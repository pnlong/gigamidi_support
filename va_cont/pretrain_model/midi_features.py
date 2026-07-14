"""Bar-level MIDI feature extraction for VA regression (no MuseTok).

Supports:
  - handcrafted: per-bar statistics from REMI/symusic (pitch, velocity, chroma, …)
  - remi: padded REMI token indices per bar (for learnable REMIBarEncoder)
"""

from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.midi_utils import (
    BEAT_RESOL,
    load_midi_symusic,
    midi_to_events_symusic,
)
from va_utils import get_bar_resol

HANDCRAFTED_FEATURE_DIM = 32
DEFAULT_REMI_MAX_TOKENS = 128
PAD_TOKEN_ID = 0


def _musetok_data_dir() -> Path:
    """MuseTok data/ dir — same repo layout as va_cont/utils/musetok_utils.py."""
    from utils.musetok_utils import MUSETOK_PATH
    return Path(MUSETOK_PATH) / "data"


def load_remi_vocab(prefer_velocity: bool = True, vocab_path: Optional[str] = None) -> dict:
    """Load MuseTok REMI event2idx vocabulary (no model checkpoint required)."""
    if vocab_path and os.path.isfile(vocab_path):
        with open(vocab_path, "rb") as f:
            vocab_data = pickle.load(f)
        return vocab_data[0]

    data_dir = _musetok_data_dir()
    candidates = []
    if prefer_velocity:
        candidates.append(data_dir / "dictionary_with_velocity.pkl")
    candidates.append(data_dir / "dictionary.pkl")
    if not prefer_velocity:
        candidates.insert(0, data_dir / "dictionary.pkl")

    for path in candidates:
        if path.is_file():
            with open(path, "rb") as f:
                vocab_data = pickle.load(f)
            return vocab_data[0]
    raise FileNotFoundError(
        f"No MuseTok vocabulary found under {data_dir} "
        f"(tried dictionary_with_velocity.pkl and dictionary.pkl). "
        "Pass --vocab_path explicitly, or ensure musetok/data/dictionary.pkl exists."
    )


def remi_vocab_size(vocab: dict) -> int:
    """MuseTok vocab size including pad token at index 0."""
    return len(vocab) + 1


def _event_to_key(event: dict) -> Optional[str]:
    name = event.get("name", "")
    value = event.get("value")
    if name not in (
        "Bar", "Beat", "EOS", "Note_Pitch", "Note_Duration",
        "Note_Velocity", "Time_Signature",
    ):
        return None
    if name.startswith("Chord"):
        return None
    if isinstance(value, (tuple, list, dict)):
        return None
    return f"{name}_{value if value is not None else 'None'}"


def events_to_token_ids(events: List[dict], vocab: dict) -> np.ndarray:
    """Map REMI event dicts to MuseTok token indices (0 = pad)."""
    ids = []
    for event in events:
        key = _event_to_key(event)
        if key is None or key not in vocab:
            continue
        ids.append(int(vocab[key]) + 1)  # reserve 0 for padding
    return np.asarray(ids, dtype=np.int32)


def split_events_by_bar(
    events: List[dict], bar_positions: List[int]
) -> List[List[dict]]:
    """Return REMI event slices per bar (excluding Bar markers themselves)."""
    slices: List[List[dict]] = []
    for i, start in enumerate(bar_positions):
        end = bar_positions[i + 1] if i + 1 < len(bar_positions) else len(events)
        chunk = [e for e in events[start + 1 : end] if e.get("name") != "Bar"]
        slices.append(chunk)
    return slices


def _chroma_from_pitch(pitch: int) -> np.ndarray:
    chroma = np.zeros(12, dtype=np.float32)
    chroma[pitch % 12] = 1.0
    return chroma


def handcrafted_features_from_bar_events(events: List[dict]) -> np.ndarray:
    """
    Compute a fixed-size feature vector for one bar from its REMI events.

    Returns shape (HANDCRAFTED_FEATURE_DIM,).
    """
    pitches, velocities, durations = [], [], []
    chroma = np.zeros(12, dtype=np.float32)

    for e in events:
        name = e.get("name", "")
        val = e.get("value")
        if name == "Note_Pitch" and val is not None:
            p = int(val)
            pitches.append(p)
            chroma += _chroma_from_pitch(p)
        elif name == "Note_Velocity" and val is not None:
            velocities.append(int(val))
        elif name == "Note_Duration" and val is not None:
            durations.append(int(val))

    n_notes = len(pitches)
    if chroma.sum() > 0:
        chroma /= chroma.sum()

    if n_notes == 0:
        return np.zeros(HANDCRAFTED_FEATURE_DIM, dtype=np.float32)

    p_arr = np.asarray(pitches, dtype=np.float32)
    v_arr = np.asarray(velocities, dtype=np.float32) if velocities else np.zeros(1)
    d_arr = np.asarray(durations, dtype=np.float32) if durations else np.zeros(1)

    feats = np.zeros(HANDCRAFTED_FEATURE_DIM, dtype=np.float32)
    feats[0] = np.log1p(n_notes)
    feats[1] = np.log1p(len(set(pitches)))
    feats[2] = float(p_arr.mean() / 127.0)
    feats[3] = float(min(p_arr.std() / 64.0, 1.0))
    feats[4] = float((p_arr.max() - p_arr.min()) / 127.0)
    feats[5] = float(v_arr.mean() / 127.0)
    feats[6] = float(v_arr.max() / 127.0)
    feats[7] = float(min(v_arr.std() / 64.0, 1.0))
    feats[8:20] = chroma
    feats[20] = float(d_arr.mean() / float(BEAT_RESOL * 4))
    feats[21] = float(min(n_notes / 16.0, 1.0))
    feats[22] = float(min(n_notes / 4.0, 1.0))
    feats[23] = float(np.mean(p_arr < 60))
    feats[24] = float(np.mean(p_arr > 72))
    return feats


def extract_handcrafted_from_midi(midi_path_or_bytes: Union[str, bytes]) -> Tuple[np.ndarray, dict]:
    """
    Extract per-bar handcrafted features from a MIDI file.

    Returns:
        features: (n_bars, HANDCRAFTED_FEATURE_DIM)
        metadata: n_bars, feature_dim, bar_resol, tpq
    """
    score = load_midi_symusic(midi_path_or_bytes)
    bar_positions, events = midi_to_events_symusic(score, has_velocity=True)
    bar_slices = split_events_by_bar(events, bar_positions)
    feats = np.stack(
        [handcrafted_features_from_bar_events(sl) for sl in bar_slices],
        axis=0,
    ).astype(np.float32)
    bar_resol = get_bar_resol(score)
    meta = {
        "n_bars": len(bar_slices),
        "feature_dim": HANDCRAFTED_FEATURE_DIM,
        "feature_mode": "handcrafted",
        "bar_resol": bar_resol,
        "tpq": int(score.ticks_per_quarter),
        "original_file_path": str(midi_path_or_bytes)[:120],
    }
    return feats, meta


def extract_remi_tokens_from_midi(
    midi_path_or_bytes: Union[str, bytes],
    vocab: dict,
    max_tokens: int = DEFAULT_REMI_MAX_TOKENS,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Extract padded REMI token indices per bar.

    Returns:
        bar_tokens: (n_bars, max_tokens) int32 — 0 = pad
        token_mask: (n_bars, max_tokens) bool — True = padded position
        metadata
    """
    score = load_midi_symusic(midi_path_or_bytes)
    bar_positions, events = midi_to_events_symusic(score, has_velocity=True)
    bar_slices = split_events_by_bar(events, bar_positions)
    n_bars = len(bar_slices)

    tokens = np.zeros((n_bars, max_tokens), dtype=np.int32)
    mask = np.ones((n_bars, max_tokens), dtype=bool)

    for i, sl in enumerate(bar_slices):
        ids = events_to_token_ids(sl, vocab)
        n = min(len(ids), max_tokens)
        if n > 0:
            tokens[i, :n] = ids[:n]
            mask[i, :n] = False

    bar_resol = get_bar_resol(score)
    meta = {
        "n_bars": n_bars,
        "max_tokens": max_tokens,
        "vocab_size": remi_vocab_size(vocab),
        "feature_mode": "remi",
        "bar_resol": bar_resol,
        "tpq": int(score.ticks_per_quarter),
        "original_file_path": str(midi_path_or_bytes)[:120],
    }
    return tokens, mask, meta


def features_dir_for_dataset(storage_dir: str, dataset_name: str, feature_mode: str) -> Path:
    """Derived feature directory under {dataset}_va/."""
    sub = {
        "handcrafted": "features_handcrafted",
        "remi": "features_remi",
        "musetok": "latents_musetok",
    }.get(feature_mode, "latents_musetok")
    return Path(storage_dir) / f"{dataset_name}_va" / sub
