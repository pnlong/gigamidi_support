"""Shared utilities for audio→MIDI V/A alignment and bar aggregation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from utils.midi_utils import load_midi_symusic, get_time_signature, BEAT_RESOL


DEFAULT_TARGET_HZ = 10.0


def get_bar_resol(score) -> int:
    """Ticks per bar for the score's time signature."""
    time_sig_num, time_sig_den = get_time_signature(score)
    quarters_per_bar = 4 * time_sig_num / time_sig_den
    return int(BEAT_RESOL * quarters_per_bar)


def ticks_to_seconds(score, tick: int) -> float:
    """Convert a tick position to seconds using the score's tempo track."""
    tpq = score.ticks_per_quarter
    tempos = sorted(score.tempos, key=lambda t: t.time) if score.tempos else []

    elapsed = 0.0
    current_tick = 0
    current_mspq = 500000  # default: 120 BPM

    for tempo in tempos:
        t_tick = int(tempo.time)
        if t_tick >= tick:
            break
        elapsed += (t_tick - current_tick) / tpq * (current_mspq / 1e6)
        current_tick = t_tick
        current_mspq = int(tempo.mspq)

    elapsed += (tick - current_tick) / tpq * (current_mspq / 1e6)
    return elapsed


def seconds_to_ticks(score, seconds: float) -> int:
    """Convert wall-clock seconds to MIDI tick using the score tempo map."""
    if seconds <= 0:
        return 0

    tpq = score.ticks_per_quarter
    tempos = sorted(score.tempos, key=lambda t: t.time) if score.tempos else []
    current_mspq = 500000
    current_tick = 0
    elapsed = 0.0

    for tempo in tempos:
        t_tick = int(tempo.time)
        mspq = int(tempo.mspq)
        seg_seconds = (t_tick - current_tick) / tpq * (current_mspq / 1e6)
        if elapsed + seg_seconds >= seconds:
            remaining = seconds - elapsed
            return int(current_tick + remaining / (current_mspq / 1e6) * tpq)
        elapsed += seg_seconds
        current_tick = t_tick
        current_mspq = mspq

    remaining = seconds - elapsed
    return max(0, int(current_tick + remaining / (current_mspq / 1e6) * tpq))


def compute_bar_start_times(score, n_bars: int) -> list:
    """Bar start times in seconds for bars 0..n_bars-1."""
    bar_resol = get_bar_resol(score)
    return [ticks_to_seconds(score, i * bar_resol) for i in range(n_bars)]


def compute_bar_start_ticks(n_bars: int, bar_resol: int) -> np.ndarray:
    """Bar start tick for each bar index."""
    return np.arange(n_bars, dtype=np.int64) * bar_resol


def resample_va_dict(
    valence: dict[float, float],
    arousal: dict[float, float],
    target_hz: float = DEFAULT_TARGET_HZ,
    min_time: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Interpolate sparse audio-time V/A onto a uniform grid in seconds.

    Returns (times_sec, valence, arousal).
    """
    times_v = sorted(t for t in valence if t >= min_time)
    times_a = sorted(t for t in arousal if t >= min_time)
    if not times_v or not times_a:
        return np.array([]), np.array([]), np.array([])

    t_min = max(min(times_v[0], times_a[0]), min_time)
    t_max = min(times_v[-1], times_a[-1])
    if t_max <= t_min:
        return np.array([]), np.array([]), np.array([])

    grid = np.arange(t_min, t_max + 0.5 / target_hz, 1.0 / target_hz)

    v_keys = np.array(times_v)
    v_vals = np.array([valence[t] for t in times_v])
    a_keys = np.array(times_a)
    a_vals = np.array([arousal[t] for t in times_a])

    v_interp = np.interp(grid, v_keys, v_vals)
    a_interp = np.interp(grid, a_keys, a_vals)
    return grid, v_interp.astype(np.float32), a_interp.astype(np.float32)


def convert_va_to_midi_ticks(
    score,
    times_sec: np.ndarray,
    valence: np.ndarray,
    arousal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    """
    Map audio-second V/A samples to MIDI tick positions.

    Returns (ticks, valence, arousal, tpq, bar_resol).
    """
    if len(times_sec) == 0:
        bar_resol = get_bar_resol(score)
        return (
            np.array([], dtype=np.int32),
            np.array([], dtype=np.float32),
            np.array([], dtype=np.float32),
            int(score.ticks_per_quarter),
            bar_resol,
        )

    ticks = np.array(
        [seconds_to_ticks(score, float(t)) for t in times_sec], dtype=np.int32
    )
    return (
        ticks,
        valence.astype(np.float32),
        arousal.astype(np.float32),
        int(score.ticks_per_quarter),
        get_bar_resol(score),
    )


def save_continuous_va(path: str | Path, ticks, valence, arousal, tpq: int, bar_resol: int):
    """Save tick-indexed continuous V/A."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        ticks=np.asarray(ticks, dtype=np.int32),
        valence=np.asarray(valence, dtype=np.float32),
        arousal=np.asarray(arousal, dtype=np.float32),
        tpq=np.int32(tpq),
        bar_resol=np.int32(bar_resol),
    )


def load_continuous_va(path: str | Path) -> dict:
    """Load tick-indexed continuous V/A from .npz."""
    data = np.load(path)
    return {
        "ticks": data["ticks"],
        "valence": data["valence"],
        "arousal": data["arousal"],
        "tpq": int(data["tpq"]),
        "bar_resol": int(data["bar_resol"]),
    }


def aggregate_va_to_bars(
    ticks: np.ndarray,
    valence: np.ndarray,
    arousal: np.ndarray,
    bar_resol: int,
    n_bars: int,
) -> list[list[float]]:
    """
    Mean V/A per bar tick window [i*bar_resol, (i+1)*bar_resol).

    Returns [[bar_idx, valence, arousal], ...] for bars with at least one sample.
    """
    labels = []
    for i in range(n_bars):
        tick_start = i * bar_resol
        tick_end = (i + 1) * bar_resol
        mask = (ticks >= tick_start) & (ticks < tick_end)
        if not mask.any():
            continue
        labels.append([
            i,
            round(float(valence[mask].mean()), 6),
            round(float(arousal[mask].mean()), 6),
        ])
    return labels


def bar_labels_from_continuous(continuous_path: str | Path, n_bars: int) -> list[list[float]]:
    """Derive per-bar labels from a continuous .npz file."""
    data = load_continuous_va(continuous_path)
    return aggregate_va_to_bars(
        data["ticks"], data["valence"], data["arousal"], data["bar_resol"], n_bars
    )


def bar_labels_from_latent_metadata(
    continuous_path: str | Path,
    metadata: Optional[dict],
) -> list[list[float]]:
    """Derive bar labels using n_bars from latent metadata."""
    if metadata is None or "n_bars" not in metadata:
        raise ValueError("metadata must contain n_bars")
    n_bars = int(metadata["n_bars"])
    return bar_labels_from_continuous(continuous_path, n_bars)


def parse_sample_ms_columns(df, id_col: str = "song_id") -> dict[int, dict[float, float]]:
    """Parse CSV with sample_XXXXms columns into {song_id: {time_sec: value}}."""
    result = {}
    for _, row in df.iterrows():
        song_id = int(row[id_col])
        annotations = {}
        for col in df.columns:
            if col.startswith("sample_") and col.endswith("ms"):
                try:
                    ms = int(col.replace("sample_", "").replace("ms", ""))
                    val = float(row[col])
                    if not np.isnan(val):
                        annotations[ms / 1000.0] = val
                except (ValueError, TypeError):
                    continue
        result[song_id] = annotations
    return result
