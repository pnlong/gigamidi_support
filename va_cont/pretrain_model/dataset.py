"""Sequence datasets for bar-level continuous valence/arousal regression.

Loads bar features (MuseTok latents, handcrafted MIDI stats, or REMI tokens) and
derives bar-level V/A from tick-indexed continuous .npz or cached bar-label JSON.
"""

import json
import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from safetensors import safe_open
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.data_utils import load_latents, load_json
from va_utils import bar_labels_from_latent_metadata
from pretrain_model.midi_features import features_dir_for_dataset, HANDCRAFTED_FEATURE_DIM


def _decode_metadata(raw: Optional[dict]) -> Optional[dict]:
    if not raw:
        return None
    meta = {}
    for key, value in raw.items():
        try:
            meta[key] = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            try:
                meta[key] = int(value) if "." not in str(value) else float(value)
            except ValueError:
                meta[key] = value
    return meta


def load_remi_bar_features(filepath: str) -> tuple[np.ndarray, np.ndarray, Optional[dict]]:
    """Load cached REMI bar tokens from safetensors."""
    with safe_open(filepath, framework="pt", device="cpu") as f:
        tokens = f.get_tensor("bar_tokens").numpy()
        mask = f.get_tensor("token_padding_mask").numpy()
        metadata = _decode_metadata(f.metadata())
    return tokens, mask, metadata


class VASequenceDataset(Dataset):
    """
    Song-level sequence dataset for continuous VA regression.

    feature_mode:
      musetok     — (T, 128) MuseTok latents from latents_musetok/
      handcrafted — (T, 32) per-bar MIDI statistics from features_handcrafted/
      remi        — (T, L) REMI token ids + token_padding_mask from features_remi/
    """

    DATASET_NAME = "VA"

    def __init__(
        self,
        latents_dir: str,
        song_list: List[str],
        labels_path: Optional[str] = None,
        continuous_dir: Optional[str] = None,
        feature_mode: str = "musetok",
        dataset_source: str = "unknown",
    ):
        self.latents_dir = latents_dir
        self.continuous_dir = continuous_dir
        self.feature_mode = feature_mode
        self.dataset_source = dataset_source
        json_labels = load_json(labels_path) if labels_path and os.path.isfile(labels_path) else {}

        self._songs: List[str] = []
        self._data: List[tuple] = []

        missing = 0
        for song_id in tqdm(song_list, desc=f"Loading {self.DATASET_NAME} sequences", unit="song"):
            if ":" in song_id:
                latent_id, labeller_id = song_id.split(":", 1)
                song_entry = json_labels.get(latent_id, {})
                bar_entries = song_entry.get(labeller_id) if isinstance(song_entry, dict) else None
            else:
                latent_id = song_id
                bar_entries = json_labels.get(song_id)

            feat_path = os.path.join(latents_dir, f"{latent_id}.safetensors")
            if not os.path.isfile(feat_path):
                missing += 1
                continue

            bar_tokens = None
            token_mask = None
            meta = None

            if feature_mode == "remi":
                bar_tokens, token_mask, meta = load_remi_bar_features(feat_path)
                n_bars = len(bar_tokens)
                # Placeholder latents for shape compatibility in collate
                latents = bar_tokens.astype(np.float32)
            else:
                latents, meta = load_latents(feat_path)
                n_bars = len(latents)

            if bar_entries is None and self.continuous_dir:
                cont_path = Path(self.continuous_dir) / f"{latent_id}.npz"
                if cont_path.is_file():
                    try:
                        bar_entries = bar_labels_from_latent_metadata(cont_path, meta)
                    except Exception as e:
                        logging.debug(f"{latent_id}: continuous→bar failed: {e}")

            if bar_entries is None:
                continue

            va_targets = np.zeros((n_bars, 2), dtype=np.float32)
            label_mask = np.zeros(n_bars, dtype=bool)

            for bar_idx, v, a in bar_entries:
                bar_idx = int(bar_idx)
                if bar_idx < n_bars:
                    va_targets[bar_idx] = [float(v), float(a)]
                    label_mask[bar_idx] = True

            if not label_mask.any():
                continue

            self._songs.append(song_id)
            if feature_mode == "remi":
                self._data.append((
                    latents, va_targets, label_mask,
                    bar_tokens.astype(np.int32), token_mask.astype(bool),
                ))
            else:
                self._data.append((latents.astype(np.float32), va_targets, label_mask))

        if missing:
            logging.warning(
                f"{self.DATASET_NAME}SequenceDataset ({feature_mode}): "
                f"{missing} songs had no feature file in {latents_dir}"
            )
        logging.info(
            f"{self.DATASET_NAME}SequenceDataset ({feature_mode}): {len(self._songs)} songs loaded "
            f"({sum(d[2].sum() for d in self._data)} annotated bars total)"
        )

    def __len__(self) -> int:
        return len(self._songs)

    def __getitem__(self, idx: int) -> dict:
        if self.feature_mode == "remi":
            latents, va_targets, label_mask, bar_tokens, token_mask = self._data[idx]
            return {
                "song_id": self._songs[idx],
                "dataset_source": self.dataset_source,
                "latents": torch.from_numpy(latents),
                "bar_tokens": torch.from_numpy(bar_tokens),
                "token_padding_mask": torch.from_numpy(token_mask),
                "va_targets": torch.from_numpy(va_targets),
                "label_mask": torch.from_numpy(label_mask),
            }
        latents, va_targets, label_mask = self._data[idx]
        return {
            "song_id": self._songs[idx],
            "dataset_source": self.dataset_source,
            "latents": torch.from_numpy(latents),
            "va_targets": torch.from_numpy(va_targets),
            "label_mask": torch.from_numpy(label_mask),
        }

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        max_len = max(item["latents"].shape[0] for item in batch)
        B = len(batch)
        has_remi = "bar_tokens" in batch[0]
        latent_dim = batch[0]["latents"].shape[1]

        latents = torch.zeros(B, max_len, latent_dim)
        va_targets = torch.zeros(B, max_len, 2)
        label_mask = torch.zeros(B, max_len, dtype=torch.bool)
        padding_mask = torch.ones(B, max_len, dtype=torch.bool)

        out = {
            "song_id": [item["song_id"] for item in batch],
            "dataset_source": [item.get("dataset_source", "unknown") for item in batch],
            "latents": latents,
            "va_targets": va_targets,
            "label_mask": label_mask,
            "padding_mask": padding_mask,
        }

        if has_remi:
            max_tok = batch[0]["bar_tokens"].shape[1]
            bar_tokens = torch.zeros(B, max_len, max_tok, dtype=torch.long)
            token_padding_mask = torch.ones(B, max_len, max_tok, dtype=torch.bool)
            out["bar_tokens"] = bar_tokens
            out["token_padding_mask"] = token_padding_mask

        for i, item in enumerate(batch):
            T = item["latents"].shape[0]
            latents[i, :T] = item["latents"]
            va_targets[i, :T] = item["va_targets"]
            label_mask[i, :T] = item["label_mask"]
            padding_mask[i, :T] = False
            if has_remi:
                out["bar_tokens"][i, :T] = item["bar_tokens"]
                out["token_padding_mask"][i, :T] = item["token_padding_mask"]

        out["latents"] = latents
        out["va_targets"] = va_targets
        out["label_mask"] = label_mask
        out["padding_mask"] = padding_mask
        return out


class DEAMSequenceDataset(VASequenceDataset):
    DATASET_NAME = "DEAM"


class Memo2496SequenceDataset(VASequenceDataset):
    DATASET_NAME = "Memo2496"


class MERPSequenceDataset(VASequenceDataset):
    DATASET_NAME = "MERP"


class CombinedVASequenceDataset(Dataset):
    """Concatenates any number of VASequenceDataset instances."""

    def __init__(self, datasets: List[VASequenceDataset]):
        if not datasets:
            raise ValueError("CombinedVASequenceDataset requires at least one dataset")
        self._datasets = datasets
        self._cum_sizes: List[int] = []
        total = 0
        for d in datasets:
            total += len(d)
            self._cum_sizes.append(total)
        names = [d.DATASET_NAME for d in datasets]
        sizes = [len(d) for d in datasets]
        logging.info(
            f"CombinedVASequenceDataset: {total} total songs "
            + ", ".join(f"{n}={s}" for n, s in zip(names, sizes))
        )

    def __len__(self) -> int:
        return self._cum_sizes[-1]

    def __getitem__(self, idx: int) -> dict:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        for i, end in enumerate(self._cum_sizes):
            if idx < end:
                start = self._cum_sizes[i - 1] if i > 0 else 0
                return self._datasets[i][idx - start]
        raise IndexError(idx)

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        return VASequenceDataset.collate_fn(batch)


def _features_dir_for_adapter(ds, feature_mode: str) -> Path:
    if feature_mode == "musetok":
        return ds.latents_dir()
    return features_dir_for_dataset(ds.storage_dir, ds.name, feature_mode)


def build_dataset_for_source(
    name: str,
    split: str,
    storage_dir: Optional[str] = None,
    feature_mode: str = "musetok",
) -> Optional[VASequenceDataset]:
    """
    Build a VASequenceDataset for a named source and split ('train' or 'valid').

    Returns None if required files are missing.
    """
    from datasets import get_dataset
    from datasets.leakage import filter_songs_for_combined_training

    ds = get_dataset(name, storage_dir)
    split_path = ds.train_songs_path() if split == "train" else ds.valid_songs_path()
    feat_dir = _features_dir_for_adapter(ds, feature_mode)

    if not (
        feat_dir.is_dir()
        and split_path.is_file()
        and (ds.continuous_dir().is_dir() or ds.bar_labels_cache_path().is_file())
    ):
        return None

    with open(split_path) as f:
        song_list = [line.strip() for line in f if line.strip()]
    song_list = filter_songs_for_combined_training(name, song_list)
    if not song_list:
        return None

    labels_path = str(ds.bar_labels_cache_path()) if ds.bar_labels_cache_path().is_file() else None
    continuous_dir = str(ds.continuous_dir()) if ds.continuous_dir().is_dir() else None

    cls_map = {"deam": DEAMSequenceDataset, "memo2496": Memo2496SequenceDataset, "merp": MERPSequenceDataset}
    cls = cls_map.get(name, VASequenceDataset)

    return cls(
        latents_dir=str(feat_dir),
        song_list=song_list,
        labels_path=labels_path,
        continuous_dir=continuous_dir,
        feature_mode=feature_mode,
        dataset_source=name,
    )
