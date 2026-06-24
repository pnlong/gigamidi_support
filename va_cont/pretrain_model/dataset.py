"""Sequence datasets for bar-level continuous valence/arousal regression.

Loads MuseTok latents and derives bar-level V/A from tick-indexed continuous .npz
(canonical) or falls back to cached bar-label JSON.
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.data_utils import load_latents, load_json
from va_utils import bar_labels_from_latent_metadata, load_continuous_va, aggregate_va_to_bars


class VASequenceDataset(Dataset):
    """
    Song-level sequence dataset for continuous VA regression.

    Each sample:
      latents    (T, 128)
      va_targets (T, 2)
      label_mask (T,) bool

    Bar labels are derived from continuous/{id}.npz when continuous_dir is set,
    otherwise loaded from labels_path JSON cache.
    """

    DATASET_NAME = "VA"

    def __init__(
        self,
        latents_dir: str,
        song_list: List[str],
        labels_path: Optional[str] = None,
        continuous_dir: Optional[str] = None,
    ):
        self.latents_dir = latents_dir
        self.continuous_dir = continuous_dir
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

            latent_path = os.path.join(latents_dir, f"{latent_id}.safetensors")
            if not os.path.isfile(latent_path):
                missing += 1
                continue

            latents, meta = load_latents(latent_path)
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
            self._data.append((latents.astype(np.float32), va_targets, label_mask))

        if missing:
            logging.warning(f"{self.DATASET_NAME}SequenceDataset: {missing} songs had no latent file")
        logging.info(
            f"{self.DATASET_NAME}SequenceDataset: {len(self._songs)} songs loaded "
            f"({sum(d[2].sum() for d in self._data)} annotated bars total)"
        )

    def __len__(self) -> int:
        return len(self._songs)

    def __getitem__(self, idx: int) -> dict:
        latents, va_targets, label_mask = self._data[idx]
        return {
            "song_id": self._songs[idx],
            "latents": torch.from_numpy(latents),
            "va_targets": torch.from_numpy(va_targets),
            "label_mask": torch.from_numpy(label_mask),
        }

    @staticmethod
    def collate_fn(batch: List[dict]) -> dict:
        max_len = max(item["latents"].shape[0] for item in batch)
        B = len(batch)
        latent_dim = batch[0]["latents"].shape[1]

        latents = torch.zeros(B, max_len, latent_dim)
        va_targets = torch.zeros(B, max_len, 2)
        label_mask = torch.zeros(B, max_len, dtype=torch.bool)
        padding_mask = torch.ones(B, max_len, dtype=torch.bool)

        for i, item in enumerate(batch):
            T = item["latents"].shape[0]
            latents[i, :T] = item["latents"]
            va_targets[i, :T] = item["va_targets"]
            label_mask[i, :T] = item["label_mask"]
            padding_mask[i, :T] = False

        return {
            "song_id": [item["song_id"] for item in batch],
            "latents": latents,
            "va_targets": va_targets,
            "label_mask": label_mask,
            "padding_mask": padding_mask,
        }


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


def build_dataset_for_source(
    name: str,
    split: str,
    storage_dir: Optional[str] = None,
) -> Optional[VASequenceDataset]:
    """
    Build a VASequenceDataset for a named source and split ('train' or 'valid').

    Returns None if required files are missing.
    """
    from datasets import get_dataset

    ds = get_dataset(name, storage_dir)
    split_path = ds.train_songs_path() if split == "train" else ds.valid_songs_path()

    if not (
        ds.latents_dir().is_dir()
        and split_path.is_file()
        and (ds.continuous_dir().is_dir() or ds.bar_labels_cache_path().is_file())
    ):
        return None

    with open(split_path) as f:
        song_list = [line.strip() for line in f if line.strip()]

    labels_path = str(ds.bar_labels_cache_path()) if ds.bar_labels_cache_path().is_file() else None
    continuous_dir = str(ds.continuous_dir()) if ds.continuous_dir().is_dir() else None

    cls_map = {"deam": DEAMSequenceDataset, "memo2496": Memo2496SequenceDataset, "merp": MERPSequenceDataset}
    cls = cls_map.get(name, VASequenceDataset)

    return cls(
        latents_dir=str(ds.latents_dir()),
        song_list=song_list,
        labels_path=labels_path,
        continuous_dir=continuous_dir,
    )
