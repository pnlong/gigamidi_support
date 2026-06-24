"""Abstract dataset adapter for VA pipeline sources."""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import numpy as np

_STORAGE_DIR = os.environ.get("XMIDI_STORAGE_DIR", "/deepfreeze/pnlong/gigamidi")


class VADatasetSource(ABC):
    """Common interface for DEAM, Memo2496, and MERP."""

    name: str = "base"

    def __init__(self, storage_dir: Optional[str] = None):
        self.storage_dir = storage_dir or _STORAGE_DIR

    @abstractmethod
    def list_song_ids(self) -> list[str]:
        """Return song IDs available from audio or annotation files."""

    def latent_id(self, song_id: str) -> str:
        """ID used for latents / continuous VA filenames."""
        return song_id

    @abstractmethod
    def audio_path(self, song_id: str) -> Path:
        ...

    @abstractmethod
    def midi_path(self, song_id: str) -> Path:
        ...

    def latents_path(self, song_id: str) -> Path:
        return self.latents_dir() / f"{self.latent_id(song_id)}.safetensors"

    def continuous_va_path(self, song_id: str) -> Path:
        return self.continuous_dir() / f"{self.latent_id(song_id)}.npz"

    @abstractmethod
    def midi_dir(self) -> Path:
        ...

    @abstractmethod
    def latents_dir(self) -> Path:
        ...

    def continuous_dir(self) -> Path:
        return Path(self.storage_dir) / f"{self.name}_va" / "continuous"

    def labels_dir(self) -> Path:
        return Path(self.storage_dir) / f"{self.name}_va" / "labels"

    def va_dir(self) -> Path:
        return Path(self.storage_dir) / f"{self.name}_va"

    @abstractmethod
    def load_audio_va_annotations(
        self, song_id: str
    ) -> tuple[dict[float, float], dict[float, float]]:
        """Return (valence_by_sec, arousal_by_sec) in audio wall-clock seconds."""

    @abstractmethod
    def min_annotation_time(self) -> float:
        ...

    @abstractmethod
    def annotation_rate_hz(self) -> float:
        ...

    def excluded_song_ids(self) -> set[str]:
        """Song IDs to exclude from this dataset's splits (overlap policy)."""
        return set()

    def train_songs_path(self) -> Path:
        return self.labels_dir() / "train_songs.txt"

    def valid_songs_path(self) -> Path:
        return self.labels_dir() / "val_songs.txt"

    def test_songs_path(self) -> Path:
        return self.labels_dir() / "test_songs.txt"

    def bar_labels_cache_path(self) -> Path:
        return self.labels_dir() / f"{self.name}_va_labels.json"

    def make_splits(
        self,
        song_ids: list[str],
        seed: int = 42,
        val_fraction: float = 0.1,
        test_fraction: float = 0.1,
    ) -> tuple[list[str], list[str], list[str]]:
        """Song-level train/val/test split."""
        ids = [s for s in song_ids if s not in self.excluded_song_ids()]
        rng = np.random.default_rng(seed)
        indices = np.arange(len(ids))
        rng.shuffle(indices)

        n_val = max(1, int(len(ids) * val_fraction)) if len(ids) > 2 else 0
        n_test = max(1, int(len(ids) * test_fraction)) if len(ids) > 2 else 0
        n_train = len(ids) - n_val - n_test
        if n_train <= 0 and len(ids) > 0:
            n_train = len(ids)
            n_val = n_test = 0

        train = [ids[i] for i in indices[:n_train]]
        val = [ids[i] for i in indices[n_train:n_train + n_val]]
        test = [ids[i] for i in indices[n_train + n_val:]]
        return train, val, test

    def write_splits(self, train, val, test) -> None:
        self.labels_dir().mkdir(parents=True, exist_ok=True)
        for name, ids in [("train", train), ("val", val), ("test", test)]:
            path = self.labels_dir() / f"{name}_songs.txt"
            path.write_text("\n".join(ids) + ("\n" if ids else ""))

    def list_audio_paths(self) -> list[Path]:
        """All audio files for AMT batch jobs."""
        return [self.audio_path(sid) for sid in self.list_song_ids()]

    def is_ready_for_training(self) -> bool:
        """Check latents, continuous VA, and split files exist."""
        return (
            self.latents_dir().is_dir()
            and self.continuous_dir().is_dir()
            and self.train_songs_path().is_file()
            and self.valid_songs_path().is_file()
        )
