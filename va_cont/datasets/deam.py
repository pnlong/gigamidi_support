"""DEAM dataset adapter."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from datasets.base import VADatasetSource
from va_utils import parse_sample_ms_columns

# MERP DEAM anchor IDs — populate via merp/songs.json "deam_anchor": true entries
MERP_DEAM_ANCHOR_IDS: set[str] = set()


class DEAMDataset(VADatasetSource):
    name = "deam"

    def __init__(self, storage_dir=None):
        super().__init__(storage_dir)
        self.base_dir = Path(self.storage_dir) / "deam"
        self.annotations_dir = (
            self.base_dir
            / "DEAM_Annotations"
            / "annotations"
            / "annotations averaged per song"
            / "dynamic (per second annotations)"
        )
        self._valence_data = None
        self._arousal_data = None

    def _load_annotation_csvs(self):
        if self._valence_data is None:
            v_csv = self.annotations_dir / "valence.csv"
            a_csv = self.annotations_dir / "arousal.csv"
            self._valence_data = parse_sample_ms_columns(pd.read_csv(v_csv))
            self._arousal_data = parse_sample_ms_columns(pd.read_csv(a_csv))

    def list_song_ids(self) -> list[str]:
        audio_dir = self.base_dir / "DEAM_audio" / "MEMD_audio"
        if audio_dir.is_dir():
            ids = sorted({p.stem for p in audio_dir.glob("*.mp3")})
            if ids:
                return ids
        self._load_annotation_csvs()
        return sorted(str(k) for k in self._valence_data.keys())

    def audio_path(self, song_id: str) -> Path:
        return self.base_dir / "DEAM_audio" / "MEMD_audio" / f"{song_id}.mp3"

    def midi_path(self, song_id: str) -> Path:
        return self.base_dir / "DEAM_midi" / "MEMD_midi" / f"{song_id}.mid"

    def midi_dir(self) -> Path:
        return self.base_dir / "DEAM_midi" / "MEMD_midi"

    def latents_dir(self) -> Path:
        return Path(self.storage_dir) / "deam_va" / "latents_musetok"

    def load_audio_va_annotations(self, song_id: str):
        self._load_annotation_csvs()
        sid = int(song_id)
        return self._valence_data.get(sid, {}), self._arousal_data.get(sid, {})

    def min_annotation_time(self) -> float:
        return 15.0

    def annotation_rate_hz(self) -> float:
        return 2.0

    def excluded_song_ids(self) -> set[str]:
        return set()
