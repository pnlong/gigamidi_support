"""Memo2496 dataset adapter."""

from __future__ import annotations

import csv
from pathlib import Path

import pandas as pd

from datasets.base import VADatasetSource
from va_utils import parse_sample_ms_columns


class Memo2496Dataset(VADatasetSource):
    name = "memo2496"

    def __init__(self, storage_dir=None):
        super().__init__(storage_dir)
        self.base_dir = Path(self.storage_dir) / "memo2496"
        self.songs_info_csv = self.base_dir / "songs_info_all.csv"
        self._uuid_to_song_id: dict[str, str] | None = None
        self._song_id_to_uuid: dict[str, str] | None = None
        self._valence_data = None
        self._arousal_data = None

    def _load_uuid_map(self):
        if self._uuid_to_song_id is not None:
            return
        self._uuid_to_song_id = {}
        self._song_id_to_uuid = {}
        if not self.songs_info_csv.is_file():
            return
        with open(self.songs_info_csv, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                song_id = row["song_id"].strip()
                uuid_stem = Path(row["file_name"].strip()).stem
                self._uuid_to_song_id[uuid_stem] = song_id
                self._song_id_to_uuid[song_id] = uuid_stem

    def _load_annotations(self):
        if self._valence_data is None:
            v_csv = self.base_dir / "valence_all_average.csv"
            a_csv = self.base_dir / "arousal_all_average.csv"
            self._valence_data = parse_sample_ms_columns(pd.read_csv(v_csv))
            self._arousal_data = parse_sample_ms_columns(pd.read_csv(a_csv))

    def list_song_ids(self) -> list[str]:
        self._load_uuid_map()
        if self._song_id_to_uuid:
            return sorted(self._song_id_to_uuid.keys())
        self._load_annotations()
        return sorted(str(k) for k in self._valence_data.keys())

    def audio_path(self, song_id: str) -> Path:
        self._load_uuid_map()
        uuid_stem = self._song_id_to_uuid.get(song_id, song_id) if self._song_id_to_uuid else song_id
        return self.base_dir / "MusicRawData" / f"{uuid_stem}.mp3"

    def midi_path(self, song_id: str) -> Path:
        self._load_uuid_map()
        uuid_stem = self._song_id_to_uuid.get(song_id, song_id) if self._song_id_to_uuid else song_id
        return Path(self.storage_dir) / "memo2496_midi" / f"{uuid_stem}.mid"

    def midi_dir(self) -> Path:
        return Path(self.storage_dir) / "memo2496_midi"

    def latents_dir(self) -> Path:
        return Path(self.storage_dir) / "memo2496_va" / "latents_musetok"

    def load_audio_va_annotations(self, song_id: str):
        self._load_annotations()
        sid = int(song_id)
        return self._valence_data.get(sid, {}), self._arousal_data.get(sid, {})

    def min_annotation_time(self) -> float:
        return 0.0

    def annotation_rate_hz(self) -> float:
        return 1.0
