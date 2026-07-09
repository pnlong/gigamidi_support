"""MERP dataset adapter."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from datasets.base import VADatasetSource
from datasets.leakage import MERP_DEAM_ANCHOR_SONG_IDS


class MERPDataset(VADatasetSource):
    """
    Music Emotion Recognition with Profile (MERP) — 54 full-length tracks.

    Expected layout under $XMIDI_STORAGE_DIR/merp/:
      audio/{song_id}.wav          (or .mp3 / .flac)
      annotations/
        averaged_valence.parquet   (or .csv with time_sec, song_id, valence)
        averaged_arousal.parquet
      songs.json                   optional manifest: [{id, deam_anchor: bool}, ...]

    Fallback CSV format (wide): song_id, sample_0ms, sample_100ms, ... (10 Hz)
    """

    name = "merp"

    def __init__(self, storage_dir=None):
        super().__init__(storage_dir)
        self.base_dir = Path(self.storage_dir) / "merp"
        self.annotations_dir = self.base_dir / "annotations"
        self._manifest: list[dict] | None = None
        self._valence_by_song: dict[str, dict[float, float]] | None = None
        self._arousal_by_song: dict[str, dict[float, float]] | None = None

    def _discover_audio_ids(self) -> list[str]:
        audio_dir = self.base_dir / "audio"
        if not audio_dir.is_dir():
            for alt in ("Audio", "audio_files", "MusicRawData"):
                candidate = self.base_dir / alt
                if candidate.is_dir():
                    audio_dir = candidate
                    break
        if audio_dir.is_dir():
            ids = []
            for ext in ("*.wav", "*.mp3", "*.flac", "*.ogg"):
                ids.extend(p.stem for p in audio_dir.glob(ext))
            return sorted(set(ids))
        return []

    def _load_manifest(self):
        if self._manifest is not None:
            return
        manifest_path = self.base_dir / "songs.json"
        if manifest_path.is_file():
            with open(manifest_path) as f:
                self._manifest = json.load(f)
        else:
            self._manifest = [{"id": sid} for sid in self._discover_audio_ids()]

    def list_song_ids(self) -> list[str]:
        self._load_manifest()
        ids = [str(entry.get("id", entry.get("song_id"))) for entry in self._manifest]
        if ids:
            return sorted(set(ids))
        return self._discover_audio_ids()

    def audio_path(self, song_id: str) -> Path:
        for sub in ("audio", "Audio", "audio_files", "MusicRawData"):
            base = self.base_dir / sub
            if not base.is_dir():
                continue
            for ext in (".wav", ".mp3", ".flac", ".ogg"):
                p = base / f"{song_id}{ext}"
                if p.is_file():
                    return p
        return self.base_dir / "audio" / f"{song_id}.wav"

    def midi_path(self, song_id: str) -> Path:
        return Path(self.storage_dir) / "merp_midi" / f"{song_id}.mid"

    def midi_dir(self) -> Path:
        return Path(self.storage_dir) / "merp_midi"

    def latents_dir(self) -> Path:
        return Path(self.storage_dir) / "merp_va" / "latents_musetok"

    def _load_long_format_parquet(self, path: Path, value_col: str) -> dict[str, dict[float, float]]:
        df = pd.read_parquet(path)
        result: dict[str, dict[float, float]] = {}
        sid_col = "song_id" if "song_id" in df.columns else "id"
        time_col = None
        for c in ("time_sec", "time", "timestamp", "t"):
            if c in df.columns:
                time_col = c
                break
        if time_col is None:
            raise ValueError(f"No time column found in {path}")
        for _, row in df.iterrows():
            sid = str(int(row[sid_col]) if str(row[sid_col]).isdigit() else row[sid_col])
            t = float(row[time_col])
            v = float(row[value_col])
            result.setdefault(sid, {})[t] = v
        return result

    def _load_wide_csv(self, path: Path) -> dict[int, dict[float, float]]:
        from va_utils import parse_sample_ms_columns
        return parse_sample_ms_columns(pd.read_csv(path))

    def _arrays_to_time_dict(self, values: list[float], hz: float = 10.0) -> dict[float, float]:
        return {i / hz: float(v) for i, v in enumerate(values)}

    @staticmethod
    def _rescale_rater_series(series: np.ndarray) -> np.ndarray:
        """Per-rater min-max to [-1, 1] (MERP filtered pipeline step)."""
        lo, hi = float(series.min()), float(series.max())
        if hi <= lo:
            return series
        return 2.0 * (series - lo) / (hi - lo) - 1.0

    def _average_rater_series(
        self,
        series_list: list,
        *,
        rescale_per_rater: bool = False,
    ) -> list[float]:
        """
        Mean across raters at 10 Hz.

        Drops trials shorter than 80% of the per-song median length, then aligns
        to the median kept length (avoids one short rater truncating full songs).
        """
        arrays = [np.asarray(s, dtype=np.float64) for s in series_list if len(s) > 0]
        if not arrays:
            return []
        if rescale_per_rater:
            arrays = [self._rescale_rater_series(a) for a in arrays]

        median_len = int(np.median([a.shape[0] for a in arrays]))
        min_accept = int(0.8 * median_len)
        arrays = [a for a in arrays if a.shape[0] >= min_accept]
        if not arrays:
            return []

        target_len = int(np.median([a.shape[0] for a in arrays]))
        arrays = [a for a in arrays if a.shape[0] >= target_len]
        if not arrays:
            return []
        stacked = np.stack([a[:target_len] for a in arrays], axis=0)
        return stacked.mean(axis=0).tolist()

    def _load_hf_parquet_annotations(self, path: Path, *, rescale_per_rater: bool = False):
        df = pd.read_parquet(path)
        if "song_id" not in df.columns or "valence" not in df.columns or "arousal" not in df.columns:
            raise ValueError(f"Unexpected schema in {path}")

        self._valence_by_song = {}
        self._arousal_by_song = {}
        for song_id, group in df.groupby("song_id"):
            sid = str(song_id)
            rescale = rescale_per_rater
            v_avg = self._average_rater_series(group["valence"].tolist(), rescale_per_rater=rescale)
            a_avg = self._average_rater_series(group["arousal"].tolist(), rescale_per_rater=rescale)
            if v_avg and a_avg:
                self._valence_by_song[sid] = self._arrays_to_time_dict(v_avg)
                self._arousal_by_song[sid] = self._arrays_to_time_dict(a_avg)

    def _ensure_annotations_loaded(self):
        if self._valence_by_song is not None:
            return

        # HuggingFace MERP layout: parquet at dataset root.
        # Prefer raw for full-song V/A (filtered is segment-trimmed ~15–20% of audio).
        raw_path = self.base_dir / "annotations_raw.parquet"
        if raw_path.is_file():
            self._load_hf_parquet_annotations(raw_path, rescale_per_rater=True)
            return

        filtered_path = self.base_dir / "annotations_filtered.parquet"
        if filtered_path.is_file():
            self._load_hf_parquet_annotations(filtered_path, rescale_per_rater=False)
            return

        v_long = self.annotations_dir / "averaged_valence.parquet"
        a_long = self.annotations_dir / "averaged_arousal.parquet"
        if v_long.is_file() and a_long.is_file():
            v_raw = self._load_long_format_parquet(v_long, "valence")
            a_raw = self._load_long_format_parquet(a_long, "arousal")
            self._valence_by_song = v_raw
            self._arousal_by_song = a_raw
            return

        v_csv = self.annotations_dir / "valence.csv"
        a_csv = self.annotations_dir / "arousal.csv"
        if v_csv.is_file() and a_csv.is_file():
            v_int = self._load_wide_csv(v_csv)
            a_int = self._load_wide_csv(a_csv)
            self._valence_by_song = {str(k): v for k, v in v_int.items()}
            self._arousal_by_song = {str(k): v for k, v in a_int.items()}
            return

        # Combined parquet from HuggingFace re-upload
        combined = self.annotations_dir / "merp_annotations.parquet"
        if combined.is_file():
            df = pd.read_parquet(combined)
            self._valence_by_song = {}
            self._arousal_by_song = {}
            sid_col = "song_id" if "song_id" in df.columns else "id"
            time_col = next(c for c in ("time_sec", "time", "timestamp") if c in df.columns)
            for _, row in df.iterrows():
                sid = str(row[sid_col])
                t = float(row[time_col])
                self._valence_by_song.setdefault(sid, {})[t] = float(row["valence"])
                self._arousal_by_song.setdefault(sid, {})[t] = float(row["arousal"])
            return

        self._valence_by_song = {}
        self._arousal_by_song = {}

    def load_audio_va_annotations(self, song_id: str):
        self._ensure_annotations_loaded()
        return (
            self._valence_by_song.get(song_id, {}),
            self._arousal_by_song.get(song_id, {}),
        )

    def min_annotation_time(self) -> float:
        return 0.0

    def annotation_rate_hz(self) -> float:
        return 10.0

    def excluded_song_ids(self) -> set[str]:
        """Exclude DEAM anchor tracks from MERP splits (overlap with DEAM)."""
        self._load_manifest()
        excluded = set(MERP_DEAM_ANCHOR_SONG_IDS)
        for entry in self._manifest:
            if entry.get("deam_anchor") or entry.get("source") == "deam":
                excluded.add(str(entry.get("id", entry.get("song_id"))))
        return excluded
