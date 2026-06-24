"""
Prepare per-bar valence/arousal labels from DEAM dynamic annotations.

Two modes:
  Averaged (default): reads CSVs from "annotations averaged per song" directory.
    Output: {song_id: [[bar_idx, valence, arousal], ...]}

  Per-labeller (--per_labeller): reads per-annotator CSV files.
    Output: {song_id: {labeller_id: [[bar_idx, valence, arousal], ...]}}
    Song lists use "song_id:labeller_id" format.

Output files:
  <labels_dir>/deam_va_labels.json
  <labels_dir>/train_songs.txt
  <labels_dir>/val_songs.txt
  <labels_dir>/test_songs.txt
"""

import os
import sys
import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.data_utils import load_latents, ensure_dir, save_json


DEAM_BASE_DIR = os.path.join(
    os.environ.get("XMIDI_STORAGE_DIR", "/deepfreeze/pnlong/gigamidi"), "deam"
)
DEAM_ANNOTATIONS_DIR = os.path.join(
    DEAM_BASE_DIR,
    "DEAM_Annotations", "annotations",
    "annotations averaged per song",
    "dynamic (per second annotations)",
)
DEAM_PER_LABELLER_DIR = os.path.join(
    DEAM_BASE_DIR,
    "DEAM_Annotations", "annotations",
    "per_subject annotations",
    "dynamic (per second annotations)",
)
DEAM_VA_DIR = os.path.join(
    os.environ.get("XMIDI_STORAGE_DIR", "/deepfreeze/pnlong/gigamidi"), "deam_va"
)
DEAM_LATENTS_DIR = os.path.join(DEAM_VA_DIR, "latents_musetok")
DEAM_LABELS_DIR = os.path.join(DEAM_VA_DIR, "labels")


def load_dynamic_annotations(csv_path: str) -> dict:
    """
    Load DEAM dynamic annotation CSV.

    Returns: {song_id (int): {time_seconds (float): value (float)}}
    """
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        song_id = int(row["song_id"])
        annotations = {}
        for col in df.columns[1:]:
            # Column format: sample_15000ms, sample_15500ms, ...
            if col.startswith("sample_") and col.endswith("ms"):
                try:
                    ms = int(col.replace("sample_", "").replace("ms", ""))
                    t = ms / 1000.0
                    val = float(row[col])
                    if not np.isnan(val):
                        annotations[t] = val
                except (ValueError, TypeError):
                    continue
        result[song_id] = annotations
    return result


def compute_bar_labels(
    bar_start_times: list,
    valence_annotations: dict,
    arousal_annotations: dict,
    min_annotation_time: float = 15.0,
) -> list:
    """
    Map dynamic annotations to per-bar labels.

    Bar i spans [bar_start_times[i], bar_start_times[i+1]).
    The last bar spans [bar_start_times[-1], +inf).
    Only bars with at least one annotation sample in their window (and >= 15s) are included.

    Returns: list of [bar_idx, valence, arousal]
    """
    n_bars = len(bar_start_times)
    annotation_times = sorted(set(valence_annotations.keys()) | set(arousal_annotations.keys()))

    labels = []
    for i in range(n_bars):
        bar_start = bar_start_times[i]
        bar_end = bar_start_times[i + 1] if i + 1 < n_bars else float("inf")

        samples_v = []
        samples_a = []
        for t in annotation_times:
            if t < min_annotation_time:
                continue
            if bar_start <= t < bar_end:
                if t in valence_annotations:
                    samples_v.append(valence_annotations[t])
                if t in arousal_annotations:
                    samples_a.append(arousal_annotations[t])

        if samples_v and samples_a:
            valence = float(np.mean(samples_v))
            arousal = float(np.mean(samples_a))
            labels.append([i, round(valence, 6), round(arousal, 6)])

    return labels


def load_per_labeller_annotations(per_labeller_dir: str) -> dict:
    """
    Load per-labeller dynamic annotation CSVs from a directory.

    Handles two common DEAM formats:
      1. Combined CSV with a 'worker_id' column alongside 'song_id' and time columns.
         (valence.csv / arousal.csv contain all labellers as rows)
      2. Separate CSV files per labeller, named valence_{worker_id}.csv, etc.

    Returns:
        {labeller_id (str): {song_id (int): {time_seconds (float): value (float)}}}
    """
    per_labeller_dir = Path(per_labeller_dir)

    def _parse_time_row(row, columns):
        annotations = {}
        for col in columns:
            if col.startswith("sample_") and col.endswith("ms"):
                try:
                    ms = int(col.replace("sample_", "").replace("ms", ""))
                    val = float(row[col])
                    if not np.isnan(val):
                        annotations[ms / 1000.0] = val
                except (ValueError, TypeError):
                    continue
        return annotations

    # Format 1: single combined CSVs with worker_id column
    valence_csv = per_labeller_dir / "valence.csv"
    arousal_csv = per_labeller_dir / "arousal.csv"
    if valence_csv.exists() and arousal_csv.exists():
        result_v: dict = {}
        result_a: dict = {}
        for csv_path, result in [(valence_csv, result_v), (arousal_csv, result_a)]:
            df = pd.read_csv(csv_path)
            if "worker_id" in df.columns:
                time_cols = [c for c in df.columns if c.startswith("sample_") and c.endswith("ms")]
                for _, row in df.iterrows():
                    worker = str(int(row["worker_id"]))
                    song_id = int(row["song_id"])
                    result.setdefault(worker, {})[song_id] = _parse_time_row(row, time_cols)
        if result_v:
            labellers = set(result_v) | set(result_a)
            combined = {}
            for lab in labellers:
                combined[lab] = {
                    "valence": result_v.get(lab, {}),
                    "arousal": result_a.get(lab, {}),
                }
            return combined

    # Format 2: separate files per labeller, e.g. valence_1.csv / arousal_1.csv
    # or worker_1/valence.csv, worker_1/arousal.csv sub-directories
    combined = {}
    # Try valence_{id}.csv pattern
    for vf in sorted(per_labeller_dir.glob("valence_*.csv")):
        labeller_id = vf.stem.replace("valence_", "")
        af = per_labeller_dir / f"arousal_{labeller_id}.csv"
        if not af.exists():
            logging.warning(f"No arousal file for labeller {labeller_id}, skipping")
            continue
        v_data = {}
        a_data = {}
        for csv_path, store in [(vf, v_data), (af, a_data)]:
            df = pd.read_csv(csv_path)
            time_cols = [c for c in df.columns if c.startswith("sample_") and c.endswith("ms")]
            for _, row in df.iterrows():
                song_id = int(row["song_id"])
                store[song_id] = _parse_time_row(row, time_cols)
        combined[labeller_id] = {"valence": v_data, "arousal": a_data}

    if not combined:
        raise FileNotFoundError(
            f"No recognisable per-labeller annotation files found in {per_labeller_dir}. "
            "Expected either a combined valence.csv/arousal.csv with a 'worker_id' column, "
            "or separate valence_{{id}}.csv / arousal_{{id}}.csv files."
        )

    logging.info(f"Loaded {len(combined)} labellers from {per_labeller_dir}")
    return combined


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare DEAM per-bar VA labels.")
    parser.add_argument("--latents_dir", type=str, default=DEAM_LATENTS_DIR)
    parser.add_argument("--annotations_dir", type=str, default=DEAM_ANNOTATIONS_DIR)
    parser.add_argument("--labels_dir", type=str, default=DEAM_LABELS_DIR)
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    # Per-labeller mode
    parser.add_argument("--per_labeller", action="store_true",
                        help="Build per-labeller labels instead of averaged")
    parser.add_argument("--per_labeller_dir", type=str, default=DEAM_PER_LABELLER_DIR,
                        help="Directory containing per-labeller annotation CSVs")
    return parser.parse_args()


def _load_bar_start_times(lf: Path) -> tuple:
    """Load (song_id_str, song_id_int, bar_start_times) from a latents safetensors file."""
    song_id_str = lf.stem
    try:
        song_id = int(song_id_str)
    except ValueError:
        return song_id_str, None, None
    _, metadata = load_latents(str(lf))
    if metadata is None or "bar_start_times_seconds" not in metadata:
        return song_id_str, song_id, None
    bar_start_times = metadata["bar_start_times_seconds"]
    if isinstance(bar_start_times, str):
        bar_start_times = json.loads(bar_start_times)
    return song_id_str, song_id, bar_start_times


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    latent_files = sorted(Path(args.latents_dir).glob("*.safetensors"))
    logging.info(f"Found {len(latent_files)} latent files in {args.latents_dir}")

    ensure_dir(args.labels_dir)

    if args.per_labeller:
        # ------------------------------------------------------------------ #
        # Per-labeller mode                                                    #
        # ------------------------------------------------------------------ #
        logging.info(f"Per-labeller mode: loading from {args.per_labeller_dir}")
        per_lab = load_per_labeller_annotations(args.per_labeller_dir)

        # all_labels: {song_id_str: {labeller_id: [[bar_idx, v, a], ...]}}
        all_labels = {}
        songs_with_labels = []   # unique song_id strings (for split)
        songs_skipped = 0

        for lf in tqdm(latent_files, desc="Building per-labeller labels"):
            song_id_str, song_id, bar_start_times = _load_bar_start_times(lf)
            if song_id is None:
                logging.warning(f"Skipping {lf.name}: cannot parse song_id as int")
                songs_skipped += 1
                continue
            if bar_start_times is None:
                logging.warning(f"Song {song_id}: missing bar_start_times_seconds, skipping")
                songs_skipped += 1
                continue

            song_lab_entries = {}
            for labeller_id, lab_data in per_lab.items():
                v_data = lab_data.get("valence", {}).get(song_id)
                a_data = lab_data.get("arousal", {}).get(song_id)
                if v_data is None or a_data is None:
                    continue
                labels = compute_bar_labels(bar_start_times, v_data, a_data)
                if labels:
                    song_lab_entries[labeller_id] = labels

            if not song_lab_entries:
                logging.debug(f"Song {song_id}: no annotated bars for any labeller, skipping")
                songs_skipped += 1
                continue

            all_labels[song_id_str] = song_lab_entries
            songs_with_labels.append(song_id_str)

        logging.info(f"Songs with labels: {len(songs_with_labels)}, skipped: {songs_skipped}")
        total_entries = sum(len(lab) for song in all_labels.values() for lab in song.values())
        logging.info(f"Total (song, labeller) training entries: {total_entries}")

        # Split at song level to avoid leakage
        rng = np.random.default_rng(args.seed)
        indices = np.arange(len(songs_with_labels))
        rng.shuffle(indices)

        n_val = max(1, int(len(songs_with_labels) * args.val_fraction))
        n_test = max(1, int(len(songs_with_labels) * args.test_fraction))
        n_train = len(songs_with_labels) - n_val - n_test

        train_songs = [songs_with_labels[i] for i in indices[:n_train]]
        val_songs   = [songs_with_labels[i] for i in indices[n_train:n_train + n_val]]
        test_songs  = [songs_with_labels[i] for i in indices[n_train + n_val:]]

        # Expand to "song_id:labeller_id" entries
        def expand(song_ids):
            entries = []
            for sid in song_ids:
                for lab in all_labels[sid]:
                    entries.append(f"{sid}:{lab}")
            return entries

        train_ids = expand(train_songs)
        val_ids   = expand(val_songs)
        test_ids  = expand(test_songs)

        logging.info(
            f"Split (songs): train={len(train_songs)}, val={len(val_songs)}, test={len(test_songs)}"
        )
        logging.info(
            f"Split (entries): train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}"
        )

        # Summary stats
        all_valence = [e[1] for song in all_labels.values() for lab in song.values() for e in lab]
        all_arousal = [e[2] for song in all_labels.values() for lab in song.values() for e in lab]

    else:
        # ------------------------------------------------------------------ #
        # Averaged mode (original behaviour)                                  #
        # ------------------------------------------------------------------ #
        valence_csv = os.path.join(args.annotations_dir, "valence.csv")
        arousal_csv = os.path.join(args.annotations_dir, "arousal.csv")

        logging.info(f"Loading valence annotations from {valence_csv}...")
        valence_data = load_dynamic_annotations(valence_csv)
        logging.info(f"Loading arousal annotations from {arousal_csv}...")
        arousal_data = load_dynamic_annotations(arousal_csv)
        logging.info(
            f"Loaded annotations for {len(valence_data)} songs (valence), "
            f"{len(arousal_data)} songs (arousal)"
        )

        all_labels = {}
        songs_with_labels = []
        songs_skipped = 0

        for lf in tqdm(latent_files, desc="Building labels"):
            song_id_str, song_id, bar_start_times = _load_bar_start_times(lf)
            if song_id is None:
                logging.warning(f"Skipping {lf.name}: cannot parse song_id as int")
                songs_skipped += 1
                continue
            if bar_start_times is None:
                logging.warning(f"Song {song_id}: missing bar_start_times_seconds, skipping")
                songs_skipped += 1
                continue
            if song_id not in valence_data or song_id not in arousal_data:
                logging.debug(f"Song {song_id}: no annotation data, skipping")
                songs_skipped += 1
                continue

            labels = compute_bar_labels(
                bar_start_times, valence_data[song_id], arousal_data[song_id]
            )
            if not labels:
                logging.debug(f"Song {song_id}: no annotated bars (all before 15s?), skipping")
                songs_skipped += 1
                continue

            all_labels[song_id_str] = labels
            songs_with_labels.append(song_id_str)

        logging.info(f"Songs with labels: {len(songs_with_labels)}, skipped: {songs_skipped}")

        rng = np.random.default_rng(args.seed)
        indices = np.arange(len(songs_with_labels))
        rng.shuffle(indices)

        n_val = max(1, int(len(songs_with_labels) * args.val_fraction))
        n_test = max(1, int(len(songs_with_labels) * args.test_fraction))
        n_train = len(songs_with_labels) - n_val - n_test

        train_ids = [songs_with_labels[i] for i in indices[:n_train]]
        val_ids   = [songs_with_labels[i] for i in indices[n_train:n_train + n_val]]
        test_ids  = [songs_with_labels[i] for i in indices[n_train + n_val:]]

        logging.info(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

        all_valence = [entry[1] for entries in all_labels.values() for entry in entries]
        all_arousal = [entry[2] for entries in all_labels.values() for entry in entries]

    save_json(os.path.join(args.labels_dir, "deam_va_labels.json"), all_labels)

    for split_name, ids in [("train", train_ids), ("val", val_ids), ("test", test_ids)]:
        path = os.path.join(args.labels_dir, f"{split_name}_songs.txt")
        with open(path, "w") as f:
            f.write("\n".join(ids) + "\n")
        logging.info(f"Wrote {split_name} split to {path}")

    all_bars = len(all_valence)
    logging.info(f"Total annotated bars: {all_bars}")
    if all_valence:
        logging.info(f"Valence  — mean: {np.mean(all_valence):.3f}, std: {np.std(all_valence):.3f}, "
                     f"min: {np.min(all_valence):.3f}, max: {np.max(all_valence):.3f}")
        logging.info(f"Arousal  — mean: {np.mean(all_arousal):.3f}, std: {np.std(all_arousal):.3f}, "
                     f"min: {np.min(all_arousal):.3f}, max: {np.max(all_arousal):.3f}")
    logging.info(f"Labels saved to {args.labels_dir}")
