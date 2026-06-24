"""
Prepare per-bar valence/arousal labels from Memo2496 dynamic annotations.

Two modes:
  Averaged (default): reads valence_all_average.csv / arousal_all_average.csv.
    Output: {song_id: [[bar_idx, valence, arousal], ...]}

  Per-labeller (--per_labeller): reads Label_pseudonymous.json.
    Output: {song_id: {labeller_id: [[bar_idx, valence, arousal], ...]}}
    Song lists use "song_id:labeller_id" format.

Annotations start at 0s (no 15s offset like DEAM) and are sampled at 1s intervals.
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
from pretrain_model.prepare_labels_deam import compute_bar_labels, _load_bar_start_times


_STORAGE_DIR = os.environ.get("XMIDI_STORAGE_DIR", "/deepfreeze/pnlong/gigamidi")

MEMO2496_DIR        = os.path.join(_STORAGE_DIR, "memo2496")
MEMO2496_VA_DIR     = os.path.join(_STORAGE_DIR, "memo2496_va")
MEMO2496_LATENTS_DIR = os.path.join(MEMO2496_VA_DIR, "latents_musetok")
MEMO2496_LABELS_DIR  = os.path.join(MEMO2496_VA_DIR, "labels")

DEFAULT_VALENCE_CSV  = os.path.join(MEMO2496_DIR, "valence_all_average.csv")
DEFAULT_AROUSAL_CSV  = os.path.join(MEMO2496_DIR, "arousal_all_average.csv")
DEFAULT_PER_LABELLER_JSON = os.path.join(MEMO2496_DIR, "Label_pseudonymous.json")


def load_averaged_annotations(csv_path: str) -> dict:
    """
    Load a Memo2496 averaged annotation CSV.

    Columns: song_id, sample_0ms, sample_1000ms, ..., sample_300000ms
    1-second intervals; values may be NaN where a song is shorter than 300s.

    Returns: {song_id (int): {time_seconds (float): value (float)}}
    """
    df = pd.read_csv(csv_path)
    result = {}
    for _, row in df.iterrows():
        song_id = int(row["song_id"])
        annotations = {}
        for col in df.columns[1:]:
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


def load_per_labeller_json(json_path: str) -> dict:
    """
    Load Label_pseudonymous.json from Memo2496.

    Each entry: {MID: int, AnnotatorID: str, V: [float, ...], A: [float, ...]}
    V[i] and A[i] are the valence/arousal at second i (0-indexed, 1s intervals).

    Returns:
        {annotator_id (str): {
            "valence": {song_id (int): {time_seconds (float): value (float)}},
            "arousal": {song_id (int): {time_seconds (float): value (float)}},
        }}
    """
    logging.info(f"Loading per-labeller JSON from {json_path}...")
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    result: dict = {}
    for entry in data:
        mid  = int(entry["MID"])
        aid  = str(entry["AnnotatorID"])
        v_list = entry["V"]
        a_list = entry["A"]

        v_dict = {float(i): float(v) for i, v in enumerate(v_list)}
        a_dict = {float(i): float(a) for i, a in enumerate(a_list)}

        if aid not in result:
            result[aid] = {"valence": {}, "arousal": {}}
        result[aid]["valence"][mid] = v_dict
        result[aid]["arousal"][mid] = a_dict

    logging.info(
        f"Loaded {len(data)} annotations across {len(result)} annotators "
        f"for {len({e['MID'] for e in data})} songs"
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Memo2496 per-bar VA labels.")
    parser.add_argument("--latents_dir",    type=str, default=MEMO2496_LATENTS_DIR)
    parser.add_argument("--valence_csv",    type=str, default=DEFAULT_VALENCE_CSV)
    parser.add_argument("--arousal_csv",    type=str, default=DEFAULT_AROUSAL_CSV)
    parser.add_argument("--labels_dir",     type=str, default=MEMO2496_LABELS_DIR)
    parser.add_argument("--val_fraction",   type=float, default=0.1)
    parser.add_argument("--test_fraction",  type=float, default=0.1)
    parser.add_argument("--seed",           type=int, default=42)
    # Per-labeller mode
    parser.add_argument("--per_labeller", action="store_true",
                        help="Build per-labeller labels from Label_pseudonymous.json")
    parser.add_argument("--per_labeller_json", type=str, default=DEFAULT_PER_LABELLER_JSON)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    latent_files = sorted(Path(args.latents_dir).glob("*.safetensors"))
    logging.info(f"Found {len(latent_files)} latent files in {args.latents_dir}")

    ensure_dir(args.labels_dir)

    # Memo2496 annotations start at 0s — no 15s exclusion
    MIN_ANNOTATION_TIME = 0.0

    if args.per_labeller:
        # ------------------------------------------------------------------ #
        # Per-labeller mode                                                    #
        # ------------------------------------------------------------------ #
        per_lab = load_per_labeller_json(args.per_labeller_json)

        all_labels = {}
        songs_with_labels = []
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
            for annotator_id, lab_data in per_lab.items():
                v_data = lab_data.get("valence", {}).get(song_id)
                a_data = lab_data.get("arousal", {}).get(song_id)
                if v_data is None or a_data is None:
                    continue
                labels = compute_bar_labels(
                    bar_start_times, v_data, a_data,
                    min_annotation_time=MIN_ANNOTATION_TIME,
                )
                if labels:
                    song_lab_entries[annotator_id] = labels

            if not song_lab_entries:
                songs_skipped += 1
                continue

            all_labels[song_id_str] = song_lab_entries
            songs_with_labels.append(song_id_str)

        logging.info(f"Songs with labels: {len(songs_with_labels)}, skipped: {songs_skipped}")
        total_entries = sum(len(lab) for song in all_labels.values() for lab in song.values())
        logging.info(f"Total (song, annotator) training entries: {total_entries}")

        # Split at song level to avoid leakage
        rng = np.random.default_rng(args.seed)
        indices = np.arange(len(songs_with_labels))
        rng.shuffle(indices)

        n_val  = max(1, int(len(songs_with_labels) * args.val_fraction))
        n_test = max(1, int(len(songs_with_labels) * args.test_fraction))
        n_train = len(songs_with_labels) - n_val - n_test

        train_songs = [songs_with_labels[i] for i in indices[:n_train]]
        val_songs   = [songs_with_labels[i] for i in indices[n_train:n_train + n_val]]
        test_songs  = [songs_with_labels[i] for i in indices[n_train + n_val:]]

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

        all_valence = [e[1] for song in all_labels.values() for lab in song.values() for e in lab]
        all_arousal = [e[2] for song in all_labels.values() for lab in song.values() for e in lab]

    else:
        # ------------------------------------------------------------------ #
        # Averaged mode (default)                                              #
        # ------------------------------------------------------------------ #
        logging.info(f"Loading valence annotations from {args.valence_csv}...")
        valence_data = load_averaged_annotations(args.valence_csv)
        logging.info(f"Loading arousal annotations from {args.arousal_csv}...")
        arousal_data = load_averaged_annotations(args.arousal_csv)
        logging.info(
            f"Loaded annotations for {len(valence_data)} songs (valence), "
            f"{len(arousal_data)} songs (arousal)"
        )

        all_labels = {}
        songs_with_labels = []
        songs_skipped = 0

        for lf in tqdm(latent_files, desc="Building averaged labels"):
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
                bar_start_times, valence_data[song_id], arousal_data[song_id],
                min_annotation_time=MIN_ANNOTATION_TIME,
            )
            if not labels:
                songs_skipped += 1
                continue

            all_labels[song_id_str] = labels
            songs_with_labels.append(song_id_str)

        logging.info(f"Songs with labels: {len(songs_with_labels)}, skipped: {songs_skipped}")

        rng = np.random.default_rng(args.seed)
        indices = np.arange(len(songs_with_labels))
        rng.shuffle(indices)

        n_val  = max(1, int(len(songs_with_labels) * args.val_fraction))
        n_test = max(1, int(len(songs_with_labels) * args.test_fraction))
        n_train = len(songs_with_labels) - n_val - n_test

        train_ids = [songs_with_labels[i] for i in indices[:n_train]]
        val_ids   = [songs_with_labels[i] for i in indices[n_train:n_train + n_val]]
        test_ids  = [songs_with_labels[i] for i in indices[n_train + n_val:]]

        logging.info(f"Split: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

        all_valence = [entry[1] for entries in all_labels.values() for entry in entries]
        all_arousal = [entry[2] for entries in all_labels.values() for entry in entries]

    save_json(os.path.join(args.labels_dir, "memo2496_va_labels.json"), all_labels)

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
