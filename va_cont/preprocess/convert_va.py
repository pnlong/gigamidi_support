"""
Convert audio-time V/A annotations to tick-indexed continuous storage on the MIDI timeline.

Usage:
    python va_cont/preprocess/convert_va.py --dataset deam
    python va_cont/preprocess/convert_va.py --dataset merp --cache-bar-labels
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datasets import get_dataset
from utils.data_utils import ensure_dir, load_latents, save_json
from utils.midi_utils import load_midi_symusic
from va_utils import (
    DEFAULT_TARGET_HZ,
    aggregate_va_to_bars,
    convert_va_to_midi_ticks,
    resample_va_dict,
    save_continuous_va,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Convert audio V/A to MIDI-tick continuous storage.")
    parser.add_argument("--dataset", type=str, required=True, choices=["deam", "memo2496", "merp"])
    parser.add_argument("--storage_dir", type=str, default=None)
    parser.add_argument("--target_hz", type=float, default=DEFAULT_TARGET_HZ)
    parser.add_argument("--cache-bar-labels", action="store_true",
                        help="Also write bar-level JSON cache for faster dataset loading")
    parser.add_argument("--val_fraction", type=float, default=0.1)
    parser.add_argument("--test_fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ds = get_dataset(args.dataset, args.storage_dir)
    continuous_dir = ds.continuous_dir()
    ensure_dir(str(continuous_dir))
    ensure_dir(str(ds.labels_dir()))

    min_t = ds.min_annotation_time()
    bar_cache: dict = {}
    converted = skipped = 0

    for song_id in tqdm(ds.list_song_ids(), desc=f"convert_va {ds.name}"):
        out_path = ds.continuous_va_path(song_id)
        if args.resume and out_path.is_file():
            continue

        v_dict, a_dict = ds.load_audio_va_annotations(song_id)
        if not v_dict or not a_dict:
            skipped += 1
            continue

        midi_path = ds.midi_path(song_id)
        if not midi_path.is_file():
            logging.debug(f"No MIDI for {song_id}, skip")
            skipped += 1
            continue

        times, v_arr, a_arr = resample_va_dict(v_dict, a_dict, args.target_hz, min_t)
        if len(times) == 0:
            skipped += 1
            continue

        score = load_midi_symusic(str(midi_path))
        ticks, v_out, a_out, tpq, bar_resol = convert_va_to_midi_ticks(score, times, v_arr, a_arr)
        save_continuous_va(out_path, ticks, v_out, a_out, tpq, bar_resol)
        converted += 1

        if args.cache_bar_labels:
            latent_path = ds.latents_path(song_id)
            n_bars = None
            if latent_path.is_file():
                _, meta = load_latents(str(latent_path))
                if meta and "n_bars" in meta:
                    n_bars = int(meta["n_bars"])
            if n_bars is None:
                max_tick = int(ticks.max()) if len(ticks) else 0
                n_bars = max_tick // bar_resol + 1
            labels = aggregate_va_to_bars(ticks, v_out, a_out, bar_resol, n_bars)
            if labels:
                bar_cache[ds.latent_id(song_id)] = labels

    logging.info(f"Converted {converted} songs, skipped {skipped}")

    # Write splits from songs that have continuous VA
    ready_ids = sorted(
        p.stem for p in continuous_dir.glob("*.npz")
        if p.stem not in ds.excluded_song_ids()
    )
    if ready_ids:
        train, val, test = ds.make_splits(
            ready_ids, seed=args.seed,
            val_fraction=args.val_fraction, test_fraction=args.test_fraction,
        )
        ds.write_splits(train, val, test)
        logging.info(f"Splits: train={len(train)}, val={len(val)}, test={len(test)}")

    if args.cache_bar_labels and bar_cache:
        save_json(str(ds.bar_labels_cache_path()), bar_cache)
        logging.info(f"Bar label cache → {ds.bar_labels_cache_path()}")
