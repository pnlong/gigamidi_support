#!/usr/bin/env python3
"""Verify dataset paths and report readiness for the VA pipeline."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import get_dataset, list_datasets


def parse_args():
    p = argparse.ArgumentParser(description="Verify DEAM / Memo2496 / MERP data on disk.")
    p.add_argument("--datasets", nargs="+", default=list_datasets())
    p.add_argument("--storage_dir", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    storage = args.storage_dir or __import__("os").environ.get(
        "XMIDI_STORAGE_DIR", "/deepfreeze/pnlong/gigamidi"
    )
    print(f"Storage: {storage}\n")

    for name in args.datasets:
        ds = get_dataset(name, args.storage_dir)
        ids = ds.list_song_ids()
        n_audio = sum(1 for s in ids if ds.audio_path(s).is_file())
        n_midi = sum(1 for s in ids if ds.midi_path(s).is_file())
        n_lat = len(list(ds.latents_dir().glob("*.safetensors"))) if ds.latents_dir().is_dir() else 0
        n_cont = len(list(ds.continuous_dir().glob("*.npz"))) if ds.continuous_dir().is_dir() else 0
        ready = ds.is_ready_for_training()

        print(f"=== {name} ===")
        print(f"  songs listed:     {len(ids)}")
        print(f"  audio present:    {n_audio}")
        print(f"  midi present:     {n_midi}")
        print(f"  latents:          {n_lat}")
        print(f"  continuous VA:    {n_cont}")
        print(f"  training ready:   {ready}")
        print(f"  audio dir:        {ds.audio_path(ids[0]) if ids else 'N/A'}")
        print(f"  midi dir:         {ds.midi_dir()}")
        print()


if __name__ == "__main__":
    main()
