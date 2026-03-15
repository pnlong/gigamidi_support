#!/usr/bin/env python3
"""
Build a cached mapping of number of bars per song in a MuseTok latents directory.

Writes latents_dir/bars_per_song.csv (columns: filename, n_bars) to disk. The cache
is shared by any config or task that uses this latents dir (emotion, genre, VA, etc.).
If the cache already exists, it is left unchanged unless --force is used. Training/eval
with bar-level chunking build the cache implicitly on first run; this script is optional
(e.g. to prebuild or to force a refresh).

Usage:
  python scripts/build_bars_index.py /path/to/latents_musetok
  python scripts/build_bars_index.py /path/to/latents_musetok --force
"""
import argparse
import os
import sys
from os.path import dirname, realpath

sys.path.insert(0, dirname(dirname(realpath(__file__))))
from utils.data_utils import (
    BARS_PER_SONG_FILENAME,
    build_bars_per_song_index,
    get_bars_per_song_index,
)


def main():
    parser = argparse.ArgumentParser(
        description="Build or verify bars_per_song.csv cache in a MuseTok latents directory (shared across configs)."
    )
    parser.add_argument(
        "latents_dir",
        type=str,
        help="Path to latents directory (e.g. latents_musetok). Cache is shared by all configs using this dir.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recompute and overwrite bars_per_song.csv even if it exists.",
    )
    args = parser.parse_args()

    latents_dir = os.path.abspath(args.latents_dir)
    if not os.path.isdir(latents_dir):
        parser.error(f"Not a directory: {latents_dir}")

    index_path = os.path.join(latents_dir, BARS_PER_SONG_FILENAME)
    if not args.force and os.path.isfile(index_path):
        idx = get_bars_per_song_index(latents_dir)
        print(f"Cache exists: {index_path} ({len(idx)} entries). Use --force to recompute.")
        return 0

    idx = build_bars_per_song_index(latents_dir, force=True)
    print(f"Wrote {index_path} with {len(idx)} entries.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
