"""
Combine MuseTok and midi2vec latents into a single directory (one vector per file).
Use --output_dir as --latents_dir for train.py / evaluate.py; input_dim is inferred.
Optional: --train_files to compute and save combined_latents_stats.npz for normalization.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.data_utils import load_latents, save_latents, ensure_dir


def load_file_list(path: str) -> list:
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser(
        description="Combine MuseTok + midi2vec latents into one dir (one vector per file)."
    )
    parser.add_argument("--latents_dir_musetok", type=str, required=True,
                        help="Directory containing MuseTok .safetensors (per-bar or song-level)")
    parser.add_argument("--latents_dir_midi2vec", type=str, required=True,
                        help="Directory containing midi2vec .safetensors")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output directory for combined .safetensors (and optional stats)")
    parser.add_argument("--train_files", type=str, default=None,
                        help="Optional path to train_files.txt; if set, compute mean/std and save combined_latents_stats.npz")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    ensure_dir(args.output_dir)

    # Discover files that exist in both dirs
    musetok_stems = {Path(p).stem for p in os.listdir(args.latents_dir_musetok) if p.endswith(".safetensors")}
    midi2vec_stems = {Path(p).stem for p in os.listdir(args.latents_dir_midi2vec) if p.endswith(".safetensors")}
    common = sorted(musetok_stems & midi2vec_stems)
    if not common:
        raise FileNotFoundError(
            f"No common .safetensors stems between {args.latents_dir_musetok} and {args.latents_dir_midi2vec}"
        )
    logging.info(f"Found {len(common)} files in both dirs; writing combined latents to {args.output_dir}")

    for stem in tqdm(common, desc="Combining"):
        path_m = os.path.join(args.latents_dir_musetok, f"{stem}.safetensors")
        path_v = os.path.join(args.latents_dir_midi2vec, f"{stem}.safetensors")
        lat_m, _ = load_latents(path_m)
        lat_v, _ = load_latents(path_v)
        if lat_m.ndim > 1:
            lat_m = np.mean(lat_m, axis=0)
        if lat_v.ndim > 1:
            lat_v = np.mean(lat_v, axis=0)
        combined = np.concatenate([lat_m.ravel(), lat_v.ravel()]).astype(np.float32)
        # Store as (1, D) for compatibility with load_latents / mean(dim=0)
        combined = combined.reshape(1, -1)
        out_path = os.path.join(args.output_dir, f"{stem}.safetensors")
        save_latents(out_path, combined, metadata={"n_bars": 1, "file_type": "combined"})

    if args.train_files:
        train_list = load_file_list(args.train_files)
        train_set = set(train_list)
        files_for_stats = [s for s in common if s in train_set]
        if not files_for_stats:
            logging.warning("No overlap between train_files and combined stems; skipping stats.")
        else:
            vectors = []
            for stem in tqdm(files_for_stats, desc="Computing stats"):
                path = os.path.join(args.output_dir, f"{stem}.safetensors")
                lat, _ = load_latents(path)
                if lat.ndim > 1:
                    lat = np.mean(lat, axis=0)
                vectors.append(lat.ravel().astype(np.float32))
            arr = np.stack(vectors, axis=0)
            mean = np.mean(arr, axis=0)
            std = np.std(arr, axis=0)
            std = np.where(std < 1e-8, 1.0, std)
            stats_path = os.path.join(args.output_dir, "combined_latents_stats.npz")
            np.savez(stats_path, mean=mean, std=std)
            logging.info(f"Saved normalization stats to {stats_path} (from {len(files_for_stats)} train files)")

    logging.info("Done.")


if __name__ == "__main__":
    main()
