"""
Preprocess XMIDI with batched midi2vec: stratified batches, run midi2edgelist -> edgelist2vec
per batch in parallel, then consolidate to a single latents dir.

Stratification is XMIDI-specific (emotion + genre from filename). Each batch is processed
by one worker; batches that already have embeddings.bin are skipped unless --reset.
"""

import os
import csv
import random
import logging
import shutil
from pathlib import Path
from multiprocessing import Pool, cpu_count
from typing import List, Tuple
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.data_utils import XMIDI_LATENTS_DIR, ensure_dir, MIDI2VEC_BATCHES_DIR
from utils.midi2vec_utils import (
    run_midi2edgelist_for_files,
    run_edgelist2vec,
    consolidate_batched_embeddings_to_safetensors,
    _list_midi_files,
)
from pretrain_model.prepare_labels import extract_labels_from_filename
from tqdm import tqdm


def find_xmidi_files(xmidi_dir: str):
    """Find all XMIDI MIDI files (absolute paths)."""
    files = _list_midi_files(xmidi_dir)
    return [str(f.resolve()) for f in files]


def _stratified_batch_assignment(
    file_paths: list,
    num_batches: int,
    seed: int,
) -> Tuple[List[List[str]], List[Tuple[str, int]]]:
    """
    Assign files to num_batches by stratum (emotion, genre). Unparseable -> ("_unknown", "_unknown").

    Process: For each (emotion, genre) group:
      1. Randomize the filepaths in that group.
      2. Randomize the batch order (which batch gets the 1st, 2nd, … file from this group).
      3. Distribute files to batches one at a time in that order (round-robin over the
         shuffled batch indices) until the group is exhausted.
    This keeps label proportions similar across all batches (no bias toward earlier batches).

    Returns (batches, assignments) where batches[i] = list of file paths,
    assignments = [(file_path, batch_id), ...].
    """
    n = len(file_paths)
    if n == 0:
        return [], []
    num_batches = max(1, min(num_batches, n))

    stratum_to_files = {}
    for fp in file_paths:
        path = Path(fp)
        emotion, genre = extract_labels_from_filename(path.name)
        if emotion is None or genre is None:
            stratum = ("_unknown", "_unknown")
        else:
            stratum = (emotion, genre)
        stratum_to_files.setdefault(stratum, []).append(fp)

    rng = random.Random(seed)
    batches = [[] for _ in range(num_batches)]
    assignments = []

    for stratum in sorted(stratum_to_files.keys()):
        files = list(stratum_to_files[stratum])
        rng.shuffle(files)
        batch_order = list(range(num_batches))
        rng.shuffle(batch_order)
        for i, fp in enumerate(files):
            batch_id = batch_order[i % num_batches]
            batches[batch_id].append(fp)
            assignments.append((fp, batch_id))

    return batches, assignments


def _compute_and_log_batch_label_statistics(
    batches: List[List[str]],
    batch_output_root: str,
) -> None:
    """
    Compute per-batch emotion and genre counts and write to batch_output_root for manual
    inspection of stratification consistency. Also logs a short summary.
    """
    from collections import defaultdict

    # Per batch: emotion counts, genre counts
    batch_emotion = []
    batch_genre = []
    for batch_files in batches:
        e_counts = defaultdict(int)
        g_counts = defaultdict(int)
        for fp in batch_files:
            emotion, genre = extract_labels_from_filename(Path(fp).name)
            e_counts[emotion if emotion is not None else "_unknown"] += 1
            g_counts[genre if genre is not None else "_unknown"] += 1
        batch_emotion.append(dict(e_counts))
        batch_genre.append(dict(g_counts))

    all_emotions = sorted(set(k for d in batch_emotion for k in d.keys()))
    all_genres = sorted(set(k for d in batch_genre for k in d.keys()))
    ensure_dir(batch_output_root)
    stats_path = os.path.join(batch_output_root, "batch_label_stats.txt")
    with open(stats_path, "w", encoding="utf-8") as f:
        f.write("Batch label statistics (stratification check)\n")
        f.write("=" * 60 + "\n\n")
        for i, (e_counts, g_counts) in enumerate(zip(batch_emotion, batch_genre)):
            n = sum(e_counts.values())
            if n == 0:
                continue
            f.write(f"Batch {i} (n={n})\n")
            f.write("  Emotions (count, proportion): " + ", ".join(
                f"{k}={e_counts.get(k, 0)} ({e_counts.get(k, 0) / n:.3f})" for k in all_emotions
            ) + "\n")
            f.write("  Genres (count, proportion):   " + ", ".join(
                f"{k}={g_counts.get(k, 0)} ({g_counts.get(k, 0) / n:.3f})" for k in all_genres
            ) + "\n\n")
        f.write("Summary: If stratification is consistent, emotion and genre counts\n")
        f.write("and proportions should be roughly similar across batches (up to rounding).\n")
    logging.info(f"Batch label statistics written to {stats_path}")
    # Optional CSV for spreadsheets (counts and proportions)
    csv_path = os.path.join(batch_output_root, "batch_label_stats.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        header = (
            ["batch_id", "n"]
            + [f"emotion_{e}" for e in all_emotions]
            + [f"emotion_{e}_prop" for e in all_emotions]
            + [f"genre_{g}" for g in all_genres]
            + [f"genre_{g}_prop" for g in all_genres]
        )
        w.writerow(header)
        for i, (e_counts, g_counts) in enumerate(zip(batch_emotion, batch_genre)):
            n = sum(e_counts.values())
            if n == 0:
                row = [i, 0] + [0] * (len(all_emotions) * 2 + len(all_genres) * 2)
            else:
                e_vals = [e_counts.get(e, 0) for e in all_emotions]
                e_props = [e_counts.get(e, 0) / n for e in all_emotions]
                g_vals = [g_counts.get(g, 0) for g in all_genres]
                g_props = [g_counts.get(g, 0) / n for g in all_genres]
                row = [i, n] + e_vals + e_props + g_vals + g_props
            w.writerow(row)
    logging.info(f"Batch label CSV written to {csv_path}")

    # PDF plot: 3 subplots. One line per series; left y = percentage, right y = count (same data, two scales).
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter, FuncFormatter
    import numpy as np

    batch_ids = list(range(len(batch_emotion)))
    batch_sizes = [sum(d.values()) for d in batch_emotion]
    total = sum(batch_sizes) or 1
    batch_proportions = [n / total for n in batch_sizes]
    mean_batch_size = total / len(batch_ids) if batch_ids else 1

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    # Subplot 1: one line (proportion). Right axis = same data as count (proportion * total).
    ax1.plot(batch_ids, batch_proportions)
    ax1.set_ylabel("Percentage")
    ax1.yaxis.set_major_formatter(PercentFormatter(1.0))
    y1_max = max(batch_proportions) * 1.15 if batch_proportions else 1.0
    ax1.set_ylim(0, y1_max)
    ax1_twin = ax1.twinx()
    ax1_twin.set_ylim(0, y1_max)
    ax1_twin.yaxis.set_major_formatter(FuncFormatter(lambda p, _: f"{p * total:.0f}"))
    ax1_twin.set_ylabel("Count")
    ax1.set_title("Batch size")
    ax1.grid(True, alpha=0.3)

    # Subplot 2: one line per emotion (proportion). Right axis = count scale (proportion * mean batch size).
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(all_emotions), 1)))
    for e, color in zip(all_emotions, colors):
        counts = [d.get(e, 0) for d in batch_emotion]
        props = [c / n if n else 0 for c, n in zip(counts, batch_sizes)]
        ax2.plot(batch_ids, props, color=color, label=e)
    ax2.set_ylabel("Percentage")
    ax2.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax2.set_ylim(0, None)
    ax2.legend(loc="upper left", fontsize=7, ncol=2)
    ax2_twin = ax2.twinx()
    ax2_twin.set_ylim(ax2.get_ylim())
    ax2_twin.yaxis.set_major_formatter(FuncFormatter(lambda p, _: f"{p * mean_batch_size:.0f}"))
    ax2_twin.set_ylabel("Count")
    ax2.set_title("Emotion labels")
    ax2.grid(True, alpha=0.3)

    # Subplot 3: one line per genre (proportion). Right axis = count scale (proportion * mean batch size).
    colors_g = plt.cm.tab10(np.linspace(0, 1, max(len(all_genres), 1)))
    for g, color in zip(all_genres, colors_g):
        counts = [d.get(g, 0) for d in batch_genre]
        props = [c / n if n else 0 for c, n in zip(counts, batch_sizes)]
        ax3.plot(batch_ids, props, color=color, label=g)
    ax3.set_ylabel("Percentage")
    ax3.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax3.set_xlabel("Batch id")
    ax3.set_ylim(0, None)
    ax3.legend(loc="upper left", fontsize=7, ncol=2)
    ax3_twin = ax3.twinx()
    ax3_twin.set_ylim(ax3.get_ylim())
    ax3_twin.yaxis.set_major_formatter(FuncFormatter(lambda p, _: f"{p * mean_batch_size:.0f}"))
    ax3_twin.set_ylabel("Count")
    ax3.set_title("Genre labels")
    ax3.grid(True, alpha=0.3)

    # One tick per batch, integer batch ids only (no float batch ids)
    for ax in (ax1, ax2, ax3):
        ax.set_xticks(batch_ids)

    plt.tight_layout()
    pdf_path = os.path.join(batch_output_root, "batch_label_stats.pdf")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Batch label plot written to {pdf_path}")


def _run_one_batch(args: tuple) -> tuple[int, bool]:
    """
    Run midi2edgelist then edgelist2vec for one batch (single-threaded).
    args = (batch_id, list_of_file_paths, xmidi_dir, batch_output_root, dimensions, reset).
    Returns (batch_id, success).
    """
    (
        batch_id,
        file_paths,
        xmidi_dir,
        batch_output_root,
        dimensions,
        reset,
    ) = args
    batch_dir = os.path.join(batch_output_root, f"batch_{batch_id}")
    embeddings_bin = os.path.join(batch_dir, "embeddings.bin")
    if not reset and os.path.isfile(embeddings_bin):
        return (batch_id, True)
    ensure_dir(batch_dir)
    edgelist_dir = os.path.join(batch_dir, "edgelist")
    # Check if edgelists already exist (names.csv is created by midi2edgelist)
    names_csv = os.path.join(edgelist_dir, "names.csv")
    if not reset and os.path.isfile(names_csv):
        # Edgelists already exist, skip midi2edgelist and go straight to edgelist2vec
        logging.info(f"Batch {batch_id}: edgelists already exist, skipping midi2edgelist")
    else:
        # Run midi2edgelist to create edgelists
        path_list = [Path(p) for p in file_paths]
        if not run_midi2edgelist_for_files(path_list, xmidi_dir, edgelist_dir, show_progress=False):
            return (batch_id, False)
    if not run_edgelist2vec(
        edgelist_dir,
        embeddings_bin,
        dimensions=dimensions,
        show_progress=False,
        workers=1,
    ):
        return (batch_id, False)
    names_src = os.path.join(edgelist_dir, "names.csv")
    names_dst = os.path.join(batch_dir, "names.csv")
    if os.path.isfile(names_src):
        shutil.copy2(names_src, names_dst)
    return (batch_id, True)


def preprocess_xmidi_midi2vec_batched(
    xmidi_dir: str,
    output_dir: str,
    batch_output_root: str = None,
    num_batches: int = 50,
    seed: int = 42,
    dimensions: int = 100,
    reset: bool = False,
    show_progress: bool = True,
    num_workers: int = None,
):
    """
    Preprocess XMIDI with batched midi2vec: stratified assignment, parallel batch runs, then consolidate.

    Batches are stable across runs: when generating, assignment is deterministic (fixed seed); when
    batch_assignments.csv already exists and --reset is not set, stratification is skipped and that
    file is loaded, so the same batches are used every time.

    Each batch is processed with one core (midi2edgelist + edgelist2vec run single-threaded per batch).
    num_workers is how many batches run in parallel (Pool size). Total cores used ≈ num_workers.

    Args:
        xmidi_dir: Directory containing XMIDI MIDI files
        output_dir: Directory for consolidated .safetensors (latents dir)
        batch_output_root: Where to write batch_*/ and batch_assignments.csv. If None, uses MIDI2VEC_BATCHES_DIR.
        num_batches: Number of batches; each (emotion, genre) group is distributed round-robin (shuffled) so proportions are similar across batches
        seed: Random seed for stratified shuffle
        dimensions: Embedding dimension
        reset: If True, recompute all batches; otherwise skip batches that already have embeddings.bin
        show_progress: If True, show tqdm over batches
        num_workers: Number of batches to run in parallel (each batch = 1 core). Default: min(num_batches, cpu_count).
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if batch_output_root is None:
        batch_output_root = MIDI2VEC_BATCHES_DIR
    ensure_dir(batch_output_root)
    xmidi_path = Path(xmidi_dir).resolve()
    assignments_path = os.path.join(batch_output_root, "batch_assignments.csv")

    # If batch assignments already exist and we're not resetting, skip stratification and
    # file discovery entirely — use the previously generated batches (same across runs).
    if not reset and os.path.isfile(assignments_path):
        with open(assignments_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = [(row["file_path"].strip().strip('"'), int(row["batch_id"].strip())) for row in reader]
        batch_id_to_files = {}
        for fp, bid in rows:
            batch_id_to_files.setdefault(bid, []).append(fp)
        batches = [batch_id_to_files[i] for i in sorted(batch_id_to_files)]
        logging.info(f"Loaded {assignments_path} ({len(batches)} batches, skipping stratification)")
    else:
        # Discover files and run stratified batch assignment (deterministic for fixed seed).
        files = find_xmidi_files(str(xmidi_path))
        if not files:
            logging.warning(f"No MIDI files found in {xmidi_dir}")
            return
        files = [str(Path(f).resolve()) for f in files]
        batches, assignments = _stratified_batch_assignment(files, num_batches, seed)
        with open(assignments_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["file_path", "batch_id"])
            for fp, bid in assignments:
                w.writerow([fp, bid])
        logging.info(f"Wrote {assignments_path} ({len(batches)} stratified batches, {len(files)} files)")
    _compute_and_log_batch_label_statistics(batches, batch_output_root)
    n_workers = num_workers
    if n_workers is None or n_workers <= 0:
        n_workers = min(len(batches), cpu_count() or 1)
    n_workers = min(n_workers, len(batches))
    worker_args = [
        (i, batch_files, str(xmidi_path), batch_output_root, dimensions, reset)
        for i, batch_files in enumerate(batches)
    ]
    root_logger = logging.getLogger()
    old_level = root_logger.level
    root_logger.setLevel(logging.WARNING)
    try:
        with Pool(processes=n_workers) as pool:
            results = list(
                tqdm(
                    pool.imap_unordered(_run_one_batch, worker_args),
                    total=len(worker_args),
                    desc="batches",
                    disable=not show_progress,
                )
            )
    finally:
        root_logger.setLevel(old_level)
    failed = [bid for bid, ok in results if not ok]
    if failed:
        logging.error(f"Failed batches: {failed}")
        return
    count = consolidate_batched_embeddings_to_safetensors(batch_output_root, output_dir)
    logging.info(f"Consolidated {count} embeddings to {output_dir}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Preprocess XMIDI with batched midi2vec")
    parser.add_argument("--xmidi_dir", required=True, help="Directory containing XMIDI MIDI files")
    parser.add_argument("--output_dir", default=XMIDI_LATENTS_DIR, help="Output directory for latents")
    parser.add_argument("--batch_output_root", default=None, help="Batch output root (default: MIDI2VEC_BATCHES_DIR)")
    parser.add_argument("--num_batches", type=int, default=50, help="Number of batches (stratified round-robin per label group)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for stratified assignment")
    parser.add_argument("--dimensions", type=int, default=64, help="Embedding dimension")
    parser.add_argument("--reset", action="store_true", help="Recompute all batches (default: skip existing)")
    parser.add_argument("--no_show_progress", action="store_true", help="Disable tqdm over batches")
    parser.add_argument("--num_workers", type=int, default=None, help="Batches to run in parallel (each batch = 1 core; default: min(batches, cpus))")
    args = parser.parse_args()
    preprocess_xmidi_midi2vec_batched(
        args.xmidi_dir,
        args.output_dir,
        batch_output_root=args.batch_output_root,
        num_batches=args.num_batches,
        seed=args.seed,
        dimensions=args.dimensions,
        reset=args.reset,
        show_progress=not args.no_show_progress,
        num_workers=args.num_workers,
    )
