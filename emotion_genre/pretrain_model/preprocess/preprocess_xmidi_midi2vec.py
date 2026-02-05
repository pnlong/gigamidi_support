"""
Preprocess XMIDI with midi2vec: extract graph embeddings from MIDI files.

midi2vec is transductive: embeddings are computed for the corpus as a whole.
No pretrained model exists for new files. We run midi2edgelist + edgelist2vec
on the XMIDI directory, or load from precomputed embeddings.bin + names.csv.

OUTPUT: Same as MuseTok - {filename}.safetensors with shape (1, dim) for dataset compatibility.
"""

import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from utils.data_utils import XMIDI_LATENTS_DIR, ensure_dir, MIDI2VEC_EMBEDDINGS_DIR
from utils.midi2vec_utils import (
    run_midi2edgelist,
    run_edgelist2vec,
    extract_embeddings_to_safetensors,
)


def find_xmidi_files(xmidi_dir: str):
    """Find all XMIDI MIDI files."""
    files = []
    xmidi_path = Path(xmidi_dir)
    for ext in ['*.midi', '*.mid', '*.MIDI', '*.MID']:
        for file_path in xmidi_path.rglob(ext):
            if str(file_path).lower().endswith(('.mid', '.midi')):
                files.append(str(file_path))
    return sorted(files)


def get_processed_files(output_dir: str):
    """Get set of already-processed files."""
    processed = set()
    if os.path.exists(output_dir):
        for f in Path(output_dir).glob("*.safetensors"):
            processed.add(f.stem)
    return processed


def preprocess_xmidi_midi2vec(
    xmidi_dir: str,
    output_dir: str,
    precomputed_dir: str = None,
    dimensions: int = 100,
    resume: bool = False,
    show_progress: bool = True,
    workers: int = 0,
):
    """
    Preprocess XMIDI with midi2vec.
    
    If precomputed_dir contains embeddings.bin and names.csv (for XMIDI ids),
    loads and extracts to output_dir. Otherwise runs full pipeline:
    midi2edgelist -> edgelist2vec -> extract.
    
    Args:
        xmidi_dir: Directory containing XMIDI MIDI files
        output_dir: Output directory for .safetensors files
        precomputed_dir: Dir with embeddings.bin and names.csv. If None, uses MIDI2VEC_EMBEDDINGS_DIR.
        dimensions: Embedding dimension (default 100)
        resume: If True, skip if all expected files already exist in output_dir
        show_progress: If True, stream midi2edgelist/edgelist2vec output to terminal
        workers: Number of parallel workers for midi2edgelist and edgelist2vec (1 = single core; 0 = use all CPUs)
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    if precomputed_dir is None:
        precomputed_dir = MIDI2VEC_EMBEDDINGS_DIR
    
    embeddings_bin = os.path.join(precomputed_dir, "embeddings.bin")
    names_csv = os.path.join(precomputed_dir, "names.csv")
    has_precomputed = os.path.isfile(embeddings_bin) and os.path.isfile(names_csv)
    
    if has_precomputed:
        logging.info(f"Using precomputed embeddings from {precomputed_dir}")
        ensure_dir(output_dir)
        count = extract_embeddings_to_safetensors(
            embeddings_bin, names_csv, output_dir
        )
        logging.info(f"Extracted {count} embeddings to {output_dir}")
        return
    
    # Run full pipeline
    logging.info("Running midi2vec pipeline (midi2edgelist -> edgelist2vec -> extract)")
    files = find_xmidi_files(xmidi_dir)
    if len(files) == 0:
        logging.warning(f"No MIDI files found in {xmidi_dir}")
        return
    
    logging.info(f"Found {len(files)} MIDI files")
    ensure_dir(precomputed_dir)
    edgelist_dir = os.path.join(precomputed_dir, "edgelist")
    ensure_dir(edgelist_dir)
    
    # Step 1: midi2edgelist (skip if edgelist output exists and resuming)
    names_csv_path = os.path.join(edgelist_dir, "names.csv")
    edgelist_done = os.path.isfile(names_csv_path)
    if edgelist_done and resume:
        logging.info(f"Edgelist output already exists at {edgelist_dir}, skipping midi2edgelist")
    else:
        if not run_midi2edgelist(xmidi_dir, edgelist_dir, show_progress=show_progress, workers=workers):
            logging.error("midi2edgelist failed. Aborting.")
            return
    
    # Step 2: edgelist2vec
    embeddings_output = os.path.join(precomputed_dir, "embeddings.bin")
    if not run_edgelist2vec(edgelist_dir, embeddings_output, dimensions=dimensions, show_progress=show_progress, workers=workers):
        logging.error("edgelist2vec failed. Aborting.")
        return
    
    # names.csv is produced by midi2edgelist in edgelist_dir
    names_csv = os.path.join(edgelist_dir, "names.csv")
    if not os.path.isfile(names_csv):
        logging.error(f"names.csv not found at {names_csv}")
        return
    
    # Step 3: Extract to safetensors
    ensure_dir(output_dir)
    if resume:
        processed = get_processed_files(output_dir)
        expected_count = len(files)
        if len(processed) >= expected_count:
            logging.info(f"Resume: {len(processed)} files already in {output_dir}, skipping")
            return
    
    count = extract_embeddings_to_safetensors(
        embeddings_output, names_csv, output_dir
    )
    logging.info(f"Extracted {count} embeddings to {output_dir}")
    
    # Copy embeddings.bin and names.csv to precomputed_dir root for future use
    import shutil
    dest_bin = os.path.join(precomputed_dir, "embeddings.bin")
    dest_csv = os.path.join(precomputed_dir, "names.csv")
    if embeddings_output != dest_bin:
        shutil.copy(embeddings_output, dest_bin)
    if names_csv != dest_csv:
        shutil.copy(names_csv, dest_csv)
    logging.info(f"Saved embeddings to {precomputed_dir} for future precomputed use")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess XMIDI with midi2vec")
    parser.add_argument("--xmidi_dir", required=True, help="Directory containing XMIDI MIDI files")
    parser.add_argument("--output_dir", default=XMIDI_LATENTS_DIR, help="Output directory for latents")
    parser.add_argument("--precomputed", default=None,
                        help="Path to dir with embeddings.bin and names.csv (optional)")
    parser.add_argument("--dimensions", type=int, default=100, help="Embedding dimension")
    parser.add_argument("--reset", action="store_true",
                        help="Reset: recompute everything (default: resume, skip existing output)")
    parser.add_argument("--no_show_progress", action="store_true",
                        help="Suppress progress output from midi2edgelist/edgelist2vec")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for midi2edgelist and edgelist2vec (1 = single core; 0 = use all CPUs)")
    args = parser.parse_args()
    
    preprocess_xmidi_midi2vec(
        args.xmidi_dir, args.output_dir,
        precomputed_dir=args.precomputed,
        dimensions=args.dimensions,
        resume=not args.reset,
        show_progress=not args.no_show_progress,
        workers=args.workers,
    )
