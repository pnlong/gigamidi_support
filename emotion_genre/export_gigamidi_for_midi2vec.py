"""
Export GigaMIDI to disk and run midi2vec pipeline for precomputed embeddings.

Streams GigaMIDI, saves each song as {md5}.mid in a flat directory, then runs
midi2edgelist and edgelist2vec to produce embeddings.bin and names.csv.
These can be used with annotate_gigamidi.py --preprocessor midi2vec.

Output: embeddings.bin and names.csv in --output_dir (default: MIDI2VEC_EMBEDDINGS_DIR)
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.data_utils import MIDI2VEC_EMBEDDINGS_DIR, ensure_dir
from utils.midi2vec_utils import run_midi2edgelist, run_edgelist2vec


def main():
    parser = argparse.ArgumentParser(
        description="Export GigaMIDI and run midi2vec for precomputed embeddings"
    )
    parser.add_argument("--output_dir", default=None,
                        help="Output dir for embeddings.bin, names.csv (default: MIDI2VEC_EMBEDDINGS_DIR)")
    parser.add_argument("--split", default="train",
                        help="GigaMIDI split (train, test, validation)")
    parser.add_argument("--streaming", action="store_true", default=True,
                        help="Use streaming mode")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Max samples to export (for testing)")
    parser.add_argument("--dimensions", type=int, default=100,
                        help="Embedding dimension")
    parser.add_argument("--skip_export", action="store_true",
                        help="Skip export; only run midi2vec on existing midi_dir")
    parser.add_argument("--midi_dir", default=None,
                        help="Existing dir with {md5}.mid files (for --skip_export)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for midi2edgelist and edgelist2vec (1 = single core; 0 = use all CPUs)")
    args = parser.parse_args()
    
    output_dir = args.output_dir or MIDI2VEC_EMBEDDINGS_DIR
    ensure_dir(output_dir)
    
    midi_dir = os.path.join(output_dir, "gigamidi_midis")
    
    if not args.skip_export:
        ensure_dir(midi_dir)
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        logging.info(f"Exporting GigaMIDI to {midi_dir}...")
        
        if args.streaming:
            dataset = load_dataset("Metacreation/GigaMIDI", split=args.split, streaming=True)
        else:
            dataset = load_dataset("Metacreation/GigaMIDI", split=args.split)
        
        count = 0
        iterator = iter(dataset)
        if args.max_samples:
            iterator = tqdm(iterator, total=args.max_samples, desc="Exporting")
        else:
            iterator = tqdm(iterator, desc="Exporting")
        
        for sample in iterator:
            if args.max_samples and count >= args.max_samples:
                break
            md5 = sample.get("md5", "")
            if not md5:
                continue
            midi_path = os.path.join(midi_dir, f"{md5}.mid")
            if os.path.exists(midi_path):
                count += 1
                continue
            try:
                midi_bytes = sample["music"]
                with open(midi_path, "wb") as f:
                    f.write(midi_bytes)
                count += 1
            except Exception as e:
                logging.warning(f"Failed to save {md5}: {e}")
        
        logging.info(f"Exported {count} MIDI files to {midi_dir}")
    else:
        midi_dir = args.midi_dir or midi_dir
        if not os.path.isdir(midi_dir):
            raise FileNotFoundError(f"midi_dir not found: {midi_dir}")
    
    # Run midi2vec pipeline
    edgelist_dir = os.path.join(output_dir, "edgelist")
    ensure_dir(edgelist_dir)
    names_csv_path = os.path.join(edgelist_dir, "names.csv")
    edgelist_done = os.path.isfile(names_csv_path)
    if edgelist_done:
        logging.info(f"Edgelist output already exists at {edgelist_dir}, skipping midi2edgelist")
    else:
        logging.info("Running midi2edgelist...")
        if not run_midi2edgelist(midi_dir, edgelist_dir, workers=args.workers):
            logging.error("midi2edgelist failed")
            sys.exit(1)
    
    logging.info("Running edgelist2vec...")
    embeddings_bin = os.path.join(output_dir, "embeddings.bin")
    if not run_edgelist2vec(edgelist_dir, embeddings_bin, dimensions=args.dimensions, workers=args.workers):
        logging.error("edgelist2vec failed")
        sys.exit(1)
    
    # Copy names.csv to output_dir root for convenience
    import shutil
    names_src = os.path.join(edgelist_dir, "names.csv")
    names_dst = os.path.join(output_dir, "names.csv")
    if names_src != names_dst and os.path.isfile(names_src):
        shutil.copy(names_src, names_dst)
    
    logging.info(f"Done. embeddings.bin and names.csv saved to {output_dir}")
    logging.info("Use: annotate_gigamidi.py --preprocessor midi2vec --embeddings_dir " + output_dir)


if __name__ == "__main__":
    main()
