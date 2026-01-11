"""
Annotate GigaMIDI dataset with valence/arousal predictions.
Uses streaming mode to avoid downloading entire dataset.
Writes CSV incrementally to avoid losing progress.
Supports resume by skipping already-processed songs.
"""

import argparse
import logging
import sys
import os
from os.path import dirname, realpath
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, dirname(dirname(realpath(__file__))))

from utils.musetok_utils import load_musetok_model, extract_latents_from_midi
from utils.midi_utils import load_midi_symusic
from pretrain_model.model import ValenceArousalMLP
from utils.data_utils import (
    TRAINED_MODEL_DIR, MUSETOK_CHECKPOINT_DIR, GIGAMIDI_ANNOTATIONS_DIR,
    ensure_dir
)

def parse_args():
    parser = argparse.ArgumentParser(prog="AnnotateGigaMIDI", description="Annotate GigaMIDI with VA predictions.")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained VA model checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to MuseTok checkpoint")
    parser.add_argument("--vocab_path", type=str, default=None,
                       help="Path to MuseTok vocabulary")
    parser.add_argument("--input_dim", type=int, default=128,
                       help="Input dimension for model (should match latent dim)")
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension")
    parser.add_argument("--use_tanh", action="store_true", default=True,
                       help="Use tanh activation")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")
    parser.add_argument("--streaming", action="store_true", default=True,
                       help="Use streaming mode")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output CSV file path (defaults to <STORAGE_DIR>/gigamidi_annotations/annotations.csv)")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Resume from existing CSV file (skip already-processed songs)")
    
    args = parser.parse_args()
    if args.hidden_dim is None:
        args.hidden_dim = args.input_dim // 2
    if args.output_path is None:
        ensure_dir(GIGAMIDI_ANNOTATIONS_DIR)
        args.output_path = os.path.join(GIGAMIDI_ANNOTATIONS_DIR, "annotations.csv")
    return args

def load_existing_annotations(csv_path):
    """Load existing annotations from CSV and return set of processed (md5, bar_number) pairs."""
    processed = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'md5' in df.columns and 'bar_number' in df.columns:
                # Create set of (md5, bar_number) tuples
                processed = set(zip(df['md5'].dropna(), df['bar_number'].dropna()))
            elif 'md5' in df.columns:
                # Fallback: if no bar_number column, just use md5 (for backward compatibility)
                processed = set(df['md5'].dropna())
            logging.info(f"Loaded {len(processed)} existing bar annotations from {csv_path}")
        except Exception as e:
            logging.warning(f"Error loading existing CSV: {e}. Starting fresh.")
    return processed

def process_song(sample, model, musetok_model, vocab, device):
    """
    Process a single song: extract latents on-the-fly and predict VA for each bar.
    
    Bar boundaries are determined by MuseTok's bar detection logic (from Phase 1):
    - Bars respect time signatures (one bar per n beats where n is the numerator)
    - New bars start when time signatures change
    - Bar positions come from REMI events where 'Bar' events mark boundaries
    
    Returns:
        list of dicts, each with 'md5', 'bar_number', 'valence', 'arousal' or None if error
    """
    try:
        md5 = sample.get("md5", "")
        if not md5:
            logging.warning("Sample missing md5, skipping")
            return None
        
        # Load MIDI from bytes
        midi_bytes = sample["music"]
        
        # Extract latents on-the-fly
        latents, bar_positions = extract_latents_from_midi(
            midi_bytes, musetok_model, vocab, device
        )
        
        if len(latents) == 0:
            logging.debug(f"No latents extracted for {md5}")
            return None
        
        # Convert to tensor
        latents_tensor = torch.from_numpy(latents).float().unsqueeze(0).to(device)  # (1, n_bars, dim)
        mask = torch.ones(1, len(latents), dtype=torch.bool).to(device)
        
        # Predict VA for each bar
        with torch.no_grad():
            outputs = model(latents_tensor, mask=mask)
            # Model outputs shape: (batch, seq_len, 2) for per-bar predictions
            if len(outputs.shape) == 3:
                # Per-bar predictions: (1, n_bars, 2)
                n_bars = outputs.shape[1]
                results = []
                for bar_idx in range(n_bars):
                    results.append({
                        "md5": md5,
                        "bar_number": bar_idx,
                        "valence": float(outputs[0, bar_idx, 0].cpu().item()),
                        "arousal": float(outputs[0, bar_idx, 1].cpu().item()),
                    })
                return results
            else:
                # Fallback: if model outputs single prediction, assign to bar 0
                return [{
                    "md5": md5,
                    "bar_number": 0,
                    "valence": float(outputs[0, 0].cpu().item()),
                    "arousal": float(outputs[0, 1].cpu().item()),
                }]
    except Exception as e:
        logging.warning(f"Error processing song {sample.get('md5', 'unknown')}: {e}")
        return None

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Using device: {device}")
    
    # Initialize CSV file
    ensure_dir(os.path.dirname(args.output_path))
    
    # Load existing annotations to get processed (md5, bar_number) pairs (only if resuming)
    processed = set()
    if args.resume and os.path.exists(args.output_path):
        try:
            processed = load_existing_annotations(args.output_path)
            logging.info(f"Resuming: found {len(processed)} existing bar annotations")
        except Exception as e:
            logging.warning(f"Error loading existing CSV: {e}. Starting fresh.")
    else:
        # Write column names if not resuming (overwrite existing file)
        df_header = pd.DataFrame(columns=['md5', 'bar_number', 'valence', 'arousal'])
        df_header.to_csv(args.output_path, mode='w', index=False)
        logging.info(f"Initialized CSV file with headers: {args.output_path}")
    
    # Load VA model
    logging.info("Loading VA prediction model...")
    va_model = ValenceArousalMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        use_tanh=args.use_tanh,
        dropout=args.dropout,
    ).to(device)
    va_model.load_state_dict(torch.load(args.model_path, map_location=device))
    va_model.eval()
    
    # Load MuseTok model
    logging.info("Loading MuseTok model...")
    musetok_model, vocab = load_musetok_model(
        checkpoint_path=args.checkpoint_path,
        vocab_path=args.vocab_path,
        device=str(device),
    )
    
    # Load GigaMIDI dataset
    logging.info(f"Loading GigaMIDI dataset (split={args.split}, streaming={args.streaming})...")
    if args.streaming:
        dataset = load_dataset("Metacreation/GigaMIDI", split=args.split, streaming=True)
    else:
        dataset = load_dataset("Metacreation/GigaMIDI", split=args.split)
    
    # Process songs
    count = 0
    skipped = 0
    errors = 0
    
    iterator = iter(dataset)
    if args.max_samples:
        iterator = tqdm(iterator, total=args.max_samples, desc="Processing")
    else:
        iterator = tqdm(iterator, desc="Processing")
    
    try:
        for sample in iterator:
            if args.max_samples and count >= args.max_samples:
                break
            
            md5 = sample.get("md5", "")
            if not md5:
                errors += 1
                continue
            
            # Process song (returns list of bar-level predictions)
            results = process_song(sample, va_model, musetok_model, vocab, device)
            
            if results is not None and len(results) > 0:
                # Filter out already-processed bars if resuming
                bars_to_write = []
                for result in results:
                    bar_key = (result["md5"], result["bar_number"])
                    if bar_key not in processed:
                        bars_to_write.append(result)
                        processed.add(bar_key)
                
                if bars_to_write:
                    # Write all bars for this song to CSV
                    df_rows = pd.DataFrame(bars_to_write)
                    df_rows.to_csv(args.output_path, mode='a', header=False, index=False)
                    count += len(bars_to_write)
                else:
                    skipped += 1
                
                # Log progress periodically
                if count % 1000 == 0:  # Changed to 1000 since we're counting bars now
                    logging.info(f"Processed {count} bars (skipped songs: {skipped}, errors: {errors})")
            else:
                errors += 1
    
    logging.info(f"\nProcessing complete!")
    logging.info(f"Successfully processed: {count} bars")
    logging.info(f"Skipped (already processed): {skipped} songs")
    logging.info(f"Errors: {errors} songs")
    logging.info(f"Annotations saved to: {args.output_path}")