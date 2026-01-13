"""
Annotate GigaMIDI dataset with emotion and genre predictions.
Uses streaming mode to avoid downloading entire dataset.
Writes CSV incrementally to avoid losing progress.
Supports resume by skipping already-processed songs.
Song-level predictions (one emotion and one genre per song).
"""

import argparse
import logging
import sys
import os
from os.path import dirname, realpath
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, dirname(dirname(realpath(__file__))))

from utils.musetok_utils import load_musetok_model, extract_latents_from_midi
from pretrain_model.model import EmotionGenreClassifier
from utils.data_utils import (
    TRAINED_MODEL_DIR, MUSETOK_CHECKPOINT_DIR,
    ensure_dir, load_json
)

def parse_args():
    parser = argparse.ArgumentParser(prog="AnnotateGigaMIDI", description="Annotate GigaMIDI with emotion/genre predictions.")
    
    parser.add_argument("--emotion_model_path", type=str, required=True,
                       help="Path to trained emotion model checkpoint")
    parser.add_argument("--genre_model_path", type=str, required=True,
                       help="Path to trained genre model checkpoint")
    parser.add_argument("--emotion_class_to_index_path", type=str, required=True,
                       help="Path to emotion_to_index.json")
    parser.add_argument("--genre_class_to_index_path", type=str, required=True,
                       help="Path to genre_to_index.json")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to MuseTok checkpoint (defaults to MUSETOK_CHECKPOINT_DIR/best_tokenizer.pt)")
    parser.add_argument("--vocab_path", type=str, default=None,
                       help="Path to MuseTok vocabulary")
    parser.add_argument("--input_dim", type=int, default=128,
                       help="Input dimension for model (should match latent dim)")
    parser.add_argument("--emotion_num_classes", type=int, default=11,
                       help="Number of emotion classes")
    parser.add_argument("--genre_num_classes", type=int, default=6,
                       help="Number of genre classes")
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU (CUDA); if not provided, use CPU")
    parser.add_argument("--streaming", action="store_true", default=True,
                       help="Use streaming mode")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output CSV file path")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Resume from existing CSV file (skip already-processed songs)")
    parser.add_argument("--velocity", action="store_true", default=False,
                       help="Prefer velocity vocabulary for MuseTok")
    
    args = parser.parse_args()
    if args.hidden_dim is None:
        args.hidden_dim = args.input_dim // 2
    return args

def load_existing_annotations(csv_path):
    """Load existing annotations from CSV and return set of processed md5s."""
    processed = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'md5' in df.columns:
                processed = set(df['md5'].dropna())
            logging.info(f"Loaded {len(processed)} existing song annotations from {csv_path}")
        except Exception as e:
            logging.warning(f"Error loading existing CSV: {e}. Starting fresh.")
    return processed

def process_song(sample, emotion_model, genre_model, musetok_model, vocab, device,
                emotion_index_to_class, genre_index_to_class):
    """
    Process a single song: extract latents on-the-fly and predict emotion/genre.
    
    Returns:
        dict with 'md5', 'emotion', 'emotion_prob', 'genre', 'genre_prob' or None if error
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
            midi_bytes, musetok_model, vocab, has_velocity=True
        )
        
        if len(latents) == 0:
            logging.debug(f"No latents extracted for {md5}")
            return None
        
        # Mean pool across bars for song-level prediction
        latents_pooled = torch.from_numpy(latents).float().mean(dim=0).unsqueeze(0).to(device)  # (1, latent_dim)
        
        # Predict emotion and genre
        with torch.no_grad():
            emotion_logits = emotion_model(latents_pooled)  # (1, num_emotion_classes)
            genre_logits = genre_model(latents_pooled)  # (1, num_genre_classes)
            
            # Apply softmax to get probabilities
            emotion_probs = F.softmax(emotion_logits, dim=1)
            genre_probs = F.softmax(genre_logits, dim=1)
            
            # Get predicted class indices
            emotion_pred_idx = torch.argmax(emotion_logits, dim=1).item()
            genre_pred_idx = torch.argmax(genre_logits, dim=1).item()
            
            # Get class names
            emotion_pred = emotion_index_to_class[emotion_pred_idx]
            genre_pred = genre_index_to_class[genre_pred_idx]
            
            # Get max probabilities (confidence)
            emotion_prob = float(emotion_probs[0, emotion_pred_idx].item())
            genre_prob = float(genre_probs[0, genre_pred_idx].item())
        
        return {
            "md5": md5,
            "emotion": emotion_pred,
            "emotion_prob": emotion_prob,
            "genre": genre_pred,
            "genre_prob": genre_prob,
        }
    except Exception as e:
        logging.warning(f"Error processing song {sample.get('md5', 'unknown')}: {e}")
        return None

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Using device: {device}")
    
    # Initialize CSV file
    if args.output_path is None:
        output_dir = os.path.join(os.environ.get("XMIDI_STORAGE_DIR", "/deepfreeze/pnlong/gigamidi"), 
                                  "xmidi_emotion_genre", "gigamidi_annotations")
        ensure_dir(output_dir)
        args.output_path = os.path.join(output_dir, "annotations.csv")
    else:
        ensure_dir(os.path.dirname(args.output_path))
    
    # Load existing annotations if resuming
    processed = set()
    if args.resume and os.path.exists(args.output_path):
        try:
            processed = load_existing_annotations(args.output_path)
            logging.info(f"Resuming: found {len(processed)} existing song annotations")
        except Exception as e:
            logging.warning(f"Error loading existing CSV: {e}. Starting fresh.")
    else:
        # Write column names if not resuming (overwrite existing file)
        df_header = pd.DataFrame(columns=['md5', 'emotion', 'emotion_prob', 'genre', 'genre_prob'])
        df_header.to_csv(args.output_path, mode='w', index=False)
        logging.info(f"Initialized CSV file with headers: {args.output_path}")
    
    # Load class mappings
    emotion_class_to_index = load_json(args.emotion_class_to_index_path)
    genre_class_to_index = load_json(args.genre_class_to_index_path)
    emotion_index_to_class = {v: k for k, v in emotion_class_to_index.items()}
    genre_index_to_class = {v: k for k, v in genre_class_to_index.items()}
    
    # Load emotion model
    logging.info("Loading emotion prediction model...")
    emotion_model = EmotionGenreClassifier(
        input_dim=args.input_dim,
        num_classes=args.emotion_num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    emotion_model.load_state_dict(torch.load(args.emotion_model_path, map_location=device))
    emotion_model.eval()
    
    # Load genre model
    logging.info("Loading genre prediction model...")
    genre_model = EmotionGenreClassifier(
        input_dim=args.input_dim,
        num_classes=args.genre_num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    genre_model.load_state_dict(torch.load(args.genre_model_path, map_location=device))
    genre_model.eval()
    
    # Load MuseTok model
    logging.info("Loading MuseTok model...")
    musetok_model, vocab, use_velocity = load_musetok_model(
        checkpoint_path=args.checkpoint_path,
        vocab_path=args.vocab_path,
        use_gpu=args.gpu,
        prefer_velocity=args.velocity,
    )
    logging.info(f"MuseTok model loaded successfully (velocity support: {use_velocity})")
    
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
            
            # Skip if already processed
            if md5 in processed:
                skipped += 1
                continue
            
            # Process song (returns dict with predictions)
            result = process_song(
                sample, emotion_model, genre_model, musetok_model, vocab, device,
                emotion_index_to_class, genre_index_to_class
            )
            
            if result is not None:
                # Write to CSV
                df_row = pd.DataFrame([result])
                df_row.to_csv(args.output_path, mode='a', header=False, index=False)
                processed.add(md5)
                count += 1
                
                # Log progress periodically
                if count % 100 == 0:
                    logging.info(f"Processed {count} songs (skipped: {skipped}, errors: {errors})")
            else:
                errors += 1
    
    except KeyboardInterrupt:
        logging.info("\nInterrupted by user. Saving progress...")
    
    logging.info(f"\nProcessing complete!")
    logging.info(f"Successfully processed: {count} songs")
    logging.info(f"Skipped (already processed): {skipped} songs")
    logging.info(f"Errors: {errors} songs")
    logging.info(f"Annotations saved to: {args.output_path}")
