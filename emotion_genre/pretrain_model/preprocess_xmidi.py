"""
Preprocess XMIDI dataset: extract MuseTok latents from MIDI files.

XMIDI DATASET STRUCTURE:
========================
- Format: MIDI files (.midi)
- Naming: XMIDI_<Emotion>_<Genre>_<ID_len_8>.midi
- Structure: Flat directory (no train/valid/test splits - created later)

OUTPUT STRUCTURE:
- Output directory: output_dir/{filename}.safetensors
- Metadata includes: emotion, genre, ID extracted from filename

RESUME LOGIC:
- Checks for existing .safetensors files in output directory
- Matches by filename (without extension) to determine if already processed
- If resume=True, skips files that already have corresponding output files
"""

import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import glob
import re
import torch
import numpy as np
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import (
    XMIDI_LATENTS_DIR, ensure_dir, save_latents,
    MUSETOK_CHECKPOINT_DIR
)
from utils.musetok_utils import (
    load_musetok_model, extract_latents_from_midi
)
from utils.midi_utils import midi_to_events_symusic, load_midi_symusic


def extract_metadata_from_filename(filename: str):
    """
    Extract emotion, genre, and ID from XMIDI filename.
    
    Format: XMIDI_<Emotion>_<Genre>_<ID_len_8>.midi
    
    Returns:
        (emotion, genre, id_str) or (None, None, None) if parsing fails
    """
    # Remove extension
    name = Path(filename).stem
    
    # Pattern: XMIDI_<Emotion>_<Genre>_<ID_len_8>
    pattern = r'^XMIDI_([^_]+)_([^_]+)_([a-zA-Z0-9]{8})$'
    match = re.match(pattern, name)
    
    if match:
        emotion = match.group(1)
        genre = match.group(2)
        id_str = match.group(3)
        return emotion, genre, id_str
    
    # Try alternative patterns
    parts = name.split('_')
    if len(parts) >= 4 and parts[0] == 'XMIDI':
        emotion = parts[1]
        genre = parts[2]
        id_str = '_'.join(parts[3:])  # ID might have underscores
        return emotion, genre, id_str
    
    return None, None, None


def is_midi_file(filepath: str) -> bool:
    """Check if file is a MIDI file."""
    return filepath.lower().endswith(('.mid', '.midi'))


def process_single_file(file_path: str,
                      output_path: str,
                      musetok_model,
                      vocab: dict):
    """
    Process a single MIDI file.
    
    Args:
        file_path: Path to input MIDI file
        output_path: Path to output .safetensors file
        musetok_model: Pre-loaded MuseTokEncoder instance
        vocab: Vocabulary dictionary
    
    Returns:
        (filename, success, error_message)
    """
    try:
        filename = Path(file_path).stem
        
        # Extract metadata from filename
        emotion, genre, id_str = extract_metadata_from_filename(filename)
        
        # Full pipeline: MIDI → REMI → latents
        # Extract per-bar latents (shape: n_bars x 128)
        # We store per-bar latents to allow flexibility in pooling strategies later
        latents, bar_positions = extract_latents_from_midi(
            file_path, musetok_model, vocab,
            has_velocity=True  # XMIDI MIDI files should have velocity
        )
        
        if len(latents) == 0:
            return (filename, False, "No latents extracted (empty or invalid file)")
        
        # Save per-bar latents with metadata
        # latents shape: (n_bars, 128) - stored per-bar, not pooled
        # The dataset will handle pooling (mean, max, attention, etc.) at load time
        ensure_dir(os.path.dirname(output_path))
        metadata = {
            "n_bars": len(latents),
            "original_file_path": str(file_path),
            "file_type": "midi",
            "bar_positions": bar_positions,
            "emotion": emotion if emotion else "unknown",
            "genre": genre if genre else "unknown",
            "id": id_str if id_str else "unknown"
        }
        save_latents(output_path, latents, metadata)
        
        return (filename, True, None)
    except Exception as e:
        return (Path(file_path).stem, False, str(e))


def process_batch(file_paths: List[str],
                 output_dir: str,
                 musetok_model,
                 vocab: dict):
    """
    Process a batch of MIDI files in parallel on GPU.
    
    This function batches the latent extraction across multiple files for GPU parallelization.
    Each file is processed to get events/bar_positions, then segments are batched together.
    
    Args:
        file_paths: List of paths to MIDI files
        output_dir: Output directory for latents
        musetok_model: Pre-loaded MuseTokEncoder instance
        vocab: Vocabulary dictionary
    
    Returns:
        List of (filename, success, error_message) tuples
    """
    results = []
    batch_data = []  # Store (file_path, filename, events, bar_positions, metadata)
    
    # Step 1: Process all files to get events and bar_positions
    for file_path in file_paths:
        try:
            filename = Path(file_path).stem
            emotion, genre, id_str = extract_metadata_from_filename(filename)
            
            # Load MIDI and convert to events
            score = load_midi_symusic(file_path)
            bar_positions, events = midi_to_events_symusic(
                score,
                has_velocity=True,
                time_first=False,
                repeat_beat=True
            )
            
            batch_data.append({
                'file_path': file_path,
                'filename': filename,
                'events': events,
                'bar_positions': bar_positions,
                'emotion': emotion if emotion else "unknown",
                'genre': genre if genre else "unknown",
                'id': id_str if id_str else "unknown"
            })
        except Exception as e:
            results.append((Path(file_path).stem, False, str(e)))
    
    if not batch_data:
        return results
    
    # Step 2: Get segments for all files and batch them together
    all_segments = []
    segment_file_map = []  # Maps segment index to (file_idx, seg_idx, n_bars_in_file)
    file_n_bars = {}  # Store actual number of bars per file
    
    for file_idx, data in enumerate(batch_data):
        try:
            # Get segments for this file
            music_data = musetok_model.get_segments(data['events'], data['bar_positions'].copy())
            n_segments = music_data['n_segment']
            n_bars = music_data['n_bar']
            file_n_bars[file_idx] = n_bars
            
            # Store segment info for later reconstruction
            for seg_idx in range(n_segments):
                all_segments.append({
                    'enc_inp': music_data['enc_inp'][seg_idx],
                    'enc_padding_mask': music_data['enc_padding_mask'][seg_idx],
                })
                segment_file_map.append((file_idx, seg_idx, n_bars))
        except Exception as e:
            results.append((data['filename'], False, f"Failed to get segments: {str(e)}"))
            continue
    
    if not all_segments:
        return results
    
    # Step 3: Batch all segments together and process on GPU
    try:
        # Stack all segments into a batch
        batch_enc_inp = np.stack([seg['enc_inp'] for seg in all_segments])  # (n_segments, n_bars, seq_len)
        batch_enc_padding_mask = np.stack([seg['enc_padding_mask'] for seg in all_segments])  # (n_segments, n_bars, seq_len)
        
        # Process batch: (n_segments, n_bars, seq_len) -> permute to (seq_len, n_segments, n_bars)
        # Use musetok's numpy_to_tensor utility (accessed through the encoder's internal utils)
        # The encoder has access to numpy_to_tensor through its internal imports
        # We'll use torch directly for conversion
        enc_inp_tensor = torch.from_numpy(batch_enc_inp).to(musetok_model.device).permute(2, 0, 1).long()
        enc_padding_mask_tensor = torch.from_numpy(batch_enc_padding_mask).to(musetok_model.device).bool()
        
        # Process batch through model
        with torch.no_grad():
            latents_batch, indices_batch = musetok_model.model.get_batch_latent(
                enc_inp_tensor, enc_padding_mask_tensor, latent_from_encoder=False
            )
        
        # Convert back to numpy
        latents_batch = latents_batch.cpu().numpy()  # (n_segments, n_bars, latent_dim)
        indices_batch = indices_batch.cpu().numpy()  # (n_segments, n_bars, ...)
        
        # Step 4: Reconstruct latents for each file
        file_latents = {}  # file_idx -> list of latents per segment
        
        for seg_idx, (file_idx, seg_idx_in_file, n_bars) in enumerate(segment_file_map):
            if file_idx not in file_latents:
                file_latents[file_idx] = []
            # Extract latents for this segment (all bars in segment, typically 16)
            seg_latents = latents_batch[seg_idx]  # (model_max_bars, latent_dim)
            file_latents[file_idx].append(seg_latents)
        
        # Step 5: Concatenate segments for each file and save
        for file_idx, data in enumerate(batch_data):
            if file_idx not in file_latents:
                results.append((data['filename'], False, "No latents extracted"))
                continue
            
            # Concatenate all segments for this file
            # Result: per-bar latents (shape: n_bars x 128)
            all_file_latents = np.concatenate(file_latents[file_idx], axis=0)  # (total_bars, latent_dim)
            
            # Get actual number of bars (may be less than segments * max_bars)
            n_bars = file_n_bars[file_idx]
            latents = all_file_latents[:n_bars]  # Trim to actual number of bars
            
            if len(latents) == 0:
                results.append((data['filename'], False, "No latents extracted (empty or invalid file)"))
                continue
            
            # Save per-bar latents (not pooled)
            # The dataset will handle pooling (mean, max, attention, etc.) at load time
            output_path = os.path.join(output_dir, f"{data['filename']}.safetensors")
            ensure_dir(os.path.dirname(output_path))
            metadata = {
                "n_bars": len(latents),
                "original_file_path": str(data['file_path']),
                "file_type": "midi",
                "bar_positions": data['bar_positions'],
                "emotion": data['emotion'],
                "genre": data['genre'],
                "id": data['id']
            }
            save_latents(output_path, latents, metadata)
            results.append((data['filename'], True, None))
    
    except Exception as e:
        # If batch processing fails, fall back to individual processing
        logging.warning(f"Batch processing failed: {str(e)}. Falling back to individual processing.")
        for data in batch_data:
            if data['filename'] not in [r[0] for r in results]:
                # Process individually
                output_path = os.path.join(output_dir, f"{data['filename']}.safetensors")
                result = process_single_file(
                    data['file_path'], output_path, musetok_model, vocab
                )
                results.append(result)
    
    return results


def get_processed_files(output_dir: str):
    """
    Get set of already-processed files by checking for existing safetensors files.
    
    Returns:
        set of filenames (without extension) that have been processed
    """
    processed = set()
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.safetensors'):
                    # Extract filename without extension
                    filename = Path(file).stem
                    processed.add(filename)
    return processed


def find_xmidi_files(xmidi_dir: str):
    """
    Find all XMIDI MIDI files.
    
    Args:
        xmidi_dir: Base XMIDI directory
    
    Returns:
        list of file paths
    """
    files = []
    xmidi_path = Path(xmidi_dir)
    
    # Find all .midi files recursively
    for ext in ['*.midi', '*.mid', '*.MIDI', '*.MID']:
        for file_path in xmidi_path.rglob(ext):
            if is_midi_file(str(file_path)):
                files.append(str(file_path))
    
    return sorted(files)


def preprocess_xmidi(xmidi_dir: str,
                    output_dir: str,
                    checkpoint_path: str = None,
                    vocab_path: str = None,
                    use_gpu: bool = False,
                    resume: bool = False,
                    batch_size: int = 1):
    """
    Preprocess XMIDI dataset.
    
    Args:
        xmidi_dir: Directory containing XMIDI MIDI files
        output_dir: Output directory for latents
        checkpoint_path: Path to MuseTok checkpoint (defaults to MUSETOK_CHECKPOINT_DIR/best_tokenizer.pt)
        vocab_path: Path to vocabulary file (defaults to musetok/data/dictionary.pkl)
        use_gpu: If True, use CUDA; otherwise use CPU
        resume: If True, skip files that have already been processed
        batch_size: Number of files to process in parallel on GPU (only used if > 1 and use_gpu=True)
    """
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Convert use_gpu to device
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    # Set default paths
    if checkpoint_path is None:
        checkpoint_path = os.path.join(MUSETOK_CHECKPOINT_DIR, "best_tokenizer.pt")
    
    # Load MuseTok model (once, before processing)
    logging.info("Loading MuseTok model...")
    musetok_model, vocab, use_velocity = load_musetok_model(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        use_gpu=use_gpu
    )
    logging.info(f"MuseTok model loaded (vocabulary size: {len(vocab)}, velocity: {use_velocity})")
    
    # Find all MIDI files
    logging.info(f"Searching for MIDI files in {xmidi_dir}...")
    files = find_xmidi_files(xmidi_dir)
    logging.info(f"Found {len(files)} MIDI files")
    
    if len(files) == 0:
        logging.warning(f"No MIDI files found in {xmidi_dir}")
        return
    
    # Get already-processed files if resuming
    processed = set()
    if resume:
        processed = get_processed_files(output_dir)
        logging.info(f"Found {len(processed)} already-processed files (will skip)")
    
    # Filter out already-processed files
    files_to_process = []
    for file_path in files:
        filename = Path(file_path).stem
        if filename not in processed:
            files_to_process.append(file_path)
        else:
            logging.debug(f"Skipping already-processed file: {filename}")
    
    logging.info(f"Processing {len(files_to_process)} files (skipped {len(files) - len(files_to_process)})")
    
    # Create output directory
    ensure_dir(output_dir)
    
    # Process files in batches for GPU parallelization
    successful = 0
    failed = 0
    errors = []
    
    # Process in batches
    if batch_size > 1 and use_gpu:
        logging.info(f"Processing files in batches of {batch_size} for GPU parallelization")
        # Progress bar tracks samples (files), not batches
        pbar = tqdm(total=len(files_to_process), desc="Processing files")
        
        for batch_start in range(0, len(files_to_process), batch_size):
            batch_files = files_to_process[batch_start:batch_start + batch_size]
            
            # Process batch
            batch_results = process_batch(
                batch_files, output_dir, musetok_model, vocab
            )
            
            # Collect results
            for result in batch_results:
                if result[1]:  # success
                    successful += 1
                else:
                    failed += 1
                    errors.append((result[0], result[2]))  # (filename, error_message)
                    logging.warning(f"Failed to process {result[0]}: {result[2]}")
            
            # Update progress bar by number of files processed (batch_size)
            pbar.update(len(batch_results))
        
        pbar.close()
    else:
        # Process files sequentially (CPU or batch_size=1)
        logging.info("Processing files sequentially")
        for file_path in tqdm(files_to_process, desc="Processing files"):
            filename = Path(file_path).stem
            output_path = os.path.join(output_dir, f"{filename}.safetensors")
            
            result = process_single_file(
                file_path, output_path, musetok_model, vocab
            )
            
            if result[1]:  # success
                successful += 1
            else:
                failed += 1
                errors.append((result[0], result[2]))  # (filename, error_message)
                logging.warning(f"Failed to process {result[0]}: {result[2]}")
    
    # Summary
    logging.info(f"\nProcessing complete!")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    if errors:
        logging.warning(f"\nErrors encountered:")
        for filename, error in errors[:10]:  # Show first 10 errors
            logging.warning(f"  {filename}: {error}")
        if len(errors) > 10:
            logging.warning(f"  ... and {len(errors) - 10} more errors")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess XMIDI dataset: extract MuseTok latents")
    parser.add_argument("--xmidi_dir", required=True,
                       help="Directory containing XMIDI MIDI files")
    parser.add_argument("--output_dir", default=XMIDI_LATENTS_DIR,
                       help="Output directory for latents")
    parser.add_argument("--checkpoint_path", default=None,
                       help="Path to MuseTok checkpoint (defaults to MUSETOK_CHECKPOINT_DIR/best_tokenizer.pt)")
    parser.add_argument("--vocab_path", default=None,
                       help="Path to vocabulary file (defaults to musetok/data/dictionary.pkl)")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU (CUDA); if not provided, use CPU")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Resume preprocessing: skip files that have already been processed")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for GPU parallel processing (only used if > 1 and --gpu is set)")
    args = parser.parse_args()
    
    preprocess_xmidi(
        args.xmidi_dir, args.output_dir, args.checkpoint_path,
        args.vocab_path, args.gpu, args.resume, args.batch_size
    )
