"""
Preprocess EMOPIA dataset: extract MuseTok latents.

EMOPIA DATASET STRUCTURE:
==========================

There are two EMOPIA variants supported:

1. Edited EMOPIA (jingyue's version):
   - Location: /deepfreeze/user_shares/jingyue/EMOPIA_data
   - Format: REMI-encoded .pkl files (already converted to REMI events)
   - Naming: Q1_*.pkl, Q2_*.pkl, Q3_*.pkl, Q4_*.pkl (Q1-Q4 indicate emotions)
   - Structure: Flat directory (no train/valid/test splits)
   - Processing: Load REMI events directly from .pkl, skip MIDI conversion

2. EMOPIA+ (original, full dataset):
   - Location: /deepfreeze/pnlong/gigamidi/emopia/emopia_plus
   - Format: MIDI files (.mid/.midi) and/or REMI representations (.pkl)
   - Naming: Q1_*.mid, Q2_*.mid, etc. (same Q1-Q4 convention)
   - Structure: 
     * midis/ directory with MIDI files (4 tracks: Melody, Texture, Bass, Chord)
     * REMI/ directory with REMI representations (optional, use --use_remi_dir)
     * split/ directory with train/valid/test splits
   - Processing: 
     * If using REMI/: Load REMI events directly (skip MIDI conversion)
     * If using midis/: Full pipeline (MIDI → REMI → latents)
     * For MIDI: Merge Melody + Texture + Bass tracks (exclude Chord track)

FILE TYPE DETECTION:
- .pkl files → Load REMI events directly → Extract latents
- .mid/.midi files → Full pipeline (MIDI → REMI → latents)

OUTPUT STRUCTURE:
- Output directory preserves input structure
- For EMOPIA+: output_dir/{split}/{filename}.safetensors
- For edited EMOPIA: output_dir/{filename}.safetensors

MULTIPROCESSING:
- Each worker process loads its own copy of the model (to avoid CUDA/device conflicts)
- Model checkpoint and vocab paths are passed to each worker

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
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import (
    EMOPIA_LATENTS_DIR, ensure_dir, save_latents, load_pickle,
    MUSETOK_CHECKPOINT_DIR
)
from utils.musetok_utils import (
    load_musetok_model, extract_latents_from_midi, 
    extract_latents_from_events
)
from utils.midi_utils import load_midi_symusic, get_bar_positions


def is_remi_file(filepath: str) -> bool:
    """Check if file is a REMI-encoded pickle file."""
    return filepath.endswith('.pkl')

def is_midi_file(filepath: str) -> bool:
    """Check if file is a MIDI file."""
    return filepath.lower().endswith(('.mid', '.midi'))

def load_remi_from_pickle(pkl_path: str):
    """
    Load REMI events from pickle file.
    
    Returns:
        events: List of REMI event dictionaries
        bar_positions: List of bar boundary positions (extracted from events)
    """
    data = load_pickle(pkl_path)
    
    # Handle different pickle formats:
    # - Tuple format (jingyue's format): (bar_positions, events) where:
    #   * bar_positions: list of indices where 'Bar' events occur in events list
    #   * events: list of dicts with 'name' and 'value' keys (REMI events)
    # - Could be just events list
    # - Could be dict with 'events' key
    # - Could be dict with 'events' and 'bar_positions' keys
    if isinstance(data, tuple) and len(data) >= 2:
        # Jingyue's format: (bar_positions, events)
        # bar_positions is a list of indices into events where 'Bar' events occur
        # events is a list of event dicts with 'name' and 'value' keys
        bar_positions_indices = data[0]  # List of indices where bars occur
        events = data[1]  # List of event dicts
        
        if not isinstance(bar_positions_indices, list):
            raise ValueError(f"Unexpected pickle format in {pkl_path}: tuple[0] (bar_positions) is type {type(bar_positions_indices)}")
        if not isinstance(events, list):
            raise ValueError(f"Unexpected pickle format in {pkl_path}: tuple[1] (events) is type {type(events)}")
        if len(events) == 0:
            raise ValueError(f"Empty events list in {pkl_path}")
        
        # Convert bar_positions from indices to actual bar positions
        # bar_positions_indices are the indices in events where 'Bar' events occur
        # We need to verify these are correct and use them directly
        bar_positions = bar_positions_indices.copy()
        
        # Verify that bar_positions indices actually point to 'Bar' events
        for idx in bar_positions[:5]:  # Check first 5 as sample
            if idx < len(events) and events[idx].get('name') != 'Bar':
                logging.warning(f"Bar position index {idx} in {pkl_path} does not point to 'Bar' event (found: {events[idx].get('name')})")
        
    elif isinstance(data, list):
        # Just events list - extract bar positions from events
        events = data
        bar_positions = get_bar_positions(events)
    elif isinstance(data, dict):
        # Try various possible keys
        events = data.get('events') or data.get('remi') or data.get('event_seq') or data.get('remi_events')
        if events is None:
            # If no standard key found, check if dict values are lists
            list_values = [v for v in data.values() if isinstance(v, list) and len(v) > 0]
            if list_values:
                events = list_values[0]  # Use first list value
            else:
                raise ValueError(f"Unexpected pickle format in {pkl_path}: dict keys are {list(data.keys())}")
        if not isinstance(events, list):
            raise ValueError(f"Unexpected pickle format in {pkl_path}: events is type {type(events)}")
        bar_positions = data.get('bar_positions') or data.get('bar_pos') or data.get('pos')
        if bar_positions is None:
            bar_positions = get_bar_positions(events)
    else:
        raise ValueError(f"Unexpected pickle format in {pkl_path}: data is type {type(data)}")
    
    return events, bar_positions

def process_single_file(file_path: str,
                      output_path: str,
                      musetok_model,
                      vocab: dict,
                      filter_velocity: bool = False,
                      is_emopia_plus: bool = False):
    """
    Process a single file (MIDI or REMI pickle).
    
    Supports:
    - .pkl files: Load REMI events directly, extract latents
    - .mid/.midi files: Load MIDI, convert to REMI, extract latents
    
    Args:
        file_path: Path to input file
        output_path: Path to output .safetensors file
        musetok_model: Pre-loaded MuseTokEncoder instance
        vocab: Vocabulary dictionary
        filter_velocity: If True, filter out Note_Velocity events before processing
    
    Returns:
        (filename, success, error_message)
    """
    try:
        filename = Path(file_path).stem
        
        # Check file type and process accordingly
        if is_remi_file(file_path):
            # Load REMI events directly from pickle
            events, bar_positions = load_remi_from_pickle(file_path)
            
            # Extract latents from REMI events
            # If vocabulary doesn't support velocity, filter velocity events
            latents = extract_latents_from_events(
                events, bar_positions, musetok_model, vocab, filter_velocity=filter_velocity
            )
            
            file_type = "remi_pkl"
            
        elif is_midi_file(file_path):
            # Full pipeline: MIDI → REMI → latents
            # Note: For EMOPIA+ MIDI files, exclude Chord track (4th track)
            # This only applies to EMOPIA+, not to other datasets
            exclude_tracks = ['Chord'] if is_emopia_plus else None
            latents, bar_positions = extract_latents_from_midi(
                file_path, musetok_model, vocab, 
                has_velocity=not filter_velocity,
                exclude_tracks=exclude_tracks
            )
            
            file_type = "midi"
        else:
            return (filename, False, f"Unsupported file type: {file_path}")
        
        if len(latents) == 0:
            return (filename, False, "No latents extracted (empty or invalid file)")
        
        # Save latents
        ensure_dir(os.path.dirname(output_path))
        metadata = {
            "n_bars": len(latents),
            "original_file_path": str(file_path),
            "file_type": file_type,
            "bar_positions": bar_positions
        }
        save_latents(output_path, latents, metadata)
        
        return (filename, True, None)
    except Exception as e:
        return (Path(file_path).stem, False, str(e))


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


def find_emopia_files(emopia_dir: str, use_remi_dir: bool = False, split: str = None):
    """
    Find all EMOPIA files (both .pkl and .mid/.midi), preserving directory structure.
    
    Args:
        emopia_dir: Base EMOPIA directory
        use_remi_dir: If True and EMOPIA+ structure, look in REMI/ subdirectory
        split: If provided, process only this split (train/valid/test) for EMOPIA+.
               If None and EMOPIA+ structure detected, processes all splits automatically.
    
    Returns:
        list of (file_path, relative_path) tuples
        relative_path is used to preserve split structure in output
    """
    files = []
    emopia_path = Path(emopia_dir)
    
    # Check for EMOPIA+ structure
    remi_dir = emopia_path / "REMI"
    midis_dir = emopia_path / "midis"
    
    if use_remi_dir and remi_dir.exists():
        # EMOPIA+ structure: use REMI directory
        search_dir = remi_dir
    elif midis_dir.exists():
        # EMOPIA+ structure: use midis directory
        search_dir = midis_dir
    else:
        # Edited EMOPIA or flat structure: use base directory
        search_dir = emopia_path
    
    # If split is provided, look in split subdirectory only
    # If split is None, process all splits (for EMOPIA+ structure)
    if split:
        search_dir = search_dir / split
        # Find all .pkl and .mid/.midi files in this split
        for ext in ['*.pkl', '*.mid', '*.midi', '*.MID', '*.MIDI']:
            for file_path in search_dir.rglob(ext):
                if is_remi_file(str(file_path)) or is_midi_file(str(file_path)):
                    # Get relative path from emopia_dir to preserve structure
                    try:
                        relative_path = file_path.relative_to(emopia_path)
                        files.append((str(file_path), str(relative_path)))
                    except ValueError:
                        # If relative path fails, just use filename
                        files.append((str(file_path), file_path.name))
    else:
        # No split specified: process all splits (train/valid/test) if they exist
        # Check if EMOPIA+ split structure exists
        split_dirs = [search_dir / s for s in ['train', 'valid', 'test']]
        has_splits = any(d.exists() and d.is_dir() for d in split_dirs)
        
        if has_splits:
            # EMOPIA+ structure: process each split
            for split_name in ['train', 'valid', 'test']:
                split_dir = search_dir / split_name
                if split_dir.exists() and split_dir.is_dir():
                    for ext in ['*.pkl', '*.mid', '*.midi', '*.MID', '*.MIDI']:
                        for file_path in split_dir.rglob(ext):
                            if is_remi_file(str(file_path)) or is_midi_file(str(file_path)):
                                try:
                                    relative_path = file_path.relative_to(emopia_path)
                                    files.append((str(file_path), str(relative_path)))
                                except ValueError:
                                    files.append((str(file_path), file_path.name))
        else:
            # No split structure: process all files recursively from base directory
            for ext in ['*.pkl', '*.mid', '*.midi', '*.MID', '*.MIDI']:
                for file_path in search_dir.rglob(ext):
                    if is_remi_file(str(file_path)) or is_midi_file(str(file_path)):
                        try:
                            relative_path = file_path.relative_to(emopia_path)
                            files.append((str(file_path), str(relative_path)))
                        except ValueError:
                            files.append((str(file_path), file_path.name))
    
    return files


def preprocess_emopia(emopia_dir: str,
                     output_dir: str,
                     checkpoint_path: str,
                     vocab_path: str,
                     use_gpu: bool = False,
                     batch_size: int = 1,
                     resume: bool = False,
                     use_remi_dir: bool = False,
                     split: str = None,
                     prefer_velocity: bool = False):
    """
    Preprocess EMOPIA dataset (supports both edited EMOPIA and EMOPIA+).
    
    Args:
        emopia_dir: Directory containing EMOPIA files (or base directory for EMOPIA+)
        output_dir: Output directory for latents
        checkpoint_path: Path to MuseTok checkpoint
        vocab_path: Path to vocabulary file
        use_gpu: If True, use CUDA; otherwise use CPU
        batch_size: Number of files to process before saving (currently unused, processes sequentially)
        resume: If True, skip files that have already been processed
        use_remi_dir: If True and EMOPIA+ structure, use REMI/ subdirectory
        split: If provided, process only this split (train/valid/test) for EMOPIA+
        prefer_velocity: If True, prefer velocity vocabulary (will fallback with warning if not available)
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Log device info
    if use_gpu and torch.cuda.is_available():
        logging.info(f"Using device: cuda")
    else:
        logging.info(f"Using device: cpu")
    
    # 1. Find all files (.pkl and .mid/.midi)
    logging.info(f"Scanning for EMOPIA files in {emopia_dir}...")
    all_files = find_emopia_files(emopia_dir, use_remi_dir=use_remi_dir, split=split)
    
    # Count by file type
    pkl_count = sum(1 for f, _ in all_files if is_remi_file(f))
    midi_count = sum(1 for f, _ in all_files if is_midi_file(f))
    logging.info(f"Found {len(all_files)} files ({pkl_count} .pkl, {midi_count} .mid/.midi)")
    
    if len(all_files) == 0:
        logging.error("No EMOPIA files found! Check emopia_dir path and file extensions.")
        return
    
    # 2. Filter out already-processed files if resuming
    files_to_process = []
    if resume:
        processed = get_processed_files(output_dir)
        skipped_count = 0
        for file_path, relative_path in all_files:
            filename = Path(relative_path).stem
            if filename not in processed:
                files_to_process.append((file_path, relative_path))
            else:
                skipped_count += 1
        logging.info(f"Resume mode: Skipping {skipped_count} already-processed files")
        logging.info(f"Processing {len(files_to_process)} remaining files")
    else:
        files_to_process = all_files
    
    if len(files_to_process) == 0:
        logging.info("All files already processed. Nothing to do.")
        return
    
    # 3. Load model once (shared across all files)
    logging.info("Loading MuseTok model...")
    musetok_model, vocab, use_velocity = load_musetok_model(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        use_gpu=use_gpu,
        prefer_velocity=prefer_velocity,
    )
    logging.info(f"Model loaded successfully (velocity support: {use_velocity})")
    
    # Determine if we need to filter velocity events
    filter_velocity = not use_velocity
    if filter_velocity:
        logging.info("Vocabulary does not include velocity - velocity events will be filtered")
    
    # 4. Detect if we're processing EMOPIA+ (to exclude Chord tracks from MIDI)
    # EMOPIA+ is located at /deepfreeze/pnlong/gigamidi/emopia/emopia_plus
    is_emopia_plus = 'emopia_plus' in emopia_dir or 'emopia/emopia_plus' in emopia_dir
    if is_emopia_plus:
        logging.info("Detected EMOPIA+ dataset - Chord tracks will be excluded from MIDI files")
    
    # 5. Process files sequentially
    logging.info(f"Processing {len(files_to_process)} files sequentially...")
    successful = 0
    failed = 0
    errors = []
    
    for file_path, relative_path in tqdm(files_to_process, desc="Processing"):
        # Create output path, removing train/valid/test split directories
        rel_path = Path(relative_path)
        # Remove split directories (train/valid/test) from path
        parts = list(rel_path.parts)
        parts = [p for p in parts if p not in ['train', 'valid', 'test']]
        # Reconstruct path without splits
        if len(parts) > 1:
            # Keep subdirectory structure but remove splits
            output_rel_path = Path(*parts).with_suffix('.safetensors')
        else:
            # Just filename
            output_rel_path = Path(parts[0]).with_suffix('.safetensors')
        output_path = os.path.join(output_dir, str(output_rel_path))
        
        filename, success, error = process_single_file(
            file_path, output_path, musetok_model, vocab, 
            filter_velocity=filter_velocity, is_emopia_plus=is_emopia_plus
        )
        
        if success:
            successful += 1
        else:
            failed += 1
            errors.append((filename, error))
    
    # 5. Log statistics
    logging.info("\n" + "="*60)
    logging.info("Preprocessing Complete!")
    logging.info(f"Total files: {len(files_to_process)}")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    if errors:
        logging.warning(f"\nFirst 10 errors:")
        for filename, error in errors[:10]:
            logging.warning(f"  {filename}: {error}")
        if len(errors) > 10:
            logging.warning(f"  ... and {len(errors) - 10} more errors")
    logging.info("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Preprocess EMOPIA dataset: extract MuseTok latents from MIDI files"
    )
    parser.add_argument("--emopia_dir", required=True, 
                       help="Directory containing EMOPIA MIDI files")
    parser.add_argument("--output_dir", type=str, default=EMOPIA_LATENTS_DIR,
                       help="Output directory for latents")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to MuseTok checkpoint (defaults to MUSETOK_CHECKPOINT_DIR/best_tokenizer.pt)")
    parser.add_argument("--vocab_path", type=str, default=None,
                       help="Path to MuseTok vocabulary (defaults to musetok/data/dictionary.pkl)")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU (CUDA); if not provided, use CPU")
    parser.add_argument("--batch_size", type=int, default=1,
                       help="Batch size for processing (currently unused, processes sequentially)")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Resume preprocessing: skip files that have already been processed")
    parser.add_argument("--use_remi_dir", action="store_true", default=False,
                       help="For EMOPIA+: use REMI/ subdirectory instead of midis/")
    parser.add_argument("--split", type=str, default=None,
                       choices=["train", "valid", "test"],
                       help="For EMOPIA+: process only this split")
    parser.add_argument("--velocity", action="store_true", default=False,
                       help="Prefer velocity vocabulary for MuseTok (will fallback with warning if checkpoint doesn't support it)")
    
    args = parser.parse_args()
    
    # Set default checkpoint path if not provided
    if args.checkpoint_path is None:
        # Default to best_tokenizer checkpoint (used for encoding/extracting latents)
        from utils.data_utils import MUSETOK_TOKENIZER_CHECKPOINT
        checkpoint_path = MUSETOK_TOKENIZER_CHECKPOINT
    else:
        checkpoint_path = args.checkpoint_path
    
    preprocess_emopia(
        args.emopia_dir,
        args.output_dir,
        checkpoint_path,
        args.vocab_path,
        args.gpu,
        args.batch_size,
        args.resume,
        args.use_remi_dir,
        args.split,
        args.velocity
    )
