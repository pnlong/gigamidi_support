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
from multiprocessing import Pool
import sys
import glob

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
    # - Could be just events list
    # - Could be dict with 'events' key
    # - Could be dict with 'events' and 'bar_positions' keys
    if isinstance(data, list):
        events = data
        bar_positions = get_bar_positions(events)
    elif isinstance(data, dict):
        events = data.get('events', data.get('remi', data))
        if not isinstance(events, list):
            raise ValueError(f"Unexpected pickle format in {pkl_path}")
        bar_positions = data.get('bar_positions', get_bar_positions(events))
    else:
        raise ValueError(f"Unexpected pickle format in {pkl_path}")
    
    return events, bar_positions

def process_single_file_worker(args_tuple):
    """
    Process a single file (MIDI or REMI pickle) for multiprocessing.
    
    This function is called by each worker process. It loads the model
    independently to avoid CUDA/device conflicts in multiprocessing.
    
    Supports:
    - .pkl files: Load REMI events directly, extract latents
    - .mid/.midi files: Load MIDI, convert to REMI, extract latents
    
    Args:
        args_tuple: (file_path, output_path, checkpoint_path, vocab_path, device)
    
    Returns:
        (filename, success, error_message)
    """
    file_path, output_path, checkpoint_path, vocab_path, device = args_tuple
    
    try:
        # Load model in this worker process (each worker needs its own model instance)
        musetok_model, vocab = load_musetok_model(
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            device=device,
        )
        
        filename = Path(file_path).stem
        
        # Check file type and process accordingly
        if is_remi_file(file_path):
            # Load REMI events directly from pickle
            events, bar_positions = load_remi_from_pickle(file_path)
            
            # Extract latents from REMI events
            latents = extract_latents_from_events(
                events, bar_positions, musetok_model, vocab, device
            )
            
            file_type = "remi_pkl"
            
        elif is_midi_file(file_path):
            # Full pipeline: MIDI → REMI → latents
            # Note: For EMOPIA MIDI files, merge Melody + Texture + Bass tracks
            # (exclude Chord track) when loading MIDI
            latents, bar_positions = extract_latents_from_midi(
                file_path, musetok_model, vocab, device
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
        split: If provided, process only this split (train/valid/test) for EMOPIA+
    
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
    
    # If split is provided, look in split subdirectory
    if split:
        search_dir = search_dir / split
    
    # Find all .pkl and .mid/.midi files recursively
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
    
    return files


def preprocess_emopia(emopia_dir: str,
                     output_dir: str,
                     checkpoint_path: str,
                     vocab_path: str,
                     device: str = 'cuda',
                     num_workers: int = 4,
                     resume: bool = False,
                     use_remi_dir: bool = False,
                     split: str = None):
    """
    Preprocess EMOPIA dataset (supports both edited EMOPIA and EMOPIA+).
    
    Args:
        emopia_dir: Directory containing EMOPIA files (or base directory for EMOPIA+)
        output_dir: Output directory for latents
        checkpoint_path: Path to MuseTok checkpoint
        vocab_path: Path to vocabulary file
        device: Device to use ('cuda' or 'cpu')
        num_workers: Number of parallel workers
        resume: If True, skip files that have already been processed
        use_remi_dir: If True and EMOPIA+ structure, use REMI/ subdirectory
        split: If provided, process only this split (train/valid/test) for EMOPIA+
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
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
    
    # 3. Prepare arguments for multiprocessing
    # Each worker will load its own model instance
    process_args = []
    for file_path, relative_path in files_to_process:
        # Preserve directory structure in output
        output_path = os.path.join(output_dir, str(Path(relative_path).with_suffix('.safetensors')))
        process_args.append((file_path, output_path, checkpoint_path, vocab_path, device))
    
    # 4. Process files in parallel
    logging.info(f"Processing {len(files_to_process)} files with {num_workers} workers...")
    successful = 0
    failed = 0
    errors = []
    
    if num_workers == 1:
        # Single-threaded processing (useful for debugging)
        for args in tqdm(process_args, desc="Processing"):
            filename, success, error = process_single_file_worker(args)
            if success:
                successful += 1
            else:
                failed += 1
                errors.append((filename, error))
    else:
        # Multiprocessing
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(
                pool.imap(process_single_file_worker, process_args),
                total=len(process_args),
                desc="Processing"
            ))
        
        for filename, success, error in results:
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
    logging.info("="*60)


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
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Resume preprocessing: skip files that have already been processed")
    parser.add_argument("--use_remi_dir", action="store_true", default=False,
                       help="For EMOPIA+: use REMI/ subdirectory instead of midis/")
    parser.add_argument("--split", type=str, default=None,
                       choices=["train", "valid", "test"],
                       help="For EMOPIA+: process only this split")
    
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
        args.device,
        args.num_workers,
        args.resume,
        args.use_remi_dir,
        args.split
    )
