"""
Preprocess EMOPIA dataset: extract MuseTok latents.

ASSUMPTIONS ABOUT EMOPIA STRUCTURE:
===================================
1. Directory Structure:
   - EMOPIA directory contains subdirectories: 'train', 'valid', 'test' (or 'val')
   - MIDI files are located in these subdirectories
   - Alternative: All MIDI files are in a flat structure within emopia_dir
   
2. File Format:
   - MIDI files have extensions: '.mid', '.midi', or '.MID'
   - Filenames are unique across splits (or we preserve split structure in output)
   
3. Output Structure:
   - Output directory mirrors input structure: output_dir/{split}/{filename}.safetensors
   - If input is flat, output is flat (or we can organize by split if metadata available)
   
4. Model Loading in Multiprocessing:
   - Each worker process loads its own copy of the model (to avoid CUDA/device conflicts)
   - Model checkpoint and vocab paths are passed to each worker
   
5. Resume Logic:
   - Checks for existing .safetensors files in output directory
   - Matches by filename (without extension) to determine if already processed
   - If resume=True, skips files that already have corresponding output files

TODO: Update these assumptions as you learn more about EMOPIA's actual structure.
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
    EMOPIA_LATENTS_DIR, ensure_dir, save_latents, 
    MUSETOK_CHECKPOINT_DIR
)
from utils.musetok_utils import load_musetok_model, extract_latents_from_midi
from utils.midi_utils import load_midi_symusic


def process_single_file_worker(args_tuple):
    """
    Process a single MIDI file (for multiprocessing).
    
    This function is called by each worker process. It loads the model
    independently to avoid CUDA/device conflicts in multiprocessing.
    
    Args:
        args_tuple: (midi_path, output_path, checkpoint_path, vocab_path, device)
    
    Returns:
        (filename, success, error_message)
    """
    midi_path, output_path, checkpoint_path, vocab_path, device = args_tuple
    
    try:
        # Load model in this worker process (each worker needs its own model instance)
        musetok_model, vocab = load_musetok_model(
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            device=device,
        )
        
        # Extract latents
        latents, bar_positions = extract_latents_from_midi(
            midi_path, musetok_model, vocab, device
        )
        
        if len(latents) == 0:
            return (Path(midi_path).stem, False, "No latents extracted (empty or invalid MIDI)")
        
        # Save latents
        ensure_dir(os.path.dirname(output_path))
        metadata = {
            "n_bars": len(latents),
            "original_midi_path": str(midi_path),
            "bar_positions": bar_positions
        }
        save_latents(output_path, latents, metadata)
        
        return (Path(midi_path).stem, True, None)
    except Exception as e:
        return (Path(midi_path).stem, False, str(e))


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


def find_midi_files(emopia_dir: str):
    """
    Find all MIDI files in emopia_dir, preserving directory structure.
    
    Returns:
        list of (midi_path, relative_path) tuples
        relative_path is used to preserve split structure in output
    """
    midi_files = []
    emopia_path = Path(emopia_dir)
    
    # Find all MIDI files recursively
    for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
        for midi_path in emopia_path.rglob(ext):
            # Get relative path from emopia_dir to preserve structure
            try:
                relative_path = midi_path.relative_to(emopia_path)
                midi_files.append((str(midi_path), str(relative_path)))
            except ValueError:
                # If relative path fails, just use filename
                midi_files.append((str(midi_path), midi_path.name))
    
    return midi_files


def preprocess_emopia(emopia_dir: str,
                     output_dir: str,
                     checkpoint_path: str,
                     vocab_path: str,
                     device: str = 'cuda',
                     num_workers: int = 4,
                     resume: bool = False):
    """
    Preprocess EMOPIA dataset.
    
    Args:
        emopia_dir: Directory containing EMOPIA MIDI files
        output_dir: Output directory for latents
        checkpoint_path: Path to MuseTok checkpoint
        vocab_path: Path to vocabulary file
        device: Device to use ('cuda' or 'cpu')
        num_workers: Number of parallel workers
        resume: If True, skip files that have already been processed
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. Find all MIDI files
    logging.info(f"Scanning for MIDI files in {emopia_dir}...")
    midi_files = find_midi_files(emopia_dir)
    logging.info(f"Found {len(midi_files)} MIDI files")
    
    if len(midi_files) == 0:
        logging.error("No MIDI files found! Check emopia_dir path and file extensions.")
        return
    
    # 2. Filter out already-processed files if resuming
    files_to_process = []
    if resume:
        processed = get_processed_files(output_dir)
        skipped_count = 0
        for midi_path, relative_path in midi_files:
            filename = Path(relative_path).stem
            if filename not in processed:
                files_to_process.append((midi_path, relative_path))
            else:
                skipped_count += 1
        logging.info(f"Resume mode: Skipping {skipped_count} already-processed files")
        logging.info(f"Processing {len(files_to_process)} remaining files")
    else:
        files_to_process = midi_files
    
    if len(files_to_process) == 0:
        logging.info("All files already processed. Nothing to do.")
        return
    
    # 3. Prepare arguments for multiprocessing
    # Each worker will load its own model instance
    process_args = []
    for midi_path, relative_path in files_to_process:
        # Preserve directory structure in output
        output_path = os.path.join(output_dir, str(Path(relative_path).with_suffix('.safetensors')))
        process_args.append((midi_path, output_path, checkpoint_path, vocab_path, device))
    
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
                       help="Path to MuseTok checkpoint (defaults to MUSETOK_CHECKPOINT_DIR/model.pt)")
    parser.add_argument("--vocab_path", type=str, default=None,
                       help="Path to MuseTok vocabulary (defaults to musetok/data/dictionary.pkl)")
    parser.add_argument("--device", type=str, default="cuda",
                       choices=["cuda", "cpu"],
                       help="Device to use")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of parallel workers")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Resume preprocessing: skip files that have already been processed")
    
    args = parser.parse_args()
    
    # Set default checkpoint path if not provided
    if args.checkpoint_path is None:
        checkpoint_path = os.path.join(MUSETOK_CHECKPOINT_DIR, "model.pt")
    else:
        checkpoint_path = args.checkpoint_path
    
    preprocess_emopia(
        args.emopia_dir,
        args.output_dir,
        checkpoint_path,
        args.vocab_path,
        args.device,
        args.num_workers,
        args.resume
    )
