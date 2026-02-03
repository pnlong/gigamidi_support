"""
Preprocess XMIDI with MuseTok: extract per-bar latents from MIDI files.

XMIDI DATASET STRUCTURE:
- Format: MIDI files (.midi)
- Naming: XMIDI_<Emotion>_<Genre>_<ID_len_8>.midi
- Structure: Flat directory (no train/valid/test splits - created later)

OUTPUT STRUCTURE:
- Output directory: output_dir/{filename}.safetensors
- Metadata includes: emotion, genre, ID extracted from filename
"""

import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
import sys
import re
import torch
import numpy as np
from typing import List
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

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
    name = Path(filename).stem
    
    pattern = r'^XMIDI_([^_]+)_([^_]+)_([a-zA-Z0-9]{8})$'
    match = re.match(pattern, name)
    
    if match:
        emotion = match.group(1)
        genre = match.group(2)
        id_str = match.group(3)
        return emotion, genre, id_str
    
    parts = name.split('_')
    if len(parts) >= 4 and parts[0] == 'XMIDI':
        emotion = parts[1]
        genre = parts[2]
        id_str = '_'.join(parts[3:])
        return emotion, genre, id_str
    
    return None, None, None


def is_midi_file(filepath: str) -> bool:
    """Check if file is a MIDI file."""
    return filepath.lower().endswith(('.mid', '.midi'))


_worker_model = None
_worker_vocab = None


def _init_worker(checkpoint_path, vocab_path, use_gpu):
    """Initialize worker process - load model once per worker."""
    global _worker_model, _worker_vocab
    _worker_model, _worker_vocab, _ = load_musetok_model(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        use_gpu=use_gpu
    )


def _process_file_worker(args_tuple):
    """Worker function for multiprocessing."""
    file_path, output_dir = args_tuple
    try:
        filename = Path(file_path).stem
        output_path = os.path.join(output_dir, f"{filename}.safetensors")
        return process_single_file(
            file_path, output_path, _worker_model, _worker_vocab
        )
    except Exception as e:
        return (Path(file_path).stem, False, str(e))


def process_single_file(file_path: str, output_path: str, musetok_model, vocab: dict):
    """Process a single MIDI file with MuseTok."""
    try:
        filename = Path(file_path).stem
        emotion, genre, id_str = extract_metadata_from_filename(filename)
        
        latents, bar_positions = extract_latents_from_midi(
            file_path, musetok_model, vocab,
            has_velocity=True
        )
        
        if len(latents) == 0:
            return (filename, False, "No latents extracted (empty or invalid file)")
        
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


def process_batch(file_paths: List[str], output_dir: str, musetok_model, vocab: dict):
    """Process a batch of MIDI files in parallel on GPU."""
    results = []
    batch_data = []
    
    for file_path in file_paths:
        try:
            filename = Path(file_path).stem
            emotion, genre, id_str = extract_metadata_from_filename(filename)
            score = load_midi_symusic(file_path)
            bar_positions, events = midi_to_events_symusic(
                score, has_velocity=True, time_first=False, repeat_beat=True
            )
            batch_data.append({
                'file_path': file_path, 'filename': filename,
                'events': events, 'bar_positions': bar_positions,
                'emotion': emotion if emotion else "unknown",
                'genre': genre if genre else "unknown",
                'id': id_str if id_str else "unknown"
            })
        except Exception as e:
            results.append((Path(file_path).stem, False, str(e)))
    
    if not batch_data:
        return results
    
    all_segments = []
    segment_file_map = []
    file_n_bars = {}
    
    for file_idx, data in enumerate(batch_data):
        try:
            music_data = musetok_model.get_segments(data['events'], data['bar_positions'].copy())
            n_segments = music_data['n_segment']
            n_bars = music_data['n_bar']
            file_n_bars[file_idx] = n_bars
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
    
    try:
        batch_enc_inp = np.stack([seg['enc_inp'] for seg in all_segments])
        batch_enc_padding_mask = np.stack([seg['enc_padding_mask'] for seg in all_segments])
        enc_inp_tensor = torch.from_numpy(batch_enc_inp).to(musetok_model.device).permute(2, 0, 1).long()
        enc_padding_mask_tensor = torch.from_numpy(batch_enc_padding_mask).to(musetok_model.device).bool()
        
        with torch.no_grad():
            latents_batch, _ = musetok_model.model.get_batch_latent(
                enc_inp_tensor, enc_padding_mask_tensor, latent_from_encoder=False
            )
        
        latents_batch = latents_batch.cpu().numpy()
        file_latents = {}
        
        for seg_idx, (file_idx, _, n_bars) in enumerate(segment_file_map):
            if file_idx not in file_latents:
                file_latents[file_idx] = []
            file_latents[file_idx].append(latents_batch[seg_idx])
        
        for file_idx, data in enumerate(batch_data):
            if file_idx not in file_latents:
                results.append((data['filename'], False, "No latents extracted"))
                continue
            all_file_latents = np.concatenate(file_latents[file_idx], axis=0)
            n_bars = file_n_bars[file_idx]
            latents = all_file_latents[:n_bars]
            if len(latents) == 0:
                results.append((data['filename'], False, "No latents extracted (empty or invalid file)"))
                continue
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
        logging.warning(f"Batch processing failed: {str(e)}. Falling back to individual processing.")
        for data in batch_data:
            if data['filename'] not in [r[0] for r in results]:
                output_path = os.path.join(output_dir, f"{data['filename']}.safetensors")
                result = process_single_file(
                    data['file_path'], output_path, musetok_model, vocab
                )
                results.append(result)
    
    return results


def get_processed_files(output_dir: str):
    """Get set of already-processed files."""
    processed = set()
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.safetensors'):
                    processed.add(Path(file).stem)
    return processed


def find_xmidi_files(xmidi_dir: str):
    """Find all XMIDI MIDI files."""
    files = []
    xmidi_path = Path(xmidi_dir)
    for ext in ['*.midi', '*.mid', '*.MIDI', '*.MID']:
        for file_path in xmidi_path.rglob(ext):
            if is_midi_file(str(file_path)):
                files.append(str(file_path))
    return sorted(files)


def preprocess_xmidi_musetok(
    xmidi_dir: str,
    output_dir: str,
    checkpoint_path: str = None,
    vocab_path: str = None,
    use_gpu: bool = False,
    resume: bool = False,
    batch_size: int = 1,
    num_workers: int = None,
):
    """Preprocess XMIDI dataset with MuseTok."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    
    if checkpoint_path is None:
        checkpoint_path = os.path.join(MUSETOK_CHECKPOINT_DIR, "best_tokenizer.pt")
    
    use_multiprocessing = not use_gpu and (num_workers is None or num_workers > 0)
    use_gpu_batching = batch_size > 1 and use_gpu
    
    if use_multiprocessing and num_workers is None:
        num_workers = max(1, cpu_count() // 4)
        logging.info(f"Auto-detected {num_workers} workers for CPU multiprocessing")
    elif num_workers == 0:
        use_multiprocessing = False
    
    if use_multiprocessing:
        logging.info(f"Using multiprocessing with {num_workers} workers (CPU mode)")
    elif use_gpu_batching:
        logging.info(f"Using GPU batching with batch_size={batch_size}")
    else:
        logging.info("Using sequential processing")
    
    musetok_model = None
    vocab = None
    if not use_multiprocessing:
        logging.info("Loading MuseTok model...")
        musetok_model, vocab, use_velocity = load_musetok_model(
            checkpoint_path=checkpoint_path,
            vocab_path=vocab_path,
            use_gpu=use_gpu
        )
        logging.info(f"MuseTok model loaded (vocabulary size: {len(vocab)}, velocity: {use_velocity})")
    
    logging.info(f"Searching for MIDI files in {xmidi_dir}...")
    files = find_xmidi_files(xmidi_dir)
    logging.info(f"Found {len(files)} MIDI files")
    
    if len(files) == 0:
        logging.warning(f"No MIDI files found in {xmidi_dir}")
        return
    
    processed = get_processed_files(output_dir) if resume else set()
    if resume:
        logging.info(f"Found {len(processed)} already-processed files (will skip)")
    
    files_to_process = [f for f in files if Path(f).stem not in processed]
    logging.info(f"Processing {len(files_to_process)} files (skipped {len(files) - len(files_to_process)})")
    
    ensure_dir(output_dir)
    successful = 0
    failed = 0
    errors = []
    
    if use_multiprocessing:
        worker_args = [(file_path, output_dir) for file_path in files_to_process]
        with Pool(processes=num_workers, initializer=_init_worker,
                  initargs=(checkpoint_path, vocab_path, use_gpu)) as pool:
            results = list(tqdm(
                pool.imap(_process_file_worker, worker_args),
                total=len(files_to_process),
                desc="Processing files"
            ))
        for result in results:
            if result[1]:
                successful += 1
            else:
                failed += 1
                errors.append((result[0], result[2]))
                logging.warning(f"Failed to process {result[0]}: {result[2]}")
    
    elif use_gpu_batching:
        pbar = tqdm(total=len(files_to_process), desc="Processing files")
        for batch_start in range(0, len(files_to_process), batch_size):
            batch_files = files_to_process[batch_start:batch_start + batch_size]
            batch_results = process_batch(batch_files, output_dir, musetok_model, vocab)
            for result in batch_results:
                if result[1]:
                    successful += 1
                else:
                    failed += 1
                    errors.append((result[0], result[2]))
                    logging.warning(f"Failed to process {result[0]}: {result[2]}")
            pbar.update(len(batch_results))
        pbar.close()
    else:
        logging.info("Processing files sequentially")
        for file_path in tqdm(files_to_process, desc="Processing files"):
            filename = Path(file_path).stem
            output_path = os.path.join(output_dir, f"{filename}.safetensors")
            result = process_single_file(file_path, output_path, musetok_model, vocab)
            if result[1]:
                successful += 1
            else:
                failed += 1
                errors.append((result[0], result[2]))
                logging.warning(f"Failed to process {result[0]}: {result[2]}")
    
    logging.info(f"\nProcessing complete!")
    logging.info(f"Successful: {successful}")
    logging.info(f"Failed: {failed}")
    if errors:
        logging.warning(f"\nErrors encountered:")
        for filename, error in errors[:10]:
            logging.warning(f"  {filename}: {error}")
        if len(errors) > 10:
            logging.warning(f"  ... and {len(errors) - 10} more errors")
