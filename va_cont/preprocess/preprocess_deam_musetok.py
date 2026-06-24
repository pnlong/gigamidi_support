"""
Preprocess DEAM dataset with MuseTok: extract per-bar latents from AMT-generated MIDI files
and save bar start times (in seconds) for temporal alignment with DEAM annotations.

Assumes AMT has already converted DEAM audio to MIDI with the same numeric song IDs:
  Input:  <midi_dir>/{song_id}.mid
  Output: <output_dir>/{song_id}.safetensors

Each .safetensors file contains:
  - latents: (n_bars, 128) float32
  - metadata: n_bars, bar_start_times_seconds (JSON list), song_id
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from utils.data_utils import ensure_dir, save_latents, MUSETOK_CHECKPOINT_DIR
from utils.musetok_utils import load_musetok_model, extract_latents_from_midi
from utils.midi_utils import load_midi_symusic, get_time_signature, BEAT_RESOL


# Default paths
DEAM_BASE_DIR = os.path.join(
    os.environ.get("XMIDI_STORAGE_DIR", "/deepfreeze/pnlong/gigamidi"),
    "deam"
)
DEAM_MIDI_DIR = os.path.join(DEAM_BASE_DIR, "DEAM_midi", "MEMD_midi")
DEAM_VA_DIR = os.path.join(
    os.environ.get("XMIDI_STORAGE_DIR", "/deepfreeze/pnlong/gigamidi"),
    "deam_va"
)
DEAM_LATENTS_DIR = os.path.join(DEAM_VA_DIR, "latents_musetok")


def ticks_to_seconds(score, tick: int) -> float:
    """Convert a tick position to seconds using the score's tempo track."""
    tpq = score.ticks_per_quarter
    tempos = sorted(score.tempos, key=lambda t: t.time) if score.tempos else []

    elapsed = 0.0
    current_tick = 0
    current_mspq = 500000  # default: 120 BPM

    for tempo in tempos:
        t_tick = int(tempo.time)
        if t_tick >= tick:
            break
        elapsed += (t_tick - current_tick) / tpq * (current_mspq / 1e6)
        current_tick = t_tick
        current_mspq = int(tempo.mspq)

    elapsed += (tick - current_tick) / tpq * (current_mspq / 1e6)
    return elapsed


def compute_bar_start_times(score, n_bars: int) -> list:
    """
    Compute the start time in seconds for each bar.

    Uses bar_resol = BEAT_RESOL * quarters_per_bar, so bar i starts at tick i * bar_resol.
    Handles tempo changes via ticks_to_seconds().
    """
    time_sig_num, time_sig_den = get_time_signature(score)
    quarters_per_bar = 4 * time_sig_num / time_sig_den
    bar_resol = int(BEAT_RESOL * quarters_per_bar)

    bar_start_times = []
    for i in range(n_bars):
        tick = i * bar_resol
        bar_start_times.append(ticks_to_seconds(score, tick))
    return bar_start_times


def process_single_file(file_path: str, output_path: str, musetok_model, vocab: dict):
    """Extract MuseTok latents and bar start times from a DEAM MIDI file."""
    try:
        song_id = Path(file_path).stem

        score = load_midi_symusic(file_path)

        latents, bar_positions = extract_latents_from_midi(
            file_path, musetok_model, vocab, has_velocity=True
        )

        if len(latents) == 0:
            return (song_id, False, "No latents extracted")

        n_bars = len(latents)
        bar_start_times = compute_bar_start_times(score, n_bars)

        ensure_dir(os.path.dirname(output_path))
        metadata = {
            "n_bars": n_bars,
            "song_id": song_id,
            "bar_start_times_seconds": bar_start_times,
            "original_file_path": str(file_path),
        }
        save_latents(output_path, latents, metadata)
        return (song_id, True, None)
    except Exception as e:
        return (Path(file_path).stem, False, str(e))


_worker_model = None
_worker_vocab = None


def _init_worker(checkpoint_path, vocab_path, use_gpu):
    global _worker_model, _worker_vocab
    _worker_model, _worker_vocab, _ = load_musetok_model(
        checkpoint_path=checkpoint_path,
        vocab_path=vocab_path,
        use_gpu=use_gpu,
    )


def _process_file_worker(args_tuple):
    file_path, output_dir = args_tuple
    song_id = Path(file_path).stem
    output_path = os.path.join(output_dir, f"{song_id}.safetensors")
    try:
        return process_single_file(file_path, output_path, _worker_model, _worker_vocab)
    except Exception as e:
        return (song_id, False, str(e))


def find_midi_files(midi_dir: str):
    files = []
    for ext in ("*.mid", "*.midi", "*.MID", "*.MIDI"):
        files.extend(Path(midi_dir).glob(ext))
    return sorted(str(p) for p in files)


def get_processed_files(output_dir: str):
    processed = set()
    if os.path.isdir(output_dir):
        for f in os.listdir(output_dir):
            if f.endswith(".safetensors"):
                processed.add(Path(f).stem)
    return processed


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess DEAM MIDI files with MuseTok.")
    parser.add_argument("--midi_dir", type=str, default=DEAM_MIDI_DIR)
    parser.add_argument("--output_dir", type=str, default=DEAM_LATENTS_DIR)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="CPU workers (0=sequential, None=auto)")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(MUSETOK_CHECKPOINT_DIR, "best_tokenizer.pt")

    logging.info(f"Searching for MIDI files in {args.midi_dir}...")
    files = find_midi_files(args.midi_dir)
    logging.info(f"Found {len(files)} MIDI files")

    if not files:
        logging.error(f"No MIDI files found in {args.midi_dir}. Exiting.")
        sys.exit(1)

    processed = get_processed_files(args.output_dir) if args.resume else set()
    if args.resume:
        logging.info(f"Found {len(processed)} already-processed files (will skip)")

    files_to_process = [f for f in files if Path(f).stem not in processed]
    logging.info(f"Processing {len(files_to_process)} files")

    ensure_dir(args.output_dir)

    use_multiprocessing = not args.gpu
    if args.num_workers is not None and args.num_workers == 0:
        use_multiprocessing = False
    num_workers = args.num_workers if args.num_workers is not None else max(1, cpu_count() // 4)

    successful = 0
    failed = 0
    errors = []

    if use_multiprocessing and len(files_to_process) > 1:
        logging.info(f"Using multiprocessing with {num_workers} workers")
        worker_args = [(fp, args.output_dir) for fp in files_to_process]
        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(args.checkpoint_path, args.vocab_path, args.gpu),
        ) as pool:
            results = list(tqdm(
                pool.imap(_process_file_worker, worker_args),
                total=len(files_to_process),
                desc="Processing",
            ))
    else:
        logging.info("Loading MuseTok model for sequential processing...")
        musetok_model, vocab, _ = load_musetok_model(
            checkpoint_path=args.checkpoint_path,
            vocab_path=args.vocab_path,
            use_gpu=args.gpu,
        )
        results = []
        for file_path in tqdm(files_to_process, desc="Processing"):
            song_id = Path(file_path).stem
            output_path = os.path.join(args.output_dir, f"{song_id}.safetensors")
            results.append(process_single_file(file_path, output_path, musetok_model, vocab))

    for song_id, ok, err in results:
        if ok:
            successful += 1
        else:
            failed += 1
            errors.append((song_id, err))
            logging.warning(f"Failed {song_id}: {err}")

    logging.info(f"Done. Successful: {successful}, Failed: {failed}")
    if errors[:10]:
        for song_id, err in errors[:10]:
            logging.warning(f"  {song_id}: {err}")
        if len(errors) > 10:
            logging.warning(f"  ... and {len(errors) - 10} more")
