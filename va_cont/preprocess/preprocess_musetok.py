"""
Generic MuseTok latent extraction for any VA dataset adapter.

Usage:
    python va_cont/preprocess/preprocess_musetok.py --dataset deam --gpu --resume
"""

import argparse
import logging
import os
import sys
from multiprocessing import Pool, cpu_count
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datasets import get_dataset
from utils.data_utils import ensure_dir, save_latents, MUSETOK_CHECKPOINT_DIR
from utils.musetok_utils import load_musetok_model, extract_latents_from_midi
from utils.midi_utils import load_midi_symusic
from va_utils import compute_bar_start_times, get_bar_resol


def process_single_file(file_path: str, output_path: str, song_id: str, musetok_model, vocab: dict):
    try:
        score = load_midi_symusic(file_path)
        latents, _ = extract_latents_from_midi(file_path, musetok_model, vocab, has_velocity=True)
        if len(latents) == 0:
            return (song_id, False, "No latents extracted")

        n_bars = len(latents)
        bar_resol = get_bar_resol(score)
        metadata = {
            "n_bars": n_bars,
            "song_id": song_id,
            "bar_start_times_seconds": compute_bar_start_times(score, n_bars),
            "bar_resol": bar_resol,
            "tpq": int(score.ticks_per_quarter),
            "original_file_path": str(file_path),
        }
        ensure_dir(os.path.dirname(output_path))
        save_latents(output_path, latents, metadata)
        return (song_id, True, None)
    except Exception as e:
        return (song_id, False, str(e))


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
    file_path, output_path, song_id = args_tuple
    return process_single_file(file_path, output_path, song_id, _worker_model, _worker_vocab)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract MuseTok latents for a VA dataset.")
    parser.add_argument("--dataset", type=str, required=True, choices=["deam", "memo2496", "merp"])
    parser.add_argument("--storage_dir", type=str, default=None)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--vocab_path", type=str, default=None)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ds = get_dataset(args.dataset, args.storage_dir)
    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(MUSETOK_CHECKPOINT_DIR, "best_tokenizer.pt")

    song_ids = ds.list_song_ids()
    logging.info(f"Dataset {ds.name}: {len(song_ids)} songs")

    latents_dir = ds.latents_dir()
    ensure_dir(str(latents_dir))

    processed = set()
    if args.resume and latents_dir.is_dir():
        processed = {p.stem for p in latents_dir.glob("*.safetensors")}
        logging.info(f"Resume: skipping {len(processed)} existing latents")

    tasks = []
    for sid in song_ids:
        latent_id = ds.latent_id(sid)
        if latent_id in processed:
            continue
        midi = ds.midi_path(sid)
        if not midi.is_file():
            logging.debug(f"Missing MIDI for {sid}: {midi}")
            continue
        out = ds.latents_path(sid)
        tasks.append((str(midi), str(out), latent_id))

    logging.info(f"Processing {len(tasks)} MIDI files → {latents_dir}")

    use_mp = not args.gpu and len(tasks) > 1
    num_workers = args.num_workers if args.num_workers is not None else max(1, cpu_count() // 4)
    if args.num_workers == 0:
        use_mp = False

    if use_mp:
        with Pool(
            processes=num_workers,
            initializer=_init_worker,
            initargs=(args.checkpoint_path, args.vocab_path, args.gpu),
        ) as pool:
            results = list(tqdm(pool.imap(_process_file_worker, tasks), total=len(tasks)))
    else:
        model, vocab, _ = load_musetok_model(
            checkpoint_path=args.checkpoint_path,
            vocab_path=args.vocab_path,
            use_gpu=args.gpu,
        )
        results = []
        for midi, out, lid in tqdm(tasks, desc="MuseTok"):
            results.append(process_single_file(midi, out, lid, model, vocab))

    ok = sum(1 for _, s, _ in results if s)
    fail = len(results) - ok
    logging.info(f"Done. Successful: {ok}, Failed: {fail}")
    for lid, success, err in results[:10]:
        if not success:
            logging.warning(f"  {lid}: {err}")
