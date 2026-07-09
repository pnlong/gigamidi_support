"""
Generic MuseTok latent extraction for any VA dataset adapter.

Usage:
    python va_cont/preprocess/preprocess_musetok.py --dataset deam --gpu --resume
    python va_cont/preprocess/preprocess_musetok.py --dataset deam --gpu --batch_size 32

Multi-GPU (3 workers on one dataset):
    CUDA_VISIBLE_DEVICES=0 python ... --dataset deam --gpu --gpu_index 0 --num_gpus 3 --batch_size 64
    CUDA_VISIBLE_DEVICES=1 python ... --dataset deam --gpu --gpu_index 1 --num_gpus 3 --batch_size 32
    CUDA_VISIBLE_DEVICES=2 python ... --dataset deam --gpu --gpu_index 2 --num_gpus 3 --batch_size 32
"""

import argparse
import logging
import os
import sys
from multiprocessing import Pool, cpu_count

import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datasets import get_dataset
from utils.data_utils import ensure_dir, save_latents, MUSETOK_CHECKPOINT_DIR
from utils.musetok_utils import (
    extract_latents_batch_from_midis,
    extract_latents_from_midi,
    load_musetok_model,
)
from utils.midi_utils import load_midi_symusic
from va_utils import compute_bar_start_times, get_bar_resol


def parse_args():
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


def save_latent_result(file_path: str, output_path: str, song_id: str, latents, score) -> tuple:
    if latents is None or len(latents) == 0:
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
    parser.add_argument(
        "--batch_size", type=int, default=1,
        help="GPU: batch this many 16-bar segments per forward pass (default: 1). Try 16–64 on a 3090.",
    )
    parser.add_argument("--gpu_index", type=int, default=0,
                        help="This worker's GPU shard index (0..num_gpus-1)")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Total GPU workers sharding one dataset in parallel")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    if args.num_gpus <= 1 and args.gpu_index != 0:
        logging.warning(
            "--gpu_index %s ignored because --num_gpus=%s. "
            "Use --num_gpus 3 and one process per GPU.",
            args.gpu_index, args.num_gpus,
        )

    if args.gpu and args.num_gpus > 1 and torch.cuda.is_available():
        if torch.cuda.device_count() == 1:
            logging.info("Single visible CUDA device (use CUDA_VISIBLE_DEVICES to pin a physical GPU).")
        elif args.gpu_index >= torch.cuda.device_count():
            raise ValueError(
                f"--gpu_index {args.gpu_index} invalid: only {torch.cuda.device_count()} CUDA device(s) visible."
            )

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
    assigned = done = missing_midi = 0
    for idx, sid in enumerate(song_ids):
        if args.num_gpus > 1 and (idx % args.num_gpus) != args.gpu_index:
            continue
        assigned += 1
        latent_id = ds.latent_id(sid)
        if latent_id in processed:
            done += 1
            continue
        midi = ds.midi_path(sid)
        if not midi.is_file():
            missing_midi += 1
            logging.debug(f"Missing MIDI for {sid}: {midi}")
            continue
        out = ds.latents_path(sid)
        tasks.append((str(midi), str(out), latent_id))

    logging.info(
        f"Shard {args.gpu_index}/{args.num_gpus}: {len(tasks)} to process "
        f"({assigned} songs assigned, {done} already done, {missing_midi} missing MIDI, "
        f"{len(song_ids)} total)"
    )
    if args.gpu and args.batch_size > 1:
        logging.info(f"GPU segment batch size: {args.batch_size}")

    use_mp = not args.gpu and len(tasks) > 1
    num_workers = args.num_workers if args.num_workers is not None else max(1, cpu_count() // 4)
    if args.num_workers == 0:
        use_mp = False

    results = []
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
        if args.gpu and args.batch_size > 1:
            songs_per_step = max(1, args.batch_size // 2)
            for i in tqdm(range(0, len(tasks), songs_per_step), desc="MuseTok"):
                batch = tasks[i : i + songs_per_step]
                midi_paths = [t[0] for t in batch]
                batch_out = extract_latents_batch_from_midis(
                    midi_paths,
                    model,
                    vocab,
                    segment_batch_size=args.batch_size,
                    has_velocity=True,
                )
                for (midi, out, lid), (latents, score, err) in zip(batch, batch_out):
                    if err:
                        results.append((lid, False, err))
                        continue
                    try:
                        results.append(save_latent_result(midi, out, lid, latents, score))
                    except Exception as exc:
                        results.append((lid, False, str(exc)))
        else:
            for midi, out, lid in tqdm(tasks, desc="MuseTok"):
                results.append(process_single_file(midi, out, lid, model, vocab))

    ok = sum(1 for _, s, _ in results if s)
    fail = len(results) - ok
    logging.info(f"Done. Successful: {ok}, Failed: {fail}")
    for lid, success, err in results[:10]:
        if not success:
            logging.warning(f"  {lid}: {err}")
