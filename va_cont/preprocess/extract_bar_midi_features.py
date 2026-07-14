"""
Extract per-bar MIDI features for VA training (no MuseTok).

Usage:
    python va_cont/preprocess/extract_bar_midi_features.py --dataset deam --feature_mode handcrafted --resume
    python va_cont/preprocess/extract_bar_midi_features.py --dataset deam --feature_mode remi --resume
"""

import argparse
import json
import logging
import os
import sys
from multiprocessing import Pool, cpu_count

import torch
from safetensors.torch import save_file
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datasets import get_dataset
from pretrain_model.midi_features import (
    DEFAULT_REMI_MAX_TOKENS,
    extract_handcrafted_from_midi,
    extract_remi_tokens_from_midi,
    features_dir_for_dataset,
    load_remi_vocab,
)
from utils.data_utils import ensure_dir, save_latents


def _process_one(args_tuple):
    midi_path, out_path, song_id, feature_mode, max_tokens, vocab = args_tuple
    try:
        if feature_mode == "handcrafted":
            feats, meta = extract_handcrafted_from_midi(midi_path)
            meta["song_id"] = song_id
            save_latents(out_path, feats, meta)
        elif feature_mode == "remi":
            tokens, mask, meta = extract_remi_tokens_from_midi(
                midi_path, vocab, max_tokens=max_tokens
            )
            meta["song_id"] = song_id
            ensure_dir(os.path.dirname(out_path))
            tensors = {
                "bar_tokens": torch.from_numpy(tokens),
                "token_padding_mask": torch.from_numpy(mask),
            }
            meta_str = {
                k: json.dumps(v) if isinstance(v, (list, dict)) else str(v)
                for k, v in meta.items()
            }
            save_file(tensors, out_path, metadata=meta_str)
        else:
            return song_id, False, f"Unknown feature_mode: {feature_mode}"
        return song_id, True, None
    except Exception as e:
        return song_id, False, str(e)


def parse_args():
    parser = argparse.ArgumentParser(description="Extract bar-level MIDI features for VA training.")
    parser.add_argument("--dataset", required=True, choices=["deam", "memo2496", "merp"])
    parser.add_argument(
        "--feature_mode", required=True, choices=["handcrafted", "remi"],
        help="handcrafted: 32-d stats per bar; remi: padded REMI token indices per bar",
    )
    parser.add_argument("--storage_dir", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=DEFAULT_REMI_MAX_TOKENS)
    parser.add_argument("--vocab_path", type=str, default=None,
                        help="MuseTok dictionary.pkl (default: musetok/data/dictionary.pkl)")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--num_workers", type=int, default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    ds = get_dataset(args.dataset, args.storage_dir)
    out_dir = features_dir_for_dataset(ds.storage_dir, ds.name, args.feature_mode)
    ensure_dir(str(out_dir))

    vocab = None
    if args.feature_mode == "remi":
        vocab = load_remi_vocab(prefer_velocity=True, vocab_path=args.vocab_path)
        logging.info(f"REMI vocab size: {len(vocab) + 1} (from {args.vocab_path or 'musetok/data'})")

    song_ids = ds.list_song_ids()
    processed = set()
    if args.resume and out_dir.is_dir():
        processed = {p.stem for p in out_dir.glob("*.safetensors")}
        logging.info(f"Resume: skipping {len(processed)} existing files")

    tasks = []
    for sid in song_ids:
        lid = ds.latent_id(sid)
        if lid in processed:
            continue
        midi = ds.midi_path(sid)
        if not midi.is_file():
            continue
        out = out_dir / f"{lid}.safetensors"
        tasks.append((str(midi), str(out), lid, args.feature_mode, args.max_tokens, vocab))

    logging.info(f"{ds.name} / {args.feature_mode}: {len(tasks)} songs to process")

    num_workers = args.num_workers if args.num_workers is not None else max(1, cpu_count() // 4)
    results = []
    if num_workers > 1 and len(tasks) > 1:
        with Pool(processes=num_workers) as pool:
            results = list(tqdm(pool.imap(_process_one, tasks), total=len(tasks)))
    else:
        for t in tqdm(tasks):
            results.append(_process_one(t))

    ok = sum(1 for _, s, _ in results if s)
    logging.info(f"Done. Successful: {ok}, Failed: {len(results) - ok}")
    for lid, success, err in results[:5]:
        if not success:
            logging.warning(f"  {lid}: {err}")
