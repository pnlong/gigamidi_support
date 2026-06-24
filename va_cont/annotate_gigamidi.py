"""
Annotate GigaMIDI with bar-level continuous valence/arousal predictions.

Loads a trained CausalVATransformer and runs inference on each GigaMIDI song.

  Model A (--va_conditioning not set):
      Single forward pass — all bars in one shot with causal mask.

  Model B (--va_conditioning):
      Sequential AR decoding — bar by bar, each bar's prediction fed back as
      prev_va for the next. O(T²) per song; add --max_bars to cap long songs.

Output CSV: md5, bar_idx, valence, arousal
"""

import argparse
import logging
import sys
import os
from os.path import dirname, realpath

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, dirname(realpath(__file__)))

from utils.musetok_utils import load_musetok_model, extract_latents_from_midi
from pretrain_model.model import CausalVATransformer
from utils.data_utils import ensure_dir, MUSETOK_CHECKPOINT_DIR


_STORAGE_DIR    = os.environ.get("XMIDI_STORAGE_DIR", "/deepfreeze/pnlong/gigamidi")
_DEAM_VA_DIR    = os.path.join(_STORAGE_DIR, "deam_va")
DEFAULT_OUT_DIR = os.path.join(_DEAM_VA_DIR, "gigamidi_annotations")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Annotate GigaMIDI with bar-level VA predictions."
    )
    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to best_model.pt checkpoint")
    parser.add_argument("--va_conditioning", action="store_true",
                        help="Model B: sequential AR decoding")
    parser.add_argument("--latent_dim",  type=int,   default=128)
    parser.add_argument("--d_model",     type=int,   default=128)
    parser.add_argument("--n_heads",     type=int,   default=4)
    parser.add_argument("--n_layers",    type=int,   default=2)
    parser.add_argument("--d_ff",        type=int,   default=256)
    parser.add_argument("--dropout",     type=float, default=0.0,
                        help="Dropout at inference — set to 0 (default)")
    parser.add_argument("--max_len",     type=int,   default=512)
    # MuseTok
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument("--vocab_path",      type=str, default=None)
    parser.add_argument("--velocity",        action="store_true")
    # Runtime
    parser.add_argument("--gpu",         action="store_true")
    parser.add_argument("--max_bars",    type=int, default=None,
                        help="Truncate songs to this many bars (useful for Model B speed)")
    parser.add_argument("--target_mode", type=str, default="absolute",
                        choices=["absolute", "differential"],
                        help="'absolute': model outputs absolute V/A (default). "
                             "'differential': model outputs ΔV/ΔA; integrated via cumsum.")
    # Dataset
    parser.add_argument("--split",       type=str, default="train")
    parser.add_argument("--streaming",   action="store_true", default=True)
    parser.add_argument("--max_samples", type=int, default=None)
    # Output
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--resume",      action="store_true")

    args = parser.parse_args()

    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(MUSETOK_CHECKPOINT_DIR, "best_tokenizer.pt")
    if args.output_path is None:
        ensure_dir(DEFAULT_OUT_DIR)
        suffix = "b" if args.va_conditioning else "a"
        args.output_path = os.path.join(DEFAULT_OUT_DIR, f"bar_va_annotations_model_{suffix}.csv")
    return args


def load_processed_md5s(csv_path: str) -> set:
    processed = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if "md5" in df.columns:
                processed = set(df["md5"].dropna())
            logging.info(f"Loaded {len(processed)} already-processed songs from {csv_path}")
        except Exception as e:
            logging.warning(f"Could not read existing CSV: {e}")
    return processed


def process_song_model_a(
    sample: dict,
    model: CausalVATransformer,
    musetok_model,
    vocab: dict,
    device: torch.device,
    max_bars: int = None,
    differential: bool = False,
) -> list:
    """Single forward pass inference (Model A)."""
    md5 = sample.get("md5", "")
    if not md5:
        return None
    try:
        latents, _ = extract_latents_from_midi(
            sample["music"], musetok_model, vocab, has_velocity=True
        )
        if len(latents) == 0:
            return None
        if max_bars:
            latents = latents[:max_bars]
        lat_t = torch.from_numpy(latents.astype(np.float32)).unsqueeze(0).to(device)  # (1, T, 128)
        with torch.no_grad():
            preds = model(lat_t).squeeze(0).cpu().numpy()  # (T, 2)
        if differential:
            preds = np.cumsum(preds, axis=0)
            preds = np.clip(preds, -1.0, 1.0)
        return [
            {"md5": md5, "bar_idx": i, "valence": round(float(v), 6), "arousal": round(float(a), 6)}
            for i, (v, a) in enumerate(preds)
        ]
    except Exception as e:
        logging.warning(f"Error processing {md5}: {e}")
        return None


def process_song_model_b(
    sample: dict,
    model: CausalVATransformer,
    musetok_model,
    vocab: dict,
    device: torch.device,
    max_bars: int = None,
    differential: bool = False,
) -> list:
    """Sequential AR decoding inference (Model B)."""
    md5 = sample.get("md5", "")
    if not md5:
        return None
    try:
        latents, _ = extract_latents_from_midi(
            sample["music"], musetok_model, vocab, has_velocity=True
        )
        if len(latents) == 0:
            return None
        if max_bars:
            latents = latents[:max_bars]
        lat_t = torch.from_numpy(latents.astype(np.float32)).unsqueeze(0).to(device)  # (1, T, 128)
        with torch.no_grad():
            preds = model.infer_sequential(lat_t, differential=differential).cpu().numpy()  # (T, 2)
        if differential:
            preds = np.cumsum(preds, axis=0)
            preds = np.clip(preds, -1.0, 1.0)
        return [
            {"md5": md5, "bar_idx": i, "valence": round(float(v), 6), "arousal": round(float(a), 6)}
            for i, (v, a) in enumerate(preds)
        ]
    except Exception as e:
        logging.warning(f"Error processing {md5}: {e}")
        return None


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Using device: {device}")
    logging.info(f"Model variant: {'B (sequential AR)' if args.va_conditioning else 'A (single pass)'}")

    logging.info(f"Loading model from {args.model_path}...")
    model = CausalVATransformer(
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        va_conditioning=args.va_conditioning,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    logging.info("Model loaded.")

    logging.info("Loading MuseTok model...")
    musetok_model, vocab, _ = load_musetok_model(
        checkpoint_path=args.checkpoint_path,
        vocab_path=args.vocab_path,
        use_gpu=args.gpu,
        prefer_velocity=args.velocity,
    )
    logging.info("MuseTok loaded.")

    ensure_dir(os.path.dirname(args.output_path))
    processed = set()
    if args.resume:
        processed = load_processed_md5s(args.output_path)
    else:
        pd.DataFrame(columns=["md5", "bar_idx", "valence", "arousal"]).to_csv(
            args.output_path, mode="w", index=False
        )

    logging.info(f"Loading GigaMIDI (split={args.split}, streaming={args.streaming})...")
    if args.streaming:
        dataset = load_dataset("Metacreation/GigaMIDI", split=args.split, streaming=True)
    else:
        dataset = load_dataset("Metacreation/GigaMIDI", split=args.split)

    process_fn = process_song_model_b if args.va_conditioning else process_song_model_a

    count = 0
    errors = 0
    skipped = 0
    iterator = tqdm(iter(dataset), total=args.max_samples, desc="Annotating")

    try:
        for sample in iterator:
            if args.max_samples and count >= args.max_samples:
                break
            md5 = sample.get("md5", "")
            if not md5:
                errors += 1
                continue
            if md5 in processed:
                skipped += 1
                continue
            rows = process_fn(
                sample, model, musetok_model, vocab, device, args.max_bars,
                differential=(args.target_mode == "differential"),
            )
            if rows:
                pd.DataFrame(rows).to_csv(args.output_path, mode="a", header=False, index=False)
                processed.add(md5)
                count += 1
                if count % 500 == 0:
                    logging.info(
                        f"Processed {count} songs (skipped={skipped}, errors={errors})"
                    )
            else:
                errors += 1
    except KeyboardInterrupt:
        logging.info("Interrupted. Progress saved.")

    logging.info(f"Done. Processed: {count}, Skipped: {skipped}, Errors: {errors}")
    logging.info(f"Output: {args.output_path}")
