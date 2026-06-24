"""
Evaluate a trained CausalVATransformer on a single MIDI file and plot the
predicted valence and arousal curves (bar number on x-axis).

Usage:
    python va_cont/evaluate_midi.py \
        --midi_path /path/to/song.mid \
        --model_path /path/to/best_model.pt \
        [--va_conditioning]          # use for Model B checkpoints
        [--output_path /path/to/out.png]
        [--show]                     # display interactively in addition to saving
"""

import argparse
import os
import sys
from os.path import dirname, realpath

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")  # non-interactive backend; overridden if --show is passed
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

sys.path.insert(0, dirname(realpath(__file__)))

from utils.musetok_utils import load_musetok_model, extract_latents_from_midi
from pretrain_model.model import CausalVATransformer
from utils.data_utils import MUSETOK_CHECKPOINT_DIR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot valence/arousal predictions for a single MIDI file."
    )
    # Input
    parser.add_argument("--midi_path", type=str, required=True,
                        help="Path to the MIDI file to evaluate")
    # Model
    parser.add_argument("--model_path", type=str, required=True,
                        help="Path to trained CausalVATransformer checkpoint (.pt)")
    parser.add_argument("--va_conditioning", action="store_true",
                        help="Model B checkpoint (uses sequential AR inference)")
    parser.add_argument("--target_mode", type=str, default="absolute",
                        choices=["absolute", "differential"],
                        help="'absolute': model outputs absolute V/A (default). "
                             "'differential': model outputs ΔV/ΔA; integrated via cumsum.")
    parser.add_argument("--latent_dim",  type=int,   default=128)
    parser.add_argument("--d_model",     type=int,   default=128)
    parser.add_argument("--n_heads",     type=int,   default=4)
    parser.add_argument("--n_layers",    type=int,   default=2)
    parser.add_argument("--d_ff",        type=int,   default=256)
    parser.add_argument("--max_len",     type=int,   default=512)
    # MuseTok
    parser.add_argument("--checkpoint_path", type=str, default=None,
                        help="MuseTok tokenizer checkpoint (default: auto-detect)")
    parser.add_argument("--vocab_path",      type=str, default=None)
    parser.add_argument("--velocity",        action="store_true")
    # Output
    parser.add_argument("--output_path", type=str, default=None,
                        help="Where to save the plot (default: <midi_stem>_va_curves.png "
                             "in the same directory as the MIDI file)")
    parser.add_argument("--show", action="store_true",
                        help="Display the plot interactively after saving")
    parser.add_argument("--gpu", action="store_true")

    args = parser.parse_args()

    if args.checkpoint_path is None:
        args.checkpoint_path = os.path.join(MUSETOK_CHECKPOINT_DIR, "best_tokenizer.pt")

    if args.output_path is None:
        stem = os.path.splitext(os.path.basename(args.midi_path))[0]
        args.output_path = os.path.join(
            os.path.dirname(os.path.abspath(args.midi_path)),
            f"{stem}_va_curves.png",
        )

    return args


def run_inference(
    midi_path: str,
    model: CausalVATransformer,
    musetok_model,
    vocab: dict,
    device: torch.device,
    target_mode: str = "absolute",
) -> np.ndarray:
    """
    Extract MuseTok latents and run the model.

    Args:
        target_mode: "absolute" (default) or "differential". When differential,
                     raw model output (ΔV/ΔA) is integrated via cumsum before return.

    Returns:
        predictions: (n_bars, 2) numpy array — columns are [valence, arousal],
                     always in absolute space.
    """
    latents, _ = extract_latents_from_midi(
        midi_path, musetok_model, vocab, has_velocity=True
    )
    if len(latents) == 0:
        raise ValueError(f"No latents extracted from {midi_path}. "
                         "The file may be empty, too short, or invalid.")

    lat_t = torch.from_numpy(latents.astype(np.float32)).unsqueeze(0).to(device)  # (1, T, 128)
    differential = (target_mode == "differential")

    with torch.no_grad():
        if model.va_conditioning:
            preds = model.infer_sequential(lat_t, differential=differential).cpu().numpy()  # (T, 2)
        else:
            preds = model(lat_t).squeeze(0).cpu().numpy()        # (T, 2)

    if differential:
        preds = np.cumsum(preds, axis=0)
        preds = np.clip(preds, -1.0, 1.0)

    return preds  # (n_bars, 2) — absolute


def plot_va_curves(
    predictions: np.ndarray,
    midi_path: str,
    model_path: str,
    va_conditioning: bool,
    output_path: str,
    show: bool = False,
):
    """
    Plot valence and arousal curves with bar number on the x-axis.

    Args:
        predictions:    (n_bars, 2) array — columns [valence, arousal]
        midi_path:      Source MIDI file path (for plot title)
        model_path:     Checkpoint path (for subtitle)
        va_conditioning: True if Model B
        output_path:    Where to save the figure
        show:           If True, also display interactively
    """
    n_bars = len(predictions)
    bars   = np.arange(n_bars)
    valence = predictions[:, 0]
    arousal = predictions[:, 1]

    midi_name  = os.path.basename(midi_path)
    model_name = os.path.basename(os.path.dirname(os.path.dirname(model_path)))
    variant    = "Model B (VA-conditioned)" if va_conditioning else "Model A (latents only)"

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    fig.suptitle(
        f"{midi_name}",
        fontsize=13, fontweight="bold", y=0.98,
    )
    fig.text(
        0.5, 0.925,
        f"{model_name}  ·  {variant}",
        ha="center", fontsize=9, color="#555555",
    )

    _ylim   = (-1.15, 1.15)
    _yticks = [-1.0, -0.5, 0.0, 0.5, 1.0]

    # — Valence —
    ax_v = axes[0]
    ax_v.plot(bars, valence, color="#2563EB", linewidth=1.5, label="valence")
    ax_v.fill_between(bars, valence, 0, where=(valence >= 0),
                      color="#2563EB", alpha=0.12)
    ax_v.fill_between(bars, valence, 0, where=(valence < 0),
                      color="#DC2626", alpha=0.12)
    ax_v.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax_v.set_ylabel("Valence", fontsize=11)
    ax_v.set_ylim(_ylim)
    ax_v.set_yticks(_yticks)
    ax_v.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax_v.grid(axis="y", linestyle=":", linewidth=0.6, color="#cccccc")
    ax_v.grid(axis="x", linestyle=":", linewidth=0.4, color="#eeeeee")
    ax_v.set_xlim(0, n_bars - 1)
    ax_v.tick_params(axis="x", which="both", bottom=False, labelbottom=False)

    # — Arousal —
    ax_a = axes[1]
    ax_a.plot(bars, arousal, color="#16A34A", linewidth=1.5, label="arousal")
    ax_a.fill_between(bars, arousal, 0, where=(arousal >= 0),
                      color="#16A34A", alpha=0.12)
    ax_a.fill_between(bars, arousal, 0, where=(arousal < 0),
                      color="#CA8A04", alpha=0.12)
    ax_a.axhline(0, color="#999999", linewidth=0.8, linestyle="--")
    ax_a.set_ylabel("Arousal", fontsize=11)
    ax_a.set_xlabel("Bar", fontsize=11)
    ax_a.set_ylim(_ylim)
    ax_a.set_yticks(_yticks)
    ax_a.yaxis.set_minor_locator(ticker.MultipleLocator(0.25))
    ax_a.grid(axis="y", linestyle=":", linewidth=0.6, color="#cccccc")
    ax_a.grid(axis="x", linestyle=":", linewidth=0.4, color="#eeeeee")
    ax_a.set_xlim(0, n_bars - 1)
    ax_a.xaxis.set_major_locator(ticker.MaxNLocator(integer=True, nbins=20))

    fig.tight_layout(rect=[0, 0, 1, 0.92])

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {output_path}")

    if show:
        matplotlib.use("TkAgg")  # switch to interactive backend
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    print(f"Loading model from {args.model_path} ...")
    model = CausalVATransformer(
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=0.0,  # no dropout at evaluation time
        max_len=args.max_len,
        va_conditioning=args.va_conditioning,
    ).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    print("Loading MuseTok model ...")
    musetok_model, vocab, _ = load_musetok_model(
        checkpoint_path=args.checkpoint_path,
        vocab_path=args.vocab_path,
        use_gpu=args.gpu,
        prefer_velocity=args.velocity,
    )

    print(f"Running inference on {args.midi_path} ...")
    predictions = run_inference(
        args.midi_path, model, musetok_model, vocab, device,
        target_mode=args.target_mode,
    )
    n_bars = len(predictions)
    print(f"  {n_bars} bars extracted")
    print(f"  Valence  — mean: {predictions[:, 0].mean():.3f}, "
          f"min: {predictions[:, 0].min():.3f}, max: {predictions[:, 0].max():.3f}")
    print(f"  Arousal  — mean: {predictions[:, 1].mean():.3f}, "
          f"min: {predictions[:, 1].min():.3f}, max: {predictions[:, 1].max():.3f}")

    plot_va_curves(
        predictions=predictions,
        midi_path=args.midi_path,
        model_path=args.model_path,
        va_conditioning=args.va_conditioning,
        output_path=args.output_path,
        show=args.show,
    )
