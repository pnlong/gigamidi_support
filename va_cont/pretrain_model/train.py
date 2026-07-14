"""
Training script for DEAM bar-level continuous valence/arousal regression
using a GPT-style causal transformer.

Two model variants (ablation):
  Model A (--va_conditioning false): conditions on bar latents only.
  Model B (--va_conditioning true):  additionally conditions on the previous
      bar's V/A prediction (teacher-forced during training).

Target modes (--target_mode):
  absolute     — MSE on continuous V/A (default)
  differential — MSE on bar-to-bar deltas
  binned       — soft-bin KL loss on [-1,1]; decode via expected bin centers

WandB metrics:
  valid/mse, valid/mae: Lower is better (teacher-forced for Model B).
  Model B also logs valid/ar_mse, valid/ar_mae (autoregressive inference).
  valid/corr_valence, valid/corr_arousal: Pearson r; Model B uses AR only (not logged for train).
  Per-dataset corr: Model A from single-pass forward; Model B from AR (valid/ar/{dataset}/corr_*).
  Best checkpoint: valid/ar_mse for Model B, valid/mse for Model A.
"""

import argparse
import logging
import pprint
import sys
import os
from os.path import dirname, realpath
from multiprocessing import cpu_count
from typing import Optional

import wandb
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import numpy as np
import warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

sys.path.insert(0, dirname(dirname(realpath(__file__))))

from pretrain_model.dataset import (
    DEAMSequenceDataset,
    Memo2496SequenceDataset,
    MERPSequenceDataset,
    CombinedVASequenceDataset,
    build_dataset_for_source,
)
from pretrain_model.model import CausalVATransformer, VAModel
from pretrain_model.va_bins import (
    binned_va_kl_loss,
    default_bin_sigma,
    va_to_soft_bin_targets,
)
from pretrain_model.midi_features import (
    HANDCRAFTED_FEATURE_DIM,
    load_remi_vocab,
    remi_vocab_size,
)
from utils.data_utils import ensure_dir, infer_input_dim
from utils.config_utils import load_config, apply_config


_STORAGE_DIR = os.environ.get("XMIDI_STORAGE_DIR", "/deepfreeze/pnlong/gigamidi")

_DEAM_VA_DIR = os.path.join(_STORAGE_DIR, "deam_va")
DEFAULT_DEAM_LATENTS_DIR = os.path.join(_DEAM_VA_DIR, "latents_musetok")
DEFAULT_DEAM_LABELS_PATH = os.path.join(_DEAM_VA_DIR, "labels", "deam_va_labels.json")
DEFAULT_DEAM_TRAIN_SONGS = os.path.join(_DEAM_VA_DIR, "labels", "train_songs.txt")
DEFAULT_DEAM_VAL_SONGS   = os.path.join(_DEAM_VA_DIR, "labels", "val_songs.txt")

_MEMO2496_VA_DIR = os.path.join(_STORAGE_DIR, "memo2496_va")
DEFAULT_MEMO2496_LATENTS_DIR = os.path.join(_MEMO2496_VA_DIR, "latents_musetok")
DEFAULT_MEMO2496_LABELS_PATH = os.path.join(_MEMO2496_VA_DIR, "labels", "memo2496_va_labels.json")
DEFAULT_MEMO2496_TRAIN_SONGS = os.path.join(_MEMO2496_VA_DIR, "labels", "train_songs.txt")
DEFAULT_MEMO2496_VAL_SONGS   = os.path.join(_MEMO2496_VA_DIR, "labels", "val_songs.txt")

_MERP_VA_DIR = os.path.join(_STORAGE_DIR, "merp_va")
DEFAULT_MERP_LATENTS_DIR = os.path.join(_MERP_VA_DIR, "latents_musetok")
DEFAULT_MERP_LABELS_PATH = os.path.join(_MERP_VA_DIR, "labels", "merp_va_labels.json")
DEFAULT_MERP_TRAIN_SONGS = os.path.join(_MERP_VA_DIR, "labels", "train_songs.txt")
DEFAULT_MERP_VAL_SONGS   = os.path.join(_MERP_VA_DIR, "labels", "val_songs.txt")

DEFAULT_DATASETS = ["deam", "memo2496", "merp"]

DEFAULT_OUTPUT_DIR = os.path.join(_DEAM_VA_DIR, "checkpoints", "trained_models")

# Keep legacy aliases for backward compatibility with old configs
DEFAULT_LATENTS_DIR = DEFAULT_DEAM_LATENTS_DIR
DEFAULT_LABELS_PATH = DEFAULT_DEAM_LABELS_PATH
DEFAULT_TRAIN_SONGS = DEFAULT_DEAM_TRAIN_SONGS
DEFAULT_VAL_SONGS   = DEFAULT_DEAM_VAL_SONGS


def _compute_differential_loss(
    pred: torch.Tensor,
    va_targets: torch.Tensor,
    label_mask: torch.Tensor,
    loss_fn: nn.Module,
) -> torch.Tensor:
    """
    MSE of bar-to-bar deltas between consecutive labeled positions, computed per song.

    For each song in the batch, finds all pairs of consecutive labeled bars (t_i, t_{i+1})
    and computes MSE( pred[t_{i+1}] - pred[t_i],  target[t_{i+1}] - target[t_i] ).
    """
    diff_p, diff_t = [], []
    for b in range(pred.shape[0]):
        idx = label_mask[b].nonzero(as_tuple=True)[0]   # sorted labeled indices
        if len(idx) < 2:
            continue
        t_prev = idx[:-1]
        t_next = idx[1:]
        diff_p.append(pred[b, t_next] - pred[b, t_prev])
        diff_t.append(va_targets[b, t_next] - va_targets[b, t_prev])
    if not diff_p:
        return pred.sum() * 0.0   # zero with grad
    return loss_fn(torch.cat(diff_p, dim=0), torch.cat(diff_t, dim=0))


def augment_va_training(
    va_targets: torch.Tensor,
    label_mask: torch.BoolTensor,
    jitter_std: float = 0.0,
    label_dropout: float = 0.0,
) -> tuple[torch.Tensor, torch.BoolTensor]:
    """
    Training-only label regularization: Gaussian jitter + random bar dropout.

    Jitter is applied on labeled bars and clamped to [-1, 1]. Label dropout
    randomly removes bars from the loss (Model B falls back to null_va for
    dropped predecessors). Each song keeps at least one labeled bar.
    """
    va_aug = va_targets
    mask_aug = label_mask

    if jitter_std > 0.0:
        va_aug = va_targets.clone()
        noise = torch.randn_like(va_aug) * jitter_std
        labeled = mask_aug.unsqueeze(-1).expand_as(va_aug)
        va_aug = torch.where(labeled, (va_aug + noise).clamp(-1.0, 1.0), va_aug)

    if label_dropout > 0.0:
        mask_aug = label_mask.clone()
        drop = (torch.rand_like(mask_aug.float()) < label_dropout) & mask_aug
        mask_aug = mask_aug & ~drop
        for b in range(mask_aug.shape[0]):
            if label_mask[b].any() and not mask_aug[b].any():
                idx = label_mask[b].nonzero(as_tuple=True)[0]
                mask_aug[b, idx[torch.randint(len(idx), (1,), device=idx.device)]] = True

    return va_aug, mask_aug


def _model_forward_kwargs(batch: dict, device: torch.device) -> dict:
    """Build optional REMI kwargs for VAModel.forward."""
    kwargs = {}
    if "bar_tokens" in batch:
        kwargs["bar_tokens"] = batch["bar_tokens"].to(device=device, dtype=torch.long)
        kwargs["token_padding_mask"] = batch["token_padding_mask"].to(device=device, dtype=torch.bool)
    return kwargs


def evaluate_batch(
    model: nn.Module,
    batch: dict,
    loss_fn: nn.Module,
    device: torch.device,
    update_parameters: bool = False,
    optimizer: torch.optim.Optimizer = None,
    return_predictions: bool = False,
    target_mode: str = "absolute",
    differential_weight: float = 0.0,
    va_jitter_std: float = 0.0,
    va_label_dropout: float = 0.0,
    max_grad_norm: float = 0.0,
    n_bins: int = 20,
    bin_sigma: float = None,
):
    """
    Forward pass, optional backward pass, and metrics.

    Loss and metrics are computed only over annotated positions (label_mask=True).
    Padding positions are excluded automatically via label_mask (which is False there).

    Args:
        target_mode:          "absolute" — MSE on absolute VA values (default);
                              "differential" — MSE on bar-to-bar deltas between
                                              consecutive labeled positions;
                              "binned" — soft-bin KL loss; metrics use expected VA.
        differential_weight:  Only used when target_mode="absolute". When > 0, adds a
                              differential MSE term:
                              loss = (1 - w)*MSE_abs + w*MSE_diff.
        va_jitter_std:        Gaussian noise std on labeled V/A during training (0=off).
        va_label_dropout:     Probability of dropping each labeled bar from the loss
                              during training (0=off).
        n_bins:               Number of equal bins on [-1, 1] for target_mode=binned.
        bin_sigma:            Gaussian soft-label width in VA units (default: one bin).
    """
    latents      = batch["latents"].to(device)       # (B, T, 128)
    va_targets   = batch["va_targets"].to(device)    # (B, T, 2)
    label_mask   = batch["label_mask"].to(device)    # (B, T) bool
    padding_mask = batch["padding_mask"].to(device)  # (B, T) bool

    if update_parameters and (va_jitter_std > 0.0 or va_label_dropout > 0.0):
        va_targets, label_mask = augment_va_training(
            va_targets, label_mask, va_jitter_std, va_label_dropout,
        )

    if update_parameters:
        optimizer.zero_grad()

    fwd_kw = _model_forward_kwargs(batch, device)
    raw = model(
        latents,
        padding_mask=padding_mask,
        va_targets=va_targets,
        label_mask=label_mask,
        **fwd_kw,
    )
    binned = target_mode == "binned"
    pred = model.decode_va(raw) if binned else raw

    # Restrict to annotated bars only
    n_labeled = int(label_mask.sum().item())
    if n_labeled == 0:
        zero_metrics = {"mse": 0.0, "mae": 0.0, "corr_valence": 0.0, "corr_arousal": 0.0,
                        "n_labeled": 0}
        if return_predictions:
            empty = np.zeros((0, 2), dtype=np.float32)
            return 0.0, zero_metrics, (empty, empty)
        return 0.0, zero_metrics

    pred_labeled   = pred[label_mask]        # (N, 2)
    target_labeled = va_targets[label_mask]  # (N, 2)
    raw_labeled    = raw[label_mask] if binned else None

    if not torch.isfinite(pred_labeled).all():
        logging.warning("Non-finite predictions detected in batch; skipping gradient update.")
        if update_parameters:
            optimizer.zero_grad(set_to_none=True)
        nan_metrics = {"mse": float("nan"), "mae": float("nan"),
                       "corr_valence": 0.0, "corr_arousal": 0.0, "n_labeled": n_labeled}
        if return_predictions:
            p = np.nan_to_num(pred_labeled.detach().cpu().numpy(), nan=0.0)
            t = target_labeled.detach().cpu().numpy()
            return float("nan"), nan_metrics, (p, t)
        return float("nan"), nan_metrics

    if target_mode == "binned":
        sigma = bin_sigma if bin_sigma is not None else default_bin_sigma(n_bins)
        centers = model.transformer.bin_centers
        soft_v, soft_a = va_to_soft_bin_targets(target_labeled, centers, sigma)
        loss = binned_va_kl_loss(raw_labeled, soft_v, soft_a, n_bins)
    elif target_mode == "differential":
        loss = _compute_differential_loss(pred, va_targets, label_mask, loss_fn)
    elif differential_weight > 0.0:
        loss_abs  = loss_fn(pred_labeled, target_labeled)
        loss_diff = _compute_differential_loss(pred, va_targets, label_mask, loss_fn)
        loss = (1.0 - differential_weight) * loss_abs + differential_weight * loss_diff
    else:
        loss = loss_fn(pred_labeled, target_labeled)

    if update_parameters:
        if not torch.isfinite(loss):
            logging.warning(f"Non-finite loss ({float(loss.detach())}); skipping optimizer step.")
            optimizer.zero_grad(set_to_none=True)
        else:
            loss.backward()
            if max_grad_norm and max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

    loss_value = float(loss.detach())
    with torch.no_grad():
        mse = ((pred_labeled - target_labeled) ** 2).mean().item()
        mae = (pred_labeled - target_labeled).abs().mean().item()
        p = pred_labeled.cpu().numpy()
        t = target_labeled.cpu().numpy()
        n = p.shape[0]
        corr_v = float(np.corrcoef(p[:, 0], t[:, 0])[0, 1]) if n > 1 else 0.0
        corr_a = float(np.corrcoef(p[:, 1], t[:, 1])[0, 1]) if n > 1 else 0.0
        corr_v = 0.0 if np.isnan(corr_v) else corr_v
        corr_a = 0.0 if np.isnan(corr_a) else corr_a

    metrics = {"mse": mse, "mae": mae, "corr_valence": corr_v, "corr_arousal": corr_a,
               "n_labeled": int(label_mask.sum().item())}
    if return_predictions:
        return loss_value, metrics, (p, t)
    return loss_value, metrics


def _pearson_corr(pred: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    """Pearson r for valence (col 0) and arousal (col 1)."""
    if pred.shape[0] < 2:
        return 0.0, 0.0
    rv = float(np.corrcoef(pred[:, 0], target[:, 0])[0, 1])
    ra = float(np.corrcoef(pred[:, 1], target[:, 1])[0, 1])
    return (0.0 if np.isnan(rv) else rv, 0.0 if np.isnan(ra) else ra)


def _checkpoint_value(
    valid_metrics: dict,
    ar_metrics: Optional[dict],
    va_conditioning: bool,
    checkpoint_metric: str,
) -> tuple[float, bool]:
    """
    Return (value, lower_is_better) for best-checkpoint comparison.
    """
    use_ar = va_conditioning and ar_metrics
    m = ar_metrics if use_ar else valid_metrics
    if checkpoint_metric == "mse":
        return float(m["mse"]), True
    if checkpoint_metric == "corr_valence":
        return float(m["corr_valence"]), False
    return float(m["corr_valence"] + m["corr_arousal"]), False


def _per_dataset_corr(
    preds: list[np.ndarray],
    targets: list[np.ndarray],
    sources: list[str],
) -> dict[str, dict[str, float]]:
    """Pooled Pearson r per dataset source."""
    by_src: dict[str, tuple[list, list]] = {}
    for p, t, src in zip(preds, targets, sources):
        by_src.setdefault(src, ([], []))
        by_src[src][0].append(p)
        by_src[src][1].append(t)
    out = {}
    for src, (pl, tl) in by_src.items():
        if not pl:
            continue
        p_all = np.concatenate(pl, axis=0)
        t_all = np.concatenate(tl, axis=0)
        rv, ra = _pearson_corr(p_all, t_all)
        out[src] = {"corr_valence": rv, "corr_arousal": ra, "n_bars": int(p_all.shape[0])}
    return out


def evaluate_autoregressive(
    model: VAModel,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    return_per_song: bool = False,
) -> dict:
    """
    Autoregressive validation for Model B (no teacher forcing).

    Returns dict with mse, mae, corr_valence, corr_arousal over all labeled bars.
    If return_per_song=True, also includes per_song_preds/targets/sources lists
    for per-dataset breakdown (same inference mode as the pooled corr metrics).
    """
    model.eval()
    all_preds: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    per_song_preds: list[np.ndarray] = []
    per_song_targets: list[np.ndarray] = []
    per_song_sources: list[str] = []

    with torch.no_grad():
        for batch in loader:
            latents = batch["latents"].to(device)
            targets = batch["va_targets"].to(device)
            mask = batch["label_mask"].to(device)
            padding_mask = batch["padding_mask"].to(device)
            sources = batch.get("dataset_source", ["unknown"] * latents.shape[0])
            for b in range(latents.shape[0]):
                seq_len = int((~padding_mask[b]).sum().item())
                if seq_len == 0:
                    continue
                if "bar_tokens" in batch:
                    bt = batch["bar_tokens"][b : b + 1, :seq_len].to(device)
                    tm = batch["token_padding_mask"][b : b + 1, :seq_len].to(device)
                    pred = model.infer_sequential(bar_tokens=bt, token_padding_mask=tm)
                else:
                    pred = model.infer_sequential(latents[b : b + 1, :seq_len])
                bar_mask = mask[b, :seq_len]
                if not bar_mask.any():
                    continue
                p = pred[bar_mask].cpu().numpy()
                t = targets[b, :seq_len][bar_mask].cpu().numpy()
                all_preds.append(p)
                all_targets.append(t)
                if return_per_song:
                    per_song_preds.append(p)
                    per_song_targets.append(t)
                    per_song_sources.append(sources[b] if b < len(sources) else "unknown")

    if not all_preds:
        out = {"mse": 0.0, "mae": 0.0, "corr_valence": 0.0, "corr_arousal": 0.0}
        if return_per_song:
            out["per_song_preds"] = []
            out["per_song_targets"] = []
            out["per_song_sources"] = []
        return out

    p_all = np.concatenate(all_preds, axis=0)
    t_all = np.concatenate(all_targets, axis=0)
    mse = float(((p_all - t_all) ** 2).mean())
    mae = float(np.abs(p_all - t_all).mean())
    corr_v, corr_a = _pearson_corr(p_all, t_all)
    out = {
        "mse": mse,
        "mae": mae,
        "corr_valence": corr_v,
        "corr_arousal": corr_a,
    }
    if return_per_song:
        out["per_song_preds"] = per_song_preds
        out["per_song_targets"] = per_song_targets
        out["per_song_sources"] = per_song_sources
    return out


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train causal transformer for bar-level VA regression (DEAM + Memo2496)."
    )
    # DEAM data
    parser.add_argument("--latents_dir",  type=str, default=DEFAULT_DEAM_LATENTS_DIR,
                        help="DEAM MuseTok latents directory")
    parser.add_argument("--labels_path",  type=str, default=DEFAULT_DEAM_LABELS_PATH,
                        help="DEAM VA labels JSON")
    parser.add_argument("--train_songs",  type=str, default=DEFAULT_DEAM_TRAIN_SONGS,
                        help="DEAM train song list")
    parser.add_argument("--valid_songs",  type=str, default=DEFAULT_DEAM_VAL_SONGS,
                        help="DEAM validation song list")
    # Memo2496 data (combined with DEAM by default; set to empty string to disable)
    parser.add_argument("--memo2496_latents_dir",  type=str, default=DEFAULT_MEMO2496_LATENTS_DIR,
                        help="Memo2496 MuseTok latents directory ('' to disable)")
    parser.add_argument("--memo2496_labels_path",  type=str, default=DEFAULT_MEMO2496_LABELS_PATH,
                        help="Memo2496 VA labels JSON ('' to disable)")
    parser.add_argument("--memo2496_train_songs",  type=str, default=DEFAULT_MEMO2496_TRAIN_SONGS,
                        help="Memo2496 train song list ('' to disable)")
    parser.add_argument("--memo2496_valid_songs",  type=str, default=DEFAULT_MEMO2496_VAL_SONGS,
                        help="Memo2496 validation song list ('' to disable)")
    # Combined multi-dataset mode (preferred)
    parser.add_argument("--datasets", nargs="+", default=None,
                        help="Dataset names to combine, e.g. deam memo2496 merp")
    parser.add_argument("--storage_dir", type=str, default=None,
                        help="XMIDI_STORAGE_DIR override for dataset adapters")
    # Model architecture
    parser.add_argument("--va_conditioning", action="store_true",
                        help="Model B: concat prev_va to each bar input")
    parser.add_argument("--latent_dim",  type=int,   default=None,
                        help="MuseTok latent dim (default: inferred from latents_dir)")
    parser.add_argument("--d_model",     type=int,   default=128)
    parser.add_argument("--n_heads",     type=int,   default=4)
    parser.add_argument("--n_layers",    type=int,   default=2)
    parser.add_argument("--d_ff",        type=int,   default=256)
    parser.add_argument("--dropout",     type=float, default=0.1)
    parser.add_argument("--max_len",     type=int,   default=512)
    # Training
    parser.add_argument("--batch_size",              type=int,   default=32,
                        help="Songs per batch")
    parser.add_argument("--learning_rate",           type=float, default=1e-4)
    parser.add_argument("--weight_decay",            type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Train for this many full passes over the training set. "
                             "Overrides --steps when set (computed after DataLoader is built).")
    parser.add_argument("--valid_every_epochs", type=float, default=1.0,
                        help="Run validation every N epochs (only when --epochs is set). "
                             "Default 1.0 = validate once per epoch.")
    parser.add_argument("--steps",                   type=int,   default=20000,
                        help="Total optimizer steps (gradient updates). Ignored if --epochs is set.")
    parser.add_argument("--valid_steps",             type=int,   default=1000,
                        help="Validate every N optimizer steps (ignored if --epochs is set).")
    parser.add_argument("--early_stopping",          action="store_true")
    parser.add_argument("--early_stopping_tolerance",type=int,   default=10)
    parser.add_argument("--gpu",                     action="store_true")
    parser.add_argument("--num_workers",             type=int,
                        default=max(1, int(cpu_count() / 4)))
    # Loss objective
    parser.add_argument("--target_mode", type=str, default="absolute",
                        choices=["absolute", "differential", "binned"],
                        help="'absolute': MSE on absolute VA values (default). "
                             "'differential': MSE on bar-to-bar deltas between "
                             "consecutive labeled positions. "
                             "'binned': soft-bin KL loss on [-1,1].")
    parser.add_argument("--n_bins", type=int, default=20,
                        help="Equal bins on [-1,1] when target_mode=binned.")
    parser.add_argument("--bin_sigma", type=float, default=None,
                        help="Soft-label Gaussian width in VA units (default: one bin width).")
    parser.add_argument("--differential_weight", type=float, default=0.0,
                        help="Only used when target_mode=absolute. Weight w in "
                             "loss = (1-w)*MSE_abs + w*MSE_diff. Default 0 (pure absolute).")
    parser.add_argument("--va_jitter_std", type=float, default=0.05,
                        help="Gaussian noise std added to labeled V/A during training "
                             "(clamped to [-1,1]). 0 disables.")
    parser.add_argument("--va_label_dropout", type=float, default=0.1,
                        help="Per-bar probability of dropping a labeled bar from the "
                             "training loss. 0 disables.")
    # Input representation
    parser.add_argument("--feature_mode", type=str, default="musetok",
                        choices=["musetok", "handcrafted", "remi"],
                        help="Bar input: musetok latents, handcrafted MIDI stats, or REMI tokens.")
    parser.add_argument("--remi_vocab_size", type=int, default=None,
                        help="REMI vocab size (inferred from MuseTok dict if omitted).")
    parser.add_argument("--remi_max_tokens", type=int, default=128)
    parser.add_argument("--remi_encoder_layers", type=int, default=2)
    parser.add_argument("--remi_encoder_d_model", type=int, default=128)
    parser.add_argument("--checkpoint_metric", type=str, default="mse",
                        choices=["mse", "corr_sum", "corr_valence"],
                        help="Metric for best checkpoint selection (lower is better for mse).")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Clip gradient norm (0 disables). Recommended for feature_mode=remi.")
    # Checkpointing / logging
    parser.add_argument("--output_dir",    type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model_name",    type=str, default="va_transformer_a")
    parser.add_argument("--wandb_project", type=str, default="gigamidi-support")
    parser.add_argument("--resume",        action="store_true")
    parser.add_argument("--config",        type=str, default=None)

    args_pre, _ = parser.parse_known_args()
    if getattr(args_pre, "config", None) and os.path.isfile(args_pre.config):
        apply_config(parser, load_config(args_pre.config))
    args = parser.parse_args()

    if args.latent_dim is None:
        try:
            if args.feature_mode == "handcrafted":
                args.latent_dim = HANDCRAFTED_FEATURE_DIM
            elif args.datasets:
                from datasets import get_dataset
                from pretrain_model.dataset import _features_dir_for_adapter
                for name in args.datasets:
                    ds = get_dataset(name, args.storage_dir)
                    feat_dir = _features_dir_for_adapter(ds, args.feature_mode)
                    if feat_dir.is_dir():
                        if args.feature_mode == "remi":
                            args.latent_dim = 128
                        else:
                            args.latent_dim = infer_input_dim(str(feat_dir))
                        break
            if args.latent_dim is None and args.feature_mode == "musetok":
                args.latent_dim = infer_input_dim(args.latents_dir)
        except FileNotFoundError:
            pass
        if args.latent_dim is None:
            args.latent_dim = 128

    if args.feature_mode == "remi" and args.remi_vocab_size is None:
        try:
            args.remi_vocab_size = remi_vocab_size(load_remi_vocab())
        except FileNotFoundError:
            args.remi_vocab_size = 0

    args.checkpoint_dir = os.path.join(args.output_dir, args.model_name, "checkpoints")
    ensure_dir(args.checkpoint_dir)
    return args


def load_song_list(path: str):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def _build_combined_datasets(dataset_names, split, storage_dir=None, feature_mode="musetok"):
    """Build CombinedVASequenceDataset from named sources; skip missing."""
    parts = []
    for name in dataset_names:
        ds_part = build_dataset_for_source(name, split, storage_dir, feature_mode=feature_mode)
        if ds_part is not None and len(ds_part) > 0:
            parts.append(ds_part)
            logging.info(f"{name} — {split}: {len(ds_part)} songs")
        else:
            logging.info(f"{name} — {split}: skipped (missing preprocessed files)")
    if not parts:
        return None
    if len(parts) == 1:
        return parts[0]
    return CombinedVASequenceDataset(parts)


def _legacy_build_datasets(args):
    """Backward-compatible DEAM + Memo2496 path-based loading."""
    train_songs = load_song_list(args.train_songs)
    valid_songs = load_song_list(args.valid_songs)
    logging.info(f"DEAM — train songs: {len(train_songs)}, valid songs: {len(valid_songs)}")

    memo2496_enabled = all([
        args.memo2496_latents_dir,
        args.memo2496_labels_path,
        args.memo2496_train_songs,
        args.memo2496_valid_songs,
        os.path.isdir(args.memo2496_latents_dir),
        os.path.isfile(args.memo2496_labels_path),
        os.path.isfile(args.memo2496_train_songs),
        os.path.isfile(args.memo2496_valid_songs),
    ])
    if memo2496_enabled:
        memo_train_songs = load_song_list(args.memo2496_train_songs)
        memo_valid_songs = load_song_list(args.memo2496_valid_songs)
        logging.info(
            f"Memo2496 — train songs: {len(memo_train_songs)}, valid songs: {len(memo_valid_songs)}"
        )
    else:
        logging.info("Memo2496 not found — DEAM only (legacy mode).")

    deam_train = DEAMSequenceDataset(
        latents_dir=args.latents_dir,
        labels_path=args.labels_path,
        song_list=train_songs,
    )
    deam_valid = DEAMSequenceDataset(
        latents_dir=args.latents_dir,
        labels_path=args.labels_path,
        song_list=valid_songs,
    )
    if memo2496_enabled:
        memo_train = Memo2496SequenceDataset(
            latents_dir=args.memo2496_latents_dir,
            labels_path=args.memo2496_labels_path,
            song_list=memo_train_songs,
        )
        memo_valid = Memo2496SequenceDataset(
            latents_dir=args.memo2496_latents_dir,
            labels_path=args.memo2496_labels_path,
            song_list=memo_valid_songs,
        )
        return (
            CombinedVASequenceDataset([deam_train, memo_train]),
            CombinedVASequenceDataset([deam_valid, memo_valid]),
        )
    return deam_train, deam_valid


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Using device: {device}")
    logging.info(f"Model variant: {'B (va_conditioning)' if args.va_conditioning else 'A (latents only)'}")
    logging.info(f"Feature mode: {args.feature_mode}")

    dataset_names = args.datasets
    if dataset_names:
        logging.info(f"Combined training datasets: {dataset_names}")
        train_dataset = _build_combined_datasets(
            dataset_names, "train", args.storage_dir, args.feature_mode,
        )
        valid_dataset = _build_combined_datasets(
            dataset_names, "valid", args.storage_dir, args.feature_mode,
        )
        if train_dataset is None or valid_dataset is None:
            logging.error("No training data available for requested datasets.")
            sys.exit(1)
    else:
        train_dataset, valid_dataset = _legacy_build_datasets(args)

    logging.info("Building DataLoaders...")
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=CombinedVASequenceDataset.collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=CombinedVASequenceDataset.collate_fn,
    )

    steps_per_epoch = max(1, len(train_loader))
    if args.epochs is not None:
        args.steps = int(args.epochs * steps_per_epoch)
        args.valid_steps = max(1, int(args.valid_every_epochs * steps_per_epoch))
        logging.info(
            f"Epoch mode: {args.epochs} epochs × {steps_per_epoch} steps/epoch "
            f"→ {args.steps} total steps, validate every {args.valid_steps} steps "
            f"({args.valid_every_epochs:g} epoch(s))"
        )
    else:
        logging.info(f"Step mode: {args.steps} total steps, {steps_per_epoch} steps/epoch "
                     f"(≈{args.steps / steps_per_epoch:.1f} epochs)")

    if args.target_mode == "binned" and args.differential_weight > 0.0:
        logging.warning("differential_weight ignored when target_mode=binned")

    output_mode = "binned" if args.target_mode == "binned" else "regression"
    bin_sigma = args.bin_sigma if args.bin_sigma is not None else default_bin_sigma(args.n_bins)

    model = VAModel(
        feature_mode=args.feature_mode,
        latent_dim=args.latent_dim,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        va_conditioning=args.va_conditioning,
        remi_vocab_size=args.remi_vocab_size or 0,
        remi_max_tokens=args.remi_max_tokens,
        remi_encoder_layers=args.remi_encoder_layers,
        remi_encoder_d_model=args.remi_encoder_d_model,
        output_mode=output_mode,
        n_bins=args.n_bins,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {n_params:,}")
    logging.info(f"Transformer dropout: {args.dropout}")
    if args.target_mode == "binned":
        logging.info(
            f"Binned targets: n_bins={args.n_bins}, bin_sigma={bin_sigma:.4f} "
            f"(bin width={2.0 / args.n_bins:.4f})"
        )
    if args.va_jitter_std > 0.0 or args.va_label_dropout > 0.0:
        logging.info(
            f"VA label regularization (train only): "
            f"jitter_std={args.va_jitter_std}, label_dropout={args.va_label_dropout}"
        )

    loss_fn   = nn.MSELoss(reduction="mean")
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )

    best_model_path     = os.path.join(args.checkpoint_dir, "best_model.pt")
    best_optimizer_path = os.path.join(args.checkpoint_dir, "best_optimizer.pt")
    stats_file = os.path.join(args.output_dir, args.model_name, "statistics.csv")
    log_file   = os.path.join(args.output_dir, args.model_name, "train.log")

    ensure_dir(os.path.dirname(log_file))
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(
        logging.FileHandler(log_file, mode="a" if args.resume else "w")
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Command: python {' '.join(sys.argv)}")
    logging.info(pprint.pformat(vars(args)))

    run_name = f"{args.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))

    if args.resume and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        optimizer.load_state_dict(torch.load(best_optimizer_path, map_location=device))
        logging.info("Resumed from checkpoint")

    stats_columns = [
        "step", "epoch", "split", "loss", "mse", "mae",
        "ar_mse", "ar_mae", "corr_valence", "corr_arousal",
    ]
    if not os.path.exists(stats_file) or not args.resume:
        pd.DataFrame(columns=stats_columns).to_csv(stats_file, index=False)

    state = {
        "best_checkpoint_value": float("inf"),
        "best_checkpoint_lower": True,
        "best_metrics": {},
        "early_stopping_counter": 0,
        "done": False,
    }
    segment_size = args.valid_steps if args.valid_steps > 0 else args.steps
    global_step  = 0
    batch_iter   = iter(train_loader)

    def run_validation():
        if train_count_accum == 0:
            return
        train_loss = train_loss_accum / train_count_accum
        train_metrics = {k: train_metrics_accum[k] / train_count_accum
                         for k in train_metrics_accum}

        model.eval()
        valid_loss = 0.0
        valid_metrics = {k: 0.0 for k in train_metrics}
        valid_count = 0
        all_preds, all_targets = [], []
        per_song_preds, per_song_targets, per_song_sources = [], [], []

        with torch.no_grad():
            for batch_v in valid_loader:
                loss_v, metrics_v, _ = evaluate_batch(
                    model, batch_v, loss_fn, device, return_predictions=True,
                    target_mode=args.target_mode,
                    differential_weight=args.differential_weight,
                    n_bins=args.n_bins,
                    bin_sigma=bin_sigma,
                )
                # Per-song preds for per-dataset breakdown (Model A only; Model B uses AR)
                if not args.va_conditioning:
                    latents_v = batch_v["latents"].to(device)
                    va_t = batch_v["va_targets"].to(device)
                    mask_v = batch_v["label_mask"].to(device)
                    padding_v = batch_v["padding_mask"].to(device)
                    fwd_kw = _model_forward_kwargs(batch_v, device)
                    raw_v = model(
                        latents_v, padding_mask=padding_v,
                        va_targets=va_t, label_mask=mask_v, **fwd_kw,
                    )
                    pred_v = model.decode_va(raw_v)
                    sources = batch_v.get("dataset_source", ["unknown"] * pred_v.shape[0])
                    for b in range(pred_v.shape[0]):
                        m = mask_v[b]
                        if not m.any():
                            continue
                        p_b = pred_v[b, m].cpu().numpy()
                        if not np.isfinite(p_b).all():
                            continue
                        per_song_preds.append(p_b)
                        per_song_targets.append(va_t[b, m].cpu().numpy())
                        per_song_sources.append(sources[b] if b < len(sources) else "unknown")
                    p_flat = pred_v[mask_v].cpu().numpy()
                    if np.isfinite(p_flat).all():
                        all_preds.append(p_flat)
                        all_targets.append(va_t[mask_v].cpu().numpy())
                else:
                    mask_v = batch_v["label_mask"].to(device)

                n = metrics_v.get("n_labeled", int(mask_v.sum().item()))
                if n > 0 and np.isfinite(loss_v):
                    valid_loss += loss_v * n
                    for k in valid_metrics:
                        valid_metrics[k] += metrics_v[k] * n
                    valid_count += n

        ar_metrics = None
        if args.va_conditioning:
            ar_metrics = evaluate_autoregressive(
                model, valid_loader, device, return_per_song=True,
            )

        if valid_count:
            valid_loss /= valid_count
            for k in valid_metrics:
                valid_metrics[k] /= valid_count
            if not args.va_conditioning:
                p_all = np.concatenate(all_preds, axis=0)
                t_all = np.concatenate(all_targets, axis=0)
                if p_all.shape[0] > 1 and np.isfinite(p_all).all():
                    rv, ra = _pearson_corr(p_all, t_all)
                    valid_metrics["corr_valence"] = rv
                    valid_metrics["corr_arousal"] = ra

        epoch = global_step / steps_per_epoch
        if args.va_conditioning and ar_metrics and ar_metrics.get("per_song_preds"):
            ds_corr = _per_dataset_corr(
                ar_metrics["per_song_preds"],
                ar_metrics["per_song_targets"],
                ar_metrics["per_song_sources"],
            )
        else:
            ds_corr = _per_dataset_corr(per_song_preds, per_song_targets, per_song_sources)
        if ds_corr:
            mode = "AR" if args.va_conditioning else "TF"
            ds_parts = [
                f"{src}: v={m['corr_valence']:.3f} a={m['corr_arousal']:.3f}"
                for src, m in sorted(ds_corr.items())
            ]
            logging.info(f"Per-dataset valid corr ({mode}) — " + ", ".join(ds_parts))

        if args.va_conditioning and ar_metrics:
            logging.info(
                f"Step {global_step} (epoch {epoch:.2f}) — Train Loss: {train_loss:.4f} | "
                f"Valid TF MSE: {valid_metrics['mse']:.4f} | "
                f"Valid AR MSE: {ar_metrics['mse']:.4f}, "
                f"corr_v: {ar_metrics['corr_valence']:.3f}, "
                f"corr_a: {ar_metrics['corr_arousal']:.3f}"
            )
        else:
            logging.info(
                f"Step {global_step} (epoch {epoch:.2f}) — Train Loss: {train_loss:.4f} | "
                f"Valid MSE: {valid_metrics['mse']:.4f}, "
                f"corr_v: {valid_metrics['corr_valence']:.3f}, "
                f"corr_a: {valid_metrics['corr_arousal']:.3f}"
            )

        for split, lv, m in [("train", train_loss, train_metrics),
                              ("valid", valid_loss, valid_metrics)]:
            row = {
                "step": global_step, "epoch": epoch, "split": split,
                "loss": lv, **m,
                "ar_mse": np.nan, "ar_mae": np.nan,
            }
            if split == "valid" and ar_metrics:
                row["ar_mse"] = ar_metrics["mse"]
                row["ar_mae"] = ar_metrics["mae"]
                row["corr_valence"] = ar_metrics["corr_valence"]
                row["corr_arousal"] = ar_metrics["corr_arousal"]
            elif split == "valid" and args.va_conditioning:
                row["corr_valence"] = np.nan
                row["corr_arousal"] = np.nan
            pd.DataFrame([row]).to_csv(stats_file, mode="a", header=False, index=False)

        wandb_payload = {
            "epoch": epoch,
            "train/loss": train_loss,
            "valid/loss": valid_loss,
            "train/mse": train_metrics["mse"],
            "train/mae": train_metrics["mae"],
            "valid/mse": valid_metrics["mse"],
            "valid/mae": valid_metrics["mae"],
        }
        if args.va_conditioning:
            if ar_metrics:
                wandb_payload["valid/ar_mse"] = ar_metrics["mse"]
                wandb_payload["valid/ar_mae"] = ar_metrics["mae"]
                wandb_payload["valid/corr_valence"] = ar_metrics["corr_valence"]
                wandb_payload["valid/corr_arousal"] = ar_metrics["corr_arousal"]
        else:
            wandb_payload["train/corr_valence"] = train_metrics["corr_valence"]
            wandb_payload["train/corr_arousal"] = train_metrics["corr_arousal"]
            wandb_payload["valid/corr_valence"] = valid_metrics["corr_valence"]
            wandb_payload["valid/corr_arousal"] = valid_metrics["corr_arousal"]
        for src, m in ds_corr.items():
            prefix = "valid/ar" if args.va_conditioning else "valid"
            wandb_payload[f"{prefix}/{src}/corr_valence"] = m["corr_valence"]
            wandb_payload[f"{prefix}/{src}/corr_arousal"] = m["corr_arousal"]
        wandb.log(wandb_payload, step=global_step)

        ckpt_val, lower_better = _checkpoint_value(
            valid_metrics, ar_metrics, args.va_conditioning, args.checkpoint_metric,
        )
        if state["best_metrics"] == {}:
            improved = True
        elif lower_better:
            improved = ckpt_val < state["best_checkpoint_value"]
        else:
            improved = ckpt_val > state["best_checkpoint_value"]
        if improved:
            state["best_checkpoint_value"] = ckpt_val
            state["best_checkpoint_lower"] = lower_better
            state["best_metrics"] = (
                {**valid_metrics, **{f"ar_{k}": v for k, v in ar_metrics.items()}}
                if ar_metrics else valid_metrics.copy()
            )
            torch.save(model.state_dict(),     best_model_path)
            torch.save(optimizer.state_dict(), best_optimizer_path)
            metric_label = args.checkpoint_metric
            logging.info(f"Saved best model (valid {metric_label}: {ckpt_val:.4f})")
            state["early_stopping_counter"] = 0
        else:
            state["early_stopping_counter"] += 1

        if args.early_stopping and state["early_stopping_counter"] >= args.early_stopping_tolerance:
            logging.info(
                f"Early stopping after {state['early_stopping_counter']} "
                "validations without improvement"
            )
            state["done"] = True
        model.train()

    logging.info(
        f"Starting training for {args.steps} steps. "
        f"Validation every {args.valid_steps} steps."
    )
    while global_step < args.steps:
        model.train()
        train_loss_accum    = 0.0
        train_metrics_accum = {k: 0.0 for k in ["mse", "mae", "corr_valence", "corr_arousal"]}
        train_count_accum   = 0
        segment_len = min(segment_size, args.steps - global_step)

        pbar = tqdm(total=segment_len, desc="Training", unit="step")
        for _ in range(segment_len):
            try:
                batch = next(batch_iter)
            except StopIteration:
                batch_iter = iter(train_loader)
                batch = next(batch_iter)

            loss, metrics = evaluate_batch(
                model, batch, loss_fn, device,
                update_parameters=True, optimizer=optimizer,
                target_mode=args.target_mode,
                differential_weight=args.differential_weight,
                va_jitter_std=args.va_jitter_std,
                va_label_dropout=args.va_label_dropout,
                max_grad_norm=args.max_grad_norm,
                n_bins=args.n_bins,
                bin_sigma=bin_sigma,
            )
            n = metrics.get("n_labeled", int(batch["label_mask"].sum().item()))
            if n > 0 and np.isfinite(loss):
                train_loss_accum += loss * n
                for k in train_metrics_accum:
                    train_metrics_accum[k] += metrics[k] * n
                train_count_accum += n
            global_step += 1
            pbar.update(1)
        pbar.close()
        run_validation()
        if state["done"]:
            break

    logging.info(
        f"Training complete. Best valid {args.checkpoint_metric}: "
        f"{state['best_checkpoint_value']:.4f}"
    )
    logging.info(f"Best metrics: {state['best_metrics']}")
    wandb.finish()
