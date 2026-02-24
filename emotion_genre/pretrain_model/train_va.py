"""
Training script for valence–arousal regression from XMIDI latents.

Uses emotion labels and a fixed mapping from emotion index to (valence, arousal).
Supports class-weighted MSE and balanced sampling so rare emotions contribute more.
"""

import argparse
import logging
import pprint
import sys
import os
from os.path import dirname, realpath
from multiprocessing import cpu_count
import wandb
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import warnings
import numpy as np
warnings.simplefilter(action="ignore", category=FutureWarning)

sys.path.insert(0, dirname(dirname(realpath(__file__))))

from pretrain_model.dataset import XMIDIDatasetVA
from pretrain_model.model import ValenceArousalRegressor
from utils.data_utils import (
    TRAINED_MODEL_DIR, XMIDI_LATENTS_DIR, XMIDI_LABELS_DIR,
    ensure_dir, load_json
)


def evaluate_batch_va(
    model: nn.Module,
    batch: dict,
    loss_fn: nn.Module,
    device: torch.device,
    update_parameters: bool = False,
    optimizer: torch.optim.Optimizer = None,
    return_predictions: bool = False,
    sample_weights: torch.Tensor = None,
):
    """Forward pass, optional backward, and metrics (MSE, MAE, correlation)."""
    latents = batch["latents"].to(device)
    va = batch["va"].to(device)  # (batch_size, 2)
    emotion_index = batch["emotion_index"].to(device)  # (batch_size,)

    if update_parameters:
        optimizer.zero_grad()

    pred = model(latents)  # (batch_size, 2)
    if sample_weights is not None:
        sq_err = ((pred - va) ** 2).sum(dim=1)
        loss = (sample_weights * sq_err).sum() / sample_weights.sum().clamp(min=1e-8)
    else:
        loss = loss_fn(pred, va)

    if update_parameters:
        loss.backward()
        optimizer.step()

    loss_value = float(loss.detach())
    with torch.no_grad():
        mse = ((pred - va) ** 2).mean().item()
        mae = (pred - va).abs().mean().item()
        pred_np = pred.cpu().numpy()
        va_np = va.cpu().numpy()
        corr_v = np.corrcoef(pred_np[:, 0], va_np[:, 0])[0, 1] if pred_np.shape[0] > 1 else 0.0
        corr_a = np.corrcoef(pred_np[:, 1], va_np[:, 1])[0, 1] if pred_np.shape[0] > 1 else 0.0
        corr_v = 0.0 if np.isnan(corr_v) else corr_v
        corr_a = 0.0 if np.isnan(corr_a) else corr_a
    metrics = {"mse": mse, "mae": mae, "corr_valence": corr_v, "corr_arousal": corr_a}
    if return_predictions:
        return loss_value, metrics, (pred_np, va_np)
    return loss_value, metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Train valence–arousal regressor from XMIDI latents.")
    parser.add_argument("--latents_dir", type=str, default=XMIDI_LATENTS_DIR, help="Directory containing latents")
    parser.add_argument("--emotion_labels_path", type=str, required=True,
                        help="Path to emotion_labels.json")
    parser.add_argument("--emotion_to_index_path", type=str, required=True,
                        help="Path to emotion_to_index.json (for class weights / sampler)")
    parser.add_argument("--train_files", type=str, required=True, help="Path to train_files.txt")
    parser.add_argument("--valid_files", type=str, required=True, help="Path to val_files.txt")
    parser.add_argument("--preprocessor", choices=["musetok", "midi2vec"], default="musetok",
                        help="Preprocessor (affects default input_dim)")
    parser.add_argument("--input_dim", type=int, default=None,
                        help="Input dimension (default: 128 musetok, 100 midi2vec)")
    parser.add_argument("--hidden_dim", type=int, default=None, help="Hidden dimension (default: input_dim // 2)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stopping_tolerance", type=int, default=10)
    parser.add_argument("--class_weight", type=str, default="balanced", choices=["none", "balanced"],
                        help="Weight MSE by inverse emotion frequency")
    parser.add_argument("--balanced_sampler", action="store_true",
                        help="WeightedRandomSampler by emotion class")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=max(1, int(cpu_count() / 4)))
    parser.add_argument("--output_dir", type=str, default=TRAINED_MODEL_DIR)
    parser.add_argument("--model_name", type=str, default="valence_arousal_regressor")
    parser.add_argument("--wandb_project", type=str, default="gigamidi-support")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.input_dim is None:
        args.input_dim = 100 if args.preprocessor == "midi2vec" else 128
    if args.hidden_dim is None:
        args.hidden_dim = args.input_dim // 2
    args.checkpoint_dir = os.path.join(args.output_dir, args.model_name, "checkpoints")
    ensure_dir(args.checkpoint_dir)
    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Using device: {device}")

    def load_file_list(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    train_files = load_file_list(args.train_files)
    valid_files = load_file_list(args.valid_files)
    logging.info(f"Train files: {len(train_files)}, Valid files: {len(valid_files)}")

    labels_dict = load_json(args.emotion_labels_path)
    class_to_index = load_json(args.emotion_to_index_path)
    index_to_class = {}
    for k, v in class_to_index.items():
        idx = int(v) if isinstance(v, str) else v
        index_to_class[idx] = k
    num_emotions = len(class_to_index)
    train_labels = [labels_dict.get(f) for f in train_files if f in labels_dict]
    valid_labels = [labels_dict.get(f) for f in valid_files if f in labels_dict]
    train_counts = np.bincount(train_labels, minlength=num_emotions)
    logging.info(f"Per-emotion count (train): {train_counts.tolist()}")

    if args.class_weight == "balanced":
        class_weights = np.zeros(num_emotions, dtype=np.float32)
        for c in range(num_emotions):
            n_c = max(1, train_counts[c])
            class_weights[c] = len(train_labels) / (num_emotions * n_c)
        class_weights_tensor = torch.from_numpy(class_weights).float().to(device)
        logging.info(f"Class weights (balanced): {[f'{w:.3f}' for w in class_weights]}")
    else:
        class_weights_tensor = None

    train_dataset = XMIDIDatasetVA(
        latents_dir=args.latents_dir,
        emotion_labels_path=args.emotion_labels_path,
        file_list=train_files,
    )
    valid_dataset = XMIDIDatasetVA(
        latents_dir=args.latents_dir,
        emotion_labels_path=args.emotion_labels_path,
        file_list=valid_files,
    )

    if args.balanced_sampler:
        train_label_per_index = [labels_dict.get(f) for f in train_files]
        sample_weights = np.array([
            1.0 / max(1, train_counts[int(l)]) if l is not None else 1.0
            for l in train_label_per_index
        ], dtype=np.float64)
        sample_weights /= sample_weights.sum()
        sampler = torch.utils.data.WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights),
            num_samples=len(train_dataset),
            replacement=True,
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, sampler=sampler,
            num_workers=args.num_workers, collate_fn=XMIDIDatasetVA.collate_fn,
        )
        logging.info("Using WeightedRandomSampler for balanced batches")
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=XMIDIDatasetVA.collate_fn,
        )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=XMIDIDatasetVA.collate_fn,
    )

    model = ValenceArousalRegressor(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.MSELoss(reduction="mean")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    run_name = f"{args.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    wandb.init(project=args.wandb_project, name=run_name, config=vars(args))
    log_file = os.path.join(args.output_dir, args.model_name, "train.log")
    ensure_dir(os.path.dirname(log_file))
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logging.FileHandler(log_file, mode="a" if args.resume else "w"))
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(f"Command: python {' '.join(sys.argv)}")
    logging.info(pprint.pformat(vars(args)))

    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    best_optimizer_path = os.path.join(args.checkpoint_dir, "best_optimizer.pt")
    if args.resume and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        optimizer.load_state_dict(torch.load(best_optimizer_path, map_location=device))
        logging.info("Resumed from checkpoint")

    stats_file = os.path.join(args.output_dir, args.model_name, "statistics.csv")
    stats_columns = ["epoch", "split", "loss", "mse", "mae", "corr_valence", "corr_arousal"]
    if not os.path.exists(stats_file) or not args.resume:
        pd.DataFrame(columns=stats_columns).to_csv(stats_file, index=False)

    best_valid_mse = float("inf")
    best_metrics = {}
    early_stopping_counter = 0

    for epoch in range(args.epochs):
        logging.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        model.train()
        train_loss = 0.0
        train_metrics = {k: 0.0 for k in ["mse", "mae", "corr_valence", "corr_arousal"]}
        train_count = 0
        for batch in tqdm(train_loader, desc="Training"):
            if class_weights_tensor is not None:
                w = class_weights_tensor[batch["emotion_index"]].to(device)
            else:
                w = None
            loss, metrics = evaluate_batch_va(
                model, batch, loss_fn, device,
                update_parameters=True, optimizer=optimizer,
                sample_weights=w,
            )
            bs = len(batch["latents"])
            train_loss += loss * bs
            for k in train_metrics:
                train_metrics[k] += metrics[k] * bs
            train_count += bs
        train_loss /= train_count
        for k in train_metrics:
            train_metrics[k] /= train_count
        logging.info(f"Train - Loss: {train_loss:.4f}, MSE: {train_metrics['mse']:.4f}, MAE: {train_metrics['mae']:.4f}, corr_v: {train_metrics['corr_valence']:.3f}, corr_a: {train_metrics['corr_arousal']:.3f}")

        model.eval()
        valid_loss = 0.0
        valid_metrics = {k: 0.0 for k in train_metrics}
        valid_count = 0
        all_preds, all_va = [], []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                loss, metrics, (pred_np, va_np) = evaluate_batch_va(
                    model, batch, loss_fn, device, return_predictions=True,
                )
                bs = len(batch["latents"])
                valid_loss += loss * bs
                for k in valid_metrics:
                    valid_metrics[k] += metrics[k] * bs
                valid_count += bs
                all_preds.append(pred_np)
                all_va.append(va_np)
        valid_loss /= valid_count
        for k in valid_metrics:
            valid_metrics[k] /= valid_count
        all_preds = np.concatenate(all_preds, axis=0)
        all_va = np.concatenate(all_va, axis=0)
        logging.info(f"Valid - Loss: {valid_loss:.4f}, MSE: {valid_metrics['mse']:.4f}, MAE: {valid_metrics['mae']:.4f}, corr_v: {valid_metrics['corr_valence']:.3f}, corr_a: {valid_metrics['corr_arousal']:.3f}")

        for split, loss_val, m in [("train", train_loss, train_metrics), ("valid", valid_loss, valid_metrics)]:
            pd.DataFrame([{"epoch": epoch + 1, "split": split, "loss": loss_val, **m}]).to_csv(
                stats_file, mode="a", header=False, index=False
            )
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss, "valid/loss": valid_loss,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"valid/{k}": v for k, v in valid_metrics.items()},
        })

        if valid_metrics["mse"] < best_valid_mse:
            best_valid_mse = valid_metrics["mse"]
            best_metrics = valid_metrics.copy()
            torch.save(model.state_dict(), best_model_path)
            torch.save(optimizer.state_dict(), best_optimizer_path)
            logging.info(f"Saved best model (valid MSE: {best_valid_mse:.4f})")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        if args.early_stopping and early_stopping_counter >= args.early_stopping_tolerance:
            logging.info(f"Early stopping after {args.early_stopping_tolerance} epochs without improvement")
            break

    logging.info("Training complete!")
    logging.info(f"Best valid MSE: {best_valid_mse:.4f}")
    logging.info(f"Best metrics: {best_metrics}")
    wandb.finish()
