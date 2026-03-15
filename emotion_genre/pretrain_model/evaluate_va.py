"""
Evaluation script for valence–arousal regression model.
"""

import argparse
import logging
import sys
import os
from os.path import dirname, realpath
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, dirname(dirname(realpath(__file__))))

from pretrain_model.dataset import XMIDIDatasetVA
from pretrain_model.model import ValenceArousalRegressor
from pretrain_model.train_va import evaluate_batch_va
from utils.data_utils import EVALUATION_RESULTS_DIR, ensure_dir, load_json, infer_input_dim


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate valence–arousal regressor.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--latents_dir", type=str, required=True, help="Directory containing latents")
    parser.add_argument("--emotion_labels_path", type=str, required=True, help="Path to emotion_labels.json")
    parser.add_argument("--test_files", type=str, required=True, help="Path to test_files.txt")
    parser.add_argument("--preprocessor", choices=["musetok", "midi2vec"], default="musetok")
    parser.add_argument("--input_dim", type=int, default=None,
                        help="Inferred from latents_dir if omitted")
    parser.add_argument("--bars_per_chunk", type=int, default=-1,
                        help="Bars per chunk: -1=song-level, N>0=N bars (MuseTok only)")
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Default: EVALUATION_RESULTS_DIR/valence_arousal")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (CLI overrides config)")
    args_pre, _ = parser.parse_known_args()
    if getattr(args_pre, "config", None) and os.path.isfile(args_pre.config):
        from utils.config_utils import load_config, apply_config
        apply_config(parser, load_config(args_pre.config))
    args = parser.parse_args()
    if args.input_dim is None:
        try:
            args.input_dim = infer_input_dim(args.latents_dir)
        except FileNotFoundError:
            args.input_dim = 100 if args.preprocessor == "midi2vec" else 128
    if args.hidden_dim is None:
        args.hidden_dim = args.input_dim // 2
    if args.output_dir is None:
        args.output_dir = os.path.join(EVALUATION_RESULTS_DIR, "valence_arousal")
    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Using device: {device}")
    ensure_dir(args.output_dir)

    model = ValenceArousalRegressor(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    logging.info(f"Loaded model from {args.checkpoint_path}")

    def load_file_list(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    test_files = load_file_list(args.test_files)
    logging.info(f"Test files: {len(test_files)}")
    test_dataset = XMIDIDatasetVA(
        latents_dir=args.latents_dir,
        emotion_labels_path=args.emotion_labels_path,
        file_list=test_files,
        bars_per_chunk=args.bars_per_chunk,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=XMIDIDatasetVA.collate_fn,
    )

    loss_fn = nn.MSELoss()
    all_preds = []
    all_va = []
    total_loss = 0.0
    total_mse = 0.0
    total_mae = 0.0
    count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            loss, metrics, (pred_np, va_np) = evaluate_batch_va(
                model, batch, loss_fn, device, return_predictions=True,
            )
            bs = len(batch["latents"])
            total_loss += loss * bs
            total_mse += metrics["mse"] * bs
            total_mae += metrics["mae"] * bs
            count += bs
            all_preds.append(pred_np)
            all_va.append(va_np)
    all_preds = np.concatenate(all_preds, axis=0)
    all_va = np.concatenate(all_va, axis=0)
    total_loss /= count
    total_mse /= count
    total_mae /= count
    corr_v = np.corrcoef(all_preds[:, 0], all_va[:, 0])[0, 1]
    corr_a = np.corrcoef(all_preds[:, 1], all_va[:, 1])[0, 1]
    corr_v = 0.0 if np.isnan(corr_v) else corr_v
    corr_a = 0.0 if np.isnan(corr_a) else corr_a

    logging.info("\nTest Results (Valence–Arousal):")
    logging.info(f"Loss (MSE): {total_loss:.4f}")
    logging.info(f"MSE: {total_mse:.4f}")
    logging.info(f"MAE: {total_mae:.4f}")
    logging.info(f"Correlation (valence):  {corr_v:.4f}")
    logging.info(f"Correlation (arousal):  {corr_a:.4f}")

    results = {
        "loss": total_loss,
        "mse": total_mse,
        "mae": total_mae,
        "corr_valence": corr_v,
        "corr_arousal": corr_a,
    }
    pd.DataFrame([results]).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    logging.info(f"Saved metrics to {os.path.join(args.output_dir, 'metrics.csv')}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.scatter(all_va[:, 0], all_preds[:, 0], alpha=0.5, s=5)
    ax1.plot([-1, 1], [-1, 1], "k--", alpha=0.5)
    ax1.set_xlabel("True valence")
    ax1.set_ylabel("Predicted valence")
    ax1.set_title(f"Valence (corr={corr_v:.3f})")
    ax1.set_xlim(-1.2, 1.2)
    ax1.set_ylim(-1.2, 1.2)
    ax1.set_aspect("equal")
    ax2.scatter(all_va[:, 1], all_preds[:, 1], alpha=0.5, s=5)
    ax2.plot([-1, 1], [-1, 1], "k--", alpha=0.5)
    ax2.set_xlabel("True arousal")
    ax2.set_ylabel("Predicted arousal")
    ax2.set_title(f"Arousal (corr={corr_a:.3f})")
    ax2.set_xlim(-1.2, 1.2)
    ax2.set_ylim(-1.2, 1.2)
    ax2.set_aspect("equal")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "valence_arousal_scatter.png"), dpi=150)
    plt.close()
    logging.info(f"Saved scatter plot to {os.path.join(args.output_dir, 'valence_arousal_scatter.png')}")
    logging.info(f"\nAll results saved to {args.output_dir}")
