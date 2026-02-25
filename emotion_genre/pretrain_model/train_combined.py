"""
Training script for emotion/genre using combined MuseTok + midi2vec features.

Loads latents from two dirs (MuseTok and midi2vec), concatenates them, normalizes
per dimension (mean/std computed on the training set), and trains the same classifier.
input_dim is inferred from the combined stats (no need to pass it).
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
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
warnings.simplefilter(action="ignore", category=FutureWarning)

sys.path.insert(0, dirname(dirname(realpath(__file__))))

from pretrain_model.dataset import CombinedLatentsDataset, compute_combined_latents_stats
from pretrain_model.model import EmotionGenreClassifier
from pretrain_model.train import evaluate_batch
from utils.data_utils import (
    TRAINED_MODEL_DIR, XMIDI_LATENTS_DIR, XMIDI_LABELS_DIR,
    ensure_dir, load_json
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train emotion/genre classifier on combined MuseTok + midi2vec latents (concatenated, per-dim normalized)."
    )
    parser.add_argument("--task", type=str, required=True, choices=["emotion", "genre"])
    parser.add_argument("--latents_dir_musetok", type=str, required=True,
                        help="Directory containing MuseTok latents (.safetensors)")
    parser.add_argument("--latents_dir_midi2vec", type=str, required=True,
                        help="Directory containing midi2vec latents (.safetensors)")
    parser.add_argument("--stats_path", type=str, default=None,
                        help="Path to .npz with mean/std for normalization. If missing, computed from train set and saved.")
    parser.add_argument("--labels_path", type=str, required=True,
                        help="Path to labels JSON (emotion_labels.json or genre_labels.json)")
    parser.add_argument("--class_to_index_path", type=str, required=True,
                        help="Path to class_to_index JSON")
    parser.add_argument("--train_files", type=str, required=True, help="Path to train_files.txt")
    parser.add_argument("--valid_files", type=str, required=True, help="Path to val_files.txt")
    parser.add_argument("--num_classes", type=int, required=True, help="11 for emotion, 6 for genre")
    parser.add_argument("--hidden_dim", type=int, default=None,
                        help="Hidden dimension (default: input_dim // 2)")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--early_stopping", action="store_true")
    parser.add_argument("--early_stopping_tolerance", type=int, default=10)
    parser.add_argument("--class_weight", type=str, default="balanced", choices=["none", "balanced"])
    parser.add_argument("--balanced_sampler", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=max(1, int(cpu_count() / 4)))
    parser.add_argument("--output_dir", type=str, default=TRAINED_MODEL_DIR)
    parser.add_argument("--model_name", type=str, default=None,
                        help="Default: {task}_classifier_combined")
    parser.add_argument("--wandb_project", type=str, default="gigamidi-support")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.model_name is None:
        args.model_name = f"{args.task}_classifier_combined"
    args.checkpoint_dir = os.path.join(args.output_dir, args.model_name, "checkpoints")
    ensure_dir(args.checkpoint_dir)
    if args.stats_path is None:
        args.stats_path = os.path.join(args.output_dir, args.model_name, "combined_latents_stats.npz")
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

    labels_dict = load_json(args.labels_path)
    class_to_index = load_json(args.class_to_index_path)
    index_to_class = {}
    for k, v in class_to_index.items():
        idx = int(v) if isinstance(v, str) else v
        index_to_class[idx] = k
    num_classes = len(class_to_index)
    train_labels = [labels_dict.get(f) for f in train_files if f in labels_dict]
    valid_labels = [labels_dict.get(f) for f in valid_files if f in labels_dict]
    train_counts = np.bincount(train_labels, minlength=num_classes)
    valid_counts = np.bincount(valid_labels, minlength=num_classes)
    logging.info(f"Per-class count in train: {train_counts.tolist()}")
    logging.info(f"Per-class count in valid: {valid_counts.tolist()}")

    # Normalization stats: compute from train set if not present
    if not os.path.isfile(args.stats_path):
        logging.info(f"Computing combined latents stats from training set -> {args.stats_path}")
        compute_combined_latents_stats(
            args.latents_dir_musetok,
            args.latents_dir_midi2vec,
            train_files,
            args.stats_path,
        )
    data = np.load(args.stats_path)
    input_dim = int(data["mean"].shape[0])
    args.input_dim = input_dim
    if args.hidden_dim is None:
        args.hidden_dim = input_dim // 2
    logging.info(f"Combined input_dim (from stats): {input_dim}")

    if args.class_weight == "balanced":
        class_weights = np.zeros(num_classes, dtype=np.float32)
        for c in range(num_classes):
            n_c = max(1, train_counts[c])
            class_weights[c] = len(train_labels) / (num_classes * n_c)
        class_weights_tensor = torch.from_numpy(class_weights).float().to(device)
        logging.info(f"Class weights (balanced): {class_weights.tolist()}")
    else:
        class_weights_tensor = None

    train_dataset = CombinedLatentsDataset(
        latents_dir_musetok=args.latents_dir_musetok,
        latents_dir_midi2vec=args.latents_dir_midi2vec,
        stats_path=args.stats_path,
        labels_path=args.labels_path,
        class_to_index_path=args.class_to_index_path,
        file_list=train_files,
        task=args.task,
    )
    valid_dataset = CombinedLatentsDataset(
        latents_dir_musetok=args.latents_dir_musetok,
        latents_dir_midi2vec=args.latents_dir_midi2vec,
        stats_path=args.stats_path,
        labels_path=args.labels_path,
        class_to_index_path=args.class_to_index_path,
        file_list=valid_files,
        task=args.task,
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
            num_workers=args.num_workers, collate_fn=CombinedLatentsDataset.collate_fn,
        )
        logging.info("Using WeightedRandomSampler for balanced batches")
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.num_workers, collate_fn=CombinedLatentsDataset.collate_fn,
        )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=CombinedLatentsDataset.collate_fn,
    )

    model = EmotionGenreClassifier(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    logging.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
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
    stats_columns = ["epoch", "split", "loss", "accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    if not os.path.exists(stats_file) or not args.resume:
        pd.DataFrame(columns=stats_columns).to_csv(stats_file, index=False)

    best_accuracy = 0.0
    best_metrics = {}
    early_stopping_counter = 0

    for epoch in range(args.epochs):
        logging.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        model.train()
        train_loss = 0.0
        train_metrics = {k: 0.0 for k in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]}
        train_count = 0
        for batch in tqdm(train_loader, desc="Training"):
            loss, metrics = evaluate_batch(
                model, batch, loss_fn, device,
                update_parameters=True, optimizer=optimizer,
            )
            bs = len(batch["latents"])
            train_loss += loss * bs
            for k in train_metrics:
                train_metrics[k] += metrics[k] * bs
            train_count += bs
        train_loss /= train_count
        for k in train_metrics:
            train_metrics[k] /= train_count
        logging.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_macro']:.4f}")

        model.eval()
        valid_loss = 0.0
        valid_metrics = {k: 0.0 for k in train_metrics}
        valid_count = 0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                loss, metrics, (preds, labels) = evaluate_batch(
                    model, batch, loss_fn, device, return_predictions=True,
                )
                bs = len(batch["latents"])
                valid_loss += loss * bs
                for k in valid_metrics:
                    valid_metrics[k] += metrics[k] * bs
                valid_count += bs
                all_preds.extend(preds.tolist())
                all_labels.extend(labels.tolist())
        valid_loss /= valid_count
        for k in valid_metrics:
            valid_metrics[k] /= valid_count
        logging.info(f"Valid - Loss: {valid_loss:.4f}, Accuracy: {valid_metrics['accuracy']:.4f}, F1: {valid_metrics['f1_macro']:.4f}")

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

        if valid_metrics["accuracy"] > best_accuracy:
            best_accuracy = valid_metrics["accuracy"]
            best_metrics = valid_metrics.copy()
            torch.save(model.state_dict(), best_model_path)
            torch.save(optimizer.state_dict(), best_optimizer_path)
            logging.info(f"Saved best model (accuracy: {best_accuracy:.4f})")
            early_stopping_counter = 0
            cm = confusion_matrix(all_labels, all_preds, labels=list(range(args.num_classes)))
            class_names = [index_to_class[i] for i in range(args.num_classes)]
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=class_names, yticklabels=class_names)
            plt.title(f"Confusion Matrix - Epoch {epoch + 1}")
            plt.ylabel("True Label")
            plt.xlabel("Predicted Label")
            plt.tight_layout()
            plt.savefig(os.path.join(args.output_dir, args.model_name, "confusion_matrix_best.png"), dpi=150)
            plt.close()
        else:
            early_stopping_counter += 1
        if args.early_stopping and early_stopping_counter >= args.early_stopping_tolerance:
            logging.info(f"Early stopping after {args.early_stopping_tolerance} epochs without improvement")
            break

    logging.info("Training complete!")
    logging.info(f"Best validation accuracy: {best_accuracy:.4f}")
    logging.info(f"Best metrics: {best_metrics}")
    wandb.finish()
