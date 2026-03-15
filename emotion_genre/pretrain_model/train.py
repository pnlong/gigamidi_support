"""
Training script for emotion/genre classification model.
"""

import argparse
import logging
import pprint
import sys
import os
from os.path import exists, dirname, realpath
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

# Add parent directory to path
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from pretrain_model.dataset import XMIDIDataset, get_bootstrap_downsampled_file_list
from pretrain_model.model import EmotionGenreClassifier
from utils.data_utils import (
    TRAINED_MODEL_DIR, XMIDI_LATENTS_DIR, XMIDI_LABELS_DIR,
    ensure_dir, save_json, load_json, infer_input_dim,
)

# ================================================== #
#  Batch Evaluation Function                        #
# ================================================== #

def evaluate_batch(
    model: nn.Module,
    batch: dict,
    loss_fn: nn.Module,
    device: torch.device,
    update_parameters: bool = False,
    optimizer: torch.optim.Optimizer = None,
    return_predictions: bool = False,
):
    """
    Evaluate model on a batch, updating parameters if specified.
    
    Returns:
        loss: float
        metrics: dict with 'accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro'
        predictions: (optional) tuple of (pred_labels, true_labels)
    """
    latents = batch["latents"].to(device)
    labels = batch["label"].to(device)
    
    # Zero gradients
    if update_parameters:
        optimizer.zero_grad()
    
    # Forward pass
    logits = model(latents)  # (batch_size, num_classes)
    
    # Compute loss
    loss = loss_fn(logits, labels)
    
    # Backward pass
    if update_parameters:
        loss.backward()
        optimizer.step()
    
    loss_value = float(loss.detach())
    
    # Compute metrics
    with torch.no_grad():
        pred_labels = torch.argmax(logits, dim=1)
        pred_labels_np = pred_labels.cpu().numpy()
        true_labels_np = labels.cpu().numpy()
        
        accuracy = accuracy_score(true_labels_np, pred_labels_np)
        f1_macro = f1_score(true_labels_np, pred_labels_np, average='macro', zero_division=0)
        f1_weighted = f1_score(true_labels_np, pred_labels_np, average='weighted', zero_division=0)
        precision_macro = precision_score(true_labels_np, pred_labels_np, average='macro', zero_division=0)
        recall_macro = recall_score(true_labels_np, pred_labels_np, average='macro', zero_division=0)
    
    metrics = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
    }
    
    # Clean up
    del latents, labels, logits, pred_labels
    
    if return_predictions:
        return loss_value, metrics, (pred_labels_np, true_labels_np)
    else:
        return loss_value, metrics

# ================================================== #
#  Argument Parsing                                 #
# ================================================== #

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="Train", description="Train emotion/genre classification model.")
    
    # Task
    parser.add_argument("--task", type=str, default=None,
                       choices=["emotion", "genre"],
                       help="Task: emotion or genre (required if not using --config)")
    
    # Data paths
    parser.add_argument("--latents_dir", type=str, default=XMIDI_LATENTS_DIR,
                       help="Directory containing XMIDI latents")
    parser.add_argument("--labels_path", type=str, default=None,
                       help="Path to labels JSON file (emotion_labels.json or genre_labels.json)")
    parser.add_argument("--class_to_index_path", type=str, default=None,
                       help="Path to class_to_index JSON file (emotion_to_index.json or genre_to_index.json)")
    parser.add_argument("--train_files", type=str, default=None,
                       help="Path to train_files.txt")
    parser.add_argument("--valid_files", type=str, default=None,
                       help="Path to val_files.txt")
    
    # Model
    parser.add_argument("--preprocessor", choices=["musetok", "midi2vec"], default="musetok",
                       help="Preprocessor used for latents (affects default input_dim)")
    parser.add_argument("--input_dim", type=int, default=None,
                       help="Input dimension (128 MuseTok, 100 midi2vec; inferred from latents_dir if omitted)")
    parser.add_argument("--bars_per_chunk", type=int, default=-1,
                       help="Bars per chunk: -1=song-level, N>0=N bars per chunk (MuseTok only)")
    parser.add_argument("--num_classes", type=int, default=None,
                       help="Number of classes (11 for emotion, 6 for genre)")
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension (default: input_dim // 2)")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--early_stopping", action="store_true",
                       help="Enable early stopping")
    parser.add_argument("--early_stopping_tolerance", type=int, default=10,
                       help="Early stopping patience")
    
    # Class imbalance
    parser.add_argument("--class_weight", type=str, default="balanced",
                       choices=["none", "balanced"],
                       help="CrossEntropyLoss class weights: 'balanced' (inverse frequency) or 'none'")
    parser.add_argument("--balanced_sampler", action="store_true",
                       help="Use WeightedRandomSampler so each batch sees more minority classes")
    
    # Others
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU (CUDA); if not provided, use CPU")
    parser.add_argument("--num_workers", type=int, default=int(cpu_count() / 4),
                       help="Number of data loader workers")
    parser.add_argument("--output_dir", type=str, default=TRAINED_MODEL_DIR,
                       help="Output directory for checkpoints")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name (default: {task}_classifier)")
    parser.add_argument("--wandb_project", type=str, default="gigamidi-support",
                       help="Wandb project name")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    parser.add_argument("--bootstrap_downsample", type=int, default=0,
                       help="If >0, downsample train set to min class size via bootstrap. 1=one run (seed=0), K=train K models (seeds 0..K-1), save best_model_fold{k}.pt")
    parser.add_argument("--config", type=str, default=None,
                       help="Path to YAML config file (CLI overrides config)")

    # Parse once to get --config; apply config as defaults; parse again so CLI overrides
    args_pre, _ = parser.parse_known_args(args=args, namespace=namespace)
    if getattr(args_pre, "config", None) and os.path.isfile(args_pre.config):
        from utils.config_utils import load_config, apply_config
        apply_config(parser, load_config(args_pre.config))
    args = parser.parse_args(args=args, namespace=namespace)

    missing = [k for k in ("task", "labels_path", "class_to_index_path", "train_files", "valid_files", "num_classes") if getattr(args, k) is None]
    if missing:
        parser.error(f"Missing required (provide via CLI or in --config): {', '.join(missing)}")

    # Infer input_dim from latents_dir if not set
    if args.input_dim is None:
        try:
            args.input_dim = infer_input_dim(args.latents_dir)
        except FileNotFoundError:
            args.input_dim = 100 if args.preprocessor == "midi2vec" else 128
    
    # Set default model name
    if args.model_name is None:
        args.model_name = f"{args.task}_classifier"
    
    # Set default hidden_dim
    if args.hidden_dim is None:
        args.hidden_dim = args.input_dim // 2
    
    # Create output directory
    args.checkpoint_dir = os.path.join(args.output_dir, args.model_name, "checkpoints")
    ensure_dir(args.checkpoint_dir)
    
    return args

# ================================================== #
#  Main Training Loop                               #
# ================================================== #

if __name__ == "__main__":
    args = parse_args()
    
    # Device
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Using device: {device}")
    
    # Load file lists
    def load_file_list(file_path: str) -> list:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    train_files = load_file_list(args.train_files)
    valid_files = load_file_list(args.valid_files)
    
    logging.info(f"Train files: {len(train_files)}, Valid files: {len(valid_files)}")
    # Log per-class counts from file lists (labels) to verify split matches prepare_labels
    labels_dict = load_json(args.labels_path)
    class_to_index = load_json(args.class_to_index_path)
    # Support both int and str values from JSON for index_to_class
    index_to_class = {}
    for k, v in class_to_index.items():
        idx = int(v) if isinstance(v, str) else v
        index_to_class[idx] = k
    num_classes = len(class_to_index)
    train_labels = [labels_dict.get(f) for f in train_files if f in labels_dict]
    valid_labels = [labels_dict.get(f) for f in valid_files if f in labels_dict]
    train_missing = sum(1 for f in train_files if f not in labels_dict)
    valid_missing = sum(1 for f in valid_files if f not in labels_dict)
    if train_missing or valid_missing:
        logging.warning(f"Files in list but not in labels: train={train_missing}, valid={valid_missing}")
    train_counts = np.bincount(train_labels, minlength=num_classes)
    valid_counts = np.bincount(valid_labels, minlength=num_classes)
    logging.info(f"Per-class count in train: {train_counts.tolist()} ({[index_to_class[i] for i in range(num_classes)]})")
    logging.info(f"Per-class count in valid: {valid_counts.tolist()}")
    if (valid_counts == 0).any():
        logging.warning(f"Classes with no samples in valid set: {[index_to_class[i] for i in range(num_classes) if valid_counts[i] == 0]}")
    
    n_folds = args.bootstrap_downsample if args.bootstrap_downsample > 0 else 1
    for fold_k in range(n_folds):
        if args.bootstrap_downsample > 0:
            train_files_use = get_bootstrap_downsampled_file_list(
                train_files, labels_dict, class_to_index, seed=fold_k
            )
            logging.info(f"Bootstrap fold {fold_k}: using {len(train_files_use)} train files (downsampled)")
        else:
            train_files_use = train_files

        train_labels_fold = [labels_dict.get(f) for f in train_files_use if f in labels_dict]
        train_counts_fold = np.bincount(train_labels_fold, minlength=num_classes)
        if args.bootstrap_downsample > 0:
            logging.info(f"Per-class count (fold {fold_k}): {train_counts_fold.tolist()}")

        # Create train dataset first (needed for chunk-level counts when bars_per_chunk > 0)
        train_dataset = XMIDIDataset(
            latents_dir=args.latents_dir,
            labels_path=args.labels_path,
            class_to_index_path=args.class_to_index_path,
            file_list=train_files_use,
            task=args.task,
            bars_per_chunk=args.bars_per_chunk,
        )
        # Chunk-level counts for class weights when using bar-level chunking
        if args.bars_per_chunk > 0:
            train_labels_chunk = [
                labels_dict.get(train_dataset._chunk_index[i][0]) for i in range(len(train_dataset))
            ]
            train_counts_fold = np.bincount(
                [l for l in train_labels_chunk if l is not None], minlength=num_classes
            )
            logging.info(f"Per-class count (chunks): {train_counts_fold.tolist()}")

        # Class weights for imbalanced training (inverse frequency)
        if args.class_weight == "balanced":
            class_weights = np.zeros(num_classes, dtype=np.float32)
            n_samples = len(train_dataset)
            for c in range(num_classes):
                n_c = max(1, train_counts_fold[c])
                class_weights[c] = n_samples / (num_classes * n_c)
            class_weights_tensor = torch.from_numpy(class_weights).float().to(device)
            logging.info(f"Class weights (balanced): {class_weights.tolist()}")
        else:
            class_weights_tensor = None

        valid_dataset = XMIDIDataset(
            latents_dir=args.latents_dir,
            labels_path=args.labels_path,
            class_to_index_path=args.class_to_index_path,
            file_list=valid_files,
            task=args.task,
            bars_per_chunk=args.bars_per_chunk,
        )

        # Train data loader: optional WeightedRandomSampler for balanced batches
        if args.balanced_sampler:
            # Per-sample labels (per-chunk when bars_per_chunk > 0)
            train_label_per_index = [
                labels_dict.get(train_dataset._chunk_index[i][0]) for i in range(len(train_dataset))
            ]
            train_counts_chunk = np.bincount(
                [l for l in train_label_per_index if l is not None],
                minlength=num_classes,
            )
            sample_weights = np.array([
                1.0 / max(1, train_counts_chunk[int(l)]) if l is not None else 1.0
                for l in train_label_per_index
            ], dtype=np.float64)
            sample_weights /= sample_weights.sum()
            sampler = torch.utils.data.WeightedRandomSampler(
                weights=torch.from_numpy(sample_weights),
                num_samples=len(train_dataset),
                replacement=True,
            )
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=sampler,
                num_workers=args.num_workers,
                collate_fn=XMIDIDataset.collate_fn,
            )
            logging.info("Using WeightedRandomSampler for balanced batches")
        else:
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=args.num_workers,
                collate_fn=XMIDIDataset.collate_fn,
            )

        valid_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=XMIDIDataset.collate_fn,
        )

        # Create model (fresh per fold)
        model = EmotionGenreClassifier(
            input_dim=args.input_dim,
            num_classes=args.num_classes,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Model parameters: {n_params:,}")

        loss_fn = nn.CrossEntropyLoss(weight=class_weights_tensor)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )

        checkpoint_suffix = f"_fold{fold_k}" if args.bootstrap_downsample > 1 else ""
        best_model_path = os.path.join(args.checkpoint_dir, f"best_model{checkpoint_suffix}.pt")
        best_optimizer_path = os.path.join(args.checkpoint_dir, f"best_optimizer{checkpoint_suffix}.pt")
        stats_file = os.path.join(args.output_dir, args.model_name, f"statistics{checkpoint_suffix}.csv")

        run_name = f"{args.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if args.bootstrap_downsample > 1:
            run_name = f"{run_name}-fold{fold_k}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )

        log_file = os.path.join(args.output_dir, args.model_name, "train.log")
        ensure_dir(os.path.dirname(log_file))
        logging.getLogger().handlers.clear()
        logging.getLogger().addHandler(logging.FileHandler(log_file, mode="a" if args.resume and fold_k == 0 else "w"))
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

        logging.info(f"Command: python {' '.join(sys.argv)}")
        logging.info(f"Arguments:\n{pprint.pformat(vars(args))}")

        if args.resume and os.path.exists(best_model_path):
            model.load_state_dict(torch.load(best_model_path, map_location=device))
            optimizer.load_state_dict(torch.load(best_optimizer_path, map_location=device))
            logging.info("Resumed from checkpoint")

        best_accuracy = 0.0
        best_metrics = {}
        stats_columns = ["epoch", "split", "loss", "accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
        if not os.path.exists(stats_file) or not args.resume:
            pd.DataFrame(columns=stats_columns).to_csv(stats_file, index=False)

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
                    update_parameters=True,
                    optimizer=optimizer,
                )
                batch_size = len(batch["latents"])
                train_loss += loss * batch_size
                for k in train_metrics:
                    train_metrics[k] += metrics[k] * batch_size
                train_count += batch_size

            train_loss /= train_count
            for k in train_metrics:
                train_metrics[k] /= train_count

            logging.info(f"Train - Loss: {train_loss:.4f}, Accuracy: {train_metrics['accuracy']:.4f}, F1: {train_metrics['f1_macro']:.4f}")

            model.eval()
            valid_loss = 0.0
            valid_metrics = {k: 0.0 for k in train_metrics.keys()}
            valid_count = 0
            all_preds = []
            all_labels = []

            with torch.no_grad():
                for batch in tqdm(valid_loader, desc="Validating"):
                    loss, metrics, (preds, labels) = evaluate_batch(
                        model, batch, loss_fn, device,
                        update_parameters=False,
                        return_predictions=True,
                    )
                    batch_size = len(batch["latents"])
                    valid_loss += loss * batch_size
                    for k in valid_metrics:
                        valid_metrics[k] += metrics[k] * batch_size
                    valid_count += batch_size
                    all_preds.extend(preds.tolist())
                    all_labels.extend(labels.tolist())

            valid_loss /= valid_count
            for k in valid_metrics:
                valid_metrics[k] /= valid_count

            logging.info(f"Valid - Loss: {valid_loss:.4f}, Accuracy: {valid_metrics['accuracy']:.4f}, F1: {valid_metrics['f1_macro']:.4f}")

            for split, loss_val, metrics_dict in [("train", train_loss, train_metrics), ("valid", valid_loss, valid_metrics)]:
                row = {"epoch": epoch + 1, "split": split, "loss": loss_val, **metrics_dict}
                pd.DataFrame([row]).to_csv(stats_file, mode="a", header=False, index=False)

            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "valid/loss": valid_loss,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"valid/{k}": v for k, v in valid_metrics.items()},
            })

            if valid_metrics['accuracy'] > best_accuracy:
                best_accuracy = valid_metrics['accuracy']
                best_metrics = valid_metrics.copy()
                torch.save(model.state_dict(), best_model_path)
                torch.save(optimizer.state_dict(), best_optimizer_path)
                logging.info(f"Saved best model (accuracy: {best_accuracy:.4f})")
                early_stopping_counter = 0
                cm = confusion_matrix(all_labels, all_preds, labels=list(range(args.num_classes)))
                class_names = [index_to_class[i] for i in range(args.num_classes)]
                plt.figure(figsize=(10, 8))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names)
                plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
                plt.ylabel('True Label')
                plt.xlabel('Predicted Label')
                plt.tight_layout()
                cm_path = os.path.join(args.output_dir, args.model_name, f"confusion_matrix_best{checkpoint_suffix}.png")
                plt.savefig(cm_path, dpi=150)
                plt.close()
            else:
                early_stopping_counter += 1

            if args.early_stopping and early_stopping_counter >= args.early_stopping_tolerance:
                logging.info(f"Early stopping after {args.early_stopping_tolerance} epochs without improvement")
                break

        logging.info(f"\nTraining complete (fold {fold_k})!")
        logging.info(f"Best validation accuracy: {best_accuracy:.4f}")
        logging.info(f"Best metrics: {best_metrics}")
        wandb.finish()
