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

from pretrain_model.dataset import XMIDIDataset
from pretrain_model.model import EmotionGenreClassifier
from utils.data_utils import (
    TRAINED_MODEL_DIR, XMIDI_LATENTS_DIR, XMIDI_LABELS_DIR,
    ensure_dir, save_json, load_json
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
    parser.add_argument("--task", type=str, required=True,
                       choices=["emotion", "genre"],
                       help="Task: emotion or genre")
    
    # Data paths
    parser.add_argument("--latents_dir", type=str, default=XMIDI_LATENTS_DIR,
                       help="Directory containing XMIDI latents")
    parser.add_argument("--labels_path", type=str, required=True,
                       help="Path to labels JSON file (emotion_labels.json or genre_labels.json)")
    parser.add_argument("--class_to_index_path", type=str, required=True,
                       help="Path to class_to_index JSON file (emotion_to_index.json or genre_to_index.json)")
    parser.add_argument("--train_files", type=str, required=True,
                       help="Path to train_files.txt")
    parser.add_argument("--valid_files", type=str, required=True,
                       help="Path to val_files.txt")
    
    # Model
    parser.add_argument("--input_dim", type=int, default=128,
                       help="Input dimension (d_vae_latent, default: 128)")
    parser.add_argument("--num_classes", type=int, required=True,
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
    
    # Others
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU (CUDA); if not provided, use CPU")
    parser.add_argument("--num_workers", type=int, default=int(cpu_count() / 4),
                       help="Number of data loader workers")
    parser.add_argument("--output_dir", type=str, default=TRAINED_MODEL_DIR,
                       help="Output directory for checkpoints")
    parser.add_argument("--model_name", type=str, default=None,
                       help="Model name (default: {task}_classifier)")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="xmidi_emotion_genre",
                       help="Wandb project name")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    
    args = parser.parse_args(args=args, namespace=namespace)
    
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
    
    # Create datasets
    train_dataset = XMIDIDataset(
        latents_dir=args.latents_dir,
        labels_path=args.labels_path,
        class_to_index_path=args.class_to_index_path,
        file_list=train_files,
        task=args.task,
    )
    valid_dataset = XMIDIDataset(
        latents_dir=args.latents_dir,
        labels_path=args.labels_path,
        class_to_index_path=args.class_to_index_path,
        file_list=valid_files,
        task=args.task,
    )
    
    # Create data loaders
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
    
    # Create model
    model = EmotionGenreClassifier(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {n_params:,}")
    
    # Loss function
    loss_fn = nn.CrossEntropyLoss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Wandb
    if args.use_wandb:
        run_name = f"{args.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
    
    # Setup logging
    log_file = os.path.join(args.output_dir, args.model_name, "train.log")
    ensure_dir(os.path.dirname(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a" if args.resume else "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    logging.info(f"Command: python {' '.join(sys.argv)}")
    logging.info(f"Arguments:\n{pprint.pformat(vars(args))}")
    
    # Resume from checkpoint
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    best_optimizer_path = os.path.join(args.checkpoint_dir, "best_optimizer.pt")
    if args.resume and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        optimizer.load_state_dict(torch.load(best_optimizer_path, map_location=device))
        logging.info("Resumed from checkpoint")
    
    # Training statistics
    best_accuracy = 0.0
    best_metrics = {}
    stats_file = os.path.join(args.output_dir, args.model_name, "statistics.csv")
    stats_columns = ["epoch", "split", "loss", "accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]
    
    if not os.path.exists(stats_file) or not args.resume:
        pd.DataFrame(columns=stats_columns).to_csv(stats_file, index=False)
    
    early_stopping_counter = 0
    
    # Training loop
    for epoch in range(args.epochs):
        logging.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
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
        
        # Validate
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
                
                all_preds.extend(preds)
                all_labels.extend(labels)
        
        valid_loss /= valid_count
        for k in valid_metrics:
            valid_metrics[k] /= valid_count
        
        logging.info(f"Valid - Loss: {valid_loss:.4f}, Accuracy: {valid_metrics['accuracy']:.4f}, F1: {valid_metrics['f1_macro']:.4f}")
        
        # Save statistics
        for split, loss_val, metrics_dict in [("train", train_loss, train_metrics), ("valid", valid_loss, valid_metrics)]:
            row = {
                "epoch": epoch + 1,
                "split": split,
                "loss": loss_val,
                **metrics_dict,
            }
            pd.DataFrame([row]).to_csv(stats_file, mode="a", header=False, index=False)
        
        # Wandb logging
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "valid/loss": valid_loss,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"valid/{k}": v for k, v in valid_metrics.items()},
            })
        
        # Save best model (based on validation accuracy)
        if valid_metrics['accuracy'] > best_accuracy:
            best_accuracy = valid_metrics['accuracy']
            best_metrics = valid_metrics.copy()
            torch.save(model.state_dict(), best_model_path)
            torch.save(optimizer.state_dict(), best_optimizer_path)
            logging.info(f"Saved best model (accuracy: {best_accuracy:.4f})")
            early_stopping_counter = 0
            
            # Save confusion matrix for best model
            cm = confusion_matrix(all_labels, all_preds)
            class_to_index = load_json(args.class_to_index_path)
            class_names = [class_to_index.get(str(i), f"Class_{i}") for i in range(len(class_to_index))]
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=class_names, yticklabels=class_names)
            plt.title(f'Confusion Matrix - Epoch {epoch + 1}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            cm_path = os.path.join(args.output_dir, args.model_name, "confusion_matrix_best.png")
            plt.savefig(cm_path, dpi=150)
            plt.close()
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if args.early_stopping and early_stopping_counter >= args.early_stopping_tolerance:
            logging.info(f"Early stopping after {args.early_stopping_tolerance} epochs without improvement")
            break
    
    logging.info(f"\nTraining complete!")
    logging.info(f"Best validation accuracy: {best_accuracy:.4f}")
    logging.info(f"Best metrics: {best_metrics}")
    
    if args.use_wandb:
        wandb.finish()
