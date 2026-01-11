"""
Training script for continuous valence/arousal prediction model.
Adapted from jingyue_latents/train.py for regression task.
"""

import argparse
import logging
import pprint
import sys
import os
from os.path import exists, dirname, realpath, basename
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
from scipy.stats import pearsonr
warnings.simplefilter(action="ignore", category=FutureWarning)

# Add parent directory to path
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from pretrain_model.dataset import ValenceArousalDataset
from pretrain_model.model import ValenceArousalMLP
from utils.data_utils import (
    TRAINED_MODEL_DIR, EMOPIA_LATENTS_DIR, EMOPIA_LABELS_DIR,
    ensure_dir, save_json
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
        metrics: dict with 'mae_valence', 'mae_arousal', 'mse_valence', 'mse_arousal', 'corr_valence', 'corr_arousal'
        predictions: (optional) tuple of (pred_valence, pred_arousal, true_valence, true_arousal)
    """
    latents = batch["latents"].to(device)
    valence_true = batch["valence"].to(device)
    arousal_true = batch["arousal"].to(device)
    mask = batch["mask"].to(device)
    
    # Zero gradients
    if update_parameters:
        optimizer.zero_grad()
    
    # Forward pass
    outputs = model(latents, mask=mask)  # (batch_size, seq_len, 2) or (batch_size, 2)
    
    # Handle sequence-level vs pooled outputs
    if len(outputs.shape) == 3:
        # Sequence-level: outputs is (batch_size, seq_len, 2)
        pred_valence = outputs[:, :, 0]  # (batch_size, seq_len)
        pred_arousal = outputs[:, :, 1]  # (batch_size, seq_len)
        
        # Apply mask and compute loss
        mask_expanded = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        pred_valence_masked = pred_valence * mask_expanded.squeeze(-1)
        pred_arousal_masked = pred_arousal * mask_expanded.squeeze(-1)
        valence_true_masked = valence_true * mask
        arousal_true_masked = arousal_true * mask
        
        # Compute loss (average over valid positions)
        loss_valence = loss_fn(pred_valence_masked, valence_true_masked)
        loss_arousal = loss_fn(pred_arousal_masked, arousal_true_masked)
        loss = (loss_valence + loss_arousal) / 2
    else:
        # Pooled: outputs is (batch_size, 2)
        pred_valence = outputs[:, 0]
        pred_arousal = outputs[:, 1]
        loss_valence = loss_fn(pred_valence, valence_true)
        loss_arousal = loss_fn(pred_arousal, arousal_true)
        loss = (loss_valence + loss_arousal) / 2
    
    # Backward pass
    if update_parameters:
        loss.backward()
        optimizer.step()
    
    loss_value = float(loss)
    
    # Compute metrics
    with torch.no_grad():
        if len(outputs.shape) == 3:
            # Flatten for metrics (only valid positions)
            valid_mask = mask.bool()
            pred_valence_flat = pred_valence[valid_mask].cpu().numpy()
            pred_arousal_flat = pred_arousal[valid_mask].cpu().numpy()
            valence_true_flat = valence_true[valid_mask].cpu().numpy()
            arousal_true_flat = arousal_true[valid_mask].cpu().numpy()
        else:
            pred_valence_flat = pred_valence.cpu().numpy()
            pred_arousal_flat = pred_arousal.cpu().numpy()
            valence_true_flat = valence_true.cpu().numpy()
            arousal_true_flat = arousal_true.cpu().numpy()
        
        mae_valence = np.mean(np.abs(pred_valence_flat - valence_true_flat))
        mae_arousal = np.mean(np.abs(pred_arousal_flat - arousal_true_flat))
        mse_valence = np.mean((pred_valence_flat - valence_true_flat) ** 2)
        mse_arousal = np.mean((pred_arousal_flat - arousal_true_flat) ** 2)
        
        # Correlation
        if len(pred_valence_flat) > 1:
            corr_valence, _ = pearsonr(pred_valence_flat, valence_true_flat)
            corr_arousal, _ = pearsonr(pred_arousal_flat, arousal_true_flat)
        else:
            corr_valence = corr_arousal = 0.0
    
    metrics = {
        "mae_valence": mae_valence,
        "mae_arousal": mae_arousal,
        "mse_valence": mse_valence,
        "mse_arousal": mse_arousal,
        "corr_valence": corr_valence,
        "corr_arousal": corr_arousal,
    }
    
    # Clean up
    del latents, valence_true, arousal_true, mask, outputs, pred_valence, pred_arousal
    
    if return_predictions:
        return loss_value, metrics, (pred_valence_flat, pred_arousal_flat, valence_true_flat, arousal_true_flat)
    else:
        return loss_value, metrics

# ================================================== #
#  Argument Parsing                                 #
# ================================================== #

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="Train", description="Train VA prediction model.")
    
    # Data paths
    parser.add_argument("--latents_dir", type=str, default=EMOPIA_LATENTS_DIR,
                       help="Directory containing EMOPIA latents")
    parser.add_argument("--labels_path", type=str, default=os.path.join(EMOPIA_LABELS_DIR, "va_labels.json"),
                       help="Path to VA labels JSON file")
    parser.add_argument("--train_split", type=str, default="train",
                       help="Train split name")
    parser.add_argument("--valid_split", type=str, default="valid",
                       help="Validation split name")
    
    # Model
    parser.add_argument("--input_dim", type=int, default=512,
                       help="Input dimension (d_vae_latent)")
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension (default: input_dim // 2)")
    parser.add_argument("--use_tanh", action="store_true", default=True,
                       help="Use tanh to constrain outputs to [-1, 1]")
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
    parser.add_argument("--max_seq_len", type=int, default=42,
                       help="Maximum sequence length (bars)")
    parser.add_argument("--pool", action="store_true",
                       help="Pool (average) across bars before model")
    parser.add_argument("--loss_type", type=str, default="smooth_l1",
                       choices=["mse", "smooth_l1", "l1"],
                       help="Loss function type")
    parser.add_argument("--early_stopping", action="store_true",
                       help="Enable early stopping")
    parser.add_argument("--early_stopping_tolerance", type=int, default=10,
                       help="Early stopping patience")
    
    # Others
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=int(cpu_count() / 4),
                       help="Number of data loader workers")
    parser.add_argument("--output_dir", type=str, default=TRAINED_MODEL_DIR,
                       help="Output directory for checkpoints")
    parser.add_argument("--model_name", type=str, default="va_mlp",
                       help="Model name")
    parser.add_argument("--wandb_project", type=str, default="valence_arousal",
                       help="Wandb project name")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    
    args = parser.parse_args(args=args, namespace=namespace)
    
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
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load file lists (assuming they exist in latents_dir/{split}/)
    def get_file_list(split):
        split_dir = os.path.join(args.latents_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        files = [f.replace(".safetensors", "") for f in os.listdir(split_dir) if f.endswith(".safetensors")]
        return files
    
    train_files = get_file_list(args.train_split)
    valid_files = get_file_list(args.valid_split)
    
    logging.info(f"Train files: {len(train_files)}, Valid files: {len(valid_files)}")
    
    # Create datasets
    train_dataset = ValenceArousalDataset(
        latents_dir=os.path.join(args.latents_dir, args.train_split),
        labels_path=args.labels_path,
        file_list=train_files,
        max_seq_len=args.max_seq_len,
        pool=args.pool,
    )
    valid_dataset = ValenceArousalDataset(
        latents_dir=os.path.join(args.latents_dir, args.valid_split),
        labels_path=args.labels_path,
        file_list=valid_files,
        max_seq_len=args.max_seq_len,
        pool=args.pool,
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ValenceArousalDataset.collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ValenceArousalDataset.collate_fn,
    )
    
    # Create model
    model = ValenceArousalMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        use_tanh=args.use_tanh,
        dropout=args.dropout,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {n_params:,}")
    
    # Loss function
    if args.loss_type == "mse":
        loss_fn = nn.MSELoss()
    elif args.loss_type == "smooth_l1":
        loss_fn = nn.SmoothL1Loss()
    elif args.loss_type == "l1":
        loss_fn = nn.L1Loss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Wandb (always enabled)
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
    best_loss = float("inf")
    best_metrics = {}
    stats_file = os.path.join(args.output_dir, args.model_name, "statistics.csv")
    stats_columns = ["epoch", "split", "loss", "mae_valence", "mae_arousal", "mse_valence", "mse_arousal", "corr_valence", "corr_arousal"]
    
    if not os.path.exists(stats_file) or not args.resume:
        pd.DataFrame(columns=stats_columns).to_csv(stats_file, index=False)
    
    early_stopping_counter = 0
    
    # Training loop
    for epoch in range(args.epochs):
        logging.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        model.train()
        train_loss = 0.0
        train_metrics = {k: 0.0 for k in ["mae_valence", "mae_arousal", "mse_valence", "mse_arousal", "corr_valence", "corr_arousal"]}
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
        
        logging.info(f"Train - Loss: {train_loss:.4f}, MAE V: {train_metrics['mae_valence']:.4f}, MAE A: {train_metrics['mae_arousal']:.4f}")
        
        # Validate
        model.eval()
        valid_loss = 0.0
        valid_metrics = {k: 0.0 for k in train_metrics.keys()}
        valid_count = 0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                loss, metrics = evaluate_batch(
                    model, batch, loss_fn, device,
                    update_parameters=False,
                )
                batch_size = len(batch["latents"])
                valid_loss += loss * batch_size
                for k in valid_metrics:
                    valid_metrics[k] += metrics[k] * batch_size
                valid_count += batch_size
        
        valid_loss /= valid_count
        for k in valid_metrics:
            valid_metrics[k] /= valid_count
        
        logging.info(f"Valid - Loss: {valid_loss:.4f}, MAE V: {valid_metrics['mae_valence']:.4f}, MAE A: {valid_metrics['mae_arousal']:.4f}")
        
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
        wandb.log({
            "epoch": epoch + 1,
            "train/loss": train_loss,
            "valid/loss": valid_loss,
            **{f"train/{k}": v for k, v in train_metrics.items()},
            **{f"valid/{k}": v for k, v in valid_metrics.items()},
        })
        
        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_metrics = valid_metrics.copy()
            torch.save(model.state_dict(), best_model_path)
            torch.save(optimizer.state_dict(), best_optimizer_path)
            logging.info(f"Saved best model (loss: {best_loss:.4f})")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if args.early_stopping and early_stopping_counter >= args.early_stopping_tolerance:
            logging.info(f"Early stopping after {args.early_stopping_tolerance} epochs without improvement")
            break
    
    logging.info(f"\nTraining complete!")
    logging.info(f"Best validation loss: {best_loss:.4f}")
    logging.info(f"Best metrics: {best_metrics}")
    
    wandb.finish()