"""
Evaluation script for VA prediction model.
"""

import argparse
import logging
import sys
import os
from os.path import exists, dirname, realpath
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import seaborn as sns

sys.path.insert(0, dirname(dirname(realpath(__file__))))

from pretrain_model.dataset import ValenceArousalDataset
from pretrain_model.model import ValenceArousalMLP
from pretrain_model.train import evaluate_batch
from utils.data_utils import EMOPIA_LATENTS_DIR, EMOPIA_LABELS_DIR, ensure_dir

def parse_args():
    parser = argparse.ArgumentParser(prog="Evaluate", description="Evaluate VA prediction model.")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--latents_dir", type=str, default=EMOPIA_LATENTS_DIR,
                       help="Directory containing latents")
    parser.add_argument("--labels_path", type=str, default=os.path.join(EMOPIA_LABELS_DIR, "va_labels.json"),
                       help="Path to VA labels")
    parser.add_argument("--test_split", type=str, default="test",
                       help="Test split name")
    parser.add_argument("--input_dim", type=int, default=512,
                       help="Input dimension")
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension")
    parser.add_argument("--use_tanh", action="store_true", default=True,
                       help="Use tanh activation")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--max_seq_len", type=int, default=42,
                       help="Max sequence length")
    parser.add_argument("--pool", action="store_true",
                       help="Pool across bars")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                       help="Output directory")
    
    args = parser.parse_args()
    if args.hidden_dim is None:
        args.hidden_dim = args.input_dim // 2
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    ensure_dir(args.output_dir)
    
    # Load model
    model = ValenceArousalMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        use_tanh=args.use_tanh,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    
    # Load test files
    test_dir = os.path.join(args.latents_dir, args.test_split)
    test_files = [f.replace(".safetensors", "") for f in os.listdir(test_dir) if f.endswith(".safetensors")]
    
    # Create dataset
    test_dataset = ValenceArousalDataset(
        latents_dir=test_dir,
        labels_path=args.labels_path,
        file_list=test_files,
        max_seq_len=args.max_seq_len,
        pool=args.pool,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ValenceArousalDataset.collate_fn,
    )
    
    # Evaluate
    all_predictions = {"valence": [], "arousal": []}
    all_targets = {"valence": [], "arousal": []}
    total_loss = 0.0
    total_metrics = {k: 0.0 for k in ["mae_valence", "mae_arousal", "mse_valence", "mse_arousal", "corr_valence", "corr_arousal"]}
    count = 0
    
    loss_fn = torch.nn.SmoothL1Loss()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            loss, metrics, (pred_v, pred_a, true_v, true_a) = evaluate_batch(
                model, batch, loss_fn, device,
                update_parameters=False,
                return_predictions=True,
            )
            batch_size = len(batch["latents"])
            total_loss += loss * batch_size
            for k in total_metrics:
                total_metrics[k] += metrics[k] * batch_size
            count += batch_size
            
            all_predictions["valence"].extend(pred_v.tolist())
            all_predictions["arousal"].extend(pred_a.tolist())
            all_targets["valence"].extend(true_v.tolist())
            all_targets["arousal"].extend(true_a.tolist())
    
    total_loss /= count
    for k in total_metrics:
        total_metrics[k] /= count
    
    # Print results
    logging.basicConfig(level=logging.INFO)
    logging.info(f"\nTest Results:")
    logging.info(f"Loss: {total_loss:.4f}")
    logging.info(f"MAE Valence: {total_metrics['mae_valence']:.4f}")
    logging.info(f"MAE Arousal: {total_metrics['mae_arousal']:.4f}")
    logging.info(f"MSE Valence: {total_metrics['mse_valence']:.4f}")
    logging.info(f"MSE Arousal: {total_metrics['mse_arousal']:.4f}")
    logging.info(f"Correlation Valence: {total_metrics['corr_valence']:.4f}")
    logging.info(f"Correlation Arousal: {total_metrics['corr_arousal']:.4f}")
    
    # Save results
    results = {
        "loss": total_loss,
        **total_metrics,
    }
    pd.DataFrame([results]).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    
    # Create scatter plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for idx, (emotion, ax) in enumerate(zip(["valence", "arousal"], axes)):
        pred = np.array(all_predictions[emotion])
        true = np.array(all_targets[emotion])
        
        ax.scatter(true, pred, alpha=0.5, s=10)
        ax.plot([-1, 1], [-1, 1], 'r--', label='Perfect prediction')
        ax.set_xlabel(f'True {emotion.capitalize()}')
        ax.set_ylabel(f'Predicted {emotion.capitalize()}')
        ax.set_title(f'{emotion.capitalize()} Prediction')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "scatter_plots.png"), dpi=150)
    plt.close()
    
    logging.info(f"Results saved to {args.output_dir}")