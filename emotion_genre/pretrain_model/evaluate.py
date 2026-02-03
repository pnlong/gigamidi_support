"""
Evaluation script for emotion/genre classification model.
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
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, 
    confusion_matrix, classification_report
)

sys.path.insert(0, dirname(dirname(realpath(__file__))))

from pretrain_model.dataset import XMIDIDataset
from pretrain_model.model import EmotionGenreClassifier
from pretrain_model.train import evaluate_batch
from utils.data_utils import EVALUATION_RESULTS_DIR, ensure_dir, load_json

def parse_args():
    parser = argparse.ArgumentParser(prog="Evaluate", description="Evaluate emotion/genre classification model.")
    
    # Task
    parser.add_argument("--task", type=str, required=True,
                       choices=["emotion", "genre"],
                       help="Task: emotion or genre")
    
    parser.add_argument("--checkpoint_path", type=str, required=True,
                       help="Path to model checkpoint")
    parser.add_argument("--latents_dir", type=str, required=True,
                       help="Directory containing latents")
    parser.add_argument("--labels_path", type=str, required=True,
                       help="Path to labels JSON file")
    parser.add_argument("--class_to_index_path", type=str, required=True,
                       help="Path to class_to_index JSON file")
    parser.add_argument("--test_files", type=str, required=True,
                       help="Path to test_files.txt")
    parser.add_argument("--preprocessor", choices=["musetok", "midi2vec"], default="musetok",
                       help="Preprocessor used for latents (affects default input_dim)")
    parser.add_argument("--input_dim", type=int, default=None,
                       help="Input dimension (128 for MuseTok, 100 for midi2vec)")
    parser.add_argument("--num_classes", type=int, required=True,
                       help="Number of classes")
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--gpu", action="store_true",
                       help="Use GPU (CUDA); if not provided, use CPU")
    parser.add_argument("--num_workers", type=int, default=4,
                       help="Number of workers")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: EVALUATION_RESULTS_DIR/{task})")
    
    args = parser.parse_args()
    if args.input_dim is None:
        args.input_dim = 100 if args.preprocessor == "midi2vec" else 128
    if args.hidden_dim is None:
        args.hidden_dim = args.input_dim // 2
    if args.output_dir is None:
        args.output_dir = os.path.join(EVALUATION_RESULTS_DIR, args.task)
    return args

if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Using device: {device}")
    
    ensure_dir(args.output_dir)
    
    # Load model
    model = EmotionGenreClassifier(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()
    logging.info(f"Loaded model from {args.checkpoint_path}")
    
    # Load file list
    def load_file_list(file_path: str) -> list:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f if line.strip()]
    
    test_files = load_file_list(args.test_files)
    logging.info(f"Test files: {len(test_files)}")
    
    # Create dataset
    test_dataset = XMIDIDataset(
        latents_dir=args.latents_dir,
        labels_path=args.labels_path,
        class_to_index_path=args.class_to_index_path,
        file_list=test_files,
        task=args.task,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=XMIDIDataset.collate_fn,
    )
    
    # Evaluate
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    total_metrics = {k: 0.0 for k in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]}
    count = 0
    
    loss_fn = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            loss, metrics, (preds, labels) = evaluate_batch(
                model, batch, loss_fn, device,
                update_parameters=False,
                return_predictions=True,
            )
            batch_size = len(batch["latents"])
            total_loss += loss * batch_size
            for k in total_metrics:
                total_metrics[k] += metrics[k] * batch_size
            count += batch_size
            
            all_predictions.extend(preds.tolist())
            all_targets.extend(labels.tolist())
    
    total_loss /= count
    for k in total_metrics:
        total_metrics[k] /= count
    
    # Print results
    logging.info(f"\nTest Results:")
    logging.info(f"Loss: {total_loss:.4f}")
    logging.info(f"Accuracy: {total_metrics['accuracy']:.4f}")
    logging.info(f"F1-score (macro): {total_metrics['f1_macro']:.4f}")
    logging.info(f"F1-score (weighted): {total_metrics['f1_weighted']:.4f}")
    logging.info(f"Precision (macro): {total_metrics['precision_macro']:.4f}")
    logging.info(f"Recall (macro): {total_metrics['recall_macro']:.4f}")
    
    # Per-class metrics
    class_to_index = load_json(args.class_to_index_path)
    index_to_class = {v: k for k, v in class_to_index.items()}
    class_names = [index_to_class.get(i, f"Class_{i}") for i in range(args.num_classes)]
    
    # Per-class F1, precision, recall
    f1_per_class = f1_score(all_targets, all_predictions, average=None, zero_division=0)
    precision_per_class = precision_score(all_targets, all_predictions, average=None, zero_division=0)
    recall_per_class = recall_score(all_targets, all_predictions, average=None, zero_division=0)
    
    logging.info(f"\nPer-class metrics:")
    for i, class_name in enumerate(class_names):
        logging.info(f"  {class_name}: F1={f1_per_class[i]:.4f}, Precision={precision_per_class[i]:.4f}, Recall={recall_per_class[i]:.4f}")
    
    # Save results
    results = {
        "loss": total_loss,
        **total_metrics,
    }
    # Add per-class metrics
    for i, class_name in enumerate(class_names):
        results[f"f1_{class_name}"] = f1_per_class[i]
        results[f"precision_{class_name}"] = precision_per_class[i]
        results[f"recall_{class_name}"] = recall_per_class[i]
    
    pd.DataFrame([results]).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    logging.info(f"Saved metrics to {os.path.join(args.output_dir, 'metrics.csv')}")
    
    # Confusion matrix
    cm = confusion_matrix(all_targets, all_predictions)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
               xticklabels=class_names, yticklabels=class_names)
    plt.title(f'Confusion Matrix - {args.task.capitalize()} Classification')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()
    logging.info(f"Saved confusion matrix to {os.path.join(args.output_dir, 'confusion_matrix.png')}")
    
    # Classification report
    report = classification_report(
        all_targets, all_predictions,
        target_names=class_names,
        output_dict=True
    )
    
    # Save classification report as text
    report_text = classification_report(
        all_targets, all_predictions,
        target_names=class_names
    )
    with open(os.path.join(args.output_dir, "classification_report.txt"), 'w') as f:
        f.write(report_text)
    logging.info(f"Saved classification report to {os.path.join(args.output_dir, 'classification_report.txt')}")
    
    # Save detailed report as JSON
    pd.DataFrame(report).transpose().to_csv(os.path.join(args.output_dir, "classification_report.csv"))
    logging.info(f"Saved detailed classification report to {os.path.join(args.output_dir, 'classification_report.csv')}")
    
    logging.info(f"\nAll results saved to {args.output_dir}")
