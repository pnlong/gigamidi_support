"""
Evaluation script for emotion/genre classifier trained on combined MuseTok + midi2vec features.

Uses the same combined latents dirs and stats file as training; input_dim is read from the stats.
"""

import argparse
import logging
import sys
import os
from os.path import dirname, realpath
import pandas as pd
import torch
import torch.utils.data
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, dirname(dirname(realpath(__file__))))

from pretrain_model.dataset import CombinedLatentsDataset
from pretrain_model.model import EmotionGenreClassifier
from pretrain_model.train import evaluate_batch
from utils.data_utils import EVALUATION_RESULTS_DIR, ensure_dir, load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate combined (MuseTok+midi2vec) classifier.")
    parser.add_argument("--task", type=str, required=True, choices=["emotion", "genre"])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--latents_dir_musetok", type=str, required=True)
    parser.add_argument("--latents_dir_midi2vec", type=str, required=True)
    parser.add_argument("--stats_path", type=str, required=True,
                        help="Path to combined_latents_stats.npz (from training)")
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--class_to_index_path", type=str, required=True)
    parser.add_argument("--test_files", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True)
    parser.add_argument("--hidden_dim", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()
    data = np.load(args.stats_path)
    args.input_dim = int(data["mean"].shape[0])
    if args.hidden_dim is None:
        args.hidden_dim = args.input_dim // 2
    if args.output_dir is None:
        args.output_dir = os.path.join(EVALUATION_RESULTS_DIR, f"{args.task}_combined")
    return args


if __name__ == "__main__":
    args = parse_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.info(f"Using device: {device}, input_dim (from stats): {args.input_dim}")
    ensure_dir(args.output_dir)

    model = EmotionGenreClassifier(
        input_dim=args.input_dim,
        num_classes=args.num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.eval()

    def load_file_list(path):
        with open(path, "r") as f:
            return [line.strip() for line in f if line.strip()]

    test_files = load_file_list(args.test_files)
    test_dataset = CombinedLatentsDataset(
        latents_dir_musetok=args.latents_dir_musetok,
        latents_dir_midi2vec=args.latents_dir_midi2vec,
        stats_path=args.stats_path,
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
        collate_fn=CombinedLatentsDataset.collate_fn,
    )

    loss_fn = torch.nn.CrossEntropyLoss()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    total_metrics = {k: 0.0 for k in ["accuracy", "f1_macro", "f1_weighted", "precision_macro", "recall_macro"]}
    count = 0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            loss, metrics, (preds, labels) = evaluate_batch(
                model, batch, loss_fn, device, return_predictions=True,
            )
            bs = len(batch["latents"])
            total_loss += loss * bs
            for k in total_metrics:
                total_metrics[k] += metrics[k] * bs
            count += bs
            all_predictions.extend(preds.tolist())
            all_targets.extend(labels.tolist())
    total_loss /= count
    for k in total_metrics:
        total_metrics[k] /= count

    logging.info("\nTest Results (combined features):")
    logging.info(f"Loss: {total_loss:.4f}")
    logging.info(f"Accuracy: {total_metrics['accuracy']:.4f}")
    logging.info(f"F1 (macro): {total_metrics['f1_macro']:.4f}")
    logging.info(f"F1 (weighted): {total_metrics['f1_weighted']:.4f}")

    class_to_index = load_json(args.class_to_index_path)
    index_to_class = {int(v) if isinstance(v, str) else v: k for k, v in class_to_index.items()}
    class_names = [index_to_class[i] for i in range(args.num_classes)]

    cm = confusion_matrix(all_targets, all_predictions, labels=list(range(args.num_classes)))
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Confusion Matrix - {args.task} (combined MuseTok+midi2vec)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    report_text = classification_report(
        all_targets, all_predictions,
        target_names=class_names,
        zero_division=0,
    )
    with open(os.path.join(args.output_dir, "classification_report.txt"), "w") as f:
        f.write(report_text)

    results = {"loss": total_loss, **total_metrics}
    pd.DataFrame([results]).to_csv(os.path.join(args.output_dir, "metrics.csv"), index=False)
    logging.info(f"Saved results to {args.output_dir}")
