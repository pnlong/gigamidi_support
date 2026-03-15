"""
Train an ensemble of sklearn classifiers (MLP, Logistic Regression, SVM, KNN, Random Forest)
on XMIDI latents for emotion or genre. Each model outputs top-3 predictions; final label
is by majority vote with uncertainty (variance across the ensemble).
"""

import argparse
import csv
import os
import sys
from collections import Counter

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pretrain_model.dataset import get_bootstrap_downsampled_file_list
from utils.data_utils import load_json, load_latents, ensure_dir


def load_file_list(path):
    with open(path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def load_latents_matrix(latents_dir, file_list):
    """Load latents for each file, mean-pool to one vector per file. Returns (X, filenames)."""
    X_list = []
    kept = []
    for f in tqdm(file_list, desc="Loading latents"):
        path = os.path.join(latents_dir, f"{f}.safetensors")
        if not os.path.isfile(path):
            continue
        latents, _ = load_latents(path)
        if latents.ndim > 1:
            latents = np.mean(latents, axis=0)
        X_list.append(latents.ravel().astype(np.float32))
        kept.append(f)
    if not X_list:
        raise FileNotFoundError(f"No latent files found under {latents_dir} for the given file list.")
    return np.stack(X_list), kept


def get_top_k_proba(estimator, X, k=3):
    """Return (top_k_indices, top_k_probs) per row. Indices are class indices."""
    proba = estimator.predict_proba(X)
    # proba can be (n_samples, n_classes); handle binary case (n_classes might be 2)
    top_indices = np.argsort(-proba, axis=1)[:, :k]
    n_samples = X.shape[0]
    top_probs = np.array([proba[i, top_indices[i]] for i in range(n_samples)])
    return top_indices, top_probs


def majority_vote(top1_preds, tie_break="first"):
    """top1_preds: (n_models,) or list of int. Returns (winner_class, vote_frac, uncertainty)."""
    if not top1_preds:
        return None, 0.0, 1.0
    counts = Counter(top1_preds)
    best_class, best_count = counts.most_common(1)[0]
    vote_frac = best_count / len(top1_preds)
    # Uncertainty: fraction of models that did not predict the winner
    uncertainty = 1.0 - vote_frac
    return best_class, vote_frac, uncertainty


def main():
    parser = argparse.ArgumentParser(
        description="Train sklearn ensemble (MLP, LogReg, SVM, KNN, RF) on XMIDI latents with optional bootstrap downsampling."
    )
    parser.add_argument("--task", type=str, required=True, choices=["emotion", "genre"])
    parser.add_argument("--latents_dir", type=str, required=True, help="Directory of .safetensors latents")
    parser.add_argument("--labels_path", type=str, required=True)
    parser.add_argument("--class_to_index_path", type=str, required=True)
    parser.add_argument("--train_files", type=str, required=True)
    parser.add_argument("--valid_files", type=str, default=None)
    parser.add_argument("--test_files", type=str, required=True)
    parser.add_argument("--num_classes", type=int, required=True, help="11 for emotion, 6 for genre")
    parser.add_argument("--bootstrap_downsample_seed", type=int, default=None,
                        help="If set, downsample train set to min class size with this seed")
    parser.add_argument("--n_bootstrap_folds", type=int, default=1,
                        help="Number of bootstrap folds; each fold trains 5 models on a downsampled set (seed=fold)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for saved models and predictions (default: ./ensemble_sklearn)")
    parser.add_argument("--save_predictions", action="store_true", help="Write predictions CSV with uncertainty")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to YAML config file (CLI overrides config)")
    args_pre, _ = parser.parse_known_args()
    if getattr(args_pre, "config", None) and os.path.isfile(args_pre.config):
        from utils.config_utils import load_config, apply_config
        apply_config(parser, load_config(args_pre.config))
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(os.path.dirname(args.latents_dir), "ensemble_sklearn")
    ensure_dir(args.output_dir)

    train_files_full = load_file_list(args.train_files)
    test_files = load_file_list(args.test_files)
    valid_files = load_file_list(args.valid_files) if args.valid_files and os.path.isfile(args.valid_files) else []

    labels_dict = load_json(args.labels_path)
    class_to_index = load_json(args.class_to_index_path)
    index_to_class = {int(v) if isinstance(v, str) else v: k for k, v in class_to_index.items()}

    # Optional single-seed downsampling of train set (for n_bootstrap_folds==1)
    train_files = train_files_full
    if args.bootstrap_downsample_seed is not None and args.n_bootstrap_folds <= 1:
        train_files = get_bootstrap_downsampled_file_list(
            train_files_full, labels_dict, class_to_index, seed=args.bootstrap_downsample_seed
        )
        print(f"Bootstrap downsampled train set (seed={args.bootstrap_downsample_seed}): {len(train_files)} files")

    # Load data (single train set when n_bootstrap_folds==1, else we load per fold)
    X_train, train_kept = load_latents_matrix(args.latents_dir, train_files)
    y_train = np.array([labels_dict[f] for f in train_kept], dtype=np.int64)
    X_test, test_kept = load_latents_matrix(args.latents_dir, test_files)
    y_test = np.array([labels_dict[f] for f in test_kept], dtype=np.int64)
    if valid_files:
        X_val, val_kept = load_latents_matrix(args.latents_dir, valid_files)
        y_val = np.array([labels_dict[f] for f in val_kept], dtype=np.int64)
    else:
        X_val, y_val, val_kept = None, None, []

    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}" + (f", Valid: {X_val.shape[0]}" if X_val is not None else ""))

    # Model factory
    def make_models(n_train):
        k = max(1, min(15, n_train // 2))
        return [
            ("mlp", MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)),
            ("logreg", LogisticRegression(max_iter=500, random_state=42)),
            ("svm", SVC(probability=True, kernel="rbf", random_state=42)),
            ("knn", KNeighborsClassifier(n_neighbors=k)),
            ("rf", RandomForestClassifier(n_estimators=100, random_state=42)),
        ]

    all_estimators = []  # list of (name_fold, estimator)
    n_folds = max(1, args.n_bootstrap_folds)

    for fold in range(n_folds):
        if n_folds > 1:
            train_files_fold = get_bootstrap_downsampled_file_list(
                train_files_full, labels_dict, class_to_index, seed=fold
            )
            X_train_fold, train_kept_fold = load_latents_matrix(args.latents_dir, train_files_fold)
            y_train_fold = np.array([labels_dict[f] for f in train_kept_fold], dtype=np.int64)
        else:
            X_train_fold, y_train_fold = X_train, y_train

        for name, est in make_models(len(y_train_fold)):
            est.fit(X_train_fold, y_train_fold)
            suffix = f"{name}_fold{fold}" if n_folds > 1 else name
            all_estimators.append((suffix, est))
            path = os.path.join(args.output_dir, f"model_{suffix}.joblib")
            joblib.dump(est, path)

    # Predict: each model gives top-3; we use top-1 for majority vote and uncertainty
    n_models = len(all_estimators)
    n_test = X_test.shape[0]
    top1_all = np.zeros((n_test, n_models), dtype=np.int64)

    for i, (suffix, est) in enumerate(all_estimators):
        top_indices, _ = get_top_k_proba(est, X_test, k=3)
        top1_all[:, i] = top_indices[:, 0]

    pred_labels = np.zeros(n_test, dtype=np.int64)
    uncertainties = np.zeros(n_test, dtype=np.float64)
    for i in range(n_test):
        winner, vote_frac, unc = majority_vote(top1_all[i].tolist())
        pred_labels[i] = winner
        uncertainties[i] = unc

    # Metrics
    acc = accuracy_score(y_test, pred_labels)
    f1_macro = f1_score(y_test, pred_labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, pred_labels, average="weighted", zero_division=0)
    print(f"\nTest accuracy: {acc:.4f}, F1 macro: {f1_macro:.4f}, F1 weighted: {f1_weighted:.4f}")
    print(classification_report(y_test, pred_labels, target_names=[index_to_class[i] for i in range(args.num_classes)]))
    cm = confusion_matrix(y_test, pred_labels, labels=list(range(args.num_classes)))
    print("Confusion matrix:\n", cm)

    # Save metrics
    metrics_path = os.path.join(args.output_dir, "metrics.txt")
    with open(metrics_path, "w") as f:
        f.write(f"accuracy,{acc}\n")
        f.write(f"f1_macro,{f1_macro}\n")
        f.write(f"f1_weighted,{f1_weighted}\n")
    print(f"Metrics written to {metrics_path}")

    if args.save_predictions:
        out_path = os.path.join(args.output_dir, "predictions.csv")
        rows = []
        for i in range(n_test):
            rows.append({
                "filename": test_kept[i],
                "true_label": index_to_class.get(int(y_test[i]), str(y_test[i])),
                "true_index": int(y_test[i]),
                "pred_label": index_to_class.get(int(pred_labels[i]), str(pred_labels[i])),
                "pred_index": int(pred_labels[i]),
                "uncertainty": float(uncertainties[i]),
            })
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)
        print(f"Predictions written to {out_path}")


if __name__ == "__main__":
    main()
