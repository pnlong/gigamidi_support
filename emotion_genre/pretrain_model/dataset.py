"""Dataset class for XMIDI latents and labels."""
import os
import sys
import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import load_latents, load_json


def get_bootstrap_downsampled_file_list(
    file_list: List[str],
    labels_dict: Dict[str, int],
    class_to_index: Dict[str, int],
    seed: int = 0,
) -> List[str]:
    """
    Bootstrap resample and downsample all classes to match the smallest class size.

    For each class, sample n_min instances with replacement (bootstrap), where
    n_min = min over classes of (count). Produces a balanced training set of
    size num_classes * n_min. Used to reduce bias from imbalanced ground truth.

    Args:
        file_list: List of filenames (without extension).
        labels_dict: Mapping filename -> label (int or str class index/name).
        class_to_index: Mapping class name -> int index (used to normalize str labels).
        seed: Random seed for reproducible bootstrap.

    Returns:
        Flat list of filenames, length num_classes * n_min.
    """
    num_classes = len(class_to_index)
    # Normalize labels to int indices
    def to_index(label):  # noqa: E306
        if isinstance(label, int) and 0 <= label < num_classes:
            return label
        if isinstance(label, str) and label in class_to_index:
            return int(class_to_index[label])
        return None

    # Group files by class index
    per_class: Dict[int, List[str]] = defaultdict(list)
    for f in file_list:
        lab = labels_dict.get(f)
        idx = to_index(lab) if lab is not None else None
        if idx is not None:
            per_class[idx].append(f)

    for c in range(num_classes):
        if c not in per_class:
            per_class[c] = []
    n_min = min(len(per_class[c]) for c in range(num_classes))
    if n_min == 0:
        raise ValueError(
            "get_bootstrap_downsampled_file_list: at least one class has no samples; "
            "cannot downsample."
        )
    rng = np.random.default_rng(seed)
    out: List[str] = []
    for c in range(num_classes):
        files = per_class[c]
        indices = rng.integers(0, len(files), size=n_min)
        out.extend(files[i] for i in indices)
    return out


def compute_combined_latents_stats(
    latents_dir_musetok: str,
    latents_dir_midi2vec: str,
    file_list: List[str],
    output_path: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load MuseTok + midi2vec latents for each file, concatenate, and compute per-dimension mean and std
    over the training set. Saves to output_path (.npz with keys 'mean', 'std').
    Returns (mean, std, input_dim).
    """
    vectors = []
    n_musetok = 0
    n_midi2vec = 0
    for filename in tqdm(file_list, desc="Computing combined stats"):
        path_m = os.path.join(latents_dir_musetok, f"{filename}.safetensors")
        path_v = os.path.join(latents_dir_midi2vec, f"{filename}.safetensors")
        has_m = os.path.isfile(path_m)
        has_v = os.path.isfile(path_v)
        if has_m:
            n_musetok += 1
        if has_v:
            n_midi2vec += 1
        if not has_m or not has_v:
            continue
        lat_m, _ = load_latents(path_m)
        lat_v, _ = load_latents(path_v)
        # Mean pool if (n_bars, dim)
        if lat_m.ndim > 1:
            lat_m = np.mean(lat_m, axis=0)
        if lat_v.ndim > 1:
            lat_v = np.mean(lat_v, axis=0)
        vec = np.concatenate([lat_m.ravel(), lat_v.ravel()]).astype(np.float32)
        vectors.append(vec)
    if not vectors:
        sample = file_list[:3]
        paths_m = [os.path.join(latents_dir_musetok, f"{f}.safetensors") for f in sample]
        paths_v = [os.path.join(latents_dir_midi2vec, f"{f}.safetensors") for f in sample]
        raise ValueError(
            "No valid (musetok, midi2vec) pairs found for the given file list.\n"
            f"  Files in list: {len(file_list)}. With MuseTok latents: {n_musetok}. With midi2vec latents: {n_midi2vec}.\n"
            f"  Check that both dirs exist and contain .safetensors for the same stems (e.g. from train_files.txt).\n"
            f"  Example stems: {sample}\n"
            f"  MuseTok paths (first 3): {paths_m}\n"
            f"  midi2vec paths (first 3): {paths_v}"
        )
    arr = np.stack(vectors, axis=0)
    mean = np.mean(arr, axis=0)
    std = np.std(arr, axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    np.savez(output_path, mean=mean, std=std)
    return mean, std, int(mean.shape[0])

class XMIDIDataset(Dataset):
    """Dataset for emotion or genre classification."""
    
    def __init__(self, 
                 latents_dir: str,
                 labels_path: str,
                 class_to_index_path: str,
                 file_list: List[str],
                 task: str = "emotion"):  # "emotion" or "genre"
        """
        Args:
            latents_dir: Directory containing latent files
            labels_path: Path to JSON file with labels (emotion or genre)
            class_to_index_path: Path to JSON file mapping class names to indices
            file_list: List of filenames (without extension)
            task: "emotion" or "genre"
        """
        self.latents_dir = latents_dir
        self.labels = load_json(labels_path)
        self.class_to_index = load_json(class_to_index_path)
        self.file_list = file_list
        self.task = task
        
        # Create index_to_class mapping
        self.index_to_class = {v: k for k, v in self.class_to_index.items()}
        self.num_classes = len(self.class_to_index)
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        # Load per-bar latents (stored as n_bars x 128)
        latents_path = os.path.join(self.latents_dir, f"{filename}.safetensors")
        latents, metadata = load_latents(latents_path)
        latents = torch.from_numpy(latents).float()  # Shape: (n_bars, latent_dim)
        
        # Mean pool across bars for song-level prediction
        # Note: We store per-bar latents to allow flexibility in pooling strategies
        # (mean, max, attention, etc.) - can be changed here without re-preprocessing
        latents_pooled = latents.mean(dim=0)  # Shape: (latent_dim,)
        
        # Load label (class index)
        label = self.labels[filename]
        label = torch.tensor(label, dtype=torch.long)
        
        return {
            "latents": latents_pooled,
            "label": label,
            "filename": filename
        }
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader."""
        latents = torch.stack([item["latents"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        filenames = [item["filename"] for item in batch]
        
        return {
            "latents": latents,
            "label": labels,
            "filenames": filenames
        }


class CombinedLatentsDataset(Dataset):
    """Dataset that concatenates MuseTok + midi2vec latents and normalizes per dimension (train-set mean/std)."""

    def __init__(
        self,
        latents_dir_musetok: str,
        latents_dir_midi2vec: str,
        stats_path: str,
        labels_path: str,
        class_to_index_path: str,
        file_list: List[str],
        task: str = "emotion",
    ):
        self.latents_dir_musetok = latents_dir_musetok
        self.latents_dir_midi2vec = latents_dir_midi2vec
        self.labels = load_json(labels_path)
        self.class_to_index = load_json(class_to_index_path)
        self.task = task
        # Only keep files that exist in both latents dirs
        self.file_list = [
            f for f in file_list
            if os.path.isfile(os.path.join(latents_dir_musetok, f"{f}.safetensors"))
            and os.path.isfile(os.path.join(latents_dir_midi2vec, f"{f}.safetensors"))
        ]
        if len(self.file_list) < len(file_list):
            skipped = len(file_list) - len(self.file_list)
            logging.warning(f"CombinedLatentsDataset: {skipped} files missing in one or both latents dirs, using {len(self.file_list)}")
        data = np.load(stats_path)
        self.mean = torch.from_numpy(data["mean"]).float()
        self.std = torch.from_numpy(data["std"]).float()
        self.input_dim = int(self.mean.shape[0])
        self.index_to_class = {int(v) if isinstance(v, str) else v: k for k, v in self.class_to_index.items()}
        self.num_classes = len(self.class_to_index)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        path_m = os.path.join(self.latents_dir_musetok, f"{filename}.safetensors")
        path_v = os.path.join(self.latents_dir_midi2vec, f"{filename}.safetensors")
        lat_m, _ = load_latents(path_m)
        lat_v, _ = load_latents(path_v)
        lat_m = torch.from_numpy(lat_m).float()
        lat_v = torch.from_numpy(lat_v).float()
        if lat_m.dim() > 1:
            lat_m = lat_m.mean(dim=0)
        if lat_v.dim() > 1:
            lat_v = lat_v.mean(dim=0)
        combined = torch.cat([lat_m, lat_v], dim=0)
        combined = (combined - self.mean) / self.std
        label = self.labels[filename]
        label = torch.tensor(label, dtype=torch.long)
        return {
            "latents": combined,
            "label": label,
            "filename": filename,
        }

    @staticmethod
    def collate_fn(batch):
        latents = torch.stack([item["latents"] for item in batch])
        labels = torch.stack([item["label"] for item in batch])
        filenames = [item["filename"] for item in batch]
        return {"latents": latents, "label": labels, "filenames": filenames}


class XMIDIDatasetVA(Dataset):
    """Dataset for valence–arousal regression from XMIDI latents and emotion labels."""

    def __init__(
        self,
        latents_dir: str,
        emotion_labels_path: str,
        file_list: List[str],
        va_mapping: List[tuple] = None,
    ):
        """
        Args:
            latents_dir: Directory containing latent files
            emotion_labels_path: Path to emotion_labels.json (filename -> emotion index 0..10)
            file_list: List of filenames (without extension)
            va_mapping: List of (valence, arousal) per emotion index; default from valence_arousal_mapping
        """
        self.latents_dir = latents_dir
        self.labels = load_json(emotion_labels_path)
        self.file_list = file_list
        if va_mapping is None:
            from pretrain_model.valence_arousal_mapping import EMOTION_INDEX_TO_VALENCE_AROUSAL
            va_mapping = EMOTION_INDEX_TO_VALENCE_AROUSAL
        self.va_mapping = va_mapping

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        filename = self.file_list[idx]
        latents_path = os.path.join(self.latents_dir, f"{filename}.safetensors")
        latents, metadata = load_latents(latents_path)
        latents = torch.from_numpy(latents).float()
        latents_pooled = latents.mean(dim=0)
        emotion_index = self.labels[filename]
        if isinstance(emotion_index, str):
            emotion_index = int(emotion_index)
        valence, arousal = self.va_mapping[emotion_index]
        va = torch.tensor([valence, arousal], dtype=torch.float32)
        return {
            "latents": latents_pooled,
            "va": va,
            "emotion_index": emotion_index,
            "filename": filename,
        }

    @staticmethod
    def collate_fn(batch):
        latents = torch.stack([item["latents"] for item in batch])
        va = torch.stack([item["va"] for item in batch])
        emotion_indices = torch.tensor([item["emotion_index"] for item in batch], dtype=torch.long)
        filenames = [item["filename"] for item in batch]
        return {
            "latents": latents,
            "va": va,
            "emotion_index": emotion_indices,
            "filenames": filenames,
        }
