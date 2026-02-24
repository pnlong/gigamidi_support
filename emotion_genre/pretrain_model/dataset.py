"""Dataset class for XMIDI latents and labels."""
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from typing import List, Dict
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import load_latents, load_json

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
