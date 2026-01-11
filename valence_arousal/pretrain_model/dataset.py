"""Dataset class for EMOPIA latents and VA labels."""
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from typing import List, Dict
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import load_latents, load_json

class ValenceArousalDataset(Dataset):
    """Dataset for continuous VA prediction."""
    
    def __init__(self, 
                 latents_dir: str,
                 labels_path: str,
                 file_list: List[str],
                 max_seq_len: int = 42,
                 pool: bool = False):
        """
        Args:
            latents_dir: Directory containing latent files
            labels_path: Path to JSON file with VA labels
            file_list: List of filenames (without extension)
            max_seq_len: Maximum sequence length (bars)
            pool: Whether to pool (average) across bars
        """
        self.latents_dir = latents_dir
        self.labels = load_json(labels_path)
        self.file_list = file_list
        self.max_seq_len = max_seq_len
        self.pool = pool
        
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx):
        filename = self.file_list[idx]
        
        # Load latents
        latents_path = os.path.join(self.latents_dir, f"{filename}.safetensors")
        latents, metadata = load_latents(latents_path)
        latents = torch.from_numpy(latents).float()
        
        # Load labels
        label_data = self.labels[filename]
        if isinstance(label_data["valence"], list):
            # Per-bar labels
            valence = torch.tensor(label_data["valence"], dtype=torch.float32)
            arousal = torch.tensor(label_data["arousal"], dtype=torch.float32)
        else:
            # Song-level labels (repeat for all bars)
            n_bars = len(latents)
            valence = torch.full((n_bars,), label_data["valence"], dtype=torch.float32)
            arousal = torch.full((n_bars,), label_data["arousal"], dtype=torch.float32)
        
        # Handle sequence length
        if len(latents) > self.max_seq_len:
            latents = latents[:self.max_seq_len]
            valence = valence[:self.max_seq_len]
            arousal = arousal[:self.max_seq_len]
        
        # Pool if requested
        if self.pool:
            latents = latents.mean(dim=0)
            valence = valence.mean()
            arousal = arousal.mean()
        
        # Create mask
        mask = torch.ones(len(latents), dtype=torch.bool)
        if len(latents) < self.max_seq_len:
            padding = self.max_seq_len - len(latents)
            latents = torch.nn.functional.pad(latents, (0, 0, 0, padding))
            valence = torch.nn.functional.pad(valence, (0, padding), value=0.0)
            arousal = torch.nn.functional.pad(arousal, (0, padding), value=0.0)
            mask = torch.nn.functional.pad(mask, (0, padding), value=False)
        
        return {
            "latents": latents,
            "valence": valence,
            "arousal": arousal,
            "mask": mask,
            "filename": filename
        }
    
    @staticmethod
    def collate_fn(batch):
        """Collate function for DataLoader."""
        latents = torch.stack([item["latents"] for item in batch])
        valence = torch.stack([item["valence"] for item in batch])
        arousal = torch.stack([item["arousal"] for item in batch])
        mask = torch.stack([item["mask"] for item in batch])
        filenames = [item["filename"] for item in batch]
        
        return {
            "latents": latents,
            "valence": valence,
            "arousal": arousal,
            "mask": mask,
            "filenames": filenames
        }