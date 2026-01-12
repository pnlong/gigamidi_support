"""Data utilities for emotion recognition pipeline."""
import os
from pathlib import Path
from typing import Optional
import json
import pickle
import numpy as np
from safetensors import safe_open
from safetensors.torch import save_file
import torch

# ================================================== #
#  Storage Directory Configuration                  #
# ================================================== #

# Base storage directory - modify this path as needed
# This should point to a location with sufficient disk space
STORAGE_DIR = os.environ.get("VALENCE_AROUSAL_STORAGE_DIR", 
                             "/deepfreeze/pnlong/gigamidi")

# Subdirectories within storage
CHECKPOINTS_DIR = os.path.join(STORAGE_DIR, "checkpoints")
EMOPIA_DATA_DIR = os.path.join(STORAGE_DIR, "emopia")
GIGAMIDI_ANNOTATIONS_DIR = os.path.join(STORAGE_DIR, "gigamidi_annotations")

# EMOPIA dataset paths
# Edited EMOPIA (jingyue's version): REMI-encoded .pkl files
EMOPIA_JINGYUE_DIR = "/deepfreeze/user_shares/jingyue/EMOPIA_data"
# EMOPIA+ (original, full dataset): MIDI files and REMI representations
EMOPIA_PLUS_DIR = os.path.join(EMOPIA_DATA_DIR, "emopia_plus")

# Specific paths
MUSETOK_CHECKPOINT_DIR = os.path.join(CHECKPOINTS_DIR, "musetok")
MUSETOK_TOKENIZER_CHECKPOINT = os.path.join(MUSETOK_CHECKPOINT_DIR, "best_tokenizer.pt")
TRAINED_MODEL_DIR = os.path.join(CHECKPOINTS_DIR, "trained_models")
EMOPIA_LATENTS_DIR = os.path.join(EMOPIA_DATA_DIR, "latents")
EMOPIA_LABELS_DIR = os.path.join(EMOPIA_DATA_DIR, "labels")

def get_storage_dir() -> str:
    """Get the base storage directory."""
    return STORAGE_DIR

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def set_storage_dir(path: str) -> None:
    """Set the storage directory (call before other operations)."""
    global STORAGE_DIR, CHECKPOINTS_DIR, EMOPIA_DATA_DIR, GIGAMIDI_ANNOTATIONS_DIR
    global MUSETOK_CHECKPOINT_DIR, MUSETOK_TOKENIZER_CHECKPOINT
    global TRAINED_MODEL_DIR, EMOPIA_LATENTS_DIR, EMOPIA_LABELS_DIR, EMOPIA_PLUS_DIR
    
    STORAGE_DIR = path
    CHECKPOINTS_DIR = os.path.join(STORAGE_DIR, "checkpoints")
    EMOPIA_DATA_DIR = os.path.join(STORAGE_DIR, "emopia")
    GIGAMIDI_ANNOTATIONS_DIR = os.path.join(STORAGE_DIR, "gigamidi_annotations")
    MUSETOK_CHECKPOINT_DIR = os.path.join(CHECKPOINTS_DIR, "musetok")
    MUSETOK_TOKENIZER_CHECKPOINT = os.path.join(MUSETOK_CHECKPOINT_DIR, "best_tokenizer.pt")
    TRAINED_MODEL_DIR = os.path.join(CHECKPOINTS_DIR, "trained_models")
    EMOPIA_LATENTS_DIR = os.path.join(EMOPIA_DATA_DIR, "latents")
    EMOPIA_LABELS_DIR = os.path.join(EMOPIA_DATA_DIR, "labels")
    EMOPIA_PLUS_DIR = os.path.join(EMOPIA_DATA_DIR, "emopia_plus")

# ================================================== #
#  File I/O Utilities                               #
# ================================================== #

def save_json(filepath: str, data: dict) -> None:
    """Save data to JSON file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)

def load_json(filepath: str) -> dict:
    """Load data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def save_pickle(filepath: str, data: any) -> None:
    """Save data to pickle file."""
    ensure_dir(os.path.dirname(filepath))
    with open(filepath, 'wb') as f:
        pickle.dump(data, f)

def load_pickle(filepath: str) -> any:
    """Load data from pickle file."""
    with open(filepath, 'rb') as f:
        return pickle.load(f)

def save_latents(filepath: str, latents: np.ndarray, metadata: Optional[dict] = None) -> None:
    """Save latents as safetensors file with optional metadata."""
    ensure_dir(os.path.dirname(filepath))
    tensors = {"latents": torch.from_numpy(latents.astype(np.float32))}
    if metadata:
        # Store metadata as JSON string in safetensors metadata
        save_file(tensors, filepath, metadata=metadata)
    else:
        save_file(tensors, filepath)

def load_latents(filepath: str) -> tuple[np.ndarray, Optional[dict]]:
    """Load latents from safetensors file."""
    with safe_open(filepath, framework="pt", device="cpu") as f:
        latents = f.get_tensor("latents").numpy()
        metadata = f.metadata() if f.metadata() else None
    return latents, metadata