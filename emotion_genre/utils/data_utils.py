"""Data utilities for XMIDI emotion and genre recognition pipeline."""
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
STORAGE_DIR = os.environ.get("XMIDI_STORAGE_DIR", 
                             "/deepfreeze/pnlong/gigamidi")

# Task-specific base directory
XMIDI_EMOTION_GENRE_DIR = os.path.join(STORAGE_DIR, "xmidi_emotion_genre")

# Subdirectories within task-specific directory
CHECKPOINTS_DIR = os.path.join(XMIDI_EMOTION_GENRE_DIR, "checkpoints")
XMIDI_DATA_DIR = os.path.join(XMIDI_EMOTION_GENRE_DIR, "xmidi_data")
XMIDI_LATENTS_DIR = os.path.join(XMIDI_DATA_DIR, "latents")
XMIDI_LABELS_DIR = os.path.join(XMIDI_DATA_DIR, "labels")
TRAINED_MODEL_DIR = os.path.join(CHECKPOINTS_DIR, "trained_models")
EVALUATION_RESULTS_DIR = os.path.join(XMIDI_EMOTION_GENRE_DIR, "evaluation_results")

# MuseTok checkpoint (shared across tasks, at storage root)
MUSETOK_CHECKPOINT_DIR = os.path.join(STORAGE_DIR, "musetok")
MUSETOK_TOKENIZER_CHECKPOINT = os.path.join(MUSETOK_CHECKPOINT_DIR, "best_tokenizer.pt")

# midi2vec precomputed embeddings (embeddings.bin, names.csv)
MIDI2VEC_EMBEDDINGS_DIR = os.environ.get(
    "MIDI2VEC_EMBEDDINGS_DIR",
    os.path.join(STORAGE_DIR, "midi2vec")
)

def get_storage_dir() -> str:
    """Get the base storage directory."""
    return STORAGE_DIR

def ensure_dir(path: str) -> None:
    """Create directory if it doesn't exist."""
    Path(path).mkdir(parents=True, exist_ok=True)

def set_storage_dir(path: str) -> None:
    """Set the storage directory (call before other operations)."""
    global STORAGE_DIR, XMIDI_EMOTION_GENRE_DIR
    global CHECKPOINTS_DIR, XMIDI_DATA_DIR, XMIDI_LATENTS_DIR, XMIDI_LABELS_DIR
    global TRAINED_MODEL_DIR, EVALUATION_RESULTS_DIR, MIDI2VEC_EMBEDDINGS_DIR
    
    STORAGE_DIR = path
    XMIDI_EMOTION_GENRE_DIR = os.path.join(STORAGE_DIR, "xmidi_emotion_genre")
    CHECKPOINTS_DIR = os.path.join(XMIDI_EMOTION_GENRE_DIR, "checkpoints")
    XMIDI_DATA_DIR = os.path.join(XMIDI_EMOTION_GENRE_DIR, "xmidi_data")
    XMIDI_LATENTS_DIR = os.path.join(XMIDI_DATA_DIR, "latents")
    XMIDI_LABELS_DIR = os.path.join(XMIDI_DATA_DIR, "labels")
    TRAINED_MODEL_DIR = os.path.join(CHECKPOINTS_DIR, "trained_models")
    EVALUATION_RESULTS_DIR = os.path.join(XMIDI_EMOTION_GENRE_DIR, "evaluation_results")
    MIDI2VEC_EMBEDDINGS_DIR = os.path.join(STORAGE_DIR, "midi2vec")

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
    """Save latents as safetensors file with optional metadata.
    
    Note: Safetensors metadata requires all values to be strings.
    Complex values (lists, dicts) are JSON-encoded.
    """
    ensure_dir(os.path.dirname(filepath))
    tensors = {"latents": torch.from_numpy(latents.astype(np.float32))}
    if metadata:
        # Convert metadata to string format (safetensors requires all values to be strings)
        metadata_str = {}
        for key, value in metadata.items():
            if isinstance(value, (list, dict)):
                # JSON-encode complex types
                metadata_str[key] = json.dumps(value)
            elif isinstance(value, (int, float)):
                # Convert numbers to strings
                metadata_str[key] = str(value)
            else:
                # Already a string or can be converted
                metadata_str[key] = str(value)
        save_file(tensors, filepath, metadata=metadata_str)
    else:
        save_file(tensors, filepath)

def load_latents(filepath: str) -> tuple[np.ndarray, Optional[dict]]:
    """Load latents from safetensors file.
    
    Note: Metadata values are decoded from JSON strings back to their original types.
    """
    with safe_open(filepath, framework="pt", device="cpu") as f:
        latents = f.get_tensor("latents").numpy()
        metadata_raw = f.metadata() if f.metadata() else None
        
        # Decode metadata from strings back to original types
        if metadata_raw:
            metadata = {}
            for key, value in metadata_raw.items():
                # Try to decode JSON strings, otherwise keep as string
                try:
                    metadata[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    # Not JSON, try to convert to int/float if possible
                    try:
                        if '.' in value:
                            metadata[key] = float(value)
                        else:
                            metadata[key] = int(value)
                    except ValueError:
                        metadata[key] = value
        else:
            metadata = None
            
    return latents, metadata
