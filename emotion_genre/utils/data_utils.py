"""Data utilities for XMIDI emotion and genre recognition pipeline."""
import csv
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

# midi2vec batched pipeline output (batch_0/, batch_1/, ..., batch_assignments.csv)
MIDI2VEC_BATCHES_DIR = os.environ.get(
    "MIDI2VEC_BATCHES_DIR",
    os.path.join(STORAGE_DIR, "midi2vec", "batches")
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
    global TRAINED_MODEL_DIR, EVALUATION_RESULTS_DIR, MIDI2VEC_EMBEDDINGS_DIR, MIDI2VEC_BATCHES_DIR

    STORAGE_DIR = path
    XMIDI_EMOTION_GENRE_DIR = os.path.join(STORAGE_DIR, "xmidi_emotion_genre")
    CHECKPOINTS_DIR = os.path.join(XMIDI_EMOTION_GENRE_DIR, "checkpoints")
    XMIDI_DATA_DIR = os.path.join(XMIDI_EMOTION_GENRE_DIR, "xmidi_data")
    XMIDI_LATENTS_DIR = os.path.join(XMIDI_DATA_DIR, "latents")
    XMIDI_LABELS_DIR = os.path.join(XMIDI_DATA_DIR, "labels")
    TRAINED_MODEL_DIR = os.path.join(CHECKPOINTS_DIR, "trained_models")
    EVALUATION_RESULTS_DIR = os.path.join(XMIDI_EMOTION_GENRE_DIR, "evaluation_results")
    MIDI2VEC_EMBEDDINGS_DIR = os.path.join(STORAGE_DIR, "midi2vec")
    MIDI2VEC_BATCHES_DIR = os.path.join(STORAGE_DIR, "midi2vec", "batches")

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

# Cache file (CSV) in latents_dir: filename (no extension) -> n_bars.
# Used by bar-level chunking to avoid reading every .safetensors file when building the chunk index.
BARS_PER_SONG_FILENAME = "bars_per_song.csv"


def get_bars_per_song_index(latents_dir: str) -> Optional[dict]:
    """Load cached mapping of filename (no extension) -> n_bars from latents_dir/bars_per_song.csv.
    Returns None if the cache file does not exist.
    """
    path = os.path.join(latents_dir, BARS_PER_SONG_FILENAME)
    if not os.path.isfile(path):
        return None
    out = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            name = row.get("filename", "").strip()
            if name:
                try:
                    out[name] = int(row.get("n_bars", 0))
                except (ValueError, TypeError):
                    pass
    return out


def build_bars_per_song_index(latents_dir: str, force: bool = False) -> dict:
    """Build and write bars_per_song.csv in latents_dir: for each .safetensors file,
    store (filename without extension) -> n_bars (shape[0], or 1 if 1D).
    If the cache file already exists and force is False, only load and return it.
    Otherwise scan the directory, load each file to get shape, write cache, return mapping.
    """
    path = os.path.join(latents_dir, BARS_PER_SONG_FILENAME)
    if not force and os.path.isfile(path):
        return get_bars_per_song_index(latents_dir) or {}
    from tqdm import tqdm
    names = [n for n in os.listdir(latents_dir) if n.endswith(".safetensors")]
    out = {}
    for name in tqdm(names, desc="Building bars_per_song cache", unit="file"):
        fpath = os.path.join(latents_dir, name)
        try:
            latents, _ = load_latents(fpath)
            n_bars = 1 if latents.ndim < 2 else int(latents.shape[0])
            out[name.replace(".safetensors", "")] = n_bars
        except Exception:
            continue
    ensure_dir(latents_dir)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["filename", "n_bars"])
        w.writerows([(k, v) for k, v in sorted(out.items())])
    return out


def ensure_bars_per_song_index(latents_dir: str) -> dict:
    """Return the bars-per-song mapping for latents_dir. If the cache CSV is missing,
    build it and write it to disk, then return the mapping. Next run will use the cache.
    """
    idx = get_bars_per_song_index(latents_dir)
    if idx is not None:
        return idx
    return build_bars_per_song_index(latents_dir, force=True)


def infer_input_dim(latents_dir: str) -> int:
    """Infer feature dimension from the first .safetensors file in latents_dir.
    Returns shape[-1] so it works for both 1D (song-level) and 2D (n_bars, dim) arrays.
    """
    for name in os.listdir(latents_dir):
        if name.endswith(".safetensors"):
            path = os.path.join(latents_dir, name)
            latents, _ = load_latents(path)
            return int(latents.shape[-1])
    raise FileNotFoundError(f"No .safetensors files found in {latents_dir}")


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
