# Implementation Plan: Continuous Valence/Arousal Prediction Pipeline

This document provides step-by-step instructions for implementing the emotion recognition pipeline. Follow the steps in order, as later steps depend on earlier ones.

---

## Phase 1: Project Setup and Utilities

### Step 1.1: Create Project Structure

**Action**: Create the directory structure and initial files.

```bash
cd /home/pnlong/gigamidi
mkdir -p valence_arousal/{config,pretrain_model,analyze_emotion_annotations,utils}
touch valence_arousal/{README.md,requirements.txt,annotate_gigamidi.py}
touch valence_arousal/utils/{__init__.py,musetok_utils.py,midi_utils.py,data_utils.py}
touch valence_arousal/pretrain_model/{__init__.py,preprocess_emopia.py,prepare_labels.py,dataset.py,model.py,train.py,evaluate.py}
touch valence_arousal/analyze_emotion_annotations/{__init__.py,plot_histograms.py,plot_boxplots.py,plot_by_genre.py,plot_song_curves.py,print_statistics.py}
touch valence_arousal/config/{musetok_config.yaml,training_config.yaml}
```

### Step 1.2: Create `utils/data_utils.py` - Storage Directory Configuration

**File**: `valence_arousal/utils/data_utils.py`

**Purpose**: Define storage directory paths and utility functions for data I/O.

**Code to write**:

```python
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
                             "/path/to/storage/valence_arousal")

# Subdirectories within storage
CHECKPOINTS_DIR = os.path.join(STORAGE_DIR, "checkpoints")
EMOPIA_DATA_DIR = os.path.join(STORAGE_DIR, "emopia")
GIGAMIDI_ANNOTATIONS_DIR = os.path.join(STORAGE_DIR, "gigamidi_annotations")

# Specific paths
MUSETOK_CHECKPOINT_DIR = os.path.join(CHECKPOINTS_DIR, "musetok")
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
    global MUSETOK_CHECKPOINT_DIR, TRAINED_MODEL_DIR, EMOPIA_LATENTS_DIR, EMOPIA_LABELS_DIR
    
    STORAGE_DIR = path
    CHECKPOINTS_DIR = os.path.join(STORAGE_DIR, "checkpoints")
    EMOPIA_DATA_DIR = os.path.join(STORAGE_DIR, "emopia")
    GIGAMIDI_ANNOTATIONS_DIR = os.path.join(STORAGE_DIR, "gigamidi_annotations")
    MUSETOK_CHECKPOINT_DIR = os.path.join(CHECKPOINTS_DIR, "musetok")
    TRAINED_MODEL_DIR = os.path.join(CHECKPOINTS_DIR, "trained_models")
    EMOPIA_LATENTS_DIR = os.path.join(EMOPIA_DATA_DIR, "latents")
    EMOPIA_LABELS_DIR = os.path.join(EMOPIA_DATA_DIR, "labels")

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
```

**Test**: Create a simple test script to verify storage directory setup works.

### Step 1.3: Create `requirements.txt`

**File**: `valence_arousal/requirements.txt`

**Content**:
```
symusic>=0.1.0
torch>=2.0.0
datasets>=2.14.0
gdown>=4.7.0
safetensors>=0.4.0
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
tqdm>=4.66.0
wandb>=0.15.0
matplotlib>=3.7.0
seaborn>=0.12.0
pyarrow>=12.0.0
```

### Step 1.4: Create `utils/midi_utils.py` - Symusic MIDI Processing

**File**: `valence_arousal/utils/midi_utils.py`

**Purpose**: Convert MIDI files to REMI events using symusic (replacing miditoolkit).

**Code structure**:

```python
"""MIDI processing utilities using symusic."""
import numpy as np
from symusic import Score
from typing import Union, List, Dict, Tuple, Optional

# Constants from musetok/data_processing/midi2events.py
BEAT_RESOL = 480
TICK_RESOL = BEAT_RESOL // 12  # 40
TRIPLET_RESOL = BEAT_RESOL // 24  # 20
DEFAULT_TEMPO = 110
DEFAULT_VELOCITY_BINS = np.linspace(4, 127, 42, dtype=int)
DEFAULT_BPM_BINS = np.linspace(32, 224, 64 + 1, dtype=int)
DEFAULT_SHIFT_BINS = np.linspace(-TICK_RESOL, TICK_RESOL, TICK_RESOL + 1, dtype=int)
DEFAULT_TIME_SIGNATURE = ['4/4', '2/4', '3/4', '2/2', '3/8', '6/8']

def load_midi_symusic(midi_path_or_bytes: Union[str, bytes]) -> Score:
    """Load MIDI file using symusic."""
    if isinstance(midi_path_or_bytes, bytes):
        return Score.from_midi(midi_path_or_bytes)
    else:
        return Score(midi_path_or_bytes)

def get_time_signature(score: Score) -> Tuple[int, int]:
    """Extract time signature from score."""
    # Implementation: extract from score.time_signatures
    # Default to 4/4 if not found
    pass

def get_tempo(score: Score) -> float:
    """Extract tempo (BPM) from score."""
    # Implementation: extract from score.tempos
    # Return median tempo, default to DEFAULT_TEMPO
    pass

def quantize_note_timing(note_start: int, bar_resol: int) -> int:
    """Quantize note start time to grid."""
    # Implementation: similar to midi2events.py quantization logic
    pass

def midi_to_events_symusic(score: Score) -> Tuple[List[int], List[Dict[str, any]]]:
    """
    Convert symusic Score to REMI events.
    
    Returns:
        bar_positions: List of event indices where bars start
        events: List of event dictionaries with 'name' and 'value' keys
    """
    # Implementation:
    # 1. Extract notes, tempo, time signature from score
    # 2. Quantize notes to grid
    # 3. Convert to REMI event format
    # 4. Return bar_positions and events
    pass

def get_bar_positions(events: List[Dict[str, any]]) -> List[int]:
    """Extract bar boundary positions from events."""
    return [i for i, event in enumerate(events) if event['name'] == 'Bar']
```

**Implementation Notes**:
- Study `musetok/data_processing/midi2events.py` carefully
- Replicate the quantization and event creation logic
- Test with sample MIDI files to ensure output matches expected REMI format

### Step 1.5: Create `utils/musetok_utils.py` - MuseTok Integration

**File**: `valence_arousal/utils/musetok_utils.py`

**Purpose**: Interface to MuseTok for extracting latents.

**Code structure**:

```python
"""MuseTok integration utilities."""
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import sys
import os

# Add musetok to path (adjust based on your setup)
MUSETOK_PATH = os.path.join(os.path.dirname(__file__), "../../musetok")
sys.path.insert(0, MUSETOK_PATH)

from musetok.encoding import MuseTokEncoder
from utils.midi_utils import midi_to_events_symusic, load_midi_symusic
from utils.data_utils import MUSETOK_CHECKPOINT_DIR

def load_musetok_model(checkpoint_path: Optional[str] = None, 
                      vocab_path: Optional[str] = None,
                      device: str = 'cuda') -> Tuple[MuseTokEncoder, dict]:
    """
    Load pre-trained MuseTok model.
    
    Args:
        checkpoint_path: Path to model checkpoint (defaults to MUSETOK_CHECKPOINT_DIR)
        vocab_path: Path to vocabulary file (defaults to musetok/data/dictionary.pkl)
        device: Device to load model on
    
    Returns:
        model: Loaded MuseTokEncoder
        vocab: Vocabulary dictionary
    """
    # Implementation:
    # 1. Set default paths if not provided
    # 2. Load vocabulary
    # 3. Initialize and load MuseTokEncoder
    # 4. Move to device
    pass

def events_to_remi_tokens(events: List[Dict], vocab: dict) -> np.ndarray:
    """Convert REMI events to token sequence."""
    # Implementation: map events to vocabulary indices
    pass

def extract_latents_from_events(events: List[Dict], 
                                bar_positions: List[int],
                                model: MuseTokEncoder,
                                vocab: dict,
                                device: str = 'cuda') -> np.ndarray:
    """
    Extract MuseTok latents from REMI events.
    
    Returns:
        latents: numpy array of shape (n_bars, d_vae_latent)
    """
    # Implementation:
    # 1. Convert events to tokens
    # 2. Prepare input for MuseTok (handle segments if >16 bars)
    # 3. Call model.get_batch_latent() or model.get_sampled_latent()
    # 4. Reshape to (n_bars, d_vae_latent)
    # 5. Convert to numpy
    pass

def extract_latents_from_midi(midi_path_or_bytes: Union[str, bytes],
                              model: MuseTokEncoder,
                              vocab: dict,
                              device: str = 'cuda') -> Tuple[np.ndarray, List[int]]:
    """
    Full pipeline: MIDI -> REMI events -> latents.
    
    Returns:
        latents: numpy array of shape (n_bars, d_vae_latent)
        bar_positions: List of bar boundary positions
    """
    # Implementation:
    # 1. Load MIDI with symusic
    # 2. Convert to REMI events
    # 3. Extract latents
    # 4. Return latents and bar_positions
    pass
```

**Implementation Notes**:
- Study `musetok/encoding.py` to understand MuseTokEncoder API
- Handle songs with >16 bars by processing in segments
- Test with sample MIDI files

### Step 1.6: Download MuseTok Checkpoints

**Action**: Download and extract MuseTok checkpoints.

```bash
cd valence_arousal
python -c "
import gdown
import os
from utils.data_utils import MUSETOK_CHECKPOINT_DIR, ensure_dir

ensure_dir(MUSETOK_CHECKPOINT_DIR)
checkpoint_id = '1HK534lEVdHYl3HMRkKvz8CWYliXRmOq_'
output_path = os.path.join(MUSETOK_CHECKPOINT_DIR, 'musetok_checkpoint.zip')
gdown.download(f'https://drive.google.com/uc?id={checkpoint_id}', output_path, quiet=False)
"
```

Then extract the zip file and verify structure.

---

## Phase 2: EMOPIA Preprocessing

### Step 2.1: Create `pretrain_model/preprocess_emopia.py`

**File**: `valence_arousal/pretrain_model/preprocess_emopia.py`

**Purpose**: Extract MuseTok latents from EMOPIA MIDI files.

**Note**: Includes `--resume` argument to skip files that have already been processed (by checking for existing `.safetensors` output files).

**Code structure**:

```python
"""Preprocess EMOPIA dataset: extract MuseTok latents."""
import os
import argparse
from pathlib import Path
from tqdm import tqdm
from multiprocessing import Pool
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import (
    EMOPIA_LATENTS_DIR, ensure_dir, save_latents, 
    MUSETOK_CHECKPOINT_DIR
)
from utils.musetok_utils import load_musetok_model, extract_latents_from_midi
from utils.midi_utils import load_midi_symusic

def process_single_file(args_tuple):
    """Process a single MIDI file (for multiprocessing)."""
    midi_path, output_dir, model, vocab, device = args_tuple
    try:
        # Load MIDI
        score = load_midi_symusic(midi_path)
        
        # Extract latents
        latents, bar_positions = extract_latents_from_midi(
            midi_path, model, vocab, device
        )
        
        # Save latents
        filename = Path(midi_path).stem
        output_path = os.path.join(output_dir, f"{filename}.safetensors")
        metadata = {
            "n_bars": len(latents),
            "original_midi_path": str(midi_path),
            "bar_positions": bar_positions
        }
        save_latents(output_path, latents, metadata)
        
        return (filename, True, None)
    except Exception as e:
        return (Path(midi_path).stem, False, str(e))

def get_processed_files(output_dir: str):
    """Get set of already-processed files by checking for existing safetensors files."""
    processed = set()
    if os.path.exists(output_dir):
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                if file.endswith('.safetensors'):
                    # Extract filename without extension
                    filename = Path(file).stem
                    processed.add(filename)
    return processed

def preprocess_emopia(emopia_dir: str,
                     output_dir: str,
                     checkpoint_path: str,
                     vocab_path: str,
                     device: str = 'cuda',
                     num_workers: int = 4,
                     resume: bool = False):
    """
    Preprocess EMOPIA dataset.
    
    Args:
        emopia_dir: Directory containing EMOPIA MIDI files
        output_dir: Output directory for latents
        checkpoint_path: Path to MuseTok checkpoint
        vocab_path: Path to vocabulary file
        device: Device to use
        num_workers: Number of parallel workers
        resume: If True, skip files that have already been processed
    """
    # Implementation:
    # 1. Load MuseTok model (once, before multiprocessing)
    # 2. Find all MIDI files in emopia_dir (recursively)
    # 3. If resume=True:
    #    - Call get_processed_files() to get set of already-processed filenames
    #    - Filter out MIDI files whose output already exists
    #    - Log how many files are being skipped
    # 4. Create output directories for train/valid/test splits (if needed)
    # 5. Prepare arguments for multiprocessing (each worker needs model, vocab, device)
    #    Note: Model loading in multiprocessing may need special handling (e.g., load per worker)
    # 6. Process files in parallel using Pool
    # 7. Collect results and log statistics (successful, failed, skipped)
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emopia_dir", required=True, help="EMOPIA dataset directory")
    parser.add_argument("--output_dir", default=EMOPIA_LATENTS_DIR)
    parser.add_argument("--checkpoint_path", default=None)
    parser.add_argument("--vocab_path", default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Resume preprocessing: skip files that have already been processed")
    args = parser.parse_args()
    
    preprocess_emopia(
        args.emopia_dir, args.output_dir, args.checkpoint_path,
        args.vocab_path, args.device, args.num_workers, args.resume
    )
```

**Test**: Run on a small subset of EMOPIA files first.

### Step 2.2: Create `pretrain_model/prepare_labels.py`

**File**: `valence_arousal/pretrain_model/prepare_labels.py`

**Purpose**: Convert EMOPIA categorical labels to continuous VA values.

**Code structure**:

```python
"""Prepare continuous valence/arousal labels from EMOPIA."""
import os
import json
import argparse
from pathlib import Path
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import EMOPIA_LABELS_DIR, ensure_dir, save_json

# Emotion to VA mapping (Russell's circumplex model)
EMOTION_TO_VA = {
    "happy": {"valence": 0.8, "arousal": 0.6},
    "angry": {"valence": -0.6, "arousal": 0.8},
    "sad": {"valence": -0.8, "arousal": -0.4},
    "relax": {"valence": 0.4, "arousal": -0.6},
}

def load_emopia_labels(emopia_dir: str) -> dict:
    """Load EMOPIA emotion labels."""
    # Implementation: load EMOPIA metadata/labels
    # Format: {filename: emotion_label}
    pass

def prepare_labels(emopia_dir: str, output_path: str, per_bar: bool = False):
    """
    Prepare continuous VA labels from EMOPIA.
    
    Args:
        emopia_dir: EMOPIA dataset directory
        output_path: Output JSON file path
        per_bar: Whether to create per-bar labels (if available)
    """
    # Implementation:
    # 1. Load EMOPIA labels
    # 2. Map emotions to VA values
    # 3. If per_bar=False, use same VA for all bars
    # 4. Save to JSON
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--emopia_dir", required=True)
    parser.add_argument("--output_path", default=os.path.join(EMOPIA_LABELS_DIR, "va_labels.json"))
    parser.add_argument("--per_bar", action="store_true")
    args = parser.parse_args()
    
    prepare_labels(args.emopia_dir, args.output_path, args.per_bar)
```

---

## Phase 3: Model Training

### Step 3.1: Create `pretrain_model/dataset.py`

**File**: `valence_arousal/pretrain_model/dataset.py`

**Purpose**: PyTorch dataset for loading EMOPIA latents and VA labels.

**Code structure**:

```python
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
```

### Step 3.2: Create `pretrain_model/model.py`

**File**: `valence_arousal/pretrain_model/model.py`

**Purpose**: MLP model for continuous VA prediction.

**Code structure**:

```python
"""MLP model for continuous valence/arousal prediction."""
import torch
import torch.nn as nn

class ValenceArousalMLP(nn.Module):
    """MLP for predicting continuous valence and arousal."""
    
    def __init__(self, 
                 input_dim: int,
                 hidden_dim: int = None,
                 use_tanh: bool = True,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of input latents (d_vae_latent)
            hidden_dim: Hidden layer dimension (default: input_dim // 2)
            use_tanh: Whether to use tanh activation to constrain outputs to [-1, 1]
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.use_tanh = use_tanh
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2)  # 2 outputs: valence, arousal
        )
        
        if use_tanh:
            self.tanh = nn.Tanh()
    
    def forward(self, latents, mask=None):
        """
        Args:
            latents: (batch_size, seq_len, input_dim) or (batch_size, input_dim) if pooled
            mask: (batch_size, seq_len) mask for valid positions
        
        Returns:
            predictions: (batch_size, seq_len, 2) or (batch_size, 2) if pooled
        """
        if len(latents.shape) == 3:
            # Sequence-level: average over valid positions
            if mask is not None:
                mask_expanded = mask.unsqueeze(-1).float()
                latents = (latents * mask_expanded).sum(dim=1) / mask.sum(dim=1, keepdim=True).float()
            else:
                latents = latents.mean(dim=1)
        
        output = self.mlp(latents)
        
        if self.use_tanh:
            output = self.tanh(output)
        
        return output
```

### Step 3.3: Create `pretrain_model/train.py`

**File**: `valence_arousal/pretrain_model/train.py`

**Purpose**: Training script for VA prediction model.

**Key components**:
- DataLoader setup
- Model initialization
- Loss function (SmoothL1Loss or MSELoss)
- Optimizer (AdamW)
- Training loop with validation
- Metrics: MAE, MSE, correlation
- Checkpoint saving
- Wandb logging (optional)

**Complete code**:

```python
"""
Training script for continuous valence/arousal prediction model.
Adapted from jingyue_latents/train.py for regression task.
"""

import argparse
import logging
import pprint
import sys
import os
from os.path import exists, dirname, realpath, basename
from multiprocessing import cpu_count
import wandb
from datetime import datetime
import pandas as pd
import torch
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm
import warnings
import numpy as np
from scipy.stats import pearsonr
warnings.simplefilter(action="ignore", category=FutureWarning)

# Add parent directory to path
sys.path.insert(0, dirname(dirname(realpath(__file__))))

from pretrain_model.dataset import ValenceArousalDataset
from pretrain_model.model import ValenceArousalMLP
from utils.data_utils import (
    TRAINED_MODEL_DIR, EMOPIA_LATENTS_DIR, EMOPIA_LABELS_DIR,
    ensure_dir, save_json
)

# ================================================== #
#  Batch Evaluation Function                        #
# ================================================== #

def evaluate_batch(
    model: nn.Module,
    batch: dict,
    loss_fn: nn.Module,
    device: torch.device,
    update_parameters: bool = False,
    optimizer: torch.optim.Optimizer = None,
    return_predictions: bool = False,
):
    """
    Evaluate model on a batch, updating parameters if specified.
    
    Returns:
        loss: float
        metrics: dict with 'mae_valence', 'mae_arousal', 'mse_valence', 'mse_arousal', 'corr_valence', 'corr_arousal'
        predictions: (optional) tuple of (pred_valence, pred_arousal, true_valence, true_arousal)
    """
    latents = batch["latents"].to(device)
    valence_true = batch["valence"].to(device)
    arousal_true = batch["arousal"].to(device)
    mask = batch["mask"].to(device)
    
    # Zero gradients
    if update_parameters:
        optimizer.zero_grad()
    
    # Forward pass
    outputs = model(latents, mask=mask)  # (batch_size, seq_len, 2) or (batch_size, 2)
    
    # Handle sequence-level vs pooled outputs
    if len(outputs.shape) == 3:
        # Sequence-level: outputs is (batch_size, seq_len, 2)
        pred_valence = outputs[:, :, 0]  # (batch_size, seq_len)
        pred_arousal = outputs[:, :, 1]  # (batch_size, seq_len)
        
        # Apply mask and compute loss
        mask_expanded = mask.unsqueeze(-1).float()  # (batch_size, seq_len, 1)
        pred_valence_masked = pred_valence * mask_expanded.squeeze(-1)
        pred_arousal_masked = pred_arousal * mask_expanded.squeeze(-1)
        valence_true_masked = valence_true * mask
        arousal_true_masked = arousal_true * mask
        
        # Compute loss (average over valid positions)
        loss_valence = loss_fn(pred_valence_masked, valence_true_masked)
        loss_arousal = loss_fn(pred_arousal_masked, arousal_true_masked)
        loss = (loss_valence + loss_arousal) / 2
    else:
        # Pooled: outputs is (batch_size, 2)
        pred_valence = outputs[:, 0]
        pred_arousal = outputs[:, 1]
        loss_valence = loss_fn(pred_valence, valence_true)
        loss_arousal = loss_fn(pred_arousal, arousal_true)
        loss = (loss_valence + loss_arousal) / 2
    
    # Backward pass
    if update_parameters:
        loss.backward()
        optimizer.step()
    
    loss_value = float(loss)
    
    # Compute metrics
    with torch.no_grad():
        if len(outputs.shape) == 3:
            # Flatten for metrics (only valid positions)
            valid_mask = mask.bool()
            pred_valence_flat = pred_valence[valid_mask].cpu().numpy()
            pred_arousal_flat = pred_arousal[valid_mask].cpu().numpy()
            valence_true_flat = valence_true[valid_mask].cpu().numpy()
            arousal_true_flat = arousal_true[valid_mask].cpu().numpy()
        else:
            pred_valence_flat = pred_valence.cpu().numpy()
            pred_arousal_flat = pred_arousal.cpu().numpy()
            valence_true_flat = valence_true.cpu().numpy()
            arousal_true_flat = arousal_true.cpu().numpy()
        
        mae_valence = np.mean(np.abs(pred_valence_flat - valence_true_flat))
        mae_arousal = np.mean(np.abs(pred_arousal_flat - arousal_true_flat))
        mse_valence = np.mean((pred_valence_flat - valence_true_flat) ** 2)
        mse_arousal = np.mean((pred_arousal_flat - arousal_true_flat) ** 2)
        
        # Correlation
        if len(pred_valence_flat) > 1:
            corr_valence, _ = pearsonr(pred_valence_flat, valence_true_flat)
            corr_arousal, _ = pearsonr(pred_arousal_flat, arousal_true_flat)
        else:
            corr_valence = corr_arousal = 0.0
    
    metrics = {
        "mae_valence": mae_valence,
        "mae_arousal": mae_arousal,
        "mse_valence": mse_valence,
        "mse_arousal": mse_arousal,
        "corr_valence": corr_valence,
        "corr_arousal": corr_arousal,
    }
    
    # Clean up
    del latents, valence_true, arousal_true, mask, outputs, pred_valence, pred_arousal
    
    if return_predictions:
        return loss_value, metrics, (pred_valence_flat, pred_arousal_flat, valence_true_flat, arousal_true_flat)
    else:
        return loss_value, metrics

# ================================================== #
#  Argument Parsing                                 #
# ================================================== #

def parse_args(args=None, namespace=None):
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(prog="Train", description="Train VA prediction model.")
    
    # Data paths
    parser.add_argument("--latents_dir", type=str, default=EMOPIA_LATENTS_DIR,
                       help="Directory containing EMOPIA latents")
    parser.add_argument("--labels_path", type=str, default=os.path.join(EMOPIA_LABELS_DIR, "va_labels.json"),
                       help="Path to VA labels JSON file")
    parser.add_argument("--train_split", type=str, default="train",
                       help="Train split name")
    parser.add_argument("--valid_split", type=str, default="valid",
                       help="Validation split name")
    
    # Model
    parser.add_argument("--input_dim", type=int, default=512,
                       help="Input dimension (d_vae_latent)")
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension (default: input_dim // 2)")
    parser.add_argument("--use_tanh", action="store_true", default=True,
                       help="Use tanh to constrain outputs to [-1, 1]")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    
    # Training
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5,
                       help="Weight decay")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of epochs")
    parser.add_argument("--max_seq_len", type=int, default=42,
                       help="Maximum sequence length (bars)")
    parser.add_argument("--pool", action="store_true",
                       help="Pool (average) across bars before model")
    parser.add_argument("--loss_type", type=str, default="smooth_l1",
                       choices=["mse", "smooth_l1", "l1"],
                       help="Loss function type")
    parser.add_argument("--early_stopping", action="store_true",
                       help="Enable early stopping")
    parser.add_argument("--early_stopping_tolerance", type=int, default=10,
                       help="Early stopping patience")
    
    # Others
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device (cuda/cpu)")
    parser.add_argument("--num_workers", type=int, default=int(cpu_count() / 4),
                       help="Number of data loader workers")
    parser.add_argument("--output_dir", type=str, default=TRAINED_MODEL_DIR,
                       help="Output directory for checkpoints")
    parser.add_argument("--model_name", type=str, default="va_mlp",
                       help="Model name")
    parser.add_argument("--use_wandb", action="store_true",
                       help="Use wandb for logging")
    parser.add_argument("--wandb_project", type=str, default="valence_arousal",
                       help="Wandb project name")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from checkpoint")
    
    args = parser.parse_args(args=args, namespace=namespace)
    
    # Set default hidden_dim
    if args.hidden_dim is None:
        args.hidden_dim = args.input_dim // 2
    
    # Create output directory
    args.checkpoint_dir = os.path.join(args.output_dir, args.model_name, "checkpoints")
    ensure_dir(args.checkpoint_dir)
    
    return args

# ================================================== #
#  Main Training Loop                               #
# ================================================== #

if __name__ == "__main__":
    args = parse_args()
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    logging.info(f"Using device: {device}")
    
    # Load file lists (assuming they exist in latents_dir/{split}/)
    def get_file_list(split):
        split_dir = os.path.join(args.latents_dir, split)
        if not os.path.exists(split_dir):
            raise ValueError(f"Split directory not found: {split_dir}")
        files = [f.replace(".safetensors", "") for f in os.listdir(split_dir) if f.endswith(".safetensors")]
        return files
    
    train_files = get_file_list(args.train_split)
    valid_files = get_file_list(args.valid_split)
    
    logging.info(f"Train files: {len(train_files)}, Valid files: {len(valid_files)}")
    
    # Create datasets
    train_dataset = ValenceArousalDataset(
        latents_dir=os.path.join(args.latents_dir, args.train_split),
        labels_path=args.labels_path,
        file_list=train_files,
        max_seq_len=args.max_seq_len,
        pool=args.pool,
    )
    valid_dataset = ValenceArousalDataset(
        latents_dir=os.path.join(args.latents_dir, args.valid_split),
        labels_path=args.labels_path,
        file_list=valid_files,
        max_seq_len=args.max_seq_len,
        pool=args.pool,
    )
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=ValenceArousalDataset.collate_fn,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=ValenceArousalDataset.collate_fn,
    )
    
    # Create model
    model = ValenceArousalMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        use_tanh=args.use_tanh,
        dropout=args.dropout,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    logging.info(f"Model parameters: {n_params:,}")
    
    # Loss function
    if args.loss_type == "mse":
        loss_fn = nn.MSELoss()
    elif args.loss_type == "smooth_l1":
        loss_fn = nn.SmoothL1Loss()
    elif args.loss_type == "l1":
        loss_fn = nn.L1Loss()
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    
    # Wandb
    if args.use_wandb:
        run_name = f"{args.model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
        )
    
    # Setup logging
    log_file = os.path.join(args.output_dir, args.model_name, "train.log")
    ensure_dir(os.path.dirname(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_file, mode="a" if args.resume else "w"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    
    logging.info(f"Command: python {' '.join(sys.argv)}")
    logging.info(f"Arguments:\n{pprint.pformat(vars(args))}")
    
    # Resume from checkpoint
    best_model_path = os.path.join(args.checkpoint_dir, "best_model.pt")
    best_optimizer_path = os.path.join(args.checkpoint_dir, "best_optimizer.pt")
    if args.resume and os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        optimizer.load_state_dict(torch.load(best_optimizer_path, map_location=device))
        logging.info("Resumed from checkpoint")
    
    # Training statistics
    best_loss = float("inf")
    best_metrics = {}
    stats_file = os.path.join(args.output_dir, args.model_name, "statistics.csv")
    stats_columns = ["epoch", "split", "loss", "mae_valence", "mae_arousal", "mse_valence", "mse_arousal", "corr_valence", "corr_arousal"]
    
    if not os.path.exists(stats_file) or not args.resume:
        pd.DataFrame(columns=stats_columns).to_csv(stats_file, index=False)
    
    early_stopping_counter = 0
    
    # Training loop
    for epoch in range(args.epochs):
        logging.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        model.train()
        train_loss = 0.0
        train_metrics = {k: 0.0 for k in ["mae_valence", "mae_arousal", "mse_valence", "mse_arousal", "corr_valence", "corr_arousal"]}
        train_count = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            loss, metrics = evaluate_batch(
                model, batch, loss_fn, device,
                update_parameters=True,
                optimizer=optimizer,
            )
            batch_size = len(batch["latents"])
            train_loss += loss * batch_size
            for k in train_metrics:
                train_metrics[k] += metrics[k] * batch_size
            train_count += batch_size
        
        train_loss /= train_count
        for k in train_metrics:
            train_metrics[k] /= train_count
        
        logging.info(f"Train - Loss: {train_loss:.4f}, MAE V: {train_metrics['mae_valence']:.4f}, MAE A: {train_metrics['mae_arousal']:.4f}")
        
        # Validate
        model.eval()
        valid_loss = 0.0
        valid_metrics = {k: 0.0 for k in train_metrics.keys()}
        valid_count = 0
        
        with torch.no_grad():
            for batch in tqdm(valid_loader, desc="Validating"):
                loss, metrics = evaluate_batch(
                    model, batch, loss_fn, device,
                    update_parameters=False,
                )
                batch_size = len(batch["latents"])
                valid_loss += loss * batch_size
                for k in valid_metrics:
                    valid_metrics[k] += metrics[k] * batch_size
                valid_count += batch_size
        
        valid_loss /= valid_count
        for k in valid_metrics:
            valid_metrics[k] /= valid_count
        
        logging.info(f"Valid - Loss: {valid_loss:.4f}, MAE V: {valid_metrics['mae_valence']:.4f}, MAE A: {valid_metrics['mae_arousal']:.4f}")
        
        # Save statistics
        for split, loss_val, metrics_dict in [("train", train_loss, train_metrics), ("valid", valid_loss, valid_metrics)]:
            row = {
                "epoch": epoch + 1,
                "split": split,
                "loss": loss_val,
                **metrics_dict,
            }
            pd.DataFrame([row]).to_csv(stats_file, mode="a", header=False, index=False)
        
        # Wandb logging
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_loss,
                "valid/loss": valid_loss,
                **{f"train/{k}": v for k, v in train_metrics.items()},
                **{f"valid/{k}": v for k, v in valid_metrics.items()},
            })
        
        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            best_metrics = valid_metrics.copy()
            torch.save(model.state_dict(), best_model_path)
            torch.save(optimizer.state_dict(), best_optimizer_path)
            logging.info(f"Saved best model (loss: {best_loss:.4f})")
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
        
        # Early stopping
        if args.early_stopping and early_stopping_counter >= args.early_stopping_tolerance:
            logging.info(f"Early stopping after {args.early_stopping_tolerance} epochs without improvement")
            break
    
    logging.info(f"\nTraining complete!")
    logging.info(f"Best validation loss: {best_loss:.4f}")
    logging.info(f"Best metrics: {best_metrics}")
    
    if args.use_wandb:
        wandb.finish()
```

### Step 3.4: Create `pretrain_model/evaluate.py`

**File**: `valence_arousal/pretrain_model/evaluate.py`

**Purpose**: Evaluate trained model on test set.

**Metrics to compute**:
- MAE, MSE, RMSE for valence and arousal
- Correlation coefficients
- Scatter plots: predicted vs. actual

**Complete code**:

```python
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
```

---

## Phase 4: GigaMIDI Annotation

### Step 4.1: Create `annotate_gigamidi.py`

**File**: `valence_arousal/annotate_gigamidi.py`

**Purpose**: Apply trained model to GigaMIDI to predict VA values.

**Key components**:
- Load trained model
- Stream GigaMIDI dataset
- Extract latents on-the-fly
- Predict VA for each bar
- Write CSV incrementally (one row at a time) to avoid losing progress
- Support resume by loading existing CSV and skipping already-processed songs

**Output Format**: CSV file with four columns: `md5`, `bar_number`, `valence`, `arousal`
Each row represents one bar of a song, so songs with multiple bars will have multiple rows.

**Note on Bar Detection**: Bar boundaries are determined by MuseTok's bar detection logic (already implemented in Phase 1 via `midi_utils.py`):
- Bars are based on time signatures: for a time signature n/beat, one bar contains n beats
- A new bar starts whenever the time signature changes
- Bar positions are extracted from REMI events (where 'Bar' events mark bar boundaries)
- The `extract_latents_from_midi()` function returns `bar_positions` which we use to index bars
- We use MuseTok's bar logic as-is since Phase 1 is complete

**Complete code**:

```python
"""
Annotate GigaMIDI dataset with valence/arousal predictions.
Uses streaming mode to avoid downloading entire dataset.
Writes CSV incrementally to avoid losing progress.
Supports resume by skipping already-processed songs.
"""

import argparse
import logging
import sys
import os
from os.path import dirname, realpath
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

sys.path.insert(0, dirname(dirname(realpath(__file__))))

from utils.musetok_utils import load_musetok_model, extract_latents_from_midi
from utils.midi_utils import load_midi_symusic
from pretrain_model.model import ValenceArousalMLP
from utils.data_utils import (
    TRAINED_MODEL_DIR, MUSETOK_CHECKPOINT_DIR, GIGAMIDI_ANNOTATIONS_DIR,
    ensure_dir
)

def parse_args():
    parser = argparse.ArgumentParser(prog="AnnotateGigaMIDI", description="Annotate GigaMIDI with VA predictions.")
    
    parser.add_argument("--model_path", type=str, required=True,
                       help="Path to trained VA model checkpoint")
    parser.add_argument("--checkpoint_path", type=str, default=None,
                       help="Path to MuseTok checkpoint")
    parser.add_argument("--vocab_path", type=str, default=None,
                       help="Path to MuseTok vocabulary")
    parser.add_argument("--input_dim", type=int, default=128,
                       help="Input dimension for model (should match latent dim)")
    parser.add_argument("--hidden_dim", type=int, default=None,
                       help="Hidden dimension")
    parser.add_argument("--use_tanh", action="store_true", default=True,
                       help="Use tanh activation")
    parser.add_argument("--dropout", type=float, default=0.1,
                       help="Dropout rate")
    parser.add_argument("--device", type=str, default="cuda",
                       help="Device")
    parser.add_argument("--streaming", action="store_true", default=True,
                       help="Use streaming mode")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split")
    parser.add_argument("--max_samples", type=int, default=None,
                       help="Maximum number of samples to process (for testing)")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output CSV file path (defaults to <STORAGE_DIR>/gigamidi_annotations/annotations.csv)")
    parser.add_argument("--resume", action="store_true", default=False,
                       help="Resume from existing CSV file (skip already-processed songs)")
    
    args = parser.parse_args()
    if args.hidden_dim is None:
        args.hidden_dim = args.input_dim // 2
    if args.output_path is None:
        ensure_dir(GIGAMIDI_ANNOTATIONS_DIR)
        args.output_path = os.path.join(GIGAMIDI_ANNOTATIONS_DIR, "annotations.csv")
    return args

def load_existing_annotations(csv_path):
    """Load existing annotations from CSV and return set of processed (md5, bar_number) pairs."""
    processed = set()
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if 'md5' in df.columns and 'bar_number' in df.columns:
                # Create set of (md5, bar_number) tuples
                processed = set(zip(df['md5'].dropna(), df['bar_number'].dropna()))
            elif 'md5' in df.columns:
                # Fallback: if no bar_number column, just use md5 (for backward compatibility)
                processed = set(df['md5'].dropna())
            logging.info(f"Loaded {len(processed)} existing bar annotations from {csv_path}")
        except Exception as e:
            logging.warning(f"Error loading existing CSV: {e}. Starting fresh.")
    return processed

def process_song(sample, model, musetok_model, vocab, device):
    """
    Process a single song: extract latents on-the-fly and predict VA for each bar.
    
    Bar boundaries are determined by MuseTok's bar detection logic (from Phase 1):
    - Bars respect time signatures (one bar per n beats where n is the numerator)
    - New bars start when time signatures change
    - Bar positions come from REMI events where 'Bar' events mark boundaries
    
    Returns:
        list of dicts, each with 'md5', 'bar_number', 'valence', 'arousal' or None if error
    """
    try:
        md5 = sample.get("md5", "")
        if not md5:
            logging.warning("Sample missing md5, skipping")
            return None
        
        # Load MIDI from bytes
        midi_bytes = sample["music"]
        
        # Extract latents on-the-fly
        latents, bar_positions = extract_latents_from_midi(
            midi_bytes, musetok_model, vocab, device
        )
        
        if len(latents) == 0:
            logging.debug(f"No latents extracted for {md5}")
            return None
        
        # Convert to tensor
        latents_tensor = torch.from_numpy(latents).float().unsqueeze(0).to(device)  # (1, n_bars, dim)
        mask = torch.ones(1, len(latents), dtype=torch.bool).to(device)
        
        # Predict VA for each bar
        with torch.no_grad():
            outputs = model(latents_tensor, mask=mask)
            # Model outputs shape: (batch, seq_len, 2) for per-bar predictions
            if len(outputs.shape) == 3:
                # Per-bar predictions: (1, n_bars, 2)
                n_bars = outputs.shape[1]
                results = []
                for bar_idx in range(n_bars):
                    results.append({
                        "md5": md5,
                        "bar_number": bar_idx,
                        "valence": float(outputs[0, bar_idx, 0].cpu().item()),
                        "arousal": float(outputs[0, bar_idx, 1].cpu().item()),
                    })
                return results
            else:
                # Fallback: if model outputs single prediction, assign to bar 0
                return [{
                    "md5": md5,
                    "bar_number": 0,
                    "valence": float(outputs[0, 0].cpu().item()),
                    "arousal": float(outputs[0, 1].cpu().item()),
                }]
    except Exception as e:
        logging.warning(f"Error processing song {sample.get('md5', 'unknown')}: {e}")
        return None

if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Using device: {device}")
    
    # Initialize CSV file
    ensure_dir(os.path.dirname(args.output_path))
    
    # Load existing annotations to get processed (md5, bar_number) pairs (only if resuming)
    processed = set()
    if args.resume and os.path.exists(args.output_path):
        try:
            processed = load_existing_annotations(args.output_path)
            logging.info(f"Resuming: found {len(processed)} existing bar annotations")
        except Exception as e:
            logging.warning(f"Error loading existing CSV: {e}. Starting fresh.")
    else:
        # Write column names if not resuming (overwrite existing file)
        df_header = pd.DataFrame(columns=['md5', 'bar_number', 'valence', 'arousal'])
        df_header.to_csv(args.output_path, mode='w', index=False)
        logging.info(f"Initialized CSV file with headers: {args.output_path}")
    
    # Load VA model
    logging.info("Loading VA prediction model...")
    va_model = ValenceArousalMLP(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        use_tanh=args.use_tanh,
        dropout=args.dropout,
    ).to(device)
    va_model.load_state_dict(torch.load(args.model_path, map_location=device))
    va_model.eval()
    
    # Load MuseTok model
    logging.info("Loading MuseTok model...")
    musetok_model, vocab = load_musetok_model(
        checkpoint_path=args.checkpoint_path,
        vocab_path=args.vocab_path,
        device=str(device),
    )
    
    # Load GigaMIDI dataset
    logging.info(f"Loading GigaMIDI dataset (split={args.split}, streaming={args.streaming})...")
    if args.streaming:
        dataset = load_dataset("Metacreation/GigaMIDI", split=args.split, streaming=True)
    else:
        dataset = load_dataset("Metacreation/GigaMIDI", split=args.split)
    
    # Process songs
    count = 0
    skipped = 0
    errors = 0
    
    iterator = iter(dataset)
    if args.max_samples:
        iterator = tqdm(iterator, total=args.max_samples, desc="Processing")
    else:
        iterator = tqdm(iterator, desc="Processing")
    
    try:
        for sample in iterator:
            if args.max_samples and count >= args.max_samples:
                break
            
            md5 = sample.get("md5", "")
            if not md5:
                errors += 1
                continue
            
            # Process song (returns list of bar-level predictions)
            results = process_song(sample, va_model, musetok_model, vocab, device)
            
            if results is not None and len(results) > 0:
                # Filter out already-processed bars if resuming
                bars_to_write = []
                for result in results:
                    bar_key = (result["md5"], result["bar_number"])
                    if bar_key not in processed:
                        bars_to_write.append(result)
                        processed.add(bar_key)
                
                if bars_to_write:
                    # Write all bars for this song to CSV
                    df_rows = pd.DataFrame(bars_to_write)
                    df_rows.to_csv(args.output_path, mode='a', header=False, index=False)
                    count += len(bars_to_write)
                else:
                    skipped += 1
                
                # Log progress periodically
                if count % 1000 == 0:  # Changed to 1000 since we're counting bars now
                    logging.info(f"Processed {count} bars (skipped songs: {skipped}, errors: {errors})")
            else:
                errors += 1
    
    logging.info(f"\nProcessing complete!")
    logging.info(f"Successfully processed: {count} bars")
    logging.info(f"Skipped (already processed): {skipped} songs")
    logging.info(f"Errors: {errors} songs")
    logging.info(f"Annotations saved to: {args.output_path}")
```

### Step 4.2: Optional - Create `save_annotations.py` (Not Required)

**Note**: Since `annotate_gigamidi.py` writes CSV directly, `save_annotations.py` is not necessary.
If you need format conversion or validation, you can create this script later.

**File**: `valence_arousal/save_annotations.py` (optional, not created by default)

**Purpose**: Convert CSV annotations to other formats or validate annotations.

**Complete code**:

```python
"""
Convert CSV annotations to other formats or validate annotations.
"""

import argparse
import os
import pandas as pd
from utils.data_utils import GIGAMIDI_ANNOTATIONS_DIR, ensure_dir

def parse_args():
    parser = argparse.ArgumentParser(prog="SaveAnnotations", description="Convert/validate annotations.")
    
    parser.add_argument("--input_path", type=str, required=True,
                       help="Input CSV file path")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output file path (optional, for format conversion)")
    parser.add_argument("--format", type=str, default="parquet",
                       choices=["parquet", "json", "jsonl"],
                       help="Output format (if converting)")
    parser.add_argument("--validate", action="store_true",
                       help="Validate annotations and print statistics")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Load CSV
    print(f"Loading annotations from {args.input_path}...")
    df = pd.read_csv(args.input_path)
    print(f"Loaded {len(df)} annotations")
    
    # Validate
    if args.validate:
        print("\nValidation Statistics:")
        print(f"Total bars: {len(df)}")
        print(f"Total songs: {df['md5'].nunique()}")
        print(f"Average bars per song: {len(df) / df['md5'].nunique():.2f}")
        if 'bar_number' in df.columns:
            print(f"Bar number range: [{df['bar_number'].min()}, {df['bar_number'].max()}]")
        print(f"Valence range: [{df['valence'].min():.3f}, {df['valence'].max():.3f}]")
        print(f"Arousal range: [{df['arousal'].min():.3f}, {df['arousal'].max():.3f}]")
        print(f"Missing values: {df.isnull().sum().sum()}")
        
        # Check for duplicates (same md5 and bar_number)
        if 'bar_number' in df.columns:
            duplicates = df.duplicated(subset=['md5', 'bar_number']).sum()
            if duplicates > 0:
                print(f"WARNING: {duplicates} duplicate (md5, bar_number) pairs found!")
            else:
                print("No duplicate (md5, bar_number) pairs found.")
        else:
            duplicates = df['md5'].duplicated().sum()
            if duplicates > 0:
                print(f"WARNING: {duplicates} duplicate md5s found!")
            else:
                print("No duplicate md5s found.")
    
    # Convert format if requested
    if args.output_path:
        ensure_dir(os.path.dirname(args.output_path))
        
        if args.format == "parquet":
            df.to_parquet(args.output_path, index=False, engine="pyarrow")
            print(f"Saved to {args.output_path}")
            
        elif args.format == "json":
            df.to_json(args.output_path, orient="records", indent=2)
            print(f"Saved to {args.output_path}")
            
        elif args.format == "jsonl":
            df.to_json(args.output_path, orient="records", lines=True)
            print(f"Saved to {args.output_path}")
```

---

## Phase 5: Analysis

Each analysis script is independent and generates a single plot or analysis.

**Note**: All scripts read from CSV file with columns: `md5`, `bar_number`, `valence`, `arousal`

**Note on Bar Numbering**: Bar numbers are 0-indexed and correspond to MuseTok's bar detection:
- Bar 0 is the first bar of the song
- Bars are determined by time signatures (one bar per n beats where n is the numerator)
- New bars start when time signatures change (handled by MuseTok in Phase 1)

### Step 5.1: Create `analyze_emotion_annotations/plot_histograms.py`

**File**: `valence_arousal/analyze_emotion_annotations/plot_histograms.py`

**Purpose**: Create histograms for valence and arousal distributions.

**Code structure**:

```python
"""
Create histograms for valence and arousal distributions.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_histograms(annotations_path: str, output_path: str, bins: int = 50):
    """Create histograms for valence and arousal."""
    df = pd.read_csv(annotations_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].hist(df['valence'], bins=bins, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Valence')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Valence Distribution')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(df['arousal'], bins=bins, edgecolor='black', alpha=0.7)
    axes[1].set_xlabel('Arousal')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title('Arousal Distribution')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved histograms to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True,
                       help="Path to annotations CSV file")
    parser.add_argument("--output_path", required=True,
                       help="Output file path for histogram plot")
    parser.add_argument("--bins", type=int, default=50,
                       help="Number of bins for histograms")
    args = parser.parse_args()
    
    plot_histograms(args.annotations_path, args.output_path, args.bins)
```

### Step 5.2: Create `analyze_emotion_annotations/plot_boxplots.py`

**File**: `valence_arousal/analyze_emotion_annotations/plot_boxplots.py`

**Purpose**: Create boxplots for valence and arousal.

**Code structure**:

```python
"""
Create boxplots for valence and arousal.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_boxplots(annotations_path: str, output_path: str):
    """Create boxplots for valence and arousal."""
    df = pd.read_csv(annotations_path)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].boxplot(df['valence'], vert=True)
    axes[0].set_ylabel('Valence')
    axes[0].set_title('Valence Boxplot')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].boxplot(df['arousal'], vert=True)
    axes[1].set_ylabel('Arousal')
    axes[1].set_title('Arousal Boxplot')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved boxplots to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True,
                       help="Path to annotations CSV file")
    parser.add_argument("--output_path", required=True,
                       help="Output file path for boxplot")
    args = parser.parse_args()
    
    plot_boxplots(args.annotations_path, args.output_path)
```

### Step 5.3: Create `analyze_emotion_annotations/plot_by_genre.py`

**File**: `valence_arousal/analyze_emotion_annotations/plot_by_genre.py`

**Purpose**: Create boxplots comparing valence/arousal by genre (requires GigaMIDI metadata).

**Code structure**:

```python
"""
Create boxplots comparing valence/arousal by genre.
Requires GigaMIDI metadata to be loaded.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datasets import load_dataset

def load_gigamidi_metadata(split='train', streaming=True):
    """
    Load GigaMIDI metadata to get genre information.
    Returns dict mapping md5 to metadata (including music_styles_curated).
    """
    metadata = {}
    try:
        if streaming:
            dataset = load_dataset("Metacreation/GigaMIDI", split=split, streaming=True)
        else:
            dataset = load_dataset("Metacreation/GigaMIDI", split=split)
        
        for sample in dataset:
            md5 = sample.get("md5", "")
            if md5:
                metadata[md5] = {
                    "music_styles_curated": sample.get("music_styles_curated", []),
                    "title": sample.get("title", ""),
                    "artist": sample.get("artist", ""),
                }
    except Exception as e:
        print(f"Warning: Could not load GigaMIDI metadata: {e}")
    return metadata

def plot_by_genre(annotations_path: str, output_path: str, top_n: int = 10, split: str = 'train', streaming: bool = True):
    """Create boxplots comparing valence/arousal by genre."""
    df = pd.read_csv(annotations_path)
    
    # Load metadata
    print("Loading GigaMIDI metadata...")
    metadata = load_gigamidi_metadata(split=split, streaming=streaming)
    
    if not metadata:
        print("Error: Could not load metadata. Cannot create genre plots.")
        return
    
    print(f"Loaded metadata for {len(metadata)} songs")
    
    # Merge metadata with annotations
    df_with_genre = df.copy()
    df_with_genre['genres'] = df_with_genre['md5'].map(
        lambda x: metadata.get(x, {}).get('music_styles_curated', [])
    )
    
    # Explode genres (songs can have multiple genres)
    df_exploded = df_with_genre.explode('genres')
    df_exploded = df_exploded[df_exploded['genres'].notna()]
    
    if len(df_exploded) == 0:
        print("Warning: No genre information available. Cannot create genre plots.")
        return
    
    # Get top N genres by count
    top_genres = df_exploded['genres'].value_counts().head(top_n).index.tolist()
    df_top = df_exploded[df_exploded['genres'].isin(top_genres)]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Valence by genre
    sns.boxplot(data=df_top, x='genres', y='valence', ax=axes[0])
    axes[0].set_title('Valence by Genre')
    axes[0].set_xlabel('Genre')
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(True, alpha=0.3)
    
    # Arousal by genre
    sns.boxplot(data=df_top, x='genres', y='arousal', ax=axes[1])
    axes[1].set_title('Arousal by Genre')
    axes[1].set_xlabel('Genre')
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved genre boxplots to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True,
                       help="Path to annotations CSV file")
    parser.add_argument("--output_path", required=True,
                       help="Output file path for genre boxplot")
    parser.add_argument("--top_n", type=int, default=10,
                       help="Number of top genres to include")
    parser.add_argument("--split", type=str, default="train",
                       help="GigaMIDI split to load metadata from")
    parser.add_argument("--no_streaming", action="store_true",
                       help="Disable streaming mode for metadata loading")
    args = parser.parse_args()
    
    plot_by_genre(args.annotations_path, args.output_path, args.top_n, args.split, not args.no_streaming)
```

### Step 5.4: Create `analyze_emotion_annotations/plot_song_curves.py`

**File**: `valence_arousal/analyze_emotion_annotations/plot_song_curves.py`

**Purpose**: Create valence/arousal curves (bar-by-bar) for example songs.

**Code structure**:

```python
"""
Create valence/arousal curves (bar-by-bar) for example songs.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import random

def plot_song_curves(annotations_path: str, output_path: str, n_examples: int = 5, seed: int = None):
    """Create valence/arousal curves for random songs."""
    if seed is not None:
        random.seed(seed)
    
    df = pd.read_csv(annotations_path)
    
    # Get unique songs
    unique_songs = df['md5'].unique()
    n_examples = min(n_examples, len(unique_songs))
    example_songs = random.sample(list(unique_songs), n_examples)
    
    fig, axes = plt.subplots(n_examples, 1, figsize=(12, 3 * n_examples))
    if n_examples == 1:
        axes = [axes]
    
    for idx, md5 in enumerate(example_songs):
        song_df = df[df['md5'] == md5].sort_values('bar_number')
        
        ax = axes[idx]
        ax.plot(song_df['bar_number'], song_df['valence'], 
                marker='o', label='Valence', linewidth=2)
        ax.plot(song_df['bar_number'], song_df['arousal'], 
                marker='s', label='Arousal', linewidth=2)
        ax.set_xlabel('Bar Number')
        ax.set_ylabel('Value')
        ax.set_title(f'Song {md5[:8]}... (Valence/Arousal over time)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-1.1, 1.1)
    
    plt.tight_layout()
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved song curves to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True,
                       help="Path to annotations CSV file")
    parser.add_argument("--output_path", required=True,
                       help="Output file path for song curves plot")
    parser.add_argument("--n_examples", type=int, default=5,
                       help="Number of example songs to plot")
    parser.add_argument("--seed", type=int, default=None,
                       help="Random seed for song selection")
    args = parser.parse_args()
    
    plot_song_curves(args.annotations_path, args.output_path, args.n_examples, args.seed)
```

### Step 5.5: Create `analyze_emotion_annotations/print_statistics.py` (Optional)

**File**: `valence_arousal/analyze_emotion_annotations/print_statistics.py`

**Purpose**: Print summary statistics about the annotations.

**Code structure**:

```python
"""
Print summary statistics about GigaMIDI annotations.
"""

import argparse
import pandas as pd

def print_statistics(annotations_path: str):
    """Print summary statistics."""
    df = pd.read_csv(annotations_path)
    
    print("\nSummary Statistics:")
    print(f"Total bars: {len(df)}")
    print(f"Total songs: {df['md5'].nunique()}")
    print(f"Average bars per song: {len(df) / df['md5'].nunique():.2f}")
    print(f"\nValence:")
    print(df['valence'].describe())
    print(f"\nArousal:")
    print(df['arousal'].describe())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations_path", required=True,
                       help="Path to annotations CSV file")
    args = parser.parse_args()
    
    print_statistics(args.annotations_path)
```

---

## Testing Checklist

After each phase, test the following:

- [ ] Phase 1: Can load MIDI, convert to events, extract latents
- [ ] Phase 2: Can preprocess EMOPIA and create labels
- [ ] Phase 3: Can train model and achieve reasonable performance
- [ ] Phase 4: Can annotate GigaMIDI successfully
- [ ] Phase 5: Can generate analysis and visualizations

---

## Next Steps

1. Start with Phase 1, Step 1.1
2. Test each component as you build it
3. Refer to `musetok` and `jingyue_latents` codebases for implementation details
4. Adjust based on actual MuseTok API and EMOPIA structure

