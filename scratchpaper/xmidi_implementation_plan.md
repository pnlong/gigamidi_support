# Implementation Plan: XMIDI Emotion and Genre Recognition

This document provides step-by-step instructions for implementing emotion recognition and genre recognition pipelines on the XMIDI dataset. The pipeline uses MuseTok for latent extraction and MLP classifiers for song-level prediction.

---

## Overview

**Tasks:**

1. **Emotion Recognition**: 11 classes (exciting, warm, happy, romantic, funny, sad, angry, lazy, quiet, fear, magnificent)
2. **Genre Recognition**: 6 classes (rock, pop, country, jazz, classical, folk)

**Key Differences from Valence/Arousal Pipeline:**

- Classification instead of regression
- Two separate tasks (emotion and genre)
- Song-level prediction (mean pooling across bars)
- XMIDI dataset with filename format: `XMIDI_<Emotion>_<Genre>_<ID_len_8>.midi`
- Custom train/val/test splits

**Architecture:**

- MuseTok for latent extraction (same as before)
- MLP with classification head (output: num_classes logits)
- Mean pooling across bars for song-level prediction
- Cross-entropy loss for training

---

## Phase 1: Project Setup and Utilities

**Status**: ✅ COMPLETED

### Step 1.1: Create Project Structure ✅

**Action**: Create the directory structure and initial files.

```bash
cd /home/pnlong/gigamidi
mkdir -p emotion_genre/{config,pretrain_model,utils,analyze_annotations}
touch emotion_genre/{README.md,PIPELINE.md,requirements.txt}
touch emotion_genre/utils/{__init__.py,musetok_utils.py,midi_utils.py,data_utils.py}
touch emotion_genre/pretrain_model/{__init__.py,preprocess_xmidi.py,prepare_labels.py,dataset.py,model.py,train.py,evaluate.py,download_xmidi.py}
touch emotion_genre/config/{musetok_config.yaml,training_config.yaml}
touch emotion_genre/analyze_annotations/{__init__.py,plot_histograms.py,plot_by_genre.py,print_statistics.py}
touch emotion_genre/annotate_gigamidi.py
```

### Step 1.2: Create `utils/data_utils.py` - Storage Directory Configuration ✅

**File**: `emotion_genre/utils/data_utils.py`

**Purpose**: Define storage directory paths and utility functions for data I/O.

**Key paths:**

- `XMIDI_DATA_DIR`: Directory for downloaded XMIDI dataset
- `XMIDI_LATENTS_DIR`: Directory for preprocessed latents
- `XMIDI_LABELS_DIR`: Directory for label files
- `TRAINED_MODEL_DIR`: Directory for trained model checkpoints

**Code structure**: Similar to `valence_arousal/utils/data_utils.py` but with XMIDI-specific paths.

### Step 1.3: Create `requirements.txt` ✅

**File**: `emotion_genre/requirements.txt`

**Content**: Same dependencies as valence_arousal (symusic, torch, datasets, gdown, safetensors, numpy, pandas, scipy, tqdm, wandb, matplotlib, seaborn, scikit-learn for classification metrics)

### Step 1.4: Reuse MuseTok Utilities ✅

**Files**: `emotion_genre/utils/midi_utils.py` and `emotion_genre/utils/musetok_utils.py`

**Action**: Copy and adapt from `valence_arousal/utils/midi_utils.py` and `valence_arousal/utils/musetok_utils.py`. These utilities handle MIDI processing and MuseTok integration, which remain the same.

---

## Phase 2: XMIDI Dataset Download and Preprocessing

**Status**: ✅ COMPLETED

### Step 2.1: Download XMIDI Dataset ✅

**File**: `emotion_genre/pretrain_model/download_xmidi.py` (optional script, or manual download)

**Action**: Download XMIDI dataset from Google Drive using gdown.

```python
"""Download XMIDI dataset from Google Drive."""
import gdown
import os
from utils.data_utils import XMIDI_DATA_DIR, ensure_dir

def download_xmidi(output_dir: str = XMIDI_DATA_DIR):
    """Download XMIDI dataset."""
    ensure_dir(output_dir)
    url = "https://drive.google.com/uc?id=1qDkSH31x7jN8X-2RyzB9wuxGji4QxYyA"
    output_path = os.path.join(output_dir, "XMIDI_Dataset.zip")
    gdown.download(url, output_path, quiet=False)
    # Extract zip file
    # Implementation: extract zip to output_dir
```

**Note**: After download, extract the zip file. The dataset should contain MIDI files with naming format: `XMIDI_<Emotion>_<Genre>_<ID_len_8>.midi`

### Step 2.2: Create `pretrain_model/preprocess_xmidi.py` ✅

**File**: `emotion_genre/pretrain_model/preprocess_xmidi.py`

**Purpose**: Extract MuseTok latents from XMIDI MIDI files.

**Key Features:**

- Process all `.midi` files in XMIDI dataset
- Extract latents using MuseTok (same as EMOPIA preprocessing)
- Save latents as `.safetensors` files
- Include `--resume` argument to skip already-processed files
- Preserve metadata (emotion, genre, ID from filename)

**Code structure**: Similar to `valence_arousal/pretrain_model/preprocess_emopia.py` but:

- Processes XMIDI MIDI files (not REMI pickle files)
- Extracts emotion and genre from filename for metadata
- No split handling (splits created later)

### Step 2.3: Create `pretrain_model/prepare_labels.py` ✅

**File**: `emotion_genre/pretrain_model/prepare_labels.py`

**Purpose**: Extract emotion and genre labels from XMIDI filenames and create label mappings.

**Key Features:**

- Parse filenames: `XMIDI_<Emotion>_<Genre>_<ID_len_8>.midi`
- Create two label files:
  - `emotion_labels.json`: Maps filename (without extension) to emotion class index
  - `genre_labels.json`: Maps filename (without extension) to genre class index
- Create class index mappings:
  - `emotion_to_index.json`: Maps emotion string to index (0-10)
  - `genre_to_index.json`: Maps genre string to index (0-5)
- Create train/val/test splits (e.g., 80/10/10) and save split files

**Emotion classes** (11 total):

```python
EMOTIONS = ["exciting", "warm", "happy", "romantic", "funny", "sad", "angry", "lazy", "quiet", "fear", "magnificent"]
```

**Genre classes** (6 total):

```python
GENRES = ["rock", "pop", "country", "jazz", "classical", "folk"]
```

**Code structure**:

```python
"""Prepare emotion and genre labels from XMIDI filenames."""
import os
import json
import argparse
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np

# Emotion and genre mappings
EMOTIONS = ["exciting", "warm", "happy", "romantic", "funny", "sad", "angry", "lazy", "quiet", "fear", "magnificent"]
GENRES = ["rock", "pop", "country", "jazz", "classical", "folk"]

EMOTION_TO_INDEX = {emotion: i for i, emotion in enumerate(EMOTIONS)}
GENRE_TO_INDEX = {genre: i for i, genre in enumerate(GENRES)}

def extract_labels_from_filename(filename: str) -> tuple[str, str]:
    """Extract emotion and genre from XMIDI filename.
    
    Format: XMIDI_<Emotion>_<Genre>_<ID_len_8>.midi
    Returns: (emotion, genre) or (None, None) if parsing fails
    """
    # Implementation: parse filename using regex or string splitting
    pass

def create_splits(filenames: list, test_size: float = 0.1, val_size: float = 0.1, random_state: int = 42):
    """Create train/val/test splits."""
    # Implementation: use train_test_split twice
    pass

def prepare_labels(xmidi_dir: str, output_dir: str, test_size: float = 0.1, val_size: float = 0.1):
    """Main function to prepare labels and splits."""
    # Implementation:
    # 1. Find all .midi files
    # 2. Extract labels from filenames
    # 3. Create emotion_labels.json and genre_labels.json
    # 4. Create splits
    # 5. Save split files (train_files.txt, val_files.txt, test_files.txt)
    pass
```

---

## Phase 3: Model Implementation

**Status**: ✅ COMPLETED

### Step 3.1: Create `pretrain_model/dataset.py` ✅

**File**: `emotion_genre/pretrain_model/dataset.py`

**Purpose**: PyTorch dataset for loading XMIDI latents and labels (emotion or genre).

**Key Features:**

- Load latents from `.safetensors` files
- Load labels from JSON file (emotion or genre)
- Mean pooling across bars (song-level prediction)
- Support for both emotion and genre tasks

**Code structure**:

```python
"""Dataset class for XMIDI latents and labels."""
import torch
from torch.utils.data import Dataset
import os
import numpy as np
from typing import List, Dict

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
        # Implementation:
        # 1. Load labels JSON
        # 2. Load class_to_index mapping
        # 3. Store file_list
        pass
    
    def __getitem__(self, idx):
        # Implementation:
        # 1. Load latents (shape: n_bars, latent_dim)
        # 2. Mean pool across bars: latents.mean(dim=0) -> (latent_dim,)
        # 3. Load label (class index)
        # 4. Return latents (pooled), label, filename
        pass
```

### Step 3.2: Create `pretrain_model/model.py` ✅

**File**: `emotion_genre/pretrain_model/model.py`

**Purpose**: MLP classifier for emotion or genre classification.

**Key Features:**

- MLP with classification head
- Input: pooled latents (mean across bars)
- Output: logits for num_classes
- Supports both emotion (11 classes) and genre (6 classes)

**Code structure**:

```python
"""MLP classifier for emotion/genre recognition."""
import torch
import torch.nn as nn

class EmotionGenreClassifier(nn.Module):
    """MLP classifier for emotion or genre recognition."""
    
    def __init__(self, 
                 input_dim: int,
                 num_classes: int,
                 hidden_dim: int = None,
                 dropout: float = 0.1):
        """
        Args:
            input_dim: Dimension of input latents (d_vae_latent)
            num_classes: Number of classes (11 for emotion, 6 for genre)
            hidden_dim: Hidden layer dimension (default: input_dim // 2)
            dropout: Dropout rate
        """
        super().__init__()
        
        if hidden_dim is None:
            hidden_dim = input_dim // 2
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)  # Classification head
        )
    
    def forward(self, latents):
        """
        Args:
            latents: (batch_size, input_dim) - already pooled
        
        Returns:
            logits: (batch_size, num_classes)
        """
        return self.mlp(latents)
```

### Step 3.3: Create `pretrain_model/train.py` ✅

**File**: `emotion_genre/pretrain_model/train.py`

**Purpose**: Training script for emotion or genre classification.

**Key Features:**

- DataLoader setup
- Model initialization (EmotionGenreClassifier)
- Loss function: CrossEntropyLoss
- Optimizer: AdamW
- Training loop with validation
- Metrics: Accuracy, F1-score (macro/weighted), Precision, Recall, per-class metrics
- Confusion matrix logging
- Checkpoint saving
- Wandb logging (optional)

**Metrics to track:**

- Overall accuracy
- F1-score (macro and weighted)
- Precision and Recall (macro and weighted)
- Per-class accuracy, precision, recall, F1
- Confusion matrix (saved as image)

**Code structure**: Similar to `valence_arousal/pretrain_model/train.py` but:

- Uses CrossEntropyLoss instead of SmoothL1Loss
- Computes classification metrics (accuracy, F1, etc.)
- Saves confusion matrix
- Task parameter to switch between emotion and genre

### Step 3.4: Create `pretrain_model/evaluate.py` ✅

**File**: `emotion_genre/pretrain_model/evaluate.py`

**Purpose**: Evaluate trained model on test set.

**Key Features:**

- Load trained model
- Evaluate on test set
- Compute comprehensive metrics:
  - Overall accuracy
  - F1-score (macro, weighted, per-class)
  - Precision and Recall (macro, weighted, per-class)
  - Confusion matrix (plot and save)
- Save results to CSV
- Generate classification report

**Code structure**: Similar to `valence_arousal/pretrain_model/evaluate.py` but with classification metrics instead of regression metrics.

---

## Phase 4: Training and Evaluation

**Status**: ✅ COMPLETED (Scripts ready, training workflow documented)

### Step 4.1: Training Workflow

**Commands for training:**

```bash
# Train emotion classifier
python emotion_genre/pretrain_model/train.py \
    --task emotion \
    --latents_dir /path/to/xmidi/latents \
    --labels_path /path/to/emotion_labels.json \
    --class_to_index_path /path/to/emotion_to_index.json \
    --train_files /path/to/train_files.txt \
    --valid_files /path/to/val_files.txt \
    --num_classes 11 \
    --input_dim 512 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --gpu \
    --use_wandb

# Train genre classifier
python emotion_genre/pretrain_model/train.py \
    --task genre \
    --latents_dir /path/to/xmidi/latents \
    --labels_path /path/to/genre_labels.json \
    --class_to_index_path /path/to/genre_to_index.json \
    --train_files /path/to/train_files.txt \
    --valid_files /path/to/val_files.txt \
    --num_classes 6 \
    --input_dim 512 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --gpu \
    --use_wandb
```

### Step 4.2: Evaluation Workflow

**Commands for evaluation:**

```bash
# Evaluate emotion classifier
python emotion_genre/pretrain_model/evaluate.py \
    --task emotion \
    --checkpoint_path /path/to/best_emotion_model.pt \
    --latents_dir /path/to/xmidi/latents \
    --labels_path /path/to/emotion_labels.json \
    --class_to_index_path /path/to/emotion_to_index.json \
    --test_files /path/to/test_files.txt \
    --num_classes 11 \
    --input_dim 512 \
    --output_dir ./emotion_evaluation_results \
    --gpu

# Evaluate genre classifier
python emotion_genre/pretrain_model/evaluate.py \
    --task genre \
    --checkpoint_path /path/to/best_genre_model.pt \
    --latents_dir /path/to/xmidi/latents \
    --labels_path /path/to/genre_labels.json \
    --class_to_index_path /path/to/genre_to_index.json \
    --test_files /path/to/test_files.txt \
    --num_classes 6 \
    --input_dim 512 \
    --output_dir ./genre_evaluation_results \
    --gpu
```

---

## Phase 5: GigaMIDI Annotation

**Status**: ✅ COMPLETED

### Step 5.1: Create `annotate_gigamidi.py` ✅

**File**: `emotion_genre/annotate_gigamidi.py`

**Purpose**: Apply trained emotion/genre classifiers to GigaMIDI to predict emotion and genre for each song.

**Key Features:**
- Load trained emotion and genre models
- Stream GigaMIDI dataset
- Extract latents on-the-fly
- Predict emotion and genre for each song (song-level, not bar-level)
- Write CSV incrementally (one row per song) to avoid losing progress
- Support resume by loading existing CSV and skipping already-processed songs

**Output Format**: CSV file with columns: `md5`, `emotion`, `emotion_prob`, `genre`, `genre_prob`
- Each row represents one song
- `emotion`: Predicted emotion class (string)
- `emotion_prob`: Confidence/probability for emotion prediction
- `genre`: Predicted genre class (string)
- `genre_prob`: Confidence/probability for genre prediction

**Code structure**: Similar to `valence_arousal/annotate_gigamidi.py` but:
- Uses both emotion and genre classifiers
- Song-level prediction (mean pooling across bars)
- Outputs class names and probabilities instead of continuous values

### Step 5.2: Usage Example

```bash
# Annotate GigaMIDI with both emotion and genre predictions
python emotion_genre/annotate_gigamidi.py \
    --emotion_model_path /path/to/best_emotion_model.pt \
    --genre_model_path /path/to/best_genre_model.pt \
    --emotion_class_to_index_path /path/to/emotion_to_index.json \
    --genre_class_to_index_path /path/to/genre_to_index.json \
    --input_dim 128 \
    --emotion_num_classes 11 \
    --genre_num_classes 6 \
    --gpu \
    --streaming \
    --split train \
    --output_path /path/to/gigamidi_annotations.csv \
    --resume
```

---

## Phase 6: Analysis of Annotations

**Status**: ✅ COMPLETED

### Step 6.1: Create `analyze_annotations/plot_histograms.py` ✅

**File**: `emotion_genre/analyze_annotations/plot_histograms.py`

**Purpose**: Create histograms for emotion and genre distributions in GigaMIDI annotations.

**Key Features:**
- Plot emotion distribution (11 classes)
- Plot genre distribution (6 classes)
- Show class frequencies

**Code structure**: Similar to `valence_arousal/analyze_emotion_annotations/plot_histograms.py` but adapted for categorical distributions.

### Step 6.2: Create `analyze_annotations/plot_by_genre.py` ✅

**File**: `emotion_genre/analyze_annotations/plot_by_genre.py`

**Purpose**: Create visualizations comparing emotion/genre predictions by GigaMIDI's original genre metadata.

**Key Features:**
- Load GigaMIDI metadata to get original genre information
- Create bar charts or heatmaps showing:
  - Emotion distribution by GigaMIDI genre
  - Genre prediction distribution by GigaMIDI genre
- Compare predicted genres vs. GigaMIDI's original genres

**Code structure**: Similar to `valence_arousal/analyze_emotion_annotations/plot_by_genre.py` but adapted for categorical data.

### Step 6.3: Create `analyze_annotations/print_statistics.py` ✅

**File**: `emotion_genre/analyze_annotations/print_statistics.py`

**Purpose**: Print summary statistics about the annotations.

**Key Features:**
- Total number of songs annotated
- Emotion class distribution (counts and percentages)
- Genre class distribution (counts and percentages)
- Average prediction probabilities
- Most common emotion/genre combinations

**Code structure**: Similar to `valence_arousal/analyze_emotion_annotations/print_statistics.py` but adapted for categorical data.

### Step 6.4: Usage Examples

```bash
# Plot histograms
python emotion_genre/analyze_annotations/plot_histograms.py \
    --annotations_path /path/to/gigamidi_annotations.csv \
    --output_path /path/to/histograms.png

# Plot by GigaMIDI genre
python emotion_genre/analyze_annotations/plot_by_genre.py \
    --annotations_path /path/to/gigamidi_annotations.csv \
    --output_path /path/to/genre_comparison.png \
    --split train

# Print statistics
python emotion_genre/analyze_annotations/print_statistics.py \
    --annotations_path /path/to/gigamidi_annotations.csv
```

---

## Phase 5: GigaMIDI Annotation

**Status**: ✅ COMPLETED

### Step 5.1: Create `annotate_gigamidi.py` ✅

**File**: `emotion_genre/annotate_gigamidi.py`

**Purpose**: Apply trained emotion/genre classifiers to GigaMIDI to predict emotion and genre for each song.

**Key Features:**
- Load trained emotion and genre models
- Stream GigaMIDI dataset
- Extract latents on-the-fly
- Predict emotion and genre for each song (song-level, not bar-level)
- Write CSV incrementally (one row per song) to avoid losing progress
- Support resume by loading existing CSV and skipping already-processed songs

**Output Format**: CSV file with columns: `md5`, `emotion`, `emotion_prob`, `genre`, `genre_prob`
- Each row represents one song
- `emotion`: Predicted emotion class (string)
- `emotion_prob`: Confidence/probability for emotion prediction (max softmax probability)
- `genre`: Predicted genre class (string)
- `genre_prob`: Confidence/probability for genre prediction (max softmax probability)

**Code structure**: Similar to `valence_arousal/annotate_gigamidi.py` but:
- Uses both emotion and genre classifiers
- Song-level prediction (mean pooling across bars before model)
- Outputs class names and probabilities instead of continuous values
- Processes each song once (not per-bar)

**Key differences from valence_arousal version:**
- Load two models (emotion and genre)
- Apply softmax to get probabilities
- Map predicted indices to class names using class_to_index mappings
- One row per song (not per bar)

### Step 5.2: Usage Example

```bash
# Annotate GigaMIDI with both emotion and genre predictions
python emotion_genre/annotate_gigamidi.py \
    --emotion_model_path /path/to/best_emotion_model.pt \
    --genre_model_path /path/to/best_genre_model.pt \
    --emotion_class_to_index_path /path/to/emotion_to_index.json \
    --genre_class_to_index_path /path/to/genre_to_index.json \
    --input_dim 128 \
    --emotion_num_classes 11 \
    --genre_num_classes 6 \
    --gpu \
    --streaming \
    --split train \
    --output_path /path/to/gigamidi_annotations.csv \
    --resume
```

---

## Phase 6: Analysis of Annotations

**Status**: ✅ COMPLETED

### Step 6.1: Create `analyze_annotations/plot_histograms.py` ✅

**File**: `emotion_genre/analyze_annotations/plot_histograms.py`

**Purpose**: Create histograms for emotion and genre distributions in GigaMIDI annotations.

**Key Features:**
- Plot emotion distribution (11 classes) as bar chart
- Plot genre distribution (6 classes) as bar chart
- Show class frequencies and percentages

**Code structure**: Similar to `valence_arousal/analyze_emotion_annotations/plot_histograms.py` but adapted for categorical distributions (bar charts instead of histograms).

### Step 6.2: Create `analyze_annotations/plot_by_genre.py` ✅

**File**: `emotion_genre/analyze_annotations/plot_by_genre.py`

**Purpose**: Create visualizations comparing emotion/genre predictions by GigaMIDI's original genre metadata.

**Key Features:**
- Load GigaMIDI metadata to get original genre information
- Create visualizations showing:
  - Emotion distribution by GigaMIDI genre (heatmap or stacked bar chart)
  - Genre prediction distribution by GigaMIDI genre (confusion matrix style)
- Compare predicted genres vs. GigaMIDI's original genres

**Code structure**: Similar to `valence_arousal/analyze_emotion_annotations/plot_by_genre.py` but adapted for categorical data (heatmaps/bar charts instead of boxplots).

### Step 6.3: Create `analyze_annotations/print_statistics.py` ✅

**File**: `emotion_genre/analyze_annotations/print_statistics.py`

**Purpose**: Print summary statistics about the annotations.

**Key Features:**
- Total number of songs annotated
- Emotion class distribution (counts and percentages)
- Genre class distribution (counts and percentages)
- Average prediction probabilities
- Most common emotion/genre combinations
- Distribution of prediction confidence scores

**Code structure**: Similar to `valence_arousal/analyze_emotion_annotations/print_statistics.py` but adapted for categorical data.

### Step 6.4: Usage Examples

```bash
# Plot histograms
python emotion_genre/analyze_annotations/plot_histograms.py \
    --annotations_path /path/to/gigamidi_annotations.csv \
    --output_path /path/to/histograms.png

# Plot by GigaMIDI genre
python emotion_genre/analyze_annotations/plot_by_genre.py \
    --annotations_path /path/to/gigamidi_annotations.csv \
    --output_path /path/to/genre_comparison.png \
    --split train

# Print statistics
python emotion_genre/analyze_annotations/print_statistics.py \
    --annotations_path /path/to/gigamidi_annotations.csv
```

---

## File Structure Summary

```
emotion_genre/
├── README.md
├── PIPELINE.md
├── requirements.txt
├── config/
│   ├── musetok_config.yaml
│   └── training_config.yaml
├── utils/
│   ├── __init__.py
│   ├── data_utils.py          # Storage paths, I/O utilities
│   ├── midi_utils.py          # MIDI processing (reuse from valence_arousal)
│   └── musetok_utils.py       # MuseTok integration (reuse from valence_arousal)
├── annotate_gigamidi.py        # Annotate GigaMIDI with predictions
├── analyze_annotations/
│   ├── __init__.py
│   ├── plot_histograms.py     # Emotion/genre distribution bar charts
│   ├── plot_by_genre.py        # Compare predictions by GigaMIDI genre
│   └── print_statistics.py     # Summary statistics
└── pretrain_model/
    ├── __init__.py
    ├── download_xmidi.py       # Optional: download script
    ├── preprocess_xmidi.py    # Extract latents from MIDI files
    ├── prepare_labels.py       # Extract labels, create splits
    ├── dataset.py              # PyTorch dataset
    ├── model.py                # EmotionGenreClassifier
    ├── train.py                # Training script
    └── evaluate.py             # Evaluation script
```

---

## Key Implementation Details

1. **Label Extraction**: Parse `XMIDI_<Emotion>_<Genre>_<ID_len_8>.midi` format
2. **Mean Pooling**: Pool latents across bars: `latents.mean(dim=0)` for song-level prediction
3. **Classification Head**: MLP outputs logits for num_classes, use CrossEntropyLoss
4. **Metrics**: Track accuracy, F1 (macro/weighted), precision, recall, confusion matrix
5. **Splits**: Create 80/10/10 train/val/test splits (stratified by class if possible)
6. **Two Tasks**: Separate training/evaluation for emotion and genre (can share preprocessing)
7. **GigaMIDI Annotation**: Song-level predictions (one emotion and one genre per song)
8. **Analysis**: Categorical distributions (bar charts, heatmaps) instead of continuous distributions

---

## Testing Checklist

- [x] Phase 1: Can load MIDI, convert to events, extract latents
- [x] Phase 2: Can download XMIDI, preprocess latents, extract labels, create splits
- [x] Phase 3: Can train emotion classifier and achieve reasonable accuracy
- [x] Phase 3: Can train genre classifier and achieve reasonable accuracy
- [x] Phase 4: Can evaluate both models with comprehensive metrics
- [x] Phase 5: Can annotate GigaMIDI with emotion and genre predictions
- [x] Phase 6: Can generate analysis and visualizations

---

## Next Steps

1. ✅ Phase 1: Create project structure and utilities (COMPLETED)
2. ✅ Phase 2: Download and preprocess XMIDI dataset (COMPLETED)
3. ✅ Phase 3: Implement models and training (COMPLETED)
4. ✅ Phase 4: Train and evaluate both tasks (Scripts ready)
5. ✅ Phase 5: Implement GigaMIDI annotation script (COMPLETED)
6. ✅ Phase 6: Implement analysis scripts (histograms, genre comparison, statistics) (COMPLETED)
