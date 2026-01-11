# Revised Plan: Predicting Continuous Valence/Arousal Values for GigaMIDI-Extended

## Overview

This project aims to predict continuous Valence and Arousal values (range: -1 to 1) at the bar level for the GigaMIDI dataset using MuseTok latents. The pipeline consists of:

1. **Preprocessing EMOPIA**: Extract MuseTok latents and prepare continuous valence/arousal labels
2. **Training Emotion Recognition Model**: Train an MLP on EMOPIA to predict continuous valence/arousal
3. **Annotating GigaMIDI**: Apply the trained model to GigaMIDI to generate bar-level emotion annotations
4. **Analysis**: Analyze the predicted annotations across genres/styles

---

## Codebase Structure

```
valence_arousal/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── config/                            # Configuration files
│   ├── musetok_config.yaml           # MuseTok model configuration
│   └── training_config.yaml           # Training hyperparameters
├── pretrain_model/                    # EMOPIA preprocessing and training
│   ├── __init__.py
│   ├── preprocess_emopia.py          # Extract latents from EMOPIA MIDI files
│   ├── prepare_labels.py             # Convert EMOPIA labels to continuous VA
│   ├── dataset.py                    # PyTorch dataset for EMOPIA
│   ├── model.py                      # MLP model for continuous prediction
│   ├── train.py                      # Training script
│   └── evaluate.py                   # Evaluation script
├── annotate_gigamidi.py               # Annotate GigaMIDI with VA predictions (extracts latents on-the-fly, writes CSV directly)
├── analyze_emotion_annotations/        # Analysis and visualization
│   ├── __init__.py
│   ├── statistics.py                 # Compute statistics by genre/tag
│   ├── visualize.py                  # Plot valence/arousal curves
│   └── report.py                     # Generate analysis report
└── utils/                             # Shared utilities
    ├── __init__.py
    ├── musetok_utils.py              # MuseTok integration helpers
    ├── midi_utils.py                  # MIDI processing with symusic
    └── data_utils.py                   # Data loading/saving utilities
```

## Storage Directory

The pipeline uses a centralized storage directory for all large files. This directory is configured in `utils/data_utils.py` and can be set via:

1. **Environment variable**: `VALENCE_AROUSAL_STORAGE_DIR`
2. **Programmatically**: Call `utils.data_utils.set_storage_dir(path)`

**Storage Directory Structure**:
```
<STORAGE_DIR>/
├── checkpoints/
│   ├── musetok/                      # MuseTok pre-trained checkpoints
│   └── trained_models/              # Trained VA prediction models
├── emopia/
│   ├── latents/                      # Preprocessed EMOPIA latents
│   │   ├── train/
│   │   ├── valid/
│   │   └── test/
│   └── labels/                        # VA labels JSON files
└── gigamidi_annotations/              # GigaMIDI annotations
    └── annotations.csv
```

**Default Location**: `/path/to/storage/valence_arousal` (modify in `utils/data_utils.py`)

**Benefits**:
- Easy to manage large files in one location
- Can be on different filesystem (e.g., network storage)
- Simple to backup or move
- Clear separation from code

---

## Step 1: Environment Setup

### 1.1 Create Conda Environment
```bash
mamba create -n gigamidi python=3.10
mamba activate gigamidi
```

### 1.2 Install Dependencies
```bash
pip install symusic torch datasets gdown safetensors numpy pandas scikit-learn tqdm wandb
```

### 1.3 Download MuseTok Checkpoints
- Download best tokenizer weights from Google Drive (ID: `1HK534lEVdHYl3HMRkKvz8CWYliXRmOq_`)
- Extract to `valence_arousal/checkpoints/musetok/`
- Verify checkpoint structure matches MuseTok expectations

**Key Questions to Resolve:**
- How many codebook levels does the pre-trained MuseTok use? (affects latent dimension)
- Should we use the quantized codes (indices) or the continuous latent vectors?
  - **Recommendation**: Use continuous latents (`vae_latent`) for better gradient flow, but test both
- What is the latent dimension `d_vae_latent`? (typically 512 or 1024)

---

## Step 2: MIDI to MuseTok Latents Pipeline

### 2.1 Create Symusic-based MIDI Processing

**File**: `utils/midi_utils.py`

**Purpose**: Replace `miditoolkit` with `symusic` for MIDI parsing, maintaining compatibility with MuseTok's event format.

**Key Functions:**
- `load_midi_symusic(midi_path_or_bytes)`: Load MIDI using symusic
- `midi_to_events_symusic(score)`: Convert symusic Score to REMI events (similar to `midi2events.py`)
- `get_bar_positions(events)`: Extract bar boundaries from events

**Implementation Notes:**
- Replicate quantization logic from `musetok/data_processing/midi2events.py`
- Use `BEAT_RESOL = 480`, `TICK_RESOL = 40` (BEAT_RESOL // 12)
- Handle time signatures, tempo changes, note quantization
- Output format: `(bar_positions, events)` where events is list of `{'name': str, 'value': Any}`

### 2.2 Create MuseTok Latent Extraction Utility

**File**: `utils/musetok_utils.py`

**Purpose**: Provide a unified interface to extract MuseTok latents from MIDI files or REMI events.

**Key Functions:**
- `load_musetok_model(checkpoint_path, device)`: Load pre-trained MuseTok tokenizer
- `extract_latents_from_midi(midi_path_or_bytes, model, vocab)`: Full pipeline from MIDI to latents
- `extract_latents_from_events(events, bar_positions, model)`: Extract latents from REMI events
- `get_bar_level_latents(latents, bar_positions)`: Split latents by bar

**Implementation Details:**
- Use `MuseTokEncoder` from `musetok/encoding.py` as reference
- Input format: REMI events with bar positions
- Output format: `numpy.ndarray` of shape `(n_bars, d_vae_latent)`
- Handle variable-length sequences (padding/truncation)

**Key Questions:**
- Should we use `get_batch_latent` or `get_sampled_latent`?
  - **Recommendation**: `get_batch_latent` for consistency with training
- How to handle songs with >16 bars? (MuseTok processes in segments)
  - **Solution**: Process in overlapping segments and concatenate bar-level latents

---

## Step 3: EMOPIA Preprocessing

### 3.1 Extract MuseTok Latents from EMOPIA

**File**: `pretrain_model/preprocess_emopia.py`

**Purpose**: Preprocess EMOPIA MIDI files to extract MuseTok latents and save them for training.

**Pipeline:**
1. Load EMOPIA dataset structure
2. For each MIDI file:
   - Load MIDI with symusic
   - Convert to REMI events using `midi_to_events_symusic`
   - Extract MuseTok latents using `extract_latents_from_events`
   - Save latents as `.safetensors` or `.npy` files
3. Organize output: `data/emopia/latents/{split}/{filename}.safetensors`

**Data Structure:**
- Each file contains: `latents` array of shape `(n_bars, d_vae_latent)`
- Metadata: `bar_positions`, `n_bars`, `original_midi_path`

**Implementation Notes:**
- Use multiprocessing for parallel processing
- Handle errors gracefully (skip invalid MIDI files)
- Preserve train/valid/test splits from EMOPIA

### 3.2 Prepare Continuous Valence/Arousal Labels

**File**: `pretrain_model/prepare_labels.py`

**Purpose**: Convert EMOPIA categorical emotion labels to continuous valence/arousal values.

**Mapping Strategy:**
- **Happy**: Valence=+0.8, Arousal=+0.6
- **Angry**: Valence=-0.6, Arousal=+0.8
- **Sad**: Valence=-0.8, Arousal=-0.4
- **Relax**: Valence=+0.4, Arousal=-0.6

**Alternative Approaches:**
1. Use existing VA annotations if available in EMOPIA metadata
2. Use a mapping based on Russell's circumplex model
3. Fine-tune mapping based on validation performance

**Output Format:**
- JSON file: `{filename: {"valence": float, "arousal": float}}`
- Per-bar labels: If EMOPIA has per-bar annotations, create `{filename: {"valence": [float], "arousal": [float]}}`

**Key Questions:**
- Does EMOPIA have per-bar emotion annotations or only song-level?
  - **If song-level only**: Use same VA values for all bars (simplest approach)
  - **If per-bar available**: Use per-bar labels (more accurate but requires verification)

---

## Step 4: Training Continuous Valence/Arousal Model

### 4.1 Dataset Class

**File**: `pretrain_model/dataset.py`

**Purpose**: PyTorch dataset for loading EMOPIA latents and continuous labels.

**Key Features:**
- Load preprocessed latents from disk
- Load valence/arousal labels from JSON
- Handle variable-length sequences (padding/truncation)
- Support both song-level and bar-level labels

**Implementation:**
- Extend `CustomDataset` from `jingyue_latents/dataset.py`
- Modify to handle continuous labels (2D output: valence, arousal)
- Add support for per-bar predictions if using bar-level labels

### 4.2 Model Architecture

**File**: `pretrain_model/model.py`

**Purpose**: MLP model for continuous valence/arousal prediction.

**Key Modifications from `jingyue_latents/model.py`:**
- Change output dimension from `N_EMOTION_CLASSES` (4) to `2` (valence, arousal)
- Remove softmax activation (continuous outputs)
- Add optional tanh activation to constrain outputs to [-1, 1] range

**Architecture:**
```python
class ValenceArousalMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim=None, use_tanh=True):
        # input_dim = d_vae_latent (from MuseTok)
        # hidden_dim = input_dim // 2 (default)
        # output_dim = 2 (valence, arousal)
```

### 4.3 Training Script

**File**: `pretrain_model/train.py`

**Purpose**: Train the MLP on EMOPIA to predict continuous valence/arousal.

**Key Modifications from `jingyue_latents/train.py`:**
- Replace `CrossEntropyLoss` with `MSELoss` or `SmoothL1Loss`
- Modify evaluation metrics:
  - Mean Absolute Error (MAE) for valence and arousal
  - Mean Squared Error (MSE)
  - Correlation coefficient
- Update accuracy computation (not applicable for regression)

**Loss Function:**
- **Option 1**: `MSELoss` - standard regression loss
- **Option 2**: `SmoothL1Loss` - less sensitive to outliers
- **Option 3**: Combined loss: `MSELoss + L1Loss` (Huber-like)

**Hyperparameters:**
- Learning rate: 1e-4 to 1e-3
- Batch size: 32-128 (depending on GPU memory)
- Epochs: 50-100 (with early stopping)
- Weight decay: 1e-5

### 4.4 Evaluation Script

**File**: `pretrain_model/evaluate.py`

**Purpose**: Evaluate trained model on EMOPIA test set.

**Metrics:**
- MAE, MSE, RMSE for valence and arousal separately
- Correlation coefficients
- Visualization: scatter plots of predicted vs. actual VA values

---

## Step 5: GigaMIDI Annotation Pipeline

### 5.1 Annotate GigaMIDI

**File**: `annotate_gigamidi.py`

**Purpose**: Apply trained model to GigaMIDI to predict valence/arousal for each bar.

**Pipeline:**
1. Load trained model checkpoint
2. Stream GigaMIDI dataset
3. For each song:
   - Extract latents on-the-fly (no intermediate storage)
   - Predict valence/arousal for each bar using trained model
   - Write predictions directly to CSV (one row per bar)

**Output Format:**
- CSV file with four columns: `md5`, `bar_number`, `valence`, `arousal`
- Written incrementally (one row at a time) to avoid losing progress
- Supports resume by skipping already-processed (md5, bar_number) pairs

**Note**: Latents are extracted on-the-fly within this script, so no separate `extract_latents.py` is needed.
CSV is written directly, so no separate `save_annotations.py` is needed.

**Note**: `annotate_gigamidi.py` writes CSV directly, so a separate save script is not necessary.
If format conversion is needed, it can be added later.

**File**: `save_annotations.py` (optional, not created by default)

**Output Structure:**
```
<STORAGE_DIR>/gigamidi_annotations/
└── annotations.csv                  # Main annotations file (md5, valence, arousal)
```

**Format:**
- **CSV**: Simple, human-readable format with three columns
- Written incrementally to avoid losing progress
- Supports resume functionality

---

## Step 6: Analysis and Visualization

### 6.1 Statistics

**File**: `analyze_emotion_annotations/statistics.py`

**Purpose**: Compute statistics about predicted valence/arousal values.

**Analyses:**
- Average valence/arousal by genre (`music_styles_curated`)
- Distribution of VA values across the dataset
- Correlation between VA and other metadata (tempo, note density, etc.)
- Temporal patterns: How VA changes within songs

### 6.2 Visualization

**File**: `analyze_emotion_annotations/visualize.py`

**Purpose**: Create visualizations of emotion annotations.

**Plots:**
- Valence/arousal scatter plots by genre
- Time-series plots: VA curves for example songs
- Heatmaps: VA distribution across genres
- Box plots: VA statistics by genre

### 6.3 Report Generation

**File**: `analyze_emotion_annotations/report.py`

**Purpose**: Generate a comprehensive analysis report.

**Contents:**
- Summary statistics
- Genre-wise analysis
- Example visualizations
- Insights and observations

---

## Implementation Order

1. **Phase 1: Setup & Utilities** (Week 1)
   - Set up environment and dependencies
   - Implement `utils/midi_utils.py` (symusic-based MIDI processing)
   - Implement `utils/musetok_utils.py` (MuseTok integration)
   - Test MIDI → Events → Latents pipeline on sample files

2. **Phase 2: EMOPIA Preprocessing** (Week 1-2)
   - Implement `preprocess_emopia.py`
   - Implement `prepare_labels.py`
   - Extract all EMOPIA latents
   - Verify data quality and format

3. **Phase 3: Model Training** (Week 2-3)
   - Implement `dataset.py`, `model.py`, `train.py`
   - Train initial model on EMOPIA
   - Implement `evaluate.py` and validate model performance
   - Iterate on hyperparameters and architecture

4. **Phase 4: GigaMIDI Annotation** (Week 3-4)
   - Implement `annotate_gigamidi.py` (extracts latents on-the-fly, writes CSV directly)
   - Run annotation pipeline on GigaMIDI (streaming mode)

5. **Phase 5: Analysis** (Week 4)
   - Implement analysis scripts
   - Generate visualizations
   - Create analysis report

---

## Key Technical Decisions

### MuseTok Latent Extraction
- **Use continuous latents** (`vae_latent`) rather than quantized codes for better gradient flow
- **Process in segments** for songs >16 bars, then concatenate bar-level latents
- **Batch processing** for efficiency when processing multiple files

### Label Mapping (EMOPIA → VA)
- **Start with simple mapping** based on Russell's circumplex model
- **Validate mapping** by checking if predictions align with human intuition
- **Consider fine-tuning** if per-bar annotations are available

### Loss Function
- **Start with SmoothL1Loss** (less sensitive to outliers than MSE)
- **Add L2 regularization** to prevent overfitting
- **Monitor both MAE and correlation** during training

### GigaMIDI Processing
- **Use streaming mode** to avoid downloading entire dataset
- **Process on-the-fly** to avoid storage overhead
- **Batch predictions** for efficiency (process multiple bars at once)

---

## Open Questions & Future Work

1. **EMOPIA Labels**: Does EMOPIA have per-bar emotion annotations, or only song-level?
2. **Codebook Levels**: How many codebook levels does the pre-trained MuseTok use?
3. **Latent Dimension**: What is the exact `d_vae_latent` dimension?
4. **Validation**: How to validate that predicted VA values are meaningful? (human evaluation?)
5. **Fine-tuning**: Should we fine-tune the mapping based on validation performance?
6. **Tension**: Future work to add tension prediction (mentioned in collaborator note)

---

## Dependencies Summary

- **symusic**: Fast MIDI parsing
- **torch**: Deep learning framework
- **datasets**: HuggingFace datasets library for GigaMIDI
- **gdown**: Download MuseTok checkpoints
- **safetensors/numpy**: Efficient tensor storage
- **pandas**: Data manipulation and analysis
- **scikit-learn**: Evaluation metrics
- **wandb**: Experiment tracking (optional)

---

## Success Criteria

1. Successfully extract MuseTok latents from EMOPIA and GigaMIDI
2. Train a model that achieves reasonable MAE (<0.3) on EMOPIA validation set
3. Generate bar-level VA annotations for entire GigaMIDI dataset
4. Produce analysis report with meaningful insights about emotion distribution across genres

---

## README: Running the Full Pipeline

The following instructions describe how to run the complete pipeline from start to finish.

### Prerequisites

1. **Set up environment**:
   ```bash
   mamba create -n gigamidi python=3.10
   mamba activate gigamidi
   cd valence_arousal
   pip install -r requirements.txt
   ```

2. **Configure storage directory**:
   ```python
   # Option 1: Set environment variable
   export VALENCE_AROUSAL_STORAGE_DIR="/path/to/your/storage"
   
   # Option 2: Modify utils/data_utils.py directly
   # Change STORAGE_DIR = "/path/to/storage/valence_arousal"
   ```

3. **Download MuseTok checkpoints**:
   ```bash
   python -c "
   import gdown
   from utils.data_utils import MUSETOK_CHECKPOINT_DIR, ensure_dir
   ensure_dir(MUSETOK_CHECKPOINT_DIR)
   gdown.download('https://drive.google.com/uc?id=1HK534lEVdHYl3HMRkKvz8CWYliXRmOq_', 
                  f'{MUSETOK_CHECKPOINT_DIR}/musetok_checkpoint.zip')
   "
   # Then extract the zip file
   unzip checkpoints/musetok/musetok_checkpoint.zip -d checkpoints/musetok/
   ```

### Step 1: Preprocess EMOPIA

**Extract MuseTok latents from EMOPIA MIDI files**:
```bash
python pretrain_model/preprocess_emopia.py \
    --emopia_dir /path/to/EMOPIA/dataset \
    --output_dir <STORAGE_DIR>/emopia/latents \
    --checkpoint_path <STORAGE_DIR>/checkpoints/musetok/model.pt \
    --vocab_path ../musetok/data/dictionary.pkl \
    --device cuda \
    --num_workers 4
```

**Prepare continuous VA labels**:
```bash
python pretrain_model/prepare_labels.py \
    --emopia_dir /path/to/EMOPIA/dataset \
    --output_path <STORAGE_DIR>/emopia/labels/va_labels.json
```

### Step 2: Train VA Prediction Model

**Train the model on EMOPIA**:
```bash
python pretrain_model/train.py \
    --latents_dir <STORAGE_DIR>/emopia/latents \
    --labels_path <STORAGE_DIR>/emopia/labels/va_labels.json \
    --output_dir <STORAGE_DIR>/checkpoints/trained_models \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --epochs 100 \
    --device cuda \
    --use_wandb  # Optional: for experiment tracking
```

**Evaluate the trained model**:
```bash
python pretrain_model/evaluate.py \
    --checkpoint_path <STORAGE_DIR>/checkpoints/trained_models/best_model.pt \
    --latents_dir <STORAGE_DIR>/emopia/latents/test \
    --labels_path <STORAGE_DIR>/emopia/labels/va_labels.json \
    --output_dir <STORAGE_DIR>/evaluation_results
```

### Step 3: Annotate GigaMIDI

**Apply trained model to GigaMIDI** (streaming mode):
```bash
python annotate_gigamidi.py \
    --model_path <STORAGE_DIR>/checkpoints/trained_models/best_model.pt \
    --checkpoint_path <STORAGE_DIR>/checkpoints/musetok/model.pt \
    --vocab_path ../musetok/data/dictionary.pkl \
    --output_dir <STORAGE_DIR>/gigamidi_annotations \
    --batch_size 64 \
    --device cuda \
    --streaming  # Use streaming mode to avoid downloading entire dataset
```

This will:
1. Stream GigaMIDI dataset (no local download required)
2. Extract latents on-the-fly for each song
3. Predict VA values (averaged across bars) for each song
4. Save annotations incrementally to `<STORAGE_DIR>/gigamidi_annotations/annotations.csv`
5. Support resume by skipping already-processed songs

### Step 4: Analyze Annotations

**Compute statistics**:
```bash
python analyze_emotion_annotations/statistics.py \
    --annotations_path <STORAGE_DIR>/gigamidi_annotations/annotations.csv \
    --output_dir <STORAGE_DIR>/analysis/statistics
```

**Generate visualizations**:
```bash
python analyze_emotion_annotations/visualize.py \
    --annotations_path <STORAGE_DIR>/gigamidi_annotations/annotations.csv \
    --output_dir <STORAGE_DIR>/analysis/visualizations
```

**Generate comprehensive report**:
```bash
python analyze_emotion_annotations/report.py \
    --annotations_path <STORAGE_DIR>/gigamidi_annotations/annotations.csv \
    --statistics_dir <STORAGE_DIR>/analysis/statistics \
    --visualizations_dir <STORAGE_DIR>/analysis/visualizations \
    --output_path <STORAGE_DIR>/analysis/report.html
```

### Quick Start (All Steps)

For convenience, you can run the entire pipeline with a single script:

```bash
# Create a script: run_full_pipeline.sh
#!/bin/bash
set -e

STORAGE_DIR="${VALENCE_AROUSAL_STORAGE_DIR:-/path/to/storage/valence_arousal}"
EMOPIA_DIR="/path/to/EMOPIA/dataset"

# Step 1: Preprocess EMOPIA
echo "Step 1: Preprocessing EMOPIA..."
python pretrain_model/preprocess_emopia.py --emopia_dir $EMOPIA_DIR
python pretrain_model/prepare_labels.py --emopia_dir $EMOPIA_DIR

# Step 2: Train model
echo "Step 2: Training model..."
python pretrain_model/train.py

# Step 3: Annotate GigaMIDI
echo "Step 3: Annotating GigaMIDI..."
python process_gigamidi/annotate_gigamidi.py --streaming

# Step 4: Analyze
echo "Step 4: Analyzing annotations..."
python analyze_emotion_annotations/statistics.py
python analyze_emotion_annotations/visualize.py
python analyze_emotion_annotations/report.py

echo "Pipeline complete! Results in $STORAGE_DIR"
```

### Troubleshooting

**Common Issues**:

1. **Storage directory not set**: Make sure to set `VALENCE_AROUSAL_STORAGE_DIR` or modify `utils/data_utils.py`
2. **Out of memory**: Reduce `--batch_size` or `--num_workers`
3. **MuseTok checkpoint not found**: Verify checkpoint path and that it's extracted correctly
4. **CUDA out of memory**: Use `--device cpu` or reduce batch size

**Verification**:

After each step, verify outputs:
- Step 1: Check that latents files exist in `<STORAGE_DIR>/emopia/latents/`
- Step 2: Check that model checkpoint exists in `<STORAGE_DIR>/checkpoints/trained_models/`
- Step 3: Check that annotations file exists in `<STORAGE_DIR>/gigamidi_annotations/`
- Step 4: Check that analysis outputs exist in `<STORAGE_DIR>/analysis/`

