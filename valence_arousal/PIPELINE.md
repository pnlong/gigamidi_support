# Valence/Arousal Prediction Pipeline

Complete step-by-step guide for training and applying the continuous valence/arousal prediction model.

## Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Storage directory**: Default is `/deepfreeze/pnlong/gigamidi`
   - Can be overridden with `export VALENCE_AROUSAL_STORAGE_DIR=/path/to/storage`

3. **MuseTok checkpoints**: Should be extracted to:
   ```
   /deepfreeze/pnlong/gigamidi/checkpoints/musetok/
   └── best_tokenizer.pt  (used for encoding/extracting latents)
   ```

---

## Phase 1: Setup

### 1.1 Download MuseTok Checkpoints (if not already done)

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

Then extract the zip file and place `best_tokenizer/model.pt` directly in `checkpoints/musetok/` as `best_tokenizer.pt`.

---

## Phase 2: EMOPIA Preprocessing

### 2.1 Extract Latents from EMOPIA

#### Option A: Process Edited EMOPIA (jingyue's version - .pkl files)

```bash
python pretrain_model/preprocess_emopia.py \
    --emopia_dir /deepfreeze/user_shares/jingyue/EMOPIA_data \
    --output_dir /deepfreeze/pnlong/gigamidi/emopia/latents/jingyue \
    --gpu \
    --batch_size 8 \
    --resume
```

#### Option B: Process EMOPIA+ (using REMI directory)

```bash
python pretrain_model/preprocess_emopia.py \
    --emopia_dir /deepfreeze/pnlong/gigamidi/emopia/emopia_plus \
    --output_dir /deepfreeze/pnlong/gigamidi/emopia/latents/emopia_plus \
    --use_remi_dir \
    --split train \
    --gpu \
    --batch_size 8 \
    --resume
```

#### Option C: Process EMOPIA+ (using MIDI files, full pipeline)

```bash
python pretrain_model/preprocess_emopia.py \
    --emopia_dir /deepfreeze/pnlong/gigamidi/emopia/emopia_plus \
    --output_dir /deepfreeze/pnlong/gigamidi/emopia/latents/emopia_plus \
    --split train \
    --gpu \
    --batch_size 8 \
    --resume
```

**Note**: 
- Use `--resume` to skip already-processed files
- For EMOPIA+ with splits: 
  - If `--split` is not provided, all splits (train/valid/test) will be processed automatically
  - To process a specific split only: `--split train`, `--split valid`, or `--split test`
- Output structure will mirror input structure (preserves train/valid/test splits)

### 2.2 Prepare Continuous VA Labels

#### For Edited EMOPIA (extracts Q1-Q4 from filenames)

```bash
python pretrain_model/prepare_labels.py \
    --emopia_dir /deepfreeze/user_shares/jingyue/EMOPIA_data \
    --output_path /deepfreeze/pnlong/gigamidi/emopia/labels/jingyue_va_labels.json
```

#### For EMOPIA+ (uses metadata if available, falls back to filenames)

```bash
python pretrain_model/prepare_labels.py \
    --emopia_dir /deepfreeze/pnlong/gigamidi/emopia/emopia_plus \
    --output_path /deepfreeze/pnlong/gigamidi/emopia/labels/emopia_plus_va_labels.json
```

**Optional**: Create per-bar labels (requires latents to be preprocessed first):

```bash
python pretrain_model/prepare_labels.py \
    --emopia_dir /deepfreeze/user_shares/jingyue/EMOPIA_data \
    --output_path /deepfreeze/pnlong/gigamidi/emopia/labels/jingyue_va_labels_per_bar.json \
    --per_bar \
    --latents_dir /deepfreeze/pnlong/gigamidi/emopia/latents/jingyue
```

---

## Phase 3: Model Training

### 3.1 Train Valence/Arousal Prediction Model

```bash
python pretrain_model/train.py \
    --latents_dir /deepfreeze/pnlong/gigamidi/emopia/latents/jingyue \
    --labels_path /deepfreeze/pnlong/gigamidi/emopia/labels/jingyue_va_labels.json \
    --train_split train \
    --valid_split valid \
    --input_dim 128 \
    --hidden_dim 64 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --epochs 100 \
    --max_seq_len 42 \
    --loss_type smooth_l1 \
    --gpu \
    --num_workers 4 \
    --output_dir /deepfreeze/pnlong/gigamidi/checkpoints/trained_models \
    --model_name va_mlp \
    --use_wandb \
    --wandb_project valence_arousal \
    --early_stopping \
    --early_stopping_tolerance 10
```

**Key Arguments**:
- `--pool`: Add this flag to pool (average) across bars before model (song-level prediction)
- `--resume`: Resume training from best checkpoint
- `--use_wandb`: Enable Weights & Biases logging (optional but recommended)

### 3.2 Evaluate Trained Model

```bash
python pretrain_model/evaluate.py \
    --checkpoint_path /deepfreeze/pnlong/gigamidi/checkpoints/trained_models/va_mlp/checkpoints/best_model.pt \
    --latents_dir /deepfreeze/pnlong/gigamidi/emopia/latents/jingyue \
    --labels_path /deepfreeze/pnlong/gigamidi/emopia/labels/jingyue_va_labels.json \
    --test_split test \
    --input_dim 128 \
    --hidden_dim 64 \
    --batch_size 32 \
    --max_seq_len 42 \
    --gpu \
    --num_workers 4 \
    --output_dir /deepfreeze/pnlong/gigamidi/evaluation_results
```

This will generate:
- `metrics.csv`: MAE, MSE, correlation coefficients
- `scatter_plots.png`: Predicted vs. actual scatter plots

---

## Phase 4: GigaMIDI Annotation

### 4.1 Annotate GigaMIDI Dataset

```bash
python annotate_gigamidi.py \
    --model_path /deepfreeze/pnlong/gigamidi/checkpoints/trained_models/va_mlp/checkpoints/best_model.pt \
    --input_dim 128 \
    --hidden_dim 64 \
    --gpu \
    --streaming \
    --split train \
    --output_path /deepfreeze/pnlong/gigamidi/gigamidi_annotations/annotations.csv \
    --resume
```

**Key Arguments**:
- `--streaming`: Use streaming mode (recommended, avoids downloading entire dataset)
- `--resume`: Resume from existing CSV (skips already-processed songs)
- `--max_samples N`: Process only N samples (useful for testing)
- `--split`: Dataset split to process (`train`, `valid`, `test`)

**Output**: CSV file with columns: `md5`, `bar_number`, `valence`, `arousal`
- Each row represents one bar of a song
- Written incrementally to avoid losing progress

---

## Phase 5: Analysis

### 5.1 Plot Histograms

```bash
python analyze_emotion_annotations/plot_histograms.py \
    --annotations_path /deepfreeze/pnlong/gigamidi/gigamidi_annotations/annotations.csv \
    --output_path /deepfreeze/pnlong/gigamidi/gigamidi_annotations/histograms.png \
    --bins 50
```

### 5.2 Plot Boxplots

```bash
python analyze_emotion_annotations/plot_boxplots.py \
    --annotations_path /deepfreeze/pnlong/gigamidi/gigamidi_annotations/annotations.csv \
    --output_path /deepfreeze/pnlong/gigamidi/gigamidi_annotations/boxplots.png
```

### 5.3 Plot by Genre

```bash
python analyze_emotion_annotations/plot_by_genre.py \
    --annotations_path /deepfreeze/pnlong/gigamidi/gigamidi_annotations/annotations.csv \
    --output_path /deepfreeze/pnlong/gigamidi/gigamidi_annotations/genre_boxplots.png \
    --top_n 10 \
    --split train
```

### 5.4 Plot Song Curves

```bash
python analyze_emotion_annotations/plot_song_curves.py \
    --annotations_path /deepfreeze/pnlong/gigamidi/gigamidi_annotations/annotations.csv \
    --output_path /deepfreeze/pnlong/gigamidi/gigamidi_annotations/song_curves.png \
    --n_examples 5 \
    --seed 42
```

### 5.5 Print Statistics

```bash
python analyze_emotion_annotations/print_statistics.py \
    --annotations_path /deepfreeze/pnlong/gigamidi/gigamidi_annotations/annotations.csv
```

---

## Quick Reference: File Structure

```
/deepfreeze/pnlong/gigamidi/
├── checkpoints/
│   ├── musetok/
│   │   └── best_tokenizer.pt
│   └── trained_models/
│       └── va_mlp/
│           └── checkpoints/
│               ├── best_model.pt
│               └── best_optimizer.pt
├── emopia/
│   ├── emopia_plus/          # EMOPIA+ dataset (to be downloaded)
│   ├── latents/
│   │   ├── jingyue/          # Latents from edited EMOPIA
│   │   └── emopia_plus/      # Latents from EMOPIA+
│   └── labels/
│       ├── jingyue_va_labels.json
│       └── emopia_plus_va_labels.json
└── gigamidi_annotations/
    └── annotations.csv        # Bar-level VA predictions

External:
/deepfreeze/user_shares/jingyue/EMOPIA_data/  # Edited EMOPIA (direct path)
```

---

## Common Workflows

### Workflow 1: Train on Edited EMOPIA, Annotate GigaMIDI

```bash
# 1. Preprocess EMOPIA
python pretrain_model/preprocess_emopia.py \
    --emopia_dir /deepfreeze/user_shares/jingyue/EMOPIA_data \
    --output_dir /deepfreeze/pnlong/gigamidi/emopia/latents/jingyue \
    --resume

# 2. Prepare labels
python pretrain_model/prepare_labels.py \
    --emopia_dir /deepfreeze/user_shares/jingyue/EMOPIA_data \
    --output_path /deepfreeze/pnlong/gigamidi/emopia/labels/jingyue_va_labels.json

# 3. Train model
python pretrain_model/train.py \
    --latents_dir /deepfreeze/pnlong/gigamidi/emopia/latents/jingyue \
    --labels_path /deepfreeze/pnlong/gigamidi/emopia/labels/jingyue_va_labels.json \
    --train_split train \
    --valid_split valid \
    --input_dim 128 \
    --batch_size 32 \
    --epochs 100 \
    --use_wandb

# 4. Evaluate
python pretrain_model/evaluate.py \
    --checkpoint_path /deepfreeze/pnlong/gigamidi/checkpoints/trained_models/va_mlp/checkpoints/best_model.pt \
    --latents_dir /deepfreeze/pnlong/gigamidi/emopia/latents/jingyue \
    --labels_path /deepfreeze/pnlong/gigamidi/emopia/labels/jingyue_va_labels.json \
    --test_split test

# 5. Annotate GigaMIDI
python annotate_gigamidi.py \
    --model_path /deepfreeze/pnlong/gigamidi/checkpoints/trained_models/va_mlp/checkpoints/best_model.pt \
    --input_dim 128 \
    --streaming \
    --resume
```

### Workflow 2: Train on EMOPIA+, Annotate GigaMIDI

```bash
# 1. Preprocess EMOPIA+ (for each split)
for split in train valid test; do
    python pretrain_model/preprocess_emopia.py \
        --emopia_dir /deepfreeze/pnlong/gigamidi/emopia/emopia_plus \
        --output_dir /deepfreeze/pnlong/gigamidi/emopia/latents/emopia_plus \
        --use_remi_dir \
        --split $split \
        --resume
done

# 2. Prepare labels
python pretrain_model/prepare_labels.py \
    --emopia_dir /deepfreeze/pnlong/gigamidi/emopia/emopia_plus \
    --output_path /deepfreeze/pnlong/gigamidi/emopia/labels/emopia_plus_va_labels.json

# 3. Train model
python pretrain_model/train.py \
    --latents_dir /deepfreeze/pnlong/gigamidi/emopia/latents/emopia_plus \
    --labels_path /deepfreeze/pnlong/gigamidi/emopia/labels/emopia_plus_va_labels.json \
    --train_split train \
    --valid_split valid \
    --input_dim 128 \
    --batch_size 32 \
    --epochs 100 \
    --use_wandb

# 4. Evaluate
python pretrain_model/evaluate.py \
    --checkpoint_path /deepfreeze/pnlong/gigamidi/checkpoints/trained_models/va_mlp/checkpoints/best_model.pt \
    --latents_dir /deepfreeze/pnlong/gigamidi/emopia/latents/emopia_plus \
    --labels_path /deepfreeze/pnlong/gigamidi/emopia/labels/emopia_plus_va_labels.json \
    --test_split test

# 5. Annotate GigaMIDI
python annotate_gigamidi.py \
    --model_path /deepfreeze/pnlong/gigamidi/checkpoints/trained_models/va_mlp/checkpoints/best_model.pt \
    --input_dim 128 \
    --streaming \
    --resume
```

---

## Tips

1. **Resume Processing**: Always use `--resume` flag to skip already-processed files
2. **Testing**: Use `--max_samples N` in `annotate_gigamidi.py` to test on a small subset first
3. **Monitoring**: Use `--use_wandb` during training to track metrics in real-time
4. **Device**: Omit `--gpu` flag to use CPU instead of GPU
5. **Batch Processing**: `--batch_size` in `preprocess_emopia.py` is currently unused (processes sequentially)
6. **DataLoader Workers**: `--num_workers` in `train.py` and `evaluate.py` is safe to use with CUDA (uses DataLoader, not multiprocessing Pool)

---

## Troubleshooting

- **Out of Memory**: Reduce `--batch_size` in training/evaluation scripts
- **CUDA Multiprocessing Error**: `preprocess_emopia.py` no longer uses multiprocessing (processes sequentially) to avoid CUDA re-initialization issues
- **Checkpoint Not Found**: Verify MuseTok checkpoint is at `checkpoints/musetok/best_tokenizer.pt`
- **File Not Found**: Check that EMOPIA directories exist and paths are correct
- **Symlink Errors**: Use direct paths instead of symlinks (see `data_utils.py` for EMOPIA_JINGYUE_DIR)
