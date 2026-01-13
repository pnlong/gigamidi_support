# XMIDI Emotion and Genre Recognition

This project implements emotion recognition (11 classes) and genre recognition (6 classes) on the XMIDI dataset using MuseTok latents and MLP classifiers.

## Overview

**Tasks:**
- **Emotion Recognition**: 11 classes (exciting, warm, happy, romantic, funny, sad, angry, lazy, quiet, fear, magnificent)
- **Genre Recognition**: 6 classes (rock, pop, country, jazz, classical, folk)

**Architecture:**
- MuseTok for latent extraction (shared checkpoint from valence_arousal task)
- MLP with classification head for song-level prediction
- Mean pooling across bars for song-level features
- Cross-entropy loss for training

## Dataset

**XMIDI Dataset:**
- Filename format: `XMIDI_<Emotion>_<Genre>_<ID_len_8>.midi`
- Download from: https://drive.google.com/file/d/1qDkSH31x7jN8X-2RyzB9wuxGji4QxYyA/view

## Quick Start

See [PIPELINE.md](PIPELINE.md) for complete step-by-step instructions.

### 1. Download and Preprocess

```bash
# Download XMIDI dataset
python pretrain_model/download_xmidi.py

# Extract latents
python pretrain_model/preprocess_xmidi.py \
    --xmidi_dir /path/to/xmidi_data \
    --output_dir /path/to/latents \
    --gpu \
    --resume

# Prepare labels and splits
python pretrain_model/prepare_labels.py \
    --xmidi_dir /path/to/xmidi_data \
    --output_dir /path/to/labels
```

### 2. Train Models

```bash
# Train emotion classifier
python pretrain_model/train.py \
    --task emotion \
    --latents_dir /path/to/latents \
    --labels_path /path/to/labels/emotion_labels.json \
    --class_to_index_path /path/to/labels/emotion_to_index.json \
    --train_files /path/to/labels/train_files.txt \
    --valid_files /path/to/labels/val_files.txt \
    --num_classes 11 \
    --input_dim 128 \
    --gpu \
    --use_wandb

# Train genre classifier
python pretrain_model/train.py \
    --task genre \
    --latents_dir /path/to/latents \
    --labels_path /path/to/labels/genre_labels.json \
    --class_to_index_path /path/to/labels/genre_to_index.json \
    --train_files /path/to/labels/train_files.txt \
    --valid_files /path/to/labels/val_files.txt \
    --num_classes 6 \
    --input_dim 128 \
    --gpu \
    --use_wandb
```

### 3. Evaluate

```bash
# Evaluate emotion classifier
python pretrain_model/evaluate.py \
    --task emotion \
    --checkpoint_path /path/to/best_model.pt \
    --latents_dir /path/to/latents \
    --labels_path /path/to/labels/emotion_labels.json \
    --class_to_index_path /path/to/labels/emotion_to_index.json \
    --test_files /path/to/labels/test_files.txt \
    --num_classes 11 \
    --input_dim 128 \
    --gpu
```

## File Structure

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
│   ├── midi_utils.py          # MIDI processing
│   └── musetok_utils.py       # MuseTok integration
└── pretrain_model/
    ├── __init__.py
    ├── download_xmidi.py      # Download XMIDI dataset
    ├── preprocess_xmidi.py    # Extract latents from MIDI files
    ├── prepare_labels.py       # Extract labels, create splits
    ├── dataset.py              # PyTorch dataset
    ├── model.py                # EmotionGenreClassifier
    ├── train.py                # Training script
    └── evaluate.py             # Evaluation script
```

## Storage Directory Structure

```
<storage_dir>/
└── xmidi_emotion_genre/
    ├── checkpoints/
    │   └── trained_models/
    │       ├── emotion_classifier/
    │       └── genre_classifier/
    ├── xmidi_data/
    │   ├── latents/          # Preprocessed latents
    │   └── labels/           # Label files and splits
    └── evaluation_results/
        ├── emotion/
        └── genre/
```

## Dependencies

See `requirements.txt` for full list. Key dependencies:
- torch>=2.0.0
- symusic>=0.1.0
- scikit-learn>=1.3.0 (for classification metrics)
- gdown>=4.7.0 (for dataset download)

## Notes

- MuseTok checkpoint is shared from the `valence_arousal` task
- Input dimension is 128 (MuseTok latent dimension)
- Mean pooling is used to aggregate bar-level latents to song-level features
- Train/val/test splits are created with stratification to maintain class distribution
