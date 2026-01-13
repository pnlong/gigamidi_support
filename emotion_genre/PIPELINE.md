# XMIDI Emotion and Genre Recognition Pipeline

Complete step-by-step guide for training emotion recognition (11 classes) and genre recognition (6 classes) models on the XMIDI dataset.

## Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Storage directory**: Default is `/deepfreeze/pnlong/gigamidi`
   - Can be overridden with `export XMIDI_STORAGE_DIR=/path/to/storage`

3. **MuseTok checkpoints**: Should be available from the valence_arousal task:
   ```
   /deepfreeze/pnlong/gigamidi/musetok/
   └── best_tokenizer.pt  (used for encoding/extracting latents)
   ```

---

## Phase 1: Dataset Download and Preprocessing

### 1.1 Download XMIDI Dataset

```bash
cd emotion_genre
python pretrain_model/download_xmidi.py \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data
```

**Note**: After download, extract the zip file. The dataset should contain MIDI files with naming format: `XMIDI_<Emotion>_<Genre>_<ID_len_8>.midi`

### 1.2 Extract Latents from XMIDI

```bash
python pretrain_model/preprocess_xmidi.py \
    --xmidi_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --gpu \
    --resume
```

**Key Arguments**:
- `--resume`: Skip already-processed files
- `--gpu`: Use GPU for faster processing

### 1.3 Prepare Labels and Create Splits

```bash
python pretrain_model/prepare_labels.py \
    --xmidi_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels \
    --test_size 0.1 \
    --val_size 0.1
```

This creates:
- `emotion_labels.json`: Maps filename to emotion class index
- `genre_labels.json`: Maps filename to genre class index
- `emotion_to_index.json`: Maps emotion string to index (0-10)
- `genre_to_index.json`: Maps genre string to index (0-5)
- `train_files.txt`, `val_files.txt`, `test_files.txt`: File lists for each split

**Emotion classes** (11 total): exciting, warm, happy, romantic, funny, sad, angry, lazy, quiet, fear, magnificent

**Genre classes** (6 total): rock, pop, country, jazz, classical, folk

---

## Phase 2: Model Training

### 2.1 Train Emotion Classifier

```bash
python pretrain_model/train.py \
    --task emotion \
    --latents_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_labels.json \
    --class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_to_index.json \
    --train_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/train_files.txt \
    --valid_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/val_files.txt \
    --num_classes 11 \
    --input_dim 128 \
    --hidden_dim 64 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --epochs 100 \
    --dropout 0.1 \
    --gpu \
    --num_workers 4 \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models \
    --model_name emotion_classifier \
    --use_wandb \
    --wandb_project xmidi_emotion_genre \
    --early_stopping \
    --early_stopping_tolerance 10
```

### 2.2 Train Genre Classifier

```bash
python pretrain_model/train.py \
    --task genre \
    --latents_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/genre_labels.json \
    --class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/genre_to_index.json \
    --train_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/train_files.txt \
    --valid_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/val_files.txt \
    --num_classes 6 \
    --input_dim 128 \
    --hidden_dim 64 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --epochs 100 \
    --dropout 0.1 \
    --gpu \
    --num_workers 4 \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models \
    --model_name genre_classifier \
    --use_wandb \
    --wandb_project xmidi_emotion_genre \
    --early_stopping \
    --early_stopping_tolerance 10
```

**Key Arguments**:
- `--task`: `emotion` or `genre`
- `--resume`: Resume training from best checkpoint
- `--use_wandb`: Enable wandb logging

---

## Phase 3: Model Evaluation

### 3.1 Evaluate Emotion Classifier

```bash
python pretrain_model/evaluate.py \
    --task emotion \
    --checkpoint_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models/emotion_classifier/checkpoints/best_model.pt \
    --latents_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_labels.json \
    --class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_to_index.json \
    --test_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/test_files.txt \
    --num_classes 11 \
    --input_dim 128 \
    --hidden_dim 64 \
    --batch_size 32 \
    --gpu \
    --num_workers 4 \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/evaluation_results/emotion
```

### 3.2 Evaluate Genre Classifier

```bash
python pretrain_model/evaluate.py \
    --task genre \
    --checkpoint_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models/genre_classifier/checkpoints/best_model.pt \
    --latents_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/genre_labels.json \
    --class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/genre_to_index.json \
    --test_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/test_files.txt \
    --num_classes 6 \
    --input_dim 128 \
    --hidden_dim 64 \
    --batch_size 32 \
    --gpu \
    --num_workers 4 \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/evaluation_results/genre
```

**Output**: Each evaluation generates:
- `metrics.csv`: Overall accuracy, F1-score (macro/weighted), Precision, Recall, per-class metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `classification_report.txt`: Detailed per-class classification report

---

## Quick Reference: File Structure

```
/deepfreeze/pnlong/gigamidi/
├── musetok/
│   └── best_tokenizer.pt  (shared MuseTok checkpoint)
└── xmidi_emotion_genre/
    ├── checkpoints/
    │   └── trained_models/
    │       ├── emotion_classifier/
    │       │   └── checkpoints/
    │       │       ├── best_model.pt
    │       │       └── best_optimizer.pt
    │       └── genre_classifier/
    │           └── checkpoints/
    │               ├── best_model.pt
    │               └── best_optimizer.pt
    ├── xmidi_data/
    │   ├── latents/          # Preprocessed latents (.safetensors files)
    │   └── labels/
    │       ├── emotion_labels.json
    │       ├── genre_labels.json
    │       ├── emotion_to_index.json
    │       ├── genre_to_index.json
    │       ├── train_files.txt
    │       ├── val_files.txt
    │       └── test_files.txt
    └── evaluation_results/
        ├── emotion/
        │   ├── metrics.csv
        │   ├── confusion_matrix.png
        │   └── classification_report.txt
        └── genre/
            ├── metrics.csv
            ├── confusion_matrix.png
            └── classification_report.txt
```

---

## Common Workflows

### Workflow 1: Complete Pipeline (Emotion + Genre)

```bash
# 1. Download and extract XMIDI dataset
python pretrain_model/download_xmidi.py

# 2. Preprocess XMIDI (extract latents)
python pretrain_model/preprocess_xmidi.py \
    --xmidi_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --gpu \
    --resume

# 3. Prepare labels and splits
python pretrain_model/prepare_labels.py \
    --xmidi_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels

# 4. Train emotion classifier
python pretrain_model/train.py \
    --task emotion \
    --latents_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_labels.json \
    --class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_to_index.json \
    --train_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/train_files.txt \
    --valid_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/val_files.txt \
    --num_classes 11 \
    --input_dim 128 \
    --batch_size 32 \
    --epochs 100 \
    --use_wandb

# 5. Train genre classifier
python pretrain_model/train.py \
    --task genre \
    --latents_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/genre_labels.json \
    --class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/genre_to_index.json \
    --train_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/train_files.txt \
    --valid_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/val_files.txt \
    --num_classes 6 \
    --input_dim 128 \
    --batch_size 32 \
    --epochs 100 \
    --use_wandb

# 6. Evaluate both models
python pretrain_model/evaluate.py \
    --task emotion \
    --checkpoint_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models/emotion_classifier/checkpoints/best_model.pt \
    --latents_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_labels.json \
    --class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_to_index.json \
    --test_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/test_files.txt \
    --num_classes 11 \
    --input_dim 128

python pretrain_model/evaluate.py \
    --task genre \
    --checkpoint_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models/genre_classifier/checkpoints/best_model.pt \
    --latents_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/genre_labels.json \
    --class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/genre_to_index.json \
    --test_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/test_files.txt \
    --num_classes 6 \
    --input_dim 128
```

---

## Tips

1. **Resume Processing**: Always use `--resume` flag to skip already-processed files
2. **Monitoring**: Use `--use_wandb` during training to track metrics in real-time
3. **Device**: Omit `--gpu` flag to use CPU instead of GPU
4. **Input Dimension**: MuseTok latents are 128-dimensional (d_vae_latent=128)
5. **Mean Pooling**: The dataset automatically pools latents across bars for song-level prediction
6. **Stratified Splits**: The prepare_labels script creates stratified splits to maintain class distribution

---

## Troubleshooting

- **Out of Memory**: Reduce `--batch_size` in training/evaluation scripts
- **Checkpoint Not Found**: Verify MuseTok checkpoint is available from valence_arousal task
- **File Not Found**: Check that XMIDI directories exist and paths are correct
- **Class Imbalance**: Check confusion matrix to see if certain classes are underrepresented
- **Low Accuracy**: Try adjusting learning rate, hidden dimension, or dropout rate
