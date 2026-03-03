# XMIDI Emotion and Genre Recognition Pipeline

Complete step-by-step guide for training emotion recognition (11 classes) and genre recognition (6 classes) models on the XMIDI dataset.

## Prerequisites

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Storage directory**: Default is `/deepfreeze/pnlong/gigamidi`
   - Can be overridden with `export XMIDI_STORAGE_DIR=/path/to/storage`

3. **MuseTok checkpoints** (for `--preprocessor musetok`): Should be available from the valence_arousal task:
   ```
   /deepfreeze/pnlong/gigamidi/musetok/
   └── best_tokenizer.pt  (used for encoding/extracting latents)
   ```

4. **Node.js** (for `--preprocessor midi2vec`): Required for midi2edgelist. Install from [nodejs.org](https://nodejs.org). Then:
   ```bash
   cd midi2vec/midi2edgelist && npm install
   ```

5. **midi2vec embeddings dir** (for GigaMIDI with midi2vec): Default is `/deepfreeze/pnlong/gigamidi/midi2vec`. Override with `export MIDI2VEC_EMBEDDINGS_DIR=/path/to/midi2vec`.

6. **midi2vec batches dir** (for batched midi2vec): Default is `$STORAGE_DIR/midi2vec/batches` (e.g. `/deepfreeze/pnlong/gigamidi/midi2vec/batches`). Override with `export MIDI2VEC_BATCHES_DIR=/path/to/batches`. Used when `--midi2vec_num_batches` is set.

---

## Phase 1: Dataset Download and Preprocessing

### 1.1 Download XMIDI Dataset

```bash
cd emotion_genre
python pretrain_model/download_xmidi.py \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data
```

**Note**: After download, extract the zip file. The dataset should contain MIDI files with naming format: `XMIDI_<Emotion>_<Genre>_<ID_len_8>.midi`

### 1.2 Extract Latents/Embeddings from XMIDI

Two preprocessors are supported: **MuseTok** (per-bar latents, 128d) and **midi2vec** (per-song embeddings, 100d).

#### Option A: MuseTok (default)

```bash
python pretrain_model/preprocess_xmidi.py \
    --preprocessor musetok \
    --xmidi_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --gpu
```

**Key Arguments**: `--gpu`, `--batch_size`, `--num_workers`. Resume is default; use `--reset` to recompute from scratch.

#### Option B: midi2vec

XMIDI midi2vec latents are written to a dedicated subdirectory so they can coexist with MuseTok latents. The standard path used in this pipeline is **`latents_midi2vec`** under the xmidi data dir (e.g. `/deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents_midi2vec`).

```bash
python pretrain_model/preprocess_xmidi.py \
    --preprocessor midi2vec \
    --xmidi_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents_midi2vec
```

Runs midi2edgelist (Node.js) and edgelist2vec (Python) on the XMIDI directory. Output is compatible with the same dataset/training pipeline. Use `--input_dim 64` when training with midi2vec.

**Key Arguments**: `--workers 1` (default) uses a single core. Use `--workers 0` to use all CPU cores, or `--workers N` for N parallel processes. Resume is default; use `--reset` to recompute from scratch. If you stop after midi2edgelist, re-running skips it and continues from edgelist2vec.

**Precomputed**: If you already have `embeddings.bin` and `names.csv` for XMIDI, place them in `$MIDI2VEC_EMBEDDINGS_DIR` and add `--precomputed /path/to/dir` to skip the pipeline.

**Batched midi2vec (stratified, parallel)**  
For large corpora or to limit memory per run, use batched mode: you set the **number of batches** (`--midi2vec_num_batches`); files are split with **stratification by XMIDI labels** (emotion and genre). For each (emotion, genre) group, filepaths are shuffled and distributed round-robin over a shuffled batch order, so label proportions stay similar across all batches. **Default** is **50 batches** when calling the batched script directly; when using `preprocess_xmidi.py` you pass `--midi2vec_num_batches 50` (or another value). Each batch runs midi2edgelist then edgelist2vec in parallel (one process per batch); batches that already have `embeddings.bin` are skipped unless `--reset`. A single **batch_assignments.csv** (columns: `file_path`, `batch_id`) and per-batch dirs (`batch_0/`, `batch_1/`, …) are written under **batch_output_root** (`MIDI2VEC_BATCHES_DIR`). A **consolidation** step then writes one `.safetensors` per file into the same latents dir as non-batched mode, so training and evaluation are unchanged.

After assignment, **per-batch label statistics** are written to `batch_output_root` for manual stratification checks:
- **batch_label_stats.txt** — per-batch emotion and genre counts (human-readable).
- **batch_label_stats.csv** — same counts in CSV form (e.g. for plotting or spreadsheets).

Use these to confirm that emotion/genre counts are roughly similar across batches.

```bash
python pretrain_model/preprocess_xmidi.py \
    --preprocessor midi2vec \
    --xmidi_dir /path/to/xmidi \
    --output_dir /path/to/latents \
    --midi2vec_num_batches 50
```

Use `--reset` to recompute all batches and overwrite `batch_assignments.csv`. Omit `--reset` to resume (skip batches that already have `embeddings.bin`).

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

**Genre classes** (6 total): rock, pop, country, jazz, classical, traditional

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
    --input_dim 64 \
    --hidden_dim 64 \
    --batch_size 8192 \
    --learning_rate 1e-3 \
    --weight_decay 1e-5 \
    --epochs 250 \
    --dropout 0.1 \
    --class_weight balanced \
    --balanced_sampler \
    --gpu \
    --num_workers 4 \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models \
    --model_name emotion_classifier \
    --wandb_project gigamidi-support \
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
    --input_dim 64 \
    --hidden_dim 64 \
    --batch_size 8192 \
    --learning_rate 1e-3 \
    --weight_decay 1e-5 \
    --epochs 250 \
    --dropout 0.1 \
    --class_weight balanced \
    --balanced_sampler \
    --gpu \
    --num_workers 4 \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models \
    --model_name genre_classifier \
    --wandb_project gigamidi-support \
    --early_stopping \
    --early_stopping_tolerance 10
```

**Key Arguments**:
- `--task`: `emotion` or `genre`
- `--class_weight`: `balanced` (default) uses inverse-frequency class weights in the loss to handle imbalance; use `none` for unweighted loss
- `--balanced_sampler`: use WeightedRandomSampler so minority classes are seen more often per epoch (recommended with `--class_weight balanced` for emotion)
- `--resume`: Resume training from best checkpoint
- `--bootstrap_downsample`: If set, use K-fold bootstrap downsampling to balance classes (downsample to min class size). `0` (default) = full train set; `1` = one balanced run (seed=0); `10` = train 10 models (seeds 0..9), save `best_model_fold0.pt` … `best_model_fold9.pt`

**Bootstrap downsampling (imbalanced data)**: To reduce bias from class imbalance, use a single balanced run:
```bash
python pretrain_model/train.py ... --bootstrap_downsample 1
```
Or train 10 models (one per bootstrap fold) for ensemble use:
```bash
python pretrain_model/train.py ... --bootstrap_downsample 10
```

### 2.3 Train Valence–Arousal Regressor

Predicts (valence, arousal) from latents using a fixed mapping from the 11 XMIDI emotion classes to VA pairs. Uses MSE loss with optional class weighting and balanced sampling (same as emotion classifier).

```bash
python pretrain_model/train_va.py \
    --latents_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --emotion_labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_labels.json \
    --emotion_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_to_index.json \
    --train_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/train_files.txt \
    --valid_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/val_files.txt \
    --input_dim 64 \
    --hidden_dim 64 \
    --batch_size 8192 \
    --learning_rate 1e-4 \
    --weight_decay 1e-5 \
    --epochs 75 \
    --dropout 0.1 \
    --class_weight balanced \
    --balanced_sampler \
    --gpu \
    --num_workers 4 \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models \
    --model_name valence_arousal_regressor \
    --wandb_project gigamidi-support \
    --early_stopping \
    --early_stopping_tolerance 10
```

**Key Arguments**:
- `--emotion_labels_path`: Path to emotion_labels.json (filename → emotion index 0..10)
- `--emotion_to_index_path`: Path to emotion_to_index.json (used for class weights and balanced sampler)
- `--class_weight`: `balanced` (default) weights MSE by inverse emotion frequency
- `--balanced_sampler`: Oversample rare emotions per epoch

### 2.4 Train Combined (MuseTok + midi2vec) Classifier

Uses both MuseTok and midi2vec latents: loads from two dirs, concatenates vectors, normalizes per dimension (mean/std on the training set), and trains the same classifier. **`input_dim` is computed automatically** (MuseTok dim + midi2vec dim, e.g. 128+64=192). You need precomputed latents in both formats (e.g. from Phase 1 with MuseTok and midi2vec).

```bash
python pretrain_model/train_combined.py \
    --task emotion \
    --latents_dir_musetok /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents_musetok \
    --latents_dir_midi2vec /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents_midi2vec \
    --labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_labels.json \
    --class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_to_index.json \
    --train_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/train_files.txt \
    --valid_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/val_files.txt \
    --num_classes 11 \
    --batch_size 8192 \
    --learning_rate 1e-3 \
    --weight_decay 1e-5 \
    --epochs 1000 \
    --dropout 0.1 \
    --class_weight balanced \
    --balanced_sampler \
    --gpu \
    --num_workers 4 \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models \
    --model_name emotion_classifier_combined \
    --wandb_project gigamidi-support \
    --early_stopping \
    --early_stopping_tolerance 10
```

**Key Arguments**:
- `--latents_dir_musetok`: Directory of MuseTok latents (e.g. from preprocess with MuseTok).
- `--latents_dir_midi2vec`: Directory of midi2vec latents (same stems as MuseTok). Standard path: `/deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents_midi2vec`.
- `--stats_path`: Optional. If omitted, normalization mean/std are computed from the training set and saved under `output_dir/model_name/combined_latents_stats.npz`. Pass the same path when evaluating.
- No `--input_dim`: it is set from the combined stats (MuseTok dim + midi2vec dim).
- `--bootstrap_downsample`: Same as in 2.1 (0=full, 1=one balanced run, K=K models with `best_model_fold{k}.pt`).

For genre, use `--task genre`, `--labels_path`/`--class_to_index_path` for genre, `--num_classes 6`, and `--model_name genre_classifier_combined`.

### 2.5 Train sklearn Ensemble (MLP, LogReg, SVM, KNN, Random Forest)

Trains five sklearn classifiers on the same (optionally downsampled) latents, then combines their top-3 predictions via majority vote and reports an uncertainty (disagreement across the ensemble). Useful for robust labels and error estimates.

```bash
python pretrain_model/train_ensemble_sklearn.py \
    --task emotion \
    --latents_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents_midi2vec \
    --labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_labels.json \
    --class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_to_index.json \
    --train_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/train_files.txt \
    --test_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/test_files.txt \
    --valid_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/val_files.txt \
    --num_classes 11 \
    --bootstrap_downsample_seed 42 \
    --n_bootstrap_folds 10 \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/ensemble_sklearn/emotion \
    --save_predictions
```

**Key Arguments**:
- `--latents_dir`: Single latents directory (e.g. midi2vec 64d); use `latents_midi2vec` path for XMIDI midi2vec output.
- `--bootstrap_downsample_seed`: If set, downsample training set to min class size with this seed (one balanced set when `--n_bootstrap_folds 1`).
- `--n_bootstrap_folds`: Number of bootstrap folds; each fold trains all 5 models on a different downsampled set. Default 1; use 10 for 50 models (10×5) and aggregate predictions.
- `--save_predictions`: Write a CSV with filename, true/pred label, and uncertainty per sample.

**Outputs**: Saved under `--output_dir`: `model_<name>_fold{k}.joblib` (or `model_<name>.joblib` when n_folds=1), `metrics.txt`, and optionally `predictions.csv` with an `uncertainty` column (fraction of models disagreeing with the chosen label). Ensemble predictions and uncertainty can be used for GigaMIDI annotation or filtering.

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
    --input_dim 64 \
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
    --input_dim 64 \
    --hidden_dim 64 \
    --batch_size 32 \
    --gpu \
    --num_workers 4 \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/evaluation_results/genre
```

### 3.3 Evaluate Valence–Arousal Regressor

```bash
python pretrain_model/evaluate_va.py \
    --checkpoint_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models/valence_arousal_regressor/checkpoints/best_model.pt \
    --latents_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --emotion_labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_labels.json \
    --test_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/test_files.txt \
    --input_dim 64 \
    --hidden_dim 64 \
    --batch_size 32 \
    --gpu \
    --num_workers 4 \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/evaluation_results/valence_arousal
```

**Output**: `metrics.csv` (MSE, MAE, correlation for valence and arousal), `valence_arousal_scatter.png` (predicted vs true).

### 3.4 Evaluate Combined (MuseTok + midi2vec) Classifier

Use the same `--stats_path` as in training so input_dim and normalization match. **Do not pass `--input_dim`**; it is read from the stats file.

```bash
python pretrain_model/evaluate_combined.py \
    --task emotion \
    --checkpoint_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models/emotion_classifier_combined/checkpoints/best_model.pt \
    --latents_dir_musetok /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --latents_dir_midi2vec /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents_midi2vec \
    --stats_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models/emotion_classifier_combined/combined_latents_stats.npz \
    --labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_labels.json \
    --class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_to_index.json \
    --test_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/test_files.txt \
    --num_classes 11 \
    --batch_size 32 \
    --gpu \
    --num_workers 4 \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/evaluation_results/emotion_combined
```

---

**Output (3.1 & 3.2)**: Each classification evaluation generates:
- `metrics.csv`: Overall accuracy, F1-score (macro/weighted), Precision, Recall, per-class metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `classification_report.txt`: Detailed per-class classification report

---

## Phase 4: GigaMIDI Annotation

After training and evaluating the emotion and genre classifiers, use them to annotate the GigaMIDI dataset.

### 4.1 Annotate GigaMIDI with Emotion and Genre Predictions

#### Option A: MuseTok (on-the-fly extraction)

```bash
python annotate_gigamidi.py \
    --preprocessor musetok \
    --emotion_model_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models/emotion_classifier/checkpoints/best_model.pt \
    --genre_model_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models/genre_classifier/checkpoints/best_model.pt \
    --emotion_class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_to_index.json \
    --genre_class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/genre_to_index.json \
    --input_dim 128 \
    --emotion_num_classes 11 \
    --genre_num_classes 6 \
    --hidden_dim 64 \
    --dropout 0.1 \
    --gpu \
    --streaming \
    --split train \
    --resume \
    --output_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/gigamidi_annotations/annotations.csv
```

**Key Arguments**:
- `--emotion_model_path`: Path to trained emotion classifier checkpoint
- `--genre_model_path`: Path to trained genre classifier checkpoint
- `--emotion_class_to_index_path`: Path to emotion_to_index.json (for mapping indices to class names)
- `--genre_class_to_index_path`: Path to genre_to_index.json (for mapping indices to class names)
- `--streaming`: Use streaming mode to avoid downloading entire dataset (recommended)
- `--split`: Dataset split to annotate (`train`, `test`, or `validation`)
- `--resume`: Resume from existing CSV file (skip already-processed songs)
- `--output_path`: Output CSV file path (defaults to `<STORAGE_DIR>/xmidi_emotion_genre/gigamidi_annotations/annotations.csv`)

**Output Format**: CSV file with columns:
- `md5`: Song identifier (MD5 hash)
- `emotion`: Predicted emotion class (string: one of 11 emotions)
- `emotion_prob`: Confidence/probability for emotion prediction (0-1)
- `genre`: Predicted genre class (string: one of 6 genres)
- `genre_prob`: Confidence/probability for genre prediction (0-1)

**Note**: 
- MuseTok: processes songs one at a time, extracting latents on-the-fly
- Mean pooling is applied across bars to get song-level predictions
- Progress is saved incrementally (one row per song) to avoid losing work
- Use `--resume` to continue from where you left off if interrupted

#### Option B: midi2vec (precomputed embeddings)

Requires precomputed GigaMIDI embeddings. Run `export_gigamidi_for_midi2vec.py` first (see below).

```bash
python annotate_gigamidi.py \
    --preprocessor midi2vec \
    --embeddings_dir /deepfreeze/pnlong/gigamidi/midi2vec \
    --emotion_model_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models/emotion_classifier/checkpoints/best_model.pt \
    --genre_model_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models/genre_classifier/checkpoints/best_model.pt \
    --emotion_class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_to_index.json \
    --genre_class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/genre_to_index.json \
    --input_dim 100 \
    --emotion_num_classes 11 \
    --genre_num_classes 6 \
    --streaming \
    --split train \
    --resume \
    --output_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/gigamidi_annotations/annotations.csv
```

**Precomputed embeddings**: Run `export_gigamidi_for_midi2vec.py` to export GigaMIDI to disk and run midi2vec pipeline. This produces `embeddings.bin` and `names.csv` in the output dir. Songs not in the lookup are skipped.

### 4.2 Export GigaMIDI for midi2vec (one-time, for Option B above)

```bash
python export_gigamidi_for_midi2vec.py \
    --output_dir /deepfreeze/pnlong/gigamidi/midi2vec \
    --split train \
    --streaming \
    --max_samples 10000
```

Omit `--max_samples` for full dataset. Use `--skip_export` to re-run midi2vec on an existing export. `--workers 1` (default) uses a single core; use `--workers 0` for all CPU cores. If you stop after midi2edgelist, re-running the script skips it and continues from edgelist2vec.

### 4.3 Analyze Annotations (Optional)

After annotation, you can analyze the distribution of predictions:

```bash
# Print statistics
python analyze_annotations/print_statistics.py \
    --annotations_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/gigamidi_annotations/annotations.csv

# Plot histograms
python analyze_annotations/plot_histograms.py \
    --annotations_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/gigamidi_annotations/annotations.csv \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/gigamidi_annotations/plots

# Plot by genre
python analyze_annotations/plot_by_genre.py \
    --annotations_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/gigamidi_annotations/annotations.csv \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/gigamidi_annotations/plots
```

---

## Quick Reference: File Structure

```
/deepfreeze/pnlong/gigamidi/
├── musetok/
│   └── best_tokenizer.pt  (shared MuseTok checkpoint)
├── midi2vec/               # midi2vec precomputed embeddings (optional)
│   ├── embeddings.bin
│   ├── names.csv
│   └── batches/            # batched midi2vec output (when --midi2vec_num_batches is used)
│       ├── batch_assignments.csv
│       ├── batch_0/
│       │   ├── edgelist/
│       │   ├── embeddings.bin
│       │   └── names.csv
│       └── batch_1/ ...
│   └── gigamidi_midis/    # Exported GigaMIDI as {md5}.mid
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
    │   ├── latents/          # MuseTok preprocessed latents (.safetensors)
    │   ├── latents_midi2vec/ # midi2vec preprocessed latents (.safetensors); standard path for XMIDI midi2vec
    │   └── labels/
    │       ├── emotion_labels.json
    │       ├── genre_labels.json
    │       ├── emotion_to_index.json
    │       ├── genre_to_index.json
    │       ├── train_files.txt
    │       ├── val_files.txt
    │       └── test_files.txt
    ├── gigamidi_annotations/
    │   ├── annotations.csv  # GigaMIDI annotations (md5, emotion, emotion_prob, genre, genre_prob)
    │   └── plots/            # Analysis plots (optional)
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
    --gpu

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
    --epochs 100

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
    --epochs 100

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

# 7. Annotate GigaMIDI dataset
python annotate_gigamidi.py \
    --preprocessor musetok \
    --emotion_model_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models/emotion_classifier/checkpoints/best_model.pt \
    --genre_model_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/checkpoints/trained_models/genre_classifier/checkpoints/best_model.pt \
    --emotion_class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_to_index.json \
    --genre_class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/genre_to_index.json \
    --input_dim 128 \
    --emotion_num_classes 11 \
    --genre_num_classes 6 \
    --hidden_dim 64 \
    --gpu \
    --streaming \
    --split train \
    --resume
```

### Workflow 2: Complete Pipeline with midi2vec

```bash
# 1. Download and extract XMIDI dataset
python pretrain_model/download_xmidi.py

# 2. Preprocess XMIDI with midi2vec
python pretrain_model/preprocess_xmidi.py \
    --preprocessor midi2vec \
    --xmidi_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents

# 3. Prepare labels and splits
python pretrain_model/prepare_labels.py \
    --xmidi_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data \
    --output_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels

# 4. Train emotion classifier (--preprocessor midi2vec sets input_dim=100)
python pretrain_model/train.py \
    --task emotion \
    --preprocessor midi2vec \
    --latents_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_labels.json \
    --class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/emotion_to_index.json \
    --train_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/train_files.txt \
    --valid_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/val_files.txt \
    --num_classes 11 \
    --batch_size 32 \
    --epochs 100

# 5. Train genre classifier
python pretrain_model/train.py \
    --task genre \
    --preprocessor midi2vec \
    --latents_dir /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/latents \
    --labels_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/genre_labels.json \
    --class_to_index_path /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/genre_to_index.json \
    --train_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/train_files.txt \
    --valid_files /deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/labels/val_files.txt \
    --num_classes 6 \
    --batch_size 32 \
    --epochs 100

# 6. Evaluate both models
python pretrain_model/evaluate.py --task emotion --preprocessor midi2vec ...
python pretrain_model/evaluate.py --task genre --preprocessor midi2vec ...

# 7. Export GigaMIDI for midi2vec (one-time)
python export_gigamidi_for_midi2vec.py \
    --output_dir /deepfreeze/pnlong/gigamidi/midi2vec \
    --split train

# 8. Annotate GigaMIDI with midi2vec
python annotate_gigamidi.py \
    --preprocessor midi2vec \
    --embeddings_dir /deepfreeze/pnlong/gigamidi/midi2vec \
    --emotion_model_path ... \
    --genre_model_path ... \
    --input_dim 100 \
    --streaming --split train --resume
```

---

## Tips

1. **Resume Processing**: Preprocessing resumes by default (skips existing output). Use `--reset` to recompute from scratch.
2. **Monitoring**: Wandb logging is enabled by default to track metrics in real-time
3. **Device**: Omit `--gpu` flag to use CPU instead of GPU
4. **Input Dimension**: MuseTok latents are 128-dimensional; midi2vec embeddings are 100-dimensional. Use `--preprocessor midi2vec` (or `--input_dim 100`) when training/evaluating with midi2vec.
5. **Mean Pooling**: The dataset automatically pools latents across bars for song-level prediction
7. **Stratified Splits**: The prepare_labels script creates stratified splits to maintain class distribution
8. **midi2vec is transductive**: Embeddings exist only for files that were in the graph when node2vec ran. No pretrained model for new files. Run the pipeline on your corpus.

---

## Troubleshooting

- **Out of Memory**: Reduce `--batch_size` in training/evaluation scripts
- **Checkpoint Not Found**: Verify MuseTok checkpoint is available from valence_arousal task (musetok only)
- **File Not Found**: Check that XMIDI directories exist and paths are correct
- **Class Imbalance**: Check confusion matrix to see if certain classes are underrepresented
- **Low Accuracy**: Try adjusting learning rate, hidden dimension, or dropout rate
- **midi2edgelist failed**: Ensure Node.js is installed and `cd midi2vec/midi2edgelist && npm install` has been run
- **md5 not in embeddings**: For midi2vec GigaMIDI annotation, ensure export_gigamidi_for_midi2vec.py was run on the same split. Songs not in the export are skipped.
