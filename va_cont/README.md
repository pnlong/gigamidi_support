# DEAM Bar-Level Continuous Valence/Arousal Prediction (`va_cont`)

This module trains a bar-level continuous valence/arousal regressor using the DEAM dataset. Unlike the `emotion_genre` approach (which maps categorical emotion labels to fixed VA pairs), this trains directly on DEAM's continuous dynamic annotations aligned to MIDI bars.

## Overview

**Task:** Predict (valence, arousal) ∈ [-1, 1]² for each bar of a MIDI file.

**Dataset:** [DEAM](https://cvml.unige.ch/databases/DEAM/) — ~1802 songs with dynamic valence/arousal annotations at 500ms intervals, already normalized to [-1, 1]. Annotations start at 15s to exclude the unstable onset region.

**Feature extraction:** [MuseTok](https://github.com/Yuer867/MuseTok) — extracts a 128-dimensional latent vector per bar from MIDI. Shared checkpoint from the `emotion_genre` task.

**Model:** 2-layer MLP (`input_dim=128 → hidden_dim=64 → 2`), trained with MSE loss.

**Downstream use:** Apply trained model bar-by-bar to GigaMIDI to generate bar-level valence/arousal annotations.

## Pipeline Summary

See [PIPELINE.md](PIPELINE.md) for full step-by-step instructions.

1. **AMT** (external): convert DEAM audio → MIDI
2. **Preprocess**: extract MuseTok latents + bar start times from MIDI
3. **Prepare labels**: align DEAM 500ms annotations to MIDI bars
4. **Train**: fit `ValenceArousalRegressor` on bar-level latents
5. **Annotate GigaMIDI**: apply trained model bar-by-bar

## File Structure

```
va_cont/
├── README.md
├── PIPELINE.md
├── requirements.txt
├── annotate_gigamidi.py          # Apply trained model to GigaMIDI (bar-by-bar)
├── preprocess/
│   └── preprocess_deam_musetok.py  # Extract MuseTok latents + bar times from DEAM MIDI
├── pretrain_model/
│   ├── prepare_labels.py         # Align DEAM annotations to bars → JSON labels + splits
│   ├── dataset.py                # DEAMDataset (bar-level continuous VA)
│   ├── model.py                  # ValenceArousalRegressor (2-layer MLP)
│   ├── train.py                  # Training script (MSE loss, WandB, early stopping)
│   └── configs/
│       ├── deam_bars1.yml        # 1 bar per sample
│       └── deam_bars4.yml        # 4-bar mean-pooled chunks
└── utils/                        # Symlink → ../emotion_genre/utils/
    ├── data_utils.py
    ├── midi_utils.py
    ├── musetok_utils.py
    └── config_utils.py
```

## Storage Directory Structure

```
$XMIDI_STORAGE_DIR/  (default: /deepfreeze/pnlong/gigamidi)
├── deam/
│   ├── DEAM_audio/MEMD_audio/     # Source MP3 files
│   ├── DEAM_midi/MEMD_midi/       # AMT-generated MIDI (same numeric IDs)
│   └── DEAM_Annotations/          # Annotation CSVs
└── deam_va/
    ├── latents_musetok/           # Preprocessed latents: {song_id}.safetensors
    ├── labels/
    │   ├── deam_va_labels.json    # {song_id: [[bar_idx, valence, arousal], ...]}
    │   ├── train_songs.txt
    │   ├── val_songs.txt
    │   └── test_songs.txt
    └── checkpoints/trained_models/
        └── deam_va_regressor_bars1/
            ├── checkpoints/
            │   ├── best_model.pt
            │   └── best_optimizer.pt
            ├── statistics.csv
            └── train.log
```

## Annotation Format

**`deam_va_labels.json`**:
```json
{
  "1000": [[5, -0.12, 0.34], [6, -0.09, 0.31], ...],
  "1001": [[8, 0.45, 0.12], ...]
}
```
Each entry: `[bar_idx, valence, arousal]`. Bars before 15s are excluded (no annotation coverage).

**`bar_va_annotations.csv`** (GigaMIDI output):
```
md5,bar_idx,valence,arousal
abc123,0,0.412,-0.088
abc123,1,0.398,-0.091
...
```

## Key Design Notes

- **Temporal alignment**: Bar `i` spans `[bar_start_times[i], bar_start_times[i+1])`. Annotation samples (500ms intervals) falling within the window are averaged. Bar start times in seconds are computed from MIDI tick positions + tempo track.
- **15s cutoff**: Bars with no annotation samples (before 15s) are excluded from training. No fallback to song-level mean.
- **Dynamic annotations**: Already in [-1, 1]; no normalization required.
- **Song-level balanced sampler** (`--balanced_sampler`): Weights bar samples by `1/n_annotated_bars_in_song` so songs contribute equally regardless of length.
- **GigaMIDI inference**: Each bar's 128d latent is fed to the model independently.

## Dependencies

See `requirements.txt`. Core packages: `torch`, `symusic`, `safetensors`, `pandas`, `numpy`, `wandb`, `tqdm`, `PyYAML`.

MuseTok checkpoint (shared with `emotion_genre`):
```
/deepfreeze/pnlong/gigamidi/musetok/best_tokenizer.pt
```
