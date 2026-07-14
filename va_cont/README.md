# Continuous Valence/Arousal (`va_cont`)

Train a **bar-level continuous valence/arousal regressor** on MIDI, using dynamic V/A annotations from **DEAM**, **Memo2496**, and **MERP**. Unlike the categorical `emotion_genre` approach, this pipeline learns directly from time-varying V/A curves aligned to transcribed MIDI.

## Overview

**Task:** Predict (valence, arousal) ∈ [-1, 1]² for each bar of a MIDI file.

**Training data:** Combined DEAM + Memo2496 + MERP (~5.3k songs after preprocessing).

**Features (choose one `feature_mode`):**

| Mode | Description | Preprocess output |
|------|-------------|-------------------|
| `musetok` (default) | Frozen MuseTok 128-d bar latents | `{dataset}_va/latents_musetok/` |
| `handcrafted` | 32-d per-bar MIDI stats (pitch, velocity, chroma, density) | `{dataset}_va/features_handcrafted/` |
| `remi` | Padded REMI token indices per bar + learnable encoder | `{dataset}_va/features_remi/` |

**Model:** `VAModel` — causal transformer over bar features, optionally preceded by a learnable `REMIBarEncoder` (for `remi` mode). Configs:

| Config | `feature_mode` | Architecture |
|--------|----------------|--------------|
| `va_transformer_a.yaml` | musetok | 2-layer, d_model=128 (original) |
| `va_transformer_a_binned.yaml` | musetok | Model A + soft-binned targets (n=20) |
| `va_transformer_b.yaml` | musetok | + VA conditioning (AR) |
| `va_transformer_c.yaml` | musetok | **Scaled** 4-layer, d_model=256 |
| `va_midi_handcrafted.yaml` | handcrafted | Scaled transformer, no MuseTok |
| `va_remi_e2e.yaml` | remi | REMI bar encoder + scaled transformer (Model A) |
| `va_remi_e2e_b.yaml` | remi | REMI e2e + VA conditioning / AR (Model B) |

**Downstream:** Apply the trained model bar-by-bar to GigaMIDI (MIDI-only, no audio).

## Documentation map

| Document | Contents |
|----------|----------|
| **[PIPELINE.md](PIPELINE.md)** | Step-by-step commands **and** per-dataset processing methodology (raw → converted MIDI) for paper writing |
| **[datasets/README.md](datasets/README.md)** | Adapter interface quick reference |

## Pipeline summary

See [PIPELINE.md](PIPELINE.md) for commands and the full methodology section.

1. **Download** raw DEAM / Memo2496 / MERP under `$XMIDI_STORAGE_DIR`
2. **AMT** (YourMT3+ multitrack): audio → MIDI
3. **MuseTok**: MIDI → bar latents (`.safetensors`)
4. **convert_va**: audio-time V/A → tick-indexed continuous storage (`.npz`)
5. **QC**: alignment plots (audio seconds vs MIDI ticks)
6. **Train** — see [Training](#training) below
7. **Annotate GigaMIDI**: bar-level V/A predictions

## File structure

```
va_cont/
├── README.md
├── PIPELINE.md                 # ← methodology for paper reproducibility
├── requirements.txt
├── va_utils.py                 # resample, tick alignment, bar aggregation
├── annotate_gigamidi.py
├── datasets/                   # DEAM, Memo2496, MERP adapters
├── preprocess/
│   ├── preprocess_musetok.py          # MuseTok latents (--dataset {deam,memo2496,merp})
│   ├── extract_bar_midi_features.py   # handcrafted or REMI bar features (no MuseTok)
│   └── convert_va.py
├── pretrain_model/
│   ├── train.py
│   ├── dataset.py
│   ├── model.py                       # CausalVATransformer, REMIBarEncoder, VAModel
│   ├── midi_features.py               # handcrafted + REMI extraction utilities
│   └── configs/
│       ├── va_transformer_a.yaml
│       ├── va_transformer_c.yaml      # scaled MuseTok baseline
│       ├── va_midi_handcrafted.yaml   # raw MIDI stats
│       ├── va_remi_e2e.yaml           # REMI encoder, Model A
│       └── va_remi_e2e_b.yaml         # REMI encoder, Model B (AR)
└── tools/
    ├── verify_datasets.py
    └── plot_va_alignment.py
```

## Storage layout (derived artifacts)

All derived data lives beside the raw datasets under `$XMIDI_STORAGE_DIR` (default `/deepfreeze/pnlong/gigamidi`):

```
{dataset}_va/
├── latents_musetok/{song_id}.safetensors      # MuseTok bar latents (feature_mode=musetok)
├── features_handcrafted/{song_id}.safetensors # 32-d bar stats (feature_mode=handcrafted)
├── features_remi/{song_id}.safetensors        # bar_tokens + token_padding_mask (feature_mode=remi)
├── continuous/{song_id}.npz                   # tick-indexed V/A (canonical labels)
└── labels/
    ├── train_songs.txt, val_songs.txt, test_songs.txt
    └── {dataset}_va_labels.json               # optional bar cache from --cache-bar-labels
```

MIDI from AMT is stored per dataset (`deam/DEAM_midi/`, `memo2496_midi/`, `merp_midi/`). See [PIPELINE.md](PIPELINE.md) for exact paths.

## Key design notes

- **Canonical label format:** tick-indexed continuous `.npz` (`ticks`, `valence`, `arousal`, `tpq`, `bar_resol`). Bar targets are derived at train time (or from optional JSON cache).
- **Unified resampling:** all sources are linearly interpolated to **10 Hz** before MIDI tick mapping, regardless of native rate (DEAM 2 Hz, Memo2496 1 Hz, MERP 10 Hz).
- **Temporal alignment:** audio-second timestamps are mapped to MIDI ticks via the AMT score's tempo track (`seconds_to_ticks` in `va_utils.py`).
- **MERP:** use `annotations_raw.parquet` (full-song), not `annotations_filtered.parquet` (segment-trimmed for the MERP paper). Details in [PIPELINE.md](PIPELINE.md).
- **Overlap policy:** MERP tracks that overlap DEAM anchor songs are excluded from MERP train/val/test splits.

## Training

### 1. Preprocess features (pick your `feature_mode`)

**MuseTok** (existing path — skip if already done):

```bash
python va_cont/preprocess/preprocess_musetok.py --dataset deam --gpu --resume
python va_cont/preprocess/preprocess_musetok.py --dataset memo2496 --gpu --resume
python va_cont/preprocess/preprocess_musetok.py --dataset merp --gpu --resume
```

**Hand-crafted MIDI features** (no MuseTok — extracts pitch/velocity/chroma stats per bar):

```bash
python va_cont/preprocess/extract_bar_midi_features.py --dataset deam --feature_mode handcrafted --resume
python va_cont/preprocess/extract_bar_midi_features.py --dataset memo2496 --feature_mode handcrafted --resume
python va_cont/preprocess/extract_bar_midi_features.py --dataset merp --feature_mode handcrafted --resume
```

**REMI tokens** (no MuseTok latents — caches padded token indices per bar for the learnable encoder):

```bash
python va_cont/preprocess/extract_bar_midi_features.py --dataset deam --feature_mode remi --resume
python va_cont/preprocess/extract_bar_midi_features.py --dataset memo2496 --feature_mode remi --resume
python va_cont/preprocess/extract_bar_midi_features.py --dataset merp --feature_mode remi --resume
```

### 2. Train

All configs support CLI overrides (e.g. `--batch_size`, `--valid_every_epochs`, `--checkpoint_metric`).

**Original MuseTok baseline (Model A):**

```bash
CUDA_VISIBLE_DEVICES=0 python va_cont/pretrain_model/train.py \
  --config va_cont/pretrain_model/configs/va_transformer_a.yaml \
  --batch_size 2048 --valid_every_epochs 20
```

**Soft-binned VA targets (Model A, n=20 bins on [-1, 1]):**

```bash
CUDA_VISIBLE_DEVICES=0 python va_cont/pretrain_model/train.py \
  --config va_cont/pretrain_model/configs/va_transformer_a_binned.yaml \
  --batch_size 2048 --valid_every_epochs 20
```

Override bin count: `--n_bins 10` or soft-label width: `--bin_sigma 0.15`.

**Scaled MuseTok (4 layers, d_model=256):**

```bash
CUDA_VISIBLE_DEVICES=0 python va_cont/pretrain_model/train.py \
  --config va_cont/pretrain_model/configs/va_transformer_c.yaml \
  --batch_size 2048 --valid_every_epochs 20
```

**Hand-crafted MIDI features (raw MIDI stats, no MuseTok):**

```bash
CUDA_VISIBLE_DEVICES=0 python va_cont/pretrain_model/train.py \
  --config va_cont/pretrain_model/configs/va_midi_handcrafted.yaml \
  --batch_size 2048 --valid_every_epochs 20
```

**End-to-end REMI encoder + transformer (Model A — latents only):**

```bash
CUDA_VISIBLE_DEVICES=0 python va_cont/pretrain_model/train.py \
  --config va_cont/pretrain_model/configs/va_remi_e2e.yaml \
  --batch_size 16 --valid_every_epochs 20
```

**End-to-end REMI encoder + transformer (Model B — VA-conditioned AR):**

```bash
CUDA_VISIBLE_DEVICES=0 python va_cont/pretrain_model/train.py \
  --config va_cont/pretrain_model/configs/va_remi_e2e_b.yaml \
  --batch_size 16 --valid_every_epochs 20
```

### Training notes

- `batch_size` is **songs per batch** (not bars). Large values (e.g. 2048) work when songs are short; use **16–32** for `remi` mode (much more memory per song).
- **Model A** (`va_conditioning: false`): validation corr uses a single forward pass over REMI/MIDI features.
- **Model B** (`va_conditioning: true`): training teacher-forces previous-bar ground-truth V/A; validation corr and best-checkpoint selection use **autoregressive** decoding (`valid/ar/corr_*` in WandB).
- `checkpoint_metric`: `mse` (default), `corr_sum` (valence+arousal Pearson r), or `corr_valence`.
- Validation logs **per-dataset** correlation to WandB: Model A → `valid/{dataset}/corr_*`; Model B → `valid/ar/{dataset}/corr_*`.
- Best checkpoint saved to `{output_dir}/{model_name}/checkpoints/best_model.pt`.

## Dependencies

See `requirements.txt`. MuseTok checkpoint (shared with `emotion_genre`):

```
$XMIDI_STORAGE_DIR/musetok/best_tokenizer.pt
```

YourMT3+ multitrack checkpoint (AMT):

```
mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt
```
