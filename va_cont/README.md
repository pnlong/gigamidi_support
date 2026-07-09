# Continuous Valence/Arousal (`va_cont`)

Train a **bar-level continuous valence/arousal regressor** on MIDI, using dynamic V/A annotations from **DEAM**, **Memo2496**, and **MERP**. Unlike the categorical `emotion_genre` approach, this pipeline learns directly from time-varying V/A curves aligned to transcribed MIDI.

## Overview

**Task:** Predict (valence, arousal) ∈ [-1, 1]² for each bar of a MIDI file.

**Training data:** Combined DEAM + Memo2496 + MERP (~5.3k songs after preprocessing).

**Features:** [MuseTok](https://github.com/Yuer867/MuseTok) — 128-dimensional latent per bar from AMT-generated MIDI.

**Model:** Causal transformer (`CausalVATransformer`, config `va_transformer_a.yaml`) or legacy 2-layer MLP.

**Downstream:** Apply the trained model bar-by-bar to GigaMIDI.

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
6. **Train**: combined `va_transformer_a` on all three sources
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
│   ├── preprocess_musetok.py   # generic --dataset {deam,memo2496,merp}
│   └── convert_va.py
├── pretrain_model/
│   ├── train.py
│   ├── dataset.py
│   ├── model.py
│   └── configs/va_transformer_a.yaml
└── tools/
    ├── verify_datasets.py
    └── plot_va_alignment.py
```

## Storage layout (derived artifacts)

All derived data lives beside the raw datasets under `$XMIDI_STORAGE_DIR` (default `/deepfreeze/pnlong/gigamidi`):

```
{dataset}_va/
├── latents_musetok/{song_id}.safetensors   # MuseTok bar latents + metadata
├── continuous/{song_id}.npz                # tick-indexed V/A (canonical labels)
└── labels/
    ├── train_songs.txt, val_songs.txt, test_songs.txt
    └── {dataset}_va_labels.json            # optional bar cache from --cache-bar-labels
```

MIDI from AMT is stored per dataset (`deam/DEAM_midi/`, `memo2496_midi/`, `merp_midi/`). See [PIPELINE.md](PIPELINE.md) for exact paths.

## Key design notes

- **Canonical label format:** tick-indexed continuous `.npz` (`ticks`, `valence`, `arousal`, `tpq`, `bar_resol`). Bar targets are derived at train time (or from optional JSON cache).
- **Unified resampling:** all sources are linearly interpolated to **10 Hz** before MIDI tick mapping, regardless of native rate (DEAM 2 Hz, Memo2496 1 Hz, MERP 10 Hz).
- **Temporal alignment:** audio-second timestamps are mapped to MIDI ticks via the AMT score's tempo track (`seconds_to_ticks` in `va_utils.py`).
- **MERP:** use `annotations_raw.parquet` (full-song), not `annotations_filtered.parquet` (segment-trimmed for the MERP paper). Details in [PIPELINE.md](PIPELINE.md).
- **Overlap policy:** MERP tracks that overlap DEAM anchor songs are excluded from MERP train/val/test splits.

## Dependencies

See `requirements.txt`. MuseTok checkpoint (shared with `emotion_genre`):

```
$XMIDI_STORAGE_DIR/musetok/best_tokenizer.pt
```

YourMT3+ multitrack checkpoint (AMT):

```
mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt
```
