# Continuous Valence/Arousal Pipeline

Unified pipeline for **DEAM + Memo2496 + MERP** combined training.  
Canonical V/A storage is **tick-indexed continuous** (`.npz`); bar-level targets are derived at train time for MuseTok.

---

## Quick reference

| Step | Command |
|------|---------|
| Verify data | `python va_cont/tools/verify_datasets.py` |
| AMT config | `python yourmt3-cc/generate_config.py --dataset deam --output deam_amt.json` |
| AMT transcribe | `cd YourMT3 && python inference.py --config_json /tmp/deam_amt.json --device cuda` |
| MuseTok latents | `python va_cont/preprocess/preprocess_musetok.py --dataset deam --gpu --resume` |
| V/A → ticks | `python va_cont/preprocess/convert_va.py --dataset deam --cache-bar-labels` |
| QC plot | `python va_cont/tools/plot_va_alignment.py --dataset deam --song_id 1000` |
| Train (combined) | `python va_cont/pretrain_model/train.py --config va_cont/pretrain_model/configs/va_transformer_a.yaml` |
| GigaMIDI annotate | `python va_cont/annotate_gigamidi.py --model_path .../best_model.pt --gpu` |

Repeat preprocess/convert steps for `memo2496` and `merp`.

---

## Dataset processing methodology (paper reference)

This section documents **exactly** how each raw public dataset becomes our derived MIDI-aligned training data. Implementation lives in [`datasets/`](datasets/) and [`va_utils.py`](va_utils.py).

### Shared pipeline (all three datasets)

Every dataset follows the same four derived stages after download:

```
Raw audio + raw V/A annotations
        │
        ▼  Phase 1: YourMT3+ AMT (multitrack)
   {dataset}_midi/{id}.mid
        │
        ▼  Phase 2: MuseTok latent extraction
   {dataset}_va/latents_musetok/{id}.safetensors
        │
        ▼  Phase 3: convert_va (audio seconds → MIDI ticks)
   {dataset}_va/continuous/{id}.npz
        │
        ▼  Phase 4: training (bar labels derived from .npz + latents)
   CausalVATransformer checkpoints
```

#### Phase 1 — Automatic music transcription (AMT)

| Setting | Value |
|---------|-------|
| Model | YourMT3+ multitrack MoE (`YPTF.MoE+Multi`) |
| Checkpoint | `mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt` |
| Script | `YourMT3/inference.py` via config from `yourmt3-cc/generate_config.py` |
| Input | Full-mix audio (MP3 or WAV per dataset) |
| Output | Flat `{song_id}.mid` in each dataset's `midi_dir()` |
| Audio loading | `soundfile` first; `librosa` fallback for problematic MP3s |

Config generation enumerates audio paths from the dataset adapter and sets `flat_output: true` so MIDI filenames match song IDs (Memo2496 uses UUID stems on disk; latents use numeric `song_id`).

#### Phase 2 — MuseTok bar latents

| Setting | Value |
|---------|-------|
| Checkpoint | `$XMIDI_STORAGE_DIR/musetok/best_tokenizer.pt` |
| Script | `va_cont/preprocess/preprocess_musetok.py --dataset {name}` |
| Input | AMT MIDI from Phase 1 |
| Output | `{dataset}_va/latents_musetok/{id}.safetensors` |
| Latent shape | `(n_bars, 128)` — one 128-d vector per bar |
| Metadata stored | `n_bars`, `bar_start_times_seconds`, `bar_resol`, `tpq`, `original_file_path` |

MIDI is tokenized with the MuseTok REMI vocabulary (Bar, Beat, Note_Pitch, Note_Duration, Note_Velocity, Time_Signature; chord events dropped). The encoder processes 16-bar segments.

#### Phase 3 — V/A alignment (`convert_va`)

| Setting | Value |
|---------|-------|
| Script | `va_cont/preprocess/convert_va.py --dataset {name}` |
| Target sample rate | **10 Hz** (`DEFAULT_TARGET_HZ` in `va_utils.py`) for all datasets |
| Resampling | Linear interpolation (`np.interp`) from native annotation timestamps onto a uniform 10 Hz grid |
| Tick mapping | Each grid time `t` (seconds) → MIDI tick via `seconds_to_ticks(score, t)` using the **AMT MIDI tempo map** |
| Output | `{dataset}_va/continuous/{id}.npz` with keys `ticks`, `valence`, `arousal`, `tpq`, `bar_resol` |
| Splits | Song-level train/val/test (90/10/10, seed 42) written to `{dataset}_va/labels/` |
| Optional cache | `--cache-bar-labels` writes `{dataset}_va_labels.json` (mean V/A per bar tick window) |

Bar aggregation (for training or cache): for bar `i`, mean all samples with `ticks ∈ [i·bar_resol, (i+1)·bar_resol)`.

#### Phase 1b — Alignment QC (recommended before bulk conversion)

`plot_va_alignment.py` produces a 2×2 figure per song: left column = V/A vs **audio seconds**; right column = same curve vs **MIDI ticks** (no tick→second conversion on the right). Compare curve **shapes** across columns to verify AMT tempo alignment.

---

### DEAM

**Source:** [DEAM — Database for Emotional Analysis of Music](https://cvml.unige.ch/databases/DEAM/) (Aljanaki et al.).

**Raw layout** (under `$XMIDI_STORAGE_DIR/deam/`):

```
DEAM_audio/MEMD_audio/{song_id}.mp3          # 45 s excerpts, numeric IDs (e.g. 1000)
DEAM_Annotations/annotations/annotations averaged per song/
  dynamic (per second annotations)/
    valence.csv, arousal.csv
```

**Raw annotations:**

| Property | Value |
|----------|-------|
| Format | Wide CSV: `song_id`, `sample_15000ms`, `sample_15500ms`, … |
| Native rate | **2 Hz** (500 ms steps) |
| Time origin | **15 s** — first column is `sample_15000ms`; onset region excluded as unstable |
| Value range | Already in **[-1, 1]** (dataset-provided dynamic averages per song) |
| Our loading | `datasets/deam.py` → `parse_sample_ms_columns()` → `{time_sec: value}` dicts |

**Our processing (no extra annotation transforms):**

- Use dynamic per-song averaged CSVs as-is.
- `min_annotation_time()` = **15.0 s** — samples before 15 s are dropped during `convert_va`.
- No per-rater averaging (DEAM release is pre-averaged).

**Derived paths:**

| Artifact | Path |
|----------|------|
| AMT MIDI | `deam/DEAM_midi/MEMD_midi/{song_id}.mid` |
| Latents | `deam_va/latents_musetok/{song_id}.safetensors` |
| Continuous V/A | `deam_va/continuous/{song_id}.npz` |

**Scale:** ~1802 songs (all clips with successful AMT + conversion).

---

### Memo2496

**Source:** Memo dataset (2496 songs with continuous V/A; see dataset paper / release for citation).

**Raw layout** (under `$XMIDI_STORAGE_DIR/memo2496/`):

```
MusicRawData/{uuid}.mp3                       # audio keyed by UUID filename
songs_info_all.csv                            # maps song_id ↔ file_name (UUID)
valence_all_average.csv, arousal_all_average.csv
```

**Raw annotations:**

| Property | Value |
|----------|-------|
| Format | Wide CSV: `song_id`, `sample_0ms`, `sample_1000ms`, … |
| Native rate | **1 Hz** (1000 ms steps) |
| Time origin | **0 s** |
| Value range | Pre-averaged across annotators, **[-1, 1]** |
| Our loading | `datasets/memo2496.py` → `parse_sample_ms_columns()` |

**ID mapping:**

- **Audio / MIDI filenames** use the UUID stem from `songs_info_all.csv` (`file_name` column).
- **Latents and continuous V/A filenames** use numeric **`song_id`** (adapter default `latent_id()`).
- `midi_path(song_id)` resolves UUID; `latents_path(song_id)` uses numeric ID.

**Our processing (no extra annotation transforms):**

- Use `valence_all_average.csv` / `arousal_all_average.csv` as-is.
- `min_annotation_time()` = **0.0 s**.

**Derived paths:**

| Artifact | Path |
|----------|------|
| AMT MIDI | `memo2496_midi/{uuid}.mid` |
| Latents | `memo2496_va/latents_musetok/{song_id}.safetensors` |
| Continuous V/A | `memo2496_va/continuous/{song_id}.npz` |

**Scale:** ~2496 songs.

---

### MERP

**Source:** [MERP — Music Emotion Recognition with Profile Information](https://huggingface.co/datasets/amaai-lab/MERP) (Koh et al., Sensors 2023).

**Raw layout** (under `$XMIDI_STORAGE_DIR/merp/`):

```
audio/{song_id}.wav                           # 54 full-length tracks (e.g. 00_145, deam_115)
annotations_raw.parquet                       # ← we use this
annotations_filtered.parquet                  # NOT used for our pipeline (see below)
songs.json                                    # optional; mark deam_anchor tracks
```

**Raw annotations (`annotations_raw.parquet`):**

| Property | Value |
|----------|-------|
| Format | Long parquet: one row per rater × song; `valence` and `arousal` are **10 Hz arrays** |
| Native rate | **10 Hz** (100 ms steps) |
| Time origin | **0 s** |
| Value range | Raw MTurk slider output (not pre-normalized) |
| Raters | Up to ~87 trials per song (before our QC) |

**Why not `annotations_filtered.parquet`?**

The MERP paper's filtered split applies 7 QC steps. **Step 6 trims each rater's annotation array to OpenSmile feature length** (~15–20% of song duration) for their segment-level BiLSTM models. Example: song `00_145` is 173.5 s audio but filtered annotations cover only ~31.5 s. Our pipeline needs **full-song** V/A aligned to full MIDI and bar latents, so we use **`annotations_raw.parquet`**.

**Our annotation processing** (`datasets/merp.py`):

1. Load `annotations_raw.parquet` (fallback: `annotations_filtered.parquet` only if raw is missing; filtered values are already rescaled/smoothed and need no extra rescaling).
2. **Per-rater min–max rescale** to [-1, 1]: `2·(x − min)/(max − min) − 1` (same rule as MERP's filtered pipeline).
3. **Drop short trials:** remove raters whose series length is **< 80%** of the per-song median length.
4. **Align length:** keep raters with length ≥ per-song median length; truncate all to that median length.
5. **Average** valence and arousal across remaining raters.
6. Convert to `{time_sec: value}` at 10 Hz: `t = i/10` for sample index `i`.

**Not applied:** Savitzky–Golay smoothing (window 15, poly 2) from the MERP filtered release — we use raw + rescale + average only.

**Overlap / splits:**

- 4 MERP tracks duplicate DEAM MEMD excerpts (`deam_115`, `deam_343`, `deam_745`, `deam_1334`).
- These are **blacklisted from all MERP splits** via [`datasets/leakage.py`](datasets/leakage.py) and `MERPDataset.excluded_song_ids()` so the same audio never appears in both DEAM train and MERP val (or vice versa).
- DEAM numeric IDs (`115`, `343`, …) remain in DEAM splits only.

**Derived paths:**

| Artifact | Path |
|----------|------|
| AMT MIDI | `merp_midi/{song_id}.mid` |
| Latents | `merp_va/latents_musetok/{song_id}.safetensors` |
| Continuous V/A | `merp_va/continuous/{song_id}.npz` |

**Scale:** 54 songs (50 FMA + 4 DEAM anchors; ~50 non-excluded for MERP-only splits).

---

### Summary comparison

| | DEAM | Memo2496 | MERP |
|---|------|----------|------|
| **Audio** | 45 s MP3 excerpts | Full MP3 (UUID names) | Full WAV |
| **Native V/A rate** | 2 Hz | 1 Hz | 10 Hz |
| **Annotation start** | 15 s | 0 s | 0 s |
| **Pre-averaged?** | Yes (per song) | Yes (all annotators) | No — we average raters |
| **Extra annotation steps** | None | None | Per-rater rescale; drop short trials; mean |
| **Annotation source file** | `valence.csv`, `arousal.csv` | `valence_all_average.csv`, … | **`annotations_raw.parquet`** |
| **convert_va target rate** | 10 Hz | 10 Hz | 10 Hz |
| **AMT output dir** | `deam/DEAM_midi/MEMD_midi/` | `memo2496_midi/` | `merp_midi/` |
| **Split exclusions** | None | None | DEAM anchor overlap |

---

## Prerequisites

1. **Dependencies:**
   ```bash
   pip install -r va_cont/requirements.txt
   ```

2. **Storage:** `export XMIDI_STORAGE_DIR=/path/to/storage` (default `/deepfreeze/pnlong/gigamidi`)

3. **MuseTok checkpoint:** `$XMIDI_STORAGE_DIR/musetok/best_tokenizer.pt`

4. **YourMT3 setup** (one-time):
   ```bash
   bash yourmt3-cc/setup_local.sh
   source YourMT3/.venv/bin/activate
   ```

---

## Phase 1: Audio → MIDI (YourMT3, local GPU)

Generate config and transcribe (flat `{song_id}.mid` output into each dataset's `midi_dir()`):

```bash
export XMIDI_STORAGE_DIR=/deepfreeze/pnlong/gigamidi

python yourmt3-cc/generate_config.py --dataset deam --output /tmp/deam_amt.json
cd YourMT3 && python inference.py --config_json /tmp/deam_amt.json --device cuda

python yourmt3-cc/generate_config.py --dataset memo2496 --output /tmp/memo2496_amt.json
cd YourMT3 && python inference.py --config_json /tmp/memo2496_amt.json --device cuda

python yourmt3-cc/generate_config.py --dataset merp --output /tmp/merp_amt.json
cd YourMT3 && python inference.py --config_json /tmp/merp_amt.json --device cuda
```

Multi-GPU sharding: run parallel workers with `--gpu_index 0 --num_gpus 3` on `inference.py` (see `yourmt3-cc` docs).

---

## Phase 1b: Alignment QC

```bash
python va_cont/tools/plot_va_alignment.py --dataset deam --song_id 1000 --show-bars
python va_cont/tools/plot_va_alignment.py --dataset merp --song_id 00_145 --show-bars
```

Compare bottom-row curve shapes across columns before bulk conversion.

---

## Phase 2: MuseTok latents

```bash
python va_cont/preprocess/preprocess_musetok.py --dataset deam --gpu --resume
python va_cont/preprocess/preprocess_musetok.py --dataset memo2496 --gpu --resume
python va_cont/preprocess/preprocess_musetok.py --dataset merp --gpu --resume
```

Multi-GPU example (3 workers, one dataset):

```bash
CUDA_VISIBLE_DEVICES=0 python va_cont/preprocess/preprocess_musetok.py --dataset deam --gpu --gpu_index 0 --num_gpus 3 --batch_size 64 --resume
CUDA_VISIBLE_DEVICES=1 python va_cont/preprocess/preprocess_musetok.py --dataset deam --gpu --gpu_index 1 --num_gpus 3 --batch_size 32 --resume
CUDA_VISIBLE_DEVICES=2 python va_cont/preprocess/preprocess_musetok.py --dataset deam --gpu --gpu_index 2 --num_gpus 3 --batch_size 32 --resume
```

---

## Phase 3: V/A conversion (continuous, tick-indexed)

```bash
python va_cont/preprocess/convert_va.py --dataset deam --cache-bar-labels
python va_cont/preprocess/convert_va.py --dataset memo2496 --cache-bar-labels
python va_cont/preprocess/convert_va.py --dataset merp --cache-bar-labels
```

Re-run **without** `--resume` after changing annotation loading (e.g. MERP raw vs filtered).

Each `.npz` contains: `ticks`, `valence`, `arousal`, `tpq`, `bar_resol`.

---

## Phase 4: Combined training

Config [`va_transformer_a.yaml`](pretrain_model/configs/va_transformer_a.yaml):

```yaml
datasets: [deam, memo2496, merp]
```

Sources with missing preprocessed files are skipped with a warning.

```bash
python va_cont/pretrain_model/train.py --config va_cont/pretrain_model/configs/va_transformer_a.yaml
```

Bar labels are derived from continuous `.npz` at load time (or from optional JSON cache).

---

## Phase 5: GigaMIDI annotation (prediction only)

```bash
python va_cont/annotate_gigamidi.py \
  --model_path .../va_transformer_a/checkpoints/best_model.pt \
  --gpu --resume
```

---

## Dataset adapters

Logic lives in [`datasets/`](datasets/): `get_dataset("deam"|"memo2496"|"merp")`.  
See [`datasets/README.md`](datasets/README.md) for the adapter interface.

Shared utilities: [`va_utils.py`](va_utils.py) (`seconds_to_ticks`, `aggregate_va_to_bars`, etc.).

---

## Legacy scripts

Per-dataset scripts still work (`preprocess_deam_musetok.py`, `prepare_labels_deam.py`, …).  
Prefer the generic `--dataset` entry points above.

---

See also [`evaluate_midi.py`](evaluate_midi.py) for single-file inference plots.
