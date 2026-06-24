# Continuous Valence/Arousal Pipeline

Unified pipeline for **DEAM + Memo2496 + MERP** combined training.  
Canonical V/A storage is **tick-indexed continuous** (`.npz`); bar-level targets are derived at train time for MuseTok.

## Quick reference

| Step | Command |
|------|---------|
| Verify data | `python va_cont/tools/verify_datasets.py` |
| AMT config | `python yourmt3-cc/generate_config.py --dataset deam --output deam_amt.json` |
| MuseTok latents | `python va_cont/preprocess/preprocess_musetok.py --dataset deam --gpu --resume` |
| V/A → ticks | `python va_cont/preprocess/convert_va.py --dataset deam --cache-bar-labels` |
| QC plot | `python va_cont/tools/plot_va_alignment.py --dataset deam --song_id 1000` |
| Train (combined) | `python va_cont/pretrain_model/train.py --config va_cont/pretrain_model/configs/va_transformer_a.yaml` |
| GigaMIDI annotate | `python va_cont/annotate_gigamidi.py --model_path .../best_model.pt --gpu` |

Repeat preprocess/convert steps for `memo2496` and `merp`.

## Prerequisites

1. **Dependencies** (from repo root or `va_cont/`):
   ```bash
   pip install -r va_cont/requirements.txt
   ```

2. **Storage**: `export XMIDI_STORAGE_DIR=/path/to/storage` (default `/deepfreeze/pnlong/gigamidi`)

3. **MuseTok checkpoint**: `$XMIDI_STORAGE_DIR/musetok/best_tokenizer.pt`

## Dataset layout

### DEAM
```
$XMIDI_STORAGE_DIR/deam/
├── DEAM_audio/MEMD_audio/{song_id}.mp3
├── DEAM_midi/MEMD_midi/{song_id}.mid          # after YourMT3 AMT
└── DEAM_Annotations/annotations/.../valence.csv, arousal.csv
$XMIDI_STORAGE_DIR/deam_va/
├── latents_musetok/{song_id}.safetensors
├── continuous/{song_id}.npz                   # ticks, valence, arousal, tpq, bar_resol
└── labels/train_songs.txt, val_songs.txt, ...
```

### Memo2496
```
$XMIDI_STORAGE_DIR/memo2496/MusicRawData/{uuid}.mp3
$XMIDI_STORAGE_DIR/memo2496_midi/{uuid}.mid
$XMIDI_STORAGE_DIR/memo2496/valence_all_average.csv, arousal_all_average.csv
$XMIDI_STORAGE_DIR/memo2496_va/...
```

### MERP
```
$XMIDI_STORAGE_DIR/merp/
├── audio/{song_id}.wav
├── annotations/               # parquet or CSV (see datasets/merp.py)
└── songs.json               # optional; mark "deam_anchor": true for overlap tracks
$XMIDI_STORAGE_DIR/merp_midi/{song_id}.mid
$XMIDI_STORAGE_DIR/merp_va/...
```

## Phase 1: Audio → MIDI (YourMT3, local GPU)

One-time setup (from repo root):

```bash
bash yourmt3-cc/setup_local.sh
source YourMT3/.venv/bin/activate
```

Generate config and transcribe (flat `{song_id}.mid` output into each dataset's `midi_dir`):

```bash
export XMIDI_STORAGE_DIR=/deepfreeze/pnlong/gigamidi

python yourmt3-cc/generate_config.py --dataset deam --output /tmp/deam_amt.json
cd YourMT3 && python inference.py --config_json /tmp/deam_amt.json --device cuda

python yourmt3-cc/generate_config.py --dataset memo2496 --output /tmp/memo2496_amt.json
cd YourMT3 && python inference.py --config_json /tmp/memo2496_amt.json --device cuda
```

You have 3× GPUs locally (3090 + 2× 2080 Ti). Shard with `--gpu_index 0 --num_gpus 3` on parallel workers if desired.

DEAM (~1802 clips) will take several hours on a 3090; Memo2496 (~2496 songs) longer.

## Phase 1b: Alignment QC

2×2 plot: left column = audio seconds; right column = MIDI ticks (no tick→second conversion on the right).

```bash
python va_cont/tools/plot_va_alignment.py --dataset deam --song_id 1000 --show-bars
```

Compare bottom-row curve **shapes** across columns before bulk conversion.

## Phase 2: MuseTok latents

```bash
python va_cont/preprocess/preprocess_musetok.py --dataset deam --gpu --resume
python va_cont/preprocess/preprocess_musetok.py --dataset memo2496 --gpu --resume
python va_cont/preprocess/preprocess_musetok.py --dataset merp --gpu --resume
```

## Phase 3: V/A conversion (continuous, tick-indexed)

```bash
python va_cont/preprocess/convert_va.py --dataset deam --cache-bar-labels
python va_cont/preprocess/convert_va.py --dataset memo2496 --cache-bar-labels
python va_cont/preprocess/convert_va.py --dataset merp --cache-bar-labels
```

Each `.npz` contains: `ticks`, `valence`, `arousal`, `tpq`, `bar_resol`.

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

## Phase 5: GigaMIDI annotation (prediction only)

```bash
python va_cont/annotate_gigamidi.py \
  --model_path .../va_transformer_a/checkpoints/best_model.pt \
  --gpu --resume
```

## Dataset adapters

Logic lives in [`datasets/`](datasets/): `get_dataset("deam"|"memo2496"|"merp")`.

Shared utilities: [`va_utils.py`](va_utils.py) (`seconds_to_ticks`, `aggregate_va_to_bars`, etc.).

## Legacy scripts

Per-dataset scripts still work (`preprocess_deam_musetok.py`, `prepare_labels_deam.py`, …).  
Prefer the generic `--dataset` entry points above.

---

See also [`evaluate_midi.py`](evaluate_midi.py) for single-file inference plots.
