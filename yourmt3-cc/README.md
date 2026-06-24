# YourMT3+ on Compute Canada

Audio-to-MIDI transcription using [YourMT3+](https://arxiv.org/abs/2407.04822), set up for the
**Fir HPC cluster** (Alliance / Compute Canada).

---

## What is YourMT3+?

YourMT3+ is a transformer model that transcribes audio into MIDI. It can work in two modes:

| Mode | Script | Use when |
|------|--------|----------|
| **Single-track** | `transcribe_stems.py` | You have source-separated audio — one file per instrument (bass, guitar, piano…). Each file → one MIDI track. |
| **Multitrack (batch)** | `inference.py` | You have full audio mixes. Each file → one multi-instrument MIDI. |

Each mode uses a different checkpoint, each designed for its task.

---

## Checkpoints

The repo ships five checkpoints. Two are relevant here:

| Checkpoint (short name) | App label | Decoder | Size | Used by |
|---|---|---|---|---|
| `ptf_all_cross_rebal5_..._b100` | YPTF+Single (noPS) | Single-channel | 361 MB | `transcribe_stems.py` (default) |
| `mc13_256_g4_..._b80_ps2` | YPTF.MoE+Multi (PS) | 13 parallel channels | 759 MB | `inference.py` |

### Why two different checkpoints?

The **decoder architecture** is different between them:

- **Single-channel decoder** (`YPTF+Single`): outputs one stream of note events. Trained specifically on single-instrument audio — one instrument at a time. This is the right tool when each audio file is already a separated stem. Smaller and faster (361 MB).

- **13-channel decoder** (`YPTF.MoE+Multi`): outputs 13 parallel streams simultaneously, one per GM instrument family (Piano, Bass, Guitar, Strings, Brass, etc.), then merges them. Trained on full mixes. Best overall quality but designed for full-mix input.

Using the single-channel checkpoint for stems and the multi-channel checkpoint for full mixes is the architecturally correct split. Both checkpoints are downloaded automatically during setup.

All five checkpoints are stored in the HuggingFace Space repo as git LFS objects and pulled by `setup.sh`.

---

## Setup (one-time)

Run from a **login node**:

```bash
cd $SCRATCH
bash yourmt3-cc/setup.sh
```

This will:
1. Load `StdEnv/2023 gcc/12.3 python/3.11`
2. Clone `https://huggingface.co/spaces/mimbres/YourMT3` (downloads ~2 GB checkpoints via git LFS)
3. Create a virtual environment at `$SCRATCH/YourMT3/.venv`
4. Install all dependencies
5. Copy the custom scripts (`model_helper.py`, `transcribe_stems.py`, `inference.py`) into the repo

> **Note:** Step 2 requires internet access. Login nodes on Fir have internet access;
> compute nodes do not. Run `setup.sh` on a login node only.

### Activate the environment

Always load the same modules before using the venv:

```bash
module purge
module load StdEnv/2023 gcc/12.3 python/3.11
source $SCRATCH/YourMT3/.venv/bin/activate
```

---

## Mode 1 — Single-track transcription (`transcribe_stems.py`)

**Use this when:** your audio files are already source-separated — one file per instrument
(e.g. `bass.wav`, `guitar.wav`, `piano.wav`). Each audio file → one MIDI file for that
single instrument.

### Basic usage

```bash
cd $SCRATCH/YourMT3

# Short name for the single-track checkpoint
EXP_SINGLE="ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100"

# Let the model infer which instrument each stem is
python transcribe_stems.py bass.wav guitar.wav piano.wav \
    --exp_id $EXP_SINGLE \
    --output_dir out/midi/

# Tell the model exactly which instrument each stem is
python transcribe_stems.py bass.wav guitar.wav piano.wav \
    --exp_id $EXP_SINGLE \
    --programs 33 27 0 --force_programs \
    --output_dir out/midi/

# Drums + one unknown instrument
python transcribe_stems.py drums.wav synth.wav \
    --exp_id $EXP_SINGLE \
    --programs 128 -1 --is_drum True False \
    --output_dir out/midi/

# Print statistics per stem
python transcribe_stems.py stems/*.wav \
    --exp_id $EXP_SINGLE \
    --output_dir out/midi/ --stats

# See the plan without running
python transcribe_stems.py stems/*.wav \
    --exp_id $EXP_SINGLE --dry_run

# Use the high-quality MoE multi-channel model instead (optional)
EXP_MULTI="mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2"
python transcribe_stems.py stems/*.wav \
    --exp_id $EXP_MULTI --preset mc13_full_plus_256 \
    --output_dir out/midi/
```

### Using a JSON manifest

For large stem lists, create a `stems_manifest.json`:

```json
[
  {"audio": "/scratch/<user>/audio/bass.wav",   "name": "bass",   "program": 33,  "is_drum": false},
  {"audio": "/scratch/<user>/audio/drums.wav",  "name": "drums",  "program": 128, "is_drum": true},
  {"audio": "/scratch/<user>/audio/guitar.wav", "name": "guitar"}
]
```

Then:

```bash
python transcribe_stems.py --stems_json stems_manifest.json \
    --exp_id ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100 \
    --output_dir out/midi/ --stats --stats_out out/stats.json
```

### Key flags

| Flag | Default | Description |
|------|---------|-------------|
| `--exp_id` | required | Checkpoint ID (see Checkpoints section) |
| `--preset` | `yptf_single` | Model architecture preset; use `mc13_full_plus_256` if switching to the MoE model |
| `--programs` | (infer) | GM program per stem; `-1` = let model decide |
| `--is_drum` | (auto) | `True`/`False` per stem; auto-set if `program=128` |
| `--force_programs` | off | Actually override model output; without it, `--programs` is just a label |
| `--output_dir` | stem dir | Where to write MIDI files |
| `--tempo` | 120 | BPM written into output MIDI |
| `--stats` | off | Print note counts per stem |
| `--stats_out` | — | Save stats to JSON |
| `--dry_run` | off | Show plan, skip inference |
| `--list_programs` | — | Print full General MIDI table and exit |

### SLURM (4 parallel MIG slices)

```bash
mkdir -p logs
sbatch slurm/transcribe_stems.sh
```

Edit `STEMS_JSON` and `OUTPUT_DIR` at the top of the script first.

---

## Mode 2 — Multitrack batch transcription (`inference.py`)

**Use this when:** your audio files are full mixes (all instruments together). Each file →
one MIDI containing all detected instruments on separate tracks.

### Create a config JSON

Copy and edit `config.json.example`:

```json
{
    "exp_id": "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2",
    "output": "/scratch/<user>/output/midi",
    "data": [
        "/scratch/<user>/audio/track001/mix.wav",
        "/scratch/<user>/audio/track002/mix.wav"
    ]
}
```

> The `data` paths must follow the pattern `.../TrackXXXX/filename.wav` — the folder name
> two levels up (`TrackXXXX`) is used as the output subdirectory name.

### Run

```bash
cd $SCRATCH/YourMT3
python inference.py --config_json my_config.json --device cuda
```

### SLURM (4 parallel MIG slices)

```bash
mkdir -p logs
sbatch slurm/inference_batch.sh
```

Edit `CONFIG_JSON` and `OUTPUT_DIR` at the top of the script first.

---

## General MIDI program numbers

Common values for `--programs`:

| Program | Instrument |
|---------|------------|
| 0 | Acoustic Grand Piano |
| 25 | Acoustic Guitar (steel) |
| 27 | Electric Guitar (clean) |
| 30 | Distortion Guitar |
| 32 | Acoustic Bass |
| 33 | Electric Bass (finger) |
| 40 | Violin |
| 42 | Cello |
| 56 | Trumpet |
| 65 | Alto Sax |
| 73 | Flute |
| 128 | Drums / Percussion |
| -1 | Let the model decide |

Full table: `python transcribe_stems.py --list_programs`

---

## SLURM notes (Fir cluster)

- Account: `def-pasquier` — **replace with your own account** in the SLURM scripts
- GPU: `nvidia_h100_80gb_hbm3_1g.10gb` is a MIG slice (small, cheap, fast enough for inference)
- No `--partition` needed — the scheduler routes automatically based on time limit
- Max job time on Fir: 7 days; these scripts request 2–3 hours which is plenty for typical datasets
- Check your account name: `sacctmgr show user $USER`

---

## File structure after setup

```
$SCRATCH/YourMT3/
├── .venv/                          ← Python virtual environment
├── transcribe_stems.py             ← single-track entrypoint
├── inference.py                    ← multitrack batch entrypoint
├── model_helper.py                 ← patched model helper
├── amt/
│   ├── src/                        ← YourMT3 source code
│   └── logs/2024/
│       ├── ptf_all_cross_..._b100/
│       │   └── checkpoints/
│       │       └── model.ckpt      ← single-track checkpoint (~361 MB)
│       ├── mc13_256_g4_..._b80_ps2/
│       │   └── checkpoints/
│       │       └── last.ckpt       ← multitrack checkpoint (~759 MB)
│       └── ...                     ← 3 other checkpoints also present
└── ...
```
