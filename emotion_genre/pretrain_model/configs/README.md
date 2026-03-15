# Training / evaluation configs

YAML configs provide default arguments for training and evaluation scripts. Any CLI argument overrides the config value.

## Usage

Configs are pre-filled with `/deepfreeze/pnlong/gigamidi/xmidi_emotion_genre/xmidi_data/` and the standard checkpoints path. Override any path via CLI.

**Helper scripts** (from the `emotion_genre` directory):
- `train.sh <config.yml> [args...]` — calls `train.py` for emotion/genre configs, `train_va.py` for valence–arousal. Extra args passed through (e.g. `--gpu`).
- `evaluate.sh <config.yml> [args...]` — calls `evaluate.py` or `evaluate_va.py` accordingly. Pass `--checkpoint_path` and `--test_files` (or rely on config). Extra args passed through (e.g. `--gpu`).

```bash
# Train (from emotion_genre/)
./train.sh pretrain_model/configs/emotion_musetok.yml --gpu
./train.sh pretrain_model/configs/valence_arousal_bars4.yml --gpu

# Evaluate (pass checkpoint and test files; paths can come from config)
./evaluate.sh pretrain_model/configs/emotion_musetok.yml --checkpoint_path .../best_model.pt --test_files .../labels/test_files.txt --gpu
./evaluate.sh pretrain_model/configs/valence_arousal_bars4.yml --checkpoint_path .../valence_arousal_regressor_bars4/checkpoints/best_model.pt --test_files .../labels/test_files.txt --gpu

# Or call scripts directly
python pretrain_model/train.py --config pretrain_model/configs/emotion_musetok.yml
python pretrain_model/evaluate.py --config pretrain_model/configs/emotion_musetok.yml \
  --checkpoint_path /path/to/best_model.pt --test_files /path/to/test_files.txt
```

## Config files

- **emotion_musetok.yml**, **genre_musetok.yml**: MuseTok latents (per-bar), song-level or `bars_per_chunk`.
- **emotion_midi2vec.yml**, **genre_midi2vec.yml**: midi2vec latents (song-level only).
- **emotion_combined.yml**, **genre_combined.yml**: Single combined latents dir (from `combine_latents.py`); `input_dim` inferred.
- **valence_arousal_musetok.yml**, **valence_arousal_midi2vec.yml**: Valence–arousal regression (song-level).
- **valence_arousal_bars4.yml**: Valence–arousal with 4 bars per chunk (MuseTok latents).
- **emotion_bootstrap.yml**, **genre_bootstrap.yml**: Bootstrap downsampling (e.g. 10 folds).
- **emotion_bars4_bootstrap.yml**, **genre_bars4_bootstrap.yml**: Bar-level (4 bars per chunk) + bootstrap; MuseTok latents only.
