# GigaMIDI Annotation Project

This repository contains code for predicting continuous Valence and Arousal values at the bar level for the GigaMIDI dataset using MuseTok latents.

## Project Structure

### `valence_arousal/`
The main emotion recognition pipeline for predicting continuous valence/arousal values. This directory contains:

- **Preprocessing**: Extract MuseTok latents from EMOPIA dataset and prepare continuous labels
- **Training**: Train an MLP model to predict valence/arousal from MuseTok latents
- **Annotation**: Apply the trained model to GigaMIDI to generate bar-level emotion annotations
- **Analysis**: Analyze and visualize the predicted annotations across genres/styles

See `scratchpaper/revised_plan.md` and `scratchpaper/implementation_plan.md` for detailed documentation.

### `jingyue_latents/`
Previous implementation of emotion recognition for the MuseTok paper. This codebase was used for categorical emotion classification on EMOPIA dataset at the sequence level. 

**Source**: [https://github.com/pnlong/jingyue_latents](https://github.com/pnlong/jingyue_latents)

**Purpose**: Reference implementation for MIR tasks (emotion recognition, chord detection, melody classification) using MuseTok latents. The `valence_arousal/` directory adapts this codebase for continuous regression instead of categorical classification.

### `musetok/`
The MuseTok symbolic music tokenization framework. This is the core library used for extracting latent representations from MIDI files.

**Source**: [https://github.com/Yuer867/MuseTok](https://github.com/Yuer867/MuseTok)

**Purpose**: Provides pre-trained models and utilities for converting MIDI files to REMI events and extracting RVQ (Residual Vector Quantization) latents. The `valence_arousal/` pipeline uses MuseTok to extract bar-level latents for emotion prediction.

### `scratchpaper/`
Markdown files containing project notes, plans, and documentation:

- `plan.md`: Initial project plan and notes
- `revised_plan.md`: Structured plan with detailed pipeline steps and codebase organization
- `implementation_plan.md`: Step-by-step implementation guide with code examples

## Quick Start

1. **Set up environment**:
   ```bash
   mamba create -n gigamidi python=3.10
   mamba activate gigamidi
   pip install -r valence_arousal/requirements.txt
   ```

2. **Configure storage directory** (see `valence_arousal/utils/data_utils.py`):
   ```bash
   export EMOTION_RECOGNITION_STORAGE_DIR="/path/to/storage"
   ```

3. **Follow the pipeline** (see `scratchpaper/revised_plan.md` for full instructions):
   - Preprocess EMOPIA dataset
   - Train valence/arousal prediction model
   - Annotate GigaMIDI dataset
   - Analyze results

## Future Tasks

Additional MIR tasks can be added as new subdirectories following the same structure as `valence_arousal/`.

## References

- **MuseTok**: [Yuer867/MuseTok](https://github.com/Yuer867/MuseTok) - Symbolic Music Tokenization for Generation and Semantic Understanding
- **jingyue_latents**: [pnlong/jingyue_latents](https://github.com/pnlong/jingyue_latents) - MIR Tasks for Jingyue's Tokenization Project
- **GigaMIDI**: [Metacreation/GigaMIDI](https://huggingface.co/datasets/Metacreation/GigaMIDI) - Large-scale symbolic music dataset

# gigamidi_support
