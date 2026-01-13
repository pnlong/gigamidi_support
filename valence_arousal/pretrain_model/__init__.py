"""
Pretraining module for continuous valence/arousal prediction.

This module contains the complete pipeline for training a model to predict continuous
valence and arousal values from MuseTok latent representations of musical pieces.

Main Components:
- preprocess_emopia.py: Extract MuseTok latents from EMOPIA dataset (supports both
  edited EMOPIA .pkl files and EMOPIA+ MIDI files)
- prepare_labels.py: Convert EMOPIA categorical emotion labels (Q1-Q4) to continuous
  valence/arousal values
- dataset.py: PyTorch Dataset class for loading latents and VA labels
- model.py: MLP model architecture for VA prediction
- train.py: Training script with wandb integration
- evaluate.py: Evaluation script for model performance metrics
"""
