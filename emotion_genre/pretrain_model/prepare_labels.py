"""
Prepare emotion and genre labels from XMIDI filenames.

Extracts labels from filenames with format: XMIDI_<Emotion>_<Genre>_<ID_len_8>.midi
Creates label mappings and train/val/test splits.
"""

import os
import json
import argparse
import re
from pathlib import Path
from sklearn.model_selection import train_test_split
import numpy as np
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import XMIDI_LABELS_DIR, ensure_dir, save_json

# Emotion and genre mappings
EMOTIONS = ["exciting", "warm", "happy", "romantic", "funny", "sad", "angry", "lazy", "quiet", "fear", "magnificent"]
GENRES = ["rock", "pop", "country", "jazz", "classical", "traditional"]

EMOTION_TO_INDEX = {emotion: i for i, emotion in enumerate(EMOTIONS)}
GENRE_TO_INDEX = {genre: i for i, genre in enumerate(GENRES)}

INDEX_TO_EMOTION = {i: emotion for i, emotion in enumerate(EMOTIONS)}
INDEX_TO_GENRE = {i: genre for i, genre in enumerate(GENRES)}


def extract_labels_from_filename(filename: str):
    """
    Extract emotion and genre from XMIDI filename.
    
    Format: XMIDI_<Emotion>_<Genre>_<ID_len_8>.midi
    
    Returns:
        (emotion, genre) or (None, None) if parsing fails
    """
    # Remove extension
    name = Path(filename).stem
    
    # Pattern: XMIDI_<Emotion>_<Genre>_<ID_len_8>
    pattern = r'^XMIDI_([^_]+)_([^_]+)_([a-zA-Z0-9]{8})$'
    match = re.match(pattern, name)
    
    if match:
        emotion = match.group(1).lower()
        genre = match.group(2).lower()
        return emotion, genre
    
    # Try alternative patterns
    parts = name.split('_')
    if len(parts) >= 4 and parts[0].upper() == 'XMIDI':
        emotion = parts[1].lower()
        genre = parts[2].lower()
        return emotion, genre
    
    return None, None


def create_splits(filenames: list, 
                  labels: list,
                  test_size: float = 0.1, 
                  val_size: float = 0.1, 
                  random_state: int = 42,
                  stratify: bool = True):
    """
    Create train/val/test splits.
    
    Args:
        filenames: List of filenames
        labels: List of labels (for stratification)
        test_size: Proportion for test set
        val_size: Proportion for validation set (of remaining after test)
        random_state: Random seed
        stratify: If True, use stratified splitting
    
    Returns:
        (train_files, val_files, test_files)
    """
    # First split: train+val vs test
    if stratify and labels:
        train_val_files, test_files = train_test_split(
            filenames,
            test_size=test_size,
            random_state=random_state,
            stratify=labels
        )
        # Get labels for train+val split
        train_val_labels = [labels[filenames.index(f)] for f in train_val_files]
    else:
        train_val_files, test_files = train_test_split(
            filenames,
            test_size=test_size,
            random_state=random_state
        )
        train_val_labels = None
    
    # Second split: train vs val
    if stratify and train_val_labels:
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_size / (1 - test_size),  # Adjust for remaining proportion
            random_state=random_state,
            stratify=train_val_labels
        )
    else:
        train_files, val_files = train_test_split(
            train_val_files,
            test_size=val_size / (1 - test_size),
            random_state=random_state
        )
    
    return train_files, val_files, test_files


def prepare_labels(xmidi_dir: str, 
                  output_dir: str, 
                  test_size: float = 0.1, 
                  val_size: float = 0.1,
                  random_state: int = 42):
    """
    Main function to prepare labels and splits.
    
    Args:
        xmidi_dir: Directory containing XMIDI MIDI files or latents
        output_dir: Output directory for label files
        test_size: Proportion for test set (default: 0.1)
        val_size: Proportion for validation set (default: 0.1)
        random_state: Random seed for splits
    """
    ensure_dir(output_dir)
    
    # Find all .midi files (or use latents directory if it exists)
    xmidi_path = Path(xmidi_dir)
    files = []
    
    # Check if latents directory exists (files already processed)
    latents_dir = xmidi_path / "latents"
    if latents_dir.exists():
        # Use latents directory - look for .safetensors files
        for file_path in latents_dir.glob("*.safetensors"):
            files.append(file_path.stem)
    else:
        # Use MIDI files directly
        for ext in ['*.midi', '*.mid', '*.MIDI', '*.MID']:
            for file_path in xmidi_path.rglob(ext):
                files.append(file_path.stem)
    
    if len(files) == 0:
        raise ValueError(f"No files found in {xmidi_dir}")
    
    print(f"Found {len(files)} files")
    
    # Extract labels from filenames
    emotion_labels = {}
    genre_labels = {}
    valid_files = []
    emotion_list = []
    genre_list = []
    
    for filename in files:
        emotion, genre = extract_labels_from_filename(filename)
        
        if emotion is None or genre is None:
            print(f"Warning: Could not parse filename: {filename}")
            continue
        
        # Validate emotion and genre
        if emotion not in EMOTION_TO_INDEX:
            print(f"Warning: Unknown emotion '{emotion}' in {filename}, skipping")
            continue
        
        if genre not in GENRE_TO_INDEX:
            print(f"Warning: Unknown genre '{genre}' in {filename}, skipping")
            continue
        
        # Add to labels
        emotion_labels[filename] = EMOTION_TO_INDEX[emotion]
        genre_labels[filename] = GENRE_TO_INDEX[genre]
        valid_files.append(filename)
        emotion_list.append(EMOTION_TO_INDEX[emotion])
        genre_list.append(GENRE_TO_INDEX[genre])
    
    print(f"Valid files: {len(valid_files)}")
    print(f"Emotion distribution: {np.bincount(emotion_list)}")
    print(f"Genre distribution: {np.bincount(genre_list)}")
    
    # Save label files
    emotion_labels_path = os.path.join(output_dir, "emotion_labels.json")
    genre_labels_path = os.path.join(output_dir, "genre_labels.json")
    
    save_json(emotion_labels_path, emotion_labels)
    save_json(genre_labels_path, genre_labels)
    print(f"Saved emotion labels to {emotion_labels_path}")
    print(f"Saved genre labels to {genre_labels_path}")
    
    # Save class-to-index mappings
    emotion_to_index_path = os.path.join(output_dir, "emotion_to_index.json")
    genre_to_index_path = os.path.join(output_dir, "genre_to_index.json")
    
    save_json(emotion_to_index_path, EMOTION_TO_INDEX)
    save_json(genre_to_index_path, GENRE_TO_INDEX)
    print(f"Saved emotion_to_index to {emotion_to_index_path}")
    print(f"Saved genre_to_index to {genre_to_index_path}")
    
    # Create splits (stratified by emotion for emotion task, by genre for genre task)
    print("\nCreating splits...")
    
    # Emotion splits (stratified by emotion)
    train_emotion, val_emotion, test_emotion = create_splits(
        valid_files, emotion_list, test_size, val_size, random_state, stratify=True
    )
    
    # Genre splits (stratified by genre)
    train_genre, val_genre, test_genre = create_splits(
        valid_files, genre_list, test_size, val_size, random_state, stratify=True
    )
    
    # Use same splits for both tasks (use emotion-based splits as default)
    # Alternatively, could use separate splits for each task
    train_files = train_emotion
    val_files = val_emotion
    test_files = test_emotion
    
    print(f"Train: {len(train_files)} files")
    print(f"Val: {len(val_files)} files")
    print(f"Test: {len(test_files)} files")
    
    # Save split files
    train_path = os.path.join(output_dir, "train_files.txt")
    val_path = os.path.join(output_dir, "val_files.txt")
    test_path = os.path.join(output_dir, "test_files.txt")
    
    with open(train_path, 'w') as f:
        f.write('\n'.join(train_files))
    with open(val_path, 'w') as f:
        f.write('\n'.join(val_files))
    with open(test_path, 'w') as f:
        f.write('\n'.join(test_files))
    
    print(f"Saved split files to {output_dir}")
    
    # Print class distribution in splits
    print("\nEmotion distribution in splits:")
    train_emotions = [emotion_list[valid_files.index(f)] for f in train_files]
    val_emotions = [emotion_list[valid_files.index(f)] for f in val_files]
    test_emotions = [emotion_list[valid_files.index(f)] for f in test_files]
    print(f"  Train: {np.bincount(train_emotions, minlength=len(EMOTIONS))}")
    print(f"  Val: {np.bincount(val_emotions, minlength=len(EMOTIONS))}")
    print(f"  Test: {np.bincount(test_emotions, minlength=len(EMOTIONS))}")
    
    print("\nGenre distribution in splits:")
    train_genres = [genre_list[valid_files.index(f)] for f in train_files]
    val_genres = [genre_list[valid_files.index(f)] for f in val_files]
    test_genres = [genre_list[valid_files.index(f)] for f in test_files]
    print(f"  Train: {np.bincount(train_genres, minlength=len(GENRES))}")
    print(f"  Val: {np.bincount(val_genres, minlength=len(GENRES))}")
    print(f"  Test: {np.bincount(test_genres, minlength=len(GENRES))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare emotion and genre labels from XMIDI filenames")
    parser.add_argument("--xmidi_dir", required=True,
                       help="Directory containing XMIDI MIDI files or latents")
    parser.add_argument("--output_dir", default=XMIDI_LABELS_DIR,
                       help="Output directory for label files")
    parser.add_argument("--test_size", type=float, default=0.1,
                       help="Proportion for test set (default: 0.1)")
    parser.add_argument("--val_size", type=float, default=0.1,
                       help="Proportion for validation set (default: 0.1)")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for splits (default: 42)")
    args = parser.parse_args()
    
    prepare_labels(
        args.xmidi_dir, args.output_dir, 
        args.test_size, args.val_size, args.random_state
    )
