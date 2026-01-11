"""
Prepare continuous valence/arousal labels from EMOPIA.

ASSUMPTIONS ABOUT EMOPIA LABEL STRUCTURE:
=========================================
1. Label Storage Format:
   - Labels might be in a CSV file (e.g., 'labels.csv', 'metadata.csv', 'emopia_labels.csv')
   - Labels might be in a JSON file (e.g., 'labels.json', 'metadata.json')
   - Labels might be encoded in filenames (e.g., 'happy_song1.mid', 'Q1_song1.mid')
   - Labels might be in a separate directory structure matching MIDI files
   
2. Label Format:
   - Emotion labels are categorical: 'happy', 'angry', 'sad', 'relax' (case-insensitive)
   - Labels are per-song (not per-bar), so all bars in a song share the same emotion
   - Each MIDI file has exactly one emotion label
   
3. File Matching:
   - Labels are matched to MIDI files by filename (without extension)
   - Filenames are unique identifiers
   - If labels are in a CSV/JSON, there's a column/key for filename and emotion
   
4. Output Format:
   - JSON file with structure: {filename: {"valence": float, "arousal": float}}
   - If per_bar=True, structure: {filename: [{"bar": int, "valence": float, "arousal": float}, ...]}
   - Filenames are without extension to match MIDI file stems
   
5. Missing Labels:
   - If a MIDI file has no label, it's skipped (with warning)
   - Unknown emotion labels are skipped (with warning)

TODO: Update these assumptions as you learn more about EMOPIA's actual label structure.
"""

import os
import json
import argparse
import logging
from pathlib import Path
import sys
import csv
import glob

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import EMOPIA_LABELS_DIR, ensure_dir, save_json, load_json

# Emotion to VA mapping (Russell's circumplex model)
EMOTION_TO_VA = {
    "happy": {"valence": 0.8, "arousal": 0.6},
    "angry": {"valence": -0.6, "arousal": 0.8},
    "sad": {"valence": -0.8, "arousal": -0.4},
    "relax": {"valence": 0.4, "arousal": -0.6},
}

# Normalize emotion names (case-insensitive, handle variations)
EMOTION_ALIASES = {
    "happy": ["happy", "happiness", "joy", "joyful"],
    "angry": ["angry", "anger", "mad"],
    "sad": ["sad", "sadness", "sorrow"],
    "relax": ["relax", "relaxed", "relaxing", "calm", "peaceful"],
}


def normalize_emotion(emotion_str: str) -> str:
    """
    Normalize emotion string to one of the four EMOPIA emotions.
    
    Args:
        emotion_str: Emotion string (case-insensitive)
    
    Returns:
        Normalized emotion name or None if not recognized
    """
    if not emotion_str:
        return None
    
    emotion_lower = emotion_str.lower().strip()
    
    # Direct match
    if emotion_lower in EMOTION_TO_VA:
        return emotion_lower
    
    # Check aliases
    for canonical, aliases in EMOTION_ALIASES.items():
        if emotion_lower in aliases:
            return canonical
    
    return None


def load_labels_from_csv(emopia_dir: str) -> dict:
    """
    Try to load labels from CSV files in emopia_dir.
    
    Assumes CSV has columns: filename (or similar) and emotion (or similar).
    """
    labels = {}
    csv_files = []
    
    # Look for common CSV filenames
    for pattern in ['*.csv', 'labels*.csv', 'metadata*.csv', '*labels*.csv']:
        csv_files.extend(glob.glob(os.path.join(emopia_dir, pattern)))
        # Also check subdirectories
        csv_files.extend(glob.glob(os.path.join(emopia_dir, '**', pattern), recursive=True))
    
    for csv_path in csv_files:
        try:
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Try to find filename and emotion columns (case-insensitive)
                    filename = None
                    emotion = None
                    
                    for key, value in row.items():
                        key_lower = key.lower()
                        if 'file' in key_lower or 'name' in key_lower or 'id' in key_lower:
                            filename = value
                        elif 'emotion' in key_lower or 'label' in key_lower or 'class' in key_lower:
                            emotion = value
                    
                    if filename and emotion:
                        filename_stem = Path(filename).stem
                        emotion_norm = normalize_emotion(emotion)
                        if emotion_norm:
                            labels[filename_stem] = emotion_norm
                        else:
                            logging.warning(f"Unknown emotion '{emotion}' for file '{filename}'")
        except Exception as e:
            logging.warning(f"Error reading CSV {csv_path}: {e}")
    
    return labels


def load_labels_from_json(emopia_dir: str) -> dict:
    """
    Try to load labels from JSON files in emopia_dir.
    
    Assumes JSON has structure: {filename: emotion} or [{filename: ..., emotion: ...}, ...]
    """
    labels = {}
    json_files = []
    
    # Look for common JSON filenames
    for pattern in ['*.json', 'labels*.json', 'metadata*.json', '*labels*.json']:
        json_files.extend(glob.glob(os.path.join(emopia_dir, pattern)))
        json_files.extend(glob.glob(os.path.join(emopia_dir, '**', pattern), recursive=True))
    
    for json_path in json_files:
        try:
            data = load_json(json_path)
            
            if isinstance(data, dict):
                # Format: {filename: emotion} or {filename: {emotion: ...}}
                for key, value in data.items():
                    filename_stem = Path(key).stem
                    if isinstance(value, str):
                        emotion_norm = normalize_emotion(value)
                        if emotion_norm:
                            labels[filename_stem] = emotion_norm
                    elif isinstance(value, dict) and 'emotion' in value:
                        emotion_norm = normalize_emotion(value['emotion'])
                        if emotion_norm:
                            labels[filename_stem] = emotion_norm
            
            elif isinstance(data, list):
                # Format: [{filename: ..., emotion: ...}, ...]
                for item in data:
                    if isinstance(item, dict):
                        filename = item.get('filename') or item.get('file') or item.get('id')
                        emotion = item.get('emotion') or item.get('label') or item.get('class')
                        if filename and emotion:
                            filename_stem = Path(filename).stem
                            emotion_norm = normalize_emotion(emotion)
                            if emotion_norm:
                                labels[filename_stem] = emotion_norm
        except Exception as e:
            logging.warning(f"Error reading JSON {json_path}: {e}")
    
    return labels


def load_labels_from_filenames(emopia_dir: str) -> dict:
    """
    Try to extract labels from MIDI filenames.
    
    Assumes format like: 'happy_song1.mid', 'Q1_song1.mid', 'song1_happy.mid', etc.
    """
    labels = {}
    midi_files = []
    
    for ext in ['*.mid', '*.midi', '*.MID', '*.MIDI']:
        midi_files.extend(glob.glob(os.path.join(emopia_dir, ext)))
        midi_files.extend(glob.glob(os.path.join(emopia_dir, '**', ext), recursive=True))
    
    for midi_path in midi_files:
        filename = Path(midi_path).stem
        
        # Try to extract emotion from filename
        # Check if emotion appears in filename
        for emotion in EMOTION_TO_VA.keys():
            if emotion.lower() in filename.lower():
                labels[filename] = emotion
                break
        
        # Check for Q1, Q2, Q3, Q4 format (mapping to emotions)
        if 'Q1' in filename.upper():
            labels[filename] = 'happy'
        elif 'Q2' in filename.upper():
            labels[filename] = 'angry'
        elif 'Q3' in filename.upper():
            labels[filename] = 'sad'
        elif 'Q4' in filename.upper():
            labels[filename] = 'relax'
    
    return labels


def load_emopia_labels(emopia_dir: str) -> dict:
    """
    Load EMOPIA emotion labels from various possible formats.
    
    Tries multiple methods in order:
    1. CSV files
    2. JSON files
    3. Filename patterns
    
    Args:
        emopia_dir: EMOPIA dataset directory
    
    Returns:
        dict mapping filename (without extension) to normalized emotion name
    """
    labels = {}
    
    logging.info("Attempting to load labels from CSV files...")
    csv_labels = load_labels_from_csv(emopia_dir)
    if csv_labels:
        labels.update(csv_labels)
        logging.info(f"Loaded {len(csv_labels)} labels from CSV")
    
    logging.info("Attempting to load labels from JSON files...")
    json_labels = load_labels_from_json(emopia_dir)
    if json_labels:
        labels.update(json_labels)
        logging.info(f"Loaded {len(json_labels)} labels from JSON")
    
    # Only try filenames if we haven't found many labels
    if len(labels) < 10:
        logging.info("Attempting to extract labels from filenames...")
        filename_labels = load_labels_from_filenames(emopia_dir)
        if filename_labels:
            labels.update(filename_labels)
            logging.info(f"Extracted {len(filename_labels)} labels from filenames")
    
    return labels


def prepare_labels(emopia_dir: str, output_path: str, per_bar: bool = False, latents_dir: str = None):
    """
    Prepare continuous VA labels from EMOPIA.
    
    Args:
        emopia_dir: EMOPIA dataset directory
        output_path: Output JSON file path
        per_bar: Whether to create per-bar labels (requires latents_dir to get bar counts)
        latents_dir: Directory containing preprocessed latents (needed for per_bar=True)
    """
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # 1. Load EMOPIA labels
    logging.info(f"Loading labels from {emopia_dir}...")
    emotion_labels = load_emopia_labels(emopia_dir)
    
    if not emotion_labels:
        logging.error("No labels found! Check emopia_dir and label file formats.")
        return
    
    logging.info(f"Loaded {len(emotion_labels)} emotion labels")
    
    # 2. Map emotions to VA values
    va_labels = {}
    missing_va = 0
    
    for filename, emotion in emotion_labels.items():
        if emotion in EMOTION_TO_VA:
            va = EMOTION_TO_VA[emotion]
            va_labels[filename] = va.copy()
        else:
            missing_va += 1
            logging.warning(f"Unknown emotion '{emotion}' for file '{filename}'")
    
    if missing_va > 0:
        logging.warning(f"Skipped {missing_va} files with unknown emotions")
    
    # 3. Handle per-bar labels if requested
    if per_bar:
        if not latents_dir:
            logging.error("per_bar=True requires --latents_dir to get bar counts")
            return
        
        logging.info("Creating per-bar labels...")
        from utils.data_utils import load_latents
        
        per_bar_labels = {}
        missing_latents = 0
        
        for filename, va in va_labels.items():
            # Try to find corresponding latent file
            latent_path = None
            for ext in ['.safetensors', '.npy']:
                candidate = os.path.join(latents_dir, filename + ext)
                if os.path.exists(candidate):
                    latent_path = candidate
                    break
            
            if latent_path:
                try:
                    latents, metadata = load_latents(latent_path)
                    n_bars = len(latents) if latents is not None else metadata.get('n_bars', 0)
                    
                    # Create per-bar labels (same VA for all bars)
                    per_bar_labels[filename] = [
                        {"bar": i, "valence": va["valence"], "arousal": va["arousal"]}
                        for i in range(n_bars)
                    ]
                except Exception as e:
                    logging.warning(f"Error loading latents for {filename}: {e}")
                    missing_latents += 1
            else:
                logging.warning(f"No latent file found for {filename}")
                missing_latents += 1
        
        if missing_latents > 0:
            logging.warning(f"Could not create per-bar labels for {missing_latents} files")
        
        va_labels = per_bar_labels
    
    # 4. Save to JSON
    ensure_dir(os.path.dirname(output_path))
    save_json(output_path, va_labels)
    
    logging.info(f"\nSaved {len(va_labels)} VA labels to {output_path}")
    if per_bar:
        total_bars = sum(len(bars) for bars in va_labels.values())
        logging.info(f"Total bars: {total_bars}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare continuous VA labels from EMOPIA emotion labels"
    )
    parser.add_argument("--emopia_dir", required=True,
                       help="EMOPIA dataset directory")
    parser.add_argument("--output_path", type=str, default=None,
                       help="Output JSON file path (defaults to EMOPIA_LABELS_DIR/va_labels.json)")
    parser.add_argument("--per_bar", action="store_true", default=False,
                       help="Create per-bar labels (requires --latents_dir)")
    parser.add_argument("--latents_dir", type=str, default=None,
                       help="Directory containing preprocessed latents (needed for per_bar=True)")
    
    args = parser.parse_args()
    
    if args.output_path is None:
        ensure_dir(EMOPIA_LABELS_DIR)
        args.output_path = os.path.join(EMOPIA_LABELS_DIR, "va_labels.json")
    
    prepare_labels(
        args.emopia_dir,
        args.output_path,
        args.per_bar,
        args.latents_dir
    )
