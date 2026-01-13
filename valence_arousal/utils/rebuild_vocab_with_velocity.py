"""
Rebuild MuseTok vocabulary dictionaries (with and without velocity).

This script builds both vocabulary dictionaries:
1. dictionary.pkl - without velocity support (167 events, matches original)
2. dictionary_with_velocity.pkl - with velocity support (209 events = 167 + 42 velocity)

The vocabulary without velocity recreates the original 167-event dictionary by combining:
- build_full_vocab(add_velocity=False): 140 events (theoretical vocabulary)
- 27 additional events from original training data (21 Beat + 6 Time_Signature events)

The vocabulary with velocity adds 42 velocity tokens to the 167-event base vocabulary.
"""

import os
import sys
import pickle
import numpy as np

# Add musetok to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'musetok'))
from data_processing.events2words import build_full_vocab

# Import our utilities (now in same directory)
# Use absolute import when running as script, relative when imported as module
try:
    from .data_utils import ensure_dir
except ImportError:
    # Running as script
    sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
    from utils.data_utils import ensure_dir

# Get musetok path
MUSETOK_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'musetok')
MUSETOK_PATH = os.path.abspath(MUSETOK_PATH)

# The 27 additional events from original training data that are not in build_full_vocab
# These appeared in the training data and were included in the original 167-event dictionary
EXTRA_TRAINING_EVENTS = [
    # 21 additional Beat events
    "Beat_102", "Beat_48", "Beat_51", "Beat_54", "Beat_56", "Beat_57",
    "Beat_60", "Beat_63", "Beat_64", "Beat_66", "Beat_69", "Beat_72",
    "Beat_78", "Beat_80", "Beat_81", "Beat_84", "Beat_87", "Beat_88",
    "Beat_90", "Beat_93", "Beat_96",
    # 6 additional Time_Signature events
    "Time_Signature_12/8", "Time_Signature_3/2", "Time_Signature_4/2",
    "Time_Signature_6/4", "Time_Signature_9/4", "Time_Signature_9/8",
]


def build_vocabulary_without_velocity():
    """
    Build MuseTok vocabulary dictionary without velocity (167 events).
    
    This recreates the original dictionary by combining:
    - build_full_vocab(add_velocity=False): 140 events
    - 27 additional events from original training data
    
    Returns:
        tuple: (event2word, word2event) dictionaries with 167 events
    """
    # Get theoretical vocabulary from build_full_vocab
    full_vocab = build_full_vocab(add_velocity=False)
    
    # Add the 27 extra events from original training data
    all_events = list(full_vocab) + EXTRA_TRAINING_EVENTS
    
    # Create unique sorted set (this matches how events2dictionary works)
    unique_events = sorted(set(all_events), key=lambda x: (not isinstance(x, int), x))
    event2word = {key: i for i, key in enumerate(unique_events)}
    word2event = {i: key for i, key in enumerate(unique_events)}
    
    return event2word, word2event


def build_vocabulary_with_velocity(base_vocab_without_velocity):
    """
    Build MuseTok vocabulary dictionary with velocity (209 events).
    
    This adds 42 velocity tokens to the base 167-event vocabulary.
    
    Args:
        base_vocab_without_velocity: (event2word, word2event) tuple from build_vocabulary_without_velocity
    
    Returns:
        tuple: (event2word, word2event) dictionaries with 209 events (167 + 42 velocity)
    """
    base_event2word, base_word2event = base_vocab_without_velocity
    
    # Generate velocity tokens (42 bins from 4-127)
    velocity_tokens = [f"Note_Velocity_{int(v)}" for v in np.linspace(4, 127, 42, dtype=int)]
    
    # Combine base vocabulary with velocity tokens
    all_events = list(base_event2word.keys()) + velocity_tokens
    
    # Create unique sorted set
    unique_events = sorted(set(all_events), key=lambda x: (not isinstance(x, int), x))
    event2word = {key: i for i, key in enumerate(unique_events)}
    word2event = {i: key for i, key in enumerate(unique_events)}
    
    return event2word, word2event


def rebuild_both_vocabularies():
    """
    Build both vocabulary dictionaries (with and without velocity).
    
    Returns:
        tuple: ((event2word_without, word2event_without), (event2word_with, word2event_with))
    """
    # Build vocabulary without velocity first (167 events, matches original)
    vocab_without_velocity = build_vocabulary_without_velocity()
    
    # Build vocabulary with velocity by adding velocity tokens to the base (209 events)
    vocab_with_velocity = build_vocabulary_with_velocity(vocab_without_velocity)
    
    return vocab_without_velocity, vocab_with_velocity


def save_vocabularies(vocab_without_velocity=None, vocab_with_velocity=None, 
                      output_dir: str = None):
    """
    Save both vocabulary dictionaries to disk.
    
    Args:
        vocab_without_velocity: (event2word, word2event) tuple for vocabulary without velocity (default)
        vocab_with_velocity: (event2word, word2event) tuple for vocabulary with velocity
        output_dir: Directory to save vocabularies (defaults to musetok/data/)
    """
    if output_dir is None:
        output_dir = os.path.join(MUSETOK_PATH, "data")
    
    ensure_dir(output_dir)
    
    # Save vocabulary without velocity (default dictionary.pkl)
    if vocab_without_velocity is not None:
        output_path = os.path.join(output_dir, "dictionary.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(vocab_without_velocity, f)
        vocab_size = len(vocab_without_velocity[0])
        print(f"Saved dictionary.pkl: {vocab_size} events (n_token={vocab_size + 1})")
    
    # Save vocabulary with velocity
    if vocab_with_velocity is not None:
        output_path = os.path.join(output_dir, "dictionary_with_velocity.pkl")
        with open(output_path, 'wb') as f:
            pickle.dump(vocab_with_velocity, f)
        vocab_size = len(vocab_with_velocity[0])
        print(f"Saved dictionary_with_velocity.pkl: {vocab_size} events (n_token={vocab_size + 1})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Rebuild MuseTok vocabulary dictionaries (with and without velocity)"
    )
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for vocabularies (defaults to musetok/data/)")
    
    args = parser.parse_args()
    
    print("Building MuseTok vocabularies...")
    vocab_without, vocab_with = rebuild_both_vocabularies()
    save_vocabularies(vocab_without, vocab_with, args.output_dir)
    print("\nDone! The system will automatically detect which vocabulary matches your checkpoint.")
