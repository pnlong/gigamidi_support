"""
MuseTok integration utilities.

This module provides functions to load MuseTok models and extract latents from MIDI files
or REMI events. It handles songs with >16 bars by processing them in segments, as the
MuseTok model architecture requires exactly 16 bars per input.
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import sys
import os
import pickle

# Add musetok to path (adjust based on your setup)
MUSETOK_PATH = os.path.join(os.path.dirname(__file__), "../../musetok")
if MUSETOK_PATH not in sys.path:
    sys.path.insert(0, MUSETOK_PATH)
# Add model directory to path (needed for musetok imports, as encoding.py does sys.path.append('./model'))
if os.path.join(MUSETOK_PATH, "model") not in sys.path:
    sys.path.insert(0, os.path.join(MUSETOK_PATH, "model"))

# Import musetok modules
# Note: We need to import from musetok directory, and encoding.py expects to be run from musetok root
try:
    from encoding import MuseTokEncoder, convert_event
except ImportError:
    # Fallback: try importing with musetok prefix
    import musetok.encoding as encoding_module
    MuseTokEncoder = encoding_module.MuseTokEncoder
    convert_event = encoding_module.convert_event

from model.musetok import TransformerResidualVQ
from utils import pickle_load, numpy_to_tensor, tensor_to_numpy

# Import our MIDI utilities (relative import)
from .midi_utils import midi_to_events_symusic, load_midi_symusic
from .data_utils import MUSETOK_CHECKPOINT_DIR, ensure_dir


def load_musetok_model(checkpoint_path: Optional[str] = None, 
                      vocab_path: Optional[str] = None,
                      device: str = 'cuda',
                      model_enc_seqlen: int = 128,
                      model_max_bars: int = 16) -> Tuple[MuseTokEncoder, dict]:
    """
    Load pre-trained MuseTok model.
    
    Args:
        checkpoint_path: Path to model checkpoint (defaults to MUSETOK_CHECKPOINT_DIR/best_tokenizer.pt)
        vocab_path: Path to vocabulary file (defaults to musetok/data/dictionary.pkl)
        device: Device to load model on ('cuda' or 'cpu')
        model_enc_seqlen: Maximum sequence length per bar (default: 128)
        model_max_bars: Maximum bars per segment (default: 16)
    
    Note: The default checkpoint is best_tokenizer.pt, which is used for encoding/extracting latents.
    
    Returns:
        encoder: Loaded MuseTokEncoder instance
        vocab: Vocabulary dictionary (event2idx mapping)
    
    Note: The model architecture parameters are hardcoded based on the pre-trained checkpoint.
    If using a different checkpoint, adjust these parameters accordingly.
    """
    # Set default paths
    if checkpoint_path is None:
        # Default to best_tokenizer checkpoint (used for encoding/extracting latents)
        checkpoint_path = os.path.join(MUSETOK_CHECKPOINT_DIR, "best_tokenizer.pt")
    
    if vocab_path is None:
        # Default to musetok/data/dictionary.pkl
        vocab_path = os.path.join(MUSETOK_PATH, "data", "dictionary.pkl")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"MuseTok checkpoint not found: {checkpoint_path}")
    if not os.path.exists(vocab_path):
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_path}")
    
    # Load vocabulary
    vocab_data = pickle_load(vocab_path)
    vocab = vocab_data[0]  # event2idx dictionary
    idx2event = vocab_data[1]  # idx2event list
    
    # Model architecture parameters (from MuseTok-tokenization.ipynb)
    # These should match the checkpoint being loaded
    enc_n_layer = 12
    enc_n_head = 8
    enc_d_model = 512
    enc_d_ff = 2048
    dec_n_layer = 12
    dec_n_head = 8
    dec_d_model = 512
    dec_d_ff = 2048
    d_vae_latent = 128  # This is the latent dimension we'll extract
    d_embed = 512
    n_token = len(vocab)  # Vocabulary size
    num_quantizers = 16
    codebook_size = 2048
    
    # Initialize model
    model = TransformerResidualVQ(
        enc_n_layer=enc_n_layer,
        enc_n_head=enc_n_head,
        enc_d_model=enc_d_model,
        enc_d_ff=enc_d_ff,
        dec_n_layer=dec_n_layer,
        dec_n_head=dec_n_head,
        dec_d_model=dec_d_model,
        dec_d_ff=dec_d_ff,
        d_vae_latent=d_vae_latent,
        d_embed=d_embed,
        n_token=n_token,
        num_quantizers=num_quantizers,
        codebook_size=codebook_size,
        rotation_trick=True,
        rvq_type='SimVQ'
    ).to(device)
    
    # Load checkpoint
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    
    # Create encoder wrapper
    encoder = MuseTokEncoder(
        model=model,
        device=device,
        vocab_file=vocab_path,
        model_enc_seqlen=model_enc_seqlen,
        model_max_bars=model_max_bars
    )
    
    return encoder, vocab


def events_to_remi_tokens(events: List[Dict], vocab: dict) -> np.ndarray:
    """
    Convert REMI events to token sequence using vocabulary.
    
    Args:
        events: List of event dictionaries with 'name' and 'value' keys
        vocab: Vocabulary dictionary (event2idx mapping)
    
    Returns:
        tokens: numpy array of token indices
    """
    # Use the convert_event function from musetok.encoding
    tokens = convert_event(events, vocab, to_ndarr=True)
    return tokens


def extract_latents_from_events(events: List[Dict], 
                                bar_positions: List[int],
                                encoder: MuseTokEncoder,
                                vocab: dict,
                                device: str = 'cuda') -> np.ndarray:
    """
    Extract MuseTok latents from REMI events.
    
    This function handles songs with >16 bars by processing them in segments.
    The MuseTok model requires exactly 16 bars per input, so longer songs are split
    into multiple segments of 16 bars each, processed independently, then concatenated.
    
    Args:
        events: List of REMI event dictionaries
        bar_positions: List of event indices where bars start
        encoder: MuseTokEncoder instance
        vocab: Vocabulary dictionary (for token conversion, though encoder has its own)
        device: Device (not used directly, encoder handles this)
    
    Returns:
        latents: numpy array of shape (n_bars, d_vae_latent) where d_vae_latent=128
    """
    # Convert events to token sequence
    # Note: encoder.get_segments() will handle token conversion internally,
    # but we need events as a list for get_segments()
    
    # Prepare bar_positions (make a copy to avoid modifying original)
    piece_bar_pos = bar_positions.copy()
    
    # Get segments (handles >16 bars automatically)
    # This splits the song into segments of 16 bars each
    music_data = encoder.get_segments(events, piece_bar_pos)
    
    # Extract latents using encoder
    # This processes all segments and returns latents for all bars
    indices, latents = encoder.encoding(music_data, return_latents=True)
    
    # latents shape: (n_bars, d_vae_latent) where d_vae_latent=128
    return latents


def extract_latents_from_midi(midi_path_or_bytes: Union[str, bytes],
                              encoder: MuseTokEncoder,
                              vocab: dict,
                              device: str = 'cuda',
                              has_velocity: bool = False,
                              time_first: bool = False,
                              repeat_beat: bool = True) -> Tuple[np.ndarray, List[int]]:
    """
    Full pipeline: MIDI -> REMI events -> latents.
    
    This function:
    1. Loads MIDI file using symusic
    2. Converts to REMI events using midi_to_events_symusic
    3. Extracts latents using MuseTok encoder
    4. Returns bar-level latents and bar positions
    
    Args:
        midi_path_or_bytes: Path to MIDI file or MIDI bytes
        encoder: MuseTokEncoder instance
        vocab: Vocabulary dictionary
        device: Device string
        has_velocity: Whether to include velocity in events
        time_first: Whether to put time signature at start of bar
        repeat_beat: Whether to repeat beat event for each note
    
    Returns:
        latents: numpy array of shape (n_bars, d_vae_latent) where d_vae_latent=128
        bar_positions: List of event indices where bars start (in the original event sequence)
    """
    # Load MIDI with symusic
    score = load_midi_symusic(midi_path_or_bytes)
    
    # Convert to REMI events
    bar_positions, events = midi_to_events_symusic(
        score,
        has_velocity=has_velocity,
        time_first=time_first,
        repeat_beat=repeat_beat
    )
    
    # Extract latents
    latents = extract_latents_from_events(
        events, bar_positions, encoder, vocab, device
    )
    
    return latents, bar_positions


def get_bar_level_latents(latents: np.ndarray, bar_positions: List[int]) -> np.ndarray:
    """
    Get bar-level latents (already bar-level, but included for API consistency).
    
    Args:
        latents: numpy array of shape (n_bars, d_vae_latent)
        bar_positions: List of bar positions (not used, but kept for API consistency)
    
    Returns:
        latents: Same as input (already bar-level)
    """
    # Latents from extract_latents_from_events are already bar-level
    # This function is included for API consistency
    return latents
