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
import logging

# Add musetok to path (adjust based on your setup)
MUSETOK_PATH = os.path.join(os.path.dirname(__file__), "../../musetok")
MUSETOK_PATH = os.path.abspath(MUSETOK_PATH)

# Import musetok modules using importlib to avoid path conflicts
# We need to import in a way that preserves musetok's internal import structure
import importlib.util

# Save original sys.path and modules
_original_sys_path = sys.path.copy()
_original_utils = sys.modules.get("utils", None)

# Import musetok/utils.py first (needed by encoding.py)
utils_spec = importlib.util.spec_from_file_location("musetok_utils_module", 
                                                     os.path.join(MUSETOK_PATH, "utils.py"))
musetok_utils_module = importlib.util.module_from_spec(utils_spec)
sys.modules["musetok_utils_module"] = musetok_utils_module
utils_spec.loader.exec_module(musetok_utils_module)
pickle_load = musetok_utils_module.pickle_load
tensor_to_numpy = musetok_utils_module.tensor_to_numpy

# Wrap numpy_to_tensor to ensure it uses the correct device
# The original function has device='cuda' as default, but we want to use the device
# passed to MuseTokEncoder. We'll create a wrapper that respects the device parameter.
_original_numpy_to_tensor = musetok_utils_module.numpy_to_tensor

def _wrapped_numpy_to_tensor(arr, use_gpu=None, device=None):
    """Wrapper for numpy_to_tensor that ensures device is used correctly.
    
    This wrapper handles both use_gpu and device parameters:
    - If device is provided, infer use_gpu from it
    - Pass correct parameters to original function
    """
    # If device is provided, infer use_gpu from it
    if device is not None:
        if isinstance(device, str):
            use_gpu = device.startswith('cuda')
        else:
            # torch.device object
            use_gpu = device.type == 'cuda'
            device = str(device)
    elif use_gpu is None:
        # Default to True if neither is provided
        use_gpu = True
        device = 'cuda'
    
    # If device wasn't set, use default based on use_gpu
    if device is None:
        device = 'cuda' if use_gpu else 'cpu'
    
    return _original_numpy_to_tensor(arr, use_gpu=use_gpu, device=device)

# Replace numpy_to_tensor in the module with our wrapper
musetok_utils_module.numpy_to_tensor = _wrapped_numpy_to_tensor

# Temporarily replace 'utils' in sys.modules so musetok imports use musetok/utils.py
sys.modules["utils"] = musetok_utils_module

# Add musetok to sys.path for imports
if MUSETOK_PATH not in sys.path:
    sys.path.insert(0, MUSETOK_PATH)
# Add model directory to path (needed for musetok imports)
model_path = os.path.join(MUSETOK_PATH, "model")
if model_path not in sys.path:
    sys.path.insert(0, model_path)

try:
    # Import model/transformer_helpers.py first (needed by model/musetok.py)
    transformer_helpers_spec = importlib.util.spec_from_file_location(
        "model.transformer_helpers",
        os.path.join(MUSETOK_PATH, "model", "transformer_helpers.py")
    )
    transformer_helpers_module = importlib.util.module_from_spec(transformer_helpers_spec)
    # Create model package structure
    if "model" not in sys.modules:
        import types
        sys.modules["model"] = types.ModuleType("model")
    sys.modules["model"].transformer_helpers = transformer_helpers_module
    sys.modules["model.transformer_helpers"] = transformer_helpers_module
    transformer_helpers_spec.loader.exec_module(transformer_helpers_module)
    
    # Now import model.musetok (it can now resolve .transformer_helpers)
    musetok_spec = importlib.util.spec_from_file_location(
        "model.musetok",
        os.path.join(MUSETOK_PATH, "model", "musetok.py")
    )
    musetok_model_module = importlib.util.module_from_spec(musetok_spec)
    sys.modules["model"].musetok = musetok_model_module
    sys.modules["model.musetok"] = musetok_model_module
    musetok_spec.loader.exec_module(musetok_model_module)
    TransformerResidualVQ = musetok_model_module.TransformerResidualVQ
    
    # Now import encoding (it can now resolve utils and model.musetok)
    encoding_spec = importlib.util.spec_from_file_location(
        "encoding",
        os.path.join(MUSETOK_PATH, "encoding.py")
    )
    encoding_module = importlib.util.module_from_spec(encoding_spec)
    sys.modules["encoding"] = encoding_module
    encoding_spec.loader.exec_module(encoding_module)
    MuseTokEncoder = encoding_module.MuseTokEncoder
    convert_event = encoding_module.convert_event
    
finally:
    # Restore original sys.path and utils module
    sys.path[:] = _original_sys_path
    if _original_utils is not None:
        sys.modules["utils"] = _original_utils
    elif "utils" in sys.modules and sys.modules["utils"] is musetok_utils_module:
        del sys.modules["utils"]

# Import our MIDI utilities (relative import)
from .midi_utils import midi_to_events_symusic, load_midi_symusic
from .data_utils import MUSETOK_CHECKPOINT_DIR, ensure_dir


def load_musetok_model(checkpoint_path: Optional[str] = None, 
                      vocab_path: Optional[str] = None,
                      use_gpu: bool = True,
                      prefer_velocity: bool = False,
                      model_enc_seqlen: int = 128,
                      model_max_bars: int = 16) -> Tuple[MuseTokEncoder, dict, bool]:
    """
    Load pre-trained MuseTok model with automatic vocabulary detection.
    
    This function:
    1. Tries loading with the preferred vocabulary (based on prefer_velocity)
    2. If checkpoint vocabulary size doesn't match, tries the other vocabulary
    3. Issues a warning if preference couldn't be satisfied
    4. Returns whether velocity is being used (for filtering events if needed)
    
    Args:
        checkpoint_path: Path to model checkpoint (defaults to MUSETOK_CHECKPOINT_DIR/best_tokenizer.pt)
        vocab_path: Path to vocabulary file (if None, auto-detects based on checkpoint)
        use_gpu: If True, use CUDA; otherwise use CPU
        prefer_velocity: If True, prefer velocity vocabulary; if False, prefer no velocity (default)
        model_enc_seqlen: Maximum sequence length per bar (default: 128)
        model_max_bars: Maximum bars per segment (default: 16)
    
    Note: The default checkpoint is best_tokenizer.pt, which is used for encoding/extracting latents.
    
    Returns:
        encoder: Loaded MuseTokEncoder instance
        vocab: Vocabulary dictionary (event2idx mapping)
        use_velocity: Whether the vocabulary includes velocity (True) or not (False)
    
    Note: The model architecture parameters are hardcoded based on the pre-trained checkpoint.
    If using a different checkpoint, adjust these parameters accordingly.
    """
    # Convert use_gpu to device string
    if use_gpu and torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    
    # Set default paths
    if checkpoint_path is None:
        # Default to best_tokenizer checkpoint (used for encoding/extracting latents)
        checkpoint_path = os.path.join(MUSETOK_CHECKPOINT_DIR, "best_tokenizer.pt")
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"MuseTok checkpoint not found: {checkpoint_path}")
    
    # Load checkpoint to check vocabulary size
    checkpoint_state = torch.load(checkpoint_path, map_location='cpu')
    if isinstance(checkpoint_state, dict) and 'model_state_dict' in checkpoint_state:
        model_state = checkpoint_state['model_state_dict']
    else:
        model_state = checkpoint_state
    
    if 'token_emb.emb_lookup.weight' not in model_state:
        raise ValueError(f"Could not find token_emb.emb_lookup.weight in checkpoint: {checkpoint_path}")
    
    checkpoint_vocab_size = model_state['token_emb.emb_lookup.weight'].shape[0]
    
    # Try to auto-detect vocabulary
    # Build list of vocabularies to try, prioritizing user preference
    vocab_paths_to_try = []
    if vocab_path is not None:
        # User specified vocab path, try it first
        vocab_paths_to_try.append((vocab_path, None, True))  # None means we don't know if it has velocity, True = user-specified
    else:
        # Auto-detect: try preferred vocabulary first, then fallback
        vocab_without_velocity = os.path.join(MUSETOK_PATH, "data", "dictionary.pkl")
        vocab_with_velocity = os.path.join(MUSETOK_PATH, "data", "dictionary_with_velocity.pkl")
        
        if prefer_velocity:
            # Prefer velocity: try with velocity first, then without
            if os.path.exists(vocab_with_velocity):
                vocab_paths_to_try.append((vocab_with_velocity, True, True))  # (path, has_velocity, is_preferred)
            if os.path.exists(vocab_without_velocity):
                vocab_paths_to_try.append((vocab_without_velocity, False, False))  # fallback
        else:
            # Prefer no velocity: try without velocity first, then with
            if os.path.exists(vocab_without_velocity):
                vocab_paths_to_try.append((vocab_without_velocity, False, True))  # (path, has_velocity, is_preferred)
            if os.path.exists(vocab_with_velocity):
                vocab_paths_to_try.append((vocab_with_velocity, True, False))  # fallback
        
        # If neither exists, try default
        if not vocab_paths_to_try:
            vocab_paths_to_try.append((vocab_without_velocity, None, None))
    
    # Try each vocabulary until we find a match
    vocab = None
    use_velocity = None
    n_token = None
    used_preferred = False
    preferred_available = False
    
    for vocab_path_attempt, has_velocity, is_preferred in vocab_paths_to_try:
        if not os.path.exists(vocab_path_attempt):
            continue
        
        if is_preferred:
            preferred_available = True
        
        # Load vocabulary
        vocab_data = pickle_load(vocab_path_attempt)
        vocab_attempt = vocab_data[0]  # event2idx dictionary
        idx2event = vocab_data[1]  # idx2event list
        
        # Calculate vocabulary size
        n_token_attempt = len(vocab_attempt) + 1  # Vocabulary size (includes pad token)
        
        # Check if this vocabulary matches the checkpoint
        if n_token_attempt == checkpoint_vocab_size:
            vocab = vocab_attempt
            n_token = n_token_attempt
            used_preferred = is_preferred if is_preferred is not None else False
            
            # Determine if velocity is used
            if has_velocity is not None:
                use_velocity = has_velocity
            else:
                # Check by looking for velocity tokens in vocabulary
                velocity_tokens = [k for k in vocab.keys() if 'Velocity' in k]
                use_velocity = len(velocity_tokens) > 0
            
            logging.info(f"Using vocabulary: {os.path.basename(vocab_path_attempt)} ({n_token} tokens, velocity={'yes' if use_velocity else 'no'})")
            
            # Warn if preference couldn't be satisfied
            if prefer_velocity and not use_velocity and preferred_available:
                logging.warning(
                    f"Requested velocity but checkpoint doesn't support it. Using non-velocity vocabulary. "
                    f"Velocity events will be filtered."
                )
            elif not prefer_velocity and use_velocity and preferred_available:
                logging.warning(
                    f"Requested no velocity but checkpoint requires it. Using velocity vocabulary."
                )
            
            break
    
    if vocab is None:
        # No matching vocabulary found - provide helpful error message
        vocab_sizes_found = []
        for vocab_path_attempt, _, _ in vocab_paths_to_try:
            if os.path.exists(vocab_path_attempt):
                try:
                    vocab_data_check = pickle_load(vocab_path_attempt)
                    vocab_size_check = len(vocab_data_check[0]) + 1
                    vocab_sizes_found.append(f"    - {vocab_path_attempt}: {vocab_size_check} tokens")
                except:
                    vocab_sizes_found.append(f"    - {vocab_path_attempt}: (could not read)")
        
        error_msg = (
            f"Could not find matching vocabulary for checkpoint!\n"
            f"  Checkpoint expects: {checkpoint_vocab_size} tokens (167 vocab + 1 pad)\n"
            f"  Checkpoint path: {checkpoint_path}\n\n"
            f"Available vocabularies:\n" + 
            "\n".join(vocab_sizes_found if vocab_sizes_found else 
                     [f"    - {path} (exists: {os.path.exists(path)})" for path, _, _ in vocab_paths_to_try]) +
            f"\n\nThis checkpoint was trained with a vocabulary of {checkpoint_vocab_size - 1} tokens.\n"
            f"The original vocabulary file may have been overwritten or is missing.\n\n"
            f"Solutions:\n"
            f"  1. Find the original vocabulary file that was used to train this checkpoint\n"
            f"  2. If you have the original training data, rebuild the vocabulary using:\n"
            f"     musetok/data_processing/events2words.py events2dictionary()\n"
            f"  3. Use a different checkpoint that matches your available vocabularies\n"
            f"  4. Retrain the MuseTok model with one of the available vocabularies"
        )
        raise ValueError(error_msg)
    
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
    num_quantizers = 16
    codebook_size = 2048
    
    # Initialize model
    logging.info(f"Initializing MuseTok model on device: {device}")
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
    logging.info(f"Loading checkpoint from {checkpoint_path}")
    model.load_state_dict(model_state)
    model.eval()
    
    # Verify model is on correct device
    first_param = next(model.parameters())
    actual_device = str(first_param.device)
    logging.info(f"Model parameter device: {actual_device} (expected: {device})")
    
    if use_gpu and torch.cuda.is_available():
        # Check that model parameters are on GPU
        if not first_param.is_cuda:
            logging.warning(f"Model was moved to {device} but parameters are on {first_param.device}. Moving to {device}...")
            model = model.to(device)
            # Verify again
            first_param = next(model.parameters())
            logging.info(f"Model parameter device after correction: {first_param.device}")
    
    # Find the actual vocab path that was used
    actual_vocab_path = None
    for vocab_path_attempt, _, _ in vocab_paths_to_try:
        if os.path.exists(vocab_path_attempt):
            # Check if this is the one we used
            vocab_data_check = pickle_load(vocab_path_attempt)
            if vocab_data_check[0] == vocab:
                actual_vocab_path = vocab_path_attempt
                break
    
    if actual_vocab_path is None:
        # Fallback to the vocab we found
        for vocab_path_attempt, _, _ in vocab_paths_to_try:
            if os.path.exists(vocab_path_attempt):
                actual_vocab_path = vocab_path_attempt
                break
    
    # Create encoder wrapper
    encoder = MuseTokEncoder(
        model=model,
        device=device,
        vocab_file=actual_vocab_path if actual_vocab_path else vocab_path,
        model_enc_seqlen=model_enc_seqlen,
        model_max_bars=model_max_bars
    )
    
    return encoder, vocab, use_velocity


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
                                filter_velocity: bool = False) -> np.ndarray:
    """
    Extract MuseTok latents from REMI events.
    
    This function handles songs with >16 bars by processing them in segments.
    The MuseTok model requires exactly 16 bars per input, so longer songs are split
    into multiple segments of 16 bars each, processed independently, then concatenated.
    
    Args:
        events: List of REMI event dictionaries
        bar_positions: List of event indices where bars start
        encoder: MuseTokEncoder instance (device is already set in encoder)
        vocab: Vocabulary dictionary (for token conversion, though encoder has its own)
        filter_velocity: If True, filter out Note_Velocity events before processing
    
    Returns:
        latents: numpy array of shape (n_bars, d_vae_latent) where d_vae_latent=128
    """
    # Filter velocity events if needed
    if filter_velocity:
        events = [e for e in events if not (isinstance(e, dict) and e.get('name', '').startswith('Note_Velocity'))]
        # Recalculate bar positions after filtering
        # Bar positions are indices into the original events list, so we need to adjust them
        # For simplicity, we'll recalculate bar positions from the filtered events
        piece_bar_pos = []
        for i, event in enumerate(events):
            if isinstance(event, dict) and event.get('name') == 'Bar':
                piece_bar_pos.append(i)
    else:
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
        encoder: MuseTokEncoder instance (device is already set in encoder)
        vocab: Vocabulary dictionary
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
        events, bar_positions, encoder, vocab
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
