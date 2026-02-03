"""
Utilities for midi2vec pipeline: run midi2edgelist, edgelist2vec, load embeddings.

midi2vec is transductive: embeddings exist only for MIDI files that were in the
graph when node2vec was run. There is no pretrained model for new files.
"""

import os
import subprocess
import csv
import logging
from pathlib import Path
from typing import Dict, Optional, Callable
import numpy as np

# Resolve path to midi2vec (sibling of emotion_genre in gigamidi)
_EMOTION_GENRE_DIR = Path(__file__).resolve().parent.parent
_GIGAMIDI_ROOT = _EMOTION_GENRE_DIR.parent
MIDI2VEC_ROOT = _GIGAMIDI_ROOT / "midi2vec"


def run_midi2edgelist(midi_dir: str, output_dir: str) -> bool:
    """
    Run midi2edgelist (Node.js) to convert MIDI files to graph edgelists.
    
    Args:
        midi_dir: Directory containing MIDI files
        output_dir: Output directory for edgelists and names.csv
        
    Returns:
        True if successful, False otherwise
    """
    index_js = MIDI2VEC_ROOT / "midi2edgelist" / "index.js"
    if not index_js.exists():
        logging.error(f"midi2edgelist not found at {index_js}. Run: cd midi2vec/midi2edgelist && npm install")
        return False
    
    cmd = [
        "node",
        str(index_js),
        "-i", os.path.abspath(midi_dir),
        "-o", os.path.abspath(output_dir),
    ]
    logging.info(f"Running midi2edgelist: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(MIDI2VEC_ROOT / "midi2edgelist"))
        if result.returncode != 0:
            logging.error(f"midi2edgelist failed: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        logging.error("Node.js not found. Install Node.js to run midi2edgelist.")
        return False


def run_edgelist2vec(
    edgelist_dir: str,
    output_bin: str,
    dimensions: int = 100,
) -> bool:
    """
    Run edgelist2vec (Python) to compute node2vec embeddings.
    
    Args:
        edgelist_dir: Directory containing .edgelist files and names.csv
        output_bin: Path to output embeddings.bin (gensim KeyedVectors)
        dimensions: Embedding dimension (default 100)
        
    Returns:
        True if successful, False otherwise
    """
    embed_py = MIDI2VEC_ROOT / "edgelist2vec" / "embed.py"
    if not embed_py.exists():
        logging.error(f"edgelist2vec not found at {embed_py}")
        return False
    
    cmd = [
        "python",
        str(embed_py),
        "-i", os.path.abspath(edgelist_dir),
        "-o", os.path.abspath(output_bin),
        "--dimensions", str(dimensions),
    ]
    logging.info(f"Running edgelist2vec: {' '.join(cmd)}")
    try:
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"edgelist2vec failed: {result.stderr}")
            return False
        return True
    except FileNotFoundError:
        logging.error("Python not found.")
        return False


def load_embeddings_lookup(
    embeddings_bin: str,
    names_csv: str,
    id_to_key_fn: Optional[Callable[[str], str]] = None,
) -> Dict[str, np.ndarray]:
    """
    Load embeddings from gensim KeyedVectors and names.csv into a lookup dict.
    
    Args:
        embeddings_bin: Path to embeddings.bin (gensim KeyedVectors)
        names_csv: Path to names.csv (id,filename)
        id_to_key_fn: Optional function to map id -> lookup key. If None, uses id as key.
                      E.g. for GigaMIDI: id is md5, so no mapping needed.
                      For XMIDI: id might need path->stem conversion.
        
    Returns:
        dict mapping key -> embedding (np.ndarray of shape (dimensions,))
    """
    try:
        from gensim.models import KeyedVectors
    except ImportError:
        raise ImportError("gensim required for load_embeddings_lookup. pip install gensim")
    
    # nodevectors saves in word2vec binary format
    try:
        kv = KeyedVectors.load_word2vec_format(str(embeddings_bin), binary=True)
    except Exception:
        kv = KeyedVectors.load(str(embeddings_bin), mmap='r')
    lookup = {}
    
    with open(names_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            midi_id = row.get('id', '').strip()
            if not midi_id:
                continue
            if midi_id not in kv:
                logging.debug(f"ID {midi_id} not in embeddings (may be note/program node, not MIDI root)")
                continue
            key = id_to_key_fn(midi_id) if id_to_key_fn else midi_id
            lookup[key] = kv[midi_id].astype(np.float32)
    
    return lookup


def extract_embeddings_to_safetensors(
    embeddings_bin: str,
    names_csv: str,
    output_dir: str,
    id_to_filename_fn: Optional[Callable[[str], str]] = None,
    save_latents_fn=None,
    ensure_dir_fn=None,
) -> int:
    """
    Extract per-file embeddings from KeyedVectors and save as .safetensors.
    
    Only MIDI root nodes (from names.csv) are extracted. Note/program nodes are skipped.
    
    Args:
        embeddings_bin: Path to embeddings.bin
        names_csv: Path to names.csv (id,filename)
        output_dir: Directory for output .safetensors files
        id_to_filename_fn: Optional. Map id -> filename (stem for output). If None, uses id.
        save_latents_fn: Function (path, latents, metadata) to save. Default uses data_utils.save_latents.
        ensure_dir_fn: Function (path) to ensure dir exists. Default uses data_utils.ensure_dir.
        
    Returns:
        Number of files saved
    """
    import sys
    sys.path.insert(0, str(_EMOTION_GENRE_DIR))
    from utils.data_utils import save_latents, ensure_dir
    
    save_fn = save_latents_fn or save_latents
    ensure_fn = ensure_dir_fn or ensure_dir
    
    lookup = load_embeddings_lookup(embeddings_bin, names_csv)
    
    # Build id -> filename mapping from names.csv
    # For XMIDI: id from midi2vec = path with slashes->dashes, no ext. We want stem.
    # names.csv has id,filename. filename is full path. Stem of filename = our output name.
    count = 0
    with open(names_csv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            midi_id = row.get('id', '').strip()
            filename_cell = row.get('filename', '').strip().strip('"')
            if not midi_id or midi_id not in lookup:
                continue
            # Output filename: use id_to_filename_fn if provided, else derive from filename column
            if id_to_filename_fn:
                output_stem = id_to_filename_fn(midi_id)
            else:
                # filename column is full path; use stem
                output_stem = Path(filename_cell).stem if filename_cell else midi_id
            vec = lookup[midi_id]
            # Store as (1, dim) for dataset compatibility (mean pool gives (dim,))
            latents = vec.reshape(1, -1)
            output_path = os.path.join(output_dir, f"{output_stem}.safetensors")
            ensure_fn(os.path.dirname(output_path))
            metadata = {
                "n_bars": 1,
                "file_type": "midi2vec",
                "original_id": midi_id,
            }
            save_fn(output_path, latents, metadata)
            count += 1
    
    return count
