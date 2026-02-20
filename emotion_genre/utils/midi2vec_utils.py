"""
Utilities for midi2vec pipeline: run midi2edgelist, edgelist2vec, load embeddings.

midi2vec is transductive: embeddings exist only for MIDI files that were in the
graph when node2vec was run. There is no pretrained model for new files.
"""

import os
import subprocess
import csv
import logging
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path
from typing import Dict, List, Optional, Callable
import numpy as np
from tqdm import tqdm

# Resolve path to midi2vec (sibling of emotion_genre in gigamidi)
_EMOTION_GENRE_DIR = Path(__file__).resolve().parent.parent
_GIGAMIDI_ROOT = _EMOTION_GENRE_DIR.parent
MIDI2VEC_ROOT = _GIGAMIDI_ROOT / "midi2vec"


def _list_midi_files(midi_dir: str) -> List[Path]:
    """List MIDI files in directory (matches midi2edgelist's klawSync filter)."""
    files = []
    midi_path = Path(midi_dir).resolve()
    for p in midi_path.rglob("*"):
        if p.is_file() and p.suffix.lower() in (".mid", ".midi"):
            files.append(p)
    return sorted(files)


def _count_midi_files(midi_dir: str) -> int:
    """Count MIDI files in directory (matches midi2edgelist's klawSync filter)."""
    return len(_list_midi_files(midi_dir))


def _run_midi2edgelist_chunk(args: tuple) -> tuple:
    """
    Worker: run midi2edgelist on a chunk. args = (chunk_files, midi_dir, chunk_output_dir).
    Returns (success, chunk_output_dir).
    """
    chunk_files, midi_dir, chunk_output_dir = args
    midi_path = Path(midi_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="midi2edgelist_chunk_") as temp_dir:
        temp_path = Path(temp_dir)
        for f in chunk_files:
            rel = f.relative_to(midi_path)
            dest = temp_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.symlink_to(f.resolve())
            except OSError:
                # Fallback: copy if symlink fails (e.g. Windows without admin)
                shutil.copy2(f, dest)
        cmd = [
            "node",
            str(MIDI2VEC_ROOT / "midi2edgelist" / "index.js"),
            "-i", str(temp_path),
            "-o", chunk_output_dir,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(MIDI2VEC_ROOT / "midi2edgelist"))
        return (result.returncode == 0, chunk_output_dir)


def run_midi2edgelist_for_files(
    file_paths: List[Path],
    midi_dir: str,
    output_dir: str,
    show_progress: bool = False,
) -> bool:
    """
    Run midi2edgelist on an explicit list of MIDI files by creating a temp dir of symlinks.

    Args:
        file_paths: List of Paths to MIDI files (must be under midi_dir or comparable).
        midi_dir: Root directory used to compute relative paths (so names.csv filenames match).
        output_dir: Where to write edgelists and names.csv.
        show_progress: If True, show progress (default False for batched workers).

    Returns:
        True if successful, False otherwise.
    """
    midi_path = Path(midi_dir).resolve()
    with tempfile.TemporaryDirectory(prefix="midi2edgelist_for_files_") as temp_dir:
        temp_path = Path(temp_dir)
        for f in file_paths:
            p = Path(f).resolve()
            try:
                rel = p.relative_to(midi_path)
            except ValueError:
                rel = p.name
            dest = temp_path / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            try:
                dest.symlink_to(p)
            except OSError:
                shutil.copy2(p, dest)
        return run_midi2edgelist(str(temp_path), output_dir, show_progress=show_progress, workers=1)


def _merge_edgelist_outputs(chunk_dirs: List[str], output_dir: str) -> None:
    """Merge edgelist outputs from multiple chunks into final output_dir."""
    edgelist_files = ["notes.edgelist", "program.edgelist", "tempo.edgelist", "time.signature.edgelist"]
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    for ef in edgelist_files:
        out_path = Path(output_dir) / ef
        with open(out_path, "w") as outf:
            for chunk_dir in chunk_dirs:
                chunk_path = Path(chunk_dir) / ef
                if chunk_path.exists():
                    outf.write(chunk_path.read_text())
    # Merge names.csv (header only once)
    names_out = Path(output_dir) / "names.csv"
    with open(names_out, "w") as outf:
        outf.write("id,filename\n")
        for chunk_dir in chunk_dirs:
            names_path = Path(chunk_dir) / "names.csv"
            if names_path.exists():
                lines = names_path.read_text().strip().split("\n")[1:]  # skip header
                for line in lines:
                    if line.strip():
                        outf.write(line + "\n")


def run_midi2edgelist(midi_dir: str, output_dir: str, show_progress: bool = True, workers: int = 1) -> bool:
    """
    Run midi2edgelist (Node.js) to convert MIDI files to graph edgelists.
    
    When workers > 1, splits files into chunks and runs multiple midi2edgelist
    processes in parallel, then merges outputs. No changes to the midi2vec repo needed.
    
    Args:
        midi_dir: Directory containing MIDI files
        output_dir: Output directory for edgelists and names.csv
        show_progress: If True, show a progress bar (one tick per file) instead of raw output
        workers: Number of parallel processes (1 = sequential, 0 = use all CPUs)
        
    Returns:
        True if successful, False otherwise
    """
    index_js = MIDI2VEC_ROOT / "midi2edgelist" / "index.js"
    if not index_js.exists():
        logging.error(f"midi2edgelist not found at {index_js}. Run: cd midi2vec/midi2edgelist && npm install")
        return False
    
    files = _list_midi_files(midi_dir)
    if not files:
        logging.warning(f"No MIDI files found in {midi_dir}")
        return True
    
    if workers == 0:
        workers = cpu_count() or 1
    if workers <= 1:
        # Sequential (original behavior)
        cmd = [
            "node",
            str(index_js),
            "-i", os.path.abspath(midi_dir),
            "-o", os.path.abspath(output_dir),
        ]
        logging.info(f"Running midi2edgelist: {' '.join(cmd)}")
        try:
            if show_progress:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    cwd=str(MIDI2VEC_ROOT / "midi2edgelist"),
                )
                with tqdm(total=len(files), desc="midi2edgelist", unit="file") as pbar:
                    for line in proc.stdout:
                        line = line.rstrip()
                        if line:
                            if ".mid" in line.lower() or ".midi" in line.lower():
                                pbar.update(1)
                            elif "error" in line.lower() or "exception" in line.lower():
                                logging.warning(line)
                proc.wait()
                if proc.returncode != 0:
                    logging.error("midi2edgelist failed (run with --no_show_progress to see full output)")
                    return False
            else:
                result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(MIDI2VEC_ROOT / "midi2edgelist"))
                if result.returncode != 0:
                    err = result.stderr if result.stderr else "(see stderr above)"
                    logging.error(f"midi2edgelist failed: {err}")
                    return False
            return True
        except FileNotFoundError:
            logging.error("Node.js not found. Install Node.js to run midi2edgelist.")
            return False
    
    # Parallel: split into chunks, run in parallel, merge
    n = min(workers, len(files))
    chunk_size = (len(files) + n - 1) // n
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]
    
    base_temp = tempfile.mkdtemp(prefix="midi2edgelist_chunks_")
    try:
        chunk_dirs = []
        for i, chunk in enumerate(chunks):
            chunk_dir = os.path.join(base_temp, f"chunk_{i}")
            os.makedirs(chunk_dir, exist_ok=True)
            chunk_dirs.append(chunk_dir)
        
        worker_args = [(chunk, midi_dir, chunk_dir) for chunk, chunk_dir in zip(chunks, chunk_dirs)]
        logging.info(f"Running midi2edgelist in parallel ({n} workers, {len(files)} files)")
        
        with ProcessPoolExecutor(max_workers=n) as executor:
            futures = {executor.submit(_run_midi2edgelist_chunk, a): a for a in worker_args}
            for future in (tqdm(as_completed(futures), total=len(futures), desc="midi2edgelist", unit="chunk") if show_progress else as_completed(futures)):
                success, _ = future.result()
                if not success:
                    logging.error("midi2edgelist chunk failed")
                    return False
        
        _merge_edgelist_outputs(chunk_dirs, output_dir)
        return True
    finally:
        shutil.rmtree(base_temp, ignore_errors=True)


def run_edgelist2vec(
    edgelist_dir: str,
    output_bin: str,
    dimensions: int = 100,
    show_progress: bool = True,
    workers: int = 1,
) -> bool:
    """
    Run edgelist2vec (Python) to compute node2vec embeddings.
    
    Args:
        edgelist_dir: Directory containing .edgelist files and names.csv
        output_bin: Path to output embeddings.bin (gensim KeyedVectors)
        dimensions: Embedding dimension (default 100)
        show_progress: If True, show progress bar
        workers: Number of parallel workers for node2vec (1 = single core, default; 0 = use all CPUs)
        
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
        "--workers", str(workers),
    ]
    if not show_progress:
        cmd.append("--quiet")
    logging.info(f"Running edgelist2vec: {' '.join(cmd)}")
    try:
        if show_progress:
            # Let embed.py's progress (loading edgelists, Nodes/Edges, Start/End learning, nodevectors verbose) go to terminal.
            result = subprocess.run(cmd)
            if result.returncode != 0:
                logging.error(
                    "edgelist2vec failed (exit code %d; run with --no_show_progress to capture stderr)",
                    result.returncode,
                )
                return False
            return True

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            err = result.stderr if result.stderr else "(no stderr captured)"
            if "FutureWarning" in err or "DeprecationWarning" in err:
                logging.warning(
                    "edgelist2vec exited with code %d (likely due to warnings): %s",
                    result.returncode, err.strip() or err,
                )
            else:
                logging.error(f"edgelist2vec failed: {err}")
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
    
    # embed.py saves in word2vec binary format
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


def consolidate_batched_embeddings_to_safetensors(
    batch_output_root: str,
    latents_output_dir: str,
    save_latents_fn=None,
    ensure_dir_fn=None,
) -> int:
    """
    Extract per-file embeddings from batched outputs into a single latents dir.

    Reads batch_assignments.csv (file_path, batch_id). For each file, loads its
    embedding from batch_{batch_id}/embeddings.bin using that batch's names.csv
    to resolve file path to midi id, then writes latents_dir/{stem}.safetensors.

    Args:
        batch_output_root: Directory containing batch_assignments.csv and batch_0/, batch_1/, ...
        latents_output_dir: Directory for output .safetensors files (one per file).
        save_latents_fn: Optional. Default uses data_utils.save_latents.
        ensure_dir_fn: Optional. Default uses data_utils.ensure_dir.

    Returns:
        Number of files written.
    """
    import sys
    sys.path.insert(0, str(_EMOTION_GENRE_DIR))
    from utils.data_utils import save_latents, ensure_dir

    save_fn = save_latents_fn or save_latents
    ensure_fn = ensure_dir_fn or ensure_dir
    batch_root = Path(batch_output_root)
    assignments_path = batch_root / "batch_assignments.csv"
    if not assignments_path.exists():
        logging.error(f"batch_assignments.csv not found at {assignments_path}")
        return 0

    # Read assignments
    rows = []
    with open(assignments_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            fp = row.get("file_path", "").strip().strip('"')
            bid = row.get("batch_id", "").strip()
            if fp and bid != "":
                try:
                    rows.append((str(Path(fp).resolve()), int(bid)))
                except ValueError:
                    continue

    # Per-batch cache: batch_id -> dict file_path_abs -> (midi_id, vec)
    batch_cache = {}

    def get_embedding(file_path_abs: str, batch_id: int):
        if batch_id not in batch_cache:
            bin_path = batch_root / f"batch_{batch_id}" / "embeddings.bin"
            csv_path = batch_root / f"batch_{batch_id}" / "names.csv"
            if not bin_path.exists() or not csv_path.exists():
                batch_cache[batch_id] = {}
                return None
            lookup = load_embeddings_lookup(str(bin_path), str(csv_path))
            path_to_id_vec = {}
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    midi_id = row.get("id", "").strip()
                    fn_cell = row.get("filename", "").strip().strip('"')
                    fn_abs = str(Path(fn_cell).resolve()) if fn_cell else ""
                    if midi_id in lookup:
                        path_to_id_vec[fn_abs] = (midi_id, lookup[midi_id])
            batch_cache[batch_id] = path_to_id_vec
        return batch_cache[batch_id].get(file_path_abs)

    ensure_fn(latents_output_dir)
    count = 0
    for file_path_abs, batch_id in rows:
        res = get_embedding(file_path_abs, batch_id)
        if res is None:
            logging.warning(f"No embedding for {file_path_abs} in batch_{batch_id}")
            continue
        vec, midi_id = res
        stem = Path(file_path_abs).stem
        output_path = os.path.join(latents_output_dir, f"{stem}.safetensors")
        ensure_fn(os.path.dirname(output_path))
        latents = vec.reshape(1, -1)
        metadata = {"n_bars": 1, "file_type": "midi2vec", "original_id": midi_id}
        save_fn(output_path, latents, metadata)
        count += 1
    return count
