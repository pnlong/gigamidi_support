"""
Preprocess XMIDI dataset: extract latents/embeddings from MIDI files.

Supports two preprocessors:
- musetok: MuseTok encoder (per-bar latents, 128d)
- midi2vec: Graph embeddings (per-song, 100d)

Usage:
  python pretrain_model/preprocess_xmidi.py --preprocessor musetok --xmidi_dir ... --output_dir ...
  python pretrain_model/preprocess_xmidi.py --preprocessor midi2vec --xmidi_dir ... --output_dir ...
"""

import argparse
import sys
import os
from pathlib import Path

# Add emotion_genre to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from utils.data_utils import XMIDI_LATENTS_DIR, MUSETOK_CHECKPOINT_DIR, MIDI2VEC_BATCHES_DIR


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess XMIDI dataset: extract latents or embeddings"
    )
    parser.add_argument("--preprocessor", choices=["musetok", "midi2vec"], default="musetok",
                        help="Preprocessor: musetok (default) or midi2vec")
    parser.add_argument("--xmidi_dir", required=True,
                        help="Directory containing XMIDI MIDI files")
    parser.add_argument("--output_dir", default=XMIDI_LATENTS_DIR,
                        help="Output directory for latents/embeddings")
    
    # MuseTok-specific
    parser.add_argument("--checkpoint_path", default=None,
                        help="Path to MuseTok checkpoint (musetok only)")
    parser.add_argument("--vocab_path", default=None,
                        help="Path to MuseTok vocabulary (musetok only)")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU (musetok only)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for GPU (musetok only)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="Number of workers for CPU multiprocessing (musetok only)")
    
    # midi2vec-specific
    parser.add_argument("--precomputed", default=None,
                        help="Path to dir with embeddings.bin and names.csv (midi2vec only)")
    parser.add_argument("--dimensions", type=int, default=100,
                        help="Embedding dimension (midi2vec only)")
    parser.add_argument("--workers", type=int, default=1,
                        help="Parallel workers for midi2edgelist and edgelist2vec (1 = single core, slow; 0 = use all CPUs, midi2vec only)")
    parser.add_argument("--midi2vec_num_batches", type=int, default=20,
                        help="If set, use batched midi2vec with this many batches (stratified by emotion+genre); output in MIDI2VEC_BATCHES_DIR (midi2vec only)")
    
    # Common
    parser.add_argument("--reset", action="store_true",
                        help="Reset: recompute everything (default: resume, skip existing output)")
    parser.add_argument("--no_show_progress", action="store_true", dest="no_show_progress",
                        help="Suppress progress output from midi2edgelist/edgelist2vec (midi2vec only)")
    
    args = parser.parse_args()
    
    if args.preprocessor == "musetok":
        from pretrain_model.preprocess.preprocess_xmidi_musetok import preprocess_xmidi_musetok
        preprocess_xmidi_musetok(
            xmidi_dir=args.xmidi_dir,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint_path,
            vocab_path=args.vocab_path,
            use_gpu=args.gpu,
            resume=not args.reset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        if args.midi2vec_num_batches is not None:
            from pretrain_model.preprocess.preprocess_xmidi_midi2vec_batched import preprocess_xmidi_midi2vec_batched
            preprocess_xmidi_midi2vec_batched(
                xmidi_dir=args.xmidi_dir,
                output_dir=args.output_dir,
                batch_output_root=MIDI2VEC_BATCHES_DIR,
                num_batches=args.midi2vec_num_batches,
                dimensions=args.dimensions,
                reset=args.reset,
                show_progress=not args.no_show_progress,
                edgelist2vec_workers=args.workers,
            )
        else:
            from pretrain_model.preprocess.preprocess_xmidi_midi2vec import preprocess_xmidi_midi2vec
            preprocess_xmidi_midi2vec(
                xmidi_dir=args.xmidi_dir,
                output_dir=args.output_dir,
                precomputed_dir=args.precomputed,
                dimensions=args.dimensions,
                resume=not args.reset,
                show_progress=not args.no_show_progress,
                workers=args.workers,
            )


if __name__ == "__main__":
    main()
