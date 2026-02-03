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

from utils.data_utils import XMIDI_LATENTS_DIR, MUSETOK_CHECKPOINT_DIR


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
    
    # Common
    parser.add_argument("--resume", action="store_true",
                        help="Resume: skip already-processed files")
    
    args = parser.parse_args()
    
    if args.preprocessor == "musetok":
        from pretrain_model.preprocess.preprocess_xmidi_musetok import preprocess_xmidi_musetok
        preprocess_xmidi_musetok(
            xmidi_dir=args.xmidi_dir,
            output_dir=args.output_dir,
            checkpoint_path=args.checkpoint_path,
            vocab_path=args.vocab_path,
            use_gpu=args.gpu,
            resume=args.resume,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    else:
        from pretrain_model.preprocess.preprocess_xmidi_midi2vec import preprocess_xmidi_midi2vec
        preprocess_xmidi_midi2vec(
            xmidi_dir=args.xmidi_dir,
            output_dir=args.output_dir,
            precomputed_dir=args.precomputed,
            dimensions=args.dimensions,
            resume=args.resume,
        )


if __name__ == "__main__":
    main()
