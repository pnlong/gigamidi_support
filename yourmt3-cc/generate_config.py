#!/usr/bin/env python3
"""
Generate YourMT3 inference JSON config from a VA dataset adapter.

Usage (from repo root or yourmt3-cc):
    python yourmt3-cc/generate_config.py --dataset deam --output deam_amt.json
    python yourmt3-cc/generate_config.py --dataset merp --output merp_amt.json --storage_dir /path
"""

import argparse
import json
import sys
from pathlib import Path

# Allow importing va_cont.datasets from repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "va_cont"))

from datasets import get_dataset  # noqa: E402

DEFAULT_EXP_ID = (
    "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2@model.ckpt"
)


def parse_args():
    p = argparse.ArgumentParser(description="Build YourMT3 batch config from dataset adapter.")
    p.add_argument("--dataset", required=True, choices=["deam", "memo2496", "merp"])
    p.add_argument("--output", required=True, help="Output JSON path")
    p.add_argument("--storage_dir", default=None)
    p.add_argument("--exp_id", default=DEFAULT_EXP_ID)
    p.add_argument("--midi_output", default=None,
                   help="MIDI output dir (default: dataset.midi_dir())")
    return p.parse_args()


def main():
    args = parse_args()
    ds = get_dataset(args.dataset, args.storage_dir)
    out_dir = args.midi_output or str(ds.midi_dir())

    data = []
    missing = 0
    for sid in ds.list_song_ids():
        ap = ds.audio_path(sid)
        if ap.is_file():
            data.append(str(ap.resolve()))
        else:
            missing += 1

    cfg = {
        "exp_id": args.exp_id,
        "output": out_dir,
        "flat_output": True,
        "data": data,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(cfg, f, indent=2)

    print(f"Wrote {out_path} — {len(data)} audio files ({missing} missing)")


if __name__ == "__main__":
    main()
