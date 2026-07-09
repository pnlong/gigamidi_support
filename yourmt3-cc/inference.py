#!/usr/bin/env python3
import json
import argparse
import torch
import sys
import os
from tqdm import tqdm
from pathlib import Path

# Add YourMT3 source
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'amt/src')))
from model_helper import load_model_checkpoint, transcribe


def shard_data(data, gpu_index, num_gpus):
    """Return (global_index, path) pairs assigned to this worker."""
    if num_gpus <= 1:
        return list(enumerate(data))
    return [(idx, path) for idx, path in enumerate(data) if (idx % num_gpus) == gpu_index]


def resolve_device(device: str, gpu_index: int, num_gpus: int) -> str:
    """Map plain 'cuda' to a concrete device when sharding across GPUs.

    gpu_index selects which *files* this worker handles. The CUDA device is:
    - cuda:0 when CUDA_VISIBLE_DEVICES pins a single GPU (always use logical 0)
    - cuda:{gpu_index} when all GPUs are visible in one process namespace
    """
    if device != "cuda" or num_gpus <= 1:
        return device
    if torch.cuda.device_count() == 1:
        return "cuda:0"
    if gpu_index >= torch.cuda.device_count():
        raise ValueError(
            f"--gpu_index {gpu_index} invalid: only {torch.cuda.device_count()} CUDA device(s) visible. "
            "Use CUDA_VISIBLE_DEVICES=<id> with --gpu_index <shard> (device will be cuda:0), "
            "or pass --device cuda:N explicitly."
        )
    return f"cuda:{gpu_index}"


def main():
    parser = argparse.ArgumentParser(description="Transcribe audio to MIDI using YourMT3")

    parser.add_argument("--config_json", required=True,
                        help="Path to a JSON config describing experiment + audio paths")

    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to use ('cpu' or 'cuda')")

    parser.add_argument("--project", type=str, default="2024")

    parser.add_argument("--output", type=str, default=None,
                        help="Output directory (overrides config file)")

    # --- GPU sharding flags ---
    parser.add_argument("--gpu_index", type=int, default=0,
                        help="This worker's GPU index (0..num_gpus-1)")
    parser.add_argument("--num_gpus", type=int, default=1,
                        help="Total number of GPUs used for distributed transcription")

    args = parser.parse_args()

    if args.num_gpus <= 1 and args.gpu_index != 0:
        print(
            f"WARNING: --gpu_index {args.gpu_index} is ignored because --num_gpus={args.num_gpus}. "
            "Pass --num_gpus 3 (and run one process per GPU) to shard work."
        )

    args.device = resolve_device(args.device, args.gpu_index, args.num_gpus)

    # Load config JSON
    with open(args.config_json, "r") as f:
        cfg = json.load(f)

    exp_id = cfg.get("exp_id")
    if not exp_id:
        raise ValueError("JSON must include 'exp_id'.")

    data = cfg.get("data")
    if not data:
        raise ValueError("JSON must include 'data' (list of input audio files).")

    flat_output = bool(cfg.get("flat_output", False))

    if not args.output:
        args.output = cfg.get("output")
        if not args.output:
            raise ValueError("Output directory must be provided (arg or JSON).")

    args.output = Path(args.output)
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"\n==== YourMT3 Transcription ====")
    print(f"Experiment: {exp_id}")
    print(f"Device: {args.device}")
    print(f"GPU shard: index={args.gpu_index} / total={args.num_gpus}")
    print("================================\n")

    # ------------------------
    # Load model once
    # ------------------------
    print("Loading model...")

    model_args = [exp_id, '-p', '2024', '-tk', 'mc13_full_plus_256', '-dec', 'multi-t5',
            '-nl', '26', '-enc', 'perceiver-tf', '-sqr', '1', '-ff', 'moe',
            '-wf', '4', '-nmoe', '8', '-kmoe', '2', '-act', 'silu', '-epe', 'rope',
            '-rp', '1', '-ac', 'spec', '-hop', '300', '-atc', '1', '-pr', '16']

    model = load_model_checkpoint(model_args, device=args.device)
    print("Model loaded.\n")

    shard = shard_data(data, args.gpu_index, args.num_gpus)
    if not shard:
        raise ValueError(
            f"No files assigned to gpu_index={args.gpu_index} with num_gpus={args.num_gpus}."
        )
    print(f"Processing {len(shard)} / {len(data)} files on {args.device}.\n")

    # ------------------------
    # Process audio files (sharded)
    # ------------------------
    processed = 0
    skipped = 0
    for idx, audio_path in tqdm(shard, desc="Transcribing"):
        audio_path = Path(audio_path)
        if flat_output:
            # VA pipeline: write {song_id}.mid directly into output dir
            output_folder = args.output
        else:
            track_id = audio_path.parts[-3]  # TrackXXXX folder (original layout)
            output_folder = args.output / track_id
            output_folder.mkdir(parents=True, exist_ok=True)

        audio_info = {
            "filepath": str(audio_path),
            "output": str(output_folder),
            "file_name": audio_path.stem,
            "flat_output": flat_output,
        }

        try:
            transcribe(model, audio_info)
            processed += 1
        except Exception as exc:
            skipped += 1
            tqdm.write(f"SKIP [{idx}] {audio_path.name}: {exc}")

    print(f"\nTranscription finished ({processed} ok, {skipped} skipped).")


if __name__ == "__main__":
    main()
