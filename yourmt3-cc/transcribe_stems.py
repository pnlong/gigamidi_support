#!/usr/bin/env python3
"""
transcribe_stems.py — YourMT3+ multi-stem single-track transcription entrypoint.

Transcribes a fixed set of audio stems independently, producing one MIDI file
per stem. Each stem can have a known MIDI program number (forced or hinted) or
let the model infer it (unknown mode).

Usage examples
--------------
# Infer programs (unknown mode) — model decides instrument for each stem
python transcribe_stems.py bass.wav guitar.wav piano.wav

# Known programs — force exact MIDI program per stem
python transcribe_stems.py bass.wav guitar.wav piano.wav \\
    --programs 33 27 0 --force_programs

# Mix: known drum, rest inferred
python transcribe_stems.py drums.wav synth.wav \\
    --programs 128 -1 --is_drum True False

# Custom output dir + names
python transcribe_stems.py stems/S00.wav stems/S01.wav \\
    --output_dir out/midi/ --output_names bass guitar

# Use a JSON stem manifest
python transcribe_stems.py --stems_json stems.json --output_dir out/midi/

# Slurm array sharding (4 GPUs)
python transcribe_stems.py stems/*.wav --gpu_index $SLURM_ARRAY_TASK_ID --num_gpus 4
"""

import argparse
import json
import os
import sys
import textwrap
from pathlib import Path

import torch

# ── YourMT3 source on path ────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "amt" / "src"))

from model_helper import load_model_checkpoint, transcribe_single_inst  # noqa: E402

# ── GM program table (for --list_programs) ────────────────────────────────────
GM_PROGRAMS = {
    0: "Acoustic Grand Piano", 1: "Bright Acoustic Piano", 2: "Electric Grand Piano",
    3: "Honky-tonk Piano", 4: "Electric Piano 1", 5: "Electric Piano 2",
    6: "Harpsichord", 7: "Clavinet", 8: "Celesta", 9: "Glockenspiel",
    10: "Music Box", 11: "Vibraphone", 12: "Marimba", 13: "Xylophone",
    14: "Tubular Bells", 15: "Dulcimer", 16: "Drawbar Organ", 17: "Percussive Organ",
    18: "Rock Organ", 19: "Church Organ", 20: "Reed Organ", 21: "Accordion",
    22: "Harmonica", 23: "Tango Accordion", 24: "Acoustic Guitar (nylon)",
    25: "Acoustic Guitar (steel)", 26: "Electric Guitar (jazz)",
    27: "Electric Guitar (clean)", 28: "Electric Guitar (muted)",
    29: "Overdriven Guitar", 30: "Distortion Guitar", 31: "Guitar Harmonics",
    32: "Acoustic Bass", 33: "Electric Bass (finger)", 34: "Electric Bass (pick)",
    35: "Fretless Bass", 36: "Slap Bass 1", 37: "Slap Bass 2",
    38: "Synth Bass 1", 39: "Synth Bass 2", 40: "Violin", 41: "Viola",
    42: "Cello", 43: "Contrabass", 44: "Tremolo Strings", 45: "Pizzicato Strings",
    46: "Orchestral Harp", 47: "Timpani", 48: "String Ensemble 1",
    49: "String Ensemble 2", 50: "Synth Strings 1", 51: "Synth Strings 2",
    52: "Choir Aahs", 53: "Voice Oohs", 54: "Synth Voice", 55: "Orchestra Hit",
    56: "Trumpet", 57: "Trombone", 58: "Tuba", 59: "Muted Trumpet",
    60: "French Horn", 61: "Brass Section", 62: "Synth Brass 1", 63: "Synth Brass 2",
    64: "Soprano Sax", 65: "Alto Sax", 66: "Tenor Sax", 67: "Baritone Sax",
    68: "Oboe", 69: "English Horn", 70: "Bassoon", 71: "Clarinet",
    72: "Piccolo", 73: "Flute", 74: "Recorder", 75: "Pan Flute",
    76: "Blown Bottle", 77: "Shakuhachi", 78: "Whistle", 79: "Ocarina",
    80: "Lead 1 (square)", 81: "Lead 2 (sawtooth)", 82: "Lead 3 (calliope)",
    83: "Lead 4 (chiff)", 84: "Lead 5 (charang)", 85: "Lead 6 (voice)",
    86: "Lead 7 (fifths)", 87: "Lead 8 (bass+lead)", 88: "Pad 1 (new age)",
    89: "Pad 2 (warm)", 90: "Pad 3 (polysynth)", 91: "Pad 4 (choir)",
    92: "Pad 5 (bowed)", 93: "Pad 6 (metallic)", 94: "Pad 7 (halo)",
    95: "Pad 8 (sweep)", 96: "FX 1 (rain)", 97: "FX 2 (soundtrack)",
    98: "FX 3 (crystal)", 99: "FX 4 (atmosphere)", 100: "Singing Voice (melody)",
    101: "Singing Voice (chorus)", 102: "FX 6 (goblin)", 103: "FX 7 (echoes)",
    104: "FX 8 (sci-fi)", 105: "Sitar", 106: "Banjo", 107: "Shamisen",
    108: "Koto", 109: "Kalimba", 110: "Bagpipe", 111: "Fiddle",
    112: "Shanai", 113: "Tinkle Bell", 114: "Agogo", 115: "Steel Drums",
    116: "Woodblock", 117: "Taiko Drum", 118: "Melodic Tom", 119: "Synth Drum",
    120: "Reverse Cymbal", 121: "Guitar Fret Noise", 122: "Breath Noise",
    123: "Seashore", 124: "Bird Tweet", 125: "Telephone Ring",
    126: "Helicopter", 127: "Applause",
    128: "Drums / Percussion (ch.9)",
}

# ── Built-in model presets ─────────────────────────────────────────────────────
# Each preset maps to the model_args list expected by load_model_checkpoint()
# (everything after exp_id).
#
# Preset guide:
#   yptf_single  — YPTF+Single (noPS): single-channel decoder, trained on single-
#                  instrument audio. Default for transcribe_stems.py. Use with:
#                  exp_id = ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100
#
#   mc13_full_plus_256 — YPTF.MoE+Multi: 13 parallel decoding channels, best
#                  overall quality. Use with exp_id = mc13_256_g4_..._b80_ps2
#
MODEL_PRESETS = {
    "yptf_single": [
        "-p", "2024",
        "-enc", "perceiver-tf",
        "-ac", "spec",
        "-hop", "300",
        "-atc", "1",
        "-pr", "16",
    ],
    "mc13_full_plus_256": [
        "-p", "2024",
        "-tk", "mc13_full_plus_256",
        "-dec", "multi-t5",
        "-nl", "26",
        "-enc", "perceiver-tf",
        "-sqr", "1",
        "-ff", "moe",
        "-wf", "4",
        "-nmoe", "8",
        "-kmoe", "2",
        "-act", "silu",
        "-epe", "rope",
        "-rp", "1",
        "-ac", "spec",
        "-hop", "300",
        "-atc", "1",
        "-pr", "16",
    ],
    "mt3_full_plus": [
        "-p", "2024",
        "-tk", "mt3_full_plus",
        "-dec", "t5",
        "-enc", "t5",
        "-pr", "32",
    ],
}
DEFAULT_PRESET = "yptf_single"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _list_programs():
    print("\nGeneral MIDI program numbers (0-127) + special values:\n")
    for prog, name in GM_PROGRAMS.items():
        marker = " [SPECIAL]" if prog >= 100 else ""
        print(f"  {prog:>3}  {name}{marker}")
    print("\nPass -1 to let the model infer the program for a stem.\n")


def _parse_bool(s: str) -> bool:
    return s.strip().lower() in ("1", "true", "yes", "y")


def _resolve_stems_and_programs(args):
    """
    Returns a list of dicts:
      { "audio": Path, "name": str, "program": int|None, "is_drum": bool }
    """
    stems = []

    if args.stems_json:
        with open(args.stems_json) as f:
            manifest = json.load(f)
        # Expected format: list of {audio, name?, program?, is_drum?}
        for entry in manifest:
            audio = Path(entry["audio"])
            name = entry.get("name", audio.stem)
            program = entry.get("program", None)
            is_drum = bool(entry.get("is_drum", False))
            stems.append({"audio": audio, "name": name, "program": program, "is_drum": is_drum})
    else:
        audio_paths = [Path(p) for p in args.stems]
        n = len(audio_paths)

        # Resolve programs: -1 → None (infer)
        programs = []
        if args.programs:
            raw = args.programs
            if len(raw) == 1:
                raw = raw * n  # broadcast single value
            for v in raw:
                programs.append(None if int(v) < 0 else int(v))
        else:
            programs = [None] * n

        # Resolve is_drum
        is_drums = []
        if args.is_drum:
            raw = args.is_drum
            if len(raw) == 1:
                raw = raw * n
            is_drums = [_parse_bool(v) for v in raw]
        else:
            # Auto-infer: program 128 implies drum
            is_drums = [p == 128 for p in programs]

        # Override is_drum for known drum programs
        for i, p in enumerate(programs):
            if p == 128:
                is_drums[i] = True

        # Output names
        out_names = []
        if args.output_names:
            raw = args.output_names
            if len(raw) != n:
                raise ValueError(f"--output_names has {len(raw)} values but {n} stems were given.")
            out_names = raw
        else:
            out_names = [p.stem for p in audio_paths]

        for audio, name, prog, is_drum in zip(audio_paths, out_names, programs, is_drums):
            stems.append({"audio": audio, "name": name, "program": prog, "is_drum": is_drum})

    return stems


def _build_model_args(args, exp_id: str) -> list:
    """Assemble the model_args list for load_model_checkpoint."""
    if args.model_args_json:
        with open(args.model_args_json) as f:
            extra = json.load(f)
        return [exp_id] + extra

    preset_key = args.preset or DEFAULT_PRESET
    if preset_key not in MODEL_PRESETS:
        raise ValueError(
            f"Unknown preset '{preset_key}'. Available: {list(MODEL_PRESETS.keys())}"
        )
    preset = MODEL_PRESETS[preset_key].copy()

    # Allow project override inside preset
    if args.project:
        try:
            idx = preset.index("-p")
            preset[idx + 1] = args.project
        except ValueError:
            preset = ["-p", args.project] + preset

    return [exp_id] + preset


def _should_process(index: int, gpu_index: int, num_gpus: int) -> bool:
    if num_gpus <= 1:
        return True
    return (index % num_gpus) == gpu_index


# ── Main ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe_stems.py",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent("""\
            YourMT3+ — multi-stem single-track transcription
            -------------------------------------------------
            Transcribe N audio stems independently and write one MIDI file per stem.

            Program numbers follow General MIDI (0-127):
              0  = Acoustic Grand Piano
              33 = Electric Bass (finger)
              40 = Violin
              128 = Drums/Percussion  (use --is_drum True)
              -1  = let model predict  (default / unknown mode)

            Run  --list_programs  to print the full GM table.
        """),
    )

    # ── Input ─────────────────────────────────────────────────────────────────
    input_grp = parser.add_argument_group("Input")
    input_grp.add_argument(
        "stems", nargs="*", metavar="STEM",
        help="Audio stem files (.wav / .flac / .mp3 …). "
             "Alternatively use --stems_json for a manifest.",
    )
    input_grp.add_argument(
        "--stems_json", metavar="FILE",
        help="JSON manifest: list of {audio, name?, program?, is_drum?}. "
             "Takes precedence over positional stems.",
    )

    # ── Program assignment ────────────────────────────────────────────────────
    prog_grp = parser.add_argument_group("Program assignment")
    prog_grp.add_argument(
        "--programs", nargs="+", metavar="P",
        help="MIDI program number(s) per stem (0-127, 128=drums, -1=infer). "
             "One value is broadcast to all stems.",
    )
    prog_grp.add_argument(
        "--is_drum", nargs="+", metavar="BOOL",
        help="Mark each stem as drum track (True/False). "
             "Auto-set when program=128.",
    )
    prog_grp.add_argument(
        "--force_programs", action="store_true",
        help="When set, override the model's predicted programs with the values "
             "from --programs. Without this flag, --programs is used only to "
             "label the output MIDI (hint mode).",
    )

    # ── Output ────────────────────────────────────────────────────────────────
    out_grp = parser.add_argument_group("Output")
    out_grp.add_argument(
        "--output_dir", "-o", metavar="DIR",
        help="Directory for output MIDI files. "
             "Defaults to the directory of each input stem.",
    )
    out_grp.add_argument(
        "--output_names", nargs="+", metavar="NAME",
        help="Output filename stems (without extension), one per input stem. "
             "Defaults to the input file stem.",
    )
    out_grp.add_argument(
        "--tempo", type=int, default=120, metavar="BPM",
        help="Assumed tempo in BPM written into output MIDI (default: 120).",
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model_grp = parser.add_argument_group("Model")
    model_grp.add_argument(
        "--exp_id", metavar="ID",
        help="YourMT3 experiment/checkpoint ID (required unless provided in "
             "--model_args_json).",
    )
    model_grp.add_argument(
        "--preset", metavar="NAME", default=DEFAULT_PRESET,
        choices=list(MODEL_PRESETS.keys()),
        help=f"Named model preset (default: {DEFAULT_PRESET}). "
             f"Choices: {list(MODEL_PRESETS.keys())}.",
    )
    model_grp.add_argument(
        "--project", metavar="NAME", default="2024",
        help="YourMT3 project name used to locate checkpoints (default: 2024).",
    )
    model_grp.add_argument(
        "--model_args_json", metavar="FILE",
        help="JSON file containing model args list (overrides --preset). "
             "Format: ['-tk', 'mc13_full_plus_256', '-enc', 'perceiver-tf', ...]",
    )

    # ── Compute ───────────────────────────────────────────────────────────────
    compute_grp = parser.add_argument_group("Compute")
    compute_grp.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
        help="Torch device (default: cuda if available, else cpu).",
    )
    compute_grp.add_argument(
        "--batch_size", type=int, default=4, metavar="N",
        help="Inference batch size (audio segments per forward pass, default: 4).",
    )
    compute_grp.add_argument(
        "--gpu_index", type=int, default=0, metavar="IDX",
        help="This worker's GPU index for sharded inference (default: 0).",
    )
    compute_grp.add_argument(
        "--num_gpus", type=int, default=1, metavar="N",
        help="Total number of GPUs for sharded inference (default: 1).",
    )

    # ── Misc ──────────────────────────────────────────────────────────────────
    misc_grp = parser.add_argument_group("Misc")
    misc_grp.add_argument(
        "--stats", action="store_true",
        help="Print per-stem transcription statistics.",
    )
    misc_grp.add_argument(
        "--stats_out", metavar="FILE",
        help="Save per-stem statistics to a JSON file.",
    )
    misc_grp.add_argument(
        "--list_programs", action="store_true",
        help="Print the General MIDI program table and exit.",
    )
    misc_grp.add_argument(
        "--list_presets", action="store_true",
        help="Print available model presets and exit.",
    )
    misc_grp.add_argument(
        "--dry_run", action="store_true",
        help="Resolve inputs and print the plan, but skip model loading and inference.",
    )

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # ── Early-exit helpers ────────────────────────────────────────────────────
    if args.list_programs:
        _list_programs()
        return

    if args.list_presets:
        print("\nAvailable model presets:\n")
        for name, flags in MODEL_PRESETS.items():
            print(f"  {name}")
            print(f"    args: {' '.join(flags)}\n")
        return

    # ── Validate inputs ───────────────────────────────────────────────────────
    if not args.stems and not args.stems_json:
        parser.error("Provide at least one stem file or use --stems_json.")

    if not args.exp_id and not args.model_args_json:
        parser.error("--exp_id is required (or supply --model_args_json with an exp_id key).")

    # If model_args_json has exp_id embedded, read it now
    exp_id = args.exp_id
    if not exp_id and args.model_args_json:
        with open(args.model_args_json) as f:
            cfg = json.load(f)
        exp_id = cfg.get("exp_id")
        if not exp_id:
            parser.error("--model_args_json must contain an 'exp_id' key when --exp_id is omitted.")

    # ── Resolve stems ─────────────────────────────────────────────────────────
    stems = _resolve_stems_and_programs(args)

    # Validate files exist
    missing = [s["audio"] for s in stems if not s["audio"].exists()]
    if missing:
        parser.error(f"Audio file(s) not found:\n  " + "\n  ".join(str(p) for p in missing))

    # ── Determine output dirs ─────────────────────────────────────────────────
    for stem in stems:
        if args.output_dir:
            stem["out_dir"] = Path(args.output_dir)
        else:
            stem["out_dir"] = stem["audio"].parent

    # ── Print plan ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  YourMT3+ — Stem Transcription")
    print(f"{'='*60}")
    print(f"  Model   : {exp_id}  [{args.preset}]")
    print(f"  Device  : {args.device}")
    if args.num_gpus > 1:
        print(f"  Shard   : GPU {args.gpu_index} / {args.num_gpus}")
    print(f"  Stems   : {len(stems)}")
    print(f"  Tempo   : {args.tempo} BPM")
    print(f"  Force   : {args.force_programs}")
    print()

    for i, stem in enumerate(stems):
        prog_str = str(stem["program"]) if stem["program"] is not None else "infer"
        drum_str = " [drum]" if stem["is_drum"] else ""
        print(f"  [{i:02d}] {stem['audio'].name:<30}  prog={prog_str}{drum_str}  → {stem['out_dir'] / (stem['name'] + '.mid')}")
    print()

    if args.dry_run:
        print("  [dry-run] Stopping before model load.\n")
        return

    # ── Load model ────────────────────────────────────────────────────────────
    model_args = _build_model_args(args, exp_id)
    print(f"Loading model: {' '.join(model_args[:6])} …")
    model = load_model_checkpoint(model_args, device=args.device)
    print("Model loaded.\n")

    # ── Transcribe each stem ──────────────────────────────────────────────────
    all_stats = []
    success = 0
    failed = 0

    for idx, stem in enumerate(stems):
        if not _should_process(idx, args.gpu_index, args.num_gpus):
            continue

        audio_path = stem["audio"]
        out_dir = stem["out_dir"]
        out_dir.mkdir(parents=True, exist_ok=True)

        program = stem["program"]
        is_drum = stem["is_drum"]

        # In hint mode (not --force_programs), don't pass program to the transcriber
        forced_program = program if args.force_programs else None
        forced_is_drum = is_drum if args.force_programs else False

        audio_info = {
            "filepath": str(audio_path),
            "output": str(out_dir),
            "file_name": stem["name"],
            "tempo": args.tempo,
        }

        prog_label = GM_PROGRAMS.get(program, f"program {program}") if program is not None else "inferred"
        print(f"[{idx:02d}/{len(stems)-1}] {audio_path.name}  ({prog_label}) …", end=" ", flush=True)

        try:
            midi_path, pred_notes, stats = transcribe_single_inst(
                model,
                audio_info,
                program=forced_program,
                is_drum=forced_is_drum,
            )
            print(f"✓  {stats['total_notes']} notes  →  {Path(midi_path).name}")

            stats["stem"] = stem["name"]
            stats["audio"] = str(audio_path)
            stats["midi"] = str(midi_path)
            stats["program_hint"] = program
            stats["is_drum_hint"] = is_drum
            all_stats.append(stats)
            success += 1

        except Exception as exc:
            print(f"✗  ERROR: {exc}")
            all_stats.append({
                "stem": stem["name"],
                "audio": str(audio_path),
                "error": str(exc),
            })
            failed += 1

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Done: {success} succeeded, {failed} failed.")
    print(f"{'='*60}\n")

    if args.stats:
        for s in all_stats:
            if "error" in s:
                print(f"  {s['stem']}: ERROR — {s['error']}")
                continue
            print(f"  {s['stem']}:")
            print(f"    total_notes   : {s['total_notes']}")
            print(f"    num_programs  : {s['num_programs']}")
            if s["program_note_counts"]:
                for prog, cnt in sorted(s["program_note_counts"].items()):
                    pct = s["program_note_percent"].get(prog, 0.0) * 100
                    name = GM_PROGRAMS.get(prog, f"prog {prog}")
                    drum_tag = " [drum]" if s["is_drum_flags"].get(prog, False) else ""
                    print(f"      prog {prog:>3} ({name}{drum_tag}): {cnt} notes ({pct:.1f}%)")
        print()

    if args.stats_out:
        with open(args.stats_out, "w") as f:
            json.dump(all_stats, f, indent=2)
        print(f"  Stats saved → {args.stats_out}\n")


if __name__ == "__main__":
    main()
