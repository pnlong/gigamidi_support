"""
2×2 alignment QC plot: audio (seconds) vs MIDI (ticks).

Also writes a per-song folder with:
  - alignment.png
  - original_audio.{mp3,wav,...}  (copy of source audio)
  - midi_synth.wav                (MIDI rendered via symusic)

Usage:
    python va_cont/tools/plot_va_alignment.py --dataset deam --song_id 1000
    python va_cont/tools/plot_va_alignment.py --dataset merp --song_id 1 --show-bars --show
"""

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from datasets import get_dataset
from utils.midi_utils import load_midi_symusic, BEAT_RESOL
from va_utils import (
    DEFAULT_TARGET_HZ,
    aggregate_va_to_bars,
    load_continuous_va,
    resample_va_dict,
)


def load_waveform(audio_path: Path):
    """Return (times_sec, amplitude) mono waveform."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(str(audio_path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        y = wav.squeeze(0).numpy()
        t = np.arange(len(y)) / sr
        return t, y
    except Exception:
        import librosa
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        t = np.arange(len(y)) / sr
        return t, y


def collect_piano_roll_notes(score):
    """Return lists of (start_tick, end_tick, pitch) for all notes."""
    starts, ends, pitches = [], [], []
    for track in score.tracks:
        if track.is_drum:
            continue
        for note in track.notes:
            starts.append(int(note.time))
            ends.append(int(note.time + note.duration))
            pitches.append(int(note.pitch))
    return starts, ends, pitches


def plot_piano_roll(ax, score, xlim=None):
    starts, ends, pitches = collect_piano_roll_notes(score)
    if not starts:
        ax.text(0.5, 0.5, "No notes", ha="center", va="center", transform=ax.transAxes)
        return
    for s, e, p in zip(starts, ends, pitches):
        ax.plot([s, e], [p, p], color="#2563EB", linewidth=1.0, solid_capstyle="butt")
    ax.set_ylabel("pitch")
    if xlim:
        ax.set_xlim(xlim)


def plot_va_curves(ax, x, valence, arousal, label_prefix=""):
    ax.plot(x, valence, color="#2563EB", linewidth=1.2, label=f"{label_prefix}valence")
    ax.plot(x, arousal, color="#16A34A", linewidth=1.2, label=f"{label_prefix}arousal")
    ax.set_ylim(-1.05, 1.05)
    ax.axhline(0, color="#9CA3AF", linewidth=0.5)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylabel("V / A")


def export_qc_audio(out_dir: Path, audio_path: Path, score, soundfont: str | None = None, sample_rate: int = 44100):
    """Copy original audio and write MIDI synthesized to WAV in out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)

    original_out = out_dir / f"original_audio{audio_path.suffix}"
    shutil.copy2(audio_path, original_out)
    logging.info(f"Copied original audio → {original_out}")

    from symusic import Synthesizer, dump_wav

    synth = Synthesizer(sf_path=soundfont, sample_rate=sample_rate) if soundfont else Synthesizer(sample_rate=sample_rate)
    buffer = synth.render(score, stereo=True)
    midi_out = out_dir / "midi_synth.wav"
    dump_wav(str(midi_out), buffer, sample_rate, use_int16=True)
    logging.info(f"Wrote MIDI synthesis → {midi_out}")


def parse_args():
    p = argparse.ArgumentParser(description="Plot audio vs MIDI-tick V/A alignment QC grid.")
    p.add_argument("--dataset", required=True, choices=["deam", "memo2496", "merp"])
    p.add_argument("--song_id", required=True)
    p.add_argument("--storage_dir", default=None)
    p.add_argument("--output_path", default=None)
    p.add_argument("--show-bars", action="store_true", help="Draw bar boundaries on right column")
    p.add_argument("--show", action="store_true")
    p.add_argument("--skip-audio", action="store_true", help="Skip copying audio / MIDI synthesis")
    p.add_argument("--soundfont", default=None, help="Optional path to .sf2/.sf3 (default: symusic built-in)")
    p.add_argument("--synth_sample_rate", type=int, default=44100)
    p.add_argument("--target_hz", type=float, default=DEFAULT_TARGET_HZ)
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    ds = get_dataset(args.dataset, args.storage_dir)
    song_id = args.song_id

    audio_path = ds.audio_path(song_id)
    midi_path = ds.midi_path(song_id)
    cont_path = ds.continuous_va_path(song_id)

    if not audio_path.is_file():
        raise FileNotFoundError(f"Audio not found: {audio_path}")
    if not midi_path.is_file():
        raise FileNotFoundError(f"MIDI not found: {midi_path}")

    # Left column: audio seconds
    t_wav, y_wav = load_waveform(audio_path)
    v_dict, a_dict = ds.load_audio_va_annotations(song_id)
    min_t = ds.min_annotation_time()
    t_va, v_audio, a_audio = resample_va_dict(v_dict, a_dict, args.target_hz, min_t)

    # Right column: MIDI ticks
    score = load_midi_symusic(str(midi_path))
    if cont_path.is_file():
        cdata = load_continuous_va(cont_path)
        ticks_r = cdata["ticks"]
        v_midi = cdata["valence"]
        a_midi = cdata["arousal"]
        bar_resol = cdata["bar_resol"]
    else:
        from va_utils import convert_va_to_midi_ticks
        ticks_r, v_midi, a_midi, _, bar_resol = convert_va_to_midi_ticks(
            score, t_va, v_audio, a_audio
        )

    fig, axes = plt.subplots(2, 2, figsize=(14, 8), sharex="col")

    # (1,1) waveform
    ax_w = axes[0, 0]
    ax_w.plot(t_wav, y_wav, color="#374151", linewidth=0.4)
    ax_w.set_ylabel("amplitude")
    ax_w.set_title(f"{ds.name} / {song_id} — audio (seconds)")

    # (2,1) audio V/A
    ax_va_l = axes[1, 0]
    if len(t_va):
        plot_va_curves(ax_va_l, t_va, v_audio, a_audio)
    ax_va_l.set_xlabel("audio seconds")

    # (1,2) piano roll — ticks
    ax_pr = axes[0, 1]
    tick_max = int(ticks_r.max()) if len(ticks_r) else bar_resol * 32
    plot_piano_roll(ax_pr, score, xlim=(0, tick_max))
    ax_pr.set_title(f"{ds.name} / {song_id} — MIDI (ticks)")

    # (2,2) converted V/A — ticks
    ax_va_r = axes[1, 1]
    if len(ticks_r):
        plot_va_curves(ax_va_r, ticks_r, v_midi, a_midi)
    ax_va_r.set_xlabel("MIDI ticks")

    if args.show_bars and bar_resol > 0:
        n_bars = tick_max // bar_resol + 1
        for i in range(n_bars + 1):
            bt = i * bar_resol
            for ax in (ax_pr, ax_va_r):
                ax.axvline(bt, color="#D1D5DB", linewidth=0.4, alpha=0.7)
        if len(ticks_r) and cont_path.is_file():
            bar_labels = aggregate_va_to_bars(ticks_r, v_midi, a_midi, bar_resol, n_bars)
            for entry in bar_labels:
                bi, bv, ba = entry
                mid = (bi + 0.5) * bar_resol
                ax_va_r.scatter([mid], [bv], color="#2563EB", s=12, alpha=0.5, zorder=5)
                ax_va_r.scatter([mid], [ba], color="#16A34A", s=12, alpha=0.5, zorder=5)

    fig.suptitle(
        "Alignment QC — compare bottom-row curve shapes (left=absolute sec, right=metrical ticks)",
        fontsize=11,
    )
    fig.tight_layout()

    if args.output_path is None:
        out_dir = ds.va_dir() / "qc_plots" / str(song_id)
        args.output_path = str(out_dir / "alignment.png")
    else:
        out_dir = Path(args.output_path).parent

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=150, bbox_inches="tight")
    logging.info(f"Saved {args.output_path}")

    if not args.skip_audio:
        try:
            export_qc_audio(
                out_dir,
                audio_path,
                score,
                soundfont=args.soundfont,
                sample_rate=args.synth_sample_rate,
            )
        except Exception as exc:
            logging.warning(f"Audio export failed (plot still saved): {exc}")

    if args.show:
        matplotlib.use("TkAgg")
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
