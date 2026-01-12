"""
MIDI processing utilities using symusic.

This module draws much inspiration from musetok/data_processing/midi2events.py,
replicating the quantization and event creation logic but using symusic instead
of miditoolkit for MIDI parsing.

Reference: musetok/data_processing/midi2events.py
"""

import numpy as np
from symusic import Score
from typing import Union, List, Dict, Tuple, Optional
from copy import deepcopy
import collections
import logging

# Constants from musetok/data_processing/midi2events.py
BEAT_RESOL = 480
TICK_RESOL = BEAT_RESOL // 12  # 40
TRIPLET_RESOL = BEAT_RESOL // 24  # 20
INSTR_NAME_MAP = {'piano': 0}
MIN_VELOCITY = 40
NOTE_SORTING = 1  # 0: ascending / 1: descending

DEFAULT_TEMPO = 110
DEFAULT_VELOCITY_BINS = np.linspace(4, 127, 42, dtype=int)
DEFAULT_BPM_BINS = np.linspace(32, 224, 64 + 1, dtype=int)
DEFAULT_SHIFT_BINS = np.linspace(-TICK_RESOL, TICK_RESOL, TICK_RESOL + 1, dtype=int)
DEFAULT_TIME_SIGNATURE = ['4/4', '2/4', '3/4', '2/2', '3/8', '6/8']


class NoteEvent(object):
    """Note event class for quantization (from midi2events.py)."""
    def __init__(self, start, end, pitch, velocity, bar_resol, default_onset):
        self.start = start
        self.quantized_start = start // bar_resol * bar_resol + default_onset[np.argmin(abs(default_onset - start % bar_resol))]
        self.is_valid_start = self.quantized_start % (TICK_RESOL * 3) == 0
        self.is_triplet_candidate = self.quantized_start % (TRIPLET_RESOL * 4) == 0
        self.is_triplet = None
        self.end = end
        self.pitch = pitch
        self.duration = None
        self.velocity = velocity
        
    def __repr__(self):
        return f'Note(quantized_start={self.quantized_start:d}, end={self.end:d}, pitch={self.pitch}, velocity={self.velocity})'


def check_triplet(start, quantized_timing, bar_resol):
    """Check if a start time is part of a valid triplet pattern (from midi2events.py)."""
    bar_idx = start // bar_resol
    if (start + TRIPLET_RESOL * 4) in quantized_timing and \
        (start + TRIPLET_RESOL * 8) in quantized_timing and \
        (start + TRIPLET_RESOL * 12) in quantized_timing and \
        (start + TRIPLET_RESOL * 16) in quantized_timing and \
        (start + TRIPLET_RESOL * 20) in quantized_timing and \
        (start + TRIPLET_RESOL * 20) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 4, start + TRIPLET_RESOL * 8, 
                    start + TRIPLET_RESOL * 12, start + TRIPLET_RESOL * 16, start + TRIPLET_RESOL * 20]
    elif (start + TRIPLET_RESOL * 8) in quantized_timing and \
        (start + TRIPLET_RESOL * 16) in quantized_timing and \
        (start + TRIPLET_RESOL * 24) in quantized_timing and \
        (start + TRIPLET_RESOL * 32) in quantized_timing and \
        (start + TRIPLET_RESOL * 40) in quantized_timing and \
        (start + TRIPLET_RESOL * 40) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 8, start + TRIPLET_RESOL * 16, 
                    start + TRIPLET_RESOL * 24, start + TRIPLET_RESOL * 32, start + TRIPLET_RESOL * 40]
    elif (start + TRIPLET_RESOL * 16) in quantized_timing and \
        (start + TRIPLET_RESOL * 32) in quantized_timing and \
        (start + TRIPLET_RESOL * 48) in quantized_timing and \
        (start + TRIPLET_RESOL * 64) in quantized_timing and \
        (start + TRIPLET_RESOL * 80) in quantized_timing and \
        (start + TRIPLET_RESOL * 80) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 16, start + TRIPLET_RESOL * 32, 
                    start + TRIPLET_RESOL * 48, start + TRIPLET_RESOL * 64, start + TRIPLET_RESOL * 80]
    elif (start + TRIPLET_RESOL * 4) in quantized_timing and \
        (start + TRIPLET_RESOL * 8) in quantized_timing and \
        (start + TRIPLET_RESOL * 8) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 4, start + TRIPLET_RESOL * 8]
    elif (start + TRIPLET_RESOL * 8) in quantized_timing and \
        (start + TRIPLET_RESOL * 16) in quantized_timing and \
        (start + TRIPLET_RESOL * 16) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 8, start + TRIPLET_RESOL * 16]
    elif (start + TRIPLET_RESOL * 16) in quantized_timing and \
        (start + TRIPLET_RESOL * 32) in quantized_timing and \
        (start + TRIPLET_RESOL * 32) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 16, start + TRIPLET_RESOL * 32]
    elif (start + TRIPLET_RESOL * 32) in quantized_timing and \
        (start + TRIPLET_RESOL * 64) in quantized_timing and \
        (start + TRIPLET_RESOL * 64) < bar_resol * (bar_idx + 1):
            return [start + TRIPLET_RESOL * 32, start + TRIPLET_RESOL * 64]
    else:
        return False


def convert_ticks_per_quarter(score: Score, target_ticks_per_quarter: int = BEAT_RESOL) -> Score:
    """
    Convert score to target ticks_per_quarter by scaling all timing values.
    
    This replicates the tick_change logic from musetok/data_processing/midi2events.py.
    All timing values (note start/end, tempo changes, time signature changes) are scaled
    proportionally: new_time = old_time * (target_ticks / orig_ticks)
    
    Args:
        score: symusic Score object
        target_ticks_per_quarter: Target ticks per quarter note (default: 480)
    
    Returns:
        Score object with converted ticks_per_quarter (may be same object if modified in-place)
    """
    if score.ticks_per_quarter == target_ticks_per_quarter:
        return score
    
    orig_ticks = score.ticks_per_quarter
    scale_factor = target_ticks_per_quarter / orig_ticks
    
    # Scale all timing values in the score
    # Scale time signatures
    for ts in score.time_signatures:
        ts.time = int(ts.time * scale_factor)
    
    # Scale tempo changes
    for tempo in score.tempos:
        tempo.time = int(tempo.time * scale_factor)
    
    # Scale all notes in all tracks
    for track in score.tracks:
        # Scale note timings
        for note in track.notes:
            new_start = int(note.start * scale_factor)
            orig_duration = note.end - note.start
            new_duration = int(orig_duration * scale_factor)
            note.start = new_start
            note.end = new_start + new_duration
        
        # Scale control changes
        for cc in track.controls:
            cc.time = int(cc.time * scale_factor)
    
    # Update ticks_per_quarter
    score.ticks_per_quarter = target_ticks_per_quarter
    
    return score


def tick_scaling_symusic(score: Score, target_ticks_per_beat: int = BEAT_RESOL) -> Score:
    """
    Scale score to target ticks_per_beat (from midi2events.py tick_change logic).
    
    This is a wrapper that calls convert_ticks_per_quarter for backward compatibility.
    """
    return convert_ticks_per_quarter(score, target_ticks_per_beat)


def load_midi_symusic(midi_path_or_bytes: Union[str, bytes], exclude_tracks: Optional[List[str]] = None) -> Score:
    """
    Load MIDI file using symusic.
    
    Args:
        midi_path_or_bytes: Path to MIDI file or MIDI bytes
        exclude_tracks: List of track names to exclude (e.g., ['Chord'] for EMOPIA)
    
    Returns:
        Score object with MIDI data
    """
    if isinstance(midi_path_or_bytes, bytes):
        score = Score.from_midi(midi_path_or_bytes)
    else:
        score = Score(midi_path_or_bytes)
    
    # For EMOPIA MIDI files: exclude Chord track, keep Melody + Texture + Bass
    if exclude_tracks:
        # Filter out tracks matching exclude_tracks names
        # This ensures we use all tracks except Chord for emotion prediction
        # Note: symusic Score tracks might be immutable, so we filter during processing
        # The actual filtering happens in score_to_corpus_strict when processing notes
        # Store exclude_tracks in score metadata for later use
        if not hasattr(score, '_exclude_tracks'):
            score._exclude_tracks = exclude_tracks
    
    # Convert ticks_per_quarter to 480 (BEAT_RESOL) if needed
    if score.ticks_per_quarter != BEAT_RESOL:
        logging.debug(f"Converting MIDI from {score.ticks_per_quarter} to {BEAT_RESOL} ticks_per_quarter")
        score = convert_ticks_per_quarter(score, BEAT_RESOL)
    
    return score


def get_time_signature(score: Score) -> Tuple[int, int]:
    """
    Extract time signature from score.
    Defaults to 4/4 if not found (from midi2events.py analyzer logic).
    """
    if len(score.time_signatures) == 0:
        return (4, 4)  # Default to 4/4
    
    # Get first time signature
    time_sig = score.time_signatures[0]
    return (time_sig.numerator, time_sig.denominator)


def get_tempo(score: Score) -> float:
    """
    Extract tempo (BPM) from score.
    Returns median tempo, default to DEFAULT_TEMPO (from midi2events.py analyzer logic).
    """
    if len(score.tempos) == 0:
        return DEFAULT_TEMPO
    
    # Get first 40 tempo changes and compute median
    tempos = [t.bpm for t in score.tempos[:40]]
    tempo_median = np.median(tempos) if tempos else DEFAULT_TEMPO
    return float(tempo_median)


def score_to_corpus_strict(score: Score, bar_resol: int, remove_overlap: bool = True):
    """
    Quantize MIDI data from symusic Score (adapted from midi2events.py midi2corpus_strict).
    
    Returns:
        song_data: dict with 'notes' (note grid) and 'metadata' (bpm, last_bar, time_sig)
    """
    # Load notes from all tracks (filtering for piano if needed)
    instr_notes = collections.defaultdict(list)
    
    # Collect all notes from all tracks
    # Note: symusic API - tracks is a list, each track has notes
    # For EMOPIA: exclude Chord track if specified
    exclude_tracks = getattr(score, '_exclude_tracks', None)
    
    all_notes = []
    for track in score.tracks:
        # Filter for piano tracks (or accept all if no name filtering)
        # For now, we'll take all non-drum tracks
        # Check if track has is_drum attribute, otherwise include all
        is_drum = getattr(track, 'is_drum', False)
        if is_drum:
            continue
        
        # Check if track should be excluded (e.g., Chord track for EMOPIA)
        if exclude_tracks:
            track_name = getattr(track, 'name', '').lower()
            should_exclude = any(excluded.lower() in track_name for excluded in exclude_tracks)
            if should_exclude:
                continue
        
        # Access notes from track
        track_notes = getattr(track, 'notes', [])
        for note in track_notes:
            all_notes.append({
                'start': int(note.start),
                'end': int(note.end),
                'pitch': int(note.pitch),
                'velocity': int(note.velocity),
            })
    
    if len(all_notes) == 0:
        raise ValueError('Detected empty MIDI file!')
    
    # Sort notes
    if NOTE_SORTING == 0:
        all_notes.sort(key=lambda x: (x['start'], x['pitch']))
    elif NOTE_SORTING == 1:
        all_notes.sort(key=lambda x: (x['start'], -x['pitch']))
    else:
        raise ValueError('Unknown type of sorting.')
    
    # Group by instrument (for now, all go to instrument 0 - piano)
    instr_notes[0] = all_notes
    
    # Load global bpm (we'll compute from score tempos)
    global_bpm = get_tempo(score)
    
    # --- step 1: adjust onset values --- #
    # Valid onset values (triplet 80*n or non-triplet 120*n)
    default_onset = np.unique(np.concatenate([
        np.arange(0, bar_resol, TRIPLET_RESOL * 4),
        np.arange(0, bar_resol, TICK_RESOL * 3),
        np.array([bar_resol])
    ]))
    default_normal_onset = np.unique(np.concatenate([
        np.arange(0, bar_resol, TICK_RESOL * 3),
        np.array([bar_resol])
    ]))  # non-triplet
    
    instr_quantized_notes = collections.defaultdict(list)
    instr_quantized_timing = collections.defaultdict(list)
    
    for key in instr_notes.keys():
        # Quantize onsets of all notes
        notes = instr_notes[key]
        for note_dict in notes:
            quantized_note = NoteEvent(
                note_dict['start'], note_dict['end'], note_dict['pitch'],
                note_dict['velocity'], bar_resol, default_onset
            )
            instr_quantized_notes[key].append(quantized_note)
            instr_quantized_timing[key].append(quantized_note.quantized_start)
        
        # Keep onsets of normal notes or valid triplets, adjust onsets of invalid triplet
        valid_timing = []
        for quantized_note in instr_quantized_notes[key]:
            # If not triplet notes or already belong to some triplets
            if not quantized_note.is_triplet_candidate:
                quantized_note.is_triplet = False
                continue
            else:
                if quantized_note.quantized_start in valid_timing:
                    quantized_note.is_triplet = True
                    continue
                triplet_result = check_triplet(quantized_note.quantized_start, instr_quantized_timing[key], bar_resol)
                if triplet_result:
                    valid_timing.append(quantized_note.quantized_start)
                    valid_timing.extend(triplet_result)
                    quantized_note.is_triplet = True
                else:
                    quantized_note.quantized_start = quantized_note.start // bar_resol * bar_resol + \
                        default_normal_onset[np.argmin(abs(default_normal_onset - quantized_note.start % bar_resol))]
                    quantized_note.is_triplet = False
        
        # --- step 2: adjust duration --- #
        if TICK_RESOL == 20:
            default_normal_duration = np.array([
                BEAT_RESOL // 8,                   # 1/32       - 60
                BEAT_RESOL // 4,                   # 1/16       - 120
                BEAT_RESOL // 4 + BEAT_RESOL // 8, # 1/16 + 1/32 - 180
                BEAT_RESOL // 2,                   # 1/8        - 240
                BEAT_RESOL // 2 + BEAT_RESOL // 4, # 1/8 + 1/16 - 360
                BEAT_RESOL,                        # 1/4        - 480
                BEAT_RESOL + BEAT_RESOL // 2,      # 1/4 + 1/8  - 720
                2 * BEAT_RESOL,                    # 1/2        - 960
                2 * BEAT_RESOL + BEAT_RESOL,       # 1/2 + 1/4  - 1440
                4 * BEAT_RESOL                     # 1          - 1920
            ])
        elif TICK_RESOL == 40:
            default_normal_duration = np.array([
                BEAT_RESOL // 4,                   # 1/16       - 120
                BEAT_RESOL // 2,                   # 1/8        - 240
                BEAT_RESOL // 2 + BEAT_RESOL // 4, # 1/8 + 1/16 - 360
                BEAT_RESOL,                        # 1/4        - 480
                BEAT_RESOL + BEAT_RESOL // 2,      # 1/4 + 1/8  - 720
                2 * BEAT_RESOL,                    # 1/2        - 960
                2 * BEAT_RESOL + BEAT_RESOL,       # 1/2 + 1/4  - 1440
                4 * BEAT_RESOL                     # 1          - 1920
            ])
        else:
            raise ValueError(f'invalid tick resolution {TICK_RESOL}')
        
        default_triplet_duration = np.array([
            BEAT_RESOL // 6,                  # 1/8 // 3   - 80
            BEAT_RESOL // 3,                   # 1/4 // 3   - 160
            2 * BEAT_RESOL // 3,               # 1/2 // 3   - 320
            4 * BEAT_RESOL // 3                # 1 // 3     - 640
        ])
        
        default_duration = np.concatenate([default_normal_duration, default_triplet_duration])
        
        for quantized_note in instr_quantized_notes[key]:
            assert quantized_note.is_triplet is not None
            duration_diff = quantized_note.end - quantized_note.quantized_start
            if quantized_note.is_triplet:
                quantized_note.duration = default_duration[np.argmin(abs(default_duration - duration_diff))]
            else:
                quantized_note.duration = default_normal_duration[np.argmin(abs(default_normal_duration - duration_diff))]
            quantized_note.end = quantized_note.quantized_start + quantized_note.duration
        
        # --- step 3: remove note overlap --- #
        if remove_overlap:
            # Remove all overlap between two consecutive notes
            onsets2notes = collections.defaultdict(list)
            onsets2ends = collections.defaultdict(int)
            for quantized_note in instr_quantized_notes[key]:
                onsets2notes[int(quantized_note.quantized_start)].append(quantized_note)
                onsets2ends[int(quantized_note.quantized_start)] = max(
                    onsets2ends[int(quantized_note.quantized_start)], quantized_note.end
                )
            onsets2notes = sorted(onsets2notes.items(), key=lambda x: x[0])
            onsets2ends = sorted(onsets2ends.items(), key=lambda x: x[0])
            
            for i in range(len(onsets2notes) - 1):
                for quantized_note in onsets2notes[i][1]:
                    j = i + 1
                    while j < len(onsets2notes) - 1 and quantized_note.end > onsets2ends[j][0]:
                        if quantized_note.end >= onsets2ends[j][1]:
                            j += 1
                            continue
                        else:
                            if onsets2ends[j][0] - quantized_note.quantized_start in default_duration:
                                quantized_note.end = onsets2ends[j][0]
                            break
        else:
            # Remove only the overlap between two notes with the same pitch
            note2onsets = collections.defaultdict(list)
            for quantized_note in instr_quantized_notes[key]:
                note2onsets[int(quantized_note.pitch)].append(quantized_note)
            for quantized_note in instr_quantized_notes[key]:
                larger_note_start = [
                    note.quantized_start for note in note2onsets[int(quantized_note.pitch)]
                    if note.quantized_start > quantized_note.quantized_start
                ]
                if len(larger_note_start) > 0:
                    closest_note_start = np.min(larger_note_start)
                    if quantized_note.end > closest_note_start and closest_note_start - quantized_note.quantized_start >= TICK_RESOL * 2:
                        duration_diff = default_duration - (closest_note_start - quantized_note.quantized_start)
                        quantized_note.duration = default_duration[
                            np.where(duration_diff <= 0)[0][np.argmax(duration_diff[duration_diff <= 0])]
                        ]
                        quantized_note.end = quantized_note.quantized_start + quantized_note.duration
        
        # Convert to simple note dicts for grid
        new_notes = []
        for quantized_note in instr_quantized_notes[key]:
            new_notes.append({
                'start': quantized_note.quantized_start,
                'end': quantized_note.end,
                'pitch': quantized_note.pitch,
                'velocity': quantized_note.velocity,
                'duration': quantized_note.duration,
            })
        instr_notes[key] = new_notes
    
    # --- process items to grid --- #
    # Compute empty bar offset at head
    first_note_time = min([instr_notes[k][0]['start'] for k in instr_notes.keys()])
    last_note_time = max([instr_notes[k][-1]['end'] for k in instr_notes.keys()])
    
    offset = first_note_time // bar_resol  # empty bar
    last_bar = int(np.ceil(last_note_time / bar_resol)) - offset
    
    # Process notes into grid
    instr_grid = dict()
    for key in instr_notes.keys():
        notes = instr_notes[key]
        note_grid = collections.defaultdict(list)
        for note in notes:
            note['start'] = note['start'] - offset * bar_resol
            note['end'] = note['end'] - offset * bar_resol
            
            # Quantize start (already quantized, but use directly)
            quant_time = note['start']
            
            # Velocity binning
            note['velocity'] = DEFAULT_VELOCITY_BINS[np.argmin(abs(DEFAULT_VELOCITY_BINS - note['velocity']))]
            
            # Duration (already set, but cap at bar_resol)
            if note['duration'] > bar_resol:
                note['duration'] = bar_resol
            
            # Append to grid
            note_grid[quant_time].append(note)
        
        # Set to track
        instr_grid[key] = note_grid.copy()
    
    # Process global bpm
    global_bpm = DEFAULT_BPM_BINS[np.argmin(abs(DEFAULT_BPM_BINS - global_bpm))]
    
    # Load time signature
    time_sig_num, time_sig_den = get_time_signature(score)
    global_time = type('TimeSignature', (), {'numerator': time_sig_num, 'denominator': time_sig_den})()
    
    # Collect
    song_data = {
        'notes': instr_grid,
        'metadata': {
            'global_bpm': global_bpm,
            'last_bar': last_bar,
            'global_time': global_time
        }
    }
    
    return song_data


def create_event(name, value):
    """Create an event dictionary (from midi2events.py)."""
    event = dict()
    event['name'] = name
    event['value'] = value
    return event


def corpus_to_events(data, bar_resol, has_velocity=False, time_first=False, repeat_beat=False, remove_short=False):
    """
    Convert quantized corpus data to REMI events (from midi2events.py corpus2events).
    
    Args:
        data: song_data dict from score_to_corpus_strict
        bar_resol: bar resolution in ticks
        has_velocity: whether to include velocity events
        time_first: whether to put time signature at start of bar
        repeat_beat: whether to repeat beat event for each note
        remove_short: whether to raise error for short pieces
    
    Returns:
        position: List of event indices where bars start
        events: List of event dictionaries
    """
    # Global tag
    global_end = data['metadata']['last_bar'] * bar_resol
    
    # Global time
    global_time = data['metadata']['global_time']
    
    # Process
    position = []
    final_sequence = []
    
    for bar_step in range(0, global_end, bar_resol):
        sequence = [create_event('Bar', None)]
        
        # --- time signature (first) --- #
        if time_first:
            sequence.append(create_event('Time_Signature', 
                                       f'{global_time.numerator}/{global_time.denominator}'))
        
        # --- piano track --- #
        for timing in range(bar_step, bar_step + bar_resol, TICK_RESOL):
            events = []
            
            # Unpack notes at this timing
            t_notes = data['notes'][0].get(timing, [])
            
            # Note events
            if len(t_notes):
                for note in t_notes:
                    if note['pitch'] not in range(21, 109):
                        raise ValueError(f'invalid pitch value {note["pitch"]}')
                    
                    if repeat_beat:
                        events.append(create_event('Beat', (timing - bar_step) // TICK_RESOL))
                    
                    if has_velocity:
                        events.extend([
                            create_event('Note_Pitch', note['pitch']),
                            create_event('Note_Duration', int(note['duration'])),
                            create_event('Note_Velocity', int(note['velocity'])),
                        ])
                    else:
                        events.extend([
                            create_event('Note_Pitch', note['pitch']),
                            create_event('Note_Duration', int(note['duration'])),
                        ])
            
            # Collect & beat
            if len(events):
                if not repeat_beat:
                    sequence.append(create_event('Beat', (timing - bar_step) // TICK_RESOL))
                sequence.extend(events)
        
        # --- time signature (last) --- #
        if not time_first:
            sequence.append(create_event('Time_Signature', 
                                       f'{global_time.numerator}/{global_time.denominator}'))
        
        # --- EOS --- #
        if bar_step == global_end - bar_resol:
            sequence.append(create_event('EOS', None))
        
        position.append(len(final_sequence))
        final_sequence.extend(sequence)
    
    if len(position) < 8 and remove_short:
        raise ValueError('music piece too short')
    
    return position, final_sequence


def midi_to_events_symusic(score: Score, 
                          has_velocity: bool = False,
                          time_first: bool = False,
                          repeat_beat: bool = True,
                          remove_overlap: bool = True) -> Tuple[List[int], List[Dict[str, any]]]:
    """
    Convert symusic Score to REMI events.
    
    This is the main entry point, replicating the flow from midi2events.py:
    1. Extract time signature and compute bar_resol
    2. Quantize notes using score_to_corpus_strict
    3. Convert to events using corpus_to_events
    
    Args:
        score: symusic Score object
        has_velocity: whether to include velocity events
        time_first: whether to put time signature at start of bar
        repeat_beat: whether to repeat beat event for each note
        remove_overlap: whether to remove note overlaps during quantization
    
    Returns:
        bar_positions: List of event indices where bars start
        events: List of event dictionaries with 'name' and 'value' keys
    """
    # Ensure ticks_per_quarter is 480 (should already be converted in load_midi_symusic)
    if score.ticks_per_quarter != BEAT_RESOL:
        logging.warning(f"Score has ticks_per_quarter={score.ticks_per_quarter}, expected {BEAT_RESOL}. Converting...")
        score = convert_ticks_per_quarter(score, BEAT_RESOL)
    
    # Get time signature and compute bar_resol
    time_sig_num, time_sig_den = get_time_signature(score)
    quarters_per_bar = 4 * time_sig_num / time_sig_den
    bar_resol = int(BEAT_RESOL * quarters_per_bar)
    
    # Convert score to quantized corpus
    song_data = score_to_corpus_strict(score, bar_resol, remove_overlap=remove_overlap)
    
    # Convert corpus to events
    bar_positions, events = corpus_to_events(
        song_data, bar_resol,
        has_velocity=has_velocity,
        time_first=time_first,
        repeat_beat=repeat_beat,
        remove_short=False
    )
    
    return bar_positions, events


def get_bar_positions(events: List[Dict[str, any]]) -> List[int]:
    """Extract bar boundary positions from events."""
    return [i for i, event in enumerate(events) if event['name'] == 'Bar']


def quantize_note_timing(note_start: int, bar_resol: int) -> int:
    """
    Quantize note start time to grid.
    
    Note: This is a simplified version. The full quantization logic is in
    score_to_corpus_strict which handles triplets and other complexities.
    """
    # Simple quantization to TICK_RESOL grid
    return int(np.round(note_start / TICK_RESOL) * TICK_RESOL)
