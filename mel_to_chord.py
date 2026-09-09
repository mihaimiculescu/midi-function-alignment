#!/usr/bin/env python3
"""
Replacement mel_to_chord.py

The CP model remains the harmonic proposal generator.  The generated note
attacks are NOT used as the final accompaniment.  Instead:

    raw CP proposal
        -> 32-step local pitch-class analysis every 8 steps
        -> duration/persistence-weighted harmonic evidence
        -> local chord interpretation
        -> temporal hysteresis
        -> simple sustained 3/4-note voicings

No global key is imposed after the initial two-bar prompt, so local harmonic
changes/modulations remain detectable.
"""

import argparse
import copy
import math
import os
import re

import mido
import numpy as np
import pretty_midi
import torch

from cp_transformer_yinyang import RoformerYinyang, PreprocessingParameters
from preprocess_large_midi_dataset import preprocess_midi, DURATION_TEMPLATES


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_BPM = 120.0
DEFAULT_NUMERATOR = 4
DEFAULT_DENOMINATOR = 4
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SAMPLES = 1
DEFAULT_SEED = 0
DEFAULT_CONDITIONING_MULTIPLIER = 1.0

# Kept for command-line compatibility.  It is intentionally NOT used to
# retain generated attacks anymore.
DEFAULT_VICINITY = 0.75

OUTPUT_VELOCITY = 100
MELODY_PROGRAM = 64
CHORD_PROGRAM = 49
CHORD_OCTAVE = 3

MODEL_MAX_LENGTH = 384

LOWEST_OUTPUT_PITCH = 36    # C2
HIGHEST_OUTPUT_PITCH = 67    # G4
MAX_VOICING_NOTES = 4

# ---------------------------------------------------------------------------
# Harmonic decoder
# ---------------------------------------------------------------------------

TYPE_CHANGE_PENALTY = 0.07

# Short horizon: actual chord identity / agility.
CHORD_WINDOW = 8
SKELETON_HOP = 4

# Long FUTURE horizon: only used to decide how trustworthy the current
# inferred key area still is.  It does NOT directly choose the chord.
KEY_CONTEXT_WINDOW = 48

# Whole-sequence transition costs.
BASE_ROOT_CHANGE_COST = 0.035
BASE_TYPE_CHANGE_COST = 0.015

# Weak tonal priors.  These break close calls; they must never dominate
# strong CP evidence.
KEY_FAMILY_BONUS = 0.015
TONIC_BONUS = 0.025
DOMINANT_MINOR_BONUS = 0.015

# Progression coherence.
FIFTH_MOTION_BONUS = 0.012

# Melody structure.
REST_CHANGE_PENALTY = 0.010
MELODY_ONSET_CHANGE_BONUS = 0.012
RETURN_AFTER_REST_BONUS = 0.018
LONG_REST_STEPS = 8

# Boundary repair: after future evidence confirms a new chord, allow the
# boundary to move backwards by up to two analysis hops.
BOUNDARY_BACKTRACK_HOPS = 2
# BOUNDARY_BACKTRACK_MARGIN = 0.040 # DEFAULT
BOUNDARY_BACKTRACK_MARGIN = 0.0

# Do not let a practically-zero-confidence inferred key keep exerting
# transition resistance.
KEY_CONFIDENCE_FLOOR = 0.10


PC_NAMES = ["C", "C#", "D", "D#", "E", "F",
            "F#", "G", "G#", "A", "A#", "B"]

PITCH_CLASSES = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7,
    "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}

# Recognition is only supporting evidence.  The evidence itself is the
# duration/persistence-weighted pitch-class distribution.
CHORD_TEMPLATES = [
    ("maj", (0, 4, 7)),
    ("min", (0, 3, 7)),
    ("dim", (0, 3, 6)),
    ("aug", (0, 4, 8)),
    ("7", (0, 4, 7, 10)),
    ("maj7", (0, 4, 7, 11)),
    ("min7", (0, 3, 7, 10)),
    # ("half_dim7", (0, 3, 6, 10)),
    # ("dim7", (0, 3, 6, 9)),
]


# ---------------------------------------------------------------------------
# MIDI metadata
# ---------------------------------------------------------------------------

def read_key_signature_map(path):
    """
    Return absolute-tick MIDI key-signature changes.

    Result:
        [(absolute_tick, "Cm"), (absolute_tick, "Eb"), ...]
    """
    midi = mido.MidiFile(path)

    result = []

    for track in midi.tracks:
        tick = 0

        for msg in track:
            tick += msg.time

            if msg.type == "key_signature":
                result.append((int(tick), str(msg.key)))

    result.sort(key=lambda x: x[0])

    # If multiple tracks contain a key signature at the same tick,
    # retain the final one deterministically.
    out = []

    for item in result:
        if out and out[-1][0] == item[0]:
            out[-1] = item
        else:
            out.append(item)

    return out

def read_tempo_map(path):
    midi = mido.MidiFile(path)
    result = []
    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                result.append((tick, float(mido.tempo2bpm(msg.tempo))))
    result.sort()
    out = []
    for item in result:
        if out and out[-1][0] == item[0]:
            out[-1] = item
        else:
            out.append(item)
    return out


def read_time_signature_map(path):
    midi = mido.MidiFile(path)
    result = []
    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "time_signature":
                result.append((tick, int(msg.numerator), int(msg.denominator)))
    result.sort()
    out = []
    for item in result:
        if out and out[-1][0] == item[0]:
            out[-1] = item
        else:
            out.append(item)
    return out


def parse_time_signature(value):
    m = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", value)
    if not m:
        raise ValueError(f"Invalid time signature '{value}', expected 4/4.")
    n, d = int(m.group(1)), int(m.group(2))
    if n <= 0 or d <= 0:
        raise ValueError("Time-signature values must be positive.")
    return n, d


def get_midi_metadata(path, cli_bpm=None, cli_time_signature=None):
    tempo_map = read_tempo_map(path)
    ts_map = read_time_signature_map(path)

    if tempo_map:
        bpm = round(tempo_map[0][1], 2)
        bpm_source = "MIDI tempo map"
    elif cli_bpm is not None:
        bpm = round(float(cli_bpm), 2)
        bpm_source = "command line"
    else:
        bpm = DEFAULT_BPM
        bpm_source = "default"

    if ts_map:
        _, numerator, denominator = ts_map[0]
        ts_source = "MIDI time-signature map"
    elif cli_time_signature:
        numerator, denominator = parse_time_signature(cli_time_signature)
        ts_source = "command line"
    else:
        numerator, denominator = DEFAULT_NUMERATOR, DEFAULT_DENOMINATOR
        ts_source = "default"

    return {
        "bpm": bpm,
        "numerator": numerator,
        "denominator": denominator,
        "bpm_source": bpm_source,
        "time_signature_source": ts_source,
        "tempo_map": [(t, round(b, 2)) for t, b in tempo_map],
        "time_signature_map": ts_map,
    }


def read_input_length(path):
    midi = mido.MidiFile(path)
    max_tick = 0
    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
        max_tick = max(max_tick, tick)
    return int(round(max_tick / midi.ticks_per_beat * 4))


# ---------------------------------------------------------------------------
# Key / artificial prompt
# ---------------------------------------------------------------------------

def parse_key(key):
    m = re.fullmatch(
        r"([A-Ga-g](?:#|b)?)(?:\s*)(major|minor|maj|min|M|m)?",
        key.strip(),
    )
    if not m:
        raise ValueError(f"Invalid key '{key}'.")
    root = m.group(1)
    root = root[0].upper() + root[1:]
    if root not in PITCH_CLASSES:
        raise ValueError(f"Unsupported key root: {root}")
    mode = (m.group(2) or "major").lower()
    mode = "minor" if mode in ("m", "min", "minor") else "major"
    return root, mode


def make_tonic_triad(key):
    root, mode = parse_key(key)
    pc = PITCH_CLASSES[root]
    third = 3 if mode == "minor" else 4
    root_pitch = 12 * (CHORD_OCTAVE + 1) + pc
    return [root_pitch, root_pitch + third, root_pitch + 7]


def calculate_bar_ticks(resolution, numerator, denominator):
    return int(round(resolution * numerator * 4.0 / denominator))


def create_melody_chord_input(input_path, output_path, key):
    """
    Create the synthetic two-track model-conditioning MIDI.

    IMPORTANT:
    The source melody is copied in MIDI TICKS, not via pretty_midi seconds.
    This avoids PrettyMIDI's 120-BPM fallback changing the musical timing
    when the source MIDI contains no tempo event.
    """

    source_mido = mido.MidiFile(input_path)
    ticks_per_beat = source_mido.ticks_per_beat

    metadata = get_midi_metadata(input_path)

    # ---------------------------------------------------------------
    # Find exactly one source track containing note-on events.
    # ---------------------------------------------------------------

    note_track_indices = []

    for track_index, track in enumerate(source_mido.tracks):
        if any(
            msg.type == "note_on" and msg.velocity > 0
            for msg in track
        ):
            note_track_indices.append(track_index)

    if len(note_track_indices) != 1:
        raise ValueError(
            "Input MIDI must contain exactly one non-empty melody track. "
            f"Found {len(note_track_indices)}."
        )

    source_note_track = source_mido.tracks[note_track_indices[0]]

    # ---------------------------------------------------------------
    # Build a clean Type-1 MIDI.
    # Track 0 = melody
    # Track 1 = two-bar tonic prompt
    # ---------------------------------------------------------------

    result = mido.MidiFile(
        type=1,
        ticks_per_beat=ticks_per_beat,
    )

    melody_track = mido.MidiTrack()
    chord_track = mido.MidiTrack()

    result.tracks.append(melody_track)
    result.tracks.append(chord_track)

    # ---------------------------------------------------------------
    # Track 0 metadata.
    # ---------------------------------------------------------------

    melody_events = []

    # Explicit tempo at tick 0 using the resolved BPM.
    melody_events.append(
        (
            0,
            0,
            mido.MetaMessage(
                "set_tempo",
                tempo=mido.bpm2tempo(metadata["bpm"]),
                time=0,
            ),
        )
    )

    # Preserve original time/key signatures at their exact ticks.
    for track in source_mido.tracks:
        absolute_tick = 0

        for msg in track:
            absolute_tick += msg.time

            if msg.type in (
                "time_signature",
                "key_signature",
            ):
                melody_events.append(
                    (
                        absolute_tick,
                        0,
                        msg.copy(time=0),
                    )
                )

    # Force melody Program 64.
    melody_events.append(
        (
            0,
            1,
            mido.Message(
                "program_change",
                program=MELODY_PROGRAM,
                channel=0,
                time=0,
            ),
        )
    )

    # ---------------------------------------------------------------
    # Copy melody notes at EXACT original absolute ticks.
    # ---------------------------------------------------------------

    absolute_tick = 0

    for msg in source_note_track:
        absolute_tick += msg.time

        if msg.type == "note_on":
            if msg.velocity > 0:
                melody_events.append(
                    (
                        absolute_tick,
                        2,
                        mido.Message(
                            "note_on",
                            note=msg.note,
                            velocity=OUTPUT_VELOCITY,
                            channel=0,
                            time=0,
                        ),
                    )
                )
            else:
                melody_events.append(
                    (
                        absolute_tick,
                        1,
                        mido.Message(
                            "note_off",
                            note=msg.note,
                            velocity=0,
                            channel=0,
                            time=0,
                        ),
                    )
                )

        elif msg.type == "note_off":
            melody_events.append(
                (
                    absolute_tick,
                    1,
                    mido.Message(
                        "note_off",
                        note=msg.note,
                        velocity=0,
                        channel=0,
                        time=0,
                    ),
                )
            )

    melody_events.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    previous_tick = 0

    for absolute_tick, _, msg in melody_events:
        melody_track.append(
            msg.copy(
                time=absolute_tick - previous_tick
            )
        )
        previous_tick = absolute_tick

    melody_track.append(
        mido.MetaMessage(
            "end_of_track",
            time=0,
        )
    )

    # ---------------------------------------------------------------
    # Track 1 = two-bar tonic prompt.
    # ---------------------------------------------------------------

    chord_track.append(
        mido.Message(
            "program_change",
            program=CHORD_PROGRAM,
            channel=1,
            time=0,
        )
    )

    pitches = make_tonic_triad(key)

    bar_ticks = calculate_bar_ticks(
        ticks_per_beat,
        metadata["numerator"],
        metadata["denominator"],
    )

    chord_events = []

    for bar in range(2):
        start_tick = bar * bar_ticks
        end_tick = (bar + 1) * bar_ticks

        for pitch in pitches:
            chord_events.append(
                (
                    start_tick,
                    1,
                    mido.Message(
                        "note_on",
                        note=pitch,
                        velocity=OUTPUT_VELOCITY,
                        channel=1,
                        time=0,
                    ),
                )
            )

            chord_events.append(
                (
                    end_tick,
                    0,
                    mido.Message(
                        "note_off",
                        note=pitch,
                        velocity=0,
                        channel=1,
                        time=0,
                    ),
                )
            )

    chord_events.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    previous_tick = 0

    for absolute_tick, _, msg in chord_events:
        chord_track.append(
            msg.copy(
                time=absolute_tick - previous_tick
            )
        )
        previous_tick = absolute_tick

    chord_track.append(
        mido.MetaMessage(
            "end_of_track",
            time=0,
        )
    )

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True,
    )

    result.save(output_path)

    return metadata


def get_original_melody(path):
    """
    Return the source melody as absolute-tick note events.

    Timing is preserved exactly from the original MIDI.
    """

    midi = mido.MidiFile(path)

    note_tracks = []

    for track_index, track in enumerate(midi.tracks):
        if any(
            msg.type == "note_on" and msg.velocity > 0
            for msg in track
        ):
            note_tracks.append(track_index)

    if len(note_tracks) != 1:
        raise ValueError(
            "Input MIDI must contain exactly one non-empty melody track. "
            f"Found {len(note_tracks)}."
        )

    track = midi.tracks[note_tracks[0]]

    events = []
    absolute_tick = 0

    for msg in track:
        absolute_tick += msg.time

        if msg.type == "note_on":
            if msg.velocity > 0:
                events.append(
                    (
                        absolute_tick,
                        "note_on",
                        int(msg.note),
                    )
                )
            else:
                events.append(
                    (
                        absolute_tick,
                        "note_off",
                        int(msg.note),
                    )
                )

        elif msg.type == "note_off":
            events.append(
                (
                    absolute_tick,
                    "note_off",
                    int(msg.note),
                )
            )

    return events

# ---------------------------------------------------------------------------
# Model helpers
# ---------------------------------------------------------------------------

def decompress(model, byte_arr):
    x = torch.tensor(byte_arr).unsqueeze(0).cuda()
    return model.preprocess(
        x,
        pitch_shift=torch.zeros(1, dtype=torch.int8, device="cuda"),
        preprocess_args=PreprocessingParameters(""),
    )[:2]


def decode_accompaniment_notes(outputs, ratio, tempo):
    """
    Decode raw CP events only for harmonic analysis.
    Nothing here is considered a final accompaniment attack.
    """
    from cp_transformer import CPTokenizer

    tokenizer = CPTokenizer(with_velocity=False)
    if not isinstance(outputs, tuple):
        outputs = (outputs,)
    if not isinstance(ratio, tuple):
        ratio = (ratio,) * len(outputs)

    step_seconds = 60.0 / tempo / 4.0
    notes = []

    for r, output in zip(ratio, outputs):
        for step, data in enumerate(output):
            content = data.squeeze(0)
            start = step * step_seconds
            for i in range(0, len(content), 2):
                program = int(content[i].item())
                if program == tokenizer.eos_token:
                    break
                if i + 1 >= len(content):
                    break

                pd = int(content[i + 1].item()) - 128
                pitch = pd % 128
                duration = pd // 128

                if not 0 <= program < 128:
                    break
                if not 0 <= pitch < 128:
                    break
                if not 0 <= duration < len(DURATION_TEMPLATES):
                    break

                end = (
                    start
                    + DURATION_TEMPLATES[duration] * step_seconds * r
                )
                notes.append(
                    pretty_midi.Note(
                        OUTPUT_VELOCITY,
                        pitch,
                        start * r,
                        end,
                    )
                )
    return notes


# ---------------------------------------------------------------------------
# Simple sustained voicing
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Whole-sequence harmonic decoder
# ---------------------------------------------------------------------------


def chord_label(root, typ):
    return f"{PC_NAMES[root]}:{typ}"


def chord_score(weights, root, typ):
    """
    Pure LOCAL CP harmonic evidence.

    No key, melody, history or future-sequence bias is applied here.
    """
    intervals = dict(CHORD_TEMPLATES)[typ]
    chord_pcs = {(root + x) % 12 for x in intervals}

    support = sum(weights[p] for p in chord_pcs)
    outside = sum(
        weights[p]
        for p in range(12)
        if p not in chord_pcs
    )

    third_interval = 3 if ("min" in typ or "dim" in typ) else 4

    root_support = weights[root]
    third_support = weights[(root + third_interval) % 12]
    fifth_support = weights[(root + 7) % 12]

    score = (
        0.72 * support
        + 0.22 * root_support
        + 0.08 * third_support
        + 0.05 * fifth_support
        - 0.26 * outside
    )

    if len(intervals) == 4:
        seventh = (root + intervals[-1]) % 12
        score += 0.10 * weights[seventh]
    else:
        score += 0.025 * max(
            weights[(root + 9) % 12],
            weights[(root + 10) % 12],
            weights[(root + 11) % 12],
        )

    if typ in ("aug", "dim", "dim7", "half_dim7"):
        score -= 0.035

    return float(score)

def family_score(weights, root, typ):
    """
    Harmonic-family evidence used by the Viterbi decoder.

    Major family:
        maj / maj7 / dominant-7

    Minor family:
        min / min7

    The decoder chooses the underlying harmonic family and root.
    Exact seventh extensions are resolved only after harmonic
    regions have been established.
    """

    if typ == "maj":
        return max(
            chord_score(weights, root, "maj"),
            chord_score(weights, root, "maj7"),
            chord_score(weights, root, "7"),
        )

    if typ == "min":
        return max(
            chord_score(weights, root, "min"),
            chord_score(weights, root, "min7"),
        )

    return chord_score(
        weights,
        root,
        typ,
    )

def all_chord_states():
    """
    Harmonic-family decoder state space.

    Seventh extensions are intentionally NOT separate temporal states.

    The Viterbi decoder chooses among:

        maj-family
        min-family
        dim
        aug

    for each of the 12 roots.

    Exact maj / maj7 / 7 and min / min7 labels are resolved
    after harmonic regions have been established.
    """
    result = []

    for typ in (
        "maj",
        "min",
        "dim",
        "aug",
    ):
        for root in range(12):
            result.append({
                "root": root,
                "type": typ,
                "label": chord_label(root, typ),
            })

    return result

def _weighted_pitch_classes(intervals, start, end):
    weights = np.zeros(12, dtype=np.float64)
    occupancy = np.zeros(12, dtype=np.float64)

    if end <= start:
        return weights, occupancy

    for a, b, pc in intervals:
        ov = max(
            0.0,
            min(b, end) - max(a, start),
        )

        if ov <= 1e-6:
            continue

        weights[pc] += math.sqrt(max(ov, 1.0))
        occupancy[pc] += ov

    total = weights.sum()

    if total:
        weights /= total

    occupancy /= float(end - start)

    return weights, occupancy


def build_pitch_class_hops(notes, generation_length, bpm):
    """
    Two simultaneous horizons.

    chord_weights:
        Short 16-step horizon.  Used to identify the chord NOW.

    key_weights:
        Longer future-looking horizon.  Used only to determine whether the
        current inferred tonal prior remains believable.

    This separation is deliberate: the long horizon must not smear chord
    boundaries or make the decoder sluggish.
    """
    sixteenth = 60.0 / bpm / 4.0

    intervals = []

    for n in notes:
        a = n.start / sixteenth
        b = n.end / sixteenth

        if b > a:
            intervals.append(
                (a, b, int(n.pitch) % 12)
            )

    hops = []

    for start in range(
        0,
        generation_length,
        SKELETON_HOP,
    ):
        chord_end = min(
            start + CHORD_WINDOW,
            generation_length,
        )

        key_end = min(
            start + KEY_CONTEXT_WINDOW,
            generation_length,
        )

        if chord_end <= start:
            break

        chord_weights, occupancy = _weighted_pitch_classes(
            intervals,
            start,
            chord_end,
        )

        key_weights, _ = _weighted_pitch_classes(
            intervals,
            start,
            key_end,
        )

        persistent = [
            pc
            for pc in range(12)
            if occupancy[pc] >= 0.20
        ]

        strong = [
            pc
            for pc in range(12)
            if chord_weights[pc] >= 0.075
        ]

        hops.append({
            "start": int(start),

            # IMPORTANT:
            # this is only the evidence-window end.
            # It is NEVER used as a harmonic-region boundary.
            "analysis_end": int(chord_end),

            "weights": chord_weights,
            "key_weights": key_weights,
            "occupancy": occupancy,
            "persistent": persistent,
            "strong": strong,
        })

    return hops


# ---------------------------------------------------------------------------
# Key handling
# ---------------------------------------------------------------------------


MAJOR_SCALE = (0, 2, 4, 5, 7, 9, 11)
MINOR_SCALE = (0, 2, 3, 5, 7, 8, 10)


def key_scale_pcs(key):
    root_name, mode = parse_key(key)
    root = PITCH_CLASSES[root_name]

    scale = MAJOR_SCALE if mode == "major" else MINOR_SCALE

    return {
        (root + interval) % 12
        for interval in scale
    }


def diatonic_triads(key):
    """
    Return the six ordinary major/minor triads belonging to the active
    major/minor pitch collection.

    The diminished seventh-degree triad is intentionally omitted from the
    'six horsemen' preference.
    """
    root_name, mode = parse_key(key)
    tonic = PITCH_CLASSES[root_name]

    if mode == "major":
        # I ii iii IV V vi
        return {
            (tonic + 0) % 12: "maj",
            (tonic + 2) % 12: "min",
            (tonic + 4) % 12: "min",
            (tonic + 5) % 12: "maj",
            (tonic + 7) % 12: "maj",
            (tonic + 9) % 12: "min",
        }

    # Natural minor:
    # i ii° III iv v VI VII
    # Omit ii° -> leaves the requested six ordinary major/minor chords.
    return {
        (tonic + 0) % 12: "min",
        (tonic + 3) % 12: "maj",
        (tonic + 5) % 12: "min",
        (tonic + 7) % 12: "min",
        (tonic + 8) % 12: "maj",
        (tonic + 10) % 12: "maj",
    }


def tonal_confidence(key_weights, key):
    """
    How compatible does the upcoming CP evidence remain with this key?

    This controls the strength of a PROMPT-INFERRED key prior.

    It does not choose the chord.
    """
    if key_weights.sum() <= 0:
        return 0.0

    root_name, mode = parse_key(key)
    tonic = PITCH_CLASSES[root_name]

    scale_pcs = key_scale_pcs(key)

    confidence = sum(
        key_weights[pc]
        for pc in scale_pcs
    )

    # In minor, allow the raised leading tone to contribute partially.
    # Example: B natural / G7 in C minor must not falsely look like an
    # immediate modulation.
    if mode == "minor":
        raised_7 = (tonic + 11) % 12

        if raised_7 not in scale_pcs:
            confidence += 0.50 * key_weights[raised_7]

    return float(
        max(0.0, min(1.0, confidence))
    )


def key_prior_bonus(state, key):
    """
    Tiny tonal tie-breaker.

    The six ordinary diatonic triads receive a small premium.
    The tonic receives a slightly larger premium.

    Minor-key dominant major / dominant-7 is also explicitly permitted.
    """
    root_name, mode = parse_key(key)

    tonic = PITCH_CLASSES[root_name]

    root = state["root"]
    typ = state["type"]

    bonus = 0.0

    horsemen = diatonic_triads(key)

    expected_type = horsemen.get(root)

    if expected_type is not None:
        if typ == expected_type:
            bonus += KEY_FAMILY_BONUS

        elif (
            expected_type == "maj"
            and typ == "maj7"
        ):
            bonus += KEY_FAMILY_BONUS * 0.75

        elif (
            expected_type == "min"
            and typ == "min7"
        ):
            bonus += KEY_FAMILY_BONUS * 0.75

    if root == tonic:
        if (
            (mode == "major" and typ in ("maj", "maj7"))
            or
            (mode == "minor" and typ in ("min", "min7"))
        ):
            bonus += TONIC_BONUS

    # Harmonic-minor dominant:
    #
    # C minor -> G / G7
    if mode == "minor":
        dominant = (tonic + 7) % 12

        if root == dominant and typ in ("maj", "7"):
            bonus += DOMINANT_MINOR_BONUS

    return float(bonus)


def build_local_key_map(
    input_path,
    prompt_key,
    prompt_steps,
):
    """
    Convert the original MIDI key-signature map to GENERATED-LOCAL step
    coordinates.

    local step 0 corresponds to the point immediately after the artificial
    two-bar prompt.

    Return:
        {
            "has_explicit_map": bool,
            "initial_key": str,
            "changes": [(local_step, key), ...],
        }
    """
    midi = mido.MidiFile(input_path)

    raw = read_key_signature_map(input_path)

    if not raw:
        return {
            "has_explicit_map": False,
            "initial_key": prompt_key,
            "changes": [],
        }

    events = []

    for tick, key in raw:
        absolute_step = int(
            round(
                tick
                / midi.ticks_per_beat
                * 4.0
            )
        )

        local_step = absolute_step - prompt_steps

        events.append(
            (local_step, key)
        )

    events.sort()

    # Determine which map entry is already active at local step zero.
    active = None

    for step, key in events:
        if step <= 0:
            active = key
        else:
            break

    if active is None:
        # A map exists but its first event occurs later.
        # Use the supplied prompt key until then.
        active = prompt_key

    future_changes = [
        (max(0, int(step)), key)
        for step, key in events
        if step > 0
    ]

    return {
        "has_explicit_map": True,
        "initial_key": active,
        "changes": future_changes,
    }


def active_key_at_step(key_map, step):
    key = key_map["initial_key"]

    for change_step, new_key in key_map["changes"]:
        if change_step > step:
            break

        key = new_key

    return key


def explicit_key_change_crossed(
    key_map,
    previous_step,
    current_step,
):
    """
    Critical rule:

    If an explicit MIDI key-signature change lies between the previous and
    current decoder position, transition resistance is EXACTLY ZERO.
    """
    if not key_map["has_explicit_map"]:
        return False

    for step, _ in key_map["changes"]:
        if previous_step < step <= current_step:
            return True

    return False


# ---------------------------------------------------------------------------
# Melody structural context
# ---------------------------------------------------------------------------


def get_local_melody_structure(
    input_path,
    prompt_steps,
):
    """
    Read original melody note timings directly in MIDI ticks and convert them
    to local sixteenth-step coordinates.

    This is independent of PrettyMIDI seconds and therefore preserves the
    timing correction already made elsewhere in the script.
    """
    midi = mido.MidiFile(input_path)

    note_tracks = []

    for track_index, track in enumerate(midi.tracks):
        tick = 0
        notes_found = False

        for msg in track:
            tick += msg.time

            if (
                msg.type == "note_on"
                and msg.velocity > 0
                and getattr(msg, "channel", 0) != 9
            ):
                notes_found = True

        if notes_found:
            note_tracks.append(track_index)

    if len(note_tracks) != 1:
        raise ValueError(
            "Input MIDI must contain exactly one non-empty melody track. "
            f"Found {len(note_tracks)}."
        )

    track = midi.tracks[note_tracks[0]]

    active = {}
    intervals = []
    onsets = []

    tick = 0

    for msg in track:
        tick += msg.time

        if not hasattr(msg, "channel"):
            continue

        if msg.channel == 9:
            continue

        key = (msg.channel, getattr(msg, "note", -1))

        if msg.type == "note_on" and msg.velocity > 0:
            active.setdefault(key, []).append(tick)

            absolute_step = (
                tick
                / midi.ticks_per_beat
                * 4.0
            )

            onsets.append(
                absolute_step - prompt_steps
            )

        elif (
            msg.type == "note_off"
            or (
                msg.type == "note_on"
                and msg.velocity == 0
            )
        ):
            starts = active.get(key)

            if not starts:
                continue

            start_tick = starts.pop(0)

            if not starts:
                active.pop(key, None)

            a = (
                start_tick
                / midi.ticks_per_beat
                * 4.0
                - prompt_steps
            )

            b = (
                tick
                / midi.ticks_per_beat
                * 4.0
                - prompt_steps
            )

            if b > a:
                intervals.append((a, b))

    onsets = sorted(onsets)
    intervals.sort()

    return {
        "onsets": onsets,
        "intervals": intervals,
    }


def melody_boundary_features(
    melody_structure,
    boundary_step,
):
    """
    Describe how structurally plausible this position is as a chord boundary.
    """
    onsets = melody_structure["onsets"]
    intervals = melody_structure["intervals"]

    eps = SKELETON_HOP * 0.35

    onset_here = any(
        abs(o - boundary_step) <= eps
        for o in onsets
    )

    sounding = any(
        a < boundary_step < b
        for a, b in intervals
    )

    previous_onset = None
    next_onset = None

    for o in onsets:
        if o < boundary_step:
            previous_onset = o
        elif o >= boundary_step:
            next_onset = o
            break

    previous_note_end = None

    for a, b in intervals:
        if b <= boundary_step:
            if (
                previous_note_end is None
                or b > previous_note_end
            ):
                previous_note_end = b

    rest_length_before = 0.0

    if onset_here and previous_note_end is not None:
        rest_length_before = max(
            0.0,
            boundary_step - previous_note_end,
        )

    return {
        "onset_here": onset_here,
        "sounding": sounding,
        "inside_rest": not sounding and not onset_here,
        "return_after_long_rest": (
            onset_here
            and rest_length_before >= LONG_REST_STEPS
        ),
        "previous_onset": previous_onset,
        "next_onset": next_onset,
    }


def melody_transition_adjustment(features):
    """
    Soft boundary prior.

    Melody silence does NOT forbid a chord change.

    A preparatory dominant during a rest can still win through CP evidence
    and progression coherence.
    """
    score = 0.0

    if features["inside_rest"]:
        score -= REST_CHANGE_PENALTY

    if features["onset_here"]:
        score += MELODY_ONSET_CHANGE_BONUS

    if features["return_after_long_rest"]:
        score += RETURN_AFTER_REST_BONUS

    return float(score)


# ---------------------------------------------------------------------------
# Progression / transition scoring
# ---------------------------------------------------------------------------


def progression_bonus(previous, current):
    """
    Very small preference for strong root motion.

    Example:
        Bb -> Eb
        G  -> C

    Both are descending-fifth / ascending-fourth relationships.
    """
    if previous["root"] == current["root"]:
        return 0.0

    motion = (
        current["root"] - previous["root"]
    ) % 12

    if motion in (5, 7):
        return FIFTH_MOTION_BONUS

    return 0.0


def build_decoder_evidence(
    hops,
    chord_states,
    key_map,
):
    """
    Local emission score for every hop × every chord state.
    """
    emissions = np.zeros(
        (len(hops), len(chord_states)),
        dtype=np.float64,
    )

    key_confidences = []

    for i, hop in enumerate(hops):
        key = active_key_at_step(
            key_map,
            hop["start"],
        )

        if key_map["has_explicit_map"]:
            # The file explicitly tells us the tonal region.
            # The prior is still tiny, but no speculative decay is needed.
            key_confidence = 1.0
        else:
            # Prompt-derived prior:
            # future evidence is allowed to dissolve it.
            key_confidence = tonal_confidence(
                hop["key_weights"],
                key,
            )

        key_confidences.append(key_confidence)

        for j, state in enumerate(chord_states):
            local = family_score(
                hop["weights"],
                state["root"],
                state["type"],
            )

            tonal = (
                key_prior_bonus(state, key)
                * key_confidence
            )

            emissions[i, j] = local + tonal

    return emissions, key_confidences

def print_truth_landmark_emissions(
    hops,
    chord_states,
    emissions,
):
    """
    Diagnostic only.

    Compare raw family-level emission scores against known harmonic truth
    landmarks for ReconstruirePredestinati.mid.

    Local generated step 0 = bar 3 beat 1.
    Therefore:
        bar 23 -> step 320
        bar 24 -> step 336
        ...
    """

    landmarks = {
        320: "BAR 23 truth=D#:maj",
        336: "BAR 24 truth=A#:maj",
        352: "BAR 25 truth=C:min",
        368: "BAR 26 truth=G#:maj",
        384: "BAR 27 truth=D#:maj",
        400: "BAR 28 truth=A#:maj",
        416: "BAR 29 truth=C:min",
        432: "BAR 30 truth=G#:maj",
        448: "BAR 31 truth=D#:maj",
    }

    watched_labels = {
        "C:min",
        "D#:maj",
        "A#:maj",
        "G#:maj",
        "G:min",
        "F:min",
        "D:min",
    }

    state_indices = {
        state["label"]: i
        for i, state in enumerate(chord_states)
        if state["label"] in watched_labels
    }

    print()
    print("  === TRUTH LANDMARK RAW EMISSIONS ===")

    for target_step, description in landmarks.items():
        hop_index = next(
            (
                i
                for i, hop in enumerate(hops)
                if hop["start"] == target_step
            ),
            None,
        )

        if hop_index is None:
            print()
            print(
                f"  step {target_step}: {description} "
                f"[NO EXACT HOP]"
            )
            continue

        ranked = []

        for label, state_index in state_indices.items():
            ranked.append(
                (
                    float(emissions[hop_index, state_index]),
                    label,
                )
            )

        ranked.sort(reverse=True)

        print()
        print(
            f"  step {target_step}: {description}"
        )

        for rank, (score, label) in enumerate(
            ranked,
            start=1,
        ):
            print(
                f"    {rank:2d}. "
                f"{label:8s} "
                f"{score:+.6f}"
            )

def print_viterbi_vs_emission_diagnostics(
    hops,
    chord_states,
    emissions,
    raw_viterbi_states,
    refined_states,
):
    """
    Compare the independent best emission at each hop with:
      1. the whole-sequence raw Viterbi path
      2. the post-refinement path

    Diagnostic only. Does not alter decoding.
    """

    print()
    print("  === EMISSION WINNER -> RAW VITERBI -> REFINED ===")
    print(
        "  step   emission       score    raw_viterbi   refined       "
        "VIT?  REF?"
    )
    print(
        "  ----   ------------  -------   ------------  ------------  "
        "----  ----"
    )

    # Our current reference area.
    TRACE_START = 312
    TRACE_END = 452

    for i, hop in enumerate(hops):
        step = hop["start"]

        if step < TRACE_START or step > TRACE_END:
            continue

        emission_idx = int(np.argmax(emissions[i]))
        emission_state = chord_states[emission_idx]
        emission_label = emission_state["label"]
        emission_score = float(emissions[i, emission_idx])

        raw_label = raw_viterbi_states[i]["label"]
        refined_label = refined_states[i]["label"]

        viterbi_changed = raw_label != emission_label
        refinement_changed = refined_label != raw_label

        print(
            f"  {step:4d}   "
            f"{emission_label:12s}  "
            f"{emission_score:+.3f}   "
            f"{raw_label:12s}  "
            f"{refined_label:12s}  "
            f"{'YES' if viterbi_changed else '-':4s}  "
            f"{'YES' if refinement_changed else '-':4s}"
        )

def print_truth_temporal_emission_trace(
    hops,
    chord_states,
    emissions,
):
    """
    Diagnostic only.

    Trace raw family-level emissions every 4 generated steps around the
    known Predestinati harmonic-reference boundaries.

    Local generated step 0 = bar 3 beat 1.

    Reference cycle from bar 23 onward:

        bar 23 / step 320   D#:maj   (Eb)
        bar 24 / step 336   A#:maj   (Bb)
        bar 25 / step 352   C:min    (Cm)
        bar 26 / step 368   G#:maj   (Ab)
        bar 27 / step 384   D#:maj
        ...

    The trace begins half a bar before bar 23 and continues 12 steps
    beyond the bar-31 boundary, so that we can inspect the temporal
    evolution of the raw emission evidence across every truth boundary.

    This function changes absolutely nothing in decoding.
    """

    truth_cycle = [
        "D#:maj",
        "A#:maj",
        "C:min",
        "G#:maj",
    ]

    core_labels = [
        "C:min",
        "D#:maj",
        "A#:maj",
        "G#:maj",
    ]

    # We deliberately trace:
    #
    #   312, 316,
    #   320, 324, 328, 332,
    #   336, 340, ...
    #   ...
    #   448, 452, 456, 460
    #
    trace_start = 312
    trace_end = 460

    # Exact hop lookup.
    hop_index_by_step = {
        int(hop["start"]): i
        for i, hop in enumerate(hops)
    }

    # Exact decoder-state lookup.
    state_index_by_label = {
        state["label"]: i
        for i, state in enumerate(chord_states)
    }

    missing_core = [
        label
        for label in core_labels
        if label not in state_index_by_label
    ]

    if missing_core:
        print()
        print("  === TRUTH TEMPORAL RAW-EMISSION TRACE ===")
        print(
            "  WARNING: missing required harmonic states: "
            + ", ".join(missing_core)
        )
        return

    print()
    print("  === TRUTH TEMPORAL RAW-EMISSION TRACE ===")
    print(
        "  Raw family emissions only; "
        "before Viterbi / refinement / reconstruction."
    )
    print(
        "  Each hop uses the existing future-looking "
        f"{CHORD_WINDOW}-step chord window."
    )
    print()
    print(
        "  step  truth     "
        "C:min      D#:maj     A#:maj     G#:maj     "
        "winner    win_score  truth_rank  gap"
    )
    print(
        "  ----  --------  "
        "---------  ---------  ---------  ---------  "
        "--------  ---------  ----------  ---------"
    )

    for step in range(
        trace_start,
        trace_end + 1,
        SKELETON_HOP,
    ):
        hop_index = hop_index_by_step.get(step)

        if hop_index is None:
            print(
                f"  {step:4d}  "
                f"{'[NO HOP]':8s}"
            )
            continue

        # Bar 23 begins at step 320.
        #
        # Python floor division is useful here:
        #
        #   step 312 -> (-8 // 16) = -1 -> cycle[-1] = G#:maj
        #
        # so the two pre-boundary samples correctly belong to the
        # preceding Ab-reference bar.
        truth_bar_offset = (
            (step - 320) // 16
        )

        truth_label = truth_cycle[
            truth_bar_offset % len(truth_cycle)
        ]

        # Rank ALL harmonic-family states, not merely the four
        # reference families.
        ranked_indices = sorted(
            range(len(chord_states)),
            key=lambda state_index: float(
                emissions[
                    hop_index,
                    state_index,
                ]
            ),
            reverse=True,
        )

        winner_index = ranked_indices[0]
        winner_label = chord_states[
            winner_index
        ]["label"]
        winner_score = float(
            emissions[
                hop_index,
                winner_index,
            ]
        )

        truth_index = state_index_by_label[
            truth_label
        ]
        truth_score = float(
            emissions[
                hop_index,
                truth_index,
            ]
        )

        truth_rank = (
            ranked_indices.index(truth_index)
            + 1
        )

        gap = winner_score - truth_score

        core_scores = {
            label: float(
                emissions[
                    hop_index,
                    state_index_by_label[label],
                ]
            )
            for label in core_labels
        }

        boundary_marker = (
            " <BOUNDARY"
            if (
                step >= 320
                and (step - 320) % 16 == 0
            )
            else ""
        )

        print(
            f"  {step:4d}  "
            f"{truth_label:8s}  "
            f"{core_scores['C:min']:+9.3f}  "
            f"{core_scores['D#:maj']:+9.3f}  "
            f"{core_scores['A#:maj']:+9.3f}  "
            f"{core_scores['G#:maj']:+9.3f}  "
            f"{winner_label:8s}  "
            f"{winner_score:+9.3f}  "
            f"{truth_rank:10d}  "
            f"{gap:+9.3f}"
            f"{boundary_marker}"
        )

        # If the winner is outside the four reference families,
        # make that explicitly visible rather than forcing the reader
        # to infer it from the compact table.
        if winner_label not in core_labels:
            print(
                f"        outside-core winner: "
                f"{winner_label} "
                f"{winner_score:+.6f}"
            )

def print_truth_four_step_harmonic_trace(
    local_notes,
    chord_states,
    bpm,
):
    """
    Diagnostic only.

    Inspect the generated accompaniment in NON-OVERLAPPING 4-step windows:

        [320,324)
        [324,328)
        [328,332)
        ...

    This does NOT alter CHORD_WINDOW.

    It reuses:
        _weighted_pitch_classes()
        family_score()

    exactly as the production harmonic analysis does.

    No key prior.
    No Viterbi.
    No continuity.
    No boundary refinement.
    No reconstruction.

    Purpose:
        reveal the actual local harmonic content generated by the model
        at quarter-bar resolution.
    """

    TRACE_START = 312
    TRACE_END = 456
    TRACE_WINDOW = 8

    truth_cycle = [
        "D#:maj",   # Eb
        "A#:maj",   # Bb
        "C:min",    # Cm
        "G#:maj",   # Ab
    ]

    state_index_by_label = {
        state["label"]: i
        for i, state in enumerate(chord_states)
    }

    sixteenth = 60.0 / bpm / 4.0

    # --------------------------------------------------------------
    # Convert generated notes to the SAME step-domain interval form
    # used by build_pitch_class_hops().
    # --------------------------------------------------------------

    intervals = []

    detailed_intervals = []

    for note in local_notes:

        start_step = note.start / sixteenth
        end_step = note.end / sixteenth

        if end_step <= start_step:
            continue

        pc = int(note.pitch) % 12

        intervals.append(
            (
                float(start_step),
                float(end_step),
                pc,
            )
        )

        detailed_intervals.append({
            "pitch": int(note.pitch),
            "pc": pc,
            "start": float(start_step),
            "end": float(end_step),
        })

    print()
    print("=" * 112)
    print("4-STEP RAW GENERATED HARMONIC TRACE")
    print("=" * 112)

    print(
        "Each row analyzes one independent 4-step window using "
        "_weighted_pitch_classes() + family_score()."
    )

    print(
        "No CHORD_WINDOW change; no key prior; no Viterbi; "
        "no refinement."
    )

    print()

    print(
        " step-window   truth     winner      score     "
        "truth_score  truth_rank   gap      dominant pitch classes"
    )

    print(
        " -----------   --------  ----------  --------  "
        "-----------  ----------  -------  ----------------------"
    )

    # --------------------------------------------------------------
    # Main 4-step trace.
    # --------------------------------------------------------------

    for start in range(
        TRACE_START,
        TRACE_END,
        TRACE_WINDOW,
    ):

        end = start + TRACE_WINDOW

        weights, occupancy = _weighted_pitch_classes(
            intervals,
            start,
            end,
        )

        # ----------------------------------------------------------
        # Score ALL 48 family states using pure LOCAL evidence.
        # ----------------------------------------------------------

        scored = []

        for state_index, state in enumerate(chord_states):

            score = family_score(
                weights,
                state["root"],
                state["type"],
            )

            scored.append(
                (
                    float(score),
                    state_index,
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        winner_score, winner_index = scored[0]

        winner_label = chord_states[
            winner_index
        ]["label"]

        # ----------------------------------------------------------
        # Reference family.
        #
        # Only claim reference truth from step 320 onward.
        # ----------------------------------------------------------

        truth_label = None

        if start >= 320:

            bar_offset = (
                (start - 320) // 16
            )

            truth_label = truth_cycle[
                bar_offset % len(truth_cycle)
            ]

        if truth_label is not None:

            truth_index = state_index_by_label[
                truth_label
            ]

            truth_score = family_score(
                weights,
                chord_states[truth_index]["root"],
                chord_states[truth_index]["type"],
            )

            ranked_indices = [
                index
                for _, index in scored
            ]

            truth_rank = (
                ranked_indices.index(truth_index)
                + 1
            )

            gap = (
                winner_score
                - truth_score
            )

        else:

            truth_score = None
            truth_rank = None
            gap = None

        # ----------------------------------------------------------
        # Summarize strongest pitch classes.
        # ----------------------------------------------------------

        pc_order = sorted(
            range(12),
            key=lambda pc: float(weights[pc]),
            reverse=True,
        )

        pc_summary_parts = []

        for pc in pc_order:

            if weights[pc] <= 0.0:
                continue

            pc_summary_parts.append(
                f"{PC_NAMES[pc]}:{weights[pc]:.3f}"
            )

            if len(pc_summary_parts) >= 5:
                break

        pc_summary = (
            " ".join(pc_summary_parts)
            if pc_summary_parts
            else "[none]"
        )

        boundary_marker = (
            " <BAR"
            if (
                start >= 320
                and (start - 320) % 16 == 0
            )
            else ""
        )

        if truth_label is not None:

            print(
                f" {start:3d}-{end:<3d}      "
                f"{truth_label:8s}  "
                f"{winner_label:10s}  "
                f"{winner_score:+8.3f}  "
                f"{truth_score:+11.3f}  "
                f"{truth_rank:10d}  "
                f"{gap:+7.3f}  "
                f"{pc_summary}"
                f"{boundary_marker}"
            )

        else:

            print(
                f" {start:3d}-{end:<3d}      "
                f"{'-':8s}  "
                f"{winner_label:10s}  "
                f"{winner_score:+8.3f}  "
                f"{'-':>11s}  "
                f"{'-':>10s}  "
                f"{'-':>7s}  "
                f"{pc_summary}"
            )

    # --------------------------------------------------------------
    # Focused verbose blocks.
    #
    # These are the most useful windows for our current test.
    # --------------------------------------------------------------

    # focus_steps = [
    #     320,
    #     324,
    #     328,
    #     332,
    #     336,
    #     340,
    #     344,
    #     348,
    #     352,
    #     356,
    #     360,
    #     364,
    #     368,
    #     372,
    #     376,
    #     380,
    #     384,
    #     388,
    #     392,
    #     396,
    #     400,
    #     404,
    #     408,
    #     412,
    #     416,
    #     420,
    #     424,
    #     428,
    #     432,
    #     436,
    #     440,
    #     444,
    #     448,
    #     452,
    # ]
    focus_steps = [
        320,
        328,
        336,
        344,
        352,
        360,
        368,
        376,
        384,
        392,
        400,
        408,
        416,
        424,
        432,
        440,
        448,
    ]
    print()
    print("=" * 112)
    print("4-STEP FOCUSED RAW-NOTE DETAILS")
    print("=" * 112)

    for start in focus_steps:

        end = start + TRACE_WINDOW

        weights, occupancy = _weighted_pitch_classes(
            intervals,
            start,
            end,
        )

        overlapping = []

        for item in detailed_intervals:

            overlap = max(
                0.0,
                min(item["end"], end)
                - max(item["start"], start),
            )
# #TEMP INSERT
#             if abs(overlap) < 1e-6:
#                 print(
#                     "    DEBUG TINY OVERLAP:"
#                     f" window=[{start:.17f}, {end:.17f})"
#                     f" note=[{item['start']:.17f}, {item['end']:.17f})"
#                     f" raw_diff="
#                     f"{min(item['end'], end) - max(item['start'], start):.17g}"
#                     f" overlap={overlap:.17g}"
#                 )
# #END TEMP INSERT
            if overlap <= 0.0:
                continue

            overlapping.append({
                **item,
                "overlap": overlap,
            })

        overlapping.sort(
            key=lambda item: (
                item["start"],
                item["pitch"],
                item["end"],
            )
        )

        scored = []

        for state_index, state in enumerate(chord_states):

            score = family_score(
                weights,
                state["root"],
                state["type"],
            )

            scored.append(
                (
                    float(score),
                    state_index,
                )
            )

        scored.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        print()
        print("-" * 112)

        if start >= 320:

            bar_offset = (
                (start - 320) // 16
            )

            truth_label = truth_cycle[
                bar_offset % len(truth_cycle)
            ]

        else:
            truth_label = "-"

        print(
            f"WINDOW [{start}, {end})"
            f"   truth={truth_label}"
        )

        print()
        print("  TOP FAMILY SCORES")

        for rank, (
            score,
            state_index,
        ) in enumerate(
            scored[:8],
            start=1,
        ):

            label = chord_states[
                state_index
            ]["label"]

            marker = ""

            if label == truth_label:
                marker = "  <REFERENCE>"

            print(
                f"    {rank:2d}. "
                f"{label:9s} "
                f"{score:+.6f}"
                f"{marker}"
            )

        print()
        print("  PITCH-CLASS EVIDENCE")

        for pc in range(12):

            if (
                weights[pc] <= 0.0
                and occupancy[pc] <= 0.0
            ):
                continue

            print(
                f"    {PC_NAMES[pc]:3s} "
                f"weight={weights[pc]:.6f} "
                f"occupancy={occupancy[pc]:.6f}"
            )

        print()
        print("  GENERATED NOTES")

        if not overlapping:

            print("    [none]")

        else:

            for item in overlapping:

                print(
                    f"    "
                    f"{_diagnostic_pitch_name(item['pitch']):6s} "
                    f"{item['start']:8.3f}"
                    f" -> "
                    f"{item['end']:8.3f}"
                    f"   overlap={item['overlap']:.3f}"
                )

def _diagnostic_pitch_name(pitch):
    """
    Human-readable MIDI pitch name.
    MIDI 60 = C4.
    """
    octave = pitch // 12 - 1
    return f"{PC_NAMES[pitch % 12]}{octave}"


def print_truth_raw_accompaniment_trace(
    local_notes,
    hops,
    chord_states,
    emissions,
    bpm,
):
    """
    Diagnostic only.

    Inspect the ACTUAL generated CP accompaniment content feeding selected
    16-step harmonic-analysis windows.

    This sits upstream of Viterbi and boundary refinement.

    For each selected hop it prints:

        - truth/reference family
        - exact 16-step analysis window
        - every generated note overlapping that window
        - note start/end in generated-local sixteenth-note steps
        - overlap with the window
        - the sqrt(overlap) contribution used by
          _weighted_pitch_classes()
        - final normalized pitch-class weights
        - pitch-class occupancies
        - raw LOCAL family_score() values
        - final build_decoder_evidence() emission values
        - winner / truth comparison

    This lets us separate:

        MODEL CONTENT
            generated notes really imply Gm / Eb / Cm / etc.

    from:

        ANALYSIS / SCORER
            generated notes imply something else, but the pitch-class
            weighting/family scorer misinterprets them.

    Local generated step 0 = bar 3 beat 1.
    """

    # ------------------------------------------------------------------
    # Focused diagnostic locations.
    #
    # These include the most informative ambiguous/wrong landmarks from
    # P1/P2 plus surrounding points.
    # ------------------------------------------------------------------

    diagnostic_steps = [
        320,
        324,
        328,
        332,

        336,
        340,
        344,
        348,

        352,
        356,
        360,
        364,

        368,
        372,
        376,
        380,

        384,
        388,
        392,
        396,

        400,
        404,
        408,
        412,

        416,
        420,
        424,
        428,

        432,
        436,
        440,
        444,

        448,
        452,
    ]

    truth_cycle = [
        "D#:maj",
        "A#:maj",
        "C:min",
        "G#:maj",
    ]

    core_labels = [
        "C:min",
        "D#:maj",
        "A#:maj",
        "G#:maj",
    ]

    state_index_by_label = {
        state["label"]: i
        for i, state in enumerate(chord_states)
    }

    hop_index_by_step = {
        int(hop["start"]): i
        for i, hop in enumerate(hops)
    }

    sixteenth = 60.0 / bpm / 4.0

    # Convert the decoded CP notes into EXACTLY the same step coordinates
    # used by build_pitch_class_hops().
    note_intervals = []

    for note in local_notes:
        start_step = note.start / sixteenth
        end_step = note.end / sixteenth

        if end_step <= start_step:
            continue

        note_intervals.append({
            "pitch": int(note.pitch),
            "pc": int(note.pitch) % 12,
            "start": float(start_step),
            "end": float(end_step),
        })

    print()
    print("=" * 110)
    print("RAW GENERATED ACCOMPANIMENT / PITCH-CLASS TRACE")
    print("=" * 110)
    print(
        "Diagnostic only: decoded CP notes -> 16-step weighting -> "
        "family_score -> final emission."
    )
    print(
        "No Viterbi / refinement / reconstruction information is used here."
    )

    for step in diagnostic_steps:

        hop_index = hop_index_by_step.get(step)

        if hop_index is None:
            print()
            print(f"STEP {step}: [NO EXACT HOP]")
            continue

        hop = hops[hop_index]

        window_start = float(hop["start"])
        window_end = float(hop["analysis_end"])

        truth_bar_offset = (
            (step - 320) // 16
        )

        truth_label = truth_cycle[
            truth_bar_offset % len(truth_cycle)
        ]

        # --------------------------------------------------------------
        # Rank final emissions exactly as the decoder sees them.
        # --------------------------------------------------------------

        ranked_indices = sorted(
            range(len(chord_states)),
            key=lambda idx: float(
                emissions[hop_index, idx]
            ),
            reverse=True,
        )

        winner_index = ranked_indices[0]
        winner_state = chord_states[winner_index]
        winner_label = winner_state["label"]

        truth_index = state_index_by_label[truth_label]

        winner_emission = float(
            emissions[hop_index, winner_index]
        )

        truth_emission = float(
            emissions[hop_index, truth_index]
        )

        truth_rank = (
            ranked_indices.index(truth_index)
            + 1
        )

        # --------------------------------------------------------------
        # Extract all actual CP notes that overlap this analysis window.
        # --------------------------------------------------------------

        overlapping = []

        for item in note_intervals:

            overlap = max(
                0.0,
                min(item["end"], window_end)
                - max(item["start"], window_start),
            )

            if overlap <= 0.0:
                continue

            contribution = math.sqrt(
                max(overlap, 1.0)
            )

            overlapping.append({
                **item,
                "overlap": overlap,
                "contribution": contribution,
            })

        overlapping.sort(
            key=lambda x: (
                x["start"],
                x["pitch"],
                x["end"],
            )
        )

        print()
        print("-" * 110)

        boundary_marker = (
            "  <REFERENCE BOUNDARY>"
            if (
                step >= 320
                and (step - 320) % 16 == 0
            )
            else ""
        )

        print(
            f"STEP {step}"
            f"   truth={truth_label}"
            f"   window=[{window_start:.1f}, {window_end:.1f})"
            f"{boundary_marker}"
        )

        print(
            f"  emission winner: "
            f"{winner_label} {winner_emission:+.6f}"
        )

        print(
            f"  reference:       "
            f"{truth_label} {truth_emission:+.6f}"
            f"   rank={truth_rank}"
            f"   gap={winner_emission - truth_emission:+.6f}"
        )

        # --------------------------------------------------------------
        # Actual generated notes.
        # --------------------------------------------------------------

        print()
        print("  GENERATED NOTES OVERLAPPING WINDOW")

        if not overlapping:
            print("    [none]")

        else:
            print(
                "    pitch   pc    note_start   note_end   "
                "overlap   sqrt(overlap)"
            )

            for item in overlapping:

                print(
                    f"    "
                    f"{_diagnostic_pitch_name(item['pitch']):6s}  "
                    f"{PC_NAMES[item['pc']]:3s}  "
                    f"{item['start']:10.3f}  "
                    f"{item['end']:8.3f}  "
                    f"{item['overlap']:7.3f}  "
                    f"{item['contribution']:13.6f}"
                )

        # --------------------------------------------------------------
        # Pitch-class evidence actually stored in the hop.
        # --------------------------------------------------------------

        print()
        print("  PITCH-CLASS EVIDENCE")

        print(
            "    pc       weight      occupancy     "
            "persistent   strong"
        )

        weights = hop["weights"]
        occupancy = hop["occupancy"]

        for pc in range(12):

            # Print PCs that have any evidence, plus anything explicitly
            # classified as persistent/strong.
            if (
                weights[pc] <= 0.0
                and occupancy[pc] <= 0.0
                and pc not in hop["persistent"]
                and pc not in hop["strong"]
            ):
                continue

            print(
                f"    {PC_NAMES[pc]:3s}   "
                f"{weights[pc]:10.6f}   "
                f"{occupancy[pc]:10.6f}     "
                f"{'YES' if pc in hop['persistent'] else 'no ':10s}   "
                f"{'YES' if pc in hop['strong'] else 'no'}"
            )

        # --------------------------------------------------------------
        # Recompute PURE LOCAL family scores.
        #
        # emissions[] contains:
        #
        #     local family score + tiny tonal prior
        #
        # Printing both tells us whether any discrepancy is coming from
        # the pitch-class scorer itself or merely from the tonal bonus.
        # --------------------------------------------------------------

        print()
        print("  FAMILY SCORE BREAKDOWN")
        print(
            "    family       local_score    final_emission    "
            "tonal_delta"
        )

        labels_to_show = list(core_labels)

        if winner_label not in labels_to_show:
            labels_to_show.append(winner_label)

        # Also show top-five emission families.  This is useful when the
        # interesting alternative is neither truth nor the winner.
        for idx in ranked_indices[:5]:
            label = chord_states[idx]["label"]

            if label not in labels_to_show:
                labels_to_show.append(label)

        for label in labels_to_show:

            state_index = state_index_by_label[label]
            state = chord_states[state_index]

            local_score = family_score(
                weights,
                state["root"],
                state["type"],
            )

            final_emission = float(
                emissions[
                    hop_index,
                    state_index,
                ]
            )

            tonal_delta = (
                final_emission
                - local_score
            )

            marker = ""

            if label == winner_label:
                marker += "  <WINNER>"

            if label == truth_label:
                marker += "  <REFERENCE>"

            print(
                f"    {label:9s}   "
                f"{local_score:+11.6f}   "
                f"{final_emission:+14.6f}   "
                f"{tonal_delta:+11.6f}"
                f"{marker}"
            )

def decode_harmonic_sequence(
    hops,
    chord_states,
    emissions,
    key_confidences,
    key_map,
    melody_structure,
):
    """
    Whole-song Viterbi decoder.

    Unlike the old causal hysteresis, every state choice is made as part of
    the best complete path through the entire generated proposal.
    """
    n_hops = len(hops)
    n_states = len(chord_states)

    if n_hops == 0:
        return []

    dp = np.full(
        (n_hops, n_states),
        -np.inf,
        dtype=np.float64,
    )

    back = np.full(
        (n_hops, n_states),
        -1,
        dtype=np.int32,
    )

    dp[0, :] = emissions[0, :]

    for i in range(1, n_hops):
        previous_step = hops[i - 1]["start"]
        current_step = hops[i]["start"]

        explicit_release = explicit_key_change_crossed(
            key_map,
            previous_step,
            current_step,
        )

        melody_features = melody_boundary_features(
            melody_structure,
            current_step,
        )

        melody_adjustment = melody_transition_adjustment(
            melody_features
        )

        # If the key is merely prompt-inferred, sustained future evidence
        # incompatible with the current tonal region progressively removes
        # resistance to leaving it.
        stability = max(
            KEY_CONFIDENCE_FLOOR,
            key_confidences[i],
        )

        for cur_idx, cur in enumerate(chord_states):
            best_value = -np.inf
            best_prev = -1

            for prev_idx, prev in enumerate(chord_states):
                value = dp[i - 1, prev_idx]

                same_state = (
                    prev["root"] == cur["root"]
                    and prev["type"] == cur["type"]
                )

                if not same_state:
                    if explicit_release:
                        # USER-SPECIFIED HARD RULE:
                        # at an explicit key-map boundary, old transition
                        # resistance is exactly zero.
                        root_cost = 0.0
                        type_cost = 0.0
                    else:
                        root_cost = (
                            BASE_ROOT_CHANGE_COST
                            if prev["root"] != cur["root"]
                            else 0.0
                        )

                        type_cost = (
                            BASE_TYPE_CHANGE_COST
                            if prev["type"] != cur["type"]
                            else 0.0
                        )

                        # Prompt-derived tonal certainty controls inertia.
                        #
                        # Explicit key maps do not need speculative modulation
                        # detection between declared map changes, but their
                        # small chord-change costs remain ordinary musical
                        # smoothing.
                        if not key_map["has_explicit_map"]:
                            root_cost *= stability
                            type_cost *= stability

                    value -= root_cost
                    value -= type_cost

                    value += progression_bonus(
                        prev,
                        cur,
                    )

                    value += melody_adjustment

                value += emissions[i, cur_idx]

                if value > best_value:
                    best_value = value
                    best_prev = prev_idx

            dp[i, cur_idx] = best_value
            back[i, cur_idx] = best_prev

    final_idx = int(np.argmax(dp[-1]))

    path = [final_idx]

    for i in range(n_hops - 1, 0, -1):
        final_idx = int(
            back[i, final_idx]
        )

        path.append(final_idx)

    path.reverse()

    states = []

    for hop_index, state_index in enumerate(path):
        state = dict(
            chord_states[state_index]
        )

        state["score"] = float(
            family_score(
                hops[hop_index]["weights"],
                state["root"],
                state["type"],
            )
        )

        states.append(state)

    return states


# ---------------------------------------------------------------------------
# Boundary repair
# ---------------------------------------------------------------------------


def refine_boundaries(
    hops,
    states,
    chord_states,
    emissions,
    melody_structure,
    key_map
):
    """
    Fix the 'slow steering wheel' problem.

    Once the whole-sequence decoder KNOWS that a chord change is real, inspect
    up to two earlier hops.  If the new chord was already competitive there,
    move the change backwards.

    This uses future confirmation without making every local decision twitchy.
    """
    if not states:
        return states

    state_to_index = {
        (s["root"], s["type"]): i
        for i, s in enumerate(chord_states)
    }

    refined = [
        dict(s)
        for s in states
    ]

    i = 1

    while i < len(refined):
        old = refined[i - 1]
        new = refined[i]

        changed = (
            old["root"] != new["root"]
            or old["type"] != new["type"]
        )

        if not changed:
            i += 1
            continue

        old_idx = state_to_index[
            (old["root"], old["type"])
        ]

        new_idx = state_to_index[
            (new["root"], new["type"])
        ]

        earliest = max(
            1,
            i - BOUNDARY_BACKTRACK_HOPS,
        )

        # Never backtrack a harmonic boundary across an explicit
        # MIDI key-signature change.  The key map is authoritative
        # structural information.
        if key_map["has_explicit_map"]:
            current_step = hops[i]["start"]

            for change_step, _ in key_map["changes"]:
                if (
                    change_step <= current_step
                    and change_step > hops[earliest]["start"]
                ):
                    while (
                        earliest < i
                        and hops[earliest]["start"] < change_step
                    ):
                        earliest += 1

        chosen = i

        for j in range(earliest, i):
            # Never cross an already-existing previous harmonic boundary.
            if j > 0:
                before = refined[j - 1]

                if (
                    before["root"] != old["root"]
                    or before["type"] != old["type"]
                ):
                    continue

            old_score = emissions[j, old_idx]
            new_score = emissions[j, new_idx]

            competitive = (
                new_score
                >= old_score - BOUNDARY_BACKTRACK_MARGIN
            )

            if not competitive:
                continue

            features = melody_boundary_features(
                melody_structure,
                hops[j]["start"],
            )

            # Prefer a melody onset if one is available in the admissible
            # backtracking range.
            if features["onset_here"]:
                chosen = j
                break

            if chosen == i:
                chosen = j

        if chosen < i:
            for j in range(chosen, i):
                refined[j] = dict(new)

                refined[j]["score"] = float(
                    family_score(
                        hops[j]["weights"],
                        new["root"],
                        new["type"],
                    )
                )

        i += 1

    return refined

def print_boundary_diagnostics(
    hops,
    states,
    chord_states,
    emissions,
    melody_structure,
):
    """
    Diagnostic only.

    For every harmonic-family transition in the final refined state path,
    print the surrounding evidence at:

        -2 hops
        -1 hop
         boundary
        +1 hop
        +2 hops

    For each surrounding hop we show the emission score for:

        OLD = family before the boundary
        NEW = family after the boundary

    This tells us whether the new harmony was already competitive before
    the selected boundary, or whether the underlying evidence itself
    arrived late.

    This function changes absolutely nothing in decoding.
    """

    if not states:
        return

    print("\n  === HARMONIC BOUNDARY DIAGNOSTICS ===")

    boundary_count = 0

    for i in range(1, len(states)):
        old_state = states[i - 1]
        new_state = states[i]

        old_identity = (
            old_state["root"],
            old_state["type"],
        )
        new_identity = (
            new_state["root"],
            new_state["type"],
        )

        if old_identity == new_identity:
            continue

        boundary_count += 1

        boundary_step = hops[i]["start"]

        features = melody_boundary_features(
            melody_structure,
            boundary_step,
        )

        print()
        print(
            f"  BOUNDARY {boundary_count:02d} "
            f"@ step {boundary_step}: "
            f"{old_state['label']} -> {new_state['label']}"
        )

        print(
            "    melody: "
            f"onset_here={features['onset_here']}  "
            f"sounding={features['sounding']}  "
            f"inside_rest={features['inside_rest']}  "
            f"return_after_long_rest="
            f"{features['return_after_long_rest']}"
        )

        old_idx = None
        new_idx = None

        for s_idx, candidate in enumerate(chord_states):
            identity = (
                candidate["root"],
                candidate["type"],
            )

            if identity == old_identity:
                old_idx = s_idx

            if identity == new_identity:
                new_idx = s_idx

        if old_idx is None or new_idx is None:
            print(
                "    WARNING: could not locate decoder states "
                "for diagnostic."
            )
            continue

        print(
            "    relative   step     old       new       "
            "new-old    melody"
        )

        for offset in range(-2, 3):
            j = i + offset

            if j < 0 or j >= len(hops):
                continue

            step = hops[j]["start"]

            old_score = float(
                emissions[j, old_idx]
            )
            new_score = float(
                emissions[j, new_idx]
            )

            delta = new_score - old_score

            local_features = melody_boundary_features(
                melody_structure,
                step,
            )

            melody_flags = []

            if local_features["onset_here"]:
                melody_flags.append("ONSET")

            if local_features["inside_rest"]:
                melody_flags.append("REST")

            if local_features["return_after_long_rest"]:
                melody_flags.append("RETURN")

            if local_features["sounding"]:
                melody_flags.append("SOUND")

            flag_text = (
                ",".join(melody_flags)
                if melody_flags
                else "-"
            )

            marker = (
                " <BOUNDARY"
                if offset == 0
                else ""
            )

            print(
                f"    {offset:+3d}      "
                f"{step:4d}   "
                f"{old_score:8.3f}  "
                f"{new_score:8.3f}  "
                f"{delta:+8.3f}   "
                f"{flag_text}"
                f"{marker}"
            )

    if boundary_count == 0:
        print("    No harmonic-family transitions.")

def print_refinement_diagnostics(
    hops,
    raw_states,
    refined_states,
):
    """
    Show exactly which harmonic boundaries were moved by
    refine_boundaries(), and by how many sixteenth-note steps.
    """

    def get_boundaries(states):
        result = []

        for i in range(1, len(states)):
            old = states[i - 1]
            new = states[i]

            if (
                old["root"] != new["root"]
                or old["type"] != new["type"]
            ):
                result.append({
                    "index": i,
                    "step": hops[i]["start"],
                    "old": (
                        old["root"],
                        old["type"],
                        old["label"],
                    ),
                    "new": (
                        new["root"],
                        new["type"],
                        new["label"],
                    ),
                })

        return result

    raw = get_boundaries(raw_states)
    refined = get_boundaries(refined_states)

    print("\n  === BOUNDARY REFINEMENT DIAGNOSTICS ===")

    if raw == refined:
        print("    No boundaries moved.")
        return

    raw_by_transition = {}

    for boundary in raw:
        key = (
            boundary["old"][0],
            boundary["old"][1],
            boundary["new"][0],
            boundary["new"][1],
        )

        raw_by_transition.setdefault(
            key,
            [],
        ).append(boundary)

    used = set()

    for refined_boundary in refined:
        key = (
            refined_boundary["old"][0],
            refined_boundary["old"][1],
            refined_boundary["new"][0],
            refined_boundary["new"][1],
        )

        candidates = raw_by_transition.get(
            key,
            [],
        )

        best = None
        best_distance = None

        for candidate in candidates:
            candidate_id = id(candidate)

            if candidate_id in used:
                continue

            distance = abs(
                candidate["step"]
                - refined_boundary["step"]
            )

            if (
                best is None
                or distance < best_distance
            ):
                best = candidate
                best_distance = distance

        if best is None:
            print(
                f"    NEW/CHANGED transition at "
                f"{refined_boundary['step']:4d}: "
                f"{refined_boundary['old'][2]} -> "
                f"{refined_boundary['new'][2]}"
            )
            continue

        used.add(id(best))

        delta = (
            refined_boundary["step"]
            - best["step"]
        )

        if delta == 0:
            status = "unchanged"
        elif delta < 0:
            status = f"moved EARLIER by {-delta} steps"
        else:
            status = f"moved LATER by {delta} steps"

        print(
            f"    "
            f"{best['old'][2]} -> "
            f"{best['new'][2]}: "
            f"raw={best['step']:4d}  "
            f"refined={refined_boundary['step']:4d}  "
            f"{status}"
        )

# ---------------------------------------------------------------------------
# Contiguous regions
# ---------------------------------------------------------------------------


def merge_regions(
    hops,
    states,
    generation_length,
):
    """
    Convert the per-hop decoded path into NON-OVERLAPPING harmonic regions.

    Region boundaries come from hop START positions.

    The analysis-window end is never used as a chord boundary.
    """
    if not hops:
        return []

    regions = []

    current = {
        "start": int(hops[0]["start"]),
        "end": None,
        "state": dict(states[0]),
        "ids": [0],
    }

    for i in range(1, len(hops)):
        same = (
            states[i]["root"]
            == current["state"]["root"]
            and
            states[i]["type"]
            == current["state"]["type"]
        )

        if same:
            current["ids"].append(i)
            continue

        boundary = int(
            hops[i]["start"]
        )

        current["end"] = boundary
        regions.append(current)

        current = {
            "start": boundary,
            "end": None,
            "state": dict(states[i]),
            "ids": [i],
        }

    current["end"] = int(generation_length)
    regions.append(current)

    # Defensive cleanup only.
    cleaned = []

    for r in regions:
        r["start"] = max(
            0,
            min(
                generation_length,
                int(r["start"]),
            ),
        )

        r["end"] = max(
            r["start"],
            min(
                generation_length,
                int(r["end"]),
            ),
        )

        if r["end"] <= r["start"]:
            continue

        if cleaned:
            # Exact shared boundary:
            r["start"] = cleaned[-1]["end"]

        cleaned.append(r)

    if cleaned:
        cleaned[0]["start"] = 0
        cleaned[-1]["end"] = generation_length

        for i in range(1, len(cleaned)):
            cleaned[i]["start"] = cleaned[i - 1]["end"]

    return cleaned

def resolve_region_extensions(
    regions,
    hops,
):
    """
    Resolve the concrete chord extension only AFTER harmonic
    regions have been established.

    This prevents:

        Cm <-> Cm7
        Eb <-> Ebmaj7
        G <-> G7

    from creating harmonic boundaries.

    Evidence is accumulated across all analysis hops belonging
    to the complete region.
    """

    resolved = []

    for region in regions:
        r = copy.deepcopy(region)

        root = r["state"]["root"]
        family = r["state"]["type"]
        ids = r["ids"]

        if family == "maj":
            candidates = (
                "maj",
                "maj7",
                "7",
            )

        elif family == "min":
            candidates = (
                "min",
                "min7",
            )

        else:
            # dim and aug already have a concrete identity.
            concrete = family

            scores = [
                chord_score(
                    hops[i]["weights"],
                    root,
                    concrete,
                )
                for i in ids
            ]

            r["state"]["type"] = concrete
            r["state"]["label"] = chord_label(
                root,
                concrete,
            )
            r["state"]["score"] = float(
                np.mean(scores)
                if scores
                else 0.0
            )

            resolved.append(r)
            continue

        best_type = None
        best_score = -np.inf

        for candidate in candidates:
            scores = [
                chord_score(
                    hops[i]["weights"],
                    root,
                    candidate,
                )
                for i in ids
            ]

            score = float(
                np.mean(scores)
                if scores
                else -np.inf
            )

            if score > best_score:
                best_score = score
                best_type = candidate

        r["state"]["type"] = best_type
        r["state"]["label"] = chord_label(
            root,
            best_type,
        )
        r["state"]["score"] = best_score

        resolved.append(r)

    return resolved

def chord_intervals(typ):
    return dict(CHORD_TEMPLATES)[typ]


def make_voicing(root, typ):
    """
    Produce a compact voicing.  Exact CP octaves are intentionally discarded.
    """
    result = []

    for interval in chord_intervals(typ):
        pc = (root + interval) % 12

        # Start around octave 3, then keep every chord tone inside C2-G4.
        pitch = 48 + ((pc - 0) % 12)
        while pitch < LOWEST_OUTPUT_PITCH:
            pitch += 12
        while pitch > HIGHEST_OUTPUT_PITCH:
            pitch -= 12

        while result and pitch <= result[-1]:
            pitch += 12

        if pitch <= HIGHEST_OUTPUT_PITCH:
            result.append(pitch)

    return sorted(set(result))[:MAX_VOICING_NOTES]


def reconstruct_accompaniment(
    regions,
    generation_length,
    bpm,
    prompt_steps,
):
    """
    One completely independent sustained voicing per harmonic region.

    ACE-Step requirement:
    every chord boundary terminates ALL outgoing notes and retriggers ALL
    incoming notes, including pitches shared by adjacent chords.
    """
    sixteenth = 60.0 / bpm / 4.0

    notes = []

    for r in regions:
        st = (
            prompt_steps + r["start"]
        ) * sixteenth

        en = min(
            (
                prompt_steps + r["end"]
            ) * sixteenth,
            (
                prompt_steps + generation_length
            ) * sixteenth,
        )

        if en <= st:
            continue

        pitches = make_voicing(
            r["state"]["root"],
            r["state"]["type"],
        )

        for pitch in pitches:
            notes.append(
                pretty_midi.Note(
                    OUTPUT_VELOCITY,
                    pitch,
                    st,
                    en,
                )
            )

    return sorted(
        notes,
        key=lambda n: (
            n.start,
            n.pitch,
            n.end,
        ),
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def write_final_midi(
    input_path,
    output_path,
    melody,
    accompaniment,
    tempo,
):
    """
    Write final Type-1 MIDI.

    Track 0:
        original melody at exact original ticks
        Program 64
        tempo / time-signature / key-signature metadata

    Track 1:
        reconstructed harmonic skeleton
        Program 0

    Melody timing never passes through seconds.
    """

    source = mido.MidiFile(input_path)
    ticks_per_beat = source.ticks_per_beat

    result = mido.MidiFile(
        type=1,
        ticks_per_beat=ticks_per_beat,
    )

    melody_track = mido.MidiTrack()
    accompaniment_track = mido.MidiTrack()

    result.tracks.append(melody_track)
    result.tracks.append(accompaniment_track)

    # ===============================================================
    # TRACK 0
    # ===============================================================

    events = []

    # Always write the resolved tempo.
    events.append(
        (
            0,
            0,
            mido.MetaMessage(
                "set_tempo",
                tempo=mido.bpm2tempo(float(tempo)),
                time=0,
            ),
        )
    )

    # Preserve time/key signature maps from original MIDI.
    for track in source.tracks:
        absolute_tick = 0

        for msg in track:
            absolute_tick += msg.time

            if msg.type in (
                "time_signature",
                "key_signature",
            ):
                events.append(
                    (
                        absolute_tick,
                        0,
                        msg.copy(time=0),
                    )
                )

    # Force Program 64.
    events.append(
        (
            0,
            1,
            mido.Message(
                "program_change",
                program=MELODY_PROGRAM,
                channel=0,
                time=0,
            ),
        )
    )

    # Original melody events already contain exact source ticks.
    for absolute_tick, event_type, pitch in melody:

        if event_type == "note_on":
            msg = mido.Message(
                "note_on",
                note=pitch,
                velocity=OUTPUT_VELOCITY,
                channel=0,
                time=0,
            )
            order = 2

        else:
            msg = mido.Message(
                "note_off",
                note=pitch,
                velocity=0,
                channel=0,
                time=0,
            )
            order = 1

        events.append(
            (
                absolute_tick,
                order,
                msg,
            )
        )

    events.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    previous_tick = 0

    for absolute_tick, _, msg in events:
        melody_track.append(
            msg.copy(
                time=absolute_tick - previous_tick
            )
        )
        previous_tick = absolute_tick

    melody_track.append(
        mido.MetaMessage(
            "end_of_track",
            time=0,
        )
    )

    # ===============================================================
    # TRACK 1
    # ===============================================================

    accompaniment_track.append(
        mido.Message(
            "program_change",
            program=CHORD_PROGRAM,
            channel=1,
            time=0,
        )
    )

    accompaniment_events = []

    seconds_per_quarter = 60.0 / float(tempo)

    for note in accompaniment:

        start_tick = int(
            round(
                float(note.start)
                / seconds_per_quarter
                * ticks_per_beat
            )
        )

        end_tick = int(
            round(
                float(note.end)
                / seconds_per_quarter
                * ticks_per_beat
            )
        )

        if end_tick <= start_tick:
            end_tick = start_tick + 1

        accompaniment_events.append(
            (
                start_tick,
                1,
                mido.Message(
                    "note_on",
                    note=int(note.pitch),
                    velocity=OUTPUT_VELOCITY,
                    channel=1,
                    time=0,
                ),
            )
        )

        accompaniment_events.append(
            (
                end_tick,
                0,
                mido.Message(
                    "note_off",
                    note=int(note.pitch),
                    velocity=0,
                    channel=1,
                    time=0,
                ),
            )
        )

    accompaniment_events.sort(
        key=lambda x: (
            x[0],
            x[1],
        )
    )

    previous_tick = 0

    for absolute_tick, _, msg in accompaniment_events:
        accompaniment_track.append(
            msg.copy(
                time=absolute_tick - previous_tick
            )
        )
        previous_tick = absolute_tick

    accompaniment_track.append(
        mido.MetaMessage(
            "end_of_track",
            time=0,
        )
    )

    os.makedirs(
        os.path.dirname(output_path) or ".",
        exist_ok=True,
    )

    result.save(output_path)

# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate(
    model,
    input_midi,
    original_input_midi,
    output_dir,
    bpm,
    prompt_length,
    generation_length,
    temperature,
    samples,
    seed,
    vicinity_fraction,
    prompt_key,
    conditioning_multiplier,
):
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=== Melody -> Chord / Pitch-Class Skeleton ===")
    print(f"Input:              {input_midi}")
    print(f"BPM:                {bpm}")
    print(f"Prompt length:      {prompt_length}")
    print(f"Generation length:  {generation_length}")
    print(f"Temperature:        {temperature}")
    print(f"Samples:            {samples}")
    print(f"Seed:               {seed}")
    print(f"Conditioning mult.: {conditioning_multiplier}")
    print(f"Chord window:       {CHORD_WINDOW} steps")
    print(f"Key context:        {KEY_CONTEXT_WINDOW} steps")
    print(f"Skeleton hop:       {SKELETON_HOP} steps")
    print()
    print(
        "The old melody-vicinity attack filter is disabled. "
        "--vicinity remains only for CLI compatibility."
    )

    original_melody = get_original_melody(original_input_midi)

    key_map = build_local_key_map(
        original_input_midi,
        prompt_key,
        prompt_length,
    )

    melody_structure = get_local_melody_structure(
        original_input_midi,
        prompt_length,
    )

    print()
    print("=== HARMONIC CONTEXT ===")

    if key_map["has_explicit_map"]:
        print(
            f"Key source:          MIDI key-signature map"
        )
        print(
            f"Initial active key:  {key_map['initial_key']}"
        )

        if key_map["changes"]:
            print("Local key changes:")

            for step, key in key_map["changes"]:
                print(
                    f"  step {step:4d} -> {key}"
                )
        else:
            print("Local key changes:   none")

    else:
        print(
            f"Key source:          prompt prior"
        )
        print(
            f"Initial active key:  {prompt_key}"
        )

    print(
        f"Chord window:        {CHORD_WINDOW} steps"
    )
    print(
        f"Key context window:  {KEY_CONTEXT_WINDOW} steps"
    )
    print(
        f"Skeleton hop:        {SKELETON_HOP} steps"
    )


    result = preprocess_midi(
        input_midi,
        16,
        ins_ids=["track-0", "track-1"],
        filter=False,
        fixed_length=generation_length,
    )
    if result is None:
        raise RuntimeError(
            "preprocess_midi() returned None. Synthetic MIDI must contain "
            "notes in track-0 and track-1."
        )

    x1, x2 = decompress(model, result[0])
    print(f"x1 shape: {x1.shape}")
    print(f"x2 shape: {x2.shape}")

    x1 = x1[:, :generation_length]
    x2 = x2[:, :prompt_length]

    if prompt_length >= MODEL_MAX_LENGTH:
        raise ValueError(
            f"Prompt length {prompt_length} must be < {MODEL_MAX_LENGTH}."
        )

    final_outputs = [[] for _ in range(samples)]
    overlap = prompt_length
    chunk_start = 0
    previous_chunk_end = 0
    chunk_number = 1

    print()
    print("=== MODEL GENERATION ===")

    while chunk_start < generation_length:
        if chunk_number == 1:
            chunk_start = 0
        else:
            chunk_start = previous_chunk_end - overlap

        chunk_end = min(chunk_start + MODEL_MAX_LENGTH, generation_length)
        melody_chunk = x1[:, chunk_start:chunk_end]

        print(
            f"CHUNK {chunk_number}: {chunk_start}:{chunk_end} "
            f"({chunk_end - chunk_start} steps)"
        )

        if chunk_number == 1:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            np.random.seed(seed)

            with torch.inference_mode():
                output = model.global_sampling(
                    melody_chunk.repeat(samples, 1, 1),
                    x2.repeat(samples, 1, 1),
                    temperature=temperature,
                    multiplier=float(conditioning_multiplier),
                )

            for i in range(samples):
                final_outputs[i].extend([
                    output[j][i:i + 1, :]
                    for j in range(len(output))
                ])

        else:
            prompt_start = chunk_start
            prompt_end = chunk_start + overlap

            prompts = []
            for i in range(samples):
                tokens = final_outputs[i][prompt_start:prompt_end]
                if len(tokens) != overlap:
                    raise RuntimeError(
                        f"Sample {i + 1}: expected {overlap} overlap tokens, "
                        f"got {len(tokens)}."
                    )
                prompts.append(torch.stack(tokens, dim=1))

            prompt_batch = torch.cat(prompts, dim=0)

            with torch.inference_mode():
                output = model.global_sampling(
                    melody_chunk.repeat(samples, 1, 1),
                    prompt_batch,
                    temperature=temperature,
                    multiplier=float(conditioning_multiplier),
                )

            for i in range(samples):
                output_i = [
                    output[j][i:i + 1, :]
                    for j in range(len(output))
                ]
                final_outputs[i].extend(output_i[overlap:])

        previous_chunk_end = chunk_end
        if previous_chunk_end >= generation_length:
            break
        chunk_number += 1

    for i, seq in enumerate(final_outputs):
        if len(seq) != generation_length:
            raise RuntimeError(
                f"Sample {i + 1}: expected {generation_length} steps, "
                f"got {len(seq)}."
            )

    # -----------------------------------------------------------------------
    # Harmonic skeletonization
    # -----------------------------------------------------------------------

    prompt_seconds = prompt_length * 60.0 / bpm / 4.0
    local_length = generation_length - prompt_length

    if local_length <= 0:
        raise RuntimeError(
            "Generation length must be greater than prompt length."
        )

    print()
    print("=== HARMONIC SKELETON EXTRACTION ===")
    print(f"Prompt cutoff: {prompt_seconds:.6f} s")

    for sample_index, seq in enumerate(final_outputs, 1):
        raw = decode_accompaniment_notes(
            (seq,),
            ratio=(model.compress_ratio_l, model.compress_ratio_r),
            tempo=bpm,
        )
        raw = [n for n in raw if n.start >= prompt_seconds]

        # Shift the generated portion to local step zero.  The prompt is never
        # allowed to influence later harmonic output except through the model.
        local_notes = [
            pretty_midi.Note(
                OUTPUT_VELOCITY,
                n.pitch,
                n.start - prompt_seconds,
                n.end - prompt_seconds,
            )
            for n in raw
            if n.end > prompt_seconds
        ]

        print()
        print(f"PROPOSAL {sample_index}/{samples}")
        print(f"  Raw CP notes after prompt: {len(local_notes)}")

        hops = build_pitch_class_hops(
            local_notes,
            local_length,
            bpm,
        )

        print(
            f"  {SKELETON_HOP}-step harmonic hops:      "
            f"{len(hops)}"
        )

        chord_states = all_chord_states()

        emissions, key_confidences = build_decoder_evidence(
            hops,
            chord_states,
            key_map,
        )

        # print_truth_landmark_emissions(
        #     hops,
        #     chord_states,
        #     emissions,
        # )

        # print_truth_temporal_emission_trace(
        #             hops,
        #             chord_states,
        #             emissions,
        #         )

        # print_truth_raw_accompaniment_trace(
        #     local_notes,
        #     hops,
        #     chord_states,
        #     emissions,
        #     bpm,
        # )

        # print_truth_four_step_harmonic_trace(
        #     local_notes,
        #     chord_states,
        #     bpm,
        # )

        states = decode_harmonic_sequence(
            hops,
            chord_states,
            emissions,
            key_confidences,
            key_map,
            melody_structure,
        )

        raw_viterbi_states = copy.deepcopy(states)

        states = refine_boundaries(
            hops,
            states,
            chord_states,
            emissions,
            melody_structure,
            key_map,
        )

        print_viterbi_vs_emission_diagnostics(
            hops,
            chord_states,
            emissions,
            raw_viterbi_states,
            states,
        )

        print_refinement_diagnostics(
            hops,
            raw_viterbi_states,
            states,
        )

        print_boundary_diagnostics(
            hops,
            states,
            chord_states,
            emissions,
            melody_structure,
        )

        family_regions = merge_regions(
            hops,
            states,
            local_length,
        )

        regions = resolve_region_extensions(
            family_regions,
            hops,
        )


        print(f"  Final harmonic regions:     {len(regions)}")
        for r in regions:
            s = r["state"]
            print(
                f"    {r['start']:4d}:{r['end']:4d}  "
                f"{s['label']}  score={s['score']:.3f}"
            )

        accompaniment = reconstruct_accompaniment(
            regions,
            local_length,
            bpm,
            prompt_length,
        )

        print(f"  Final chord notes:           {len(accompaniment)}")

        output_path = os.path.join(
            output_dir,
            f"proposal_{sample_index:02d}.mid",
        )

        write_final_midi(
            original_input_midi,
            output_path,
            copy.deepcopy(original_melody),
            accompaniment,
            bpm,
        )
        print(f"  Writing: {output_path}")

    print()
    print("DONE.")
    print(f"Output directory: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Melody -> chord inference with harmonic skeleton output."
    )

    parser.add_argument("input", help="Input melody MIDI.")
    parser.add_argument(
        "--key",
        required=True,
        help="Key for the initial two-bar prompt, e.g. 'Eb major'.",
    )
    parser.add_argument("--bpm", type=float, default=None)
    parser.add_argument("--time-signature", default=None)

    parser.add_argument(
        "--model",
        default=(
            "ckpt/mel_to_chord/"
            "cp_transformer_yinyang_v5.1_lora_batch_8_"
            "nottingham_cp8_v2_chord_mel_rev_mask0.0-10-step1."
            "epoch=last.ckpt"
        ),
        help="Melody -> chord checkpoint.",
    )
    parser.add_argument(
        "--output-dir",
        default="./test/outputs",
    )
    parser.add_argument(
        "--generation-length",
        type=int,
        default=None,
        help="16th-note steps; defaults to input MIDI length.",
    )
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument(
        "--conditioning-multiplier",
        type=float,
        default=DEFAULT_CONDITIONING_MULTIPLIER,
        help=(
            "YinYang melody-conditioning strength. "
            "1.0 = normal conditioning; "
            "0.0 = disable melody cross-attention contribution."
        ),
    )
    parser.add_argument("--samples", type=int, default=DEFAULT_SAMPLES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)

    # Compatibility only.
    parser.add_argument(
        "--vicinity",
        type=float,
        default=DEFAULT_VICINITY,
        help="Retained for compatibility; no longer filters CP attacks.",
    )

    args = parser.parse_args()

    if args.samples < 1:
        parser.error("--samples must be >= 1")
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if not 0 <= args.vicinity <= 1:
        parser.error("--vicinity must be between 0 and 1")

    if args.generation_length is None:
        args.generation_length = read_input_length(args.input)

    metadata = get_midi_metadata(
        args.input,
        cli_bpm=args.bpm,
        cli_time_signature=args.time_signature,
    )

    bpm = metadata["bpm"]
    numerator = metadata["numerator"]
    denominator = metadata["denominator"]

    quarter_notes_per_bar = numerator * 4.0 / denominator
    prompt_length = int(round(2 * quarter_notes_per_bar * 4))

    print()
    print("=== MIDI METADATA ===")
    print(f"Tempo:          {bpm:.3f} BPM ({metadata['bpm_source']})")
    print(
        f"Time signature: {numerator}/{denominator} "
        f"({metadata['time_signature_source']})"
    )
    print(f"Prompt length:   {prompt_length} 16th-note steps")
    print(f"Generation:      {args.generation_length} 16th-note steps")

    base = os.path.splitext(os.path.basename(args.input))[0]
    safe_key = args.key.replace(" ", "_")
    synthetic_input = os.path.join(
        args.output_dir,
        f"{base}_{safe_key}_prompt.mid",
    )

    print()
    print("Creating melody + two-bar chord prompt...")
    print(f"Key:             {args.key}")
    print(f"Chord pitches:   {make_tonic_triad(args.key)}")
    print(f"Prompt MIDI:     {synthetic_input}")

    create_melody_chord_input(
        args.input,
        synthetic_input,
        args.key,
    )

    print()
    print("Loading model...")
    print(args.model)

    model = RoformerYinyang.load_from_checkpoint(
        args.model,
        strict=False,
    )
    model.save_name = os.path.basename(args.model)
    model.cuda()
    model.eval()

    print()
    print("=== YINYANG CONDITIONING GATES ===")

    gate_values = []

    for i, attn in enumerate(model.yinyang_attn):
        gate = float(attn.gates.detach().float().cpu().item())
        gate_values.append(gate)

        print(
            f"adapter {i:02d}: "
            f"gate={gate:+.8f}"
        )

    if gate_values:
        gate_abs = [abs(x) for x in gate_values]

        print(
            f"gate summary: "
            f"count={len(gate_values)}  "
            f"min={min(gate_values):+.8f}  "
            f"max={max(gate_values):+.8f}  "
            f"mean={np.mean(gate_values):+.8f}  "
            f"mean_abs={np.mean(gate_abs):.8f}"
        )

    generate(
        model=model,
        input_midi=synthetic_input,
        original_input_midi=args.input,
        output_dir=args.output_dir,
        bpm=bpm,
        prompt_length=prompt_length,
        generation_length=args.generation_length,
        temperature=args.temperature,
        samples=args.samples,
        seed=args.seed,
        vicinity_fraction=args.vicinity,
        prompt_key=args.key,
        conditioning_multiplier=args.conditioning_multiplier,
    )


if __name__ == "__main__":
    main()

