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

# Kept for command-line compatibility.  It is intentionally NOT used to
# retain generated attacks anymore.
DEFAULT_VICINITY = 0.75

OUTPUT_VELOCITY = 100
MELODY_PROGRAM = 64
CHORD_PROGRAM = 49
CHORD_OCTAVE = 3

MODEL_MAX_LENGTH = 384

# SKELETON_WINDOW = 32       # two bars in 4/4
# SKELETON_HOP = 8            # analyse every half bar
# MIN_STATE_HOPS = 2          # hysteresis
# MIN_REGION_STEPS = 16       # do not output 1/2-bar chord blips

#TESTING
SKELETON_WINDOW = 16
SKELETON_HOP = 4
MIN_STATE_HOPS = 2
MIN_REGION_STEPS = 8


ROOT_CHANGE_PENALTY = 0.02
# ROOT_CHANGE_PENALTY = 0.001
# TYPE_CHANGE_PENALTY = 0.035
TYPE_CHANGE_PENALTY = 0.07

# New chord candidate/challenger must genuinely win, but not by an enormous amount
STATE_CHANGE_MARGIN = 0.02

LOWEST_OUTPUT_PITCH = 36    # C2
HIGHEST_OUTPUT_PITCH = 67    # G4
MAX_VOICING_NOTES = 4

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
# Pitch-class skeleton
# ---------------------------------------------------------------------------

def build_pitch_class_hops(notes, generation_length, bpm):
    """
    One record every 8 steps, looking at a 32-step window.

    Each pitch class receives sqrt(overlap_duration) evidence.  This makes
    sustained notes stronger than attacks, without allowing a 16-step note
    to count 16 times as much as a one-step note.

    No global key is consulted.
    """
    sixth = 60.0 / bpm / 4.0
    intervals = []

    for n in notes:
        a = n.start / sixth
        b = n.end / sixth
        if b > a:
            intervals.append((a, b, int(n.pitch) % 12))

    hops = []

    for start in range(0, generation_length, SKELETON_HOP):
        end = min(start + SKELETON_WINDOW, generation_length)
        if end <= start:
            break

        weights = np.zeros(12, dtype=np.float64)
        occupancy = np.zeros(12, dtype=np.float64)

        for a, b, pc in intervals:
            ov = max(0.0, min(b, end) - max(a, start))
            if ov <= 0:
                continue
            weights[pc] += math.sqrt(max(ov, 1.0))
            occupancy[pc] += ov

        total = weights.sum()
        weights = weights / total if total else weights

        occ = occupancy / float(end - start)
        persistent = [pc for pc in range(12) if occ[pc] >= 0.20]
        strong = [pc for pc in range(12) if weights[pc] >= 0.075]

        hops.append({
            "start": start,
            "end": end,
            "weights": weights,
            "occupancy": occ,
            "persistent": persistent,
            "strong": strong,
        })

    return hops


def chord_label(root, typ):
    return f"{PC_NAMES[root]}:{typ}"


def chord_score(weights, root, typ):
    intervals = dict(CHORD_TEMPLATES)[typ]
    chord_pcs = {(root + x) % 12 for x in intervals}

    support = sum(weights[p] for p in chord_pcs)
    outside = sum(weights[p] for p in range(12) if p not in chord_pcs)

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


def rank_chords(hop, k=10):
    result = []
    for typ, _ in CHORD_TEMPLATES:
        for root in range(12):
            result.append({
                "root": root,
                "type": typ,
                "label": chord_label(root, typ),
                "score": chord_score(hop["weights"], root, typ),
            })
    result.sort(key=lambda x: (-x["score"], x["root"], x["type"]))
    return result[:k]


def select_harmonic_states(hops):
    """
    Per-hop chord labels are NOT trusted directly.

    A candidate must beat the current chord and remain competitive for two
    consecutive hops before a change is committed.  This removes the
    8-step label flicker seen in the diagnostic while retaining local changes.
    """
    if not hops:
        return []

    states = []
    current = None
    pending_key = None
    pending_count = 0

    for hop in hops:
        candidates = rank_chords(hop)

        scored = []
        for c in candidates:
            s = c["score"]
            if current is not None:
                if c["root"] != current["root"]:
                    s -= ROOT_CHANGE_PENALTY
                if c["type"] != current["type"]:
                    s -= TYPE_CHANGE_PENALTY
            scored.append((s, c))

        scored.sort(key=lambda x: (-x[0], x[1]["root"], x[1]["type"]))
        best_score, best = scored[0]

        if current is None:
            current = dict(best)
            states.append(dict(current))
            continue

        stay = chord_score(hop["weights"], current["root"], current["type"])

        if best["root"] == current["root"] and best["type"] == current["type"]:
            pending_key = None
            pending_count = 0
            states.append(dict(current))
            continue

        # Hysteresis: a new label must have a meaningful local advantage.
        if best["score"] - stay < STATE_CHANGE_MARGIN:
            states.append(dict(current))
            continue

        key = (best["root"], best["type"])
        if key == pending_key:
            pending_count += 1
        else:
            pending_key = key
            pending_count = 1

        if pending_count >= MIN_STATE_HOPS:
            current = dict(best)
            pending_key = None
            pending_count = 0

        states.append(dict(current))

    return states


def cosine(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def region_vector(region, hops):
    ids = region["ids"]
    if not ids:
        return np.zeros(12)
    return np.mean([hops[i]["weights"] for i in ids], axis=0)


def merge_regions(hops, states, generation_length):
    """
    Convert per-hop harmonic states into contiguous, NON-OVERLAPPING
    harmonic regions.

    Important:
    A hop's `end` is the end of its ANALYSIS WINDOW.  It is NOT the
    end of the harmonic state.

    Therefore region boundaries are defined by hop START positions.
    """

    if not hops:
        return []

    if len(hops) != len(states):
        raise ValueError(
            "hops/states length mismatch: "
            f"{len(hops)} != {len(states)}"
        )

    # ---------------------------------------------------------------
    # First create contiguous runs of identical harmonic states.
    # ---------------------------------------------------------------

    regions = []

    cur = {
        "start": int(hops[0]["start"]),
        "end": None,
        "state": dict(states[0]),
        "ids": [0],
    }

    for i in range(1, len(hops)):

        same = (
            states[i]["root"] == cur["state"]["root"]
            and
            states[i]["type"] == cur["state"]["type"]
        )

        if same:
            cur["ids"].append(i)
            continue

        # The new state's hop START is the exact boundary.
        boundary = int(hops[i]["start"])

        cur["end"] = boundary
        regions.append(cur)

        cur = {
            "start": boundary,
            "end": None,
            "state": dict(states[i]),
            "ids": [i],
        }

    # Last state runs to the end of generated material.
    cur["end"] = int(generation_length)
    regions.append(cur)

    # ---------------------------------------------------------------
    # Remove zero/negative regions defensively.
    # ---------------------------------------------------------------

    regions = [
        r for r in regions
        if r["end"] > r["start"]
    ]

    # ---------------------------------------------------------------
    # Suppress short regions.
    #
    # IMPORTANT:
    # Merging must move ONE SHARED BOUNDARY.
    # Never use min(start)/max(end) on both sides because that recreates
    # overlapping regions.
    # ---------------------------------------------------------------

    changed = True

    while changed and len(regions) > 1:

        changed = False

        for i, r in enumerate(regions):

            length = r["end"] - r["start"]

            if length >= MIN_REGION_STEPS:
                continue

            # -------------------------------------------------------
            # Decide whether this short region belongs left or right.
            # -------------------------------------------------------

            if i == 0:
                target = i + 1

            elif i == len(regions) - 1:
                target = i - 1

            else:
                rv = region_vector(r, hops)

                left_similarity = cosine(
                    rv,
                    region_vector(regions[i - 1], hops),
                )

                right_similarity = cosine(
                    rv,
                    region_vector(regions[i + 1], hops),
                )

                target = (
                    i - 1
                    if left_similarity >= right_similarity
                    else i + 1
                )

            # -------------------------------------------------------
            # Absorb the short region WITHOUT overlap.
            # -------------------------------------------------------

            if target < i:
                # Absorb into left neighbor.
                regions[target]["end"] = r["end"]
                regions[target]["ids"].extend(r["ids"])

            else:
                # Absorb into right neighbor.
                regions[target]["start"] = r["start"]
                regions[target]["ids"].extend(r["ids"])

            regions.pop(i)

            changed = True
            break

    # ---------------------------------------------------------------
    # Final normalization.
    # ---------------------------------------------------------------

    for i, r in enumerate(regions):

        r["start"] = max(
            0,
            int(r["start"]),
        )

        r["end"] = min(
            int(generation_length),
            int(r["end"]),
        )

        r["ids"] = sorted(set(r["ids"]))

        # Force perfect continuity with the next region.
        if i > 0:
            r["start"] = regions[i - 1]["end"]

    if regions:
        regions[0]["start"] = 0
        regions[-1]["end"] = int(generation_length)

    # ---------------------------------------------------------------
    # Sanity check: overlap is a programming error.
    # ---------------------------------------------------------------

    for i in range(1, len(regions)):

        if regions[i]["start"] != regions[i - 1]["end"]:
            raise RuntimeError(
                "Non-contiguous harmonic regions: "
                f"{regions[i - 1]['start']}:"
                f"{regions[i - 1]['end']} followed by "
                f"{regions[i]['start']}:"
                f"{regions[i]['end']}"
            )

    return regions

# ---------------------------------------------------------------------------
# Simple sustained voicing
# ---------------------------------------------------------------------------

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


def reconstruct_accompaniment(regions, generation_length, bpm, prompt_steps):
    """
    One sustained voicing per harmonic region.

    `regions` are in generated-local step coordinates. The returned notes are
    shifted past the artificial prompt.

    IMPORTANT:
    Notes are NOT merged across harmonic-region boundaries.
    Every new chord retriggers all of its pitches, including pitches shared
    with the previous chord.
    """
    sixth = 60.0 / bpm / 4.0
    notes = []

    for r in regions:
        st = (prompt_steps + r["start"]) * sixth
        en = min(
            (prompt_steps + r["end"]) * sixth,
            (prompt_steps + generation_length) * sixth,
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
    print(f"Skeleton window:    {SKELETON_WINDOW} steps")
    print(f"Skeleton hop:       {SKELETON_HOP} steps")
    print(f"Minimum region:     {MIN_REGION_STEPS} steps")
    print()
    print(
        "The old melody-vicinity attack filter is disabled. "
        "--vicinity remains only for CLI compatibility."
    )

    original_melody = get_original_melody(original_input_midi)

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
        print(f"  8-step harmonic hops:      {len(hops)}")

        states = select_harmonic_states(hops)
        regions = merge_regions(hops, states, local_length)

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
    )


if __name__ == "__main__":
    main()

