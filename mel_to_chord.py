import argparse
import copy
import os
import re

import mido
import numpy as np
import pretty_midi
import torch

from cp_transformer_yinyang import RoformerYinyang, PreprocessingParameters
from preprocess_large_midi_dataset import (
    preprocess_midi,
    DURATION_TEMPLATES,
)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_BPM = 120.0
DEFAULT_NUMERATOR = 4
DEFAULT_DENOMINATOR = 4
DEFAULT_QUANTIZATION = 16  # 16ths

DEFAULT_TEMPERATURE = 1.0
DEFAULT_SAMPLES = 1
DEFAULT_SEED = 0

# Fraction of one 16th-note used as the melody-onset vicinity.
#
# Example:
#     0.15 = +/- 15% of a 16th note around every melody onset.
#
# This is deliberately expressed in musical time rather than milliseconds.
DEFAULT_VICINITY = 0.475

# All output note velocities are forced to this value.
OUTPUT_VELOCITY = 100

# Nottingham-style melody -> chord configuration.
MELODY_PROGRAM = 64
CHORD_PROGRAM = 0

# Chord notes are placed in octave 3.
CHORD_OCTAVE = 3


# ---------------------------------------------------------------------------
# MIDI metadata
# ---------------------------------------------------------------------------

def read_tempo_map(midi_path):
    """
    Read the actual MIDI tempo map using mido.

    Returns:
        list of (absolute_tick, bpm)
    """
    midi = mido.MidiFile(midi_path)

    tempo_map = []

    for track in midi.tracks:
        absolute_tick = 0

        for msg in track:
            absolute_tick += msg.time

            if msg.type == "set_tempo":
                bpm = mido.tempo2bpm(msg.tempo)
                tempo_map.append(
                    (absolute_tick, float(bpm))
                )

    tempo_map.sort(key=lambda x: x[0])

    # Avoid duplicate tempo events at the same tick.
    result = []

    for tick, bpm in tempo_map:
        if result and result[-1][0] == tick:
            result[-1] = (tick, bpm)
        else:
            result.append((tick, bpm))

    return result


def read_time_signature_map(midi_path):
    """
    Read the actual MIDI time-signature map using mido.

    Returns:
        list of (absolute_tick, numerator, denominator)
    """
    midi = mido.MidiFile(midi_path)

    time_signature_map = []

    for track in midi.tracks:
        absolute_tick = 0

        for msg in track:
            absolute_tick += msg.time

            if msg.type == "time_signature":
                time_signature_map.append(
                    (
                        absolute_tick,
                        int(msg.numerator),
                        int(msg.denominator),
                    )
                )

    time_signature_map.sort(key=lambda x: x[0])

    result = []

    for tick, numerator, denominator in time_signature_map:
        if result and result[-1][0] == tick:
            result[-1] = (
                tick,
                numerator,
                denominator,
            )
        else:
            result.append(
                (
                    tick,
                    numerator,
                    denominator,
                )
            )

    return result


def parse_time_signature(value):
    match = re.fullmatch(
        r"\s*(\d+)\s*/\s*(\d+)\s*",
        value,
    )

    if not match:
        raise ValueError(
            f"Invalid time signature '{value}'. "
            "Expected e.g. 4/4 or 3/4."
        )

    numerator = int(match.group(1))
    denominator = int(match.group(2))

    if numerator <= 0 or denominator <= 0:
        raise ValueError(
            "Time-signature values must be positive."
        )

    return numerator, denominator


def get_midi_metadata(
    midi_path,
    cli_bpm=None,
    cli_time_signature=None,
):
    """
    Determine BPM and time signature.

    Priority:
        1. MIDI tempo map / time-signature map
        2. user supplied values
        3. defaults

    The resolved BPM is normalized to 2 decimal places.

    The complete MIDI tempo/time-signature maps are preserved.
    """

    tempo_map = read_tempo_map(midi_path)
    time_signature_map = read_time_signature_map(midi_path)

    # ------------------------------------------------------------------
    # Tempo
    # ------------------------------------------------------------------

    if tempo_map:
        bpm = round(
            float(tempo_map[0][1]),
            2,
        )
        bpm_source = "MIDI tempo map"

    elif cli_bpm is not None:
        bpm = round(
            float(cli_bpm),
            2,
        )
        bpm_source = "command line"

    else:
        bpm = round(
            float(DEFAULT_BPM),
            2,
        )
        bpm_source = "default"

    # Normalize EVERY tempo-map entry immediately.
    normalized_tempo_map = [
        (
            tick,
            round(
                float(event_bpm),
                2,
            ),
        )
        for tick, event_bpm in tempo_map
    ]

    # ------------------------------------------------------------------
    # Time signature
    # ------------------------------------------------------------------

    if time_signature_map:
        _, numerator, denominator = (
            time_signature_map[0]
        )

        time_signature_source = (
            "MIDI time-signature map"
        )

    elif cli_time_signature is not None:
        numerator, denominator = (
            parse_time_signature(
                cli_time_signature
            )
        )

        time_signature_source = "command line"

    else:
        numerator = DEFAULT_NUMERATOR
        denominator = DEFAULT_DENOMINATOR

        time_signature_source = "default"

    return {
        "bpm": bpm,
        "numerator": numerator,
        "denominator": denominator,
        "bpm_source": bpm_source,
        "time_signature_source": time_signature_source,

        "tempo_map": normalized_tempo_map,
        "time_signature_map": time_signature_map,
    }


# ---------------------------------------------------------------------------
# INPUT MIDI FILE LENGTH
# ---------------------------------------------------------------------------

def read_input_length(midi_path):
    midi = mido.MidiFile(midi_path)

    max_tick = 0

    for track in midi.tracks:
        absolute_tick = 0

        for msg in track:
            absolute_tick += msg.time

        max_tick = max(
            max_tick,
            absolute_tick,
        )

    return int(
        round(
            (
                max_tick
                / midi.ticks_per_beat
            )
            * DEFAULT_QUANTIZATION
            / 4
        )
    )


# ---------------------------------------------------------------------------
# Key / chord handling
# ---------------------------------------------------------------------------

PITCH_CLASSES = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}


def parse_key(key):
    """
    Parse keys such as:

        C minor
        Cm
        Eb major
        Eb
        F# minor
        Bb major
    """

    value = key.strip()

    match = re.fullmatch(
        r"([A-Ga-g](?:#|b)?)(?:\s*)"
        r"(major|minor|maj|min|M|m)?",
        value,
    )

    if not match:
        raise ValueError(
            f"Invalid key '{key}'. "
            "Examples: 'C minor', 'Cm', 'Eb major'."
        )

    root = match.group(1)
    mode = match.group(2)

    root = (
        root[0].upper()
        + root[1:]
    )

    if root not in PITCH_CLASSES:
        raise ValueError(
            f"Unsupported key root: {root}"
        )

    if mode is None:
        mode = "major"

    mode = mode.lower()

    if mode in (
        "m",
        "min",
        "minor",
    ):
        mode = "minor"
    else:
        mode = "major"

    return root, mode


def make_tonic_triad(key):
    """
    Return MIDI pitches for the tonic triad.

    Examples:
        Cm         -> C3 Eb3 G3
        Eb major   -> Eb3 G3 Bb3
    """

    root_name, mode = parse_key(key)

    root_pc = PITCH_CLASSES[root_name]

    third = (
        3
        if mode == "minor"
        else 4
    )

    fifth = 7

    root_pitch = (
        12 * (CHORD_OCTAVE + 1)
        + root_pc
    )

    return [
        root_pitch,
        root_pitch + third,
        root_pitch + fifth,
    ]


# ---------------------------------------------------------------------------
# Synthetic melody -> chord input
# ---------------------------------------------------------------------------

def calculate_bar_ticks(
    resolution,
    numerator,
    denominator,
):
    """
    Length of one bar in MIDI ticks.

    MIDI resolution is ticks per quarter note.
    """

    quarter_notes_per_bar = (
        numerator
        * (4.0 / denominator)
    )

    return int(
        round(
            resolution
            * quarter_notes_per_bar
        )
    )


def create_melody_chord_input(
    input_midi_path,
    output_midi_path,
    key,
):
    """
    Create the exact two-track structure required by the
    melody -> chord model.

    Track 0:
        original melody

    Track 1:
        tonic triad, whole-bar notes, for first two bars

    This MIDI is only the model-conditioning MIDI.
    The final output is reconstructed separately from the
    original input melody plus the generated accompaniment.
    """

    source = pretty_midi.PrettyMIDI(
        input_midi_path
    )

    non_empty = [
        ins
        for ins in source.instruments
        if len(ins.notes) > 0
    ]

    if len(non_empty) != 1:
        raise ValueError(
            "Melody → chord input must contain exactly "
            "one non-empty melody track. "
            f"Found {len(non_empty)}."
        )

    melody_source = non_empty[0]

    metadata = get_midi_metadata(
        input_midi_path,
        cli_bpm=None,
        cli_time_signature=None,
    )

    bpm = metadata["bpm"]
    numerator = metadata["numerator"]
    denominator = metadata["denominator"]

    result = pretty_midi.PrettyMIDI(
        resolution=source.resolution,
        initial_tempo=bpm,
    )

    result.time_signature_changes = copy.deepcopy(
        source.time_signature_changes
    )

    result.key_signature_changes = copy.deepcopy(
        source.key_signature_changes
    )

    # ------------------------------------------------------------------
    # Track 0 = melody
    #
    # All velocities are deliberately normalized to 100.
    # ------------------------------------------------------------------

    melody = pretty_midi.Instrument(
        program=melody_source.program,
        is_drum=melody_source.is_drum,
        name="Melody",
    )

    melody.notes = [
        pretty_midi.Note(
            velocity=OUTPUT_VELOCITY,
            pitch=note.pitch,
            start=note.start,
            end=note.end,
        )
        for note in melody_source.notes
    ]

    result.instruments.append(melody)

    # ------------------------------------------------------------------
    # Track 1 = tonic chord prompt
    # ------------------------------------------------------------------

    chord = pretty_midi.Instrument(
        program=CHORD_PROGRAM,
        is_drum=False,
        name=f"{key} prompt",
    )

    chord_pitches = make_tonic_triad(key)

    bar_ticks = calculate_bar_ticks(
        source.resolution,
        numerator,
        denominator,
    )

    # Two bars.
    for bar in range(2):

        start_tick = (
            bar * bar_ticks
        )

        end_tick = (
            (bar + 1) * bar_ticks
        )

        start_time = source.tick_to_time(
            start_tick
        )

        end_time = source.tick_to_time(
            end_tick
        )

        for pitch in chord_pitches:

            chord.notes.append(
                pretty_midi.Note(
                    velocity=OUTPUT_VELOCITY,
                    pitch=pitch,
                    start=start_time,
                    end=end_time,
                )
            )

    result.instruments.append(chord)

    output_parent = os.path.dirname(
        output_midi_path
    )

    if output_parent:
        os.makedirs(
            output_parent,
            exist_ok=True,
        )

    result.write(output_midi_path)

    return metadata, chord_pitches


# ---------------------------------------------------------------------------
# Model inference helpers
# ---------------------------------------------------------------------------

def decompress(
    model,
    byte_arr,
):
    x = torch.tensor(
        byte_arr
    ).unsqueeze(0)

    x = x.cuda()

    return model.preprocess(
        x,
        pitch_shift=torch.zeros(
            1,
            dtype=torch.int8,
            device="cuda",
        ),
        preprocess_args=PreprocessingParameters(""),
    )[:2]


# ---------------------------------------------------------------------------
# Original melody extraction
# ---------------------------------------------------------------------------

def get_original_melody(
    midi_path,
):
    """
    Copy the melody directly from the ORIGINAL INPUT MIDI.

    No quantization.
    No re-timing.
    No reconstruction from preprocess_midi().

    All output melody velocities are normalized to 100.
    """

    source = pretty_midi.PrettyMIDI(
        midi_path
    )

    non_empty = [
        ins
        for ins in source.instruments
        if len(ins.notes) > 0
    ]

    if len(non_empty) != 1:
        raise ValueError(
            "Input MIDI must contain exactly one "
            "non-empty melody track. "
            f"Found {len(non_empty)}."
        )

    source_melody = non_empty[0]

    melody = pretty_midi.Instrument(
        program=source_melody.program,
        is_drum=source_melody.is_drum,
        name="Melody",
    )

    melody.notes = [
        pretty_midi.Note(
            velocity=OUTPUT_VELOCITY,
            pitch=note.pitch,
            start=note.start,
            end=note.end,
        )
        for note in source_melody.notes
    ]

    return melody


# ---------------------------------------------------------------------------
# Generated CP -> ONE accompaniment note list
# ---------------------------------------------------------------------------

def decode_accompaniment_notes(
    outputs,
    ratio,
    tempo,
    min_pitch=None,
):
    """
    Decode generated CP output into a SINGLE list of
    accompaniment notes.

    IMPORTANT:

    The original decode_output() creates a separate instrument
    for every distinct generated program. We explicitly do NOT
    do that here.

    Every generated note is placed into the same accompaniment
    instrument.

    Every note velocity is forced to 100.
    """

    from cp_transformer import CPTokenizer

    tokenizer = CPTokenizer(
        with_velocity=False
    )

    if not isinstance(outputs, tuple):
        outputs = (outputs,)

    if not isinstance(ratio, tuple):
        ratio = (
            ratio,
        ) * len(outputs)

    notes = []

    time_step_length = (
        60.0 / tempo / 4.0
    )

    for r, output in zip(
        ratio,
        outputs,
    ):

        for time_step, data in enumerate(
            output
        ):

            content = data.squeeze(0)

            start_time = (
                time_step
                * time_step_length
            )

            for i in range(
                0,
                len(content),
                2,
            ):

                program = int(
                    content[i].item()
                )

                if program == tokenizer.eos_token:
                    break

                if i + 1 >= len(content):
                    print(
                        "Incomplete note @",
                        time_step,
                        i,
                    )
                    break

                pitch_duration = (
                    int(
                        content[i + 1].item()
                    )
                    - 128
                )

                pitch = (
                    pitch_duration
                    % 128
                )

                duration = (
                    pitch_duration
                    // 128
                )

                # The generated program is intentionally ignored
                # for instrument creation. It only needs to be
                # syntactically valid CP data.
                if program < 0 or program >= 128:
                    print(
                        "Invalid program:",
                        program,
                        "@",
                        time_step,
                        i,
                    )
                    break

                if pitch < 0 or pitch >= 128:
                    print(
                        "Invalid pitch:",
                        pitch,
                        "@",
                        time_step,
                        i,
                    )
                    break

                if min_pitch is not None:

                    while pitch < min_pitch:
                        pitch += 12

                    if pitch >= 128:
                        continue

                if (
                    duration < 0
                    or duration >= len(
                        DURATION_TEMPLATES
                    )
                ):
                    print(
                        "Invalid duration:",
                        duration,
                        "@",
                        time_step,
                        i,
                    )
                    break

                end_time = (
                    DURATION_TEMPLATES[
                        duration
                    ]
                    * time_step_length
                    * r
                    + start_time * r
                )

                note_start = (
                    start_time * r
                )

                notes.append(
                    pretty_midi.Note(
                        velocity=OUTPUT_VELOCITY,
                        pitch=pitch,
                        start=note_start,
                        end=end_time,
                    )
                )

    return notes


# ---------------------------------------------------------------------------
# Artificial prompt removal
# ---------------------------------------------------------------------------

def remove_initial_accompaniment_notes(
    notes,
    cutoff_time,
):
    """
    Remove complete accompaniment notes whose onset is inside
    the artificial prompt region.

    IMPORTANT:

    The remaining notes are NOT shifted.

    This is deliberately done on complete Note objects rather
    than individual MIDI note_on/note_off events.

    Therefore a note starting after the prompt retains its
    original absolute start time.
    """

    return [
        note
        for note in notes
        if note.start >= cutoff_time
    ]


# ---------------------------------------------------------------------------
# Accompaniment vicinity filtering
# ---------------------------------------------------------------------------

def filter_accompaniment_by_melody_vicinity(
    accompaniment_notes,
    melody_notes,
    vicinity_seconds,
):
    """
    Retain accompaniment notes whose START is within
    +/- vicinity_seconds of at least one melody-note start.

    The vicinity itself is specified by the user as a fraction
    of a 16th note. This function receives the internally
    converted seconds value.

    Notes are NOT moved. Only notes outside the permitted
    onset windows are discarded.
    """

    if not accompaniment_notes:
        return []

    if not melody_notes:
        return []

    melody_starts = np.asarray(
        sorted(
            note.start
            for note in melody_notes
        ),
        dtype=np.float64,
    )

    retained = []

    for note in accompaniment_notes:

        start = float(
            note.start
        )

        # Position where this accompaniment onset would be
        # inserted in the sorted melody onset array.
        index = np.searchsorted(
            melody_starts,
            start,
            side="left",
        )

        matched = False

        # Melody onset immediately before this accompaniment onset.
        if index > 0:

            previous_start = (
                melody_starts[index - 1]
            )

            if (
                abs(
                    start
                    - previous_start
                )
                <= vicinity_seconds
            ):
                matched = True

        # Melody onset immediately after this accompaniment onset.
        if index < len(melody_starts):

            next_start = (
                melody_starts[index]
            )

            if (
                abs(
                    start
                    - next_start
                )
                <= vicinity_seconds
            ):
                matched = True

        if matched:
            retained.append(note)

    return retained


# ---------------------------------------------------------------------------
# Accompaniment deduplication
# ---------------------------------------------------------------------------

def deduplicate_accompaniment_notes(
    notes,
):
    """
    For notes having the same pitch and same start time,
    retain only the note with the longest duration.

    Velocity is always 100, so velocity does not participate
    in the comparison.

    The start-time rounding prevents insignificant floating-point
    noise from preventing legitimate duplicates from being merged.
    """

    best = {}

    for note in notes:

        key = (
            int(note.pitch),
            round(
                float(note.start),
                9,
            ),
        )

        previous = best.get(key)

        if previous is None:

            best[key] = note

            continue

        previous_duration = (
            previous.end
            - previous.start
        )

        current_duration = (
            note.end
            - note.start
        )

        if current_duration > previous_duration:
            best[key] = note

    return sorted(
        best.values(),
        key=lambda note: (
            note.start,
            note.pitch,
            note.end,
        ),
    )


# ---------------------------------------------------------------------------
# Accompaniment duration extension
# ---------------------------------------------------------------------------

def extend_accompaniment_durations_to_next_melody(
    accompaniment_notes,
    melody_notes,
):
    """
    For every accompaniment note:

        find the first melody onset strictly AFTER
        the accompaniment start.

    If the accompaniment note ends before that melody onset,
    extend it to that melody onset.

    Existing longer durations are preserved.

    Notes occurring after the final melody onset are left unchanged.
    """

    if not accompaniment_notes:
        return accompaniment_notes

    if not melody_notes:
        return accompaniment_notes

    melody_starts = np.asarray(
        sorted(
            note.start
            for note in melody_notes
        ),
        dtype=np.float64,
    )

    for note in accompaniment_notes:

        index = np.searchsorted(
            melody_starts,
            note.start,
            side="right",
        )

        # No later melody note.
        if index >= len(melody_starts):
            continue

        next_melody_start = (
            float(
                melody_starts[index]
            )
        )

        required_end = (
            next_melody_start
        )

        # Never shorten an existing note.
        if note.end < required_end:

            note.end = required_end

    return accompaniment_notes

# ---------------------------------------------------------------------------
# Accompaniment removal of overlapping notes
# ---------------------------------------------------------------------------

def remove_overlapping_same_pitch_notes(notes):
    """
    For notes with the same pitch that overlap:
      - retain the earliest-starting note
      - extend its end to the maximum end time of the overlapping notes
      - discard the later overlapping notes

    Overlap is defined as:
        later.start < current.end

    Notes that merely touch at the boundary are NOT considered overlapping.
    """

    if not notes:
        return []

    # Group by pitch.
    notes_by_pitch = {}

    for note in notes:
        pitch = int(note.pitch)
        notes_by_pitch.setdefault(pitch, []).append(note)

    result = []

    for pitch_notes in notes_by_pitch.values():

        # Earliest start first.
        # For identical starts, longest duration first.
        pitch_notes.sort(
            key=lambda n: (
                float(n.start),
                -float(n.end),
            )
        )

        merged = []

        for note in pitch_notes:

            if not merged:
                merged.append(note)
                continue

            current = merged[-1]

            # Same-pitch notes are already grouped together.
            # If the new note starts before the retained note ends,
            # they overlap.
            if float(note.start) < float(current.end):

                # Keep the earliest note and extend its end if necessary.
                if float(note.end) > float(current.end):
                    current.end = float(note.end)

            else:
                # No overlap.
                merged.append(note)

        result.extend(merged)

    # Restore chronological order.
    result.sort(
        key=lambda n: (
            float(n.start),
            int(n.pitch),
            float(n.end),
        )
    )

    return result

# ---------------------------------------------------------------------------
# Final MIDI writer
# ---------------------------------------------------------------------------

def write_final_midi(
    input_midi_path,
    output_path,
    melody,
    accompaniment_notes,
    tempo,
):
    """
    Write the final MIDI.

    Exactly two musical instruments are written:

        Track/instrument 0 = original melody
        Track/instrument 1 = all accompaniment

    All note velocities are 100.
    """

    source = pretty_midi.PrettyMIDI(
        input_midi_path
    )

    # ---------------------------------------------------------------
    # Ensure the melody itself is normalized to velocity 100.
    # ---------------------------------------------------------------

    for note in melody.notes:
        note.velocity = OUTPUT_VELOCITY

    # ---------------------------------------------------------------
    # Single accompaniment instrument.
    # ---------------------------------------------------------------

    accompaniment = pretty_midi.Instrument(
        program=CHORD_PROGRAM,
        is_drum=False,
        name="Accompaniment",
    )

    for note in accompaniment_notes:
        note.velocity = OUTPUT_VELOCITY

    accompaniment.notes = accompaniment_notes

    # ---------------------------------------------------------------
    # Construct clean output MIDI.
    # ---------------------------------------------------------------

    result = pretty_midi.PrettyMIDI(
        resolution=source.resolution,
        initial_tempo=tempo,
    )

    # Preserve original time signatures.
    result.time_signature_changes = copy.deepcopy(
        source.time_signature_changes
    )

    # Preserve original key signatures.
    result.key_signature_changes = copy.deepcopy(
        source.key_signature_changes
    )

    # Exactly two musical instruments.
    result.instruments.append(
        melody
    )

    result.instruments.append(
        accompaniment
    )

    output_parent = os.path.dirname(
        output_path
    )

    if output_parent:
        os.makedirs(
            output_parent,
            exist_ok=True,
        )

    result.write(
        output_path
    )


# ---------------------------------------------------------------------------
# Main generation
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
    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print()
    print("=== Melody → Chord ===")
    print(f"Input:              {input_midi}")
    print(f"BPM:                {bpm}")
    print(f"Prompt length:      {prompt_length}")
    print(f"Generation:         {generation_length}")
    print(f"Temperature:        {temperature}")
    print(f"Samples:            {samples}")
    print(f"Seed:               {seed}")
    print(
        f"Vicinity:           "
        f"{vicinity_fraction} × 16th"
    )
    print(
        f"Output velocity:    "
        f"{OUTPUT_VELOCITY}"
    )
    print()

    # ------------------------------------------------------------------
    # Original melody.
    #
    # This is used ONLY for the final output.
    # The synthetic MIDI remains the model input.
    # ------------------------------------------------------------------

    original_melody = get_original_melody(
        original_input_midi
    )

    # ------------------------------------------------------------------
    # Synthetic MIDI is already a two-track MIDI:
    #
    # track-0 = melody
    # track-1 = artificial chord prompt
    # ------------------------------------------------------------------

    ins_ids = [
        "track-0",
        "track-1",
    ]

    fixed_program = [
        MELODY_PROGRAM,
        CHORD_PROGRAM,
    ]

    fixed_velocity = [
        OUTPUT_VELOCITY,
        OUTPUT_VELOCITY,
    ]

    # Keep these variables because they document the intended
    # preprocessing configuration and preserve compatibility with
    # the existing script.
    _ = fixed_program
    _ = fixed_velocity

    print("Preprocessing MIDI...")

    result = preprocess_midi(
        input_midi,
        16,
        ins_ids=ins_ids,
        filter=False,
        fixed_length=generation_length,
    )

    if result is None:
        raise RuntimeError(
            "preprocess_midi() returned None.\n"
            "The synthetic MIDI must contain notes in both "
            "track-0 and track-1."
        )

    byte_arr = result[0]

    x1, x2 = decompress(
        model,
        byte_arr,
    )

    print(
        f"x1 shape: {x1.shape}"
    )

    print(
        f"x2 shape: {x2.shape}"
    )

    # x1 = conditioning melody
    # x2 = chord-generation stream

    x1 = x1[
        :,
        :generation_length,
    ]

    x2 = x2[
        :,
        :prompt_length,
    ]

    print(
        "Conditioning stream:",
        x1.shape,
    )

    print(
        "Generation prompt:",
        x2.shape,
    )

    # ------------------------------------------------------------------
    # Chord generation
    #
    # The model was trained with a 384-step sequence length.
    #
    # For longer inputs, generate overlapping chunks:
    #
    #   Chunk 1:  0       -> 384
    #   Chunk 2:  352     -> 736
    #   Chunk 3:  704     -> 1088
    #   ...
    #
    # Each new chunk uses the previous 32 generated chord steps
    # as its prompt.
    # ------------------------------------------------------------------

    MODEL_MAX_LENGTH = 384

    overlap = prompt_length

    if overlap >= MODEL_MAX_LENGTH:
        raise ValueError(
            f"Prompt length ({overlap}) must be smaller than "
            f"model maximum length ({MODEL_MAX_LENGTH})."
        )

    # One output list per requested sample.
    final_outputs = [
        []
        for _ in range(samples)
    ]

    chunk_start = 0
    chunk_number = 1

    print("Generating...")

    while chunk_start < generation_length:

        # --------------------------------------------------------------
        # First chunk starts at zero.
        #
        # Subsequent chunks start `overlap` steps before the previous
        # chunk ended.
        # --------------------------------------------------------------

        if chunk_number == 1:
            chunk_start = 0

        else:
            chunk_start = (
                previous_chunk_end
                - overlap
            )

        chunk_end = min(
            chunk_start
            + MODEL_MAX_LENGTH,
            generation_length,
        )

        chunk_length = (
            chunk_end
            - chunk_start
        )

        print()
        print(
            f"CHUNK {chunk_number}: "
            f"{chunk_start}:{chunk_end} "
            f"({chunk_length} steps)"
        )

        # --------------------------------------------------------------
        # Melody conditioning for this chunk.
        # --------------------------------------------------------------

        melody_chunk = x1[
            :,
            chunk_start:chunk_end,
        ]

        # ==============================================================
        # FIRST CHUNK
        # ==============================================================

        if chunk_number == 1:

            chord_prompt = x2

            print(
                f"  Melody: "
                f"{chunk_start}:{chunk_end}"
            )

            print(
                f"  Chord prompt: artificial "
                f"{prompt_length} steps"
            )

            # Generate all samples together.
            with torch.inference_mode():

                torch.manual_seed(
                    seed
                )

                torch.cuda.manual_seed_all(
                    seed
                )

                np.random.seed(
                    seed
                )

                melody_batch = (
                    melody_chunk.repeat(
                        samples,
                        1,
                        1,
                    )
                )

                prompt_batch = (
                    chord_prompt.repeat(
                        samples,
                        1,
                        1,
                    )
                )

                output = (
                    model.global_sampling(
                        melody_batch,
                        prompt_batch,
                        temperature=temperature,
                    )
                )

            for i in range(samples):

                output_i = [
                    output[j][
                        i:i + 1,
                        :
                    ]
                    for j in range(
                        len(output)
                    )
                ]

                final_outputs[i].extend(
                    output_i
                )

        # ==============================================================
        # SUBSEQUENT CHUNKS
        # ==============================================================

        else:

            prompt_start = (
                chunk_start
            )

            prompt_end = (
                chunk_start
                + overlap
            )

            print(
                f"  Melody: "
                f"{chunk_start}:{chunk_end}"
            )

            print(
                f"  Chord prompt: "
                f"{prompt_start}:{prompt_end}"
            )

            print(
                f"  Generated chunk: "
                f"{chunk_start}:{chunk_end}"
            )

            print(
                f"  Retained: "
                f"{prompt_end}:{chunk_end}"
            )

            # ----------------------------------------------------------
            # Each sample has its own generated chord history.
            #
            # Build one prompt per sample, then batch them.
            # ----------------------------------------------------------

            chord_prompts = []

            for i in range(samples):

                previous_output = (
                    final_outputs[i]
                )

                prompt_tokens = (
                    previous_output[
                        prompt_start:
                        prompt_end
                    ]
                )

                # 32 x [1, 32]
                # ->
                # [1, 32, 32]
                chord_prompt = (
                    torch.stack(
                        prompt_tokens,
                        dim=1,
                    )
                )

                chord_prompts.append(
                    chord_prompt
                )

            # [samples, prompt_length, 32]
            prompt_batch = torch.cat(
                chord_prompts,
                dim=0,
            )

            # Same melody conditioning for every sample.
            melody_batch = (
                melody_chunk.repeat(
                    samples,
                    1,
                    1,
                )
            )

            with torch.inference_mode():

                output = (
                    model.global_sampling(
                        melody_batch,
                        prompt_batch,
                        temperature=temperature,
                    )
                )

            # Each output is:
            #
            #   [prompt overlap][new generation]
            #
            # Keep only the new portion.
            for i in range(samples):

                output_i = [
                    output[j][
                        i:i + 1,
                        :
                    ]
                    for j in range(
                        len(output)
                    )
                ]

                new_output = (
                    output_i[overlap:]
                )

                final_outputs[i].extend(
                    new_output
                )

        previous_chunk_end = chunk_end

        if (
            previous_chunk_end
            >= generation_length
        ):
            break

        chunk_number += 1

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------

    for i, output_i in enumerate(
        final_outputs
    ):

        if len(output_i) != generation_length:
            raise RuntimeError(
                f"Sample {i + 1}: expected "
                f"{generation_length} chord steps, got "
                f"{len(output_i)}"
            )

    # ------------------------------------------------------------------
    # Musical vicinity
    #
    # User specifies vicinity as a fraction of a 16th note.
    #
    # One quarter note = 60 / BPM seconds.
    # One 16th = quarter / 4.
    # ------------------------------------------------------------------

    sixteenth_seconds = (
        60.0
        / bpm
        / 4.0
    )

    vicinity_seconds = (
        vicinity_fraction
        * sixteenth_seconds
    )

    print()
    print(
        "=== FINAL ACCOMPANIMENT FILTER ==="
    )

    print(
        f"16th-note duration: "
        f"{sixteenth_seconds:.6f} s"
    )

    print(
        f"Vicinity: "
        f"{vicinity_fraction} × 16th "
        f"= ±{vicinity_seconds:.6f} s"
    )

    # ------------------------------------------------------------------
    # Artificial prompt duration.
    #
    # The first prompt_length 16th-note positions correspond to the
    # artificial tonic-chord prompt.
    #
    # IMPORTANT:
    # We remove notes beginning in this region, but NEVER shift the
    # remaining notes.
    # ------------------------------------------------------------------

    prompt_duration_seconds = (
        prompt_length
        * sixteenth_seconds
    )

    print(
        f"Prompt cutoff: "
        f"{prompt_duration_seconds:.6f} s"
    )

    # ------------------------------------------------------------------
    # Write final MIDI files.
    # ------------------------------------------------------------------

    for i, output_i in enumerate(
        final_outputs
    ):

        output_file = os.path.join(
            output_dir,
            f"proposal_{i + 1:02d}.mid",
        )

        print()
        print(
            f"Processing proposal "
            f"{i + 1}/{samples}"
        )

        # --------------------------------------------------------------
        # Decode CP sequence.
        #
        # This creates ONE flat list of accompaniment notes.
        # It deliberately does NOT create one instrument per program.
        # --------------------------------------------------------------

        accompaniment_notes = (
            decode_accompaniment_notes(
                (output_i,),
                ratio=(
                    model.compress_ratio_l,
                    model.compress_ratio_r,
                ),
                tempo=bpm,
                min_pitch=45,
            )
        )

        print(
            f"  Decoded accompaniment notes: "
            f"{len(accompaniment_notes)}"
        )

        # --------------------------------------------------------------
        # 1. Remove artificial prompt notes.
        #
        # NO shifting.
        # --------------------------------------------------------------

        accompaniment_notes = (
            remove_initial_accompaniment_notes(
                accompaniment_notes,
                cutoff_time=(
                    prompt_duration_seconds
                ),
            )
        )

        print(
            f"  After prompt removal: "
            f"{len(accompaniment_notes)}"
        )

        # --------------------------------------------------------------
        # 2. Melody-onset vicinity filtering.
        #
        # Only accompaniment onsets sufficiently close to a melody
        # onset survive.
        # --------------------------------------------------------------

        accompaniment_notes = (
            filter_accompaniment_by_melody_vicinity(
                accompaniment_notes,
                original_melody.notes,
                vicinity_seconds=(
                    vicinity_seconds
                ),
            )
        )

        print(
            f"  After melody vicinity filter: "
            f"{len(accompaniment_notes)}"
        )

        # --------------------------------------------------------------
        # 3. Deduplicate:
        #
        # same pitch + same start
        # ->
        # retain longest duration
        # --------------------------------------------------------------

        accompaniment_notes = (
            deduplicate_accompaniment_notes(
                accompaniment_notes
            )
        )

        print(
            f"  After deduplication: "
            f"{len(accompaniment_notes)}"
        )

        # --------------------------------------------------------------
        # 4. Duration extension.
        #
        # A retained accompaniment note must reach at least the next
        # melody onset after its own start.
        #
        # Existing longer notes are preserved.
        # --------------------------------------------------------------

        accompaniment_notes = (
            extend_accompaniment_durations_to_next_melody(
                accompaniment_notes,
                original_melody.notes,
            )
        )

        # --------------------------------------------------------------
        # 5. Remove overlapping notes
        #
        # --------------------------------------------------------------

        accompaniment_notes = remove_overlapping_same_pitch_notes(
            accompaniment_notes
        )
        # --------------------------------------------------------------
        # 5. FINAL VELOCITY NORMALIZATION.
        #
        # No matter what happened upstream:
        # EVERYTHING is 100.
        # --------------------------------------------------------------

        for note in original_melody.notes:
            note.velocity = OUTPUT_VELOCITY

        for note in accompaniment_notes:
            note.velocity = OUTPUT_VELOCITY

        # --------------------------------------------------------------
        # 6. Write exactly two musical tracks.
        # --------------------------------------------------------------

        print(
            f"  Writing: {output_file}"
        )

        write_final_midi(
            input_midi_path=input_midi,
            output_path=output_file,
            melody=copy.deepcopy(
                original_melody
            ),
            accompaniment_notes=(
                accompaniment_notes
            ),
            tempo=bpm,
        )

    print()
    print("DONE.")
    print(
        f"Output directory: {output_dir}"
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Melody → Chord inference for "
            "MIDI Function Alignment."
        )
    )

    parser.add_argument(
        "input",
        help="Input melody MIDI.",
    )

    parser.add_argument(
        "--key",
        required=True,
        help=(
            "Key used for the two-bar tonic prompt, "
            "e.g. 'C minor' or 'Eb major'."
        ),
    )

    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help=(
            "Fallback BPM if the MIDI contains "
            "no tempo map."
        ),
    )

    parser.add_argument(
        "--time-signature",
        default=None,
        help=(
            "Fallback time signature if the MIDI "
            "contains none, e.g. 4/4."
        ),
    )

    parser.add_argument(
        "--model",
        default=(
            "ckpt/mel_to_chord/"
            "cp_transformer_yinyang_v5.1_lora_batch_8_"
            "nottingham_cp8_v2_chord_mel_rev_mask0.0-10-step1."
            "epoch=last.ckpt"
        ),
        help="Melody → chord checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        default="./test/outputs",
        help="Output directory.",
    )

    parser.add_argument(
        "--generation-length",
        type=int,
        default=None,
        help=(
            "Generation length in 16th-note steps. "
            "Defaults to the input MIDI length."
        ),
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    parser.add_argument(
        "--vicinity",
        type=float,
        default=DEFAULT_VICINITY,
        help=(
            "Melody-onset vicinity as a fraction of one "
            "16th note. Default: 0.15."
        ),
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Validate vicinity.
    # ------------------------------------------------------------------

    if args.vicinity < 0:
        parser.error(
            "--vicinity must be >= 0."
        )

    if args.vicinity > 1.0:
        parser.error(
            "--vicinity must not exceed 1.0 "
            "(one full 16th note)."
        )

    # ------------------------------------------------------------------
    # Determine generation length.
    # ------------------------------------------------------------------

    if args.generation_length is None:

        args.generation_length = (
            read_input_length(
                args.input
            )
        )

    # ------------------------------------------------------------------
    # Read metadata BEFORE creating the synthetic prompt MIDI.
    # ------------------------------------------------------------------

    metadata = get_midi_metadata(
        args.input,
        cli_bpm=args.bpm,
        cli_time_signature=(
            args.time_signature
        ),
    )

    bpm = metadata["bpm"]
    numerator = metadata["numerator"]
    denominator = metadata["denominator"]

    print()
    print("=== MIDI METADATA ===")

    print(
        f"Tempo:          "
        f"{bpm:.3f} BPM "
        f"({metadata['bpm_source']})"
    )

    print(
        f"Time signature: "
        f"{numerator}/{denominator} "
        f"({metadata['time_signature_source']})"
    )

    if metadata["tempo_map"]:

        print("Tempo map:")

        for tick, tempo in (
            metadata["tempo_map"]
        ):

            print(
                f"  tick {tick}: "
                f"{tempo:.3f} BPM"
            )

    else:

        print(
            "Tempo map:       none"
        )

    if metadata["time_signature_map"]:

        print(
            "Time-signature map:"
        )

        for (
            tick,
            num,
            den,
        ) in metadata[
            "time_signature_map"
        ]:

            print(
                f"  tick {tick}: "
                f"{num}/{den}"
            )

    else:

        print(
            "Time-signature map: none"
        )

    # ------------------------------------------------------------------
    # Two bars × quarter-note units × four 16th notes.
    # ------------------------------------------------------------------

    quarter_notes_per_bar = (
        numerator
        * (4.0 / denominator)
    )

    prompt_length = int(
        round(
            2
            * quarter_notes_per_bar
            * 4
        )
    )

    print(
        f"Prompt length:   "
        f"{prompt_length} 16th-note steps"
    )

    print(
        f"Vicinity:        "
        f"{args.vicinity} × 16th"
    )

    print(
        f"Velocity:        "
        f"{OUTPUT_VELOCITY}"
    )

    print()

    # ------------------------------------------------------------------
    # Create synthetic two-track MIDI.
    # ------------------------------------------------------------------

    base_name = os.path.splitext(
        os.path.basename(
            args.input
        )
    )[0]

    safe_key = args.key.replace(
        " ",
        "_",
    )

    synthetic_input = os.path.join(
        args.output_dir,
        f"{base_name}_{safe_key}_prompt.mid",
    )

    print(
        "Creating melody + "
        "two-bar chord prompt..."
    )

    print(
        f"Key:             "
        f"{args.key}"
    )

    print(
        f"Chord pitches:   "
        f"{make_tonic_triad(args.key)}"
    )

    print(
        f"Prompt MIDI:     "
        f"{synthetic_input}"
    )

    create_melody_chord_input(
        args.input,
        synthetic_input,
        args.key,
    )

    # ------------------------------------------------------------------
    # Load downstream model.
    # ------------------------------------------------------------------

    print()
    print("Loading model...")
    print(args.model)

    model = (
        RoformerYinyang.load_from_checkpoint(
            args.model,
            strict=False,
        )
    )

    model.save_name = os.path.basename(
        args.model
    )

    model.cuda()
    model.eval()

    # ------------------------------------------------------------------
    # Generate.
    # ------------------------------------------------------------------

    generate(
        model=model,
        input_midi=synthetic_input,
        original_input_midi=args.input,
        output_dir=args.output_dir,
        bpm=bpm,
        prompt_length=prompt_length,
        generation_length=(
            args.generation_length
        ),
        temperature=args.temperature,
        samples=args.samples,
        seed=args.seed,
        vicinity_fraction=args.vicinity,
    )


if __name__ == "__main__":
    main()
