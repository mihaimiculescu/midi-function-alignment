#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4 -> CP16 preprocessing

Converts the clean MIDI corpus produced by stage4.py into the exact
tensor representation expected by the existing CP transformer / Yinyang
training code.

IMPORTANT
---------
This script intentionally does NOT use:
    UglyMIDI
    pretty_midi
    the old preprocess_midi()
    the old quantization filter

Stage 4 MIDI semantics are authoritative.

INPUT
-----
Dataset/LAMDselection/selection_stage4/

    1/*.mid
    2/*.mid
    ...
    f/*.mid

Each Stage 4 MIDI contains exactly two musical tracks:

    Track 0 = melody
    Track 1 = accompaniment

OUTPUT
------
data/<dataset_name>.pt
data/<dataset_name>.length.pt
data/<dataset_name>.pitch_shift_range.pt
data/<dataset_name>.txt

The .pt tensor has shape:

    [total_CP_timesteps, 64]

because:

    2 tracks
    x 8 notes
    x 4 CP fields
    = 64 values per timestep

Each CP note contains:

    [program, pitch, duration, velocity]

Padding:

    program = 255
    pitch   = 255
    duration= 255
    velocity= 255

EOS:

    program = 254

The existing CP transformer converts these fields into its tokenizer
representation. See cp_transformer.py.

CP16
----
beat_div = 4

Therefore:

    1 quarter-note beat = 4 CP positions
    1 CP position       = 1/16 note

The duration templates are exactly those used by the original
CP preprocessing:

    1, 2, 3, 4, 6, 8, 12, ... 4096

PROGRAMS
--------
Because Stage 4 deliberately contains only two semantic roles rather
than original General MIDI instruments, fixed programs are assigned:

    Track 0 melody         -> program 64
    Track 1 accompaniment  -> program 0

This matches the program convention used by the existing melody/chord
Yinyang inference examples.

PITCH SHIFT
-----------
The original preprocessing records the possible whole-semitone pitch
shift range:

    minimum shift = -minimum_pitch
    maximum shift = 127 - maximum_pitch

The existing OverlapFramedDataset subsequently clamps this to:

    [-5, +6]

for training augmentation.

NO SONG IS SPLIT HERE
---------------------
Each MIDI is converted as one complete song.

OverlapFramedDataset is responsible later for:

    384-step windows
    192-step stride
    final end-aligned window
    shuffling
    batching

This script must therefore preserve song boundaries through
length.pt.

USAGE
-----
Default:

    python preprocess_stage4.py

Explicit dataset name:

    python preprocess_stage4.py --dataset-name lamd_stage4_cp8_v1

Optional:

    python preprocess_stage4.py \
        --input Dataset/LAMDselection/selection_stage4 \
        --output data \
        --dataset-name lamd_stage4_cp8_v1

"""

import argparse
import gc
import json
import math
import os
import sys
from pathlib import Path

import torch


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_INPUT_ROOT = Path(
    "Dataset/LAMDselection/selection_stage4"
)

DEFAULT_OUTPUT_ROOT = Path("data")

DEFAULT_DATASET_NAME = "lamd_stage4_cp8_v1"

# Exactly the convention used by the original CP preprocessing.
BEAT_DIV = 4

# Stage 4 always has:
#
#     track 0 = melody
#     track 1 = accompaniment
#
TRACK_COUNT = 2

MAX_POLYPHONY = 8

# CP fields:
#
#     program
#     pitch
#     duration
#     velocity
#
FIELDS_PER_NOTE = 4

SUBSEQ_LENGTH = (
    TRACK_COUNT
    * MAX_POLYPHONY
    * FIELDS_PER_NOTE
)

# Fixed semantic programs.
#
# These are deliberately not taken from MIDI program-change messages.
# Stage 4 stripped that information because its two tracks represent
# musical functions, not original instrument identities.
MELODY_PROGRAM = 64
ACCOMPANIMENT_PROGRAM = 0

PROGRAMS = (
    MELODY_PROGRAM,
    ACCOMPANIMENT_PROGRAM,
)

# MIDI note range.
MIN_PITCH = 0
MAX_PITCH = 127

# MIDI percussion channel.
GM_DRUM_CHANNEL = 9

# Exact duration templates from preprocess_large_midi_dataset.py.
DURATION_TEMPLATES = (
    1,
    2,
    3,
    4,
    6,
    8,
    12,
    16,
    24,
    32,
    48,
    64,
    96,
    128,
    192,
    256,
    384,
    512,
    768,
    1024,
    1536,
    2048,
    3072,
    4096,
)

# Token values used by the original CP representation.
PAD_VALUE = 255
EOS_VALUE = 254


# ============================================================================
# IMPORT
# ============================================================================

def import_mido():
    try:
        import mido
        return mido
    except Exception as exc:
        print()
        print("=" * 78)
        print("ERROR: Could not import mido.")
        print("=" * 78)
        print()
        print(f"{type(exc).__name__}: {exc}")
        print()
        print("Install it with:")
        print("    pip install mido")
        print()
        raise


# ============================================================================
# MIDI MESSAGE HELPERS
# ============================================================================

def is_note_on(message):
    """
    True for a genuine MIDI note-on.

    note_on velocity=0 is handled as note-off.
    """
    return (
        not message.is_meta
        and message.type == "note_on"
        and int(message.velocity) > 0
    )


def is_note_off(message):
    """
    True for note-off or note_on velocity=0.
    """
    if message.is_meta:
        return False

    if message.type == "note_off":
        return True

    return (
        message.type == "note_on"
        and int(message.velocity) == 0
    )


# ============================================================================
# NOTE EXTRACTION
# ============================================================================

def extract_notes(track):
    """
    Extract complete notes from one MIDI track.

    MIDI delta times are converted to absolute ticks.

    Notes are paired per:

        channel + pitch

    using FIFO pairing, matching the approach used by Stage 4 itself.
    """

    active = {}
    notes = []

    absolute_tick = 0

    for message in track:

        absolute_tick += int(message.time)

        if not (
            is_note_on(message)
            or is_note_off(message)
        ):
            continue

        channel = int(
            getattr(message, "channel", 0)
        )

        pitch = int(message.note)

        key = (
            channel,
            pitch,
        )

        if is_note_on(message):

            active.setdefault(
                key,
                []
            ).append(
                (
                    absolute_tick,
                    int(message.velocity),
                )
            )

        elif is_note_off(message):

            queue = active.get(key)

            if not queue:
                continue

            start_tick, velocity = queue.pop(0)

            if absolute_tick < start_tick:
                continue

            notes.append(
                {
                    "start": int(start_tick),
                    "end": int(absolute_tick),
                    "pitch": pitch,
                    "velocity": velocity,
                    "channel": channel,
                }
            )

    return notes


# ============================================================================
# TICK -> CP POSITION
# ============================================================================

def ticks_to_cp_position(ticks, ticks_per_beat):
    """
    Convert MIDI ticks into CP16 positions.

    beat_div = 4 means:

        ticks_per_CP_position = ticks_per_beat / 4

    The MIDI is therefore quantized onto a 16th-note grid by rounding
    to the nearest CP position.

    No tempo information is consulted.

    This is intentional.

    MIDI ticks already represent musical position relative to the MIDI
    quarter-note beat. Tempo affects real time, not the symbolic beat grid.
    """

    if ticks_per_beat <= 0:
        raise ValueError(
            f"Invalid ticks_per_beat: {ticks_per_beat}"
        )

    cp_ticks = (
        float(ticks_per_beat)
        / float(BEAT_DIV)
    )

    return int(
        math.floor(
            (float(ticks) / cp_ticks)
            + 0.5
        )
    )


# ============================================================================
# DURATION ENCODING
# ============================================================================

def duration_to_template(duration):
    """
    Map a CP duration to the same duration-template vocabulary used by
    the original preprocessing.

    The original implementation uses:

        searchsorted(boundaries, duration)

    where each boundary is the midpoint between adjacent templates.

    We reproduce that behavior without NumPy.
    """

    if duration <= 0:
        duration = 1

    templates = DURATION_TEMPLATES

    # Equivalent to np.searchsorted(boundaries, duration).
    for index in range(len(templates) - 1):

        left = templates[index]
        right = templates[index + 1]

        boundary = (
            float(left + right)
            / 2.0
        )

        if duration < boundary:
            return left

    return templates[-1]


# ============================================================================
# NOTE CONVERSION
# ============================================================================

def convert_notes_to_cp(
    notes,
    ticks_per_beat,
    program,
):
    """
    Convert MIDI notes into CP note records.

    Returns:

        [
            {
                "start": CP position,
                "pitch": MIDI pitch,
                "duration": CP duration,
                "velocity": MIDI velocity,
                "program": CP program,
            },
            ...
        ]
    """

    converted = []

    for note in notes:

        # Stage 4 already removes percussion, but retain this guard so
        # this preprocessing cannot accidentally introduce drums if a
        # malformed Stage 4 file is encountered.
        if int(note["channel"]) == GM_DRUM_CHANNEL:
            continue

        pitch = int(note["pitch"])

        if not (
            MIN_PITCH
            <= pitch
            <= MAX_PITCH
        ):
            continue

        start = ticks_to_cp_position(
            note["start"],
            ticks_per_beat,
        )

        end = ticks_to_cp_position(
            note["end"],
            ticks_per_beat,
        )

        # A note whose two endpoints collapse onto the same CP position
        # cannot have a zero-length CP duration.
        if end <= start:
            end = start + 1

        duration = end - start

        duration = duration_to_template(
            duration
        )

        velocity = int(note["velocity"])

        velocity = max(
            0,
            min(
                127,
                velocity,
            )
        )

        converted.append(
            {
                "start": int(start),
                "pitch": int(pitch),
                "duration": int(duration),
                "velocity": int(velocity),
                "program": int(program),
            }
        )

    return converted


# ============================================================================
# CP SONG CONSTRUCTION
# ============================================================================

def build_song_tensor(
    melody_notes,
    accompaniment_notes,
    ticks_per_beat,
):
    """
    Build one complete Stage 4 song tensor.

    Shape:

        [song_length, 64]

    where:

        2 tracks
        x 8 notes
        x 4 fields
        = 64
    """

    melody = convert_notes_to_cp(
        melody_notes,
        ticks_per_beat,
        MELODY_PROGRAM,
    )

    accompaniment = convert_notes_to_cp(
        accompaniment_notes,
        ticks_per_beat,
        ACCOMPANIMENT_PROGRAM,
    )

    track_notes = (
        melody,
        accompaniment,
    )

    # --------------------------------------------------------------
    # Determine complete song length.
    #
    # We include the end of the longest note so that the song metadata
    # describes the complete Stage 4 song rather than merely its last
    # onset.
    # --------------------------------------------------------------

    song_length = 0

    for notes in track_notes:

        for note in notes:

            note_end = (
                note["start"]
                + note["duration"]
            )

            song_length = max(
                song_length,
                int(note_end),
            )

    if song_length <= 0:
        return None

    # --------------------------------------------------------------
    # Allocate CP tensor.
    #
    # Each slot initially contains PAD.
    # --------------------------------------------------------------

    song = torch.full(
        (
            song_length,
            TRACK_COUNT
            * MAX_POLYPHONY
            * FIELDS_PER_NOTE,
        ),
        PAD_VALUE,
        dtype=torch.long,
    )

    # --------------------------------------------------------------
    # Fill each track.
    # --------------------------------------------------------------

    for track_index, notes in enumerate(
        track_notes
    ):

        # Group notes by onset position.
        by_start = {}

        for note in notes:

            by_start.setdefault(
                int(note["start"]),
                []
            ).append(note)

        for start, onset_notes in by_start.items():

            if start < 0 or start >= song_length:
                continue

            # ------------------------------------------------------
            # Reproduce the ordering used by the original CP
            # preprocessing:
            #
            #     velocity
            #     program
            #     pitch
            #     duration
            #
            # np.lexsort((duration, pitch, program, velocity))
            # makes velocity the primary key.
            # ------------------------------------------------------

            onset_notes.sort(
                key=lambda note: (
                    note["velocity"],
                    note["program"],
                    note["pitch"],
                    note["duration"],
                )
            )

            # ------------------------------------------------------
            # CP representation permits at most 8 simultaneous
            # onset notes.
            #
            # The original preprocessing silently stops accepting
            # additional notes once max_polyphony is reached.
            # ------------------------------------------------------

            onset_notes = onset_notes[
                :MAX_POLYPHONY
            ]

            for slot, note in enumerate(
                onset_notes
            ):

                base = (
                    track_index
                    * MAX_POLYPHONY
                    * FIELDS_PER_NOTE
                    +
                    slot
                    * FIELDS_PER_NOTE
                )

                song[start, base + 0] = (
                    note["program"]
                )

                song[start, base + 1] = (
                    note["pitch"]
                )

                song[start, base + 2] = (
                    note["duration"]
                )

                song[start, base + 3] = (
                    note["velocity"]
                )

            # ------------------------------------------------------
            # EOS is placed immediately after the final note if
            # polyphony is not full.
            #
            # This exactly mirrors the original CP representation:
            #
            #     rolls[i, polyphony_counts[i], 0] = 254
            # ------------------------------------------------------

            note_count = len(onset_notes)

            if note_count < MAX_POLYPHONY:

                eos_base = (
                    track_index
                    * MAX_POLYPHONY
                    * FIELDS_PER_NOTE
                    +
                    note_count
                    * FIELDS_PER_NOTE
                )

                song[
                    start,
                    eos_base
                ] = EOS_VALUE

    return song


# ============================================================================
# ONE MIDI FILE
# ============================================================================

def preprocess_one_midi(
    midi_path,
):
    """
    Convert one Stage 4 MIDI into:

        tensor
        pitch_shift_range
        metadata
    """

    mido = import_mido()

    try:
        midi = mido.MidiFile(
            str(midi_path)
        )
    except Exception as exc:
        return {
            "ok": False,
            "path": str(midi_path),
            "reason": (
                "midi_read_error:"
                + type(exc).__name__
            ),
            "error": str(exc),
        }

    try:

        # Stage 4 is required to have exactly two musical tracks.
        if len(midi.tracks) != 2:

            return {
                "ok": False,
                "path": str(midi_path),
                "reason": "expected_exactly_two_tracks",
                "track_count": len(midi.tracks),
            }

        ticks_per_beat = int(
            midi.ticks_per_beat
        )

        if ticks_per_beat <= 0:

            return {
                "ok": False,
                "path": str(midi_path),
                "reason": "invalid_ticks_per_beat",
            }

        melody_notes = extract_notes(
            midi.tracks[0]
        )

        accompaniment_notes = extract_notes(
            midi.tracks[1]
        )

        song = build_song_tensor(
            melody_notes,
            accompaniment_notes,
            ticks_per_beat,
        )

        if song is None:

            return {
                "ok": False,
                "path": str(midi_path),
                "reason": "no_usable_notes",
            }

        # ----------------------------------------------------------
        # Calculate pitch-shift range exactly from the musical
        # material represented in the tensor.
        #
        # This corresponds to the original preprocessing's:
        #
        #     pitch_shift_max = 127 - max_pitch
        #     pitch_shift_min = -min_pitch
        #
        # The existing OverlapFramedDataset later clamps this range
        # to [-5, +6].
        # ----------------------------------------------------------

        pitches = []

        for notes in (
            melody_notes,
            accompaniment_notes,
        ):

            for note in notes:

                if (
                    int(note["channel"])
                    == GM_DRUM_CHANNEL
                ):
                    continue

                pitch = int(
                    note["pitch"]
                )

                if (
                    MIN_PITCH
                    <= pitch
                    <= MAX_PITCH
                ):
                    pitches.append(
                        pitch
                    )

        if not pitches:

            return {
                "ok": False,
                "path": str(midi_path),
                "reason": "no_non_drum_pitches",
            }

        minimum_pitch = min(
            pitches
        )

        maximum_pitch = max(
            pitches
        )

        pitch_shift_min = (
            -minimum_pitch
        )

        pitch_shift_max = (
            127
            - maximum_pitch
        )

        return {
            "ok": True,
            "path": str(midi_path),
            "tensor": song,
            "length": int(song.shape[0]),
            "pitch_shift_min": int(
                pitch_shift_min
            ),
            "pitch_shift_max": int(
                pitch_shift_max
            ),
            "melody_notes": len(
                melody_notes
            ),
            "accompaniment_notes": len(
                accompaniment_notes
            ),
            "ticks_per_beat": ticks_per_beat,
        }

    except Exception as exc:

        return {
            "ok": False,
            "path": str(midi_path),
            "reason": (
                "preprocessing_error:"
                + type(exc).__name__
            ),
            "error": str(exc),
        }

    finally:

        midi = None
        gc.collect()


# ============================================================================
# FILE DISCOVERY
# ============================================================================

def find_stage4_midis(
    input_root,
):
    """
    Discover Stage 4 MIDI files.

    Only the Stage 4 hexadecimal subdirectories are considered:

        1..9
        a..f

    The relative ordering is deterministic.
    """

    input_root = Path(
        input_root
    )

    if not input_root.is_dir():

        raise FileNotFoundError(
            "Stage 4 directory does not exist:\n"
            f"{input_root}"
        )

    files = []

    for subdir in "123456789abcdef":

        directory = (
            input_root
            / subdir
        )

        if not directory.is_dir():
            continue

        for path in sorted(
            directory.iterdir(),
            key=lambda p: p.name.lower()
        ):

            if not path.is_file():
                continue

            if path.suffix.lower() != ".mid":
                continue

            files.append(path)

    return files


# ============================================================================
# ATOMIC TORCH SAVE
# ============================================================================

def atomic_torch_save(
    obj,
    path,
):
    """
    Save with a temporary file and atomic rename.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:

        torch.save(
            obj,
            temporary,
        )

        os.replace(
            temporary,
            path,
        )

    finally:

        if temporary.exists():

            try:
                temporary.unlink()
            except OSError:
                pass


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Preprocess Stage 4 MIDI files into "
            "the CP16 dataset format used by "
            "Yinyang/LoRA training."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
        help=(
            "Stage 4 MIDI root "
            f"(default: {DEFAULT_INPUT_ROOT})"
        ),
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help=(
            "Output dataset directory "
            f"(default: {DEFAULT_OUTPUT_ROOT})"
        ),
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help=(
            "Dataset basename "
            f"(default: {DEFAULT_DATASET_NAME})"
        ),
    )

    args = parser.parse_args()

    input_root = args.input
    output_root = args.output
    dataset_name = args.dataset_name

    if not dataset_name:
        raise ValueError(
            "dataset-name cannot be empty."
        )

    # --------------------------------------------------------------
    # Banner
    # --------------------------------------------------------------

    print()
    print("=" * 78)
    print("STAGE 4 -> CP16 PREPROCESSING")
    print("=" * 78)
    print()
    print(f"Input root       : {input_root}")
    print(f"Output root      : {output_root}")
    print(f"Dataset name     : {dataset_name}")
    print()
    print(f"beat_div         : {BEAT_DIV}")
    print(f"max_polyphony    : {MAX_POLYPHONY}")
    print(f"tracks           : {TRACK_COUNT}")
    print(f"fields/note      : {FIELDS_PER_NOTE}")
    print(f"tensor width     : {SUBSEQ_LENGTH}")
    print()
    print("Track 0 program  : "
          f"{MELODY_PROGRAM} (melody)")
    print("Track 1 program  : "
          f"{ACCOMPANIMENT_PROGRAM} (accompaniment)")
    print()
    print("UglyMIDI         : NO")
    print("Old quantization : NO")
    print("Stage 4 MIDI     : AUTHORITATIVE")
    print()

    # --------------------------------------------------------------
    # Discover files
    # --------------------------------------------------------------

    midi_files = find_stage4_midis(
        input_root
    )

    total_files = len(
        midi_files
    )

    print(
        f"Found {total_files:,} Stage 4 MIDI files."
    )
    print()

    if total_files == 0:

        raise RuntimeError(
            "No Stage 4 MIDI files were found."
        )

    # --------------------------------------------------------------
    # Process files.
    #
    # We deliberately process sequentially.
    #
    # The expensive object is the complete song tensor. Keeping
    # dozens of such tensors alive simultaneously provides no useful
    # benefit and can create unnecessary RAM pressure.
    # --------------------------------------------------------------

    tensors = []
    lengths = []
    pitch_ranges = []
    manifest = []

    rejected = []

    for index, midi_path in enumerate(
        midi_files,
        start=1,
    ):

        result = preprocess_one_midi(
            midi_path
        )

        if not result["ok"]:

            rejected.append(
                result
            )

            print(
                f"[{index:>7,}/{total_files:,}] "
                f"REJECT "
                f"{midi_path} "
                f"({result['reason']})"
            )

            if result.get("error"):
                print(
                    f"    ERROR: {result['error']}"
                )

            continue

        tensor = result["tensor"]

        # Sanity check.
        if tensor.ndim != 2:

            raise RuntimeError(
                "Internal tensor rank error for:\n"
                f"{midi_path}\n"
                f"shape={tuple(tensor.shape)}"
            )

        if tensor.shape[1] != SUBSEQ_LENGTH:

            raise RuntimeError(
                "Internal tensor width error for:\n"
                f"{midi_path}\n"
                f"shape={tuple(tensor.shape)}\n"
                f"expected width={SUBSEQ_LENGTH}"
            )

        tensors.append(
            tensor
        )

        lengths.append(
            int(tensor.shape[0])
        )

        pitch_ranges.append(
            [
                int(
                    result["pitch_shift_min"]
                ),
                int(
                    result["pitch_shift_max"]
                ),
            ]
        )

        relative_path = str(
            midi_path.relative_to(
                input_root
            )
        )

        manifest.append(
            {
                "index": len(manifest),
                "path": relative_path,
                "length": int(
                    tensor.shape[0]
                ),
                "pitch_shift_min": int(
                    result[
                        "pitch_shift_min"
                    ]
                ),
                "pitch_shift_max": int(
                    result[
                        "pitch_shift_max"
                    ]
                ),
                "melody_notes": int(
                    result[
                        "melody_notes"
                    ]
                ),
                "accompaniment_notes": int(
                    result[
                        "accompaniment_notes"
                    ]
                ),
                "ticks_per_beat": int(
                    result[
                        "ticks_per_beat"
                    ]
                ),
            }
        )

        if (
            index == 1
            or index % 100 == 0
            or index == total_files
        ):

            total_steps = sum(
                lengths
            )

            print(
                f"[{index:>7,}/{total_files:,}] "
                f"OK "
                f"survivors={len(tensors):,} "
                f"rejected={len(rejected):,} "
                f"CP steps={total_steps:,}"
            )

    # --------------------------------------------------------------
    # Verify that something survived.
    # --------------------------------------------------------------

    if not tensors:

        raise RuntimeError(
            "No Stage 4 MIDI files could be converted."
        )

    # --------------------------------------------------------------
    # Concatenate complete songs.
    #
    # IMPORTANT:
    #
    # Song boundaries are retained separately in length.pt.
    #
    # OverlapFramedDataset reconstructs each song's absolute range
    # from these lengths and therefore never creates a window crossing
    # from one song into another.
    # --------------------------------------------------------------

    print()
    print(
        "Concatenating CP tensors..."
    )

    data = torch.cat(
        tensors,
        dim=0,
    )

    lengths_tensor = torch.tensor(
        lengths,
        dtype=torch.long,
    )

    pitch_shift_tensor = torch.tensor(
        pitch_ranges,
        dtype=torch.int8,
    )

    # --------------------------------------------------------------
    # Final consistency checks.
    # --------------------------------------------------------------

    if data.dtype != torch.long:

        raise RuntimeError(
            f"Unexpected tensor dtype: "
            f"{data.dtype}"
        )

    if data.ndim != 2:

        raise RuntimeError(
            f"Unexpected tensor shape: "
            f"{tuple(data.shape)}"
        )

    if data.shape[1] != SUBSEQ_LENGTH:

        raise RuntimeError(
            "Final tensor width mismatch: "
            f"{data.shape[1]} != {SUBSEQ_LENGTH}"
        )

    if int(
        lengths_tensor.sum().item()
    ) != int(
        data.shape[0]
    ):

        raise RuntimeError(
            "length.pt does not describe "
            "the final tensor correctly."
        )

    if pitch_shift_tensor.shape != (
        len(tensors),
        2,
    ):

        raise RuntimeError(
            "pitch_shift_range.pt shape mismatch: "
            f"{tuple(pitch_shift_tensor.shape)}"
        )

    # --------------------------------------------------------------
    # Output paths
    # --------------------------------------------------------------

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    data_path = (
        output_root
        / f"{dataset_name}.pt"
    )

    length_path = (
        output_root
        / f"{dataset_name}.length.pt"
    )

    pitch_path = (
        output_root
        / f"{dataset_name}.pitch_shift_range.pt"
    )

    manifest_path = (
        output_root
        / f"{dataset_name}.txt"
    )

    metadata_path = (
        output_root
        / f"{dataset_name}.json"
    )

    # --------------------------------------------------------------
    # Save main tensors
    # --------------------------------------------------------------

    print()
    print(
        "Saving dataset..."
    )

    atomic_torch_save(
        data,
        data_path,
    )

    atomic_torch_save(
        lengths_tensor,
        length_path,
    )

    atomic_torch_save(
        pitch_shift_tensor,
        pitch_path,
    )

    # --------------------------------------------------------------
    # Save manifest.
    #
    # Format deliberately remains compatible with the original
    # dataset creator's .txt convention:
    #
    #     index<TAB>relative_path
    # --------------------------------------------------------------

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as fh:

        for record in manifest:

            fh.write(
                str(
                    record["index"]
                )
                + "\t"
                + str(
                    record["path"]
                )
                + "\n"
            )

    # --------------------------------------------------------------
    # Save human-readable metadata.
    # This is additional information and is not consumed by the
    # existing training code.
    # --------------------------------------------------------------

    metadata = {
        "dataset": (
            "Los-Angeles-MIDI-Dataset-"
            "Ver-4-0-CC-BY-NC-SA"
        ),
        "source": str(
            input_root
        ),
        "preprocessing": (
            "Stage 4 specific CP16"
        ),
        "uses_uglymidi": False,
        "uses_old_quantization_filter": False,
        "beat_div": BEAT_DIV,
        "max_polyphony": MAX_POLYPHONY,
        "track_count": TRACK_COUNT,
        "fields_per_note": FIELDS_PER_NOTE,
        "tensor_width": SUBSEQ_LENGTH,
        "melody_track": 0,
        "accompaniment_track": 1,
        "melody_program": MELODY_PROGRAM,
        "accompaniment_program": (
            ACCOMPANIMENT_PROGRAM
        ),
        "pad_value": PAD_VALUE,
        "eos_value": EOS_VALUE,
        "duration_templates": list(
            DURATION_TEMPLATES
        ),
        "file_count_input": total_files,
        "file_count_survived": len(
            tensors
        ),
        "file_count_rejected": len(
            rejected
        ),
        "total_cp_steps": int(
            data.shape[0]
        ),
        "dtype": str(
            data.dtype
        ),
        "shape": [
            int(data.shape[0]),
            int(data.shape[1]),
        ],
        "songs": manifest,
        "rejections": rejected,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as fh:

        json.dump(
            metadata,
            fh,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------------------
    # Release large objects before reporting memory-sensitive
    # statistics.
    # --------------------------------------------------------------

    tensor_count = len(
        tensors
    )

    del tensors
    gc.collect()

    # --------------------------------------------------------------
    # Final report
    # --------------------------------------------------------------

    print()
    print("=" * 78)
    print("STAGE 4 -> CP16 COMPLETE")
    print("=" * 78)
    print()
    print(
        f"Input MIDI files       : {total_files:,}"
    )
    print(
        f"Converted songs        : {tensor_count:,}"
    )
    print(
        f"Rejected               : {len(rejected):,}"
    )
    print(
        f"Total CP timesteps     : "
        f"{data.shape[0]:,}"
    )
    print(
        f"Tensor shape           : "
        f"{tuple(data.shape)}"
    )
    print(
        f"Tensor dtype           : "
        f"{data.dtype}"
    )
    print()
    print(
        f"Main dataset           : {data_path}"
    )
    print(
        f"Song lengths           : {length_path}"
    )
    print(
        f"Pitch shift ranges     : {pitch_path}"
    )
    print(
        f"Manifest               : {manifest_path}"
    )
    print(
        f"Metadata               : {metadata_path}"
    )
    print()
    print(
        "The dataset is ready for "
        "OverlapFramedDataset."
    )
    print()


if __name__ == "__main__":
    main()
    