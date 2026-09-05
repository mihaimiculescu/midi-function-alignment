#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4 -> CP16 preprocessing
PARALLEL / MEMORY-SAFE VERSION

Converts the clean MIDI corpus produced by stage4.py into the exact
CP tensor representation expected by the existing CP Transformer /
Yinyang training pipeline.

IMPORTANT:
    This script does NOT use UglyMIDI.
    This script does NOT use pretty_midi.
    This script does NOT use the old LA quantization filter.

Stage 4 MIDI files are authoritative.

CP representation:

    2 tracks
    x 8 notes
    x 4 fields

    = 64 values per CP timestep

Each note:

    [program, pitch, duration_index, velocity]

IMPORTANT DURATION REPRESENTATION
---------------------------------

The CP Transformer does NOT store the actual duration template value.

It stores the INDEX into DURATION_TEMPLATES.

Therefore:

    template value       stored index

        1                    0
        2                    1
        3                    2
        4                    3
        6                    4
        ...
        256                 15
        384                 16
        512                 17
        ...
        4096                23

This is required by cp_transformer.py, whose tokenizer has exactly
24 duration categories.

The tensor is torch.long because the representation also uses values
such as 255 padding and 254 EOS, and because keeping the representation
in int64 avoids the uint8 overflow that occurs with actual duration
values above 255.

The ACTUAL quantized duration in CP steps is used only to determine
the song's temporal extent. The duration FIELD stored in the tensor
is the duration-template INDEX.

Input:

    Dataset/LAMDselection/selection_stage4/
        1/*.mid
        2/*.mid
        ...
        f/*.mid

Each Stage 4 MIDI must contain exactly two tracks:

    Track 0 = melody
    Track 1 = accompaniment

Output:

    data/<dataset_name>.pt
    data/<dataset_name>.length.pt
    data/<dataset_name>.pitch_shift_range.pt
    data/<dataset_name>.txt
    data/<dataset_name>.json

Default dataset name:

    lamd_stage4_cp8_v1

Default workers:

    16

Usage:

    python preprocess_stage4.py --workers 16

A previous staging directory is deliberately NOT reused. Delete it
before restarting an interrupted run.
"""

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import time
from pathlib import Path


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_INPUT_ROOT = Path(
    "Dataset/LAMDselection/selection_stage4"
)

DEFAULT_OUTPUT_ROOT = Path(
    "data"
)

DEFAULT_DATASET_NAME = (
    "lamd_stage4_cp8_v1"
)

# 16 is intentionally chosen as the practical default.
DEFAULT_WORKERS = 16

DEFAULT_MAX_TASKS_PER_CHILD = 100

BEAT_DIV = 4

TRACK_COUNT = 2

MAX_POLYPHONY = 8

FIELDS_PER_NOTE = 4

SUBSEQ_LENGTH = (
    TRACK_COUNT
    * MAX_POLYPHONY
    * FIELDS_PER_NOTE
)

MELODY_PROGRAM = 64

ACCOMPANIMENT_PROGRAM = 0

GM_DRUM_CHANNEL = 9

MIN_PITCH = 0

MAX_PITCH = 127


# These are the exact CP duration templates used by the original
# preprocess_large_midi_dataset.py.
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

# CP representation values.
PAD_VALUE = 255
EOS_VALUE = 254


# ============================================================================
# DURATION BOUNDARIES
# ============================================================================

# The original code computes:
#
#   duration_boundaries =
#       (DURATION_TEMPLATES[1:] + DURATION_TEMPLATES[:-1]) / 2
#
# and then:
#
#   np.searchsorted(duration_boundaries, duration)
#
# which returns the TEMPLATE INDEX.
#
# We calculate the same result without numpy.

DURATION_BOUNDARIES = tuple(
    (
        DURATION_TEMPLATES[i]
        + DURATION_TEMPLATES[i + 1]
    )
    / 2.0
    for i in range(
        len(DURATION_TEMPLATES) - 1
    )
)


def duration_to_template_index(duration):
    """
    Convert an actual CP duration into the exact duration-template
    INDEX expected by the CP Transformer.

    Returns:

        0 .. 23

    This deliberately does NOT return the duration value.
    """

    duration = int(duration)

    if duration <= 0:
        duration = 1

    for index, boundary in enumerate(
        DURATION_BOUNDARIES
    ):
        if duration < boundary:
            return index

    return len(DURATION_TEMPLATES) - 1


def duration_template_value(index):
    """
    Return the actual CP duration represented by a duration-template
    index.
    """

    index = int(index)

    if not (
        0 <= index < len(DURATION_TEMPLATES)
    ):
        raise ValueError(
            f"Invalid duration template index: {index}"
        )

    return DURATION_TEMPLATES[index]


# ============================================================================
# WORKER INITIALIZATION
# ============================================================================

def worker_init():
    """
    Prevent multiprocessing workers from multiplying CPU thread pools.
    """

    os.environ.setdefault(
        "OMP_NUM_THREADS",
        "1",
    )

    os.environ.setdefault(
        "MKL_NUM_THREADS",
        "1",
    )

    os.environ.setdefault(
        "OPENBLAS_NUM_THREADS",
        "1",
    )

    try:
        import torch

        torch.set_num_threads(1)

        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    except Exception:
        pass


# ============================================================================
# MIDO
# ============================================================================

def import_mido():
    import mido
    return mido


# ============================================================================
# MIDI MESSAGE HELPERS
# ============================================================================

def is_note_on(message):
    return (
        not message.is_meta
        and message.type == "note_on"
        and int(message.velocity) > 0
    )


def is_note_off(message):
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
    Convert delta-time MIDI messages into complete notes.

    Notes are paired FIFO by:

        channel + pitch
    """

    active = {}

    notes = []

    absolute_tick = 0

    for message in track:

        absolute_tick += int(
            message.time
        )

        if not (
            is_note_on(message)
            or is_note_off(message)
        ):
            continue

        channel = int(
            getattr(
                message,
                "channel",
                0,
            )
        )

        pitch = int(
            message.note
        )

        key = (
            channel,
            pitch,
        )

        if is_note_on(message):

            active.setdefault(
                key,
                [],
            ).append(
                (
                    absolute_tick,
                    int(message.velocity),
                )
            )

        else:

            queue = active.get(
                key
            )

            if not queue:
                continue

            start_tick, velocity = (
                queue.pop(0)
            )

            if absolute_tick < start_tick:
                continue

            notes.append(
                {
                    "start": int(
                        start_tick
                    ),
                    "end": int(
                        absolute_tick
                    ),
                    "pitch": int(
                        pitch
                    ),
                    "velocity": int(
                        velocity
                    ),
                    "channel": int(
                        channel
                    ),
                }
            )

    return notes


# ============================================================================
# TICKS -> CP POSITION
# ============================================================================

def ticks_to_cp_position(
    ticks,
    ticks_per_beat,
):
    """
    Convert MIDI ticks to CP16 positions.

    beat_div = 4

    Therefore:

        1 quarter-note beat = 4 CP positions
        1 CP position       = 1/16 note
    """

    if ticks_per_beat <= 0:
        raise ValueError(
            f"Invalid ticks_per_beat: "
            f"{ticks_per_beat}"
        )

    cp_ticks = (
        float(ticks_per_beat)
        / float(BEAT_DIV)
    )

    return int(
        math.floor(
            (
                float(ticks)
                / cp_ticks
            )
            + 0.5
        )
    )


# ============================================================================
# NOTE -> CP
# ============================================================================

def convert_notes_to_cp(
    notes,
    ticks_per_beat,
    program,
):
    """
    Convert MIDI notes to CP notes.

    Each returned note contains:

        start
        pitch
        duration_index
        duration_steps
        velocity
        program
    """

    converted = []

    for note in notes:

        if (
            int(note["channel"])
            == GM_DRUM_CHANNEL
        ):
            continue

        pitch = int(
            note["pitch"]
        )

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

        if end <= start:
            end = start + 1

        duration_steps = (
            end - start
        )

        duration_index = (
            duration_to_template_index(
                duration_steps
            )
        )

        velocity = max(
            0,
            min(
                127,
                int(
                    note["velocity"]
                ),
            ),
        )

        converted.append(
            {
                "start": int(
                    start
                ),
                "pitch": int(
                    pitch
                ),

                # IMPORTANT:
                # Store the TEMPLATE INDEX,
                # not the template value.
                "duration_index": int(
                    duration_index
                ),

                # Keep actual quantized duration
                # separately for temporal extent.
                "duration_steps": int(
                    duration_steps
                ),

                "velocity": int(
                    velocity
                ),

                "program": int(
                    program
                ),
            }
        )

    return converted


# ============================================================================
# BUILD SONG TENSOR
# ============================================================================

def build_song_tensor(
    melody_notes,
    accompaniment_notes,
    ticks_per_beat,
):
    """
    Build one complete CP16 song.

    Result:

        [song_length, 64]

    dtype:

        torch.long
    """

    import torch

    melody = convert_notes_to_cp(
        melody_notes,
        ticks_per_beat,
        MELODY_PROGRAM,
    )

    accompaniment = (
        convert_notes_to_cp(
            accompaniment_notes,
            ticks_per_beat,
            ACCOMPANIMENT_PROGRAM,
        )
    )

    tracks = (
        melody,
        accompaniment,
    )

    # ------------------------------------------------------------------------
    # IMPORTANT:
    #
    # Song length is based on the actual CP position of the note END,
    # NOT on the duration-template index.
    #
    # This prevents the old bug where a duration index such as 23 could
    # accidentally be interpreted as 23 CP steps.
    # ------------------------------------------------------------------------

    song_length = 0

    for notes in tracks:

        for note in notes:

            note_end = (
                int(note["start"])
                + int(note["duration_steps"])
            )

            if note_end > song_length:
                song_length = int(
                    note_end
                )

    if song_length <= 0:
        return None

    song = torch.full(
        (
            song_length,
            SUBSEQ_LENGTH,
        ),
        PAD_VALUE,
        dtype=torch.long,
    )

    for track_index, notes in enumerate(
        tracks
    ):

        by_start = {}

        for note in notes:

            by_start.setdefault(
                int(note["start"]),
                [],
            ).append(
                note
            )

        for start, onset_notes in (
            by_start.items()
        ):

            if (
                start < 0
                or start >= song_length
            ):
                continue

            # Same ordering as the original CP preprocessing:
            #
            # primary   = velocity
            # secondary = program
            # tertiary  = pitch
            # quaternary= duration
            #
            # np.lexsort((duration, pitch, program, velocity))
            onset_notes.sort(
                key=lambda note: (
                    note["velocity"],
                    note["program"],
                    note["pitch"],
                    note["duration_index"],
                )
            )

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

                song[
                    start,
                    base + 0
                ] = note["program"]

                song[
                    start,
                    base + 1
                ] = note["pitch"]

                # IMPORTANT:
                # duration field = template INDEX
                song[
                    start,
                    base + 2
                ] = note["duration_index"]

                song[
                    start,
                    base + 3
                ] = note["velocity"]

            note_count = len(
                onset_notes
            )

            if (
                note_count
                < MAX_POLYPHONY
            ):

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
# VALIDATE ONE SONG
# ============================================================================

def validate_song_tensor(song):
    """
    Validate the CP representation before it is allowed into the corpus.

    This catches representation mistakes early.
    """

    import torch

    if song is None:
        raise RuntimeError(
            "Song tensor is None"
        )

    if song.ndim != 2:
        raise RuntimeError(
            f"Invalid tensor rank: "
            f"{tuple(song.shape)}"
        )

    if song.shape[1] != SUBSEQ_LENGTH:
        raise RuntimeError(
            f"Invalid tensor width: "
            f"{tuple(song.shape)}"
        )

    if song.dtype != torch.long:
        raise RuntimeError(
            f"Invalid tensor dtype: "
            f"{song.dtype}"
        )

    if song.numel() == 0:
        raise RuntimeError(
            "Empty song tensor"
        )

    # ------------------------------------------------------------
    # Validate every field independently.
    #
    # Layout:
    #
    #   program, pitch, duration_index, velocity
    # ------------------------------------------------------------

    reshaped = song.view(
        song.shape[0],
        TRACK_COUNT,
        MAX_POLYPHONY,
        FIELDS_PER_NOTE,
    )

    programs = reshaped[:, :, :, 0]
    pitches = reshaped[:, :, :, 1]
    durations = reshaped[:, :, :, 2]
    velocities = reshaped[:, :, :, 3]

    # Padding/EOS occupy the program field.
    valid_program_mask = (
        (programs == PAD_VALUE)
        | (programs == EOS_VALUE)
        | (
            (
                programs >= 0
            )
            & (
                programs <= 127
            )
        )
    )

    if not bool(
        torch.all(valid_program_mask)
    ):
        raise RuntimeError(
            "Invalid program value detected"
        )

    valid_pitch_mask = (
        (pitches == PAD_VALUE)
        | (
            (pitches >= MIN_PITCH)
            & (pitches <= MAX_PITCH)
        )
    )

    if not bool(
        torch.all(valid_pitch_mask)
    ):
        raise RuntimeError(
            "Invalid pitch value detected"
        )

    # Duration is now an INDEX 0..23.
    #
    # Padding is 255.
    # EOS slots are initialized as padding, except their program field.
    valid_duration_mask = (
        (durations == PAD_VALUE)
        | (
            (durations >= 0)
            & (
                durations
                < len(DURATION_TEMPLATES)
            )
        )
    )

    if not bool(
        torch.all(valid_duration_mask)
    ):
        bad = durations[
            ~valid_duration_mask
        ]

        raise RuntimeError(
            "Invalid duration-template index "
            f"detected. First bad values: "
            f"{bad[:20].tolist()}"
        )

    valid_velocity_mask = (
        (velocities == PAD_VALUE)
        | (
            (velocities >= 0)
            & (velocities <= 127)
        )
    )

    if not bool(
        torch.all(valid_velocity_mask)
    ):
        raise RuntimeError(
            "Invalid velocity value detected"
        )


# ============================================================================
# PITCH SHIFT RANGE
# ============================================================================

def calculate_pitch_shift_range(
    melody_notes,
    accompaniment_notes,
):
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
        return None

    minimum_pitch = min(
        pitches
    )

    maximum_pitch = max(
        pitches
    )

    return (
        -minimum_pitch,
        127 - maximum_pitch,
    )


# ============================================================================
# ONE MIDI -> TEMPORARY SONG
# ============================================================================

def preprocess_one_midi_worker(
    task,
):
    """
    Worker entry point.

    The worker writes the large tensor to disk.

    Only lightweight metadata is returned to the parent.
    """

    import torch

    (
        input_index,
        midi_path,
        staging_dir,
    ) = task

    midi_path = Path(
        midi_path
    )

    midi = None
    song = None

    try:

        mido = import_mido()

        midi = mido.MidiFile(
            str(midi_path)
        )

        if len(midi.tracks) != 2:

            return {
                "ok": False,
                "index": input_index,
                "path": str(
                    midi_path
                ),
                "reason":
                    "expected_exactly_two_tracks",
                "track_count":
                    len(midi.tracks),
            }

        ticks_per_beat = int(
            midi.ticks_per_beat
        )

        if ticks_per_beat <= 0:

            return {
                "ok": False,
                "index": input_index,
                "path": str(
                    midi_path
                ),
                "reason":
                    "invalid_ticks_per_beat",
            }

        melody_notes = extract_notes(
            midi.tracks[0]
        )

        accompaniment_notes = (
            extract_notes(
                midi.tracks[1]
            )
        )

        pitch_range = (
            calculate_pitch_shift_range(
                melody_notes,
                accompaniment_notes,
            )
        )

        if pitch_range is None:

            return {
                "ok": False,
                "index": input_index,
                "path": str(
                    midi_path
                ),
                "reason":
                    "no_non_drum_pitches",
            }

        song = build_song_tensor(
            melody_notes,
            accompaniment_notes,
            ticks_per_beat,
        )

        if song is None:

            return {
                "ok": False,
                "index": input_index,
                "path": str(
                    midi_path
                ),
                "reason":
                    "no_usable_notes",
            }

        # ------------------------------------------------------------
        # CRITICAL VALIDATION
        # ------------------------------------------------------------

        validate_song_tensor(
            song
        )

        # ------------------------------------------------------------
        # Temporary worker output
        # ------------------------------------------------------------

        staging_dir = Path(
            staging_dir
        )

        staging_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        temp_path = (
            staging_dir
            / f"{input_index:08d}.pt"
        )

        torch.save(
            song,
            temp_path,
        )

        length = int(
            song.shape[0]
        )

        pitch_shift_min, pitch_shift_max = (
            pitch_range
        )

        melody_note_count = len(
            melody_notes
        )

        accompaniment_note_count = len(
            accompaniment_notes
        )

        del song
        song = None

        gc.collect()

        return {
            "ok": True,
            "index": input_index,
            "path": str(
                midi_path
            ),
            "temp_path": str(
                temp_path
            ),
            "length": length,
            "pitch_shift_min": int(
                pitch_shift_min
            ),
            "pitch_shift_max": int(
                pitch_shift_max
            ),
            "melody_notes":
                melody_note_count,
            "accompaniment_notes":
                accompaniment_note_count,
            "ticks_per_beat":
                ticks_per_beat,
        }

    except Exception as exc:

        return {
            "ok": False,
            "index": input_index,
            "path": str(
                midi_path
            ),
            "reason":
                "preprocessing_error:"
                + type(exc).__name__,
            "error": str(exc),
        }

    finally:

        midi = None
        song = None

        gc.collect()


# ============================================================================
# DISCOVER MIDI FILES
# ============================================================================

def find_stage4_midis(
    input_root,
):
    """
    Discover Stage 4 MIDI files in deterministic order.
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

    for subdir in (
        "123456789abcdef"
    ):

        directory = (
            input_root
            / subdir
        )

        if not directory.is_dir():
            continue

        for path in sorted(
            directory.iterdir(),
            key=lambda p:
                p.name.lower(),
        ):

            if not path.is_file():
                continue

            if (
                path.suffix.lower()
                != ".mid"
            ):
                continue

            files.append(
                path
            )

    return files


# ============================================================================
# ATOMIC TORCH SAVE
# ============================================================================

def atomic_torch_save(
    obj,
    path,
):
    """
    Atomically save a torch object.
    """

    import torch

    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix + ".tmp"
    )

    try:

        torch.save(
            obj,
            temp,
        )

        os.replace(
            temp,
            path,
        )

    finally:

        if temp.exists():

            try:
                temp.unlink()
            except OSError:
                pass


# ============================================================================
# FORMAT HELPERS
# ============================================================================

def format_seconds(
    seconds,
):
    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes = (
        seconds / 60
    )

    if minutes < 60:
        return f"{minutes:.1f}m"

    hours = (
        minutes / 60
    )

    return f"{hours:.1f}h"


def format_bytes(
    value,
):
    value = float(
        value
    )

    units = (
        "B",
        "KiB",
        "MiB",
        "GiB",
        "TiB",
    )

    for unit in units:

        if value < 1024.0:
            return (
                f"{value:.2f} {unit}"
            )

        value /= 1024.0

    return (
        f"{value:.2f} PiB"
    )


# ============================================================================
# FINAL DATASET VALIDATION
# ============================================================================

def validate_final_dataset(
    data_path,
    length_path,
    pitch_shift_path,
    expected_song_count,
):
    """
    Reload the final dataset and perform structural checks.

    This is deliberately performed before the script declares success.
    """

    import torch

    print()
    print(
        "Validating final dataset..."
    )

    data = torch.load(
        data_path,
        weights_only=True,
    )

    lengths = torch.load(
        length_path,
        weights_only=True,
    )

    pitch_ranges = torch.load(
        pitch_shift_path,
        weights_only=True,
    )

    if data.dtype != torch.long:

        raise RuntimeError(
            "FINAL VALIDATION FAILED: "
            f"dataset dtype is {data.dtype}"
        )

    if data.ndim != 2:

        raise RuntimeError(
            "FINAL VALIDATION FAILED: "
            f"dataset shape is "
            f"{tuple(data.shape)}"
        )

    if data.shape[1] != SUBSEQ_LENGTH:

        raise RuntimeError(
            "FINAL VALIDATION FAILED: "
            f"dataset width is "
            f"{data.shape[1]}"
        )

    if len(lengths) != expected_song_count:

        raise RuntimeError(
            "FINAL VALIDATION FAILED: "
            f"{len(lengths)} lengths for "
            f"{expected_song_count} songs"
        )

    pitch_ranges = pitch_ranges.reshape(
        -1,
        2,
    )

    if len(pitch_ranges) != expected_song_count:

        raise RuntimeError(
            "FINAL VALIDATION FAILED: "
            f"{len(pitch_ranges)} pitch ranges for "
            f"{expected_song_count} songs"
        )

    if int(
        lengths.sum().item()
    ) != int(
        data.shape[0]
    ):

        raise RuntimeError(
            "FINAL VALIDATION FAILED: "
            "sum(lengths) != data.shape[0]"
        )

    # Validate duration indices globally.
    #
    # This is potentially expensive over 34M rows, but it is performed
    # once at the very end and guarantees that the corpus cannot silently
    # contain the old actual-duration representation.

    reshaped = data.view(
        data.shape[0],
        TRACK_COUNT,
        MAX_POLYPHONY,
        FIELDS_PER_NOTE,
    )

    duration_values = (
        reshaped[:, :, :, 2]
    )

    valid_duration_mask = (
        (duration_values == PAD_VALUE)
        |
        (
            (duration_values >= 0)
            &
            (
                duration_values
                < len(DURATION_TEMPLATES)
            )
        )
    )

    if not bool(
        torch.all(
            valid_duration_mask
        )
    ):

        bad = duration_values[
            ~valid_duration_mask
        ]

        raise RuntimeError(
            "FINAL VALIDATION FAILED: "
            "invalid duration-template indices. "
            f"First bad values: "
            f"{bad[:20].tolist()}"
        )

    print(
        "Final validation: OK"
    )

    del data
    del lengths
    del pitch_ranges

    gc.collect()


# ============================================================================
# MAIN
# ============================================================================

def main():

    parser = argparse.ArgumentParser(
        description=(
            "Parallel Stage 4 -> CP16 "
            "preprocessing."
        )
    )

    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT_ROOT,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
    )

    parser.add_argument(
        "--dataset-name",
        type=str,
        default=DEFAULT_DATASET_NAME,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
    )

    parser.add_argument(
        "--maxtasksperchild",
        type=int,
        default=DEFAULT_MAX_TASKS_PER_CHILD,
    )

    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
    )

    args = parser.parse_args()

    if args.workers <= 0:
        raise ValueError(
            "--workers must be > 0"
        )

    if args.maxtasksperchild <= 0:
        raise ValueError(
            "--maxtasksperchild must be > 0"
        )

    input_root = Path(
        args.input
    )

    output_root = Path(
        args.output
    )

    dataset_name = (
        args.dataset_name
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Staging directory
    # ------------------------------------------------------------------------

    if args.staging_dir is None:

        staging_root = (
            output_root
            / (
                f".{dataset_name}"
                ".staging"
            )
        )

    else:

        staging_root = Path(
            args.staging_dir
        )

    if staging_root.exists():

        raise RuntimeError(
            "Staging directory already exists:\n"
            f"    {staging_root}\n\n"
            "This script deliberately refuses to mix "
            "partial runs.\n"
            "Remove that directory before starting "
            "a fresh preprocessing run."
        )

    staging_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 78
    )
    print(
        "STAGE 4 -> CP16 PREPROCESSING"
    )
    print(
        "PARALLEL / MEMORY-SAFE / CORRECTED DURATION INDEX"
    )
    print(
        "=" * 78
    )
    print()

    print(
        f"Input root       : {input_root}"
    )

    print(
        f"Output root      : {output_root}"
    )

    print(
        f"Dataset name     : {dataset_name}"
    )

    print(
        f"Workers          : {args.workers}"
    )

    print(
        f"Max tasks/worker : "
        f"{args.maxtasksperchild}"
    )

    print(
        f"Staging dir      : {staging_root}"
    )

    print()

    print(
        f"beat_div         : {BEAT_DIV}"
    )

    print(
        f"tracks           : {TRACK_COUNT}"
    )

    print(
        f"max_polyphony    : {MAX_POLYPHONY}"
    )

    print(
        f"fields/note      : {FIELDS_PER_NOTE}"
    )

    print(
        f"tensor width     : {SUBSEQ_LENGTH}"
    )

    print()

    print(
        "Track 0 program  : "
        f"{MELODY_PROGRAM} (melody)"
    )

    print(
        "Track 1 program  : "
        f"{ACCOMPANIMENT_PROGRAM} "
        "(accompaniment)"
    )

    print()

    print(
        "Tensor dtype     : torch.long"
    )

    print(
        "Duration field   : TEMPLATE INDEX 0..23"
    )

    print(
        "Duration values  : "
        f"{DURATION_TEMPLATES[0]}.."
        f"{DURATION_TEMPLATES[-1]}"
    )

    print(
        "Padding          : "
        f"{PAD_VALUE}"
    )

    print(
        "EOS program      : "
        f"{EOS_VALUE}"
    )

    print(
        "UglyMIDI         : NO"
    )

    print(
        "pretty_midi      : NO"
    )

    print(
        "Old quantization : NO"
    )

    print(
        "Stage 4 MIDI     : AUTHORITATIVE"
    )

    print()

    # ------------------------------------------------------------------------
    # Discover files
    # ------------------------------------------------------------------------

    midi_files = find_stage4_midis(
        input_root
    )

    total_files = len(
        midi_files
    )

    print(
        f"Found {total_files:,} "
        "Stage 4 MIDI files."
    )

    print()

    if total_files == 0:

        raise RuntimeError(
            "No Stage 4 MIDI files were found."
        )

    # ------------------------------------------------------------------------
    # Worker tasks
    # ------------------------------------------------------------------------

    tasks = (
        (
            index,
            str(midi_path),
            str(staging_root),
        )
        for index, midi_path
        in enumerate(
            midi_files
        )
    )

    results = []

    rejected = []

    completed = 0

    successful = 0

    start_time = time.monotonic()

    # ------------------------------------------------------------------------
    # Multiprocessing
    # ------------------------------------------------------------------------

    context = mp.get_context(
        "spawn"
    )

    print(
        "Starting workers..."
    )

    print()

    try:

        with context.Pool(
            processes=args.workers,
            initializer=worker_init,
            maxtasksperchild=(
                args.maxtasksperchild
            ),
        ) as pool:

            for result in pool.imap_unordered(
                preprocess_one_midi_worker,
                tasks,
                chunksize=1,
            ):

                completed += 1

                if result["ok"]:

                    results.append(
                        result
                    )

                    successful += 1

                else:

                    rejected.append(
                        result
                    )

                if (
                    completed == 1
                    or completed % 250 == 0
                    or completed == total_files
                ):

                    elapsed = (
                        time.monotonic()
                        - start_time
                    )

                    rate = (
                        completed
                        / elapsed
                        if elapsed > 0
                        else 0
                    )

                    remaining = (
                        total_files
                        - completed
                    )

                    eta = (
                        remaining / rate
                        if rate > 0
                        else 0
                    )

                    print(
                        f"\rProcessed "
                        f"{completed:,}/"
                        f"{total_files:,} "
                        f"({completed / total_files * 100:6.2f}%) "
                        f"| OK {successful:,} "
                        f"| rejected {len(rejected):,} "
                        f"| {rate:.1f}/s "
                        f"| ETA {format_seconds(eta)}",
                        end="",
                        flush=True,
                    )

        print()
        print()

    except KeyboardInterrupt:

        print()
        print(
            "Interrupted."
        )

        raise

    # ------------------------------------------------------------------------
    # Deterministic ordering
    # ------------------------------------------------------------------------

    results.sort(
        key=lambda item:
            item["index"]
    )

    # ------------------------------------------------------------------------
    # Report rejections
    # ------------------------------------------------------------------------

    if rejected:

        print(
            f"Rejected {len(rejected):,} files:"
        )

        for item in rejected[:20]:

            print(
                f"  {item['path']}"
            )

            print(
                f"      {item.get('reason', 'unknown')}"
            )

            if "error" in item:

                print(
                    f"      {item['error']}"
                )

        if len(rejected) > 20:

            print(
                f"  ... and "
                f"{len(rejected) - 20:,} more"
            )

        print()

    if not results:

        raise RuntimeError(
            "No MIDI files were successfully converted."
        )

    # ------------------------------------------------------------------------
    # Prepare final metadata
    # ------------------------------------------------------------------------

    import torch

    song_lengths = []

    pitch_shift_ranges = []

    manifest_lines = []

    total_cp_steps = 0

    for result in results:

        length = int(
            result["length"]
        )

        song_lengths.append(
            length
        )

        pitch_shift_ranges.append(
            [
                int(
                    result[
                        "pitch_shift_min"
                    ]
                ),
                int(
                    result[
                        "pitch_shift_max"
                    ]
                ),
            ]
        )

        total_cp_steps += length

        manifest_lines.append(
            (
                f"{len(manifest_lines)}\t"
                f"{os.path.relpath(result['path'], input_root)}"
                "\n"
            )
        )

    # ------------------------------------------------------------------------
    # Estimate final tensor RAM
    # ------------------------------------------------------------------------

    final_bytes = (
        total_cp_steps
        * SUBSEQ_LENGTH
        * 8
    )

    print(
        "Assembly:"
    )

    print(
        f"  Successful songs : "
        f"{len(results):,}"
    )

    print(
        f"  Total CP steps   : "
        f"{total_cp_steps:,}"
    )

    print(
        f"  Tensor shape     : "
        f"({total_cp_steps}, "
        f"{SUBSEQ_LENGTH})"
    )

    print(
        f"  Tensor dtype     : "
        f"torch.int64"
    )

    print(
        f"  Final tensor RAM : "
        f"{format_bytes(final_bytes)}"
    )

    print()

    # ------------------------------------------------------------------------
    # Allocate final tensor ONCE.
    #
    # This avoids:
    #
    #     torch.cat(list_of_18k_tensors)
    #
    # and therefore avoids a large temporary peak.
    # ------------------------------------------------------------------------

    final_data = torch.empty(
        (
            total_cp_steps,
            SUBSEQ_LENGTH,
        ),
        dtype=torch.long,
    )

    offset = 0

    for result in results:

        temp_path = Path(
            result["temp_path"]
        )

        song = torch.load(
            temp_path,
            weights_only=True,
        )

        # Defensive validation one final time.
        validate_song_tensor(
            song
        )

        length = int(
            song.shape[0]
        )

        final_data[
            offset:
            offset + length
        ].copy_(
            song
        )

        offset += length

        del song

        # Delete staging file immediately.
        try:
            temp_path.unlink()
        except OSError:
            pass

    if offset != total_cp_steps:

        del final_data

        raise RuntimeError(
            "Assembly error: "
            f"copied {offset} steps, "
            f"expected {total_cp_steps}"
        )

    # ------------------------------------------------------------------------
    # Final paths
    # ------------------------------------------------------------------------

    data_path = (
        output_root
        / f"{dataset_name}.pt"
    )

    length_path = (
        output_root
        / f"{dataset_name}.length.pt"
    )

    pitch_shift_path = (
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

    # ------------------------------------------------------------------------
    # Save final dataset
    # ------------------------------------------------------------------------

    print(
        "Saving final dataset..."
    )

    atomic_torch_save(
        final_data,
        data_path,
    )

    del final_data

    gc.collect()

    # ------------------------------------------------------------------------
    # Save lengths
    # ------------------------------------------------------------------------

    atomic_torch_save(
        torch.tensor(
            song_lengths,
            dtype=torch.long,
        ),
        length_path,
    )

    # ------------------------------------------------------------------------
    # Save pitch ranges
    # ------------------------------------------------------------------------

    atomic_torch_save(
        torch.tensor(
            pitch_shift_ranges,
            dtype=torch.long,
        ),
        pitch_shift_path,
    )

    # ------------------------------------------------------------------------
    # Save manifest
    # ------------------------------------------------------------------------

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as handle:

        handle.writelines(
            manifest_lines
        )

    # ------------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------------

    elapsed = (
        time.monotonic()
        - start_time
    )

    metadata = {
        "dataset_name":
            dataset_name,

        "input_root":
            str(input_root),

        "output_root":
            str(output_root),

        "source":
            "Stage 4 MIDI corpus",

        "stage4_is_authoritative":
            True,

        "uses_uglymidi":
            False,

        "uses_pretty_midi":
            False,

        "uses_old_quantization_filter":
            False,

        "beat_div":
            BEAT_DIV,

        "cp_resolution":
            "1/16 note",

        "tracks":
            TRACK_COUNT,

        "track_semantics": {
            "0": "melody",
            "1": "accompaniment",
        },

        "programs": {
            "melody":
                MELODY_PROGRAM,
            "accompaniment":
                ACCOMPANIMENT_PROGRAM,
        },

        "max_polyphony":
            MAX_POLYPHONY,

        "fields_per_note":
            FIELDS_PER_NOTE,

        "tensor_width":
            SUBSEQ_LENGTH,

        "dtype":
            "torch.int64",

        "padding_value":
            PAD_VALUE,

        "eos_program_value":
            EOS_VALUE,

        "duration_representation":
            "duration-template-index",

        "duration_templates":
            list(
                DURATION_TEMPLATES
            ),

        "duration_template_count":
            len(
                DURATION_TEMPLATES
            ),

        "duration_index_min":
            0,

        "duration_index_max":
            len(
                DURATION_TEMPLATES
            ) - 1,

        "worker_count":
            args.workers,

        "max_tasks_per_child":
            args.maxtasksperchild,

        "input_file_count":
            total_files,

        "converted_song_count":
            len(results),

        "rejected_song_count":
            len(rejected),

        "total_cp_timesteps":
            total_cp_steps,

        "tensor_shape": [
            total_cp_steps,
            SUBSEQ_LENGTH,
        ],

        "estimated_tensor_ram_bytes":
            final_bytes,

        "elapsed_seconds":
            elapsed,

        "rejected": rejected,
    }

    with open(
        metadata_path,
        "w",
        encoding="utf-8",
    ) as handle:

        json.dump(
            metadata,
            handle,
            indent=2,
            ensure_ascii=False,
        )

    # ------------------------------------------------------------------------
    # Final validation
    # ------------------------------------------------------------------------

    validate_final_dataset(
        data_path,
        length_path,
        pitch_shift_path,
        len(results),
    )

    # ------------------------------------------------------------------------
    # Remove staging directory
    # ------------------------------------------------------------------------

    if staging_root.exists():

        shutil.rmtree(
            staging_root
        )

    # ------------------------------------------------------------------------
    # COMPLETE
    # ------------------------------------------------------------------------

    print()
    print(
        "=" * 78
    )
    print(
        "STAGE 4 -> CP16 COMPLETE"
    )
    print(
        "=" * 78
    )
    print()

    print(
        f"Input MIDI files       : "
        f"{total_files:,}"
    )

    print(
        f"Converted songs        : "
        f"{len(results):,}"
    )

    print(
        f"Rejected               : "
        f"{len(rejected):,}"
    )

    print(
        f"Total CP timesteps     : "
        f"{total_cp_steps:,}"
    )

    print(
        f"Tensor shape           : "
        f"({total_cp_steps}, "
        f"{SUBSEQ_LENGTH})"
    )

    print(
        f"Tensor dtype           : "
        f"torch.int64"
    )

    print(
        f"Final tensor RAM       : "
        f"{format_bytes(final_bytes)}"
    )

    print(
        f"Total elapsed          : "
        f"{format_seconds(elapsed)}"
    )

    print()

    print(
        f"Main dataset           : "
        f"{data_path}"
    )

    print(
        f"Song lengths           : "
        f"{length_path}"
    )

    print(
        f"Pitch shift ranges     : "
        f"{pitch_shift_path}"
    )

    print(
        f"Manifest               : "
        f"{manifest_path}"
    )

    print(
        f"Metadata               : "
        f"{metadata_path}"
    )

    print()

    print(
        "Duration representation: "
        "TEMPLATE INDEX 0..23"
    )

    print(
        "Dataset is ready for "
        "OverlapFramedDataset."
    )

    print()


if __name__ == "__main__":
    mp.freeze_support()
    main()
    