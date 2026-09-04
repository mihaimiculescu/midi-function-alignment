#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Stage 4 -> CP16 preprocessing
PARALLEL VERSION

Converts the clean MIDI corpus produced by stage4.py into the CP tensor
representation expected by the existing CP transformer / Yinyang training
code.

This version is specifically designed for a large Stage 4 corpus.

IMPORTANT
---------
This script intentionally does NOT use:

    UglyMIDI
    pretty_midi
    the old preprocess_midi()
    the old quantization filter

Stage 4 MIDI semantics are authoritative.

PARALLELISM
-----------
Default:

    24 worker processes

Each worker:

    1. reads one MIDI
    2. extracts its notes
    3. converts it to CP16
    4. writes that song tensor to a temporary staging file
    5. returns only lightweight metadata to the parent

Large tensors are therefore NOT sent through multiprocessing pipes.

This is important for RAM usage.

The parent subsequently assembles the final contiguous tensor in deterministic
input order.

PYTORCH THREADING
-----------------
Each worker is restricted to one PyTorch CPU thread.

Otherwise:

    24 processes × N PyTorch threads

could easily overwhelm the machine.

MEMORY MANAGEMENT
-----------------
Only one song tensor exists in each worker at a time.

Workers use:

    maxtasksperchild

so worker processes are periodically replaced, preventing long-running
Python/native allocations from accumulating indefinitely.

The parent does not retain individual song tensors while preprocessing.

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
data/<dataset_name>.json

The .pt tensor has shape:

    [total_CP_timesteps, 64]

because:

    2 tracks
    x 8 notes
    x 4 fields
    = 64 values per timestep

Each CP note contains:

    [program, pitch, duration, velocity]

Padding:

    255

EOS:

    program = 254

CP16
----
beat_div = 4

Therefore:

    1 quarter-note beat = 4 CP positions
    1 CP position       = 1/16 note

Duration vocabulary:

    1, 2, 3, 4, 6, 8, 12, ... 4096

The tensor MUST use torch.long because duration values above 255 are valid.

PROGRAMS
--------
Stage 4 contains semantic tracks rather than original instrument identity.

Therefore:

    Track 0 melody         -> program 64
    Track 1 accompaniment  -> program 0

PITCH SHIFT
-----------
For every song:

    minimum shift = -minimum_pitch
    maximum shift = 127 - maximum_pitch

The existing OverlapFramedDataset subsequently clamps these to:

    [-5, +6]

SONG BOUNDARIES
---------------
No song is split here.

Each song remains a complete CP tensor in the staging area.

The final .length.pt contains one length per surviving song.

OverlapFramedDataset uses these lengths to guarantee that its
384-step / 192-step windows never cross song boundaries.

USAGE
-----
Default:

    python preprocess_stage4.py

Default workers:

    24

Explicit:

    python preprocess_stage4.py --workers 24

For example:

    python preprocess_stage4.py \
        --workers 24 \
        --dataset-name lamd_stage4_cp8_v1

Other options:

    --input
    --output
    --dataset-name
    --workers
    --maxtasksperchild
    --staging-dir

"""

import argparse
import gc
import json
import math
import multiprocessing as mp
import os
import shutil
import sys
import tempfile
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

DEFAULT_WORKERS = 24

# Replace workers periodically.
#
# This is not needed because a song is retained in memory after completion;
# it is protection against gradual allocator/native-library growth in a
# long-running worker process.
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

MIN_PITCH = 0

MAX_PITCH = 127

GM_DRUM_CHANNEL = 9

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

PAD_VALUE = 255

EOS_VALUE = 254


# ============================================================================
# WORKER INITIALIZATION
# ============================================================================

def worker_init():
    """
    Initialize one multiprocessing worker.

    Critical:
    prevent each worker from starting its own large pool of PyTorch
    CPU threads.

    24 workers × many BLAS/PyTorch threads would be counterproductive.
    """

    try:
        import torch

        torch.set_num_threads(1)

        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # Can occur if another torch operation initialized the
            # inter-op pool before this call. Not fatal.
            pass

    except Exception:
        pass

    # Avoid OpenMP / MKL thread multiplication where these libraries
    # are present.
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


# ============================================================================
# IMPORT MIDO
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
    Convert delta-time MIDI events into complete notes.

    Notes are paired by:

        channel + pitch

    using FIFO pairing.
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

    means:

        4 CP positions per quarter-note beat.
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
# DURATION -> CP TEMPLATE
# ============================================================================

def duration_to_template(
    duration,
):
    """
    Match the duration quantization of the original CP preprocessing.

    This is equivalent to selecting the nearest duration template using
    midpoint boundaries.
    """

    if duration <= 0:
        duration = 1

    templates = DURATION_TEMPLATES

    for index in range(
        len(templates) - 1
    ):

        left = templates[index]

        right = templates[
            index + 1
        ]

        boundary = (
            float(left + right)
            / 2.0
        )

        if duration < boundary:
            return left

    return templates[-1]


# ============================================================================
# NOTE -> CP
# ============================================================================

def convert_notes_to_cp(
    notes,
    ticks_per_beat,
    program,
):
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

        duration = (
            end - start
        )

        duration = (
            duration_to_template(
                duration
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
                "duration": int(
                    duration
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

    The long dtype is mandatory because valid CP durations include:

        256 ... 4096
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

    song_length = 0

    for notes in tracks:

        for note in notes:

            note_end = (
                note["start"]
                + note["duration"]
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

            # Same ordering as original CP preprocessing.
            onset_notes.sort(
                key=lambda note: (
                    note["velocity"],
                    note["program"],
                    note["pitch"],
                    note["duration"],
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

                song[
                    start,
                    base + 2
                ] = note["duration"]

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
# ONE MIDI -> TEMPORARY SONG
# ============================================================================

def preprocess_one_midi_worker(
    task,
):
    """
    Worker entry point.

    Input:

        (
            input_index,
            midi_path,
            staging_dir
        )

    Output is deliberately small:

        metadata dictionary

    The large tensor stays on disk.
    """

    import torch

    input_index, midi_path, staging_dir = (
        task
    )

    midi_path = Path(
        midi_path
    )

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

        if song.ndim != 2:

            raise RuntimeError(
                "Internal tensor rank error: "
                f"{tuple(song.shape)}"
            )

        if (
            song.shape[1]
            != SUBSEQ_LENGTH
        ):

            raise RuntimeError(
                "Internal tensor width error: "
                f"{tuple(song.shape)}"
            )

        if song.dtype != torch.long:

            raise RuntimeError(
                "Internal tensor dtype error: "
                f"{song.dtype}"
            )

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
                "index": input_index,
                "path": str(
                    midi_path
                ),
                "reason":
                    "no_non_drum_pitches",
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
            127 - maximum_pitch
        )

        # --------------------------------------------------------------
        # One temporary file per song.
        #
        # The parent receives only this path, not the tensor itself.
        # --------------------------------------------------------------

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

        # Explicitly release the tensor before the worker is reused.
        length = int(
            song.shape[0]
        )

        del song
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
            "pitch_shift_min":
                int(
                    pitch_shift_min
                ),
            "pitch_shift_max":
                int(
                    pitch_shift_max
                ),
            "melody_notes":
                len(melody_notes),
            "accompaniment_notes":
                len(
                    accompaniment_notes
                ),
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

        try:
            midi = None
        except Exception:
            pass

        gc.collect()


# ============================================================================
# DISCOVER MIDI FILES
# ============================================================================

def find_stage4_midis(
    input_root,
):
    """
    Discover Stage 4 files in deterministic order.
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
    path = Path(
        path
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp = path.with_suffix(
        path.suffix
        + ".tmp"
    )

    try:

        import torch

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
# PROGRESS DISPLAY
# ============================================================================

def format_seconds(
    seconds,
):
    if seconds < 60:
        return f"{seconds:.0f}s"

    minutes = seconds / 60

    if minutes < 60:
        return f"{minutes:.1f}m"

    hours = minutes / 60

    return f"{hours:.1f}h"


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
            "Output directory "
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

    parser.add_argument(
        "--workers",
        type=int,
        default=DEFAULT_WORKERS,
        help=(
            "Number of preprocessing workers "
            f"(default: {DEFAULT_WORKERS})"
        ),
    )

    parser.add_argument(
        "--maxtasksperchild",
        type=int,
        default=DEFAULT_MAX_TASKS_PER_CHILD,
        help=(
            "Tasks handled by each worker before "
            "replacement "
            f"(default: "
            f"{DEFAULT_MAX_TASKS_PER_CHILD})"
        ),
    )

    parser.add_argument(
        "--staging-dir",
        type=Path,
        default=None,
        help=(
            "Temporary staging directory. "
            "If omitted, a temporary directory is "
            "created under the output directory."
        ),
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

    input_root = (
        args.input
    )

    output_root = (
        args.output
    )

    dataset_name = (
        args.dataset_name
    )

    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Staging directory.
    #
    # Keeping it under output/ makes it obvious what belongs to this
    # preprocessing run and also avoids surprises with /tmp space.
    # ------------------------------------------------------------------

    if args.staging_dir is None:

        staging_root = (
            output_root
            / (
                f".{dataset_name}"
                ".staging"
            )
        )

    else:

        staging_root = (
            args.staging_dir
        )

    if staging_root.exists():

        print()
        print(
            "WARNING: staging directory already exists:"
        )
        print(
            f"    {staging_root}"
        )
        print()

        # It is safer to refuse than to accidentally mix the results
        # of two preprocessing runs.
        raise RuntimeError(
            "Staging directory already exists. "
            "Remove it manually before starting a new run."
        )

    staging_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Banner
    # ------------------------------------------------------------------

    print()
    print(
        "=" * 78
    )
    print(
        "STAGE 4 -> CP16 PREPROCESSING"
    )
    print(
        "PARALLEL VERSION"
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
        f"max_polyphony    : "
        f"{MAX_POLYPHONY}"
    )

    print(
        f"tracks           : "
        f"{TRACK_COUNT}"
    )

    print(
        f"fields/note      : "
        f"{FIELDS_PER_NOTE}"
    )

    print(
        f"tensor width     : "
        f"{SUBSEQ_LENGTH}"
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
        "UglyMIDI         : NO"
    )

    print(
        "Old quantization : NO"
    )

    print(
        "Stage 4 MIDI     : AUTHORITATIVE"
    )

    print()

    # ------------------------------------------------------------------
    # Discover files.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Create worker tasks.
    #
    # IMPORTANT:
    #
    # The task contains only:
    #
    #     index
    #     path
    #     staging directory
    #
    # No MIDI data or tensors are sent from the parent.
    # ------------------------------------------------------------------

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

    successful = 0

    completed = 0

    start_time = time.monotonic()

    # ------------------------------------------------------------------
    # multiprocessing
    #
    # "spawn" is safest across platforms and avoids inheriting unwanted
    # PyTorch state.
    #
    # On Linux, fork is faster to start, but workers import almost
    # nothing expensive here. Spawn provides safer isolation.
    # ------------------------------------------------------------------

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

                if not result["ok"]:

                    rejected.append(
                        result
                    )

                    if (
                        completed <= 20
                        or completed % 100 == 0
                    ):

                        print(
                            f"[{completed:>7,}/"
                            f"{total_files:,}] "
                            f"REJECT "
                            f"{result['path']} "
                            f"("
                            f"{result['reason']}"
                            f")"
                        )

                        if result.get(
                            "error"
                        ):

                            print(
                                "    ERROR: "
                                f"{result['error']}"
                            )

                else:

                    results.append(
                        result
                    )

                    successful += 1

                # --------------------------------------------------
                # Progress.
                # --------------------------------------------------

                if (
                    completed == 1
                    or completed % 100 == 0
                    or completed
                    == total_files
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

                    total_steps = sum(
                        int(
                            r["length"]
                        )
                        for r in results
                    )

                    print(
                        f"[{completed:>7,}/"
                        f"{total_files:,}] "
                        f"done="
                        f"{successful:,} "
                        f"rejected="
                        f"{len(rejected):,} "
                        f"CP="
                        f"{total_steps:,} "
                        f"rate="
                        f"{rate:.1f}/s "
                        f"ETA="
                        f"{format_seconds(eta)}"
                    )

    except KeyboardInterrupt:

        print()
        print(
            "=" * 78
        )
        print(
            "INTERRUPTED"
        )
        print(
            "=" * 78
        )
        print()

        print(
            "Terminating workers..."
        )

        raise

    except Exception:

        print()
        print(
            "Worker pool failed."
        )

        raise

    # ------------------------------------------------------------------
    # Ensure all expected successful staging files exist.
    # ------------------------------------------------------------------

    print()
    print(
        "Worker preprocessing complete."
    )

    print()

    missing = []

    for result in results:

        temp_path = Path(
            result["temp_path"]
        )

        if not temp_path.is_file():

            missing.append(
                result
            )

    if missing:

        raise RuntimeError(
            "Missing worker staging files: "
            f"{len(missing)}"
        )

    # ------------------------------------------------------------------
    # Sort successful results by original MIDI order.
    #
    # imap_unordered() gives us parallel throughput.
    # Sorting here restores deterministic dataset order.
    # ------------------------------------------------------------------

    results.sort(
        key=lambda r:
            r["index"]
    )

    # ------------------------------------------------------------------
    # Build metadata arrays.
    # ------------------------------------------------------------------

    lengths = [
        int(
            r["length"]
        )
        for r in results
    ]

    pitch_ranges = [
        [
            int(
                r["pitch_shift_min"]
            ),
            int(
                r["pitch_shift_max"]
            ),
        ]
        for r in results
    ]

    manifest = []

    for output_index, result in enumerate(
        results
    ):

        midi_path = Path(
            result["path"]
        )

        manifest.append(
            {
                "index": output_index,
                "input_index": int(
                    result["index"]
                ),
                "path": str(
                    midi_path.relative_to(
                        input_root
                    )
                ),
                "length": int(
                    result["length"]
                ),
                "pitch_shift_min":
                    int(
                        result[
                            "pitch_shift_min"
                        ]
                    ),
                "pitch_shift_max":
                    int(
                        result[
                            "pitch_shift_max"
                        ]
                    ),
                "melody_notes":
                    int(
                        result[
                            "melody_notes"
                        ]
                    ),
                "accompaniment_notes":
                    int(
                        result[
                            "accompaniment_notes"
                        ]
                    ),
                "ticks_per_beat":
                    int(
                        result[
                            "ticks_per_beat"
                        ]
                    ),
            }
        )

    # ------------------------------------------------------------------
    # Total tensor size.
    # ------------------------------------------------------------------

    total_cp_steps = sum(
        lengths
    )

    print(
        f"Successful songs : "
        f"{len(results):,}"
    )

    print(
        f"Rejected files   : "
        f"{len(rejected):,}"
    )

    print(
        f"Total CP steps   : "
        f"{total_cp_steps:,}"
    )

    # ------------------------------------------------------------------
    # Estimate final RAM requirement.
    #
    # torch.long = 8 bytes.
    #
    # This is useful because the final .pt is one contiguous tensor.
    # ------------------------------------------------------------------

    final_bytes = (
        total_cp_steps
        * SUBSEQ_LENGTH
        * 8
    )

    final_gib = (
        final_bytes
        / (1024 ** 3)
    )

    print(
        f"Final tensor RAM : "
        f"{final_gib:.2f} GiB"
    )

    print()

    # ------------------------------------------------------------------
    # Allocate final tensor.
    #
    # This is the only intentionally large parent-side allocation.
    #
    # We do NOT use:
    #
    #     torch.cat(list_of_20k_tensors)
    #
    # because that would require all individual tensors to remain alive
    # simultaneously and would create unnecessary peak RAM usage.
    #
    # Instead, allocate the final tensor once and fill it sequentially
    # from the worker staging files.
    # ------------------------------------------------------------------

    import torch

    print(
        "Allocating final tensor..."
    )

    data = torch.empty(
        (
            total_cp_steps,
            SUBSEQ_LENGTH,
        ),
        dtype=torch.long,
    )

    print(
        "Final tensor allocated."
    )

    print()

    # ------------------------------------------------------------------
    # Assemble.
    # ------------------------------------------------------------------

    offset = 0

    assembly_start = (
        time.monotonic()
    )

    for position, result in enumerate(
        results,
        start=1,
    ):

        temp_path = Path(
            result["temp_path"]
        )

        song = torch.load(
            temp_path,
            weights_only=True,
        )

        expected_length = int(
            result["length"]
        )

        if song.dtype != torch.long:

            raise RuntimeError(
                f"Unexpected dtype in "
                f"{temp_path}: "
                f"{song.dtype}"
            )

        if song.ndim != 2:

            raise RuntimeError(
                f"Unexpected shape in "
                f"{temp_path}: "
                f"{tuple(song.shape)}"
            )

        if (
            song.shape[0]
            != expected_length
        ):

            raise RuntimeError(
                f"Length mismatch in "
                f"{temp_path}: "
                f"metadata="
                f"{expected_length}, "
                f"tensor="
                f"{song.shape[0]}"
            )

        if (
            song.shape[1]
            != SUBSEQ_LENGTH
        ):

            raise RuntimeError(
                f"Width mismatch in "
                f"{temp_path}: "
                f"{tuple(song.shape)}"
            )

        end = (
            offset
            + expected_length
        )

        data[
            offset:end
        ] = song

        offset = end

        del song

        # Immediately remove the staging tensor.
        try:
            temp_path.unlink()
        except OSError:
            pass

        if (
            position == 1
            or position % 100 == 0
            or position == len(results)
        ):

            elapsed = (
                time.monotonic()
                - assembly_start
            )

            rate = (
                position
                / elapsed
                if elapsed > 0
                else 0
            )

            remaining = (
                len(results)
                - position
            )

            eta = (
                remaining / rate
                if rate > 0
                else 0
            )

            print(
                f"[ASSEMBLY "
                f"{position:>7,}/"
                f"{len(results):,}] "
                f"CP="
                f"{offset:,} "
                f"rate="
                f"{rate:.1f}/s "
                f"ETA="
                f"{format_seconds(eta)}"
            )

    # ------------------------------------------------------------------
    # Final sanity checks.
    # ------------------------------------------------------------------

    if offset != total_cp_steps:

        raise RuntimeError(
            "Final assembly length mismatch: "
            f"offset={offset}, "
            f"expected={total_cp_steps}"
        )

    if data.ndim != 2:

        raise RuntimeError(
            "Final tensor rank mismatch: "
            f"{tuple(data.shape)}"
        )

    if data.shape[1] != SUBSEQ_LENGTH:

        raise RuntimeError(
            "Final tensor width mismatch: "
            f"{data.shape[1]} != "
            f"{SUBSEQ_LENGTH}"
        )

    if data.dtype != torch.long:

        raise RuntimeError(
            "Final tensor dtype mismatch: "
            f"{data.dtype}"
        )

    # ------------------------------------------------------------------
    # Metadata tensors.
    # ------------------------------------------------------------------

    lengths_tensor = torch.tensor(
        lengths,
        dtype=torch.long,
    )

    pitch_shift_tensor = torch.tensor(
        pitch_ranges,
        dtype=torch.long,
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
        len(results),
        2,
    ):

        raise RuntimeError(
            "pitch_shift_range.pt shape "
            "mismatch: "
            f"{tuple(pitch_shift_tensor.shape)}"
        )

    # ------------------------------------------------------------------
    # Output paths.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Save final tensors.
    # ------------------------------------------------------------------

    print()
    print(
        "Saving final dataset..."
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

    # ------------------------------------------------------------------
    # Save manifest.
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Save detailed JSON metadata.
    # ------------------------------------------------------------------

    metadata = {
        "dataset": (
            "Los-Angeles-MIDI-Dataset-"
            "Ver-4-0-CC-BY-NC-SA"
        ),
        "source": str(
            input_root
        ),
        "preprocessing": (
            "Stage 4 specific CP16 "
            "parallel preprocessing"
        ),
        "uses_uglymidi": False,
        "uses_old_quantization_filter":
            False,
        "workers": args.workers,
        "maxtasksperchild":
            args.maxtasksperchild,
        "beat_div": BEAT_DIV,
        "max_polyphony":
            MAX_POLYPHONY,
        "track_count":
            TRACK_COUNT,
        "fields_per_note":
            FIELDS_PER_NOTE,
        "tensor_width":
            SUBSEQ_LENGTH,
        "dtype":
            "torch.long",
        "melody_track":
            0,
        "accompaniment_track":
            1,
        "melody_program":
            MELODY_PROGRAM,
        "accompaniment_program":
            ACCOMPANIMENT_PROGRAM,
        "pad_value":
            PAD_VALUE,
        "eos_value":
            EOS_VALUE,
        "duration_templates":
            list(
                DURATION_TEMPLATES
            ),
        "file_count_input":
            total_files,
        "file_count_survived":
            len(results),
        "file_count_rejected":
            len(rejected),
        "total_cp_steps":
            int(
                data.shape[0]
            ),
        "shape": [
            int(
                data.shape[0]
            ),
            int(
                data.shape[1]
            ),
        ],
        "final_tensor_bytes":
            int(final_bytes),
        "final_tensor_gib":
            float(final_gib),
        "songs":
            manifest,
        "rejections":
            rejected,
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

    # ------------------------------------------------------------------
    # Cleanup staging directory.
    # ------------------------------------------------------------------

    print()
    print(
        "Cleaning staging directory..."
    )

    try:

        shutil.rmtree(
            staging_root
        )

    except OSError as exc:

        print(
            "WARNING: Could not completely "
            "remove staging directory:"
        )

        print(
            f"    {exc}"
        )

    # ------------------------------------------------------------------
    # Release auxiliary tensors.
    # ------------------------------------------------------------------

    del lengths_tensor
    del pitch_shift_tensor

    gc.collect()

    # ------------------------------------------------------------------
    # Final report.
    # ------------------------------------------------------------------

    total_elapsed = (
        time.monotonic()
        - start_time
    )

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

    print(
        f"Final tensor RAM       : "
        f"{final_gib:.2f} GiB"
    )

    print(
        f"Total elapsed          : "
        f"{format_seconds(total_elapsed)}"
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
        f"{pitch_path}"
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
        "The dataset is ready for "
        "OverlapFramedDataset."
    )

    print()


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":

    # Required for multiprocessing with spawn.
    mp.freeze_support()

    main()

