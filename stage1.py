#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Los Angeles MIDI Dataset
Stage 1 — Direct MIDI Structural Filtering

This version deliberately DOES NOT use LAMDa_META_DATA.pickle.

Instead, every physical MIDI file is inspected directly.

INPUT
-----

Dataset/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/MIDIs/
    1/
    2/
    ...
    9/
    a/
    ...
    f/

OUTPUT
------

All output files are written into ONE directory:

Dataset/LAMDselection/selection_stage1/

One independent output set is produced for every input hexadecimal
subdirectory.

Example:

    candidates_1.txt
    candidates_1.json
    rejections_1.json
    summary_1.json

    candidates_2.txt
    candidates_2.json
    rejections_2.json
    summary_2.json

    ...

No MIDI files are copied, moved, or modified.

FILTERS
-------

The filters reproduce the existing Stage 1 logic in la_filter.py,
but the values are calculated directly from the MIDI.

FIRST FILTER:

    MIDI format/type 0 -> reject immediately.

Then:

    min_tracks             = 2
    max_tracks             = 16
    min_score_events       = 100
    max_score_events       = 250000
    min_chords             = 20
    min_chords_ms          = 2000
    min_pitches_ms         = 10000
    max_tempo_changes      = None
    reject_lyrics          = False
    max_text_events        = None

The definition of "total_number_of_chords" intentionally follows the
original LAMDa metadata generator:

    number of distinct MIDI patches/programs

It does NOT mean the number of simultaneous pitch-class chord shapes.

WORKERS
-------

24 processes by default.

The number of outstanding jobs is bounded so that the parent process
does not accumulate 100k+ completed futures/results in memory.
"""

import gc
import json
import os
import struct
import sys
from collections import Counter
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    wait,
)
from pathlib import Path


# ============================================================================
# PATHS
# ============================================================================

MIDI_ROOT = Path(
    "Dataset/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/MIDIs"
)

OUTPUT_DIR = Path(
    "Dataset/LAMDselection/selection_stage1"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

WORKERS = 24

# Number of futures allowed to exist at once.
#
# This is deliberately larger than WORKERS, but bounded.
#
# We do NOT submit all MIDI files to ProcessPoolExecutor simultaneously.
#
MAX_PENDING = WORKERS * 2

# The dataset is organized as hexadecimal directories.
INPUT_SUBDIRECTORIES = tuple(
    "123456789abcdef"
)


# ============================================================================
# STAGE 1 FILTERS
# ============================================================================
#
# These values are intentionally identical to the existing la_filter.py
# Stage 1 structural filters.
#
# ============================================================================

FILTERS = {
    "min_tracks": 2,
    "max_tracks": 16,

    "min_score_events": 100,
    "max_score_events": 250000,

    "min_chords": 20,
    "min_chords_ms": 2000,

    "min_pitches_ms": 10000,

    "max_tempo_changes": None,

    "reject_lyrics": False,
    "max_text_events": None,
}


# ============================================================================
# MIDI IMPORT
# ============================================================================
#
# Import MIDI only after basic configuration has been established.
#
# The project already depends on MIDI.py.
# ============================================================================

try:
    import MIDI
except Exception as exc:
    print()
    print("=" * 70)
    print("ERROR: Could not import the project's MIDI module.")
    print("=" * 70)
    print()
    print(f"{type(exc).__name__}: {exc}")
    print()
    print(
        "Run this script from the midi-function-alignment repository "
        "environment."
    )
    print()
    sys.exit(1)


# ============================================================================
# BASIC MIDI HEADER INSPECTION
# ============================================================================

def inspect_midi_header(midi_path):
    """
    Inspect only the MIDI header.

    Returns:

        {
            "format": 0 / 1 / 2,
            "tracks": integer,
            "division": integer,
        }

    or:

        (None, reason)

    This is intentionally done before MIDI.midi2opus().

    In particular, MIDI format 0 is rejected before the expensive parser
    is invoked.
    """

    try:
        with open(midi_path, "rb") as fh:
            header = fh.read(14)

    except Exception as exc:
        return None, (
            "file_read_error:"
            + type(exc).__name__
        )

    if len(header) < 14:
        return None, "midi_header_too_short"

    if header[0:4] != b"MThd":
        return None, "invalid_midi_header"

    try:
        header_length = struct.unpack(
            ">I",
            header[4:8],
        )[0]

        midi_format = struct.unpack(
            ">H",
            header[8:10],
        )[0]

        track_count = struct.unpack(
            ">H",
            header[10:12],
        )[0]

        division = struct.unpack(
            ">H",
            header[12:14],
        )[0]

    except struct.error:
        return None, "invalid_midi_header"

    if header_length < 6:
        return None, "invalid_midi_header_length"

    if midi_format not in (0, 1, 2):
        return None, "invalid_midi_format"

    return {
        "format": midi_format,
        "tracks": track_count,
        "division": division,
    }, None


# ============================================================================
# DIRECT MIDI PARSING
# ============================================================================

def analyze_midi(midi_path):
    """
    Reproduce the relevant LAMDa metadata calculations directly from
    the physical MIDI file.

    Returns:

        (result_dict, None)

    or:

        (None, rejection_reason)
    """

    # ------------------------------------------------------------------
    # FIRST CRITERION:
    #
    # MIDI type 0 is rejected immediately.
    #
    # No full MIDI parsing is performed.
    # ------------------------------------------------------------------

    header, reason = inspect_midi_header(
        midi_path
    )

    if reason is not None:
        return None, reason

    if header["format"] == 0:
        return None, "midi_type_0"

    # ------------------------------------------------------------------
    # Parse MIDI.
    #
    # Read bytes locally in the worker. Do not return the bytes to the
    # parent process.
    # ------------------------------------------------------------------

    try:
        with open(
            midi_path,
            "rb",
        ) as fh:
            midi_data = fh.read()

        opus = MIDI.midi2opus(
            midi_data
        )

        # Drop raw MIDI bytes as soon as the parser has consumed them.
        del midi_data

    except Exception as exc:
        return None, (
            "midi_parse_error:"
            + type(exc).__name__
        )

    # ------------------------------------------------------------------
    # Validate opus.
    # ------------------------------------------------------------------

    if not opus or len(opus) < 2:
        del opus
        gc.collect()

        return None, "empty_midi"

    # ------------------------------------------------------------------
    # The original LAMDa generator creates:
    #
    #     opus_events_matrix
    #
    # by concatenating all events from all tracks.
    #
    # This is used for:
    #
    #     total_number_of_opus_midi_events
    #
    # It is NOT directly used by Stage 1, but calculating it here keeps
    # the direct MIDI analysis faithful to the original representation.
    # ------------------------------------------------------------------

    try:
        opus_track_count = len(opus) - 1

        opus_event_count = 0

        for track in opus[1:]:
            opus_event_count += len(track)

        # ------------------------------------------------------------------
        # Convert opus to millisecond score.
        #
        # This is exactly the representation used by the original
        # metadata generator.
        # ------------------------------------------------------------------

        ms_score = MIDI.opus2score(
            MIDI.to_millisecs(opus)
        )

        # ------------------------------------------------------------------
        # Extract note events.
        #
        # Original code:
        #
        #     if event[0] == 'note':
        #         ms_events_matrix.append(event)
        # ------------------------------------------------------------------

        ms_events = []

        for track in ms_score[1:]:
            for event in track:
                if event and event[0] == "note":
                    ms_events.append(event)

        ms_events.sort(
            key=lambda event: event[1]
        )

        # ------------------------------------------------------------------
        # Convert original opus to score representation.
        # ------------------------------------------------------------------

        score = MIDI.opus2score(
            opus
        )

        # ------------------------------------------------------------------
        # Reproduce the original event matrix.
        #
        # The original metadata generator includes:
        #
        #     note
        #     patch_change
        #
        # events here.
        # ------------------------------------------------------------------

        events_matrix = []
        full_events_matrix = []

        for track in score[1:]:
            for event in track:

                if event[0] in (
                    "note",
                    "patch_change",
                ):
                    events_matrix.append(event)

                full_events_matrix.append(event)

        full_events_matrix.sort(
            key=lambda event: event[1]
        )

        events_matrix.sort(
            key=lambda event: event[1]
        )

        # ------------------------------------------------------------------
        # Determine the active patch/program per MIDI channel.
        #
        # This reproduces:
        #
        #     patches = [0] * 16
        #
        #     patch_change -> update channel
        #
        #     note -> append current patch
        #
        # ------------------------------------------------------------------

        patches = [0] * 16

        events_matrix1 = []

        for event in events_matrix:

            if event[0] == "patch_change":

                try:
                    channel = int(event[2])
                    patch = int(event[3])

                    if 0 <= channel < 16:
                        patches[channel] = patch

                except (
                    TypeError,
                    ValueError,
                    IndexError,
                ):
                    pass

            elif event[0] == "note":

                # The original code does:
                #
                #     event.extend([patches[event[3]]])
                #
                # We preserve that logic, but defensively handle malformed
                # channel values.
                try:
                    channel = int(event[3])

                    if not 0 <= channel < 16:
                        continue

                    event_copy = list(event)
                    event_copy.append(
                        patches[channel]
                    )

                    events_matrix1.append(
                        event_copy
                    )

                except (
                    TypeError,
                    ValueError,
                    IndexError,
                ):
                    continue

        # ------------------------------------------------------------------
        # The original metadata maker only generated metadata for files
        # having > 32 note events.
        #
        # Such files cannot pass the existing Stage 1 min_score_events=100
        # anyway, so reject them directly.
        # ------------------------------------------------------------------

        if len(events_matrix1) <= 32:
            return None, "too_few_note_events"

        # ------------------------------------------------------------------
        # total_number_of_score_midi_events
        #
        # IMPORTANT:
        # This is len(full_events_matrix), not len(events_matrix1).
        # ------------------------------------------------------------------

        score_event_count = len(
            full_events_matrix
        )

        # ------------------------------------------------------------------
        # MIDI patches.
        #
        # This is used to reproduce total_number_of_chords.
        #
        # Original LAMDa definition:
        #
        #     total_number_of_chords =
        #         len(set([y[1] for y in events_matrix1]))
        #
        # where y[1] is the MIDI program/patch appended above.
        # ------------------------------------------------------------------

        distinct_patches = set()

        for event in events_matrix1:
            try:
                distinct_patches.add(
                    int(event[5])
                )
            except (
                TypeError,
                ValueError,
                IndexError,
            ):
                pass

        total_number_of_chords = len(
            distinct_patches
        )

        # ------------------------------------------------------------------
        # Calculate time activity exactly as the original metadata maker.
        #
        # It takes the first note timestamp and records a new timestamp
        # whenever the millisecond onset changes.
        # ------------------------------------------------------------------

        if not ms_events:
            return None, "no_note_events"

        times = []

        previous_time = ms_events[0][1]
        first = True

        for event in ms_events:

            current_time = event[1]

            if (
                current_time != previous_time
                or first
            ):
                times.append(
                    current_time - previous_time
                )
                first = False

            previous_time = current_time

        # Original:
        #
        #     times_sum = min(10000000, sum(times))
        #
        pitches_times_sum_ms = min(
            10000000,
            sum(times),
        )

        # ------------------------------------------------------------------
        # Original metadata calls len(times):
        #
        #     total_number_of_chords_ms
        #
        # despite this really being the number of distinct note-onset
        # intervals/timestamps.
        # ------------------------------------------------------------------

        total_number_of_chords_ms = len(
            times
        )

        # ------------------------------------------------------------------
        # Tempo changes
        # ------------------------------------------------------------------

        tempo_change_count = sum(
            1
            for event in full_events_matrix
            if event[0] == "set_tempo"
        )

        # ------------------------------------------------------------------
        # Text / lyric events
        # ------------------------------------------------------------------

        text_event_types = {
            "text_event",
            "text_event_08",
            "text_event_09",
            "text_event_0a",
            "text_event_0b",
            "text_event_0c",
            "text_event_0d",
            "text_event_0e",
            "text_event_0f",
        }

        text_events_count = sum(
            1
            for event in full_events_matrix
            if event[0] in text_event_types
        )

        lyric_events_count = sum(
            1
            for event in full_events_matrix
            if event[0] == "lyric"
        )

        # ------------------------------------------------------------------
        # Original LAMDa metadata used:
        #
        #     total_number_of_tracks = itrack
        #
        # where itrack starts at 1 and increments once per score track.
        #
        # Therefore this is effectively len(score)-1.
        #
        # We use the actual parsed score count rather than trusting the
        # MIDI header.
        # ------------------------------------------------------------------

        total_tracks = len(score) - 1

        # ------------------------------------------------------------------
        # Return ONLY compact information.
        #
        # Do not return opus, score, note lists, etc.
        #
        # This is important for multiprocessing memory usage.
        # ------------------------------------------------------------------

        result = {
            "total_number_of_tracks":
                total_tracks,

            "total_number_of_opus_midi_events":
                opus_event_count,

            "total_number_of_score_midi_events":
                score_event_count,

            "total_number_of_chords":
                total_number_of_chords,

            "total_number_of_chords_ms":
                total_number_of_chords_ms,

            "pitches_times_sum_ms":
                pitches_times_sum_ms,

            "tempo_change_count":
                tempo_change_count,

            "text_events_count":
                text_events_count,

            "lyric_events_count":
                lyric_events_count,

            "midi_format":
                header["format"],
        }

        # ------------------------------------------------------------------
        # Explicitly destroy large intermediate structures before the
        # worker returns.
        # ------------------------------------------------------------------

        del ms_events
        del ms_score
        del events_matrix
        del events_matrix1
        del full_events_matrix
        del score
        del opus

        gc.collect()

        return result, None

    except Exception as exc:

        # Make absolutely sure large structures are not retained by a
        # failed worker invocation.
        gc.collect()

        return None, (
            "midi_analysis_error:"
            + type(exc).__name__
        )


# ============================================================================
# WORKER
# ============================================================================

def process_one_midi(midi_path):
    """
    Worker entry point.

    Returns a compact result suitable for transfer between processes.
    """

    path = str(midi_path)

    try:
        analysis, reason = analyze_midi(
            path
        )

        if reason is not None:
            return {
                "path": path,
                "md5": Path(path).stem.lower(),
                "passed": False,
                "reason": reason,
                "analysis": None,
            }

        passed, filter_reason = passes_stage1(
            analysis
        )

        return {
            "path": path,
            "md5": Path(path).stem.lower(),
            "passed": passed,
            "reason": filter_reason,
            "analysis": analysis,
        }

    except Exception as exc:

        return {
            "path": path,
            "md5": Path(path).stem.lower(),
            "passed": False,
            "reason": (
                "worker_error:"
                + type(exc).__name__
            ),
            "analysis": None,
        }

    finally:
        gc.collect()


# ============================================================================
# STAGE 1 FILTER
# ============================================================================

def passes_stage1(analysis):
    """
    Apply exactly the existing Stage 1 structural thresholds to the
    values calculated from the physical MIDI.

    Returns:

        (True, None)

    or:

        (False, rejection_reason)
    """

    tracks = int(
        analysis["total_number_of_tracks"]
    )

    score_events = int(
        analysis["total_number_of_score_midi_events"]
    )

    chords = int(
        analysis["total_number_of_chords"]
    )

    chords_ms = int(
        analysis["total_number_of_chords_ms"]
    )

    pitches_ms = int(
        analysis["pitches_times_sum_ms"]
    )

    tempo_changes = int(
        analysis["tempo_change_count"]
    )

    text_events = int(
        analysis["text_events_count"]
    )

    lyric_events = int(
        analysis["lyric_events_count"]
    )

    # ------------------------------------------------------------------
    # Track count
    # ------------------------------------------------------------------

    if tracks < FILTERS["min_tracks"]:
        return False, "too_few_tracks"

    if FILTERS["max_tracks"] is not None:

        if tracks > FILTERS["max_tracks"]:
            return False, "too_many_tracks"

    # ------------------------------------------------------------------
    # Score event count
    # ------------------------------------------------------------------

    if score_events < FILTERS["min_score_events"]:
        return False, "too_few_score_events"

    if FILTERS["max_score_events"] is not None:

        if score_events > FILTERS["max_score_events"]:
            return False, "too_many_score_events"

    # ------------------------------------------------------------------
    # Harmonic / patch activity
    # ------------------------------------------------------------------

    if chords < FILTERS["min_chords"]:
        return False, "too_few_chords"

    if chords_ms < FILTERS["min_chords_ms"]:
        return False, "too_little_chord_activity"

    # ------------------------------------------------------------------
    # Pitched activity
    # ------------------------------------------------------------------

    if pitches_ms < FILTERS["min_pitches_ms"]:
        return False, "too_little_pitch_activity"

    # ------------------------------------------------------------------
    # Tempo changes
    # ------------------------------------------------------------------

    if FILTERS["max_tempo_changes"] is not None:

        if tempo_changes > FILTERS["max_tempo_changes"]:
            return False, "too_many_tempo_changes"

    # ------------------------------------------------------------------
    # Text
    # ------------------------------------------------------------------

    if FILTERS["max_text_events"] is not None:

        if text_events > FILTERS["max_text_events"]:
            return False, "too_many_text_events"

    # ------------------------------------------------------------------
    # Lyrics
    # ------------------------------------------------------------------

    if FILTERS["reject_lyrics"]:

        if lyric_events > 0:
            return False, "contains_lyrics"

    return True, None


# ============================================================================
# JSON HELPERS
# ============================================================================

def atomic_json_write(path, data):
    """
    Atomically write JSON.

    This prevents leaving a half-written JSON file if the process is
    interrupted during the final write.
    """

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = path.with_suffix(
        path.suffix + ".tmp"
    )

    with open(
        temporary_path,
        "w",
        encoding="utf-8",
    ) as fh:

        json.dump(
            data,
            fh,
            indent=2,
            ensure_ascii=False,
        )

        fh.flush()
        os.fsync(
            fh.fileno()
        )

    os.replace(
        temporary_path,
        path,
    )


# ============================================================================
# PROCESS ONE INPUT SUBDIRECTORY
# ============================================================================

def process_subdirectory(subdir):
    """
    Process exactly one hexadecimal MIDI directory.

    All results for this directory are written as:

        candidates_<subdir>.txt
        candidates_<subdir>.json
        rejections_<subdir>.json
        summary_<subdir>.json
    """

    input_dir = MIDI_ROOT / subdir

    print()
    print("=" * 70)
    print(f"STAGE 1 — INPUT SUBDIRECTORY {subdir}")
    print("=" * 70)
    print()
    print(f"Input : {input_dir}")
    print(f"Output: {OUTPUT_DIR}")
    print(f"Workers: {WORKERS}")
    print(f"Max pending jobs: {MAX_PENDING}")
    print()

    if not input_dir.is_dir():
        print(
            f"WARNING: input directory does not exist: {input_dir}"
        )
        return

    # ------------------------------------------------------------------
    # Locate files.
    #
    # We only retain Path objects here.
    # We do not read file contents.
    # ------------------------------------------------------------------

    midi_paths = sorted(
        (
            path
            for path in input_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() == ".mid"
        ),
        key=lambda path: path.name.lower(),
    )

    total_files = len(
        midi_paths
    )

    print(
        f"MIDI files found: {total_files:,}"
    )
    print()

    if total_files == 0:
        return

    # ------------------------------------------------------------------
    # Result accumulators.
    #
    # These contain only compact dictionaries.
    #
    # The actual MIDI data lives only inside workers.
    # ------------------------------------------------------------------

    candidates = []

    rejection_details = []

    rejection_counts = Counter()

    completed = 0

    # ------------------------------------------------------------------
    # Bounded multiprocessing.
    #
    # We deliberately DO NOT use:
    #
    #     executor.map(...)
    #
    # over the entire 100k+ file list because the parent can end up
    # maintaining a very large amount of future/result bookkeeping.
    #
    # Instead we keep at most MAX_PENDING jobs alive.
    # ------------------------------------------------------------------

    next_index = 0

    with ProcessPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        pending = {}

        # --------------------------------------------------------------
        # Initial batch
        # --------------------------------------------------------------

        while (
            next_index < total_files
            and len(pending) < MAX_PENDING
        ):

            path = midi_paths[next_index]

            future = executor.submit(
                process_one_midi,
                path,
            )

            pending[future] = path

            next_index += 1

        # --------------------------------------------------------------
        # Consume results and continuously refill the bounded queue.
        # --------------------------------------------------------------

        while pending:

            done, _ = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )

            for future in done:

                path = pending.pop(
                    future
                )

                completed += 1

                try:
                    result = future.result()

                except Exception as exc:

                    result = {
                        "path": str(path.resolve()),
                        "md5": path.stem.lower(),
                        "passed": False,
                        "reason": (
                            "future_error:"
                            + type(exc).__name__
                        ),
                        "analysis": None,
                    }

                # ------------------------------------------------------
                # Candidate
                # ------------------------------------------------------

                if result["passed"]:

                    candidate = {
                        "md5":
                            result["md5"],

                        "path":
                            str(
                                path.resolve()
                            ),

                        "input_subdirectory":
                            subdir,

                        "midi_format":
                            result["analysis"][
                                "midi_format"
                            ],

                        "total_number_of_tracks":
                            result["analysis"][
                                "total_number_of_tracks"
                            ],

                        "total_number_of_opus_midi_events":
                            result["analysis"][
                                "total_number_of_opus_midi_events"
                            ],

                        "total_number_of_score_midi_events":
                            result["analysis"][
                                "total_number_of_score_midi_events"
                            ],

                        "total_number_of_chords":
                            result["analysis"][
                                "total_number_of_chords"
                            ],

                        "total_number_of_chords_ms":
                            result["analysis"][
                                "total_number_of_chords_ms"
                            ],

                        "pitches_times_sum_ms":
                            result["analysis"][
                                "pitches_times_sum_ms"
                            ],

                        "tempo_change_count":
                            result["analysis"][
                                "tempo_change_count"
                            ],

                        "text_events_count":
                            result["analysis"][
                                "text_events_count"
                            ],

                        "lyric_events_count":
                            result["analysis"][
                                "lyric_events_count"
                            ],
                    }

                    candidates.append(
                        candidate
                    )

                # ------------------------------------------------------
                # Rejection
                # ------------------------------------------------------

                else:

                    reason = result[
                        "reason"
                    ]

                    rejection_counts[
                        reason
                    ] += 1

                    rejection_details.append(
                        {
                            "md5":
                                result["md5"],

                            "path":
                                str(
                                    path.resolve()
                                ),

                            "reason":
                                reason,
                        }
                    )

                # ------------------------------------------------------
                # Keep queue full, but bounded.
                # ------------------------------------------------------

                if next_index < total_files:

                    next_path = midi_paths[
                        next_index
                    ]

                    next_future = executor.submit(
                        process_one_midi,
                        next_path,
                    )

                    pending[
                        next_future
                    ] = next_path

                    next_index += 1

                # ------------------------------------------------------
                # Progress.
                # ------------------------------------------------------

                if (
                    completed % 1000 == 0
                    or completed == total_files
                ):

                    print(
                        f"[{subdir}] "
                        f"{completed:,}/{total_files:,} "
                        f"({completed / total_files * 100:6.2f}%) "
                        f"candidates={len(candidates):,} "
                        f"rejected={completed - len(candidates):,}",
                        flush=True,
                    )

    # ------------------------------------------------------------------
    # Deterministic output ordering.
    # ------------------------------------------------------------------

    candidates.sort(
        key=lambda item: item["path"]
    )

    rejection_details.sort(
        key=lambda item: item["path"]
    )

    # ------------------------------------------------------------------
    # Output filenames.
    # ------------------------------------------------------------------

    candidates_txt = (
        OUTPUT_DIR
        / f"candidates_{subdir}.txt"
    )

    candidates_json = (
        OUTPUT_DIR
        / f"candidates_{subdir}.json"
    )

    rejections_json = (
        OUTPUT_DIR
        / f"rejections_{subdir}.json"
    )

    summary_json = (
        OUTPUT_DIR
        / f"summary_{subdir}.json"
    )

    # ------------------------------------------------------------------
    # candidates.txt
    # ------------------------------------------------------------------

    with open(
        candidates_txt,
        "w",
        encoding="utf-8",
    ) as fh:

        for candidate in candidates:

            fh.write(
                candidate["path"]
                + "\n"
            )

    # ------------------------------------------------------------------
    # candidates.json
    # ------------------------------------------------------------------

    atomic_json_write(
        candidates_json,
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

            "stage":
                1,

            "input_subdirectory":
                subdir,

            "description":
                (
                    "Stage 1 candidates selected by inspecting "
                    "the physical MIDI files directly. "
                    "LAMDa metadata is not used."
                ),

            "midi_root":
                str(MIDI_ROOT),

            "filters":
                FILTERS,

            "additional_first_filter":
                {
                    "midi_type_0":
                        "reject"
                },

            "workers":
                WORKERS,

            "candidate_count":
                len(candidates),

            "candidates":
                candidates,
        },
    )

    # ------------------------------------------------------------------
    # rejections.json
    # ------------------------------------------------------------------

    atomic_json_write(
        rejections_json,
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

            "stage":
                1,

            "input_subdirectory":
                subdir,

            "description":
                (
                    "Stage 1 MIDI files rejected after direct "
                    "inspection of the physical MIDI."
                ),

            "rejection_counts":
                dict(
                    rejection_counts.most_common()
                ),

            "rejected_count":
                len(
                    rejection_details
                ),

            "rejections":
                rejection_details,
        },
    )

    # ------------------------------------------------------------------
    # summary.json
    # ------------------------------------------------------------------

    candidate_count = len(
        candidates
    )

    rejected_count = len(
        rejection_details
    )

    summary = {
        "dataset":
            "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

        "stage":
            1,

        "input_subdirectory":
            subdir,

        "input_directory":
            str(input_dir),

        "output_directory":
            str(OUTPUT_DIR),

        "metadata_source":
            None,

        "metadata_used":
            False,

        "physical_midi_inspection":
            True,

        "total_midi_files":
            total_files,

        "candidates":
            candidate_count,

        "rejected":
            rejected_count,

        "candidate_ratio":
            (
                candidate_count / total_files
                if total_files
                else 0
            ),

        "rejection_counts":
            dict(
                rejection_counts.most_common()
            ),

        "workers":
            WORKERS,

        "max_pending":
            MAX_PENDING,

        "filters":
            FILTERS,

        "additional_first_filter":
            {
                "midi_type_0":
                    "reject"
            },

        "outputs":
            {
                "candidates_txt":
                    str(candidates_txt),

                "candidates_json":
                    str(candidates_json),

                "rejections_json":
                    str(rejections_json),

                "summary_json":
                    str(summary_json),
            },
    }

    atomic_json_write(
        summary_json,
        summary,
    )

    # ------------------------------------------------------------------
    # Final report.
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print(f"SUBDIRECTORY {subdir} COMPLETE")
    print("=" * 70)
    print()
    print(
        f"Input files       : {total_files:,}"
    )
    print(
        f"Candidates        : {candidate_count:,}"
    )
    print(
        f"Rejected          : {rejected_count:,}"
    )

    if total_files:
        print(
            f"Candidate ratio   : "
            f"{candidate_count / total_files:.4f}"
        )

    print()
    print("Rejection reasons:")
    print("-" * 70)

    for reason, count in (
        rejection_counts.most_common()
    ):
        print(
            f"{reason:35s} {count:10,}"
        )

    print()
    print("Output:")
    print(
        candidates_txt
    )
    print(
        candidates_json
    )
    print(
        rejections_json
    )
    print(
        summary_json
    )
    print()

    # ------------------------------------------------------------------
    # Drop large parent-process result structures before moving to the
    # next hexadecimal directory.
    # ------------------------------------------------------------------

    del candidates
    del rejection_details
    del rejection_counts
    del midi_paths

    gc.collect()


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 70)
    print("Los Angeles MIDI Dataset")
    print("STAGE 1 — DIRECT MIDI FILTERING")
    print("=" * 70)
    print()
    print(
        "LAMDa_META_DATA.pickle is NOT used."
    )
    print(
        f"MIDI root : {MIDI_ROOT}"
    )
    print(
        f"Output    : {OUTPUT_DIR}"
    )
    print(
        f"Workers   : {WORKERS}"
    )
    print(
        f"Max pending jobs: {MAX_PENDING}"
    )
    print()

    # ------------------------------------------------------------------
    # Validate directories.
    # ------------------------------------------------------------------

    if not MIDI_ROOT.is_dir():

        print(
            "ERROR: MIDI dataset directory does not exist:"
        )
        print(
            MIDI_ROOT
        )
        sys.exit(1)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Process hexadecimal directories independently.
    #
    # Processing one directory at a time has two advantages:
    #
    # 1. output is naturally separated
    # 2. parent-process memory can be reclaimed between directories
    #
    # Workers are reused only for the duration of one subdirectory.
    # ------------------------------------------------------------------

    for subdir in INPUT_SUBDIRECTORIES:

        process_subdirectory(
            subdir
        )

    print()
    print("=" * 70)
    print("STAGE 1 COMPLETE")
    print("=" * 70)
    print()


if __name__ == "__main__":
    main()
