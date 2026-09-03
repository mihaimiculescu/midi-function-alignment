#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Los Angeles MIDI Dataset
Stage 3 — Direct MIDI Musical Structure Analysis

IMPORTANT PIPELINE CHANGE
-------------------------
Stage 3 consumes Stage 1 output DIRECTLY.

Stage 2 is deliberately NOT USED.

INPUT
-----
Dataset/LAMDselection/selection_stage1/
    candidates_1.json
    candidates_2.json
    ...
    candidates_f.json

The candidate JSON files were produced by stage1.py, which inspected
the physical MIDI files directly.

MIDI files:
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
All Stage 3 output is written into:

Dataset/LAMDselection/selection_stage3/

One independent output set is produced for every input hexadecimal
subdirectory:

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

PROCESSING
----------
24 worker processes by default.

Outstanding work is bounded to prevent the parent process from
accumulating tens of thousands of Future objects/results.

MEMORY
------
Each worker parses one MIDI, extracts only the information needed
for Stage 3, then explicitly releases the PrettyMIDI object and
runs garbage collection.

The parent receives only compact Python dictionaries.

CHECKPOINTING
-------------
Checkpoints are NOT accumulated in one ever-growing JSON file.

Instead:

    checkpoint1.json
    checkpoint2.json
    checkpoint3.json
    ...

Each checkpoint contains ONLY the newly completed batch.

On resume, all checkpoint files belonging to the current input
subdirectory are read and reconstructed.

When the COMPLETE Stage 3 run succeeds, all checkpoint*.json files
are deleted.

A failed/interrupted run therefore leaves the checkpoints intact.
"""

import gc
import hashlib
import json
import os
import sys
import tempfile
from collections import Counter
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path


# ============================================================================
# PATHS
# ============================================================================

MIDI_ROOT = Path(
    "Dataset/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/MIDIs"
)

STAGE1_DIR = Path(
    "Dataset/LAMDselection/selection_stage1"
)

STAGE3_DIR = Path(
    "Dataset/LAMDselection/selection_stage3"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

WORKERS = 24

# Never allow the parent to have an unlimited number of outstanding jobs.
MAX_PENDING = WORKERS * 2

# Number of completed MIDI files represented by one checkpoint.
CHECKPOINT_INTERVAL = 1000

# Same maximum number of ranked track candidates as la_filter.py.
MAX_CANDIDATES_PER_FILE = 8

# Same minimum number of notes required for a track to participate
# in melody-candidate ranking.
MIN_NOTES_FOR_CANDIDATE = 8

INPUT_SUBDIRECTORIES = tuple(
    "123456789abcdef"
)


# ============================================================================
# MIDI IMPORT
# ============================================================================

try:
    import pretty_midi
except Exception as exc:
    print()
    print("=" * 70)
    print("ERROR: Could not import pretty_midi.")
    print("=" * 70)
    print()
    print(f"{type(exc).__name__}: {exc}")
    print()
    sys.exit(1)


# ============================================================================
# JSON SERIALIZATION
# ============================================================================

def make_json_serializable(obj):
    """
    Recursively convert NumPy/scalar/container values into objects
    accepted by json.dump().

    This is deliberately applied BEFORE every JSON write.

    In particular, NumPy int64/float64 values are converted through
    .item(), preventing the failure seen in the previous Stage 3:

        TypeError:
        Object of type int64 is not JSON serializable
    """

    if isinstance(obj, dict):
        return {
            make_json_serializable(key):
                make_json_serializable(value)
            for key, value in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [
            make_json_serializable(value)
            for value in obj
        ]

    if isinstance(obj, set):
        return [
            make_json_serializable(value)
            for value in sorted(obj)
        ]

    if hasattr(obj, "item"):
        try:
            return obj.item()
        except (ValueError, TypeError):
            pass

    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except (ValueError, TypeError):
            pass

    return obj


def atomic_json_write(path, data):
    """
    Atomically write JSON.

    The complete object is normalized before json.dump(), so NumPy
    scalar values cannot leak into the encoder.
    """

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = make_json_serializable(data)

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            delete=False,
            prefix=f".{path.name}.",
            suffix=".tmp",
        ) as fh:

            json.dump(
                data,
                fh,
                indent=2,
                ensure_ascii=False,
            )

            fh.flush()
            os.fsync(fh.fileno())

            temp_path = Path(fh.name)

        os.replace(
            temp_path,
            path,
        )

    finally:
        if temp_path is not None and temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


# ============================================================================
# INPUT
# ============================================================================

def load_stage1_candidates(subdir):
    """
    Load candidates_<subdir>.json from Stage 1.

    The Stage 1 file is the ONLY source of Stage 3 input.

    Stage 2 is never consulted.
    """

    path = STAGE1_DIR / f"candidates_{subdir}.json"

    if not path.is_file():
        raise FileNotFoundError(
            f"Stage 1 candidate file not found:\n{path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as fh:

        data = json.load(fh)

    candidates = data.get("candidates")

    if not isinstance(candidates, list):
        raise RuntimeError(
            f"Invalid Stage 1 candidate file:\n{path}\n"
            "Expected a top-level 'candidates' list."
        )

    return candidates


def normalize_candidate(candidate, subdir):
    """
    Normalize the small amount of Stage 1 information Stage 3 needs.

    The original candidate dictionary is retained, because Stage 3
    output should preserve Stage 1 information.
    """

    if not isinstance(candidate, dict):
        raise RuntimeError(
            f"Invalid Stage 1 candidate in subdirectory {subdir}: "
            f"expected object, got {type(candidate).__name__}"
        )

    path = candidate.get("path")
    md5 = candidate.get("md5")

    if not path:
        raise RuntimeError(
            f"Stage 1 candidate has no path in subdirectory {subdir}."
        )

    if not md5:
        md5 = Path(path).stem.lower()

    result = dict(candidate)

    result["path"] = str(path)
    result["md5"] = str(md5).lower()
    result["input_subdirectory"] = str(
        candidate.get(
            "input_subdirectory",
            subdir,
        )
    )

    return result


# ============================================================================
# INPUT FINGERPRINT
# ============================================================================

def candidate_fingerprint(candidates):
    """
    Stable fingerprint of the Stage 1 candidate ordering.

    The fingerprint is based on the ordered MD5/path sequence.

    This prevents a checkpoint from silently being applied to a
    different candidate list.
    """

    digest = hashlib.sha256()

    for candidate in candidates:
        md5 = str(
            candidate.get(
                "md5",
                "",
            )
        ).lower()

        path = str(
            candidate.get(
                "path",
                "",
            )
        )

        digest.update(
            md5.encode(
                "utf-8",
                errors="replace",
            )
        )

        digest.update(b"\0")

        digest.update(
            path.encode(
                "utf-8",
                errors="replace",
            )
        )

        digest.update(b"\0")

    return digest.hexdigest()


# ============================================================================
# STAGE 3 MUSICAL ANALYSIS
# ============================================================================

def safe_mean(values):
    if not values:
        return 0.0

    return float(
        sum(values) / len(values)
    )


def safe_median(values):
    if not values:
        return 0.0

    values = sorted(values)
    n = len(values)

    if n % 2:
        return float(
            values[n // 2]
        )

    return float(
        (
            values[n // 2 - 1]
            +
            values[n // 2]
        )
        / 2
    )


def calculate_polyphony(notes):
    """
    Same sweep-line polyphony calculation used by la_filter.py.

    Returns the time-weighted average number of simultaneously
    sounding notes.
    """

    if not notes:
        return 0.0

    events = []

    for note in notes:

        start = float(note.start)
        end = float(note.end)

        if end <= start:
            continue

        events.append(
            (
                start,
                1,
            )
        )

        events.append(
            (
                end,
                -1,
            )
        )

    if not events:
        return 0.0

    # Note-offs (-1) sort before note-ons (+1) at equal timestamps.
    events.sort(
        key=lambda event: (
            event[0],
            event[1],
        )
    )

    active = 0
    previous_time = events[0][0]

    weighted_polyphony = 0.0
    total_time = 0.0

    i = 0

    while i < len(events):

        current_time = events[i][0]

        if current_time > previous_time:

            duration = (
                current_time
                -
                previous_time
            )

            weighted_polyphony += (
                active
                *
                duration
            )

            total_time += duration

        while (
            i < len(events)
            and events[i][0] == current_time
        ):
            active += events[i][1]
            i += 1

        previous_time = current_time

    if total_time <= 0:
        return 0.0

    return float(
        weighted_polyphony
        /
        total_time
    )


def calculate_pitch_range(notes):
    if not notes:
        return 0

    pitches = [
        int(note.pitch)
        for note in notes
    ]

    return int(
        max(pitches)
        -
        min(pitches)
    )


def calculate_pitch_mean(notes):
    if not notes:
        return 0.0

    return safe_mean(
        [
            float(note.pitch)
            for note in notes
        ]
    )


def calculate_note_duration_stats(notes):

    durations = []

    for note in notes:

        duration = (
            float(note.end)
            -
            float(note.start)
        )

        if duration > 0:
            durations.append(
                duration
            )

    if not durations:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "mean":
            safe_mean(durations),

        "median":
            safe_median(durations),

        "min":
            float(min(durations)),

        "max":
            float(max(durations)),
    }


def calculate_activity_span(notes):

    if not notes:
        return 0.0

    start = min(
        float(note.start)
        for note in notes
    )

    end = max(
        float(note.end)
        for note in notes
    )

    return max(
        0.0,
        end - start,
    )


def calculate_note_density(notes):

    if not notes:
        return 0.0

    span = calculate_activity_span(
        notes
    )

    if span <= 0:
        return 0.0

    return float(
        len(notes)
        /
        span
    )


def calculate_monophonic_fraction(notes):
    """
    Same definition as la_filter.py:

        fraction of note onsets which do not have another note
        sounding simultaneously.

    Uses a min-heap rather than O(n²) note comparisons.
    """

    if not notes:
        return 0.0

    import heapq

    sorted_notes = sorted(
        notes,
        key=lambda note: (
            float(note.start),
            float(note.end),
        )
    )

    active_end_times = []

    isolated = 0

    for note in sorted_notes:

        start = float(note.start)
        end = float(note.end)

        while (
            active_end_times
            and active_end_times[0] <= start
        ):
            heapq.heappop(
                active_end_times
            )

        if not active_end_times:
            isolated += 1

        heapq.heappush(
            active_end_times,
            end,
        )

    return float(
        isolated
        /
        len(notes)
    )


def melody_score(
    notes,
    polyphony,
    pitch_range,
    activity_span,
    note_density,
    monophonic_fraction,
):
    """
    Same melody ranking heuristic as la_filter.py.
    """

    if not notes:
        return 0.0

    score = 0.0

    # --------------------------------------------------------------
    # Monophonic material.
    # --------------------------------------------------------------

    score += (
        monophonic_fraction
        *
        0.35
    )

    # --------------------------------------------------------------
    # Polyphony.
    # --------------------------------------------------------------

    if polyphony <= 1.15:

        polyphony_score = 1.0

    elif polyphony <= 1.5:

        polyphony_score = 0.75

    elif polyphony <= 2.0:

        polyphony_score = 0.45

    else:

        polyphony_score = 0.10

    score += (
        polyphony_score
        *
        0.25
    )

    # --------------------------------------------------------------
    # Useful melodic pitch range.
    # --------------------------------------------------------------

    if 12 <= pitch_range <= 48:

        range_score = 1.0

    elif 7 <= pitch_range < 12:

        range_score = 0.65

    elif 48 < pitch_range <= 60:

        range_score = 0.70

    else:

        range_score = 0.30

    score += (
        range_score
        *
        0.15
    )

    # --------------------------------------------------------------
    # Sustained activity.
    # --------------------------------------------------------------

    if activity_span > 0:

        density_score = min(
            1.0,
            note_density / 2.0,
        )

    else:

        density_score = 0.0

    score += (
        density_score
        *
        0.10
    )

    # --------------------------------------------------------------
    # Number of notes.
    # --------------------------------------------------------------

    if len(notes) >= 100:

        note_count_score = 1.0

    elif len(notes) >= 50:

        note_count_score = 0.8

    elif len(notes) >= 20:

        note_count_score = 0.6

    else:

        note_count_score = 0.3

    score += (
        note_count_score
        *
        0.05
    )

    return float(
        max(
            0.0,
            min(
                1.0,
                score,
            ),
        )
    )


def get_instrument_channels(instrument):
    """
    PrettyMIDI exposes the MIDI channel at instrument level.

    Unlike UglyMIDI, PrettyMIDI Note objects do not normally expose
    individual MIDI channels.

    Therefore Stage 3 does NOT pretend that note.channel exists.

    If an instrument exposes a channel, it is reported as metadata.
    """

    channels = []

    channel = getattr(
        instrument,
        "channel",
        None,
    )

    if channel is not None:

        try:
            channels.append(
                int(channel)
            )
        except (
            TypeError,
            ValueError,
        ):
            pass

    return sorted(
        set(channels)
    )


def analyze_midi(midi_path):
    """
    Analyze one physical MIDI.

    Returns:

        (analysis, None)

    or:

        (None, rejection_reason)
    """

    midi = None

    try:

        midi = pretty_midi.PrettyMIDI(
            str(midi_path)
        )

    except Exception as exc:

        return (
            None,
            "midi_parse_error:"
            +
            type(exc).__name__,
        )

    try:

        track_reports = []

        tracks_with_notes = 0
        total_notes = 0

        for track_index, instrument in enumerate(
            midi.instruments
        ):

            raw_notes = instrument.notes

            if not raw_notes:
                continue

            tracks_with_notes += 1

            raw_note_count = len(
                raw_notes
            )

            total_notes += (
                raw_note_count
            )

            # ----------------------------------------------------------
            # IMPORTANT:
            #
            # PrettyMIDI Notes do not reliably carry channel information.
            #
            # We therefore use the actual instrument's channel only as
            # descriptive metadata and do not manufacture note-level
            # channel filtering.
            # ----------------------------------------------------------

            filtered_notes = list(
                raw_notes
            )

            if len(filtered_notes) < MIN_NOTES_FOR_CANDIDATE:
                continue

            pitch_range = calculate_pitch_range(
                filtered_notes
            )

            polyphony = calculate_polyphony(
                filtered_notes
            )

            activity_span = calculate_activity_span(
                filtered_notes
            )

            note_density = calculate_note_density(
                filtered_notes
            )

            monophonic_fraction = calculate_monophonic_fraction(
                filtered_notes
            )

            duration_stats = calculate_note_duration_stats(
                filtered_notes
            )

            pitch_mean = calculate_pitch_mean(
                filtered_notes
            )

            score = melody_score(
                filtered_notes,
                polyphony,
                pitch_range,
                activity_span,
                note_density,
                monophonic_fraction,
            )

            channels = get_instrument_channels(
                instrument
            )

            report = {
                "track":
                    int(track_index),

                "channels":
                    channels,

                "program":
                    int(
                        getattr(
                            instrument,
                            "program",
                            0,
                        )
                    ),

                "program_name":
                    str(
                        pretty_midi.program_to_instrument_name(
                            getattr(
                                instrument,
                                "program",
                                0,
                            )
                        )
                    ),

                "is_drum":
                    bool(
                        getattr(
                            instrument,
                            "is_drum",
                            False,
                        )
                    ),

                "raw_notes":
                    int(raw_note_count),

                "non_percussion_notes":
                    int(len(filtered_notes)),

                "polyphony":
                    round(
                        float(polyphony),
                        4,
                    ),

                "monophonic_fraction":
                    round(
                        float(monophonic_fraction),
                        4,
                    ),

                "pitch_range":
                    int(pitch_range),

                "pitch_mean":
                    round(
                        float(pitch_mean),
                        2,
                    ),

                "note_density":
                    round(
                        float(note_density),
                        4,
                    ),

                "activity_span":
                    round(
                        float(activity_span),
                        4,
                    ),

                "duration":
                    duration_stats,

                "melody_score":
                    round(
                        float(score),
                        4,
                    ),
            }

            track_reports.append(
                report
            )

        # --------------------------------------------------------------
        # Rank exactly as la_filter.py.
        # --------------------------------------------------------------

        track_reports.sort(
            key=lambda item: (
                item["melody_score"],
                item["monophonic_fraction"],
                item["pitch_mean"],
            ),
            reverse=True,
        )

        track_reports = track_reports[
            :MAX_CANDIDATES_PER_FILE
        ]

        analysis = {
            "tracks_with_notes":
                int(tracks_with_notes),

            "total_notes":
                int(total_notes),

            "candidate_count":
                int(len(track_reports)),

            "candidates":
                track_reports,
        }

        return (
            analysis,
            None,
        )

    except Exception as exc:

        return (
            None,
            "midi_analysis_error:"
            +
            type(exc).__name__,
        )

    finally:

        del midi

        gc.collect()


# ============================================================================
# WORKER
# ============================================================================

def stage3_worker(candidate):
    """
    Worker entry point.

    Only compact dictionaries are returned to the parent.
    """

    path = candidate["path"]

    try:

        analysis, reason = analyze_midi(
            path
        )

        return {
            "candidate":
                candidate,

            "analysis":
                analysis,

            "reason":
                reason,
        }

    except Exception as exc:

        return {
            "candidate":
                candidate,

            "analysis":
                None,

            "reason":
                "worker_error:"
                +
                type(exc).__name__,
        }

    finally:

        gc.collect()


# ============================================================================
# CHECKPOINTS
# ============================================================================

def checkpoint_paths():
    """
    Return all checkpoint files in numeric order.

    Checkpoints are intentionally shared by all hexadecimal subdirectories.
    Each file contains its own subdirectory identifier.
    """

    paths = list(
        STAGE3_DIR.glob(
            "checkpoint*.json"
        )
    )

    def checkpoint_number(path):

        name = path.stem

        suffix = name[
            len("checkpoint"):
        ]

        try:
            return int(suffix)
        except ValueError:
            return -1

    paths.sort(
        key=checkpoint_number
    )

    return [
        path
        for path in paths
        if checkpoint_number(path) >= 1
    ]


def next_checkpoint_number():
    numbers = []

    for path in checkpoint_paths():

        suffix = path.stem[
            len("checkpoint"):
        ]

        try:
            numbers.append(
                int(suffix)
            )
        except ValueError:
            continue

    if not numbers:
        return 1

    return max(numbers) + 1


def write_checkpoint(
    checkpoint_number,
    subdir,
    input_count,
    input_fingerprint,
    completed_start,
    completed_end,
    results,
):
    """
    Write ONE batch checkpoint.

    Critically, this contains only the batch represented by this
    checkpoint. It does NOT contain every result accumulated so far.
    """

    checkpoint_path = (
        STAGE3_DIR
        /
        f"checkpoint{checkpoint_number}.json"
    )

    checkpoint = {
        "stage": "3",

        "subdirectory":
            str(subdir),

        "input_count":
            int(input_count),

        "input_fingerprint":
            str(input_fingerprint),

        "completed_start":
            int(completed_start),

        "completed_end":
            int(completed_end),

        "result_count":
            int(len(results)),

        "results":
            results,
    }

    atomic_json_write(
        checkpoint_path,
        checkpoint,
    )

    print(
        f"  checkpoint: "
        f"{checkpoint_path} "
        f"[{completed_start:,}..{completed_end:,}]"
    )

    return checkpoint_path


def load_checkpoints_for_subdir(
    subdir,
    candidates,
    input_fingerprint,
):
    """
    Reconstruct completed work for one subdirectory.

    Returns:

        completed_count
        completed_results

    Checkpoints must form a contiguous prefix of the Stage 1 input.

    This means an interrupted run can safely resume without assuming
    that checkpoint filenames themselves describe processing order.
    """

    checkpoint_files = checkpoint_paths()

    if not checkpoint_files:
        return 0, []

    expected_next = 0
    reconstructed = []

    relevant = []

    for path in checkpoint_files:

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as fh:

                checkpoint = json.load(fh)

        except Exception as exc:

            raise RuntimeError(
                f"Could not read checkpoint:\n"
                f"  {path}\n"
                f"{type(exc).__name__}: {exc}"
            )

        if str(
            checkpoint.get(
                "subdirectory",
                "",
            )
        ) != str(subdir):

            continue

        relevant.append(
            (
                path,
                checkpoint,
            )
        )

    if not relevant:
        return 0, []

    # --------------------------------------------------------------
    # Validate all checkpoints before reconstructing.
    # --------------------------------------------------------------

    for path, checkpoint in relevant:

        checkpoint_input_count = int(
            checkpoint.get(
                "input_count",
                -1,
            )
        )

        if checkpoint_input_count != len(
            candidates
        ):
            raise RuntimeError(
                "Checkpoint input-count mismatch.\n"
                f"  checkpoint: {path}\n"
                f"  checkpoint count: "
                f"{checkpoint_input_count:,}\n"
                f"  current count: "
                f"{len(candidates):,}\n"
                "\n"
                "The Stage 1 candidate list has changed."
            )

        checkpoint_fingerprint = checkpoint.get(
            "input_fingerprint"
        )

        if checkpoint_fingerprint != input_fingerprint:
            raise RuntimeError(
                "Checkpoint input-fingerprint mismatch.\n"
                f"  checkpoint: {path}\n"
                f"  checkpoint fingerprint: "
                f"{checkpoint_fingerprint}\n"
                f"  current fingerprint: "
                f"{input_fingerprint}\n"
                "\n"
                "The Stage 1 candidate ordering/content has changed."
            )

    relevant.sort(
        key=lambda item: int(
            item[1].get(
                "completed_start",
                -1,
            )
        )
    )

    for path, checkpoint in relevant:

        start = int(
            checkpoint.get(
                "completed_start",
                -1,
            )
        )

        end = int(
            checkpoint.get(
                "completed_end",
                -1,
            )
        )

        results = checkpoint.get(
            "results",
            [],
        )

        if start != expected_next:

            raise RuntimeError(
                "Checkpoint sequence gap/overlap.\n"
                f"  checkpoint: {path}\n"
                f"  expected start: "
                f"{expected_next:,}\n"
                f"  actual start: "
                f"{start:,}"
            )

        expected_count = (
            end
            -
            start
            +
            1
        )

        if expected_count != len(results):

            raise RuntimeError(
                "Checkpoint result-count mismatch.\n"
                f"  checkpoint: {path}\n"
                f"  range: {start:,}..{end:,}\n"
                f"  expected results: "
                f"{expected_count:,}\n"
                f"  actual results: "
                f"{len(results):,}"
            )

        reconstructed.extend(
            results
        )

        expected_next = end + 1

    # --------------------------------------------------------------
    # Verify reconstructed results against current input ordering.
    # --------------------------------------------------------------

    if len(reconstructed) != expected_next:
        raise RuntimeError(
            "Internal checkpoint reconstruction error."
        )

    for index, result in enumerate(
        reconstructed
    ):

        candidate = result.get(
            "candidate",
            {}
        )

        expected_candidate = candidates[
            index
        ]

        if str(
            candidate.get(
                "md5",
                ""
            )
        ).lower() != str(
            expected_candidate.get(
                "md5",
                ""
            )
        ).lower():

            raise RuntimeError(
                "Checkpoint candidate ordering mismatch.\n"
                f"  subdirectory: {subdir}\n"
                f"  candidate index: {index:,}\n"
                f"  checkpoint MD5: "
                f"{candidate.get('md5')}\n"
                f"  current MD5: "
                f"{expected_candidate.get('md5')}"
            )

    return (
        expected_next,
        reconstructed,
    )


# ============================================================================
# RESULT ACCUMULATION
# ============================================================================

def summarize_results(
    results,
):
    """
    Convert worker results into final Stage 3 structures.

    Stage 3 is fundamentally a ranking stage.

    A file with at least one ranked track is a Stage 3 survivor.

    A successfully parsed file with zero qualifying tracks is rejected
    as having no usable melody candidate.

    A parse/worker error is also recorded as a rejection.
    """

    reports = []
    survivors = []
    rejection_details = []
    rejection_counts = Counter()

    for result in results:

        candidate = result[
            "candidate"
        ]

        analysis = result.get(
            "analysis"
        )

        reason = result.get(
            "reason"
        )

        if reason is not None:

            rejection_counts[
                reason
            ] += 1

            rejection_details.append(
                {
                    "md5":
                        candidate["md5"],

                    "path":
                        candidate["path"],

                    "reason":
                        reason,
                }
            )

            continue

        stage3_candidate = dict(
            candidate
        )

        stage3_candidate[
            "stage3"
        ] = analysis

        reports.append(
            stage3_candidate
        )

        if analysis[
            "candidate_count"
        ] > 0:

            survivors.append(
                candidate["path"]
            )

        else:

            rejection_counts[
                "no_melody_candidate"
            ] += 1

            rejection_details.append(
                {
                    "md5":
                        candidate["md5"],

                    "path":
                        candidate["path"],

                    "reason":
                        "no_melody_candidate",
                }
            )

    return (
        reports,
        survivors,
        rejection_counts,
        rejection_details,
    )


def merge_checkpoint_results(
    results,
):
    """
    Sort reconstructed results by their original Stage 1 order.
    """

    return list(results)


# ============================================================================
# FINAL OUTPUT
# ============================================================================

def output_paths(subdir):

    return {
        "candidates_txt":
            STAGE3_DIR
            /
            f"candidates_{subdir}.txt",

        "candidates_json":
            STAGE3_DIR
            /
            f"candidates_{subdir}.json",

        "rejections_json":
            STAGE3_DIR
            /
            f"rejections_{subdir}.json",

        "summary_json":
            STAGE3_DIR
            /
            f"summary_{subdir}.json",
    }


def write_final_outputs(
    subdir,
    candidates,
    results,
):
    """
    Write the four independent Stage 3 outputs for one hexadecimal
    input directory.
    """

    (
        reports,
        survivors,
        rejection_counts,
        rejection_details,
    ) = summarize_results(
        results
    )

    paths = output_paths(
        subdir
    )

    input_count = len(
        candidates
    )

    survivor_count = len(
        survivors
    )

    rejected_count = (
        input_count
        -
        survivor_count
    )

    # --------------------------------------------------------------
    # candidates_<subdir>.txt
    # --------------------------------------------------------------

    with open(
        paths["candidates_txt"],
        "w",
        encoding="utf-8",
    ) as fh:

        for survivor in survivors:

            fh.write(
                survivor
                +
                "\n"
            )

    # --------------------------------------------------------------
    # candidates_<subdir>.json
    # --------------------------------------------------------------

    candidate_output = {
        "dataset":
            "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

        "stage":
            "3",

        "input_subdirectory":
            str(subdir),

        "description":
            (
                "Stage 3 MIDI musical-structure analysis and "
                "melody-candidate ranking. Stage 3 consumes "
                "Stage 1 output directly; Stage 2 is not used."
            ),

        "input":
            str(
                STAGE1_DIR
                /
                f"candidates_{subdir}.json"
            ),

        "midi_root":
            str(
                MIDI_ROOT
                /
                str(subdir)
            ),

        "workers":
            WORKERS,

        "min_notes_for_candidate":
            MIN_NOTES_FOR_CANDIDATE,

        "max_candidates_per_file":
            MAX_CANDIDATES_PER_FILE,

        "input_candidates":
            input_count,

        "survivors":
            survivor_count,

        "rejected":
            rejected_count,

        "survivor_ratio":
            (
                survivor_count / input_count
                if input_count
                else 0.0
            ),

        "candidates":
            survivors,

        "reports":
            reports,
    }

    atomic_json_write(
        paths["candidates_json"],
        candidate_output,
    )

    # --------------------------------------------------------------
    # rejections_<subdir>.json
    # --------------------------------------------------------------

    rejection_output = {
        "dataset":
            "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

        "stage":
            "3",

        "input_subdirectory":
            str(subdir),

        "input_candidates":
            input_count,

        "rejected":
            rejected_count,

        "rejection_counts":
            dict(
                rejection_counts
            ),

        "rejections":
            rejection_details,
    }

    atomic_json_write(
        paths["rejections_json"],
        rejection_output,
    )

    # --------------------------------------------------------------
    # summary_<subdir>.json
    # --------------------------------------------------------------

    summary = {
        "dataset":
            "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

        "stage":
            "3",

        "input_subdirectory":
            str(subdir),

        "description":
            (
                "Direct MIDI musical-structure analysis and "
                "melody-candidate ranking."
            ),

        "input":
            str(
                STAGE1_DIR
                /
                f"candidates_{subdir}.json"
            ),

        "output":
            str(
                paths["candidates_txt"]
            ),

        "input_candidates":
            input_count,

        "survivors":
            survivor_count,

        "rejected":
            rejected_count,

        "survivor_ratio":
            (
                survivor_count / input_count
                if input_count
                else 0.0
            ),

        "workers":
            WORKERS,

        "checkpoint_interval":
            CHECKPOINT_INTERVAL,

        "min_notes_for_candidate":
            MIN_NOTES_FOR_CANDIDATE,

        "max_candidates_per_file":
            MAX_CANDIDATES_PER_FILE,

        "rejection_counts":
            dict(
                rejection_counts
            ),

        "outputs": {
            "candidate_paths":
                str(
                    paths["candidates_txt"]
                ),

            "candidate_json":
                str(
                    paths["candidates_json"]
                ),

            "rejections_json":
                str(
                    paths["rejections_json"]
                ),

            "summary_json":
                str(
                    paths["summary_json"]
                ),
        },
    }

    atomic_json_write(
        paths["summary_json"],
        summary,
    )

    return (
        survivor_count,
        rejected_count,
        rejection_counts,
    )


# ============================================================================
# PROCESS ONE HEX SUBDIRECTORY
# ============================================================================

def process_subdirectory(
    subdir,
):
    print()
    print("=" * 78)
    print(
        f"STAGE 3 — INPUT SUBDIRECTORY {subdir}"
    )
    print("=" * 78)
    print()

    input_file = (
        STAGE1_DIR
        /
        f"candidates_{subdir}.json"
    )

    midi_dir = (
        MIDI_ROOT
        /
        str(subdir)
    )

    print(
        f"Input candidates : {input_file}"
    )

    print(
        f"MIDI root        : {midi_dir}"
    )

    print(
        f"Output directory : {STAGE3_DIR}"
    )

    print(
        f"Workers          : {WORKERS}"
    )

    print(
        f"Max pending      : {MAX_PENDING}"
    )

    print()

    candidates = load_stage1_candidates(
        subdir
    )

    candidates = [
        normalize_candidate(
            candidate,
            subdir,
        )
        for candidate in candidates
    ]

    input_count = len(
        candidates
    )

    print(
        f"Stage 1 candidates: "
        f"{input_count:,}"
    )

    if input_count == 0:

        print(
            "No candidates. Writing empty Stage 3 output."
        )

        write_final_outputs(
            subdir,
            candidates,
            [],
        )

        return

    input_fingerprint = candidate_fingerprint(
        candidates
    )

    print(
        f"Input fingerprint : "
        f"{input_fingerprint}"
    )

    # --------------------------------------------------------------
    # Resume from existing checkpoints.
    # --------------------------------------------------------------

    completed_count, completed_results = (
        load_checkpoints_for_subdir(
            subdir,
            candidates,
            input_fingerprint,
        )
    )

    if completed_count:

        print()
        print(
            "CHECKPOINT RESUME"
        )
        print(
            "-" * 78
        )

        print(
            f"Completed already : "
            f"{completed_count:,}"
        )

        print(
            f"Remaining         : "
            f"{input_count - completed_count:,}"
        )

        print()

    if completed_count >= input_count:

        print(
            "All input files already completed "
            "according to checkpoints."
        )

        results = completed_results

    else:

        remaining = candidates[
            completed_count:
        ]

        next_checkpoint = (
            next_checkpoint_number()
        )

        current_completed = (
            completed_count
        )

        batch_results = []

        # ----------------------------------------------------------
        # Bounded ProcessPool.
        # ----------------------------------------------------------

        with ProcessPoolExecutor(
            max_workers=WORKERS
        ) as executor:

            pending = {}
            next_index = 0

            # ------------------------------------------------------
            # Initial bounded submission.
            # ------------------------------------------------------

            while (
                next_index < len(remaining)
                and len(pending) < MAX_PENDING
            ):

                candidate = remaining[
                    next_index
                ]

                future = executor.submit(
                    stage3_worker,
                    candidate,
                )

                pending[
                    future
                ] = (
                    current_completed
                    +
                    next_index
                )

                next_index += 1

            # ------------------------------------------------------
            # Consume results and refill queue.
            # ------------------------------------------------------

            while pending:

                done, _ = wait(
                    pending,
                    return_when=FIRST_COMPLETED,
                )

                for future in done:

                    original_index = pending.pop(
                        future
                    )

                    try:

                        result = future.result()

                    except Exception as exc:

                        candidate = candidates[
                            original_index
                        ]

                        result = {
                            "candidate":
                                candidate,

                            "analysis":
                                None,

                            "reason":
                                "future_error:"
                                +
                                type(exc).__name__,
                        }

                    batch_results.append(
                        (
                            original_index,
                            result,
                        )
                    )

                    # --------------------------------------------------
                    # Refill immediately.
                    # --------------------------------------------------

                    if next_index < len(remaining):

                        candidate = remaining[
                            next_index
                        ]

                        future = executor.submit(
                            stage3_worker,
                            candidate,
                        )

                        pending[
                            future
                        ] = (
                            current_completed
                            +
                            next_index
                        )

                        next_index += 1

                    # --------------------------------------------------
                    # Checkpoint only contiguous completed prefix.
                    #
                    # Futures finish out of order. We therefore sort
                    # the accumulated batch by original index and write
                    # only a contiguous prefix.
                    # --------------------------------------------------

                    batch_results.sort(
                        key=lambda item: item[0]
                    )

                    while batch_results:

                        first_index = batch_results[0][0]

                        expected_index = (
                            current_completed
                            +
                            len(
                                completed_results
                            )
                            -
                            current_completed
                        )

                        # Simpler explicit contiguous position:
                        expected_index = (
                            current_completed
                            +
                            sum(
                                1
                                for _ in []
                            )
                        )

                        break

                # ------------------------------------------------------
                # Checkpoint outside the individual future loop.
                #
                # Reconstruct the contiguous sequence beginning at
                # current_completed.
                # ------------------------------------------------------

                batch_results.sort(
                    key=lambda item: item[0]
                )

                contiguous = []

                expected_index = (
                    current_completed
                )

                for index, result in batch_results:

                    if index != expected_index:
                        break

                    contiguous.append(
                        result
                    )

                    expected_index += 1

                if (
                    len(contiguous)
                    >=
                    CHECKPOINT_INTERVAL
                ):

                    checkpoint_results = (
                        contiguous[
                            :CHECKPOINT_INTERVAL
                        ]
                    )

                    start = (
                        current_completed
                        +
                        1
                    )

                    end = (
                        current_completed
                        +
                        len(
                            checkpoint_results
                        )
                    )

                    write_checkpoint(
                        next_checkpoint,
                        subdir,
                        input_count,
                        input_fingerprint,
                        start,
                        end,
                        checkpoint_results,
                    )

                    next_checkpoint += 1

                    completed_results.extend(
                        checkpoint_results
                    )

                    current_completed = end

                    # Remove the checkpointed prefix.
                    batch_results = (
                        batch_results[
                            len(
                                checkpoint_results
                            ):
                        ]
                    )

                    del checkpoint_results
                    del contiguous

                    gc.collect()

            # ------------------------------------------------------
            # After the executor exits, every remaining result exists.
            # Write the final partial checkpoint if necessary.
            # ------------------------------------------------------

            batch_results.sort(
                key=lambda item: item[0]
            )

            remaining_results = [
                result
                for _, result in batch_results
            ]

            if remaining_results:

                start = (
                    current_completed
                    +
                    1
                )

                end = (
                    current_completed
                    +
                    len(
                        remaining_results
                    )
                )

                write_checkpoint(
                    next_checkpoint,
                    subdir,
                    input_count,
                    input_fingerprint,
                    start,
                    end,
                    remaining_results,
                )

                completed_results.extend(
                    remaining_results
                )

                current_completed = end

        results = completed_results

    # --------------------------------------------------------------
    # Verify every input was processed.
    # --------------------------------------------------------------

    if len(results) != input_count:

        raise RuntimeError(
            "Stage 3 completed with an unexpected result count.\n"
            f"  input files : {input_count:,}\n"
            f"  results     : {len(results):,}"
        )

    # --------------------------------------------------------------
    # Verify ordering one final time.
    # --------------------------------------------------------------

    for index, result in enumerate(
        results
    ):

        expected_md5 = str(
            candidates[index]["md5"]
        ).lower()

        actual_md5 = str(
            result["candidate"].get(
                "md5",
                "",
            )
        ).lower()

        if actual_md5 != expected_md5:

            raise RuntimeError(
                "Final Stage 3 result ordering mismatch.\n"
                f"  index       : {index:,}\n"
                f"  expected MD5: {expected_md5}\n"
                f"  actual MD5  : {actual_md5}"
            )

    # --------------------------------------------------------------
    # Write final outputs.
    # --------------------------------------------------------------

    (
        survivor_count,
        rejected_count,
        rejection_counts,
    ) = write_final_outputs(
        subdir,
        candidates,
        results,
    )

    print()
    print("=" * 78)
    print(
        f"STAGE 3 COMPLETE — SUBDIRECTORY {subdir}"
    )
    print("=" * 78)
    print()

    print(
        f"Input candidates : "
        f"{input_count:,}"
    )

    print(
        f"Survivors        : "
        f"{survivor_count:,}"
    )

    print(
        f"Rejected         : "
        f"{rejected_count:,}"
    )

    if input_count:

        print(
            f"Survivor ratio   : "
            f"{survivor_count / input_count:.4f}"
        )

    print()
    print(
        "Rejection reasons:"
    )
    print(
        "-" * 78
    )

    for reason, count in (
        rejection_counts.most_common()
    ):

        print(
            f"{reason:40s}"
            f"{count:10,}"
        )

    print()

    print(
        "Outputs:"
    )

    for path in output_paths(
        subdir
    ).values():

        print(
            f"  {path}"
        )

    print()


# ============================================================================
# CHECKPOINT CLEANUP
# ============================================================================

def delete_all_checkpoints():
    """
    Delete checkpoint*.json ONLY after the complete Stage 3 run
    has succeeded.
    """

    paths = checkpoint_paths()

    if not paths:
        return

    print()
    print("=" * 78)
    print("Removing Stage 3 checkpoints")
    print("=" * 78)
    print()

    for path in paths:

        try:

            path.unlink()

            print(
                f"  deleted: {path}"
            )

        except OSError as exc:

            raise RuntimeError(
                f"Could not delete checkpoint:\n"
                f"{path}\n"
                f"{type(exc).__name__}: {exc}"
            )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 78)
    print("Los Angeles MIDI Dataset")
    print("STAGE 3 — DIRECT MIDI MUSICAL STRUCTURE ANALYSIS")
    print("=" * 78)
    print()

    print(
        "Stage 2 is DISABLED."
    )

    print(
        "Stage 3 input = Stage 1 candidates directly."
    )

    print()

    if not STAGE1_DIR.exists():

        raise RuntimeError(
            f"Stage 1 directory does not exist:\n"
            f"{STAGE1_DIR}"
        )

    if not MIDI_ROOT.exists():

        raise RuntimeError(
            f"MIDI root does not exist:\n"
            f"{MIDI_ROOT}"
        )

    STAGE3_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        f"Stage 1 directory : "
        f"{STAGE1_DIR}"
    )

    print(
        f"MIDI root         : "
        f"{MIDI_ROOT}"
    )

    print(
        f"Stage 3 directory  : "
        f"{STAGE3_DIR}"
    )

    print(
        f"Workers            : "
        f"{WORKERS}"
    )

    print(
        f"Max pending        : "
        f"{MAX_PENDING}"
    )

    print(
        f"Checkpoint interval: "
        f"{CHECKPOINT_INTERVAL}"
    )

    print()

    try:

        for subdir in INPUT_SUBDIRECTORIES:

            input_file = (
                STAGE1_DIR
                /
                f"candidates_{subdir}.json"
            )

            if not input_file.is_file():

                print()
                print(
                    f"Skipping {subdir}: "
                    f"{input_file} does not exist."
                )

                continue

            process_subdirectory(
                subdir
            )

        # ----------------------------------------------------------
        # THIS IS THE ONLY PLACE WHERE CHECKPOINTS ARE DELETED.
        #
        # If anything above raises, this code is not reached.
        # Therefore interrupted/failed runs retain checkpoints.
        # ----------------------------------------------------------

        delete_all_checkpoints()

        print()
        print("=" * 78)
        print("STAGE 3 — ALL SUBDIRECTORIES COMPLETE")
        print("=" * 78)
        print()

        print(
            "All checkpoint files were successfully removed."
        )

        print()

    except KeyboardInterrupt:

        print()
        print("=" * 78)
        print("STAGE 3 INTERRUPTED")
        print("=" * 78)
        print()

        print(
            "Checkpoints have been preserved."
        )

        print(
            "Re-run stage3.py to resume."
        )

        print()

        raise

    except Exception:

        print()
        print("=" * 78)
        print("STAGE 3 FAILED")
        print("=" * 78)
        print()

        print(
            "Existing checkpoints have been preserved."
        )

        print(
            "Fix the problem and re-run stage3.py."
        )

        print()

        raise


if __name__ == "__main__":
    main()
