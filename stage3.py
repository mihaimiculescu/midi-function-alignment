#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Los Angeles MIDI Dataset
Stage 3 — Direct physical-MIDI-track musical analysis

Stage 3 deliberately:
    * consumes Stage 1 candidates directly
    * never uses Stage 2
    * never uses LAMDa metadata
    * never uses PrettyMIDI / UglyMIDI
    * never re-maps physical MIDI tracks into logical instruments
    * removes only MIDI channel 9 notes before ranking
    * preserves the actual physical MIDI track number

For every physical MIDI track:
    1. Count the track if it contains any note events.
    2. Mark is_drum=True if the track contains channel-9 notes.
    3. Remove channel-9 notes.
    4. If fewer than MIN_NOTES_FOR_CANDIDATE notes remain, do not rank it.
    5. Otherwise calculate the same musical features and melody score used
       by la_filter.py.
    6. Rank by melody_score descending and retain at most 8 tracks.

File-level rule:
    candidate_count == 0 -> reject
    candidate_count == 1 -> reject
    candidate_count >= 2 -> survive

Checkpoints:
    checkpoint<number>.json
    Each checkpoint contains only one newly completed contiguous batch.
    Checkpoints remain after interruption/failure.
    ALL checkpoints are deleted only after the entire Stage 3 run succeeds.

Memory:
    * 24 workers by default.
    * At most WORKERS*2 futures are outstanding.
    * Workers return only compact dictionaries.
    * Large MIDI structures are explicitly deleted and gc.collect() is called.
"""

import gc
import hashlib
import json
import os
import sys
import tempfile
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

# Bound the number of futures/results held by the parent process.
MAX_PENDING = WORKERS * 2

# One checkpoint per 1000 completed input files.
CHECKPOINT_INTERVAL = 1000

# Same candidate limits as la_filter.py.
MIN_NOTES_FOR_CANDIDATE = 8
MAX_CANDIDATES_PER_FILE = 8

INPUT_SUBDIRECTORIES = tuple(
    "123456789abcdef"
)

# MIDI.py uses zero-based MIDI channels.
# GM MIDI channel 10 therefore appears as channel 9.
GM_DRUM_CHANNEL = 9


# ============================================================================
# JSON
# ============================================================================

def atomic_json_write(path, data):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

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
            os.fsync(
                fh.fileno()
            )

            temporary_path = Path(
                fh.name
            )

        os.replace(
            temporary_path,
            path,
        )

    finally:

        if (
            temporary_path is not None
            and temporary_path.exists()
        ):
            try:
                temporary_path.unlink()
            except OSError:
                pass


# ============================================================================
# STAGE 1 INPUT
# ============================================================================

def load_stage1_candidates(subdir):
    """
    Stage 1 is the ONLY Stage 3 input.

    Stage 2 is deliberately not consulted.
    """

    path = (
        STAGE1_DIR
        /
        f"candidates_{subdir}.json"
    )

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

    candidates = data.get(
        "candidates"
    )

    if not isinstance(
        candidates,
        list,
    ):
        raise RuntimeError(
            f"Invalid Stage 1 candidate file:\n{path}\n"
            "Expected top-level 'candidates' list."
        )

    normalized = []

    for candidate in candidates:

        if not isinstance(
            candidate,
            dict,
        ):
            raise RuntimeError(
                f"Invalid Stage 1 candidate in {path}"
            )

        if not candidate.get(
            "path"
        ):
            raise RuntimeError(
                f"Stage 1 candidate has no path in {path}"
            )

        item = dict(
            candidate
        )

        item["path"] = str(
            item["path"]
        )

        item["md5"] = str(
            item.get(
                "md5",
                Path(
                    item["path"]
                ).stem,
            )
        ).lower()

        item["input_subdirectory"] = str(
            item.get(
                "input_subdirectory",
                subdir,
            )
        )

        normalized.append(
            item
        )

    return normalized


def candidate_fingerprint(candidates):
    """
    Stable fingerprint of the exact ordered Stage 1 input list.
    """

    digest = hashlib.sha256()

    for candidate in candidates:

        digest.update(
            str(
                candidate.get(
                    "md5",
                    "",
                )
            )
            .lower()
            .encode(
                "utf-8",
                errors="replace",
            )
        )

        digest.update(
            b"\0"
        )

        digest.update(
            str(
                candidate.get(
                    "path",
                    "",
                )
            )
            .encode(
                "utf-8",
                errors="replace",
            )
        )

        digest.update(
            b"\0"
        )

    return digest.hexdigest()


# ============================================================================
# MUSICAL CALCULATIONS
# ============================================================================

# A note is represented compactly as:
#
#     (start_seconds, end_seconds, pitch)
#
# This avoids constructing PrettyMIDI Note objects.


def safe_mean(values):

    if not values:
        return 0.0

    return float(
        sum(values)
        /
        len(values)
    )


def safe_median(values):

    if not values:
        return 0.0

    values = sorted(
        values
    )

    n = len(
        values
    )

    if n % 2:
        return float(
            values[
                n // 2
            ]
        )

    return float(
        (
            values[
                n // 2 - 1
            ]
            +
            values[
                n // 2
            ]
        )
        /
        2
    )


def calculate_polyphony(notes):
    """
    Same sweep-line calculation as la_filter.py.

    Returns time-weighted average simultaneous-note count.
    """

    events = []

    for start, end, pitch in notes:

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

    # Note-offs before note-ons at identical timestamps.
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
            and
            events[i][0] == current_time
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


def calculate_monophonic_fraction(notes):
    """
    Same definition as la_filter.py.

    Fraction of note onsets for which no other note is sounding.
    """

    if not notes:
        return 0.0

    import heapq

    sorted_notes = sorted(
        notes,
        key=lambda note: (
            float(note[0]),
            float(note[1]),
        ),
    )

    active_end_times = []

    isolated = 0

    for start, end, pitch in sorted_notes:

        while (
            active_end_times
            and
            active_end_times[0] <= start
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


def calculate_pitch_range(notes):

    if not notes:
        return 0

    pitches = [
        int(note[2])
        for note in notes
    ]

    return int(
        max(pitches)
        -
        min(pitches)
    )


def calculate_pitch_mean(notes):

    return safe_mean(
        [
            float(note[2])
            for note in notes
        ]
    )


def calculate_activity_span(notes):

    if not notes:
        return 0.0

    start = min(
        float(note[0])
        for note in notes
    )

    end = max(
        float(note[1])
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


def calculate_note_duration_stats(notes):

    durations = []

    for note in notes:

        duration = (
            float(note[1])
            -
            float(note[0])
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
            safe_mean(
                durations
            ),

        "median":
            safe_median(
                durations
            ),

        "min":
            float(
                min(durations)
            ),

        "max":
            float(
                max(durations)
            ),
    }


def melody_score(
    notes,
    polyphony,
    pitch_range,
    activity_span,
    note_density,
    monophonic_fraction,
):
    """
    Exact scoring model currently used by la_filter.py.
    """

    if not notes:
        return 0.0

    score = 0.0

    # Monophonic material.
    score += (
        monophonic_fraction
        *
        0.35
    )

    # Polyphony.
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

    # Useful melodic pitch range.
    if (
        12
        <=
        pitch_range
        <=
        48
    ):

        range_score = 1.0

    elif (
        7
        <=
        pitch_range
        <
        12
    ):

        range_score = 0.65

    elif (
        48
        <
        pitch_range
        <=
        60
    ):

        range_score = 0.70

    else:

        range_score = 0.30

    score += (
        range_score
        *
        0.15
    )

    # Sustained activity.
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

    # Number of notes.
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


# ============================================================================
# DIRECT PHYSICAL MIDI ANALYSIS
# ============================================================================

def analyze_midi(midi_path):
    """
    Analyze the ORIGINAL physical MIDI tracks.

    MIDI.py score representation is used because Stage 1 already uses it.
    score[1:] corresponds to the physical MIDI tracks.

    No PrettyMIDI/UglyMIDI processing occurs here.
    """

    try:

        import MIDI

        with open(
            midi_path,
            "rb",
        ) as fh:

            midi_data = fh.read()

        opus = MIDI.midi2opus(
            midi_data
        )

        del midi_data

        # Convert each physical track to millisecond score events.
        ms_score = MIDI.opus2score(
            MIDI.to_millisecs(
                opus
            )
        )

        del opus

    except Exception as exc:

        gc.collect()

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

        # IMPORTANT:
        #
        # score[1:] is the physical MIDI-track sequence.
        #
        # Therefore enumerate() here produces the REAL MIDI track number.
        #
        for track_index, track in enumerate(
            ms_score[1:]
        ):

            raw_note_events = [
                event
                for event in track
                if (
                    event
                    and
                    event[0] == "note"
                )
            ]

            # A physical track with no notes is not counted.
            if not raw_note_events:
                continue

            tracks_with_notes += 1

            raw_note_count = len(
                raw_note_events
            )

            total_notes += (
                raw_note_count
            )

            # --------------------------------------------------------------
            # Build non-percussion notes.
            #
            # MIDI event format:
            #
            #     ["note", start, duration, channel, pitch, velocity]
            #
            # Channel 9 is GM percussion.
            # --------------------------------------------------------------

            filtered_notes = []

            is_drum = False

            for event in raw_note_events:

                try:

                    start_ms = float(
                        event[1]
                    )

                    duration_ms = float(
                        event[2]
                    )

                    channel = int(
                        event[3]
                    )

                    pitch = int(
                        event[4]
                    )

                except (
                    TypeError,
                    ValueError,
                    IndexError,
                ):
                    # Malformed note: it cannot participate in ranking.
                    continue

                if channel == GM_DRUM_CHANNEL:

                    is_drum = True

                    # Remove percussion note before ALL ranking calculations.
                    continue

                start_seconds = (
                    start_ms
                    /
                    1000.0
                )

                end_seconds = (
                    start_ms
                    +
                    duration_ms
                ) / 1000.0

                filtered_notes.append(
                    (
                        start_seconds,
                        end_seconds,
                        pitch,
                    )
                )

            # --------------------------------------------------------------
            # Fewer than 8 non-percussion notes:
            # track remains a real note-bearing track, but does not qualify
            # for candidate ranking.
            # --------------------------------------------------------------

            if (
                len(filtered_notes)
                <
                MIN_NOTES_FOR_CANDIDATE
            ):
                continue

            pitch_range = calculate_pitch_range(
                filtered_notes
            )

            pitch_mean = calculate_pitch_mean(
                filtered_notes
            )

            polyphony = calculate_polyphony(
                filtered_notes
            )

            monophonic_fraction = calculate_monophonic_fraction(
                filtered_notes
            )

            activity_span = calculate_activity_span(
                filtered_notes
            )

            note_density = calculate_note_density(
                filtered_notes
            )

            duration_stats = calculate_note_duration_stats(
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

            # --------------------------------------------------------------
            # Program/channel fields are deliberately NOT inferred.
            #
            # They are irrelevant to this Stage 3 selection logic.
            # Keep the output schema, but do not let absent/inconsistent
            # program metadata affect ranking.
            # --------------------------------------------------------------

            track_reports.append(
                {
                    "track":
                        int(track_index),

                    "channels":
                        [],

                    "program":
                        0,

                    "program_name":
                        "Acoustic Grand Piano",

                    "is_drum":
                        bool(is_drum),

                    "raw_notes":
                        int(raw_note_count),

                    "non_percussion_notes":
                        int(
                            len(
                                filtered_notes
                            )
                        ),

                    "polyphony":
                        round(
                            float(
                                polyphony
                            ),
                            4,
                        ),

                    "monophonic_fraction":
                        round(
                            float(
                                monophonic_fraction
                            ),
                            4,
                        ),

                    "pitch_range":
                        int(
                            pitch_range
                        ),

                    "pitch_mean":
                        round(
                            float(
                                pitch_mean
                            ),
                            2,
                        ),

                    "note_density":
                        round(
                            float(
                                note_density
                            ),
                            4,
                        ),

                    "activity_span":
                        round(
                            float(
                                activity_span
                            ),
                            4,
                        ),

                    "duration":
                        duration_stats,

                    "melody_score":
                        round(
                            float(
                                score
                            ),
                            4,
                        ),
                }
            )

        # --------------------------------------------------------------
        # EXACT ranking order of la_filter.py:
        #
        # melody_score descending.
        #
        # No PrettyMIDI/instrument ordering and no additional tie-breaker.
        # --------------------------------------------------------------

        track_reports.sort(
            key=lambda item:
                item["melody_score"],
            reverse=True,
        )

        # Keep at most 8.
        track_reports = track_reports[
            :MAX_CANDIDATES_PER_FILE
        ]

        analysis = {
            "tracks_with_notes":
                int(
                    tracks_with_notes
                ),

            "total_notes":
                int(
                    total_notes
                ),

            "candidate_count":
                int(
                    len(
                        track_reports
                    )
                ),

            "candidates":
                track_reports,
        }

        del ms_score

        gc.collect()

        return (
            analysis,
            None,
        )

    except Exception as exc:

        try:
            del ms_score
        except UnboundLocalError:
            pass

        gc.collect()

        return (
            None,
            "midi_analysis_error:"
            +
            type(exc).__name__,
        )


# ============================================================================
# WORKER
# ============================================================================

def stage3_worker(candidate):

    try:

        analysis, reason = analyze_midi(
            candidate["path"]
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


def classify_result(result):
    """
    Convert one worker result into either:
        report
    or:
        rejection
    """

    candidate = result[
        "candidate"
    ]

    analysis = result[
        "analysis"
    ]

    reason = result[
        "reason"
    ]

    if reason is not None:

        return {
            "kind":
                "rejection",

            "record":
                {
                    "md5":
                        candidate[
                            "md5"
                        ],

                    "path":
                        candidate[
                            "path"
                        ],

                    "reason":
                        reason,
                },
        }

    enriched = dict(
        candidate
    )

    enriched[
        "stage3"
    ] = analysis

    candidate_count = int(
        analysis[
            "candidate_count"
        ]
    )

    # --------------------------------------------------------------
    # Stage 3 file-level rule:
    #
    # 0 candidates -> reject
    # 1 candidate  -> reject
    # 2+ candidates -> survive
    # --------------------------------------------------------------

    if candidate_count < 2:

        if candidate_count == 1:

            rejection_reason = (
                "fewer_than_two_melody_candidates"
            )

        else:

            rejection_reason = (
                "no_melody_candidates"
            )

        return {
            "kind":
                "rejection",

            "record":
                {
                    "md5":
                        candidate[
                            "md5"
                        ],

                    "path":
                        candidate[
                            "path"
                        ],

                    "reason":
                        rejection_reason,

                    "candidate_count":
                        candidate_count,
                },
        }

    return {
        "kind":
            "report",

        "record":
            enriched,
    }


# ============================================================================
# CHECKPOINTS
# ============================================================================

def checkpoint_paths():

    paths = []

    for path in STAGE3_DIR.glob(
        "checkpoint*.json"
    ):

        suffix = path.stem[
            len("checkpoint"):
        ]

        if suffix.isdigit():

            paths.append(
                path
            )

    return sorted(
        paths,
        key=lambda path:
            int(
                path.stem[
                    len("checkpoint"):
                ]
            ),
    )


def next_checkpoint_number():

    paths = checkpoint_paths()

    if not paths:
        return 1

    return (
        int(
            paths[-1].stem[
                len("checkpoint"):
            ]
        )
        +
        1
    )


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
    Write ONLY the newly completed contiguous batch.

    completed_start/completed_end are zero-based half-open indexes.
    """

    checkpoint_path = (
        STAGE3_DIR
        /
        f"checkpoint{checkpoint_number}.json"
    )

    checkpoint = {
        "stage":
            "3",

        "subdirectory":
            str(subdir),

        "input_count":
            int(input_count),

        "input_fingerprint":
            str(
                input_fingerprint
            ),

        "completed_start":
            int(
                completed_start
            ),

        "completed_end":
            int(
                completed_end
            ),

        "result_count":
            int(
                len(results)
            ),

        "results":
            results,
    }

    atomic_json_write(
        checkpoint_path,
        checkpoint,
    )

    print(
        f"  checkpoint{checkpoint_number}.json "
        f"[{completed_start + 1:,}..{completed_end:,}]"
    )


def load_checkpoints(
    subdir,
    candidates,
    input_fingerprint,
):
    """
    Reconstruct the completed prefix for one subdirectory.

    Checkpoints must form a contiguous prefix.
    """

    relevant = []

    for path in checkpoint_paths():

        try:

            with open(
                path,
                "r",
                encoding="utf-8",
            ) as fh:

                checkpoint = json.load(
                    fh
                )

        except Exception as exc:

            raise RuntimeError(
                f"Could not read checkpoint:\n"
                f"{path}\n"
                f"{type(exc).__name__}: {exc}"
            )

        if str(
            checkpoint.get(
                "subdirectory",
                "",
            )
        ) != str(
            subdir
        ):
            continue

        if int(
            checkpoint.get(
                "input_count",
                -1,
            )
        ) != len(
            candidates
        ):

            raise RuntimeError(
                f"Checkpoint input count mismatch:\n"
                f"{path}"
            )

        if checkpoint.get(
            "input_fingerprint"
        ) != input_fingerprint:

            raise RuntimeError(
                f"Checkpoint input fingerprint mismatch:\n"
                f"{path}"
            )

        relevant.append(
            (
                path,
                checkpoint,
            )
        )

    relevant.sort(
        key=lambda item:
            int(
                item[1][
                    "completed_start"
                ]
            )
    )

    reconstructed = []

    expected_start = 0

    for path, checkpoint in relevant:

        start = int(
            checkpoint[
                "completed_start"
            ]
        )

        end = int(
            checkpoint[
                "completed_end"
            ]
        )

        results = checkpoint.get(
            "results",
            [],
        )

        if start != expected_start:

            raise RuntimeError(
                f"Checkpoint sequence gap/overlap for "
                f"subdirectory {subdir}:\n"
                f"  checkpoint: {path}\n"
                f"  expected start: {expected_start}\n"
                f"  actual start: {start}"
            )

        if len(results) != (
            end - start
        ):

            raise RuntimeError(
                f"Checkpoint result count mismatch:\n"
                f"{path}"
            )

        reconstructed.extend(
            results
        )

        expected_start = end

    # Verify reconstructed order against Stage 1.

    for index, result in enumerate(
        reconstructed
    ):

        actual_md5 = str(
            result[
                "record"
            ].get(
                "md5",
                "",
            )
        ).lower()

        expected_md5 = str(
            candidates[
                index
            ].get(
                "md5",
                "",
            )
        ).lower()

        if actual_md5 != expected_md5:

            raise RuntimeError(
                "Checkpoint ordering mismatch:\n"
                f"  index: {index}\n"
                f"  checkpoint MD5: {actual_md5}\n"
                f"  expected MD5: {expected_md5}"
            )

    return reconstructed


# ============================================================================
# FINAL OUTPUTS
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
    Write one complete output set for this input subdirectory.
    """

    reports = []
    rejections = []

    rejection_counts = {}

    for result in results:

        if result[
            "kind"
        ] == "report":

            reports.append(
                result[
                    "record"
                ]
            )

        else:

            record = result[
                "record"
            ]

            rejections.append(
                record
            )

            reason = record[
                "reason"
            ]

            rejection_counts[
                reason
            ] = (
                rejection_counts.get(
                    reason,
                    0,
                )
                +
                1
            )

    paths = output_paths(
        subdir
    )

    # --------------------------------------------------------------
    # candidates_<subdir>.txt
    # --------------------------------------------------------------

    with open(
        paths[
            "candidates_txt"
        ],
        "w",
        encoding="utf-8",
    ) as fh:

        for record in reports:

            fh.write(
                str(
                    record[
                        "path"
                    ]
                )
                +
                "\n"
            )

    # --------------------------------------------------------------
    # candidates_<subdir>.json
    # --------------------------------------------------------------

    atomic_json_write(
        paths[
            "candidates_json"
        ],
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

            "stage":
                "3",

            "input_subdirectory":
                str(subdir),

            "description":
                (
                    "Stage 3 direct physical-MIDI-track "
                    "musical analysis. Stage 3 consumes "
                    "Stage 1 output directly and does not "
                    "use Stage 2, PrettyMIDI, UglyMIDI, or "
                    "LAMDa metadata."
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
                int(
                    WORKERS
                ),

            "candidate_count":
                int(
                    len(reports)
                ),

            "candidates":
                reports,
        },
    )

    # --------------------------------------------------------------
    # rejections_<subdir>.json
    # --------------------------------------------------------------

    atomic_json_write(
        paths[
            "rejections_json"
        ],
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

            "stage":
                "3",

            "input_subdirectory":
                str(subdir),

            "rejection_count":
                int(
                    len(rejections)
                ),

            "rejection_reasons":
                rejection_counts,

            "rejections":
                rejections,
        },
    )

    # --------------------------------------------------------------
    # summary_<subdir>.json
    # --------------------------------------------------------------

    atomic_json_write(
        paths[
            "summary_json"
        ],
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

            "stage":
                "3",

            "input_subdirectory":
                str(subdir),

            "input_count":
                int(
                    len(candidates)
                ),

            "survivor_count":
                int(
                    len(reports)
                ),

            "rejected_count":
                int(
                    len(rejections)
                ),

            "rejection_reasons":
                rejection_counts,
        },
    )

    return (
        len(reports),
        len(rejections),
    )


# ============================================================================
# PROCESS ONE SUBDIRECTORY
# ============================================================================

def process_subdirectory(subdir):

    candidates = load_stage1_candidates(
        subdir
    )

    input_count = len(
        candidates
    )

    fingerprint = candidate_fingerprint(
        candidates
    )

    results = load_checkpoints(
        subdir,
        candidates,
        fingerprint,
    )

    completed = len(
        results
    )

    print()
    print("=" * 78)
    print(
        f"STAGE 3 — SUBDIRECTORY {subdir}"
    )
    print("=" * 78)
    print(
        f"Stage 1 input      : {input_count:,}"
    )
    print(
        f"Already completed  : {completed:,}"
    )
    print(
        f"Remaining          : "
        f"{input_count - completed:,}"
    )
    print(
        f"Workers            : {WORKERS}"
    )
    print(
        f"Max pending        : {MAX_PENDING}"
    )
    print(
        f"Checkpoint interval: "
        f"{CHECKPOINT_INTERVAL}"
    )
    print()

    if completed < input_count:

        pending = {}

        ready = {}

        next_submit = completed

        next_commit = completed

        checkpoint_number = (
            next_checkpoint_number()
        )

        # ----------------------------------------------------------
        # Results accumulated after the last checkpoint only.
        # ----------------------------------------------------------

        uncheckpointed_results = []

        with ProcessPoolExecutor(
            max_workers=WORKERS
        ) as executor:

            # Initial bounded submission.
            while (
                next_submit < input_count
                and
                len(pending) < MAX_PENDING
            ):

                future = executor.submit(
                    stage3_worker,
                    candidates[
                        next_submit
                    ],
                )

                pending[
                    future
                ] = next_submit

                next_submit += 1

            # Consume and refill.
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

                        worker_result = (
                            future.result()
                        )

                        classified = (
                            classify_result(
                                worker_result
                            )
                        )

                    except Exception as exc:

                        candidate = candidates[
                            original_index
                        ]

                        classified = {
                            "kind":
                                "rejection",

                            "record":
                                {
                                    "md5":
                                        candidate[
                                            "md5"
                                        ],

                                    "path":
                                        candidate[
                                            "path"
                                        ],

                                    "reason":
                                        "future_error:"
                                        +
                                        type(
                                            exc
                                        ).__name__,
                                },
                        }

                    ready[
                        original_index
                    ] = classified

                    # Refill immediately.
                    if next_submit < input_count:

                        future2 = executor.submit(
                            stage3_worker,
                            candidates[
                                next_submit
                            ],
                        )

                        pending[
                            future2
                        ] = next_submit

                        next_submit += 1

                # --------------------------------------------------
                # Commit ONLY the contiguous completed prefix.
                # --------------------------------------------------

                while next_commit in ready:

                    classified = ready.pop(
                        next_commit
                    )

                    results.append(
                        classified
                    )

                    uncheckpointed_results.append(
                        classified
                    )

                    next_commit += 1

                    # --------------------------------------------------
                    # Write exactly one checkpoint per interval.
                    # --------------------------------------------------

                    if len(
                        uncheckpointed_results
                    ) >= CHECKPOINT_INTERVAL:

                        start = (
                            next_commit
                            -
                            len(
                                uncheckpointed_results
                            )
                        )

                        end = next_commit

                        write_checkpoint(
                            checkpoint_number,
                            subdir,
                            input_count,
                            fingerprint,
                            start,
                            end,
                            uncheckpointed_results,
                        )

                        checkpoint_number += 1

                        uncheckpointed_results = []

                        gc.collect()

        # --------------------------------------------------------------
        # Final partial checkpoint.
        # --------------------------------------------------------------

        if uncheckpointed_results:

            start = (
                next_commit
                -
                len(
                    uncheckpointed_results
                )
            )

            end = next_commit

            write_checkpoint(
                checkpoint_number,
                subdir,
                input_count,
                fingerprint,
                start,
                end,
                uncheckpointed_results,
            )

            uncheckpointed_results = []

        # --------------------------------------------------------------
        # Reconstruct from disk.
        # This also verifies checkpoint integrity.
        # --------------------------------------------------------------

        results = load_checkpoints(
            subdir,
            candidates,
            fingerprint,
        )

    # --------------------------------------------------------------
    # Final verification.
    # --------------------------------------------------------------

    if len(results) != input_count:

        raise RuntimeError(
            f"Stage 3 result count mismatch for "
            f"subdirectory {subdir}:\n"
            f"  expected: {input_count:,}\n"
            f"  actual:   {len(results):,}"
        )

    for index, result in enumerate(
        results
    ):

        expected_md5 = str(
            candidates[
                index
            ]["md5"]
        ).lower()

        actual_md5 = str(
            result[
                "record"
            ].get(
                "md5",
                "",
            )
        ).lower()

        if actual_md5 != expected_md5:

            raise RuntimeError(
                "Final ordering verification failed:\n"
                f"  subdirectory: {subdir}\n"
                f"  index: {index}\n"
                f"  expected MD5: {expected_md5}\n"
                f"  actual MD5:   {actual_md5}"
            )

    survivors, rejected = write_final_outputs(
        subdir,
        candidates,
        results,
    )

    print()
    print(
        f"STAGE 3 COMPLETE — {subdir}"
    )
    print(
        f"  Input     : {input_count:,}"
    )
    print(
        f"  Survivors : {survivors:,}"
    )
    print(
        f"  Rejected  : {rejected:,}"
    )

    return (
        survivors,
        rejected,
    )


# ============================================================================
# CHECKPOINT CLEANUP
# ============================================================================

def delete_all_checkpoints():

    paths = checkpoint_paths()

    if not paths:
        return

    print()
    print("=" * 78)
    print(
        "REMOVING STAGE 3 CHECKPOINTS"
    )
    print("=" * 78)

    for path in paths:

        path.unlink()

        print(
            f"  deleted: {path}"
        )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 78)
    print(
        "Los Angeles MIDI Dataset"
    )
    print(
        "STAGE 3 — DIRECT PHYSICAL MIDI TRACK ANALYSIS"
    )
    print("=" * 78)
    print()
    print(
        "Stage 3 input = Stage 1 directly."
    )
    print(
        "Stage 2 = DISABLED."
    )
    print(
        "PrettyMIDI/UglyMIDI = NOT USED."
    )
    print(
        "LAMDa metadata = NOT USED."
    )
    print(
        f"Workers = {WORKERS}"
    )
    print()

    if not STAGE1_DIR.is_dir():

        raise RuntimeError(
            f"Stage 1 directory does not exist:\n"
            f"{STAGE1_DIR}"
        )

    if not MIDI_ROOT.is_dir():

        raise RuntimeError(
            f"MIDI root does not exist:\n"
            f"{MIDI_ROOT}"
        )

    STAGE3_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        for subdir in INPUT_SUBDIRECTORIES:

            input_file = (
                STAGE1_DIR
                /
                f"candidates_{subdir}.json"
            )

            if not input_file.is_file():

                print(
                    f"Skipping {subdir}: "
                    f"{input_file} not found."
                )

                continue

            process_subdirectory(
                subdir
            )

        # ----------------------------------------------------------
        # ONLY successful completion reaches here.
        # ----------------------------------------------------------

        delete_all_checkpoints()

        print()
        print("=" * 78)
        print(
            "STAGE 3 — ALL SUBDIRECTORIES COMPLETE"
        )
        print("=" * 78)
        print()
        print(
            "All checkpoint files were deleted."
        )

    except KeyboardInterrupt:

        print()
        print("=" * 78)
        print(
            "STAGE 3 INTERRUPTED"
        )
        print("=" * 78)
        print(
            "Checkpoints have been retained."
        )

        raise

    except Exception:

        print()
        print("=" * 78)
        print(
            "STAGE 3 FAILED"
        )
        print("=" * 78)
        print(
            "Checkpoints have been retained."
        )

        raise


if __name__ == "__main__":
    main()
