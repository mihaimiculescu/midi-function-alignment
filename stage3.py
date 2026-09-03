#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Los Angeles MIDI Dataset
Stage 3 — Direct MIDI musical-structure / melody-candidate analysis

Stage 3 input:
    Dataset/LAMDselection/selection_stage1/candidates_1.json
    ...
    Dataset/LAMDselection/selection_stage1/candidates_f.json

Stage 2 is deliberately NOT used.

Stage 3 reads the physical MIDI files referenced by the Stage 1 manifests,
analyzes every MIDI track/instrument, ranks melody-like tracks using the
same melody scoring function as la_filter.py, and writes one independent
set of outputs for each hexadecimal input directory.

Output directory:
    Dataset/LAMDselection/selection_stage3/

For each subdirectory:
    candidates_1.txt
    candidates_1.json
    reports_1.json
    rejections_1.json
    summary_1.json

    ...
    candidates_f.txt
    candidates_f.json
    reports_f.json
    rejections_f.json
    summary_f.json

Checkpointing:
    checkpoint1.json
    checkpoint2.json
    ...

A checkpoint is a complete snapshot of the accumulated state at that point.
A new checkpoint file is written for every checkpoint interval; the previous
checkpoint is NEVER appended to.

After successful completion of ALL input subdirectories, every checkpoint*.json
is deleted.

Important:
    The checkpoint files are deliberately self-contained so an interrupted
    run can resume without rebuilding results already processed.

Memory:
    - one hexadecimal input directory is processed at a time
    - at most MAX_PENDING futures exist simultaneously
    - workers return compact dictionaries only
    - MIDI objects are destroyed in workers before returning
    - accumulated results are released between directories
"""

import gc
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

INPUT_DIR = Path(
    "Dataset/LAMDselection/selection_stage1"
)

OUTPUT_DIR = Path(
    "Dataset/LAMDselection/selection_stage3"
)

# Physical MIDI files are still the authoritative source for Stage 3.
MIDI_ROOT = Path(
    "Dataset/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/MIDIs"
)

INPUT_SUBDIRECTORIES = tuple("123456789abcdef")

# ============================================================================
# EXECUTION / MEMORY CONFIGURATION
# ============================================================================

WORKERS = 24

# Do not submit the entire candidate list at once.
# 2x workers keeps the CPU busy without accumulating tens of thousands of
# futures/results in the parent process.
MAX_PENDING = WORKERS * 2

# A checkpoint is written after this many completed MIDI files within a
# subdirectory.
CHECKPOINT_INTERVAL = 1000

# Maximum melody-like tracks retained per MIDI file, exactly as in la_filter.py.
MAX_CANDIDATES_PER_FILE = 8

# Tiny tracks are ignored for melody-candidate analysis, exactly as in
# la_filter.py.
MIN_NOTES_FOR_CANDIDATE = 8

# ============================================================================
# MELODY RANKING CONFIGURATION
# ============================================================================
#
# These are the scoring rules from la_filter.py.
#
# The Stage 3 script does NOT apply the later Stage 4 acceptance thresholds
# (pitch_mean >= 40, melody_score >= 0.86, monophonic_fraction >= 0.95).
# Those are a separate final-selection operation.
#
# Stage 3 ranks tracks. It does not make that final decision.
# ============================================================================

EXCLUDED_MIDI_CHANNELS = {9, 15}

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
    print(
        "Run this script from the midi-function-alignment repository "
        "environment."
    )
    print()
    sys.exit(1)


# ============================================================================
# JSON / FILE HELPERS
# ============================================================================

def atomic_json_write(path, data):
    """Atomically write JSON so interruption cannot leave a half-written file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
        text=True,
    )

    try:
        with os.fdopen(
            fd,
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
            os.fsync(fh.fileno())

        os.replace(tmp_name, path)

    except Exception:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def make_json_serializable(obj):
    """Convert NumPy/scalar/container values into json-compatible values."""
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


def load_candidates_file(subdir):
    """
    Load the Stage 1 candidates_<subdir>.json file.

    The Stage 1 JSON contains:
        {
            ...
            "candidates": [
                {
                    "md5": "...",
                    "path": "...",
                    ...
                }
            ]
        }

    Only compact candidate dictionaries are retained.
    """
    path = INPUT_DIR / f"candidates_{subdir}.json"

    if not path.is_file():
        raise FileNotFoundError(
            f"Stage 1 candidate file does not exist: {path}"
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
            f"Invalid Stage 1 candidate file: {path}\n"
            "Expected a JSON object containing a 'candidates' list."
        )

    normalized = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        midi_path = candidate.get("path")
        md5 = candidate.get("md5")

        if not midi_path:
            continue

        if not md5:
            md5 = Path(midi_path).stem.lower()

        normalized.append({
            "md5": str(md5).lower(),
            "path": str(midi_path),
            "input_subdirectory": subdir,
        })

    normalized.sort(
        key=lambda item: item["path"]
    )

    return normalized


# ============================================================================
# MELODY-SCORE HELPERS
# ============================================================================
#
# These calculations intentionally mirror la_filter.py.
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
        return float(values[n // 2])

    return float(
        (
            values[n // 2 - 1]
            + values[n // 2]
        ) / 2
    )


def calculate_polyphony(notes):
    """
    Average simultaneous-note polyphony using a sweep-line algorithm.

    Same definition as la_filter.py.
    """
    if not notes:
        return 0.0

    events = []

    for note in notes:
        start = float(note.start)
        end = float(note.end)

        if end <= start:
            continue

        events.append((start, 1))
        events.append((end, -1))

    if not events:
        return 0.0

    # At identical timestamps note-offs occur before note-ons.
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
                - previous_time
            )

            weighted_polyphony += (
                active
                * duration
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
        / total_time
    )


def calculate_pitch_range(notes):
    if not notes:
        return 0

    pitches = [
        int(note.pitch)
        for note in notes
    ]

    return max(pitches) - min(pitches)


def calculate_pitch_mean(notes):
    if not notes:
        return 0.0

    return safe_mean([
        float(note.pitch)
        for note in notes
    ])


def calculate_note_duration_stats(notes):
    durations = []

    for note in notes:
        duration = (
            float(note.end)
            - float(note.start)
        )

        if duration > 0:
            durations.append(duration)

    if not durations:
        return {
            "mean": 0.0,
            "median": 0.0,
            "min": 0.0,
            "max": 0.0,
        }

    return {
        "mean": safe_mean(durations),
        "median": safe_median(durations),
        "min": float(min(durations)),
        "max": float(max(durations)),
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

    span = calculate_activity_span(notes)

    if span <= 0:
        return 0.0

    return float(
        len(notes) / span
    )


def calculate_monophonic_fraction(notes):
    """
    Fraction of note onsets having no other currently sounding note.

    Same definition as la_filter.py.
    """
    if not notes:
        return 0.0

    import heapq

    sorted_notes = sorted(
        notes,
        key=lambda note: (
            float(note.start),
            float(note.end),
        ),
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
        isolated / len(notes)
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
    Exact melody-score weighting used by la_filter.py.
    """
    if not notes:
        return 0.0

    score = 0.0

    # Monophonic material.
    score += (
        monophonic_fraction
        * 0.35
    )

    # Moderate polyphony.
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
        * 0.25
    )

    # Useful melodic pitch range.
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
        * 0.15
    )

    # Sustained activity / density.
    if activity_span > 0:
        density_score = min(
            1.0,
            note_density / 2.0,
        )
    else:
        density_score = 0.0

    score += (
        density_score
        * 0.10
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
        * 0.05
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
# TRACK / CHANNEL HELPERS
# ============================================================================

def instrument_channels(instrument):
    """
    Return the MIDI channel(s) represented by a PrettyMIDI instrument.

    PrettyMIDI normally groups notes by program/channel, but the public Note
    object does not expose a channel. Therefore channel information is taken
    from the instrument where available.

    For percussion, instrument.is_drum is authoritative.
    """
    channels = set()

    for attr in (
        "channel",
        "channels",
    ):
        if not hasattr(instrument, attr):
            continue

        value = getattr(
            instrument,
            attr,
        )

        if value is None:
            continue

        if isinstance(value, (list, tuple, set)):
            values = value
        else:
            values = [value]

        for channel in values:
            try:
                channel = int(channel)
            except (TypeError, ValueError):
                continue

            if 0 <= channel < 16:
                channels.add(channel)

    return sorted(channels)


def is_excluded_track(instrument):
    """
    Exclude General MIDI percussion.

    PrettyMIDI marks percussion instruments with is_drum=True.

    If explicit instrument channel information is available, channels 9 and
    15 are also treated as excluded, matching la_filter.py's 0-based channel
    representation.
    """
    if bool(
        getattr(
            instrument,
            "is_drum",
            False,
        )
    ):
        return True

    channels = instrument_channels(
        instrument
    )

    return any(
        channel in EXCLUDED_MIDI_CHANNELS
        for channel in channels
    )


# ============================================================================
# ONE-MIDI ANALYSIS
# ============================================================================

def analyze_one_midi(candidate):
    """
    Analyze one physical MIDI file.

    Returns a compact result only. The PrettyMIDI object and all Note objects
    remain inside the worker and are released before returning.
    """
    midi_path = candidate["path"]
    md5 = candidate["md5"]

    try:
        midi = pretty_midi.PrettyMIDI(
            midi_path
        )

    except Exception as exc:
        return {
            "candidate": candidate,
            "analysis": None,
            "reason": (
                "midi_parse_error:"
                + type(exc).__name__
            ),
        }

    try:
        track_reports = []
        tracks_with_notes = 0
        total_non_percussion_notes = 0
        excluded_percussion_notes = 0

        for track_index, instrument in enumerate(
            midi.instruments
        ):
            raw_notes = list(
                instrument.notes
            )

            if not raw_notes:
                continue

            tracks_with_notes += 1

            if is_excluded_track(
                instrument
            ):
                excluded_percussion_notes += len(
                    raw_notes
                )
                continue

            filtered_notes = raw_notes

            total_non_percussion_notes += len(
                filtered_notes
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

            monophonic_fraction = (
                calculate_monophonic_fraction(
                    filtered_notes
                )
            )

            duration_stats = (
                calculate_note_duration_stats(
                    filtered_notes
                )
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

            channels = instrument_channels(
                instrument
            )

            track_reports.append({
                "track": track_index,
                "channels": channels,
                "program": getattr(
                    instrument,
                    "program",
                    None,
                ),
                "program_name": (
                    pretty_midi.program_to_instrument_name(
                        instrument.program
                    )
                    if instrument.program is not None
                    else None
                ),
                "is_drum": bool(
                    getattr(
                        instrument,
                        "is_drum",
                        False,
                    )
                ),
                "raw_notes": len(raw_notes),
                "non_percussion_notes": len(
                    filtered_notes
                ),
                "polyphony": round(
                    polyphony,
                    4,
                ),
                "monophonic_fraction": round(
                    monophonic_fraction,
                    4,
                ),
                "pitch_range": pitch_range,
                "pitch_mean": round(
                    pitch_mean,
                    2,
                ),
                "note_density": round(
                    note_density,
                    4,
                ),
                "activity_span": round(
                    activity_span,
                    4,
                ),
                "duration": duration_stats,
                "melody_score": round(
                    score,
                    4,
                ),
            })

        # Highest melody score first.
        track_reports.sort(
            key=lambda item: (
                item["melody_score"],
                item["monophonic_fraction"],
                item["pitch_mean"],
                item["non_percussion_notes"],
                -item["track"],
            ),
            reverse=True,
        )

        all_candidate_count = len(
            track_reports
        )

        track_reports = track_reports[
            :MAX_CANDIDATES_PER_FILE
        ]

        analysis = {
            "md5": md5,
            "tracks_with_notes": tracks_with_notes,
            "non_percussion_notes": (
                total_non_percussion_notes
            ),
            "excluded_percussion_notes": (
                excluded_percussion_notes
            ),
            "candidate_count": len(
                track_reports
            ),
            "all_candidate_count": (
                all_candidate_count
            ),
            "candidates": track_reports,
        }

        return {
            "candidate": candidate,
            "analysis": analysis,
            "reason": None,
        }

    except Exception as exc:
        return {
            "candidate": candidate,
            "analysis": None,
            "reason": (
                "midi_analysis_error:"
                + type(exc).__name__
            ),
        }

    finally:
        # Release the complete MIDI object before this worker returns.
        try:
            del midi
        except UnboundLocalError:
            pass

        gc.collect()


# ============================================================================
# CHECKPOINTS
# ============================================================================

def checkpoint_path(checkpoint_number):
    return (
        OUTPUT_DIR
        / f"checkpoint{checkpoint_number}.json"
    )


def next_global_checkpoint_number():
    numbers = []

    for path in OUTPUT_DIR.glob(
        "checkpoint*.json"
    ):
        stem = path.stem

        if not stem.startswith("checkpoint"):
            continue

        suffix = stem[len("checkpoint"):]

        try:
            numbers.append(int(suffix))
        except ValueError:
            continue

    return (
        max(numbers) + 1
        if numbers
        else 1
    )


def write_checkpoint(
    checkpoint_number,
    subdir,
    start_index,
    end_index,
    input_count,
    results,
):
    """
    Write one bounded checkpoint.

    IMPORTANT:
        A checkpoint contains ONLY the results for this interval.
        It does not contain all previous reports/survivors.

    Therefore every checkpoint file is approximately the size of one
    checkpoint interval instead of becoming progressively larger.
    """
    path = checkpoint_path(
        checkpoint_number
    )

    state = {
        "stage": 3,
        "input_subdirectory": subdir,

        "input_count": input_count,

        "start_index": start_index,
        "end_index": end_index,

        "completed_count": end_index,

        "checkpoint_number":
            checkpoint_number,

        "results": results,
    }

    atomic_json_write(
        path,
        make_json_serializable(state),
    )

    print(
        f"  checkpoint: {path} "
        f"[{start_index:,}..{end_index:,}]"
    )


def load_checkpoint_results(
    subdir,
    candidates,
):
    """
    Load all checkpoint fragments belonging to one input subdirectory.

    Each checkpoint is a fixed-size fragment. The fragments are merged in
    index order. No single checkpoint contains the accumulated history.
    """
    paths = []

    for path in OUTPUT_DIR.glob(
        "checkpoint*.json"
    ):
        try:
            with open(
                path,
                "r",
                encoding="utf-8",
            ) as fh:
                state = json.load(fh)
        except Exception:
            continue

        if (
            state.get("stage") != 3
            or state.get("input_subdirectory") != subdir
        ):
            continue

        paths.append(
            (
                int(
                    state.get(
                        "start_index",
                        -1,
                    )
                ),
                path,
                state,
            )
        )

    if not paths:
        return None

    paths.sort(
        key=lambda item: item[0]
    )

    expected_start = 0
    fragments = []

    for start_index, path, state in paths:
        end_index = int(
            state.get(
                "end_index",
                -1,
            )
        )

        input_count = int(
            state.get(
                "input_count",
                -1,
            )
        )

        if input_count != len(
            candidates
        ):
            raise RuntimeError(
                "Checkpoint is incompatible with the current Stage 1 "
                "candidate list.\n"
                f"  checkpoint: {path}\n"
                f"  checkpoint input count: {input_count:,}\n"
                f"  current input count:    {len(candidates):,}"
            )

        if start_index != expected_start:
            raise RuntimeError(
                "Stage 3 checkpoint sequence has a gap or overlap.\n"
                f"  checkpoint: {path}\n"
                f"  expected start: {expected_start:,}\n"
                f"  actual start:   {start_index:,}"
            )

        if (
            end_index <= start_index
            or end_index > len(candidates)
        ):
            raise RuntimeError(
                "Invalid checkpoint range.\n"
                f"  checkpoint: {path}\n"
                f"  range: {start_index:,}..{end_index:,}"
            )

        results = state.get(
            "results",
            []
        )

        if not isinstance(
            results,
            list,
        ):
            raise RuntimeError(
                f"Invalid results in checkpoint: {path}"
            )

        expected_result_count = (
            end_index - start_index
        )

        if len(results) != expected_result_count:
            raise RuntimeError(
                "Checkpoint result count mismatch.\n"
                f"  checkpoint: {path}\n"
                f"  expected: {expected_result_count:,}\n"
                f"  actual:   {len(results):,}"
            )

        # Verify the candidate identity at both ends of the fragment.
        first_result = results[0]
        last_result = results[-1]

        if first_result.get("md5") != candidates[
            start_index
        ]["md5"]:
            raise RuntimeError(
                "Checkpoint first-MD5 mismatch.\n"
                f"  checkpoint: {path}"
            )

        if last_result.get("md5") != candidates[
            end_index - 1
        ]["md5"]:
            raise RuntimeError(
                "Checkpoint last-MD5 mismatch.\n"
                f"  checkpoint: {path}"
            )

        fragments.append(
            (
                start_index,
                end_index,
                results,
            )
        )

        expected_start = end_index

    return {
        "completed_count": expected_start,
        "fragments": fragments,
        "last_checkpoint_number": max(
            int(
                state["checkpoint_number"]
            )
            for _, _, state in paths
        ),
    }


def reconstruct_state_from_fragments(
    fragments,
):
    """
    Rebuild the in-memory Stage 3 state from fixed-size checkpoint fragments.
    """
    reports = []
    survivors = []
    rejection_counts = Counter()
    rejection_details = []

    for _, _, results in fragments:
        for result in results:
            reason = result.get(
                "reason"
            )

            candidate = result[
                "candidate"
            ]

            if reason is not None:
                rejection_counts[
                    reason
                ] += 1

                rejection_details.append({
                    "md5":
                        candidate["md5"],

                    "path":
                        candidate["path"],

                    "reason":
                        reason,
                })

                continue

            analysis = result.get(
                "analysis"
            )

            if analysis is None:
                raise RuntimeError(
                    "Checkpoint contains neither a reason nor analysis."
                )

            report = {
                "md5":
                    candidate["md5"],

                "path":
                    candidate["path"],

                "input_subdirectory":
                    candidate[
                        "input_subdirectory"
                    ],

                "stage3":
                    analysis,
            }

            reports.append(
                report
            )

            if analysis[
                "candidate_count"
            ] > 0:
                survivors.append(
                    candidate["path"]
                )
            else:
                rejection_counts[
                    "no_non_percussion_candidate"
                ] += 1

                rejection_details.append({
                    "md5":
                        candidate["md5"],

                    "path":
                        candidate["path"],

                    "reason":
                        "no_non_percussion_candidate",
                })

    return (
        reports,
        survivors,
        rejection_counts,
        rejection_details,
    )


def delete_all_checkpoints():
    """
    Delete every Stage 3 checkpoint only after ALL subdirectories completed.
    """
    paths = sorted(
        OUTPUT_DIR.glob(
            "checkpoint*.json"
        )
    )

    deleted = 0

    for path in paths:
        try:
            path.unlink()
            deleted += 1
        except FileNotFoundError:
            pass

    return deleted


# ============================================================================
# RESULT OUTPUT
# ============================================================================

def write_subdirectory_outputs(
    subdir,
    input_candidates,
    reports,
    survivors,
    rejection_counts,
    rejection_details,
):
    """
    Write one complete output set for one hexadecimal input directory.
    """
    candidates_txt = (
        OUTPUT_DIR
        / f"candidates_{subdir}.txt"
    )

    candidates_json = (
        OUTPUT_DIR
        / f"candidates_{subdir}.json"
    )

    reports_json = (
        OUTPUT_DIR
        / f"reports_{subdir}.json"
    )

    rejections_json = (
        OUTPUT_DIR
        / f"rejections_{subdir}.json"
    )

    summary_json = (
        OUTPUT_DIR
        / f"summary_{subdir}.json"
    )

    # Deterministic ordering.
    reports.sort(
        key=lambda item: item["path"]
    )

    survivors.sort()

    rejection_details.sort(
        key=lambda item: item["path"]
    )

    # ------------------------------------------------------------------
    # candidates_<subdir>.txt
    # ------------------------------------------------------------------

    with open(
        candidates_txt,
        "w",
        encoding="utf-8",
    ) as fh:
        for path in survivors:
            fh.write(
                path
                + "\n"
            )

    # ------------------------------------------------------------------
    # candidates_<subdir>.json
    # ------------------------------------------------------------------

    atomic_json_write(
        candidates_json,
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

            "stage": 3,

            "input_subdirectory": subdir,

            "description": (
                "Stage 3 MIDI files having at least one "
                "melody-like track candidate. "
                "Stage 2 is not used."
            ),

            "input_manifest": str(
                INPUT_DIR
                / f"candidates_{subdir}.json"
            ),

            "midi_root": str(
                MIDI_ROOT
            ),

            "survivor_count": len(
                survivors
            ),

            "candidates": [
                {
                    "md5": report["md5"],
                    "path": report["path"],
                    "best_track": (
                        report["stage3"]["candidates"][0]
                        if report["stage3"]["candidates"]
                        else None
                    ),
                }
                for report in reports
                if report.get("stage3", {}).get(
                    "candidate_count",
                    0,
                ) > 0
            ],
        },
    )

    # ------------------------------------------------------------------
    # reports_<subdir>.json
    # ------------------------------------------------------------------

    atomic_json_write(
        reports_json,
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

            "stage": 3,

            "input_subdirectory": subdir,

            "description": (
                "Direct physical-MIDI musical-structure "
                "analysis and melody-candidate ranking."
            ),

            "input_manifest": str(
                INPUT_DIR
                / f"candidates_{subdir}.json"
            ),

            "ranking": {
                "max_candidates_per_file":
                    MAX_CANDIDATES_PER_FILE,

                "min_notes_for_candidate":
                    MIN_NOTES_FOR_CANDIDATE,

                "melody_score": {
                    "monophonic_fraction_weight":
                        0.35,

                    "polyphony_weight":
                        0.25,

                    "pitch_range_weight":
                        0.15,

                    "density_weight":
                        0.10,

                    "note_count_weight":
                        0.05,
                },
            },

            "excluded_channels": sorted(
                EXCLUDED_MIDI_CHANNELS
            ),

            "reports": reports,
        },
    )

    # ------------------------------------------------------------------
    # rejections_<subdir>.json
    # ------------------------------------------------------------------

    atomic_json_write(
        rejections_json,
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

            "stage": 3,

            "input_subdirectory": subdir,

            "rejection_count": len(
                rejection_details
            ),

            "rejection_counts": dict(
                rejection_counts.most_common()
            ),

            "rejections": rejection_details,
        },
    )

    # ------------------------------------------------------------------
    # summary_<subdir>.json
    # ------------------------------------------------------------------

    input_count = len(
        input_candidates
    )

    survivor_count = len(
        survivors
    )

    rejected_count = (
        input_count
        - survivor_count
    )

    summary = {
        "dataset":
            "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

        "stage": 3,

        "input_subdirectory": subdir,

        "input_manifest": str(
            INPUT_DIR
            / f"candidates_{subdir}.json"
        ),

        "physical_midi_root": str(
            MIDI_ROOT
        ),

        "stage2_used": False,

        "input_candidates": input_count,

        "survivors": survivor_count,

        "rejected": rejected_count,

        "survivor_ratio": (
            survivor_count / input_count
            if input_count
            else 0
        ),

        "workers": WORKERS,

        "max_pending": MAX_PENDING,

        "checkpoint_interval":
            CHECKPOINT_INTERVAL,

        "ranking": {
            "max_candidates_per_file":
                MAX_CANDIDATES_PER_FILE,

            "min_notes_for_candidate":
                MIN_NOTES_FOR_CANDIDATE,

            "score_definition":
                "same melody_score weighting and "
                "piecewise functions as la_filter.py",
        },

        "excluded_channels": sorted(
            EXCLUDED_MIDI_CHANNELS
        ),

        "rejection_counts": dict(
            rejection_counts.most_common()
        ),

        "outputs": {
            "candidates_txt":
                str(candidates_txt),

            "candidates_json":
                str(candidates_json),

            "reports_json":
                str(reports_json),

            "rejections_json":
                str(rejections_json),

            "summary_json":
                str(summary_json),
        },
    }

    atomic_json_write(
        summary_json,
        make_json_serializable(
            summary
        ),
    )


# ============================================================================
# PROCESS ONE HEX DIRECTORY
# ============================================================================

def process_subdirectory(subdir):
    print()
    print("=" * 78)
    print(
        f"STAGE 3 — INPUT SUBDIRECTORY {subdir}"
    )
    print("=" * 78)
    print()

    print(
        f"Input candidates : "
        f"{INPUT_DIR / f'candidates_{subdir}.json'}"
    )

    print(
        f"MIDI root        : "
        f"{MIDI_ROOT / subdir}"
    )

    print(
        f"Output directory : "
        f"{OUTPUT_DIR}"
    )

    print(
        f"Workers          : {WORKERS}"
    )

    print(
        f"Max pending      : {MAX_PENDING}"
    )

    print()

    midi_dir = (
        MIDI_ROOT / subdir
    )

    if not midi_dir.is_dir():
        raise FileNotFoundError(
            f"MIDI input directory does not exist: "
            f"{midi_dir}"
        )

    candidates = load_candidates_file(
        subdir
    )

    total = len(
        candidates
    )

    print(
        f"Stage 1 candidates: {total:,}"
    )

    if total == 0:
        print(
            "No candidates. Writing empty Stage 3 outputs."
        )

        write_subdirectory_outputs(
            subdir,
            candidates,
            [],
            [],
            Counter(),
            [],
        )

        return

    # ------------------------------------------------------------------
    # Resume from fixed-size checkpoint fragments.
    # ------------------------------------------------------------------

    checkpoint_state = load_checkpoint_results(
        subdir,
        candidates,
    )

    if checkpoint_state is None:
        completed_count = 0
        reports = []
        survivors = []
        rejection_counts = Counter()
        rejection_details = []

        next_checkpoint_number = (
            next_global_checkpoint_number()
        )

        print(
            "No Stage 3 checkpoints found."
        )

    else:
        completed_count = checkpoint_state[
            "completed_count"
        ]

        (
            reports,
            survivors,
            rejection_counts,
            rejection_details,
        ) = reconstruct_state_from_fragments(
            checkpoint_state[
                "fragments"
            ]
        )

        next_checkpoint_number = (
            max(
                checkpoint_state[
                    "last_checkpoint_number"
                ] + 1,
                next_global_checkpoint_number(),
            )
        )

        print(
            f"Resuming at candidate "
            f"{completed_count:,}/{total:,}"
        )

        print(
            f"Checkpoint fragments loaded: "
            f"{len(checkpoint_state['fragments']):,}"
        )

    if completed_count >= total:
        print(
            "All files in this subdirectory were already processed."
        )

        write_subdirectory_outputs(
            subdir,
            candidates,
            reports,
            survivors,
            rejection_counts,
            rejection_details,
        )

        return

    # ------------------------------------------------------------------
    # Process with bounded ProcessPoolExecutor.
    # ------------------------------------------------------------------

    next_index = completed_count
    checkpoint_results = []
    checkpoint_start_index = completed_count

    with ProcessPoolExecutor(
        max_workers=WORKERS
    ) as executor:

        pending = {}

        # Fill bounded queue.
        while (
            next_index < total
            and len(pending) < MAX_PENDING
        ):
            candidate = candidates[
                next_index
            ]

            future = executor.submit(
                analyze_one_midi,
                candidate,
            )

            pending[future] = (
                next_index,
                candidate,
            )

            next_index += 1

        while pending:
            done, _ = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                index, candidate = pending.pop(
                    future
                )

                try:
                    result = future.result()

                except Exception as exc:
                    result = {
                        "candidate": candidate,
                        "analysis": None,
                        "reason": (
                            "future_error:"
                            + type(exc).__name__
                        ),
                    }

                # ------------------------------------------------------
                # IMPORTANT:
                #
                # Results may complete out of order. Checkpoint progress
                # therefore advances only as a contiguous prefix.
                #
                # To make this simple and robust, completed results are
                # stored temporarily by input index.
                # ------------------------------------------------------

                # We attach the result to the candidate itself and use a
                # private in-memory result table. Only MAX_PENDING entries
                # can accumulate here.
                if "completed_results" not in locals():
                    completed_results = {}

                completed_results[index] = result

                # ------------------------------------------------------
                # Consume every contiguous completed result.
                #
                # Results are committed in input order even though worker
                # processes finish in arbitrary order.
                # ------------------------------------------------------

                while completed_count in completed_results:
                    current = completed_results.pop(
                        completed_count
                    )

                    current_candidate = current[
                        "candidate"
                    ]

                    reason = current[
                        "reason"
                    ]

                    # Keep exactly one compact result in the current
                    # checkpoint fragment.
                    checkpoint_results.append({
                        "candidate":
                            current_candidate,

                        "analysis":
                            current.get(
                                "analysis"
                            ),

                        "reason":
                            reason,
                    })

                    if reason is not None:
                        rejection_counts[
                            reason
                        ] += 1

                        rejection_details.append({
                            "md5":
                                current_candidate["md5"],

                            "path":
                                current_candidate["path"],

                            "reason":
                                reason,
                        })

                    else:
                        analysis = current[
                            "analysis"
                        ]

                        report = {
                            "md5":
                                current_candidate["md5"],

                            "path":
                                current_candidate["path"],

                            "input_subdirectory":
                                subdir,

                            "stage3":
                                analysis,
                        }

                        reports.append(
                            report
                        )

                        if analysis[
                            "candidate_count"
                        ] > 0:
                            survivors.append(
                                current_candidate["path"]
                            )
                        else:
                            rejection_counts[
                                "no_non_percussion_candidate"
                            ] += 1

                            rejection_details.append({
                                "md5":
                                    current_candidate["md5"],

                                "path":
                                    current_candidate["path"],

                                "reason":
                                    "no_non_percussion_candidate",
                            })

                    completed_count += 1

                    # --------------------------------------------------
                    # Checkpoint.
                    #
                    # The checkpoint contains ONLY the results collected
                    # since the previous checkpoint.
                    # --------------------------------------------------

                    if (
                        len(checkpoint_results)
                        >= CHECKPOINT_INTERVAL
                    ):
                        write_checkpoint(
                            next_checkpoint_number,
                            subdir,
                            checkpoint_start_index,
                            completed_count,
                            total,
                            checkpoint_results,
                        )

                        next_checkpoint_number += 1

                        checkpoint_start_index = (
                            completed_count
                        )

                        checkpoint_results = []

                    # --------------------------------------------------
                    # Progress.
                    # --------------------------------------------------

                    if (
                        completed_count % 250 == 0
                        or completed_count == total
                    ):
                        print(
                            f"[{subdir}] "
                            f"{completed_count:,}/{total:,} "
                            f"({completed_count / total * 100:6.2f}%) "
                            f"survivors={len(survivors):,} "
                            f"rejected="
                            f"{completed_count - len(survivors):,}",
                            flush=True,
                        )

                # ------------------------------------------------------
                # Refill bounded queue.
                # ------------------------------------------------------

                while (
                    next_index < total
                    and len(pending) < MAX_PENDING
                ):
                    next_candidate = candidates[
                        next_index
                    ]

                    next_future = executor.submit(
                        analyze_one_midi,
                        next_candidate,
                    )

                    pending[next_future] = (
                        next_index,
                        next_candidate,
                    )

                    next_index += 1

    # --------------------------------------------------------------
    # Write a final partial checkpoint fragment if the subdirectory
    # ended between checkpoint intervals.
    # --------------------------------------------------------------

    if checkpoint_results:
        write_checkpoint(
            next_checkpoint_number,
            subdir,
            checkpoint_start_index,
            completed_count,
            total,
            checkpoint_results,
        )

        checkpoint_results = []

    # The result table should be empty because every input has completed.
    if (
        "completed_results" in locals()
        and completed_results
    ):
        raise RuntimeError(
            "Internal ordering error: "
            f"{len(completed_results)} results remain."
        )

    # ------------------------------------------------------------------
    # Final output for this subdirectory.
    # ------------------------------------------------------------------

    write_subdirectory_outputs(
        subdir,
        candidates,
        reports,
        survivors,
        rejection_counts,
        rejection_details,
    )

    print()
    print("=" * 78)
    print(
        f"SUBDIRECTORY {subdir} COMPLETE"
    )
    print("=" * 78)
    print()

    print(
        f"Input candidates : {total:,}"
    )

    print(
        f"Survivors        : {len(survivors):,}"
    )

    print(
        f"Rejected         : "
        f"{total - len(survivors):,}"
    )

    if total:
        print(
            f"Survivor ratio   : "
            f"{len(survivors) / total:.4f}"
        )

    print()
    print("Rejection reasons:")
    print("-" * 78)

    for reason, count in (
        rejection_counts.most_common()
    ):
        print(
            f"{reason:45s} {count:10,}"
        )

    print()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("=" * 78)
    print("Los Angeles MIDI Dataset")
    print("STAGE 3 — DIRECT MIDI MUSICAL-STRUCTURE ANALYSIS")
    print("=" * 78)
    print()

    print(
        "Stage 2 is NOT used."
    )

    print(
        "Input: Stage 1 candidates_<subdir>.json"
    )

    print(
        "Source of truth: physical MIDI files"
    )

    print(
        f"Workers: {WORKERS}"
    )

    print(
        f"Max pending: {MAX_PENDING}"
    )

    print(
        f"Checkpoint interval: {CHECKPOINT_INTERVAL}"
    )

    print()

    if not INPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"Stage 1 output directory does not exist: "
            f"{INPUT_DIR}"
        )

    if not MIDI_ROOT.is_dir():
        raise FileNotFoundError(
            f"MIDI dataset directory does not exist: "
            f"{MIDI_ROOT}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Every subdirectory is independent.
    #
    # If the run is interrupted, completed subdirectories remain complete
    # and any incomplete subdirectory can resume from its checkpoints.
    # ------------------------------------------------------------------

    for subdir in INPUT_SUBDIRECTORIES:
        process_subdirectory(
            subdir
        )

        # Release parent-process references before moving to the next
        # hexadecimal directory.
        gc.collect()

    # ------------------------------------------------------------------
    # ONLY HERE do we remove checkpoints.
    #
    # If any exception/KeyboardInterrupt occurred above, this code is never
    # reached, so checkpoints remain available for resume.
    # ------------------------------------------------------------------

    deleted = delete_all_checkpoints()

    print()
    print("=" * 78)
    print("STAGE 3 COMPLETE")
    print("=" * 78)
    print()

    print(
        f"Checkpoint files removed: {deleted}"
    )

    print()
    print(
        "No MIDI files were copied, moved, or modified."
    )

    print()


if __name__ == "__main__":
    main()