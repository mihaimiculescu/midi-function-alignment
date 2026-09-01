
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Los Angeles MIDI Dataset
Stage 1 — Structural Metadata Filtering

PURPOSE
-------
Use the LAMDa metadata to reduce the full dataset to a manageable
candidate pool BEFORE doing expensive MIDI parsing.

IMPORTANT
---------
This script DOES NOT:

    - copy MIDI files
    - move MIDI files
    - modify MIDI files
    - parse MIDI files
    - render MIDI files
    - remove drums
    - classify genre
    - identify melody/chord tracks

It only selects existing MIDI files and writes their absolute paths
to manifest files.

INPUT
-----
Metadata:

    ~/midi-function-alignment/Dataset/
        Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/
        META-DATA/LAMDa_META_DATA.pickle

MIDI dataset:

    ~/midi-function-alignment/Dataset/
        Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/MIDIs/

OUTPUT
------
Only manifests are written here:

    ~/Dataset/LAMDselection/selection_stage1/

    candidates.txt
        One absolute MIDI path per line.

    candidates.json
        Candidate paths plus selected structural metadata.

    summary.json
        Filter configuration and rejection statistics.

The MIDI files themselves remain completely untouched.

RUN
---
    cd ~/midi-function-alignment
    python3 stage1_filter_la_midi.py
"""

import json
import pickle
import sys
from pathlib import Path
from collections import Counter
from tqdm import tqdm
import numpy as np
from pretty_midi_fix import UglyMIDI
# import pretty_midi
import MIDI
import math
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor


# ============================================================================
# PATHS
# ============================================================================

METADATA_FILE = Path(
    "Dataset/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/"
    "META_DATA/LAMDa_META_DATA.pickle"
)

MIDI_ROOT = Path(
    "Dataset/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/"
    "MIDIs"
)

OUTPUT_DIR = Path(
    "Dataset/LAMDselection/selection_stage1"
)
CANDIDATES_TXT = OUTPUT_DIR / "candidates.txt"
CANDIDATES_JSON = OUTPUT_DIR / "candidates.json"
SUMMARY_JSON = OUTPUT_DIR / "summary.json"
REJECTIONS_JSON = OUTPUT_DIR / "rejections.json"
#+++
INPUT_MANIFEST = Path(
    "Dataset/LAMDselection/selection_stage1/candidates.txt"
)

OUTPUT_DIR1B = Path(
    "Dataset/LAMDselection/selection_stage1b"
)

SURVIVORS_TXT1B = OUTPUT_DIR1B / "candidates_track0_notes.txt"
SUMMARY_JSON1B = OUTPUT_DIR1B / "summary.json"
REJECTIONS_JSON1B = OUTPUT_DIR1B / "rejections.json"

# ============================================================================
# STAGE 3 EXECUTION
# ============================================================================

STAGE3_WORKERS = 16

# Save completed Stage 3 results every N files.
STAGE3_CHECKPOINT_INTERVAL = 1000

STAGE3_DIR = (
    OUTPUT_DIR /
    "stage3"
)
STAGE3_CANDIDATES_TXT = (
    STAGE3_DIR /
    "candidates.txt"
)

STAGE3_REPORTS_JSON = (
    STAGE3_DIR /
    "reports.json"
)

STAGE3_SUMMARY_JSON = (
    STAGE3_DIR /
    "summary.json"
)

STAGE3_REJECTIONS_JSON = (
    STAGE3_DIR /
    "rejections.json"
)


# ============================================================================
# STAGE 1 FILTERS
# ============================================================================
#
# These are deliberately conservative.
#
# Stage 1 is only supposed to remove obviously unsuitable material.
# We will use actual MIDI inspection in later stages to make musical
# decisions.
#
# DO NOT start by trying to identify "pop" here.
#
# ============================================================================

FILTERS = {

    # Files with fewer than two tracks are unlikely to give us a useful
    # melody/harmony relationship.
    "min_tracks": 2,

    # Avoid pathological multi-track files.
    "max_tracks": 16,

    # Reject almost-empty MIDI files.
    "min_score_events": 100,

    # Avoid obviously enormous/pathological files for the next stage.
    "max_score_events": 250000,

    # We ultimately need harmonic information.
    "min_chords": 20,

    # At least some meaningful chord activity.
    "min_chords_ms": 2000,

    # At least some meaningful pitched activity.
    "min_pitches_ms": 10000,

    # Do NOT reject tempo changes yet.
    "max_tempo_changes": None,

    # Lyrics are not a problem.
    "reject_lyrics": False,

    # Text events are not a problem.
    "max_text_events": None,
}
# ============================================================================
# STAGE 2 — MIDI-LEVEL MUSICAL FILTERS
# ============================================================================

STAGE2_FILTERS = {

    # Quantization:
    #
    # The project considers:
    #
    #   <= 0.45       good quantization
    #   0.45-0.65     suspicious
    #   > 0.65        bad
    #
    # We keep the original project's accepted regions.
    "quantization_good_max": 0.45,
    "quantization_bad_min": 0.65,

    # Instruments/tracks with fewer notes than this are ignored when
    # evaluating quantization.
    "min_notes_for_quantization": 20,

    # Reject files that cannot be parsed by UglyMIDI.
    "reject_parse_errors": True,

}

# ============================================================================
# STAGE 2 RESUME CHECK
# ============================================================================

STAGE2_CANDIDATES_TXT = (
    OUTPUT_DIR
    / "stage2"
    / "candidates.txt"
)

STAGE2_CANDIDATES_JSON = (
    OUTPUT_DIR
    / "stage2"
    / "candidates.json"
)

STAGE2_SUMMARY_JSON = (
    OUTPUT_DIR
    / "stage2"
    / "summary.json"
)
STAGE2_REJECTIONS_JSON = (
    OUTPUT_DIR
    / "stage2"
    / "rejections.json"
)


def load_manifest(path):
    """
    Load a newline-separated MIDI path manifest.
    """
    with open(
        path,
        "r",
        encoding="utf-8"
    ) as fh:

        return [
            line.strip()
            for line in fh
            if line.strip()
        ]


def stage2_is_complete():

    required_files = [
        STAGE2_CANDIDATES_TXT,
        STAGE2_CANDIDATES_JSON,
        STAGE2_SUMMARY_JSON,
        STAGE2_REJECTIONS_JSON,
    ]

    return all(
        path.is_file()
        for path in required_files
    )

# ============================================================================
# STAGE 3 — MIDI MUSICAL STRUCTURE ANALYSIS
# ============================================================================

STAGE3_FILTERS = {

    # General MIDI percussion channel.
    # MIDI channel numbers in the MIDI file representation are 0-based,
    # therefore GM channel 10 is represented as channel 9.
    "excluded_channels": {9, 15},

    # Ignore tiny tracks when considering melody candidates.
    "min_notes_for_candidate": 8,

    # Maximum number of candidate tracks retained in the report.
    "max_candidates_per_file": 8,

}

# ============================================================================
# HELPERS
# ============================================================================

def validate_metadata_record(record):
    """
    Validate and normalize one LAMDa metadata record for Stage 1.

    Stage 1 deliberately validates ONLY the metadata fields that it
    actually needs. Other LAMDa fields may contain lists, nested lists,
    counters, etc. and are left completely alone.

    Returns:
        (md5, metadata_dict, None)

    or:
        (md5, None, rejection_reason)
    """

    # --------------------------------------------------------------
    # Record shape
    # --------------------------------------------------------------
    if not isinstance(record, (list, tuple)) or len(record) < 2:
        return None, None, "metadata_record_wrong_shape"

    # --------------------------------------------------------------
    # MD5
    # --------------------------------------------------------------
    md5 = record[0]

    if not isinstance(md5, str) or not md5.strip():
        return None, None, "metadata_record_wrong_shape"

    md5 = md5.lower()

    # --------------------------------------------------------------
    # Metadata list
    # --------------------------------------------------------------
    raw_metadata = record[1]

    if raw_metadata is None:
        return md5, None, "metadata_list_missing"

    if not isinstance(raw_metadata, (list, tuple)):
        return md5, None, "metadata_list_missing"

    # --------------------------------------------------------------
    # Build a dictionary.
    #
    # IMPORTANT:
    #
    # We do NOT validate every metadata entry.
    #
    # LAMDa contains perfectly legitimate fields whose values are:
    #
    #     lists
    #     nested lists
    #     counters
    #     distributions
    #
    # Stage 1 does not care about those.
    #
    # We only extract the fields needed below.
    # --------------------------------------------------------------
    meta = {}

    for entry in raw_metadata:

        if not isinstance(entry, (list, tuple)):
            continue

        if len(entry) < 2:
            continue

        key = entry[0]

        if not isinstance(key, str):
            continue

        meta[key] = entry[1]

    # --------------------------------------------------------------
    # Required Stage 1 fields
    # --------------------------------------------------------------
    required_fields = [
        "total_number_of_tracks",
        "total_number_of_score_midi_events",
        "total_number_of_chords",
        "total_number_of_chords_ms",
        "pitches_times_sum_ms",
    ]

    for field in required_fields:

        if field not in meta:
            return md5, None, "required_field_absent"

    # --------------------------------------------------------------
    # Fields that Stage 1 interprets numerically
    #
    # DO NOT include list-valued metadata fields here.
    # --------------------------------------------------------------
    numeric_fields = [
        "total_number_of_tracks",
        "total_number_of_score_midi_events",
        "total_number_of_chords",
        "total_number_of_chords_ms",
        "pitches_times_sum_ms",
    ]

    # Optional scalar fields used by Stage 1.
    optional_numeric_fields = [
        "tempo_change_count",
        "text_events_count",
        "lyric_events_count",
    ]

    # --------------------------------------------------------------
    # Validate required numeric fields
    # --------------------------------------------------------------
    for field in numeric_fields:

        value = meta[field]

        try:
            int(value)

        except (TypeError, ValueError):
            return md5, None, "numeric_field_invalid"

    # --------------------------------------------------------------
    # Validate optional numeric fields if present.
    #
    # If absent, that is fine.
    # --------------------------------------------------------------
    for field in optional_numeric_fields:

        if field not in meta:
            continue

        try:
            int(meta[field])

        except (TypeError, ValueError):
            return md5, None, "numeric_field_invalid"

    # --------------------------------------------------------------
    # midi_patches
    #
    # This is relevant metadata, but its value is EXPECTED to be a
    # list/tuple. It must therefore NOT be treated as a scalar.
    # --------------------------------------------------------------
    if "midi_patches" not in meta:

        return md5, None, "midi_patches_missing_or_malformed"

    if not isinstance(
        meta["midi_patches"],
        (list, tuple)
    ):

        return md5, None, "midi_patches_missing_or_malformed"

    # --------------------------------------------------------------
    # Explicit validation of total_number_of_tracks
    # --------------------------------------------------------------
    try:

        tracks = int(
            meta["total_number_of_tracks"]
        )

        if tracks < 0:
            return md5, None, "total_number_of_tracks_invalid"

    except (TypeError, ValueError):

        return md5, None, "total_number_of_tracks_invalid"

    # --------------------------------------------------------------
    # Explicit validation of total_number_of_chords
    # --------------------------------------------------------------
    try:

        chords = int(
            meta["total_number_of_chords"]
        )

        if chords < 0:
            return md5, None, "total_number_of_chords_invalid"

    except (TypeError, ValueError):

        return md5, None, "total_number_of_chords_invalid"

    # --------------------------------------------------------------
    # Explicit validation of total_number_of_chords_ms
    # --------------------------------------------------------------
    try:

        chords_ms = int(
            meta["total_number_of_chords_ms"]
        )

        if chords_ms < 0:
            return md5, None, "total_number_of_chords_ms_invalid"

    except (TypeError, ValueError):

        return md5, None, "total_number_of_chords_ms_invalid"

    return md5, meta, None

def get_int(meta, key, default=0):
    """Safely return an integer metadata value."""

    value = meta.get(key, default)

    try:
        return int(value)

    except (TypeError, ValueError):
        return default

# ============================================================================
# HELPER STAGE 1B
# ============================================================================

def first_track_has_notes(midi_path):
    """
    Check whether the first actual MIDI track contains at least
    one sounding note.

    This intentionally does NOT use MIDI.midi2opus().

    We only need the first MTrk chunk, so we inspect the Standard
    MIDI File container directly and never parse subsequent tracks.

    Returns:
        (True, None)

    or:
        (False, reason)
    """

    try:

        with open(midi_path, "rb") as fh:

            # ----------------------------------------------------------
            # MIDI header
            # ----------------------------------------------------------

            header = fh.read(14)

            if len(header) < 14:

                return False, "midi_too_short"

            if header[0:4] != b"MThd":

                return False, "missing_mthd"

            header_length = int.from_bytes(
                header[4:8],
                byteorder="big"
            )

            if header_length < 6:

                return False, "invalid_midi_header"

            # ----------------------------------------------------------
            # Locate first track.
            #
            # Normally the header length is 6, but honor the actual
            # value in case of unusual files.
            # ----------------------------------------------------------

            if header_length > 6:

                fh.seek(
                    header_length - 6,
                    1
                )

            track_header = fh.read(8)

            if len(track_header) < 8:

                return False, "no_first_track"

            # ----------------------------------------------------------
            # First actual MIDI track must begin with MTrk.
            # ----------------------------------------------------------

            if track_header[0:4] != b"MTrk":

                return False, "first_track_not_mtrk"

            track_length = int.from_bytes(
                track_header[4:8],
                byteorder="big"
            )

            # ----------------------------------------------------------
            # Read ONLY the first track.
            # ----------------------------------------------------------

            track_data = fh.read(
                track_length
            )

            if len(track_data) != track_length:

                return False, "truncated_first_track"

            # ----------------------------------------------------------
            # Parse MIDI events sufficiently to identify note_on.
            #
            # We do not need to understand the entire musical event
            # structure. We only need to identify:
            #
            #   0x9n = note_on
            #
            # A note_on with velocity 0 is note_off.
            #
            # Running status is supported.
            # ----------------------------------------------------------

            i = 0
            running_status = None

            while i < len(track_data):

                # ------------------------------------------------------
                # Skip delta-time (variable-length quantity)
                # ------------------------------------------------------

                delta_bytes = 0

                while True:

                    if i >= len(track_data):

                        return False, "invalid_delta_time"

                    byte = track_data[i]
                    i += 1

                    delta_bytes += 1

                    if byte < 0x80:

                        break

                    if delta_bytes > 4:

                        return False, "invalid_delta_time"

                if i >= len(track_data):

                    break

                status_or_data = track_data[i]

                # ------------------------------------------------------
                # New status byte
                # ------------------------------------------------------

                if status_or_data & 0x80:

                    status = status_or_data
                    i += 1

                    # Meta event
                    if status == 0xFF:

                        if i >= len(track_data):

                            return False, "truncated_meta_event"

                        i += 1

                        length = 0

                        while True:

                            if i >= len(track_data):

                                return False, "truncated_meta_length"

                            byte = track_data[i]
                            i += 1

                            length = (
                                (length << 7)
                                | (byte & 0x7F)
                            )

                            if byte < 0x80:

                                break

                        i += length

                        if i > len(track_data):

                            return False, "truncated_meta_event"

                        running_status = None

                        continue

                    # SysEx
                    if status in (0xF0, 0xF7):

                        length = 0

                        while True:

                            if i >= len(track_data):

                                return False, "truncated_sysex_length"

                            byte = track_data[i]
                            i += 1

                            length = (
                                (length << 7)
                                | (byte & 0x7F)
                            )

                            if byte < 0x80:

                                break

                        i += length

                        if i > len(track_data):

                            return False, "truncated_sysex"

                        running_status = None

                        continue

                    # Channel message
                    running_status = status

                else:

                    # --------------------------------------------------
                    # Running status
                    # --------------------------------------------------

                    if running_status is None:

                        return False, "running_status_missing"

                    status = running_status

                # ------------------------------------------------------
                # Channel message
                # ------------------------------------------------------

                message_type = status & 0xF0

                # Note On
                if message_type == 0x90:

                    # If this was running status, status byte was not
                    # consumed above, so current byte is still data.
                    #
                    # If it was a new status, current byte is first
                    # data byte.
                    if i >= len(track_data):

                        return False, "truncated_note_on"

                    note = track_data[i]
                    i += 1

                    if i >= len(track_data):

                        return False, "truncated_note_on"

                    velocity = track_data[i]
                    i += 1

                    # Real sounding note.
                    if velocity != 0:

                        return True, None

                    continue

                # ------------------------------------------------------
                # Other channel messages
                # ------------------------------------------------------

                if message_type in (
                    0x80,  # note off
                    0xA0,  # polyphonic aftertouch
                    0xB0,  # controller
                    0xE0,  # pitch bend
                ):

                    i += 2

                elif message_type in (
                    0xC0,  # program change
                    0xD0,  # channel pressure
                ):

                    i += 1

                else:

                    return False, "unknown_midi_event"

                if i > len(track_data):

                    return False, "truncated_midi_event"

            return False, "first_track_has_no_notes"

    except OSError:

        return False, "file_read_error"

    except Exception as exc:

        return False, (
            "first_track_scan_error:"
            + type(exc).__name__
        )

# ============================================================================
# STAGE 2 HELPERS
# ============================================================================

def get_midi_time_signature(midi_path):
    """
    Read the first MIDI time-signature event directly through
    the project's MIDI.py parser.

    MIDI.py represents a time-signature event as:

        ['time_signature',
         delta_time,
         numerator,
         denominator,
         clocks_per_click,
         notated_32nd_notes_per_beat]

    Returns:

        ((numerator, denominator), None)

    or:

        (None, reason)

    If the MIDI contains no time-signature event, 4/4 is used.

    PrettyMIDI is deliberately NOT used here.
    """

    try:

        with open(
            midi_path,
            "rb"
        ) as fh:

            midi_data = fh.read()

        opus = MIDI.midi2opus(
            midi_data
        )

    except Exception as exc:

        return None, (
            "time_signature_parse_error:"
            + type(exc).__name__
        )

    if not opus or len(opus) < 2:

        return (4, 4), None

    # Search ALL tracks.
    #
    # This is important for Type 1 MIDI files because the
    # time-signature event is not required to be on track 0.

    for track in opus[1:]:

        for event in track:

            if not event:
                continue

            if event[0] != "time_signature":
                continue

            try:

                numerator = int(
                    event[2]
                )

                denominator = int(
                    event[3]
                )

            except (
                TypeError,
                ValueError,
                IndexError
            ):

                return None, (
                    "invalid_time_signature_event"
                )

            if (
                numerator <= 0
                or denominator <= 0
            ):

                return None, (
                    "invalid_time_signature_event"
                )

            return (
                numerator,
                denominator
            ), None

    # No explicit time signature.
    # MIDI default = 4/4.

    return (4, 4), None


def beat_div_from_time_signature(
    numerator,
    denominator,
):
    """
    Determine the number of subdivisions per beat used by the
    CP-transformer preprocessing grid.

    Our definition of "beat" follows the actual denominator:

        4/4 -> quarter-note beat -> 4 sixteenth subdivisions
        3/4 -> quarter-note beat -> 4 sixteenth subdivisions

        6/8 -> eighth-note beat  -> 2 sixteenth subdivisions
        9/8 -> eighth-note beat  -> 2 sixteenth subdivisions
        12/8 -> eighth-note beat -> 2 sixteenth subdivisions

    This deliberately does NOT assume that every meter has a
    quarter-note beat.
    """

    if denominator == 8:
        return 2

    if denominator == 16:
        return 1

    if denominator == 2:
        return 8

    if denominator == 1:
        return 16

    # Quarter-note denominator.
    return 4


def analyze_stage2_quantization(
    midi_path,
):
    """
    Analyze MIDI quantization using the project's UglyMIDI
    representation.

    Returns:

        {
            "beat_div": ...,
            "time_signature": [numerator, denominator],
            "quantization_score": ...,
            "usable_instruments": ...,
            "total_notes": ...,
        }

    or:

        (None, reason)
    """

    time_signature, reason = get_midi_time_signature(
        midi_path
    )

    if reason is not None:

        return None, reason

    numerator, denominator = time_signature    

    beat_div = beat_div_from_time_signature(
        numerator,
        denominator,
    )

    try:

        midi = UglyMIDI(
            str(midi_path),
            constant_tempo=60.0 / beat_div,
        )

    except Exception as exc:

        return None, (
            "midi_parse_error:"
            + type(exc).__name__
        )

    try:

        midi_end_time = int(
            midi.get_end_time()
        )

    except Exception as exc:

        return None, (
            "midi_analysis_error:"
            + type(exc).__name__
        )

    if midi_end_time <= 0:

        return None, "empty_midi"

    best_statistics = 1.0

    usable_instruments = 0
    total_notes = 0

    for ins in midi.instruments:

        note_count = len(ins.notes)

        if note_count <= STAGE2_FILTERS[
            "min_notes_for_quantization"
        ]:

            continue

        usable_instruments += 1
        total_notes += note_count

        statistics = np.zeros(
            beat_div,
            dtype=np.uint32,
        )

        for note in ins.notes:

            start_time = int(
                round(note.start)
            )

            statistics[
                start_time % beat_div
            ] += 1

        # Same statistic used by the project's
        # filter_la_quantization().
        statistic = (
            statistics[1::2].sum()
            / len(ins.notes)
        )

        best_statistics = min(
            best_statistics,
            statistic,
        )

    # No sufficiently populated instrument.
    if usable_instruments == 0:

        return None, (
            "no_instrument_with_enough_notes"
        )

    # Same acceptance logic as the existing
    # preprocessing code:
    #
    # <= 0.45  -> likely well quantized
    # > 0.65   -> bad
    #
    # The 0.45-0.65 region is intentionally rejected.
    if not (
        best_statistics <= STAGE2_FILTERS[
            "quantization_good_max"
        ]
        or
        (
            best_statistics >
            STAGE2_FILTERS[
                "quantization_bad_min"
            ]
            and
            best_statistics < 1.0
        )
    ):

        return None, (
            "poor_or_ambiguous_quantization"
        )

    return {
        "beat_div": beat_div,
        "time_signature": [
            numerator,
            denominator,
        ],
        "quantization_score":
            float(best_statistics),
        "usable_instruments":
            usable_instruments,
        "total_notes":
            total_notes,
    }, None

# ============================================================================
# STAGE 3 HELPERS
# ============================================================================

def _safe_mean(values):
    if not values:
        return 0.0

    return float(
        sum(values) / len(values)
    )


def _safe_median(values):
    if not values:
        return 0.0

    values = sorted(values)
    n = len(values)

    if n % 2:
        return float(values[n // 2])

    return float(
        (values[n // 2 - 1] + values[n // 2]) / 2
    )


def _calculate_polyphony(notes):
    """
    Calculate average simultaneous-note polyphony.

    Uses a sweep-line event algorithm rather than comparing every
    note against every other note.

    Returns:
        average number of simultaneously sounding notes.
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
            (start, 1)
        )

        events.append(
            (end, -1)
        )

    if not events:
        return 0.0

    # At identical timestamps, note-offs must occur before note-ons.
    events.sort(
        key=lambda event: (
            event[0],
            event[1]
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

        # Apply all events at this timestamp.
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


def _calculate_pitch_range(notes):

    if not notes:
        return 0

    pitches = [
        int(note.pitch)
        for note in notes
    ]

    return max(pitches) - min(pitches)


def _calculate_pitch_mean(notes):

    if not notes:
        return 0.0

    return _safe_mean(
        [
            float(note.pitch)
            for note in notes
        ]
    )


def _calculate_note_duration_stats(notes):

    durations = []

    for note in notes:

        duration = (
            float(note.end)
            -
            float(note.start)
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
        "mean":
            _safe_mean(durations),

        "median":
            _safe_median(durations),

        "min":
            float(min(durations)),

        "max":
            float(max(durations)),
    }


def _calculate_activity_span(notes):

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
        end - start
    )


def _calculate_note_density(notes):

    if not notes:
        return 0.0

    span = _calculate_activity_span(
        notes
    )

    if span <= 0:
        return 0.0

    return float(
        len(notes) / span
    )


def _calculate_monophonic_fraction(notes):
    """
    Fraction of note onsets which do not have another note sounding
    simultaneously.

    Uses an interval sweep rather than O(n²) note comparisons.
    """

    if not notes:
        return 0.0

    # Sort by start time.
    sorted_notes = sorted(
        notes,
        key=lambda note: (
            float(note.start),
            float(note.end)
        )
    )

    # Min-heap of active note end times.
    import heapq

    active_end_times = []

    isolated = 0

    for note in sorted_notes:

        start = float(note.start)
        end = float(note.end)

        # Remove notes which have already ended.
        while (
            active_end_times
            and active_end_times[0] <= start
        ):
            heapq.heappop(
                active_end_times
            )

        # If no other note is sounding, this onset is isolated.
        if not active_end_times:
            isolated += 1

        heapq.heappush(
            active_end_times,
            end
        )

    return float(
        isolated / len(notes)
    )


def _calculate_channel_information(
    instrument
):

    channels = set()

    # PrettyMIDI instruments normally have one program/channel.
    # Some project representations may expose channel information
    # differently, so be defensive.
    if hasattr(
        instrument,
        "channel"
    ):

        try:
            channels.add(
                int(instrument.channel)
            )
        except (
            TypeError,
            ValueError
        ):
            pass

    if hasattr(
        instrument,
        "channels"
    ):

        try:

            for channel in instrument.channels:

                channels.add(
                    int(channel)
                )

        except Exception:
            pass

    return sorted(
        channels
    )


def _melody_score(
    notes,
    polyphony,
    pitch_range,
    activity_span,
    note_density,
    monophonic_fraction,
):
    """
    Heuristic melody likelihood.

    This is NOT intended to identify the melody with certainty.

    It produces a ranking score so that Stage 3 can tell us:

        "These tracks look melody-like."

    rather than making an irreversible selection.
    """

    if not notes:
        return 0.0

    score = 0.0

    # --------------------------------------------------------------
    # Monophonic material is strongly melody-like.
    # --------------------------------------------------------------

    score += (
        monophonic_fraction
        * 0.35
    )

    # --------------------------------------------------------------
    # Moderate polyphony is acceptable.
    #
    # A melody can occasionally contain doubled notes/chords.
    # Penalize increasingly dense polyphony.
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
        * 0.25
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
        * 0.15
    )

    # --------------------------------------------------------------
    # Some sustained activity across the piece.
    # --------------------------------------------------------------

    if activity_span > 0:

        density_score = min(
            1.0,
            note_density / 2.0
        )

    else:

        density_score = 0.0

    score += (
        density_score
        * 0.10
    )

    # --------------------------------------------------------------
    # Number of notes.
    #
    # Enough notes to constitute a meaningful musical line.
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
        * 0.05
    )

    return float(
        max(
            0.0,
            min(
                1.0,
                score
            )
        )
    )


def analyze_stage3_midi(
    midi_path
):

    """
    Analyze one MIDI file for melody-like track candidates.

    Returns:

        {
            "md5": ...,
            "tracks_with_notes": ...,
            "non_percussion_notes": ...,
            "candidates": [...]
        }

    or:

        (None, reason)
    """

    try:

        midi = UglyMIDI(
            str(midi_path)
        )

    except Exception as exc:

        return None, (
            "midi_parse_error:"
            + type(exc).__name__
        )

    try:

        track_reports = []

        total_non_percussion_notes = 0

        tracks_with_notes = 0

        # ----------------------------------------------------------
        # Examine every instrument/track.
        # ----------------------------------------------------------

        for track_index, instrument in enumerate(
            midi.instruments
        ):

            raw_notes = list(
                instrument.notes
            )

            if not raw_notes:
                continue

            tracks_with_notes += 1

            channels = (
                _calculate_channel_information(
                    instrument
                )
            )

            # ------------------------------------------------------
            # Remove percussion channels.
            #
            # We deliberately remove the NOTES rather than deleting
            # the whole track.
            # ------------------------------------------------------

            filtered_notes = []

            for note in raw_notes:

                channel = getattr(
                    note,
                    "channel",
                    None
                )

                if channel in STAGE3_FILTERS[
                    "excluded_channels"
                ]:
                    continue

                filtered_notes.append(
                    note
                )

            if not filtered_notes:
                continue

            total_non_percussion_notes += (
                len(filtered_notes)
            )

            if len(filtered_notes) < STAGE3_FILTERS[
                "min_notes_for_candidate"
            ]:
                continue

            pitch_range = (
                _calculate_pitch_range(
                    filtered_notes
                )
            )

            polyphony = (
                _calculate_polyphony(
                    filtered_notes
                )
            )

            activity_span = (
                _calculate_activity_span(
                    filtered_notes
                )
            )

            note_density = (
                _calculate_note_density(
                    filtered_notes
                )
            )

            monophonic_fraction = (
                _calculate_monophonic_fraction(
                    filtered_notes
                )
            )

            duration_stats = (
                _calculate_note_duration_stats(
                    filtered_notes
                )
            )

            pitch_mean = (
                _calculate_pitch_mean(
                    filtered_notes
                )
            )

            score = _melody_score(
                filtered_notes,
                polyphony,
                pitch_range,
                activity_span,
                note_density,
                monophonic_fraction,
            )

            track_reports.append(
                {
                    "track":
                        track_index,

                    "channels":
                        channels,

                    "program":
                        getattr(
                            instrument,
                            "program",
                            None
                        ),

                    "is_drum":
                        bool(
                            getattr(
                                instrument,
                                "is_drum",
                                False
                            )
                        ),

                    "raw_notes":
                        len(raw_notes),

                    "non_percussion_notes":
                        len(filtered_notes),

                    "polyphony":
                        round(
                            polyphony,
                            4
                        ),

                    "monophonic_fraction":
                        round(
                            monophonic_fraction,
                            4
                        ),

                    "pitch_range":
                        pitch_range,

                    "pitch_mean":
                        round(
                            pitch_mean,
                            2
                        ),

                    "note_density":
                        round(
                            note_density,
                            4
                        ),

                    "activity_span":
                        round(
                            activity_span,
                            4
                        ),

                    "duration":
                        duration_stats,

                    "melody_score":
                        round(
                            score,
                            4
                        ),
                }
            )

        # ----------------------------------------------------------
        # Rank candidates.
        # ----------------------------------------------------------

        track_reports.sort(
            key=lambda item:
                item["melody_score"],
            reverse=True,
        )

        track_reports = track_reports[
            :STAGE3_FILTERS[
                "max_candidates_per_file"
            ]
        ]

        return {
            "tracks_with_notes":
                tracks_with_notes,

            "non_percussion_notes":
                total_non_percussion_notes,

            "candidate_count":
                len(track_reports),

            "candidates":
                track_reports,
        }, None

    except Exception as exc:

        return None, (
            "midi_analysis_error:"
            + type(exc).__name__
        )


def stage3_worker(candidate):
    """
    Worker executed in a separate process.

    Returns:
        {
            "candidate": original Stage 2 candidate,
            "analysis": Stage 3 analysis,
            "reason": None
        }

    or the same structure with reason populated.
    """

    midi_path = candidate["path"]

    analysis, reason = analyze_stage3_midi(
        midi_path
    )

    return {
        "candidate": candidate,
        "analysis": analysis,
        "reason": reason,
    }


def make_json_serializable(obj):
    """
    Recursively convert NumPy/scalar/container values into ordinary
    Python objects accepted by json.dump().
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
        except (
            ValueError,
            TypeError
        ):
            pass

    if hasattr(obj, "tolist"):

        try:
            return obj.tolist()
        except (
            ValueError,
            TypeError
        ):
            pass

    return obj

def write_stage3_checkpoint(
    checkpoint_path,
    reports,
    survivors,
    rejection_counts,
    rejection_details,
    completed_count,
    input_count,
    last_md5,
):
    """
    Atomically save Stage 3 progress.

    The checkpoint records both the number of completed files and
    the MD5 of the last completed candidate, allowing the next run
    to verify that the Stage 2 candidate ordering has not changed.
    """

    checkpoint = {
        "completed_count":
            completed_count,

        "input_count":
            input_count,

        "last_md5":
            last_md5,

        "reports":
            reports,

        "survivors":
            survivors,

        "rejection_counts":
            dict(rejection_counts),

        "rejection_details":
            rejection_details,
    }

    checkpoint = make_json_serializable(
        checkpoint
    )

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=checkpoint_path.parent,
        delete=False,
        prefix=".stage3_checkpoint_",
        suffix=".tmp",
    ) as fh:

        json.dump(
            checkpoint,
            fh,
            indent=2,
            ensure_ascii=False,
        )

        temp_path = Path(
            fh.name
        )

    os.replace(
        temp_path,
        checkpoint_path
    )


def load_stage3_checkpoint(
    checkpoint_path
):
    """
    Load an existing Stage 3 checkpoint.

    Returns None if no checkpoint exists.
    """

    if not checkpoint_path.is_file():
        return None

    with open(
        checkpoint_path,
        "r",
        encoding="utf-8",
    ) as fh:

        return json.load(fh)

    
# ============================================================================
# BUILD MIDI INDEX
# ============================================================================

def build_midi_index():
    """
    Build:

        MD5 -> absolute MIDI path

    LA MIDI files are named after their MD5 hash.

    Example:

        d9a7e1c6a375b8e560155a5977fc10f8

    becomes:

        /.../MIDIs/d/
            d9a7e1c6a375b8e560155a5977fc10f8.mid

    No MIDI files are opened here.
    """

    print("=" * 70)
    print("Creating MIDI path index...")
    print("=" * 70)

    if not MIDI_ROOT.exists():

        print()
        print("ERROR: MIDI root does not exist:")
        print(MIDI_ROOT)
        print()

        sys.exit(1)

    midi_index = {}

    total_files = 0
    duplicate_files = 0

    # The dataset is organized into hexadecimal subdirectories.
    # rglob is used only to locate paths; files are not read.
    for path in MIDI_ROOT.rglob("*.mid"):

        total_files += 1

        md5 = path.stem.lower()

        if md5 in midi_index:

            duplicate_files += 1

            print()
            print("WARNING: duplicate MD5 filename:")
            print("  MD5:", md5)
            print("  Existing:", midi_index[md5])
            print("  Duplicate:", path)

            continue

        midi_index[md5] = str(path.resolve())

    print()
    print(f"Physical .mid files found : {total_files:,}")
    print(f"Unique MD5 filenames       : {len(midi_index):,}")
    print(f"Duplicate filenames        : {duplicate_files:,}")
    print()

    return midi_index


# ============================================================================
# STRUCTURAL FILTER
# ============================================================================

def passes_stage1(meta):
    """
    Apply metadata-only Stage 1 filtering.

    Returns:

        (True, None)

    for a surviving record.

    Or:

        (False, reason)

    for a rejected record.
    """

    tracks = get_int(
        meta,
        "total_number_of_tracks"
    )

    score_events = get_int(
        meta,
        "total_number_of_score_midi_events"
    )

    chords = get_int(
        meta,
        "total_number_of_chords"
    )

    chords_ms = get_int(
        meta,
        "total_number_of_chords_ms"
    )

    pitches_ms = get_int(
        meta,
        "pitches_times_sum_ms"
    )

    tempo_changes = get_int(
        meta,
        "tempo_change_count"
    )

    text_events = get_int(
        meta,
        "text_events_count"
    )

    lyric_events = get_int(
        meta,
        "lyric_events_count"
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
    # Event count
    # ------------------------------------------------------------------

    if score_events < FILTERS["min_score_events"]:
        return False, "too_few_score_events"

    if FILTERS["max_score_events"] is not None:

        if score_events > FILTERS["max_score_events"]:
            return False, "too_many_score_events"

    # ------------------------------------------------------------------
    # Harmonic activity
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
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 70)
    print("Los Angeles MIDI Dataset")
    print("STAGE 1 — STRUCTURAL METADATA FILTERING")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Validate input paths
    # ------------------------------------------------------------------

    if not METADATA_FILE.exists():

        print("ERROR: Metadata file does not exist:")
        print(METADATA_FILE)
        sys.exit(1)

    if not MIDI_ROOT.exists():

        print("ERROR: MIDI dataset directory does not exist:")
        print(MIDI_ROOT)
        sys.exit(1)

    # ------------------------------------------------------------------
    # Create output directory.
    #
    # IMPORTANT:
    # This directory contains MANIFESTS ONLY.
    # No MIDI files are copied here.
    # ------------------------------------------------------------------

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------------
    # Load metadata
    # ------------------------------------------------------------------

    print("=" * 70)
    print("Loading LAMDa META-DATA...")
    print("=" * 70)

    print(METADATA_FILE)
    print()

    with open(
        METADATA_FILE,
        "rb"
    ) as fh:

        meta_data = pickle.load(fh)

    print(f"Metadata records: {len(meta_data):,}")
    print()

    # ------------------------------------------------------------------
    # Build MD5 -> path index
    # ------------------------------------------------------------------

    midi_index = build_midi_index()

    # ------------------------------------------------------------------
    # Apply filters
    # ------------------------------------------------------------------

    print("=" * 70)
    print("Applying Stage 1 structural filters...")
    print("=" * 70)
    print()

    candidates = []

    rejection_counts = Counter()

    missing_midi = 0

    rejection_details = []

    for record in meta_data:

        # --------------------------------------------------------------
        # Validate metadata record
        # --------------------------------------------------------------

        md5, meta, metadata_reason = validate_metadata_record(
            record
        )

        if metadata_reason is not None:

            rejection_counts[
                metadata_reason
            ] += 1

            rejection_details.append(
                {
                    "md5": md5,
                    "reason": metadata_reason,
                }
            )

            continue

        # --------------------------------------------------------------
        # Locate physical MIDI
        # --------------------------------------------------------------

        midi_path = midi_index.get(md5)

        if midi_path is None:

            missing_midi += 1

            rejection_counts[
                "missing_midi"
            ] += 1

            rejection_details.append(
                {
                    "md5": md5,
                    "reason": "missing_midi",
                }
            )

            continue

        # --------------------------------------------------------------
        # Apply structural filters
        # --------------------------------------------------------------

        passed, reason = passes_stage1(meta)

        if not passed:

            rejection_counts[
                reason
            ] += 1

            rejection_details.append(
                {
                    "md5": md5,
                    "path": midi_path,
                    "reason": reason,
                }
            )

            continue

        # --------------------------------------------------------------
        # Preserve useful metadata.
        #
        # This is metadata only; the MIDI itself is untouched.
        # --------------------------------------------------------------

        candidate = {

            "md5":
                md5,

            "path":
                midi_path,

            "total_number_of_tracks":
                get_int(
                    meta,
                    "total_number_of_tracks"
                ),

            "total_number_of_opus_midi_events":
                get_int(
                    meta,
                    "total_number_of_opus_midi_events"
                ),

            "total_number_of_score_midi_events":
                get_int(
                    meta,
                    "total_number_of_score_midi_events"
                ),

            "average_median_mode_time_ms":
                meta.get(
                    "average_median_mode_time_ms"
                ),

            "average_median_mode_dur_ms":
                meta.get(
                    "average_median_mode_dur_ms"
                ),

            "average_median_mode_vel":
                meta.get(
                    "average_median_mode_vel"
                ),

            "total_number_of_chords":
                get_int(
                    meta,
                    "total_number_of_chords"
                ),

            "total_number_of_chords_ms":
                get_int(
                    meta,
                    "total_number_of_chords_ms"
                ),

            "pitches_times_sum_ms":
                get_int(
                    meta,
                    "pitches_times_sum_ms"
                ),

            "midi_patches":
                meta.get(
                    "midi_patches",
                    []
                ),

            "total_patches_counts":
                meta.get(
                    "total_patches_counts",
                    []
                ),

            "tempo_change_count":
                get_int(
                    meta,
                    "tempo_change_count"
                ),

            "text_events_count":
                get_int(
                    meta,
                    "text_events_count"
                ),

            "lyric_events_count":
                get_int(
                    meta,
                    "lyric_events_count"
                ),
        }

        candidates.append(candidate)

    # =========================================================================
    # REPORT
    # =========================================================================

    total_records = len(meta_data)
    candidate_count = len(candidates)
    rejected_count = total_records - candidate_count

    print()
    print("=" * 70)
    print("STAGE 1 COMPLETE")
    print("=" * 70)

    print()
    print(f"Metadata records       : {total_records:,}")
    print(f"MIDI files indexed     : {len(midi_index):,}")
    print(f"Candidates             : {candidate_count:,}")
    print(f"Rejected               : {rejected_count:,}")
    print(f"Missing MIDI           : {missing_midi:,}")
    # print(f"Malformed metadata     : {malformed_metadata:,}")
    metadata_rejection_total = sum(
        count
        for reason, count in rejection_counts.items()
        if reason in {
            "metadata_record_wrong_shape",
            "metadata_list_missing",
            "required_field_absent",
            "required_field_wrong_type",
            "numeric_field_invalid",
            "midi_patches_missing_or_malformed",
            "total_number_of_tracks_invalid",
            "total_number_of_chords_invalid",
            "total_number_of_chords_ms_invalid",
        }
    )

    print(
        f"Metadata validation rejects : "
        f"{metadata_rejection_total:,}"
    )

    if total_records > 0:

        print(
            f"Candidate ratio       : "
            f"{candidate_count / total_records:.4f}"
        )

    print()
    print("Rejection reasons:")
    print("-" * 70)

    for reason, count in rejection_counts.most_common():

        print(
            f"{reason:35s} {count:10,}"
        )

    # =========================================================================
    # WRITE candidates.txt
    # =========================================================================

    print()
    print("=" * 70)
    print("Writing candidate path manifest...")
    print("=" * 70)

    with open(
        CANDIDATES_TXT,
        "w",
        encoding="utf-8"
    ) as fh:

        for candidate in candidates:

            fh.write(
                candidate["path"]
                + "\n"
            )

    print()
    print(CANDIDATES_TXT)

    # =========================================================================
    # WRITE candidates.json
    # =========================================================================

    with open(
        CANDIDATES_JSON,
        "w",
        encoding="utf-8"
    ) as fh:

        json.dump(
            {
                "dataset":
                    "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

                "stage":
                    1,

                "description":
                    (
                        "Structurally filtered MIDI candidates. "
                        "MIDI files are not copied or modified."
                    ),

                "metadata_source":
                    str(METADATA_FILE),

                "midi_root":
                    str(MIDI_ROOT),

                "filters":
                    FILTERS,

                "candidate_count":
                    candidate_count,

                "candidates":
                    candidates,
            },
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(CANDIDATES_JSON)

    # =========================================================================
    # WRITE summary.json
    # =========================================================================

    summary = {

        "dataset":
            "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

        "stage":
            1,

        "metadata_file":
            str(METADATA_FILE),

        "midi_root":
            str(MIDI_ROOT),

        "output_directory":
            str(OUTPUT_DIR),

        "metadata_records":
            total_records,

        "midi_files_indexed":
            len(midi_index),

        "candidates":
            candidate_count,

        "rejected":
            rejected_count,

        "missing_midi":
            missing_midi,

        "metadata_validation_rejects":
            {
                reason: rejection_counts[reason]
                for reason in (
                    "metadata_record_wrong_shape",
                    "metadata_list_missing",
                    "required_field_absent",
                    "required_field_wrong_type",
                    "numeric_field_invalid",
                    "midi_patches_missing_or_malformed",
                    "total_number_of_tracks_invalid",
                    "total_number_of_chords_invalid",
                    "total_number_of_chords_ms_invalid",
                )
                if rejection_counts[reason] > 0
            },

        "candidate_ratio":
            (
                candidate_count / total_records
                if total_records
                else 0
            ),

        "rejection_counts":
            dict(rejection_counts),

        "filters":
            FILTERS,

        "outputs":
            {
                "candidate_paths":
                    str(CANDIDATES_TXT),

                "candidate_metadata":
                    str(CANDIDATES_JSON),

                "summary":
                    str(SUMMARY_JSON),

                "rejections":
                    str(REJECTIONS_JSON),
            },
    }

    with open(
        SUMMARY_JSON,
        "w",
        encoding="utf-8"
    ) as fh:

        json.dump(
            summary,
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print()
    print(SUMMARY_JSON)

    # =========================================================================
    # WRITE rejection details
    # =========================================================================

    print()
    print("=" * 70)
    print("Writing rejection details...")
    print("=" * 70)

    with open(
        REJECTIONS_JSON,
        "w",
        encoding="utf-8"
    ) as fh:

        json.dump(
            {
                "dataset":
                    "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

                "stage":
                    1,

                "description":
                    (
                        "Every MIDI rejected by Stage 1, with its "
                        "primary rejection reason."
                    ),

                "rejection_count":
                    len(rejection_details),

                "rejections":
                    rejection_details,
            },

            fh,

            indent=2,

            ensure_ascii=False,
        )

    print()
    print(REJECTIONS_JSON)

    # =========================================================================
    # FINAL
    # =========================================================================

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    print()
    print(
        "No MIDI files were copied, moved, opened, "
        "or modified."
    )

    print()
    print(
        f"{candidate_count:,} existing MIDI files survived Stage 1."
    )

    print()
    print(
        "The main manifest is:"
    )

    print(
        f"  {CANDIDATES_TXT}"
    )

    print()
#STAGE1B
    # =========================================================================
    # STAGE 1B — FIRST TRACK NOTE FILTER
    # =========================================================================

    print()
    print("=" * 70)
    print("STAGE 1B — FIRST TRACK NOTE FILTER")
    print("=" * 70)
    print()

    # ------------------------------------------------------------------
    # Stage 1 output becomes Stage 1B input
    # ------------------------------------------------------------------

    INPUT_MANIFEST = OUTPUT_DIR / "candidates.txt"

    OUTPUT_DIR1B = OUTPUT_DIR / "stage1b"

    SURVIVORS_TXT1B = OUTPUT_DIR1B / "candidates.txt"
    SUMMARY_JSON1B = OUTPUT_DIR1B / "summary.json"
    REJECTIONS_JSON1B = OUTPUT_DIR1B / "rejections.json"

    OUTPUT_DIR1B.mkdir(
        parents=True,
        exist_ok=True
    )

    # ------------------------------------------------------------------
    # Load Stage 1 candidates
    # ------------------------------------------------------------------

    print("=" * 70)
    print("Loading Stage 1 candidate manifest...")
    print("=" * 70)
    print()

    with open(
        INPUT_MANIFEST,
        "r",
        encoding="utf-8"
    ) as fh:

        midi_paths = [
            line.strip()
            for line in fh
            if line.strip()
        ]

    print(
        f"Input candidates: {len(midi_paths):,}"
    )
    print()

    # ------------------------------------------------------------------
    # Process first MIDI track
    # ------------------------------------------------------------------

    survivors = []
    rejection_counts = Counter()
    rejection_details = []

    print("=" * 70)
    print("Scanning first MIDI track...")
    print("=" * 70)
    print()

    for midi_path_string in tqdm(midi_paths):

        midi_path = Path(midi_path_string)

        # --------------------------------------------------------------
        # Check that the path still exists
        # --------------------------------------------------------------

        if not midi_path.exists():

            reason = "missing_midi"

            rejection_counts[reason] += 1

            rejection_details.append(
                {
                    "path": midi_path_string,
                    "reason": reason,
                }
            )

            continue

        # --------------------------------------------------------------
        # Inspect first actual MIDI track
        # --------------------------------------------------------------

        passed, reason = first_track_has_notes(
            midi_path
        )

        if not passed:

            rejection_counts[reason] += 1

            rejection_details.append(
                {
                    "path": midi_path_string,
                    "reason": reason,
                }
            )

            continue

        # --------------------------------------------------------------
        # SURVIVED
        # --------------------------------------------------------------

        survivors.append(
            midi_path_string
        )

    # =========================================================================
    # STAGE 1B REPORT
    # =========================================================================

    input_count = len(midi_paths)
    survivor_count = len(survivors)
    rejected_count = input_count - survivor_count

    print()
    print("=" * 70)
    print("STAGE 1B COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Input candidates       : {input_count:,}"
    )

    print(
        f"Survivors              : {survivor_count:,}"
    )

    print(
        f"Rejected               : {rejected_count:,}"
    )

    if input_count:

        print(
            f"Survivor ratio         : "
            f"{survivor_count / input_count:.4f}"
        )

    print()

    print("Rejection reasons:")
    print("-" * 70)

    for reason, count in rejection_counts.most_common():

        print(
            f"{reason:35s} {count:10,}"
        )

    # =========================================================================
    # WRITE STAGE 1B SURVIVOR MANIFEST
    # =========================================================================

    print()
    print("=" * 70)
    print("Writing Stage 1B survivor manifest...")
    print("=" * 70)
    print()

    with open(
        SURVIVORS_TXT1B,
        "w",
        encoding="utf-8"
    ) as fh:

        for path in survivors:

            fh.write(
                path
                + "\n"
            )

    print(SURVIVORS_TXT1B)

    # =========================================================================
    # WRITE STAGE 1B SUMMARY
    # =========================================================================

    summary_1b = {

        "stage":
            "1B",

        "description":
            (
                "Keeps Stage 1 candidates whose first actual "
                "MIDI track contains at least one note event."
            ),

        "input_manifest":
            str(INPUT_MANIFEST),

        "output_manifest":
            str(SURVIVORS_TXT1B),

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
                else 0
            ),

        "rejection_counts":
            dict(rejection_counts),
    }

    with open(
        SUMMARY_JSON1B,
        "w",
        encoding="utf-8"
    ) as fh:

        json.dump(
            summary_1b,
            fh,
            indent=2,
            ensure_ascii=False
        )

    # =========================================================================
    # WRITE STAGE 1B REJECTION DETAILS
    # =========================================================================

    with open(
        REJECTIONS_JSON1B,
        "w",
        encoding="utf-8"
    ) as fh:

        json.dump(
            rejection_details,
            fh,
            indent=2,
            ensure_ascii=False
        )

    print()
    print(SUMMARY_JSON1B)
    print(REJECTIONS_JSON1B)

    # ============================================================================
    # STAGE 2
    # ============================================================================

    if stage2_is_complete():

        print()
        print("=" * 70)
        print("STAGE 2 ALREADY COMPLETE")
        print("=" * 70)
        print()

        print(
            "Stage 2 survivor manifest found:"
        )

        print(
            f"{STAGE2_CANDIDATES_TXT}"
        )

        print()

        stage2_survivors = load_manifest(
            STAGE2_CANDIDATES_TXT
        )

        print(
            f"Loaded Stage 2 survivors: "
            f"{len(stage2_survivors):,}"
        )

        with open(
            STAGE2_CANDIDATES_JSON,
            "r",
            encoding="utf-8"
        ) as fh:

            stage2_candidates = json.load(fh)["candidates"]


        print()

        print(
            "Skipping Stage 2 processing."
        )

        print(
            "Proceeding directly to Stage 3."
        )

    else:

        print()
        print("=" * 70)
        print("STAGE 2 NOT COMPLETE")
        print("=" * 70)
        print()

        print(
            "Stage 2 survivor manifest not found."
        )

        print(
            "Running Stage 2..."
        )

        print()

        # ------------------------------------------------------------------------
        # EXISTING STAGE 2 CODE GOES HERE
        # ------------------------------------------------------------------------

        print()
        print("=" * 70)
        print("STAGE 2 — MIDI-LEVEL QUANTIZATION FILTER")
        print("=" * 70)
        print()

        stage2_candidates = []
        stage2_rejection_counts = Counter()
        stage2_rejection_details = []

        input_candidates = len(candidates)

        print(
            f"Input Stage 1 candidates: "
            f"{input_candidates:,}"
        )

        print()
        print(
            "Parsing MIDI files and analyzing quantization..."
        )
        print()

        for candidate in tqdm(
            candidates,
            total=len(candidates),
        ):

            midi_path = candidate["path"]

            analysis, reason = analyze_stage2_quantization(
                midi_path
            )

            if reason is not None:

                stage2_rejection_counts[
                    reason
                ] += 1

                stage2_rejection_details.append(
                    {
                        "md5":
                            candidate["md5"],

                        "path":
                            midi_path,

                        "reason":
                            reason,
                    }
                )

                continue

            # --------------------------------------------------------------
            # Preserve Stage 1 information and append Stage 2 information.
            # --------------------------------------------------------------

            stage2_candidate = dict(
                candidate
            )

            stage2_candidate[
                "stage2"
            ] = analysis

            stage2_candidates.append(
                stage2_candidate
            )

        # =========================================================================
        # STAGE 2 REPORT
        # =========================================================================

        stage2_input_count = (
            len(candidates)
        )

        stage2_survivor_count = (
            len(stage2_candidates)
        )

        stage2_rejected_count = (
            stage2_input_count
            -
            stage2_survivor_count
        )

        print()
        print("=" * 70)
        print("STAGE 2 COMPLETE")
        print("=" * 70)
        print()

        print(
            f"Input candidates       : "
            f"{stage2_input_count:,}"
        )

        print(
            f"Survivors              : "
            f"{stage2_survivor_count:,}"
        )

        print(
            f"Rejected               : "
            f"{stage2_rejected_count:,}"
        )

        if stage2_input_count:

            print(
                f"Survivor ratio         : "
                f"{stage2_survivor_count / stage2_input_count:.4f}"
            )

        print()

        print(
            "Rejection reasons:"
        )

        print("-" * 70)

        for reason, count in (
            stage2_rejection_counts
            .most_common()
        ):

            print(
                f"{reason:40s}"
                f"{count:10,}"
            )

        # =========================================================================
        # STAGE 2 OUTPUT
        # =========================================================================

        STAGE2_DIR = (
            OUTPUT_DIR /
            "stage2"
        )

        STAGE2_DIR.mkdir(
            parents=True,
            exist_ok=True,
        )

        # STAGE2_CANDIDATES_TXT = (
        #     STAGE2_DIR /
        #     "candidates.txt"
        # )

        # STAGE2_CANDIDATES_JSON = (
        #     STAGE2_DIR /
        #     "candidates.json"
        # )

        # STAGE2_SUMMARY_JSON = (
        #     STAGE2_DIR /
        #     "summary.json"
        # )

        # STAGE2_REJECTIONS_JSON = (
        #     STAGE2_DIR /
        #     "rejections.json"
        # )

        print()
        print("=" * 70)
        print("Writing Stage 2 survivor manifests...")
        print("=" * 70)
        print()

        # --------------------------------------------------------------
        # Plain path manifest
        # --------------------------------------------------------------

        with open(
            STAGE2_CANDIDATES_TXT,
            "w",
            encoding="utf-8",
        ) as fh:

            for candidate in stage2_candidates:

                fh.write(
                    candidate["path"]
                    + "\n"
                )

        print(
            STAGE2_CANDIDATES_TXT
        )

        # --------------------------------------------------------------
        # Detailed candidate JSON
        # --------------------------------------------------------------

        with open(
            STAGE2_CANDIDATES_JSON,
            "w",
            encoding="utf-8",
        ) as fh:

            json.dump(
                {
                    "dataset":
                        "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

                    "stage":
                        "2",

                    "description":
                        (
                            "Stage 1 candidates filtered by "
                            "actual MIDI parsing and "
                            "time-signature-aware quantization."
                        ),

                    "input_manifest":
                        str(CANDIDATES_TXT),

                    "candidates":
                        stage2_candidates,

                },
                fh,
                indent=2,
                ensure_ascii=False,
            )

        print(
            STAGE2_CANDIDATES_JSON
        )

        # --------------------------------------------------------------
        # Summary
        # --------------------------------------------------------------

        stage2_summary = {

            "stage":
                "2",

            "description":
                (
                    "MIDI-level quantization filtering "
                    "of Stage 1 candidates."
                ),

            "input_candidates":
                stage2_input_count,

            "survivors":
                stage2_survivor_count,

            "rejected":
                stage2_rejected_count,

            "survivor_ratio":
                (
                    stage2_survivor_count /
                    stage2_input_count
                    if stage2_input_count
                    else 0
                ),

            "filters":
                STAGE2_FILTERS,

            "rejection_counts":
                dict(
                    stage2_rejection_counts
                ),

        }

        with open(
            STAGE2_SUMMARY_JSON,
            "w",
            encoding="utf-8",
        ) as fh:

            json.dump(
                stage2_summary,
                fh,
                indent=2,
                ensure_ascii=False,
            )

        print(
            STAGE2_SUMMARY_JSON
        )

        # --------------------------------------------------------------
        # Rejection details
        # --------------------------------------------------------------

        with open(
            STAGE2_REJECTIONS_JSON,
            "w",
            encoding="utf-8",
        ) as fh:

            json.dump(
                stage2_rejection_details,
                fh,
                indent=2,
                ensure_ascii=False,
            )

        print(
            STAGE2_REJECTIONS_JSON
        )

        print()
        print("=" * 70)
        print("STAGE 2 DONE")
        print("=" * 70)
        print()

        print(
            "No MIDI files were copied, moved, "
            "or modified."
        )

        print()

    # =========================================================================
    # STAGE 3 — MUSICAL STRUCTURE / MELODY CANDIDATE ANALYSIS
    # =========================================================================

    print()
    print("=" * 70)
    print("STAGE 3 — MUSICAL STRUCTURE ANALYSIS")
    print("=" * 70)
    print()


    STAGE3_DIR.mkdir(
        parents=True,
        exist_ok=True
    )


    print(
        f"Input Stage 2 candidates: "
        f"{len(stage2_candidates):,}"
    )

    print()
    print(
        "Analyzing MIDI track structure..."
    )
    print()

    stage3_reports = []
    stage3_survivors = []
    stage3_rejection_counts = Counter()
    stage3_rejection_details = []

    # =========================================================================
    # STAGE 3 CHECKPOINT
    # =========================================================================

    STAGE3_CHECKPOINT = (
        STAGE3_DIR /
        "checkpoint.json"
    )

    checkpoint = load_stage3_checkpoint(
        STAGE3_CHECKPOINT
    )


    if checkpoint is not None:

        print()
        print("=" * 70)
        print("STAGE 3 CHECKPOINT FOUND")
        print("=" * 70)
        print()

        completed_count = int(
            checkpoint.get(
                "completed_count",
                0
            )
        )

        checkpoint_input_count = int(
            checkpoint.get(
                "input_count",
                -1
            )
        )

        checkpoint_last_md5 = (
            checkpoint.get(
                "last_md5"
            )
        )

        current_input_count = len(
            stage2_candidates
        )

        # --------------------------------------------------------------
        # Validate that the checkpoint belongs to this exact Stage 2
        # candidate list.
        # --------------------------------------------------------------

        if checkpoint_input_count != current_input_count:

            raise RuntimeError(
                "Stage 3 checkpoint is incompatible with the "
                "current Stage 2 candidate list:\n"
                f"  checkpoint input count: "
                f"{checkpoint_input_count:,}\n"
                f"  current input count:    "
                f"{current_input_count:,}\n"
                "\n"
                "The Stage 2 candidate list has changed. "
                "Delete checkpoint.json and restart Stage 3."
            )

        # --------------------------------------------------------------
        # Verify the last processed MD5.
        # --------------------------------------------------------------

        if completed_count > 0:

            if completed_count > current_input_count:

                raise RuntimeError(
                    "Stage 3 checkpoint claims more completed "
                    "files than exist in the current Stage 2 list."
                )

            expected_md5 = stage2_candidates[
                completed_count - 1
            ]["md5"]

            if checkpoint_last_md5 != expected_md5:

                raise RuntimeError(
                    "Stage 3 checkpoint ordering mismatch.\n"
                    f"  checkpoint last MD5: "
                    f"{checkpoint_last_md5}\n"
                    f"  current expected MD5: "
                    f"{expected_md5}\n"
                    "\n"
                    "The Stage 2 candidate ordering has changed. "
                    "Delete checkpoint.json and restart Stage 3."
                )

        stage3_reports = (
            checkpoint.get(
                "reports",
                []
            )
        )

        stage3_survivors = (
            checkpoint.get(
                "survivors",
                []
            )
        )

        stage3_rejection_counts = Counter(
            checkpoint.get(
                "rejection_counts",
                {}
            )
        )

        stage3_rejection_details = (
            checkpoint.get(
                "rejection_details",
                []
            )
        )

        print(
            f"Checkpoint contains: "
            f"{completed_count:,} completed files"
        )

        print(
            f"Checkpoint ordering verified."
        )

    else:

        completed_count = 0

    # =========================================================================
    # DETERMINE REMAINING WORK
    # =========================================================================

    remaining_candidates = (
        stage2_candidates[
            completed_count:
        ]
    )

    print()
    print(
        f"Stage 3 workers       : "
        f"{STAGE3_WORKERS}"
    )

    print(
        f"Checkpoint interval   : "
        f"{STAGE3_CHECKPOINT_INTERVAL:,}"
    )

    print(
        f"Already completed     : "
        f"{completed_count:,}"
    )

    print(
        f"Remaining files       : "
        f"{len(remaining_candidates):,}"
    )

    print()

    # =========================================================================
    # PARALLEL STAGE 3
    # =========================================================================

    with ProcessPoolExecutor(
        max_workers=STAGE3_WORKERS
    ) as executor:

        results = executor.map(
            stage3_worker,
            remaining_candidates,
            chunksize=1,
        )

        for result in tqdm(
            results,
            total=len(remaining_candidates),
            initial=0,
        ):

            candidate = result["candidate"]
            analysis = result["analysis"]
            reason = result["reason"]

            completed_count += 1
            last_md5 = candidate["md5"]

            if reason is not None:

                stage3_rejection_counts[
                    reason
                ] += 1

                stage3_rejection_details.append(
                    {
                        "md5":
                            candidate["md5"],

                        "path":
                            candidate["path"],

                        "reason":
                            reason,
                    }
                )

            else:

                stage3_candidate = dict(
                    candidate
                )

                stage3_candidate[
                    "stage3"
                ] = analysis

                stage3_reports.append(
                    stage3_candidate
                )

                if analysis[
                    "candidate_count"
                ] > 0:

                    stage3_survivors.append(
                        stage3_candidate
                    )

                else:

                    stage3_rejection_counts[
                        "no_non_percussion_candidate"
                    ] += 1

                    stage3_rejection_details.append(
                        {
                            "md5":
                                candidate["md5"],

                            "path":
                                candidate["path"],

                            "reason":
                                "no_non_percussion_candidate",
                        }
                    )

            # --------------------------------------------------------------
            # CHECKPOINT
            # --------------------------------------------------------------

            if (
                completed_count
                % STAGE3_CHECKPOINT_INTERVAL
                == 0
            ):

                write_stage3_checkpoint(
                    STAGE3_CHECKPOINT,
                    stage3_reports,
                    stage3_survivors,
                    stage3_rejection_counts,
                    stage3_rejection_details,
                    completed_count,
                    len(stage2_candidates),
                    last_md5,
                )


    # =========================================================================
    # STAGE 3 REPORT
    # =========================================================================

    stage3_input_count = (
        len(stage2_candidates)
    )

    stage3_survivor_count = (
        len(stage3_survivors)
    )

    stage3_rejected_count = (
        stage3_input_count
        -
        stage3_survivor_count
    )

    print()
    print("=" * 70)
    print("STAGE 3 COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Input candidates       : "
        f"{stage3_input_count:,}"
    )

    print(
        f"Survivors              : "
        f"{stage3_survivor_count:,}"
    )

    print(
        f"Rejected               : "
        f"{stage3_rejected_count:,}"
    )

    if stage3_input_count:

        print(
            f"Survivor ratio         : "
            f"{stage3_survivor_count / stage3_input_count:.4f}"
        )

    print()
    print(
        "Rejection reasons:"
    )
    print("-" * 70)

    for reason, count in (
        stage3_rejection_counts
        .most_common()
    ):

        print(
            f"{reason:40s}"
            f"{count:10,}"
        )

    # =========================================================================
    # WRITE STAGE 3 PATH MANIFEST
    # =========================================================================

    print()
    print("=" * 70)
    print("Writing Stage 3 survivor manifest...")
    print("=" * 70)
    print()

    with open(
        STAGE3_CANDIDATES_TXT,
        "w",
        encoding="utf-8",
    ) as fh:

        for candidate in stage3_survivors:

            fh.write(
                candidate["path"]
                + "\n"
            )

    print(
        STAGE3_CANDIDATES_TXT
    )

    # =========================================================================
    # WRITE DETAILED REPORTS
    # =========================================================================

    with open(
        STAGE3_REPORTS_JSON,
        "w",
        encoding="utf-8",
    ) as fh:

        json.dump(
            make_json_serializable(
                {
                    "dataset":
                        "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

                    "stage":
                        "3",

                    "description":
                        (
                            "MIDI structural analysis and melody "
                            "candidate ranking. No MIDI files are "
                            "copied, moved, or modified."
                        ),

                    "input_manifest":
                        str(
                            STAGE2_CANDIDATES_TXT
                        ),

                    "excluded_channels":
                        sorted(
                            STAGE3_FILTERS[
                                "excluded_channels"
                            ]
                        ),

                    "reports":
                        stage3_reports,
                }
            ),
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print(
        STAGE3_REPORTS_JSON
    )

    # =========================================================================
    # WRITE SUMMARY
    # =========================================================================

    stage3_summary = {

        "stage":
            "3",

        "description":
            (
                "MIDI musical-structure analysis and "
                "melody-candidate ranking."
            ),

        "input_manifest":
            str(
                STAGE2_CANDIDATES_TXT
            ),

        "output_manifest":
            str(
                STAGE3_CANDIDATES_TXT
            ),

        "input_candidates":
            stage3_input_count,

        "survivors":
            stage3_survivor_count,

        "rejected":
            stage3_rejected_count,

        "survivor_ratio":
            (
                stage3_survivor_count
                / stage3_input_count
                if stage3_input_count
                else 0
            ),

        "excluded_channels":
            sorted(
                STAGE3_FILTERS[
                    "excluded_channels"
                ]
            ),

        "rejection_counts":
            dict(
                stage3_rejection_counts
            ),

        "outputs":
            {
                "candidate_paths":
                    str(
                        STAGE3_CANDIDATES_TXT
                    ),

                "reports":
                    str(
                        STAGE3_REPORTS_JSON
                    ),

                "summary":
                    str(
                        STAGE3_SUMMARY_JSON
                    ),

                "rejections":
                    str(
                        STAGE3_REJECTIONS_JSON
                    ),
            },
    }

    with open(
        STAGE3_SUMMARY_JSON,
        "w",
        encoding="utf-8",
    ) as fh:

        json.dump(
            make_json_serializable(
                stage3_summary
            ),
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print(
        STAGE3_SUMMARY_JSON
    )

    # =========================================================================
    # WRITE REJECTIONS
    # =========================================================================

    with open(
        STAGE3_REJECTIONS_JSON,
        "w",
        encoding="utf-8",
    ) as fh:

        json.dump(
            make_json_serializable(
                stage3_rejection_details
            ),
            fh,
            indent=2,
            ensure_ascii=False,
        )

    print(
        STAGE3_REJECTIONS_JSON
    )

    print()
    print("=" * 70)
    print("STAGE 3 DONE")
    print("=" * 70)
    print()

    print(
        "No MIDI files were copied, moved, or modified."
    )

    print()
    if STAGE3_CHECKPOINT.is_file():

        STAGE3_CHECKPOINT.unlink()

        print()
        print(
            "Stage 3 checkpoint removed after "
            "successful completion."
        )


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

