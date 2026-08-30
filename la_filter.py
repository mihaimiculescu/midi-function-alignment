
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

import MIDI

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
    Read one MIDI file and determine whether its first actual MIDI track
    contains at least one note event.

    MIDI score structure:

        score[0] = ticks
        score[1] = first actual MIDI track
        score[2] = second actual MIDI track
        ...

    Returns:

        (True, None)

    or:

        (False, reason)
    """

    try:

        # --------------------------------------------------------------
        # Read MIDI
        # --------------------------------------------------------------

        opus = MIDI.midi2opus(
            str(midi_path)
        )

        # --------------------------------------------------------------
        # Basic opus sanity check
        # --------------------------------------------------------------

        if not isinstance(opus, (list, tuple)):
            return False, "invalid_opus"

        if len(opus) < 2:
            return False, "no_midi_tracks"

        # --------------------------------------------------------------
        # Convert to SCORE representation.
        #
        # score[0] = ticks
        # score[1] = first actual MIDI track
        # --------------------------------------------------------------

        score = MIDI.opus2score(opus)

        if not isinstance(score, (list, tuple)):
            return False, "invalid_score"

        if len(score) < 2:
            return False, "no_midi_tracks"

        first_track = score[1]

        if not isinstance(first_track, (list, tuple)):
            return False, "first_track_invalid"

        # --------------------------------------------------------------
        # Look for actual note events.
        # --------------------------------------------------------------

        for event in first_track:

            if not isinstance(event, (list, tuple)):
                continue

            if len(event) == 0:
                continue

            if event[0] == "note":
                return True, None

        return False, "first_track_has_no_notes"

    except Exception as exc:

        return False, (
            "midi_parse_error:"
            + type(exc).__name__
        )


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
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()

