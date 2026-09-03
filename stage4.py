#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Los Angeles MIDI Dataset
Stage 4 — Melody Selection and MIDI Reconstruction

Stage 3 has already analyzed the MIDI through UglyMIDI.

Stage 4 therefore uses UglyMIDI.instruments[] DIRECTLY.

For every Stage 3 report:

    1. Find candidates satisfying:
           pitch_mean >= 40
           melody_score >= 0.86
           monophonic_fraction >= 0.95

    2. Select the qualifying candidate with the highest melody_score.

    3. Read the ORIGINAL MIDI with Mido.

    4. Read the same MIDI with UglyMIDI.

    5. Stage 3's "track" value is the zero-based index into
       UglyMIDI.instruments[].

    6. Winner:
           UglyMIDI instrument -> output MIDI track 0

       The winner is NEVER channel-filtered.

    7. All other UglyMIDI instruments:
           - discard instruments on EXCLUDED_CHANNELS
           - merge all remaining notes
           - transpose notes below C2 upward by octaves
           - remove duplicate notes
           - optionally quantize
           - write to output MIDI track 1

    8. Output is MIDI Type 1 with exactly two tracks.

    9. Original tempo / time-signature / key-signature events are
       preserved on output track 0.

Checkpoints:

    stage4/checkpoint1.json
    stage4/checkpoint2.json
    ...

Never one growing checkpoint.json.

Checkpoints are deleted only after complete successful processing.
"""

import json
import math
import os
import re
import sys
import tempfile

from concurrent.futures import ProcessPoolExecutor

import mido
from tqdm import tqdm

from pretty_midi_fix import UglyMIDI


# ============================================================================
# PATHS
# ============================================================================

OUTPUT_DIR = "Dataset/LAMDselection/selection_stage1"

STAGE3_DIR = os.path.join(
    OUTPUT_DIR,
    "stage3",
)

STAGE3_REPORTS_JSON = os.path.join(
    STAGE3_DIR,
    "reports.json",
)

STAGE4_DIR = os.path.join(
    OUTPUT_DIR,
    "stage4",
)


# ============================================================================
# CONFIGURATION
# ============================================================================

PITCH_MEAN = 40.0
SCORE_THRESHOLD = 0.86
MONO_THRESHOLD = 0.95

WORKERS = 24

# Number of files submitted to one worker pool at a time.
#
# Keep this at 1 while debugging.
# It can later be increased.
BATCH_SIZE = 1

# Write one independent checkpoint every N completed files.
CHECKPOINT_INTERVAL = 1000

# MIDI channel numbers are ZERO-BASED.
#
# Currently:
#     9  = human MIDI channel 10
#     15 = human MIDI channel 16
#
# Keep these as parameters. We may change them later.
EXCLUDED_CHANNELS = {
    9,
    15,
}

# MIDI C2.
C2 = 36

# Future hook.
QUANTIZE_OUTPUT = False


# ============================================================================
# QUANTIZATION PLACEHOLDER
# ============================================================================

def quantize_output(notes):
    """
    Future output quantization hook.

    Currently intentionally does nothing.
    """
    return notes


# ============================================================================
# JSON HELPERS
# ============================================================================

def atomic_json_write(path, data):
    """
    Atomically write JSON.
    """

    path = os.path.abspath(path)

    parent = os.path.dirname(path)
    os.makedirs(parent, exist_ok=True)

    fd, temp_path = tempfile.mkstemp(
        dir=parent,
        prefix=".stage4_json_",
        suffix=".tmp",
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

            fh.write("\n")

        os.replace(
            temp_path,
            path,
        )

    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

        raise


# ============================================================================
# REPORT LOADING
# ============================================================================

def load_reports():
    """
    Load Stage 3 reports.json.
    """

    if not os.path.isfile(STAGE3_REPORTS_JSON):
        raise FileNotFoundError(
            "Stage 3 reports.json not found:\n"
            f"  {STAGE3_REPORTS_JSON}"
        )

    with open(
        STAGE3_REPORTS_JSON,
        "r",
        encoding="utf-8",
    ) as fh:

        data = json.load(fh)

    reports = data.get("reports")

    if not isinstance(reports, list):
        raise ValueError(
            "reports.json does not contain a valid 'reports' list."
        )

    return reports


# ============================================================================
# STAGE 4 QUALIFICATION
# ============================================================================

def get_qualifying_candidates(report):
    """
    Return all Stage 3 candidates satisfying all Stage 4 thresholds.
    """

    stage3 = report.get(
        "stage3",
        {},
    )

    candidates = stage3.get(
        "candidates",
        [],
    )

    qualifying = []

    for candidate in candidates:

        try:
            pitch_mean = float(
                candidate["pitch_mean"]
            )

            melody_score = float(
                candidate["melody_score"]
            )

            monophonic_fraction = float(
                candidate["monophonic_fraction"]
            )

        except (
            KeyError,
            TypeError,
            ValueError,
        ):
            continue

        if (
            pitch_mean >= PITCH_MEAN
            and
            melody_score >= SCORE_THRESHOLD
            and
            monophonic_fraction >= MONO_THRESHOLD
        ):
            qualifying.append(candidate)

    return qualifying


def get_winner(report):
    """
    Return the qualifying candidate with the highest melody_score.

    Returns None if no candidate qualifies.
    """

    qualifying = get_qualifying_candidates(report)

    if not qualifying:
        return None

    return max(
        qualifying,
        key=lambda candidate: float(
            candidate["melody_score"]
        ),
    )


# ============================================================================
# INPUT / OUTPUT PATH
# ============================================================================

def make_output_path(input_path):
    """
    Preserve:

        MIDIs/<hex>/<filename>.mid

    as:

        stage4/<hex>/<filename>.mid
    """

    absolute_input = os.path.abspath(input_path)

    marker = os.sep + "MIDIs" + os.sep

    position = absolute_input.find(marker)

    if position < 0:
        raise ValueError(
            "Cannot locate '/MIDIs/' in input path:\n"
            f"  {input_path}"
        )

    relative = absolute_input[
        position + len(marker):
    ]

    return os.path.join(
        os.path.abspath(STAGE4_DIR),
        relative,
    )


# ============================================================================
# NOTE HELPERS
# ============================================================================

def transpose_to_c2(pitch):
    """
    Raise pitch by octaves until it is >= C2 (36).
    """

    pitch = int(pitch)

    if pitch >= C2:
        return pitch

    octaves = (
        (C2 - pitch + 11)
        // 12
    )

    return pitch + (12 * octaves)


def remove_duplicate_notes(notes):
    """
    Remove duplicate notes after merging/transposition.

    Identity:

        start_tick
        end_tick
        pitch

    Channel and velocity are NOT part of duplicate identity.
    """

    seen = set()
    result = []

    notes.sort(
        key=lambda note: (
            note["start"],
            note["end"],
            note["pitch"],
        )
    )

    for note in notes:

        key = (
            int(note["start"]),
            int(note["end"]),
            int(note["pitch"]),
        )

        if key in seen:
            continue

        seen.add(key)
        result.append(note)

    return result


# ============================================================================
# PRETTY MIDI TIME -> MIDI TICK
# ============================================================================

def seconds_to_tick(ugly_midi, seconds):
    """
    Convert an UglyMIDI/PrettyMIDI note time back to MIDI ticks.

    UglyMIDI stores notes in seconds.
    Stage 4 output is written in MIDI ticks.
    """

    return int(
        ugly_midi.time_to_tick(
            float(seconds)
        )
    )


# ============================================================================
# MIDI METADATA
# ============================================================================

def collect_metadata_events(original_midi):
    """
    Collect tempo / key / time-signature events from the original MIDI.

    UglyMIDI may move these events to physical track 0, but for Stage 4
    we simply collect them directly from the original Mido representation.

    Returned events use absolute MIDI ticks.
    """

    events = []

    for track_index, track in enumerate(
        original_midi.tracks
    ):

        absolute_tick = 0

        for event in track:

            absolute_tick += int(
                event.time
            )

            if event.type in {
                "set_tempo",
                "time_signature",
                "key_signature",
            }:

                events.append(
                    (
                        absolute_tick,
                        track_index,
                        event.copy(time=0),
                    )
                )

    # Stable deterministic ordering.
    events.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return events


def append_metadata_events(track, metadata_events):
    """
    Append metadata events using delta times.
    """

    previous_tick = 0

    for (
        absolute_tick,
        _source_track,
        event,
    ) in metadata_events:

        delta = (
            absolute_tick
            - previous_tick
        )

        event.time = delta

        track.append(event)

        previous_tick = absolute_tick


# ============================================================================
# NOTE EVENT WRITING
# ============================================================================

def append_note_events(
    track,
    notes,
):
    """
    Write absolute-tick note dictionaries as delta-time Mido events.

    Output notes are placed on channel 0.
    """

    events = []

    for note in notes:

        start = int(note["start"])
        end = int(note["end"])

        pitch = int(note["pitch"])
        velocity = int(note["velocity"])

        events.append(
            (
                start,
                1,
                mido.Message(
                    "note_on",
                    channel=0,
                    note=pitch,
                    velocity=velocity,
                    time=0,
                ),
            )
        )

        events.append(
            (
                end,
                0,
                mido.Message(
                    "note_off",
                    channel=0,
                    note=pitch,
                    velocity=0,
                    time=0,
                ),
            )
        )

    # Note-off before note-on at the same tick.
    events.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    previous_tick = 0

    for (
        absolute_tick,
        _order,
        message,
    ) in events:

        delta = (
            absolute_tick
            - previous_tick
        )

        message.time = delta

        track.append(message)

        previous_tick = absolute_tick


# ============================================================================
# UGLYMIDI -> STAGE 4 NOTES
# ============================================================================

def extract_instrument_notes(
    ugly_midi,
    instrument,
):
    """
    Convert one UglyMIDI Instrument into Stage 4 tick-based notes.

    IMPORTANT:

    We do NOT attempt to rediscover the instrument.

    UglyMIDI has already done that.
    """

    notes = []

    for ugly_note in instrument.notes:

        start_tick = seconds_to_tick(
            ugly_midi,
            ugly_note.start,
        )

        end_tick = seconds_to_tick(
            ugly_midi,
            ugly_note.end,
        )

        if end_tick <= start_tick:
            continue

        notes.append(
            {
                "start": start_tick,
                "end": end_tick,
                "pitch": int(
                    ugly_note.pitch
                ),
                "velocity": int(
                    ugly_note.velocity
                ),
            }
        )

    return notes


# ============================================================================
# OUTPUT MIDI
# ============================================================================

def build_output_midi(
    original_midi,
    ugly_midi,
    winning_track_index,
):
    """
    Build the Stage 4 Type-1 output MIDI.

    Track 0 = winning UglyMIDI instrument.

    Track 1 = all other UglyMIDI instruments after:

        channel filtering
        octave transposition
        duplicate removal
        optional quantization
    """

    instruments = ugly_midi.instruments

    if (
        winning_track_index < 0
        or
        winning_track_index >= len(instruments)
    ):
        raise IndexError(
            "Stage 3 winning instrument index "
            f"{winning_track_index} is outside "
            f"UglyMIDI.instruments[] range "
            f"0..{len(instruments) - 1}"
        )

    # ------------------------------------------------------------------
    # Output MIDI.
    # ------------------------------------------------------------------

    output_midi = mido.MidiFile(
        type=1,
        ticks_per_beat=original_midi.ticks_per_beat,
    )

    melody_track = mido.MidiTrack()
    other_track = mido.MidiTrack()

    output_midi.tracks.append(
        melody_track
    )

    output_midi.tracks.append(
        other_track
    )

    # ------------------------------------------------------------------
    # Original metadata goes to output track 0.
    # ------------------------------------------------------------------

    metadata_events = collect_metadata_events(
        original_midi
    )

    append_metadata_events(
        melody_track,
        metadata_events,
    )

    # ------------------------------------------------------------------
    # TRACK 0 — WINNER
    # ------------------------------------------------------------------

    winning_instrument = instruments[
        winning_track_index
    ]

    melody_track.append(
        mido.MetaMessage(
            "track_name",
            name="Stage4 Melody",
            time=0,
        )
    )

    # PrettyMIDI uses General MIDI program numbers 0..127.
    winning_program = int(
        winning_instrument.program
    )

    melody_track.append(
        mido.Message(
            "program_change",
            channel=0,
            program=winning_program,
            time=0,
        )
    )

    winning_notes = (
        extract_instrument_notes(
            ugly_midi,
            winning_instrument,
        )
    )

    append_note_events(
        melody_track,
        winning_notes,
    )

    # ------------------------------------------------------------------
    # TRACK 1 — EVERYTHING ELSE
    # ------------------------------------------------------------------

    merged_notes = []

    for instrument_index, instrument in enumerate(
        instruments
    ):

        if instrument_index == winning_track_index:
            continue

        channel = int(
            instrument.channel
        )

        # --------------------------------------------------------------
        # Channel filtering.
        #
        # This is done at the UglyMIDI Instrument level.
        #
        # Every note belonging to this instrument therefore gets
        # discarded together.
        # --------------------------------------------------------------

        if channel in EXCLUDED_CHANNELS:
            continue

        instrument_notes = (
            extract_instrument_notes(
                ugly_midi,
                instrument,
            )
        )

        for note in instrument_notes:

            note["pitch"] = transpose_to_c2(
                note["pitch"]
            )

            merged_notes.append(note)

    # ------------------------------------------------------------------
    # Duplicate removal AFTER merging/transposition.
    # ------------------------------------------------------------------

    merged_notes = remove_duplicate_notes(
        merged_notes
    )

    # ------------------------------------------------------------------
    # Optional quantization.
    # ------------------------------------------------------------------

    if QUANTIZE_OUTPUT:

        merged_notes = quantize_output(
            merged_notes
        )

    other_track.append(
        mido.MetaMessage(
            "track_name",
            name="Stage4 Other",
            time=0,
        )
    )

    append_note_events(
        other_track,
        merged_notes,
    )

    # ------------------------------------------------------------------
    # End markers.
    # ------------------------------------------------------------------

    melody_track.append(
        mido.MetaMessage(
            "end_of_track",
            time=0,
        )
    )

    other_track.append(
        mido.MetaMessage(
            "end_of_track",
            time=0,
        )
    )

    return output_midi


# ============================================================================
# ATOMIC MIDI WRITING
# ============================================================================

def write_midi_atomic(
    midi,
    output_path,
):
    """
    Atomically write a MIDI file.
    """

    output_path = os.path.abspath(
        output_path
    )

    parent = os.path.dirname(
        output_path
    )

    os.makedirs(
        parent,
        exist_ok=True,
    )

    fd, temp_path = tempfile.mkstemp(
        dir=parent,
        prefix=".stage4_midi_",
        suffix=".mid",
    )

    os.close(fd)

    try:

        midi.save(
            filename=temp_path
        )

        os.replace(
            temp_path,
            output_path,
        )

    except Exception:

        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass

        raise


# ============================================================================
# ONE FILE
# ============================================================================

def process_one(report):
    """
    Process one Stage 3 report.
    """

    input_path = report.get("path")

    if not input_path:
        return {
            "status": "error",
            "reason": "missing_input_path",
        }

    winner = get_winner(report)

    if winner is None:
        return {
            "status": "discarded",
            "input_path": input_path,
        }

    winning_track_index = int(
        winner["track"]
    )

    try:

        # --------------------------------------------------------------
        # Read original MIDI.
        # --------------------------------------------------------------

        original_midi = mido.MidiFile(
            filename=input_path,
            clip=True,
        )

        # --------------------------------------------------------------
        # Stage 4 requires Type 1 input.
        # --------------------------------------------------------------

        if original_midi.type != 1:
            raise ValueError(
                "Stage 4 requires MIDI Type 1 input; "
                f"file is Type {original_midi.type}"
            )

        # --------------------------------------------------------------
        # THIS IS THE IMPORTANT PART.
        #
        # UglyMIDI performs exactly the logical-instrument separation
        # used by Stage 3.
        # --------------------------------------------------------------

        ugly_midi = UglyMIDI(
            input_path
        )

        # --------------------------------------------------------------
        # Verify that Stage 3's logical index exists in the actual
        # UglyMIDI instrument list.
        # --------------------------------------------------------------

        instrument_count = len(
            ugly_midi.instruments
        )

        if (
            winning_track_index < 0
            or
            winning_track_index >= instrument_count
        ):
            raise IndexError(
                "Stage 3 winning instrument index "
                f"{winning_track_index} does not exist in "
                f"UglyMIDI.instruments[] "
                f"(count={instrument_count})"
            )

        winning_instrument = ugly_midi.instruments[
            winning_track_index
        ]

        # --------------------------------------------------------------
        # Build output.
        # --------------------------------------------------------------

        output_midi = build_output_midi(
            original_midi=original_midi,
            ugly_midi=ugly_midi,
            winning_track_index=winning_track_index,
        )

        # --------------------------------------------------------------
        # Output path.
        # --------------------------------------------------------------

        output_path = make_output_path(
            input_path
        )

        # --------------------------------------------------------------
        # Write.
        # --------------------------------------------------------------

        write_midi_atomic(
            output_midi,
            output_path,
        )

        return {
            "status": "retained",
            "input_path": input_path,
            "output_path": output_path,
            "winning_track": winning_track_index,
            "winning_channel": int(
                winning_instrument.channel
            ),
            "winning_program": int(
                winning_instrument.program
            ),
            "ugly_instruments": instrument_count,
            "melody_score": float(
                winner["melody_score"]
            ),
            "pitch_mean": float(
                winner["pitch_mean"]
            ),
            "monophonic_fraction": float(
                winner["monophonic_fraction"]
            ),
        }

    except Exception as exc:

        return {
            "status": "error",
            "input_path": input_path,
            "reason": type(exc).__name__,
            "message": str(exc),
        }

    finally:

        try:
            del ugly_midi
        except UnboundLocalError:
            pass

        try:
            del original_midi
        except UnboundLocalError:
            pass

        try:
            del output_midi
        except UnboundLocalError:
            pass


# ============================================================================
# CHECKPOINT MANAGEMENT
# ============================================================================

CHECKPOINT_PATTERN = re.compile(
    r"^checkpoint([0-9]+)\.json$"
)


def checkpoint_path(
    checkpoint_number,
):
    return os.path.join(
        STAGE4_DIR,
        f"checkpoint{checkpoint_number}.json",
    )


def find_latest_checkpoint():
    """
    Return:

        (number, path)

    or:

        (0, None)
    """

    if not os.path.isdir(STAGE4_DIR):
        return 0, None

    candidates = []

    for filename in os.listdir(
        STAGE4_DIR
    ):

        match = CHECKPOINT_PATTERN.match(
            filename
        )

        if not match:
            continue

        number = int(
            match.group(1)
        )

        candidates.append(
            (
                number,
                os.path.join(
                    STAGE4_DIR,
                    filename,
                ),
            )
        )

    if not candidates:
        return 0, None

    return max(
        candidates,
        key=lambda item: item[0],
    )


def write_checkpoint(
    checkpoint_number,
    completed_count,
    total_count,
    retained_count,
    discarded_count,
    error_count,
    last_path,
):
    """
    Write one completely independent checkpoint file.
    """

    checkpoint = {
        "stage": "4",
        "checkpoint_number": checkpoint_number,
        "completed_count": completed_count,
        "total_count": total_count,
        "retained_count": retained_count,
        "discarded_count": discarded_count,
        "error_count": error_count,
        "last_path": last_path,

        "configuration": {
            "pitch_mean": PITCH_MEAN,
            "score_threshold": SCORE_THRESHOLD,
            "monophonic_threshold": MONO_THRESHOLD,
            "workers": WORKERS,
            "batch_size": BATCH_SIZE,
            "checkpoint_interval": CHECKPOINT_INTERVAL,
            "excluded_channels": sorted(
                EXCLUDED_CHANNELS
            ),
            "c2": C2,
            "quantize_output": QUANTIZE_OUTPUT,
        },
    }

    atomic_json_write(
        checkpoint_path(
            checkpoint_number
        ),
        checkpoint,
    )


def load_latest_checkpoint():
    """
    Load the newest checkpoint.

    Returns None if none exists.
    """

    number, path = find_latest_checkpoint()

    if path is None:
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as fh:

        checkpoint = json.load(fh)

    checkpoint["_checkpoint_number"] = number

    return checkpoint


def delete_all_checkpoints():
    """
    Delete every checkpoint<number>.json.
    """

    if not os.path.isdir(STAGE4_DIR):
        return

    deleted = 0

    for filename in os.listdir(
        STAGE4_DIR
    ):

        if not CHECKPOINT_PATTERN.match(
            filename
        ):
            continue

        path = os.path.join(
            STAGE4_DIR,
            filename,
        )

        try:
            os.unlink(path)
            deleted += 1

        except FileNotFoundError:
            pass

    print(
        f"Deleted {deleted} checkpoint file(s)."
    )


# ============================================================================
# WORK LIST
# ============================================================================

def build_work_list(reports):
    """
    Determine which reports have a Stage 4 winner.

    No MIDI files are opened here.
    """

    eligible = []
    threshold_rejected = 0

    for report in reports:

        if not report.get("path"):
            continue

        winner = get_winner(report)

        if winner is None:
            threshold_rejected += 1
            continue

        eligible.append(report)

    return (
        eligible,
        threshold_rejected,
    )


# ============================================================================
# MAIN
# ============================================================================

def main():

    print()
    print("=" * 70)
    print("Los Angeles MIDI Dataset")
    print("STAGE 4 — MELODY SELECTION / MIDI RECONSTRUCTION")
    print("=" * 70)
    print()

    os.makedirs(
        STAGE4_DIR,
        exist_ok=True,
    )

    print(
        f"Stage 3 reports : {STAGE3_REPORTS_JSON}"
    )

    print(
        f"Stage 4 output  : {STAGE4_DIR}"
    )

    print(
        f"Workers         : {WORKERS}"
    )

    print(
        f"Batch size      : {BATCH_SIZE}"
    )

    print(
        f"Checkpoint every: {CHECKPOINT_INTERVAL:,} files"
    )

    print(
        f"Excluded channels: {sorted(EXCLUDED_CHANNELS)}"
    )

    print()

    # ------------------------------------------------------------------
    # Load reports.
    # ------------------------------------------------------------------

    reports = load_reports()

    print(
        f"Reports loaded  : {len(reports):,}"
    )

    print()

    # ------------------------------------------------------------------
    # Stage 4 qualification.
    # ------------------------------------------------------------------

    eligible_reports, threshold_rejected = (
        build_work_list(reports)
    )

    print(
        f"Threshold survivors : {len(eligible_reports):,}"
    )

    print(
        f"Threshold rejected  : {threshold_rejected:,}"
    )

    print()

    # ------------------------------------------------------------------
    # Resume.
    # ------------------------------------------------------------------

    checkpoint = load_latest_checkpoint()

    if checkpoint is not None:

        completed_count = int(
            checkpoint["completed_count"]
        )

        retained_count = int(
            checkpoint.get(
                "retained_count",
                0,
            )
        )

        discarded_count = int(
            checkpoint.get(
                "discarded_count",
                0,
            )
        )

        error_count = int(
            checkpoint.get(
                "error_count",
                0,
            )
        )

        checkpoint_number = int(
            checkpoint["_checkpoint_number"]
        )

        last_path = checkpoint.get(
            "last_path"
        )

        checkpoint_total = int(
            checkpoint["total_count"]
        )

        if checkpoint_total != len(
            eligible_reports
        ):
            raise RuntimeError(
                "Stage 4 cannot safely resume because "
                "the number of eligible reports changed.\n\n"
                f"Checkpoint: {checkpoint_total:,}\n"
                f"Current   : {len(eligible_reports):,}\n\n"
                "Do not modify reports.json while Stage 4 is running."
            )

        print("=" * 70)
        print("RESUMING STAGE 4")
        print("=" * 70)

        print(
            f"Checkpoint       : "
            f"checkpoint{checkpoint_number}.json"
        )

        print(
            f"Already completed: {completed_count:,}"
        )

        print(
            f"Last path        : {last_path}"
        )

        print()

    else:

        completed_count = 0
        retained_count = 0
        discarded_count = 0
        error_count = 0
        checkpoint_number = 0

    # ------------------------------------------------------------------
    # Check checkpoint sanity.
    # ------------------------------------------------------------------

    if (
        completed_count < 0
        or
        completed_count > len(
            eligible_reports
        )
    ):
        raise RuntimeError(
            "Invalid completed_count in checkpoint."
        )

    remaining = eligible_reports[
        completed_count:
    ]

    # ------------------------------------------------------------------
    # Process bounded batches.
    # ------------------------------------------------------------------

    for batch_offset in range(
        0,
        len(remaining),
        BATCH_SIZE,
    ):

        batch = remaining[
            batch_offset:
            batch_offset + BATCH_SIZE
        ]

        absolute_start = completed_count

        absolute_end = (
            completed_count
            + len(batch)
        )

        print("=" * 70)

        print(
            f"STAGE 4 BATCH "
            f"{absolute_start + 1:,} - "
            f"{absolute_end:,} "
            f"of "
            f"{len(eligible_reports):,}"
        )

        print("=" * 70)

        # --------------------------------------------------------------
        # Fresh process pool for each bounded batch.
        # --------------------------------------------------------------

        with ProcessPoolExecutor(
            max_workers=WORKERS
        ) as executor:

            results = executor.map(
                process_one,
                batch,
            )

            for result in tqdm(
                results,
                total=len(batch),
                desc="Stage 4",
            ):

                status = result.get(
                    "status"
                )

                completed_count += 1

                if status == "retained":

                    retained_count += 1

                elif status == "discarded":

                    discarded_count += 1

                else:

                    error_count += 1

                    print()
                    print(
                        "ERROR processing:"
                    )

                    print(
                        f"  {result.get('input_path')}"
                    )

                    print(
                        f"  {result.get('reason')}: "
                        f"{result.get('message')}"
                    )

                # ------------------------------------------------------
                # Independent checkpoint.
                # ------------------------------------------------------

                if (
                    completed_count
                    % CHECKPOINT_INTERVAL
                    == 0
                ):

                    checkpoint_number += 1

                    write_checkpoint(
                        checkpoint_number=
                            checkpoint_number,

                        completed_count=
                            completed_count,

                        total_count=
                            len(eligible_reports),

                        retained_count=
                            retained_count,

                        discarded_count=
                            discarded_count,

                        error_count=
                            error_count,

                        last_path=
                            result.get(
                                "input_path"
                            ),
                    )

                    print()

                    print(
                        "Checkpoint written: "
                        f"checkpoint{checkpoint_number}.json"
                    )

        del batch

    # ------------------------------------------------------------------
    # Completion.
    # ------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 4 COMPLETE")
    print("=" * 70)
    print()

    print(
        f"Stage 4 eligible input : "
        f"{len(eligible_reports):,}"
    )

    print(
        f"Threshold rejected     : "
        f"{threshold_rejected:,}"
    )

    print(
        f"Retained / written     : "
        f"{retained_count:,}"
    )

    print(
        f"Discarded during work  : "
        f"{discarded_count:,}"
    )

    print(
        f"Processing errors      : "
        f"{error_count:,}"
    )

    print()

    if error_count > 0:

        print("WARNING:")
        print("Processing errors occurred.")
        print("Checkpoint files have NOT been deleted.")
        print("Fix the problem and resume Stage 4.")
        print()

        return 1

    # ------------------------------------------------------------------
    # Only a completely successful run deletes checkpoints.
    # ------------------------------------------------------------------

    delete_all_checkpoints()

    print()
    print(
        "Stage 4 finished successfully."
    )

    print(
        "Output directory:"
    )

    print(
        f"  {STAGE4_DIR}"
    )

    print()

    return 0


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    sys.exit(main())