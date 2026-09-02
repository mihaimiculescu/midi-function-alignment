
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Los Angeles MIDI Dataset
Stage 4 — Melody Selection and MIDI Reconstruction

INPUT
-----
Dataset/LAMDselection/selection_stage1/stage3/reports.json

For every MIDI file represented in reports.json:

    pitch_mean >= 40
    melody_score >= 0.86
    monophonic_fraction >= 0.95

A file with no qualifying candidate is discarded.

For retained files:

    1. Select the qualifying candidate with the highest melody_score.
    2. Read the ORIGINAL MIDI using Mido.
    3. Reconstruct the same logical-track indexing used by Stage 3.
    4. Write the winning logical track to output track 0.
    5. For all other logical tracks:
         - discard notes whose actual MIDI channel is 9 or 15
         - merge remaining notes
         - transpose notes below C2 upward by octaves
         - remove duplicates
         - optionally quantize
         - write to output track 1
    6. Preserve the hexadecimal directory structure.

OUTPUT
------
Dataset/LAMDselection/selection_stage1/stage4/

Example:

    MIDIs/5/5cc6b240f8acbd56ab93decd8993ed96.mid

becomes:

    stage4/5/5cc6b240f8acbd56ab93decd8993ed96.mid

CHECKPOINTS
-----------
Checkpoint files are written under:

    stage4/checkpoint1.json
    stage4/checkpoint2.json
    ...

They are NEVER appended to one growing checkpoint.json.

After successful completion all checkpoint<number>.json files are
deleted.

If the process is interrupted, the latest checkpoint remains and
the next run resumes from that checkpoint.

PARALLELISM
-----------
24 worker processes.

Processing is bounded in batches to avoid accumulating a huge number
of pending process jobs/results in memory.
"""

import json
import math
import os
import re
import sys
import tempfile
from collections import OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor

import mido
from tqdm import tqdm
from dataclasses import dataclass

# ============================================================================
# PATHS
# ============================================================================

OUTPUT_DIR = (
    "Dataset/LAMDselection/selection_stage1"
)

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

# Number of files submitted to the worker pool at one time.
BATCH_SIZE = 1

# Write one checkpoint every N completed files.
CHECKPOINT_INTERVAL = 1000

# MIDI channel numbers are zero-based.
#
# GM channel 10 == MIDI channel 9
# MIDI channel 16 == MIDI channel 15
EXCLUDED_CHANNELS = {
    9,
    15,
}

# MIDI C2 = note number 36.
C2 = 36

# Requested by specification.
QUANTIZE_OUTPUT = False


# ============================================================================
# QUANTIZATION PLACEHOLDER
# ============================================================================

def quantize_output(notes):
    """
    Placeholder for future output quantization.

    This intentionally does NOTHING.

    It is called AFTER duplicate removal and BEFORE writing track 1.
    """

    return notes


# ============================================================================
# JSON HELPERS
# ============================================================================

def atomic_json_write(
    path,
    data,
):
    """
    Atomically write JSON.
    """

    path = os.path.abspath(path)

    parent = os.path.dirname(path)

    os.makedirs(
        parent,
        exist_ok=True,
    )

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

    if not os.path.isfile(
        STAGE3_REPORTS_JSON
    ):
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

    reports = data.get(
        "reports"
    )

    if not isinstance(
        reports,
        list,
    ):
        raise ValueError(
            "reports.json does not contain "
            "a valid 'reports' list."
        )

    return reports


# ============================================================================
# STAGE 4 QUALIFICATION
# ============================================================================

def get_qualifying_candidates(
    report,
):
    """
    Return all Stage 3 candidates satisfying ALL Stage 4 conditions.
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
            qualifying.append(
                candidate
            )

    return qualifying


def get_winner(
    report,
):
    """
    Return the highest-melody_score candidate among the qualifying
    candidates.

    Returns None when no candidate qualifies.
    """

    qualifying = (
        get_qualifying_candidates(
            report
        )
    )

    if not qualifying:
        return None

    return max(
        qualifying,
        key=lambda candidate:
            float(
                candidate["melody_score"]
            ),
    )


# ============================================================================
# INPUT / OUTPUT PATH
# ============================================================================

def make_output_path(
    input_path,
):
    """
    Preserve the original MIDIs/<hex>/filename structure.

    Example:

        .../MIDIs/5/file.mid

    becomes:

        .../stage4/5/file.mid
    """

    absolute_input = os.path.abspath(
        input_path
    )

    marker = os.sep + "MIDIs" + os.sep

    position = absolute_input.find(
        marker
    )

    if position < 0:

        raise ValueError(
            "Cannot locate '/MIDIs/' in input path:\n"
            f"  {input_path}"
        )

    relative = absolute_input[
        position + len(marker):
    ]

    return os.path.join(
        os.path.abspath(
            STAGE4_DIR
        ),
        relative,
    )


# ============================================================================
# MIDI NOTE REPRESENTATION
# ============================================================================

class LogicalTrack:
    __slots__ = (
        "source_track",
        "channel",
        "program",
        "notes",
    )

    def __init__(
        self,
        source_track,
        channel,
        program,
        notes,
    ):
        self.source_track = int(source_track)
        self.channel = int(channel)
        self.program = int(program)
        self.notes = notes

class NoteRecord:
    """
    Lightweight MIDI note representation.

    All timing is kept in MIDI ticks.

    Fields:

        source_track
        channel
        program
        pitch
        velocity
        start
        end
    """

    __slots__ = (
        "source_track",
        "channel",
        "program",
        "pitch",
        "velocity",
        "start",
        "end",
    )

    def __init__(
        self,
        source_track,
        channel,
        program,
        pitch,
        velocity,
        start,
        end,
    ):

        self.source_track = int(
            source_track
        )

        self.channel = int(
            channel
        )

        self.program = int(
            program
        )

        self.pitch = int(
            pitch
        )

        self.velocity = int(
            velocity
        )

        self.start = int(
            start
        )

        self.end = int(
            end
        )


# ============================================================================
# LOGICAL TRACK RECONSTRUCTION
# ============================================================================

def parse_midi_logical_tracks(midi):
    """
    Reconstruct the logical instrument list produced by UglyMIDI.

    Stage 3 stores the index of UglyMIDI.instruments[] as "track".

    UglyMIDI logical identity is:

        (program, channel, physical_track)

    Instrument insertion order is the order in which a logical
    instrument is first created.
    """

    instrument_map = OrderedDict()

    # Controller/pitch events occurring before an instrument has
    # a note are stored as stragglers by UglyMIDI.
    #
    # They do NOT create entries in instrument_map.
    stragglers = {}

    for track_idx, track in enumerate(midi.tracks):

        # EXACTLY as UglyMIDI:
        # program 0 for every channel at the start of every
        # physical MIDI track.
        current_instrument = [0] * 16

        # EXACTLY as UglyMIDI:
        # reset open-note state for every physical track.
        last_note_on = defaultdict(list)

        # Mido gives us delta times.
        # UglyMIDI converts them to absolute ticks before
        # _load_instruments().
        absolute_tick = 0

        for event in track:

            absolute_tick += int(event.time)

            # ==========================================================
            # TRACK NAME
            #
            # Doesn't affect logical identity, so nothing to store.
            # ==========================================================

            if event.type == "track_name":
                continue

            # ==========================================================
            # PROGRAM CHANGE
            # ==========================================================

            if event.type == "program_change":

                current_instrument[event.channel] = (
                    event.program
                )

                continue

            # ==========================================================
            # NOTE ON
            # ==========================================================

            elif (
                event.type == "note_on"
                and event.velocity > 0
            ):

                note_on_index = (
                    event.channel,
                    event.note,
                )

                last_note_on[note_on_index].append(
                    (
                        absolute_tick,
                        event.velocity,
                    )
                )

            # ==========================================================
            # NOTE OFF
            #
            # Also handles NOTE ON velocity=0.
            # ==========================================================

            elif (
                event.type == "note_off"
                or (
                    event.type == "note_on"
                    and event.velocity == 0
                )
            ):

                key = (
                    event.channel,
                    event.note,
                )

                # UglyMIDI ignores spurious note-offs.
                if key not in last_note_on:
                    continue

                end_tick = absolute_tick

                open_notes = last_note_on[key]

                # IMPORTANT:
                # This is copied from UglyMIDI:
                #
                #     start_tick != end_tick
                #
                # NOT "< end_tick".
                notes_to_close = [
                    (start_tick, velocity)
                    for start_tick, velocity
                    in open_notes
                    if start_tick != end_tick
                ]

                notes_to_keep = [
                    (start_tick, velocity)
                    for start_tick, velocity
                    in open_notes
                    if start_tick == end_tick
                ]

                for start_tick, velocity in notes_to_close:

                    # UglyMIDI determines the program at NOTE OFF,
                    # not NOTE ON.
                    program = current_instrument[
                        event.channel
                    ]

                    logical_key = (
                        int(program),
                        int(event.channel),
                        int(track_idx),
                    )

                    # --------------------------------------------------
                    # This corresponds directly to:
                    #
                    # __get_instrument(
                    #     program,
                    #     event.channel,
                    #     track_idx,
                    #     1
                    # )
                    # --------------------------------------------------

                    if logical_key not in instrument_map:

                        instrument_map[logical_key] = LogicalTrack(
                            source_track=track_idx,
                            channel=event.channel,
                            program=program,
                            notes=[],
                        )

                        # UglyMIDI transfers any straggler state
                        # into the newly created instrument.
                        stragglers.pop(
                            (
                                event.channel,
                                track_idx,
                            ),
                            None,
                        )

                    instrument = instrument_map[
                        logical_key
                    ]

                    instrument.notes.append(
                        NoteRecord(
                            source_track=track_idx,
                            channel=event.channel,
                            program=program,
                            pitch=event.note,
                            velocity=velocity,
                            start=start_tick,
                            end=end_tick,
                        )
                    )

                # ------------------------------------------------------
                # Exactly reproduce UglyMIDI's handling of notes
                # which started at the same tick as this note-off.
                # ------------------------------------------------------

                if (
                    len(notes_to_close) > 0
                    and len(notes_to_keep) > 0
                ):

                    last_note_on[key] = notes_to_keep

                else:

                    del last_note_on[key]

            # ==========================================================
            # PITCH WHEEL
            #
            # create_new=0
            #
            # This can create a STRAGGLER but cannot create a logical
            # instrument index.
            # ==========================================================

            elif event.type == "pitchwheel":

                program = current_instrument[
                    event.channel
                ]

                logical_key = (
                    int(program),
                    int(event.channel),
                    int(track_idx),
                )

                if logical_key not in instrument_map:

                    straggler_key = (
                        event.channel,
                        track_idx,
                    )

                    if straggler_key not in stragglers:

                        stragglers[
                            straggler_key
                        ] = True

            # ==========================================================
            # CONTROL CHANGE
            #
            # Same semantics as pitchwheel.
            # ==========================================================

            elif event.type == "control_change":

                program = current_instrument[
                    event.channel
                ]

                logical_key = (
                    int(program),
                    int(event.channel),
                    int(track_idx),
                )

                if logical_key not in instrument_map:

                    straggler_key = (
                        event.channel,
                        track_idx,
                    )

                    if straggler_key not in stragglers:

                        stragglers[
                            straggler_key
                        ] = True

    # IMPORTANT:
    #
    # UglyMIDI does:
    #
    #     self.instruments = [i for i in instrument_map.values()]
    #
    # Therefore OrderedDict insertion order is the Stage 3
    # logical track index.
    return {
        "logical_tracks": list(
            instrument_map.values()
        ),
        "ticks_per_beat": midi.ticks_per_beat,
    }


# ============================================================================
# OUTPUT NOTE HELPERS
# ============================================================================

def transpose_to_c2(
    pitch,
):
    """
    Raise a pitch by octaves until it is >= C2.

    MIDI C2 = 36.
    """

    pitch = int(
        pitch
    )

    if pitch >= C2:
        return pitch

    octaves = (
        (C2 - pitch + 11)
        // 12
    )

    return pitch + (
        12 * octaves
    )


def remove_duplicate_notes(
    notes,
):
    """
    Remove duplicates after merging/transposition.

    A duplicate note is defined by:

        start tick
        end tick
        pitch

    Channel and velocity are deliberately NOT part of the identity.

    This is important because once all surviving material is merged
    into track 1, two notes at the same pitch and same time are the
    same musical note even if they came from different source
    channels or had different velocities.

    The strongest/first surviving velocity is retained.
    """

    seen = set()

    result = []

    # Deterministic ordering.
    notes.sort(
        key=lambda note: (
            note.start,
            note.end,
            note.pitch,
            note.channel,
        )
    )

    for note in notes:

        key = (
            int(note.start),
            int(note.end),
            int(note.pitch),
        )

        if key in seen:
            continue

        seen.add(
            key
        )

        result.append(
            note
        )

    return result


# ============================================================================
# OUTPUT MIDI CREATION
# ============================================================================

def add_note_events(
    track,
    notes,
):
    """
    Add absolute-tick NoteRecords to a Mido track.

    Output events are generated in absolute time and then converted
    to delta times.

    Notes are written on channel 0 because each output track is a
    reconstructed musical track rather than a preservation of the
    original multi-channel arrangement.

    The winning track retains its program through a program_change.
    """

    events = []

    for note in notes:

        events.append(
            (
                int(note.start),
                0,
                mido.Message(
                    "note_on",
                    channel=0,
                    note=int(note.pitch),
                    velocity=int(note.velocity),
                    time=0,
                ),
            )
        )

        events.append(
            (
                int(note.end),
                1,
                mido.Message(
                    "note_off",
                    channel=0,
                    note=int(note.pitch),
                    velocity=0,
                    time=0,
                ),
            )
        )

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

        track.append(
            message
        )

        previous_tick = (
            absolute_tick
        )


def build_output_midi(
    parsed,
    winning_track_index,
):
    """
    Construct a new two-track Mido MIDI file.

    Track 0:
        winning logical track.

    Track 1:
        all other logical tracks after channel filtering,
        transposition and duplicate removal.
    """

    logical_tracks = parsed[
        "logical_tracks"
    ]

    if (
        winning_track_index < 0
        or
        winning_track_index >= len(
            logical_tracks
        )
    ):

        raise IndexError(
            "Stage 3 winning track index "
            f"{winning_track_index} is outside "
            f"the reconstructed logical-track range "
            f"0..{len(logical_tracks) - 1}"
        )

    midi = mido.MidiFile(
        type=1,
        ticks_per_beat=
            parsed["ticks_per_beat"],
    )

    # ==================================================================
    # OUTPUT TRACK 0 — WINNING MELODY
    # ==================================================================

    winning = logical_tracks[
        winning_track_index
    ]

    melody_track = mido.MidiTrack()

    midi.tracks.append(
        melody_track
    )

    # The winner is intentionally exempt from the channel 9/15
    # filtering. This is the explicit Stage 4 specification.
    #
    # Use a single output channel (0), but preserve the winning
    # program number.
    melody_track.append(
        mido.MetaMessage(
            "track_name",
            name="Stage4 Melody",
            time=0,
        )
    )

    melody_track.append(
        mido.Message(
            "program_change",
            channel=0,
            program=int(
                winning["program"]
            ),
            time=0,
        )
    )

    winning_notes = sorted(
        winning["notes"],
        key=lambda note: (
            note.start,
            note.end,
            note.pitch,
        )
    )

    add_note_events(
        melody_track,
        winning_notes,
    )

    # ==================================================================
    # OUTPUT TRACK 1 — EVERYTHING ELSE
    # ==================================================================

    merged_notes = []

    for logical_index, logical in enumerate(
        logical_tracks
    ):

        if (
            logical_index
            == winning_track_index
        ):
            continue

        # --------------------------------------------------------------
        # IMPORTANT:
        #
        # Channel filtering happens PER NOTE.
        #
        # A physical MIDI track can contain multiple channels.
        # Therefore we do NOT discard an entire source track simply
        # because one of its channels is 9 or 15.
        # --------------------------------------------------------------

        for note in logical["notes"]:

            if (
                note.channel
                in EXCLUDED_CHANNELS
            ):
                continue

            # ----------------------------------------------------------
            # Transpose below C2.
            # ----------------------------------------------------------

            note.pitch = (
                transpose_to_c2(
                    note.pitch
                )
            )

            merged_notes.append(
                note
            )

    # ------------------------------------------------------------------
    # Remove duplicates AFTER transposition and merging.
    # ------------------------------------------------------------------

    merged_notes = (
        remove_duplicate_notes(
            merged_notes
        )
    )

    # ------------------------------------------------------------------
    # Optional quantization.
    #
    # This is deliberately AFTER duplicate removal.
    # ------------------------------------------------------------------

    if QUANTIZE_OUTPUT:

        merged_notes = (
            quantize_output(
                merged_notes
            )
        )

    other_track = mido.MidiTrack()

    midi.tracks.append(
        other_track
    )

    other_track.append(
        mido.MetaMessage(
            "track_name",
            name="Stage4 Other",
            time=0,
        )
    )

    add_note_events(
        other_track,
        merged_notes,
    )

    # ------------------------------------------------------------------
    # Ensure both tracks terminate properly.
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

    return midi


# ============================================================================
# OUTPUT WRITING
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
            os.unlink(
                temp_path
            )
        except FileNotFoundError:
            pass

        raise


# ============================================================================
# WORKER
# ============================================================================

def process_one(
    report,
):
    """
    Process one retained Stage 3 report.

    Returns a compact result so that large MIDI objects are never
    sent back to the parent process.
    """

    input_path = report.get(
        "path"
    )

    if not input_path:

        return {
            "status": "error",
            "reason":
                "missing_input_path",
        }

    winner = get_winner(
        report
    )

    # This should normally be impossible because the parent only
    # submits qualifying reports.
    if winner is None:

        return {
            "status": "discarded",
            "path": input_path,
        }

    winning_track_index = int(
        winner["track"]
    )

    try:

        # --------------------------------------------------------------
        # Read original MIDI with Mido.
        # --------------------------------------------------------------

        midi = mido.MidiFile(
            filename=input_path,
            clip=True,
        )

        # --------------------------------------------------------------
        # Reconstruct the logical tracks used by Stage 3.
        # --------------------------------------------------------------

        parsed = parse_midi_logical_tracks(
            midi
        )

        # --------------------------------------------------------------
        # Build output.
        # --------------------------------------------------------------

        output_midi = (
            build_output_midi(
                parsed,
                winning_track_index,
            )
        )

        # --------------------------------------------------------------
        # Determine output path.
        # --------------------------------------------------------------

        output_path = (
            make_output_path(
                input_path
            )
        )

        # --------------------------------------------------------------
        # Write atomically.
        # --------------------------------------------------------------

        write_midi_atomic(
            output_midi,
            output_path,
        )

        return {
            "status": "retained",
            "input_path":
                input_path,
            "output_path":
                output_path,
            "winning_track":
                winning_track_index,
            "melody_score":
                float(
                    winner[
                        "melody_score"
                    ]
                ),
            "pitch_mean":
                float(
                    winner[
                        "pitch_mean"
                    ]
                ),
            "monophonic_fraction":
                float(
                    winner[
                        "monophonic_fraction"
                    ]
                ),
        }

    except Exception as exc:

        return {
            "status": "error",
            "input_path":
                input_path,
            "reason":
                type(exc).__name__,
            "message":
                str(exc),
        }

    finally:

        # Explicitly release references before this worker processes
        # another file.
        try:
            del parsed
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
    Find the highest-numbered checkpoint.

    Returns:

        (number, path)

    or:

        (0, None)
    """

    if not os.path.isdir(
        STAGE4_DIR
    ):
        return 0, None

    candidates = []

    for filename in os.listdir(
        STAGE4_DIR
    ):

        match = (
            CHECKPOINT_PATTERN.match(
                filename
            )
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
    Write one independent checkpoint file.

    NEVER modifies a previous checkpoint.
    """

    checkpoint = {
        "stage": "4",

        "checkpoint_number":
            checkpoint_number,

        "completed_count":
            completed_count,

        "total_count":
            total_count,

        "retained_count":
            retained_count,

        "discarded_count":
            discarded_count,

        "error_count":
            error_count,

        "last_path":
            last_path,

        "configuration": {
            "pitch_mean":
                PITCH_MEAN,

            "score_threshold":
                SCORE_THRESHOLD,

            "monophonic_threshold":
                MONO_THRESHOLD,

            "workers":
                WORKERS,

            "batch_size":
                BATCH_SIZE,

            "checkpoint_interval":
                CHECKPOINT_INTERVAL,

            "excluded_channels":
                sorted(
                    EXCLUDED_CHANNELS
                ),

            "c2":
                C2,

            "quantize_output":
                QUANTIZE_OUTPUT,
        },
    }

    path = checkpoint_path(
        checkpoint_number
    )

    atomic_json_write(
        path,
        checkpoint,
    )


def load_latest_checkpoint():
    """
    Load the latest checkpoint.

    Returns:

        None

    or the checkpoint dictionary.
    """

    number, path = (
        find_latest_checkpoint()
    )

    if path is None:
        return None

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as fh:

        checkpoint = json.load(
            fh
        )

    checkpoint["_checkpoint_number"] = (
        number
    )

    return checkpoint


def delete_all_checkpoints():
    """
    Delete all checkpoint<number>.json files.
    """

    if not os.path.isdir(
        STAGE4_DIR
    ):
        return

    deleted = 0

    for filename in os.listdir(
        STAGE4_DIR
    ):

        if not (
            CHECKPOINT_PATTERN.match(
                filename
            )
        ):
            continue

        path = os.path.join(
            STAGE4_DIR,
            filename,
        )

        try:
            os.unlink(
                path
            )

            deleted += 1

        except FileNotFoundError:
            pass

    print(
        f"Deleted {deleted} checkpoint file(s)."
    )


# ============================================================================
# PREPARE WORK LIST
# ============================================================================

def build_work_list(
    reports,
):
    """
    Evaluate Stage 4 thresholds using reports.json only.

    No MIDI files are opened here.

    Returns:

        eligible_reports
        threshold_rejected
    """

    eligible = []

    threshold_rejected = 0

    for report in reports:

        if not report.get(
            "path"
        ):
            continue

        winner = get_winner(
            report
        )

        if winner is None:

            threshold_rejected += 1

            continue

        eligible.append(
            report
        )

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
        f"Stage 3 reports : "
        f"{STAGE3_REPORTS_JSON}"
    )

    print(
        f"Stage 4 output  : "
        f"{STAGE4_DIR}"
    )

    print(
        f"Workers         : "
        f"{WORKERS}"
    )

    print(
        f"Batch size      : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Checkpoint every: "
        f"{CHECKPOINT_INTERVAL:,} files"
    )

    print()

    # ------------------------------------------------------------------
    # Load reports.
    # ------------------------------------------------------------------

    reports = load_reports()

    print(
        f"Reports loaded  : "
        f"{len(reports):,}"
    )

    print()

    # ------------------------------------------------------------------
    # Determine which files need actual MIDI processing.
    # ------------------------------------------------------------------

    eligible_reports, threshold_rejected = (
        build_work_list(
            reports
        )
    )

    print(
        f"Threshold survivors : "
        f"{len(eligible_reports):,}"
    )

    print(
        f"Threshold rejected  : "
        f"{threshold_rejected:,}"
    )

    print()

    # ------------------------------------------------------------------
    # Resume.
    # ------------------------------------------------------------------

    checkpoint = (
        load_latest_checkpoint()
    )

    if checkpoint is not None:

        completed_count = int(
            checkpoint[
                "completed_count"
            ]
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
            checkpoint[
                "_checkpoint_number"
            ]
        )

        last_path = checkpoint.get(
            "last_path"
        )

        # --------------------------------------------------------------
        # Verify that the work-list size did not change.
        # --------------------------------------------------------------

        checkpoint_total = int(
            checkpoint[
                "total_count"
            ]
        )

        if checkpoint_total != len(
            eligible_reports
        ):

            raise RuntimeError(
                "Stage 4 cannot safely resume because "
                "the number of eligible reports changed.\n\n"
                f"Checkpoint: {checkpoint_total:,}\n"
                f"Current   : {len(eligible_reports):,}\n\n"
                "Do not delete or modify reports.json while a Stage 4 "
                "run is in progress."
            )

        print(
            "=" * 70
        )

        print(
            "RESUMING STAGE 4"
        )

        print(
            "=" * 70
        )

        print(
            f"Checkpoint       : "
            f"checkpoint{checkpoint_number}.json"
        )

        print(
            f"Already completed: "
            f"{completed_count:,}"
        )

        print(
            f"Last path        : "
            f"{last_path}"
        )

        print()

    else:

        completed_count = 0

        retained_count = 0

        discarded_count = 0

        error_count = 0

        checkpoint_number = 0

    # ------------------------------------------------------------------
    # Sanity-check checkpoint bounds.
    # ------------------------------------------------------------------

    if (
        completed_count < 0
        or
        completed_count >
        len(eligible_reports)
    ):

        raise RuntimeError(
            "Invalid completed_count in checkpoint."
        )

    remaining = eligible_reports[
        completed_count:
    ]

    # ------------------------------------------------------------------
    # Process remaining files in bounded batches.
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

        absolute_start = (
            completed_count
        )

        absolute_end = (
            completed_count
            + len(batch)
        )

        print(
            "=" * 70
        )

        print(
            f"STAGE 4 BATCH "
            f"{absolute_start + 1:,} - "
            f"{absolute_end:,} "
            f"of "
            f"{len(eligible_reports):,}"
        )

        print(
            "=" * 70
        )

        # --------------------------------------------------------------
        # A fresh process pool per bounded batch.
        #
        # This is deliberate memory management.
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
                # Checkpoint.
                #
                # Each checkpoint is a NEW file.
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
                            len(
                                eligible_reports
                            ),

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
                        f"Checkpoint written: "
                        f"checkpoint"
                        f"{checkpoint_number}"
                        f".json"
                    )

        # --------------------------------------------------------------
        # Release the batch before creating the next worker pool.
        # --------------------------------------------------------------

        del batch

    # ------------------------------------------------------------------
    # IMPORTANT:
    #
    # If there were processing errors, DO NOT delete checkpoints.
    #
    # This preserves the recovery point and makes failures diagnosable.
    # A completely successful run deletes them below.
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

        print(
            "WARNING:"
        )

        print(
            "Processing errors occurred."
        )

        print(
            "Checkpoint files have NOT been deleted."
        )

        print(
            "Fix the problem and resume Stage 4."
        )

        print()

        return 1

    # ------------------------------------------------------------------
    # Successful completion.
    #
    # Only now delete ALL checkpoint<number>.json files.
    # ------------------------------------------------------------------

    delete_all_checkpoints()

    print()

    print(
        "Stage 4 finished successfully."
    )

    print(
        f"Output directory:"
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

    sys.exit(
        main()
    )
