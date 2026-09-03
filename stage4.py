#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Los Angeles MIDI Dataset
Stage 4 — Melody/Accompaniment Extraction

INPUT
-----
Dataset/LAMDselection/selection_stage3/
    candidates_1.json
    candidates_2.json
    ...
    candidates_f.json

The Stage 3 candidate JSON files contain the physical MIDI track numbers
and the Stage 3 musical-analysis metrics.

The ORIGINAL physical MIDI files are reopened for Stage 4 processing:

Dataset/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/MIDIs/
    1/
    2/
    ...
    9/
    a/
    ...
    f/

NO LAMDa metadata is used.

STAGE 4 QUALIFICATION
---------------------
A MIDI file survives iff at least one Stage 3 candidate track satisfies:

    pitch_mean          >= 40
    melody_score        >= 0.86
    monophonic_fraction >= 0.95

Among qualifying tracks, the winner is the one with the highest
melody_score.

OUTPUT
------
Dataset/LAMDselection/selection_stage4/

MIDI files:

    1/<filename>.mid
    2/<filename>.mid
    ...
    f/<filename>.mid

For each input subdirectory, one independent manifest set is produced
directly in the Stage 4 directory:

    candidates_1.txt
    candidates_1.json
    rejections_1.json
    summary_1.json

    candidates_2.txt
    candidates_2.json
    rejections_2.json
    summary_2.json

    ...

CHECKPOINTING
-------------
Checkpoints are NOT accumulated in one ever-growing file.

Instead:

    checkpoint1.json
    checkpoint2.json
    checkpoint3.json
    ...

Each checkpoint contains only one newly completed contiguous batch.

A failed/interrupted run leaves checkpoints intact.

After ALL subdirectories complete successfully, ALL checkpoint*.json files
are deleted.

PROCESSING
----------
24 worker processes by default.

Outstanding work is bounded to prevent tens of thousands of Future objects
from accumulating in the parent process.

MEMORY
------
Each worker handles one MIDI file at a time and releases the MidiFile
object before returning.

OUTPUT MIDI STRUCTURE
---------------------
Track 0:
    - notes from the winning physical MIDI track
    - channel 9 notes removed
    - tempo-map messages copied from wherever they occur in the input
    - key-signature messages copied from wherever they occur
    - time-signature messages copied from wherever they occur

Track 1:
    - all notes from every NON-winning physical track
    - channel 9 notes removed
    - pitches below C2 (MIDI 36) repeatedly transposed +12 until >= C2
    - duplicate notes removed after transposition/merging

Other MIDI events such as program changes and controller events are not
copied to Track 1 because the Stage 4 specification concerns the note
material itself.

QUANTIZATION
------------
quantize_output() intentionally does nothing.

QUANTIZE_OUTPUT=False is provided for future development.

When enabled in the future, quantization will happen AFTER duplicate
removal and BEFORE Track 1 is written.
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

STAGE3_DIR = Path(
    "Dataset/LAMDselection/selection_stage3"
)

STAGE4_DIR = Path(
    "Dataset/LAMDselection/selection_stage4"
)


# ============================================================================
# CONFIGURATION
# ============================================================================

WORKERS = 24

# Keep the parent bounded.
MAX_PENDING = WORKERS * 2

# Number of completed MIDI files represented by one checkpoint.
CHECKPOINT_INTERVAL = 1000

INPUT_SUBDIRECTORIES = tuple(
    "123456789abcdef"
)

# Stage 4 thresholds.
PITCH_MEAN = 40.0
SCORE_THRESHOLD = 0.86
MONO_THRESHOLD = 0.94

# MIDI General MIDI percussion channel.
GM_DRUM_CHANNEL = 9

# C2.
C2 = 36

# Future quantization switch.
QUANTIZE_OUTPUT = False


# ============================================================================
# JSON UTILITIES
# ============================================================================

def make_json_serializable(obj):
    """
    Convert NumPy/scalar/container values into JSON-safe objects.
    """

    if isinstance(obj, dict):
        return {make_json_serializable(key) : make_json_serializable(value) for key, value in obj.items()}

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
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    data = make_json_serializable(data)

    temporary = None

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

            temporary = Path(fh.name)

        os.replace(
            temporary,
            path,
        )

    finally:

        if (
            temporary is not None
            and temporary.exists()
        ):
            try:
                temporary.unlink()
            except OSError:
                pass


# ============================================================================
# STAGE 3 INPUT
# ============================================================================

def load_stage3_candidates(subdir):
    """
    Load candidates_<subdir>.json from Stage 3.
    """

    path = (
        STAGE3_DIR
        /
        f"candidates_{subdir}.json"
    )

    if not path.is_file():
        raise FileNotFoundError(
            f"Stage 3 candidate file not found:\n{path}"
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
            f"Invalid Stage 3 candidate file:\n{path}\n"
            "Expected a top-level 'candidates' list."
        )

    return candidates


def candidate_fingerprint(candidates):
    """
    Stable fingerprint of the Stage 3 input ordering.
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
# QUALIFICATION
# ============================================================================

def get_stage3_analysis(candidate):
    """
    Return the Stage 3 candidate-track reports.

    Stage 3 stores them as:

        candidate["stage3"]["candidates"]
    """

    stage3 = candidate.get("stage3")

    if not isinstance(stage3, dict):
        return []

    tracks = stage3.get("candidates")

    if not isinstance(tracks, list):
        return []

    return tracks


def qualifying_tracks(candidate):
    """
    Return all Stage 3 candidate tracks satisfying ALL Stage 4 thresholds.

    The Stage 3 candidate list contains the highest-ranked tracks, so it is
    sufficient for Stage 4 qualification.
    """

    qualifying = []

    for track_report in get_stage3_analysis(candidate):

        try:
            pitch_mean = float(
                track_report.get(
                    "pitch_mean",
                    0.0,
                )
            )

            melody_score = float(
                track_report.get(
                    "melody_score",
                    0.0,
                )
            )

            monophonic_fraction = float(
                track_report.get(
                    "monophonic_fraction",
                    0.0,
                )
            )

        except (
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
                track_report
            )

    qualifying.sort(
        key=lambda item: float(
            item.get(
                "melody_score",
                0.0,
            )
        ),
        reverse=True,
    )

    return qualifying


# ============================================================================
# MIDI IMPORT
# ============================================================================

def import_mido():

    try:
        import mido

        return mido

    except Exception as exc:

        print()
        print("=" * 78)
        print("ERROR: Could not import mido.")
        print("=" * 78)
        print()
        print(
            f"{type(exc).__name__}: {exc}"
        )
        print()

        raise


# ============================================================================
# MIDI MESSAGE HELPERS
# ============================================================================

def absolute_messages(track):
    """
    Convert a MidiTrack's delta-time messages into:

        (absolute_tick, message)

    The original Message object is copied before returning.
    """

    result = []

    absolute_tick = 0

    for message in track:

        absolute_tick += int(
            message.time
        )

        result.append(
            (
                absolute_tick,
                message.copy(),
            )
        )

    return result


def note_message_type(message):
    """
    Return True for MIDI note-on and note-off messages.

    A note_on with velocity 0 is also a note-off semantically, but for
    extraction purposes we retain the original message type and pair it
    correctly.
    """

    return (
        not message.is_meta
        and
        message.type in (
            "note_on",
            "note_off",
        )
    )


def is_note_on(message):
    return (
        not message.is_meta
        and
        message.type == "note_on"
        and
        int(message.velocity) > 0
    )


def is_note_off(message):
    if message.is_meta:
        return False

    if message.type == "note_off":
        return True

    return (
        message.type == "note_on"
        and
        int(message.velocity) == 0
    )


# ============================================================================
# NOTE EXTRACTION
# ============================================================================

def extract_track_notes(track):
    """
    Extract complete MIDI notes from one physical MIDI track.

    Returns dictionaries containing:

        start
        end
        pitch
        velocity
        channel

    Notes are paired per MIDI channel/pitch using FIFO pairing.

    This is deliberately done at the physical MIDI-message level so that
    channel 9 can be filtered correctly.
    """

    absolute = absolute_messages(track)

    active = {}

    notes = []

    for tick, message in absolute:

        if not note_message_type(message):
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
                    tick,
                    int(message.velocity),
                )
            )

        elif is_note_off(message):

            queue = active.get(key)

            if not queue:
                continue

            start_tick, velocity = queue.pop(0)

            if tick < start_tick:
                continue

            notes.append(
                {
                    "start": int(start_tick),
                    "end": int(tick),
                    "pitch": pitch,
                    "velocity": velocity,
                    "channel": channel,
                }
            )

    return notes


# ============================================================================
# NOTE TRANSFORMATION
# ============================================================================

def transpose_to_c2(pitch):
    """
    Repeatedly transpose a pitch upward by octaves until it is >= C2.
    """

    pitch = int(pitch)

    while pitch < C2:
        pitch += 12

    return pitch


def merge_and_transform_notes(
    tracks,
    winning_track_index,
):
    """
    Collect all notes from every non-winning physical track.

    Channel 9 notes are discarded.

    Remaining notes are transposed upward until >= C2.

    Duplicate notes are removed AFTER transposition.

    Duplicate identity is:

        (start_tick, end_tick, pitch)

    Channel and velocity deliberately do not participate in duplicate
    identity because the purpose of the operation is to collapse duplicate
    musical notes introduced by merging.
    """

    transformed = []

    for track_index, track in enumerate(tracks):

        if track_index == winning_track_index:
            continue

        notes = extract_track_notes(
            track
        )

        for note in notes:

            if note["channel"] == GM_DRUM_CHANNEL:
                continue

            transformed_pitch = transpose_to_c2(
                note["pitch"]
            )

            transformed.append(
                {
                    "start": int(
                        note["start"]
                    ),
                    "end": int(
                        note["end"]
                    ),
                    "pitch": int(
                        transformed_pitch
                    ),
                    "velocity": int(
                        note["velocity"]
                    ),
                    "channel": int(
                        note["channel"]
                    ),
                }
            )

    # --------------------------------------------------------------
    # Duplicate removal AFTER transposition.
    # --------------------------------------------------------------

    transformed.sort(
        key=lambda note: (
            note["start"],
            note["end"],
            note["pitch"],
            note["channel"],
            note["velocity"],
        )
    )

    seen = set()
    unique = []

    for note in transformed:

        identity = (
            note["start"],
            note["end"],
            note["pitch"],
        )

        if identity in seen:
            continue

        seen.add(identity)
        unique.append(note)

    # Restore deterministic chronological ordering.
    unique.sort(
        key=lambda note: (
            note["start"],
            note["end"],
            note["pitch"],
            note["channel"],
            note["velocity"],
        )
    )

    return unique


# ============================================================================
# WINNING TRACK
# ============================================================================

def extract_winning_track_notes(
    track,
):
    """
    Extract notes from the winning physical track.

    Channel 9 notes are removed.

    No transposition or duplicate removal is performed on the winning
    track.
    """

    notes = extract_track_notes(
        track
    )

    return [
        note
        for note in notes
        if note["channel"] != GM_DRUM_CHANNEL
    ]


# ============================================================================
# META MAPS
# ============================================================================

MAP_MESSAGE_TYPES = frozenset(
    {
        "set_tempo",
        "key_signature",
        "time_signature",
    }
)


def collect_map_messages(tracks):
    """
    Collect tempo/key-signature/time-signature messages from ALL physical
    MIDI tracks.

    Their original absolute tick positions are retained.

    All such messages are placed into Track 0.
    """

    collected = []

    for track_index, track in enumerate(tracks):

        for tick, message in absolute_messages(track):

            if (
                message.is_meta
                and
                message.type in MAP_MESSAGE_TYPES
            ):
                collected.append(
                    (
                        int(tick),
                        int(track_index),
                        message.copy(),
                    )
                )

    # Deterministic ordering.
    collected.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return collected


# ============================================================================
# OUTPUT TRACK CONSTRUCTION
# ============================================================================

def append_absolute_messages(
    output_track,
    messages,
):
    """
    Append:

        (absolute_tick, Message)

    to a MidiTrack using delta times.
    """

    previous_tick = 0

    for absolute_tick, message in messages:

        absolute_tick = int(
            absolute_tick
        )

        delta = (
            absolute_tick
            -
            previous_tick
        )

        if delta < 0:
            raise RuntimeError(
                "Negative MIDI delta time while constructing output."
            )

        output_track.append(
            message.copy(
                time=delta
            )
        )

        previous_tick = absolute_tick


def notes_to_messages(
    notes,
):
    """
    Convert complete note dictionaries into absolute MIDI messages.
    """

    messages = []

    for note in notes:

        start = int(
            note["start"]
        )

        end = int(
            note["end"]
        )

        if end <= start:
            continue

        pitch = int(
            note["pitch"]
        )

        if not 0 <= pitch <= 127:
            continue

        velocity = int(
            note["velocity"]
        )

        velocity = max(
            0,
            min(
                127,
                velocity,
            )
        )

        channel = int(
            note["channel"]
        )

        channel = max(
            0,
            min(
                15,
                channel,
            )
        )

        messages.append(
            (
                start,
                {
                    "kind": "on",
                    "pitch": pitch,
                    "velocity": velocity,
                    "channel": channel,
                },
            )
        )

        messages.append(
            (
                end,
                {
                    "kind": "off",
                    "pitch": pitch,
                    "velocity": 0,
                    "channel": channel,
                },
            )
        )

    # At equal ticks note-offs precede note-ons.
    messages.sort(
        key=lambda item: (
            item[0],
            0 if item[1]["kind"] == "off" else 1,
        )
    )

    return messages


def build_note_track(
    mido,
    notes,
):
    """
    Build a MidiTrack containing only note messages.
    """

    track = mido.MidiTrack()

    absolute = notes_to_messages(
        notes
    )

    previous_tick = 0

    for tick, data in absolute:

        delta = (
            int(tick)
            -
            previous_tick
        )

        if delta < 0:
            raise RuntimeError(
                "Negative note delta time."
            )

        if data["kind"] == "on":

            message = mido.Message(
                "note_on",
                channel=data["channel"],
                note=data["pitch"],
                velocity=data["velocity"],
                time=delta,
            )

        else:

            message = mido.Message(
                "note_off",
                channel=data["channel"],
                note=data["pitch"],
                velocity=0,
                time=delta,
            )

        track.append(
            message
        )

        previous_tick = int(tick)

    return track


def build_track_zero(
    mido,
    winning_notes,
    map_messages,
):
    """
    Build Track 0 from:

        - tempo map
        - key-signature map
        - time-signature map
        - winning-track notes
    """

    events = []

    # Map messages.
    for tick, track_index, message in map_messages:

        events.append(
            (
                int(tick),
                0,
                int(track_index),
                message.copy(),
            )
        )

    # Winning notes.
    for tick, data in notes_to_messages(
        winning_notes
    ):

        if data["kind"] == "on":

            message = mido.Message(
                "note_on",
                channel=data["channel"],
                note=data["pitch"],
                velocity=data["velocity"],
                time=0,
            )

        else:

            message = mido.Message(
                "note_off",
                channel=data["channel"],
                note=data["pitch"],
                velocity=0,
                time=0,
            )

        # Note-offs before note-ons at equal ticks.
        priority = (
            0
            if data["kind"] == "off"
            else 1
        )

        events.append(
            (
                int(tick),
                priority,
                0,
                message,
            )
        )

    events.sort(
        key=lambda item: (
            item[0],
            item[1],
            item[2],
        )
    )

    output = mido.MidiTrack()

    previous_tick = 0

    for tick, _, _, message in events:

        delta = (
            int(tick)
            -
            previous_tick
        )

        if delta < 0:
            raise RuntimeError(
                "Negative Track 0 delta time."
            )

        output.append(
            message.copy(
                time=delta
            )
        )

        previous_tick = int(tick)

    return output


# ============================================================================
# QUANTIZATION
# ============================================================================

def quantize_output(notes):
    """
    Future quantization hook.

    Currently deliberately does NOTHING.
    """

    return notes


# ============================================================================
# MIDI OUTPUT
# ============================================================================

def output_midi_path(
    subdir,
    input_path,
):
    """
    Preserve the original filename and the input hexadecimal subdirectory.
    """

    output_directory = (
        STAGE4_DIR
        /
        str(subdir)
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return (
        output_directory
        /
        Path(input_path).name
    )


def atomic_midi_save(
    midi,
    output_path,
):
    """
    Atomically save a MIDI file.

    This prevents a worker interruption from leaving a partially written
    final MIDI.
    """

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            delete=False,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        ) as fh:

            temporary = Path(
                fh.name
            )

        midi.save(
            filename=str(
                temporary
            )
        )

        os.replace(
            temporary,
            output_path,
        )

    finally:

        if (
            temporary is not None
            and temporary.exists()
        ):
            try:
                temporary.unlink()
            except OSError:
                pass


# ============================================================================
# PROCESS ONE MIDI
# ============================================================================

def process_midi(
    candidate,
    subdir,
):
    """
    Process one Stage 3 survivor.

    Returns a compact dictionary suitable for transfer to the parent.
    """

    md5 = str(
        candidate.get(
            "md5",
            "",
        )
    ).lower()

    input_path = Path(
        str(
            candidate.get(
                "path",
                "",
            )
        )
    )

    result_base = {
        "md5": md5,
        "path": str(input_path),
    }

    # --------------------------------------------------------------
    # Stage 4 qualification.
    # --------------------------------------------------------------

    qualifying = qualifying_tracks(
        candidate
    )

    if not qualifying:

        result_base.update(
            {
                "kind": "rejection",
                "reason":
                    "no_track_satisfies_stage4_thresholds",
            }
        )

        return result_base

    winner = qualifying[0]

    try:
        winning_track_index = int(
            winner["track"]
        )

    except (
        KeyError,
        TypeError,
        ValueError,
    ):

        result_base.update(
            {
                "kind": "rejection",
                "reason":
                    "invalid_winning_track_index",
            }
        )

        return result_base

    if not input_path.is_file():

        result_base.update(
            {
                "kind": "rejection",
                "reason":
                    "midi_file_not_found",
            }
        )

        return result_base

    # --------------------------------------------------------------
    # Read original physical MIDI.
    # --------------------------------------------------------------

    mido = import_mido()

    midi = None

    try:

        midi = mido.MidiFile(
            str(input_path)
        )

        tracks = midi.tracks

        if (
            winning_track_index < 0
            or
            winning_track_index >= len(tracks)
        ):
            result_base.update(
                {
                    "kind": "rejection",
                    "reason":
                        "winning_track_index_out_of_range",
                    "winning_track":
                        winning_track_index,
                    "track_count":
                        len(tracks),
                }
            )

            return result_base

        # ----------------------------------------------------------
        # Winning track.
        # ----------------------------------------------------------

        winning_notes = extract_winning_track_notes(
            tracks[
                winning_track_index
            ]
        )

        # ----------------------------------------------------------
        # Map messages from ALL tracks.
        # ----------------------------------------------------------

        map_messages = collect_map_messages(
            tracks
        )

        # ----------------------------------------------------------
        # Remaining tracks.
        # ----------------------------------------------------------

        accompaniment_notes = (
            merge_and_transform_notes(
                tracks,
                winning_track_index,
            )
        )

        # ----------------------------------------------------------
        # Future quantization hook.
        # ----------------------------------------------------------

        if QUANTIZE_OUTPUT:

            accompaniment_notes = quantize_output(
                accompaniment_notes
            )

        # ----------------------------------------------------------
        # Build output MIDI.
        # ----------------------------------------------------------

        output_midi = mido.MidiFile(
            type=1,
            ticks_per_beat=midi.ticks_per_beat,
        )

        output_track_0 = build_track_zero(
            mido,
            winning_notes,
            map_messages,
        )

        output_track_1 = build_note_track(
            mido,
            accompaniment_notes,
        )

        output_midi.tracks.append(
            output_track_0
        )

        output_midi.tracks.append(
            output_track_1
        )

        # ----------------------------------------------------------
        # Save.
        # ----------------------------------------------------------

        output_path = output_midi_path(
            subdir,
            input_path,
        )

        atomic_midi_save(
            output_midi,
            output_path,
        )

        result_base.update(
            {
                "kind": "candidate",
                "output_path":
                    str(output_path),
                "winning_track":
                    winning_track_index,
                "winning_track_melody_score":
                    float(
                        winner.get(
                            "melody_score",
                            0.0,
                        )
                    ),
                "winning_track_pitch_mean":
                    float(
                        winner.get(
                            "pitch_mean",
                            0.0,
                        )
                    ),
                "winning_track_monophonic_fraction":
                    float(
                        winner.get(
                            "monophonic_fraction",
                            0.0,
                        )
                    ),
                "qualifying_track_count":
                    int(
                        len(
                            qualifying
                        )
                    ),
                "winning_notes":
                    int(
                        len(
                            winning_notes
                        )
                    ),
                "track1_notes":
                    int(
                        len(
                            accompaniment_notes
                        )
                    ),
                "input_tracks":
                    int(
                        len(tracks)
                    ),
                "output_tracks":
                    2,
            }
        )

        return result_base

    except Exception as exc:

        result_base.update(
            {
                "kind": "rejection",
                "reason":
                    "midi_processing_error:"
                    +
                    type(exc).__name__,
                "error":
                    str(exc),
            }
        )

        return result_base

    finally:

        midi = None

        try:
            del midi
        except UnboundLocalError:
            pass

        gc.collect()


# ============================================================================
# WORKER
# ============================================================================

def stage4_worker(
    candidate,
    subdir,
):

    try:

        return process_midi(
            candidate,
            subdir,
        )

    except Exception as exc:

        return {
            "kind": "rejection",
            "md5":
                str(
                    candidate.get(
                        "md5",
                        "",
                    )
                ).lower(),
            "path":
                str(
                    candidate.get(
                        "path",
                        "",
                    )
                ),
            "reason":
                "worker_error:"
                +
                type(exc).__name__,
            "error":
                str(exc),
        }

    finally:

        gc.collect()


# ============================================================================
# CHECKPOINTS
# ============================================================================

def checkpoint_paths():

    paths = []

    for path in STAGE4_DIR.glob(
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
    Write ONLY one newly completed contiguous batch.
    """

    checkpoint_path = (
        STAGE4_DIR
        /
        f"checkpoint{checkpoint_number}.json"
    )

    checkpoint = {
        "stage": "4",
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
        ) != str(subdir):

            continue

        if int(
            checkpoint.get(
                "input_count",
                -1,
            )
        ) != len(candidates):

            raise RuntimeError(
                f"Checkpoint input count mismatch:\n"
                f"{path}"
            )

        if (
            checkpoint.get(
                "input_fingerprint"
            )
            !=
            input_fingerprint
        ):

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
                f"Checkpoint sequence gap/overlap "
                f"for subdirectory {subdir}:\n"
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

    # Verify order against input.
    for index, result in enumerate(
        reconstructed
    ):

        expected_md5 = str(
            candidates[index].get(
                "md5",
                "",
            )
        ).lower()

        actual_md5 = str(
            result.get(
                "md5",
                "",
            )
        ).lower()

        if actual_md5 != expected_md5:

            raise RuntimeError(
                "Checkpoint ordering mismatch:\n"
                f"  subdirectory: {subdir}\n"
                f"  index: {index}\n"
                f"  expected MD5: {expected_md5}\n"
                f"  actual MD5: {actual_md5}"
            )

    return reconstructed


# ============================================================================
# FINAL OUTPUTS
# ============================================================================

def output_manifest_paths(subdir):

    return {
        "candidates_txt":
            STAGE4_DIR
            /
            f"candidates_{subdir}.txt",

        "candidates_json":
            STAGE4_DIR
            /
            f"candidates_{subdir}.json",

        "rejections_json":
            STAGE4_DIR
            /
            f"rejections_{subdir}.json",

        "summary_json":
            STAGE4_DIR
            /
            f"summary_{subdir}.json",
    }


def write_final_outputs(
    subdir,
    candidates,
    results,
):
    """
    Write one independent manifest set for this input subdirectory.
    """

    survivors = []
    rejections = []

    rejection_counts = {}

    for result in results:

        if result.get(
            "kind"
        ) == "candidate":

            survivors.append(
                result
            )

        else:

            rejections.append(
                result
            )

            reason = str(
                result.get(
                    "reason",
                    "unknown",
                )
            )

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

    paths = output_manifest_paths(
        subdir
    )

    # --------------------------------------------------------------
    # candidates_<subdir>.txt
    # --------------------------------------------------------------

    with open(
        paths["candidates_txt"],
        "w",
        encoding="utf-8",
    ) as fh:

        for record in survivors:

            fh.write(
                str(
                    record[
                        "output_path"
                    ]
                )
                +
                "\n"
            )

    # --------------------------------------------------------------
    # candidates_<subdir>.json
    # --------------------------------------------------------------

    atomic_json_write(
        paths["candidates_json"],
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

            "stage":
                "4",

            "input_subdirectory":
                str(subdir),

            "input":
                str(
                    STAGE3_DIR
                    /
                    f"candidates_{subdir}.json"
                ),

            "midi_root":
                str(
                    MIDI_ROOT
                    /
                    str(subdir)
                ),

            "output_root":
                str(STAGE4_DIR),

            "workers":
                int(WORKERS),

            "pitch_mean_threshold":
                float(PITCH_MEAN),

            "melody_score_threshold":
                float(SCORE_THRESHOLD),

            "monophonic_fraction_threshold":
                float(MONO_THRESHOLD),

            "quantize_output":
                bool(QUANTIZE_OUTPUT),

            "candidate_count":
                int(
                    len(survivors)
                ),

            "candidates":
                survivors,
        }
    )

    # --------------------------------------------------------------
    # rejections_<subdir>.json
    # --------------------------------------------------------------

    atomic_json_write(
        paths["rejections_json"],
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

            "stage":
                "4",

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
        }
    )

    # --------------------------------------------------------------
    # summary_<subdir>.json
    # --------------------------------------------------------------

    atomic_json_write(
        paths["summary_json"],
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",

            "stage":
                "4",

            "input_subdirectory":
                str(subdir),

            "input_count":
                int(
                    len(candidates)
                ),

            "survivor_count":
                int(
                    len(survivors)
                ),

            "rejected_count":
                int(
                    len(rejections)
                ),

            "rejection_reasons":
                rejection_counts,

            "thresholds":
                {
                    "pitch_mean":
                        float(PITCH_MEAN),

                    "melody_score":
                        float(SCORE_THRESHOLD),

                    "monophonic_fraction":
                        float(MONO_THRESHOLD),
                },

            "quantize_output":
                bool(QUANTIZE_OUTPUT),
        }
    )

    return (
        len(survivors),
        len(rejections),
    )


# ============================================================================
# PROCESS SUBDIRECTORY
# ============================================================================

def process_subdirectory(
    subdir,
):
    candidates = load_stage3_candidates(
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
        f"STAGE 4 — SUBDIRECTORY {subdir}"
    )
    print("=" * 78)

    print(
        f"Stage 3 input      : {input_count:,}"
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

        uncheckpointed_results = []

        with ProcessPoolExecutor(
            max_workers=WORKERS
        ) as executor:

            # ------------------------------------------------------
            # Initial bounded submission.
            # ------------------------------------------------------

            while (
                next_submit < input_count
                and
                len(pending) < MAX_PENDING
            ):

                future = executor.submit(
                    stage4_worker,
                    candidates[
                        next_submit
                    ],
                    subdir,
                )

                pending[
                    future
                ] = next_submit

                next_submit += 1

            # ------------------------------------------------------
            # Consume/refill.
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
                            "kind":
                                "rejection",

                            "md5":
                                str(
                                    candidate.get(
                                        "md5",
                                        "",
                                    )
                                ).lower(),

                            "path":
                                str(
                                    candidate.get(
                                        "path",
                                        "",
                                    )
                                ),

                            "reason":
                                "future_error:"
                                +
                                type(exc).__name__,

                            "error":
                                str(exc),
                        }

                    ready[
                        original_index
                    ] = result

                    # --------------------------------------------------
                    # Refill immediately.
                    # --------------------------------------------------

                    if (
                        next_submit
                        <
                        input_count
                    ):

                        future2 = executor.submit(
                            stage4_worker,
                            candidates[
                                next_submit
                            ],
                            subdir,
                        )

                        pending[
                            future2
                        ] = next_submit

                        next_submit += 1

                # --------------------------------------------------
                # Commit only contiguous completed prefix.
                # --------------------------------------------------

                while next_commit in ready:

                    result = ready.pop(
                        next_commit
                    )

                    results.append(
                        result
                    )

                    uncheckpointed_results.append(
                        result
                    )

                    next_commit += 1

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

        # ----------------------------------------------------------
        # Final partial checkpoint.
        # ----------------------------------------------------------

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
            f"Stage 4 result count mismatch for "
            f"subdirectory {subdir}:\n"
            f"  expected: {input_count:,}\n"
            f"  actual:   {len(results):,}"
        )

    for index, result in enumerate(
        results
    ):

        expected_md5 = str(
            candidates[index].get(
                "md5",
                "",
            )
        ).lower()

        actual_md5 = str(
            result.get(
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

    survivor_count, rejected_count = (
        write_final_outputs(
            subdir,
            candidates,
            results,
        )
    )

    print()
    print(
        f"STAGE 4 COMPLETE — {subdir}"
    )

    print(
        f"  Input     : {input_count:,}"
    )

    print(
        f"  Survivors : {survivor_count:,}"
    )

    print(
        f"  Rejected  : {rejected_count:,}"
    )

    return (
        survivor_count,
        rejected_count,
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
        "REMOVING STAGE 4 CHECKPOINTS"
    )
    print("=" * 78)
    print()

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
        "STAGE 4 — MELODY / ACCOMPANIMENT EXTRACTION"
    )
    print("=" * 78)
    print()

    print(
        "Stage 4 input = Stage 3 candidates directly."
    )

    print(
        "LAMDa metadata = NOT USED."
    )

    print(
        f"Pitch mean threshold = {PITCH_MEAN}"
    )

    print(
        f"Melody score threshold = {SCORE_THRESHOLD}"
    )

    print(
        f"Monophonic fraction threshold = "
        f"{MONO_THRESHOLD}"
    )

    print(
        f"Workers = {WORKERS}"
    )

    print()

    if not STAGE3_DIR.is_dir():

        raise RuntimeError(
            f"Stage 3 directory does not exist:\n"
            f"{STAGE3_DIR}"
        )

    if not MIDI_ROOT.is_dir():

        raise RuntimeError(
            f"MIDI root does not exist:\n"
            f"{MIDI_ROOT}"
        )

    STAGE4_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    try:

        for subdir in INPUT_SUBDIRECTORIES:

            input_file = (
                STAGE3_DIR
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
            "STAGE 4 — ALL SUBDIRECTORIES COMPLETE"
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
            "STAGE 4 INTERRUPTED"
        )
        print("=" * 78)
        print()

        print(
            "Checkpoints have been retained."
        )

        raise

    except Exception:

        print()
        print("=" * 78)
        print(
            "STAGE 4 FAILED"
        )
        print("=" * 78)
        print()

        print(
            "Checkpoints have been retained."
        )

        raise


if __name__ == "__main__":
    main()
