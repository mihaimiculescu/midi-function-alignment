#!/usr/bin/env python3

import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import MIDI
import mido
from collections import OrderedDict

# ============================================================
# CONFIGURATION
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent

DATASET_DIR = (
    PROJECT_ROOT
    / "Dataset"
)

MIDI_ROOT = DATASET_DIR / "MIDIs"

STAGE3_DIR = (
    DATASET_DIR
    / "LAMDselection"
    / "selection_stage1"
    / "stage3"
)

STAGE3_REPORTS = STAGE3_DIR / "reports.json"

STAGE4_DIR = STAGE3_DIR / "stage4"

# Separate checkpoint files:
# checkpoint1.json
# checkpoint2.json
# ...
CHECKPOINT_DIR = STAGE4_DIR

# ------------------------------------------------------------
# Stage 3 selection criteria
# ------------------------------------------------------------

SCORE_THRESHOLD = 0.86
MONO_THRESHOLD = 0.95
PITCH_MEAN_THRESHOLD = 39.0

# ------------------------------------------------------------
# Accompaniment processing
# ------------------------------------------------------------

# User requested:
# keep this FALSE for now.
#
# When switched to True, the execution path will call
# apply_quantization() immediately after accompaniment
# merging and before octave normalization / duplicate removal.
APPLY_16TH_QUANTIZATION = False

# C2 = MIDI note 36
MIN_ACCOMPANIMENT_PITCH = 36

# Stage 3 excluded percussion channels.
# MIDI channels are 0-based internally.
EXCLUDED_ACCOMP_CHANNELS = {9, 15}

# ------------------------------------------------------------
# Performance
# ------------------------------------------------------------

MAX_WORKERS = 24
BATCH_SIZE = 1


# ============================================================
# QUANTIZATION HOOK
# ============================================================

def apply_quantization(score_track, ticks_per_quarter):
    """
    Apply the project's 16th-note quantization.

    This is intentionally a NO-OP for now.

    The execution path is already wired so that when
    APPLY_16TH_QUANTIZATION becomes True, this function is
    called immediately after accompaniment merging and before
    octave normalization / duplicate suppression.

    Parameters
    ----------
    score_track:
        One MIDI score track containing absolute-time events.

    ticks_per_quarter:
        MIDI ticks per quarter note.

    Returns
    -------
    list
        The unchanged score track for now.
    """

    return score_track


# ============================================================
# STAGE 3 SELECTION
# ============================================================

def select_winning_candidate(report):
    """
    Return the highest-scoring Stage 3 candidate satisfying
    all Stage 4 selection criteria.

    Returns
    -------
    dict or None
    """

    stage3 = report.get("stage3", {})
    candidates = stage3.get("candidates", [])

    qualifying = []

    for candidate in candidates:
        try:
            pitch_mean = float(candidate["pitch_mean"])
            melody_score = float(candidate["melody_score"])
            mono_fraction = float(candidate["monophonic_fraction"])
        except (KeyError, TypeError, ValueError):
            continue

        if pitch_mean < PITCH_MEAN_THRESHOLD:
            continue

        if melody_score < SCORE_THRESHOLD:
            continue

        if mono_fraction < MONO_THRESHOLD:
            continue

        qualifying.append(candidate)

    if not qualifying:
        return None

    return max(
        qualifying,
        key=lambda candidate: float(candidate["melody_score"])
    )


# ============================================================
# MIDI HELPERS
# ============================================================

def extract_instrument_track(
    midi_path,
    raw_track_index,
    channel,
    program,
):
    """
    Extract exactly the notes belonging to:

        (program, channel, raw_track_index)

    using the same program-change and note-pairing semantics as
    UglyMIDI.
    """

    midi_data = mido.MidiFile(
        filename=str(midi_path),
        clip=True,
    )

    if raw_track_index >= len(midi_data.tracks):
        raise ValueError(
            f"Raw MIDI track {raw_track_index} does not exist. "
            f"MIDI contains {len(midi_data.tracks)} tracks."
        )

    track = midi_data.tracks[
        raw_track_index
    ]

    absolute_tick = 0

    current_program = [0] * 16

    last_note_on = {}

    notes = []

    for event in track:

        absolute_tick += event.time

        if event.type == "program_change":

            current_program[
                event.channel
            ] = event.program

            continue

        if event.type == "note_on" and event.velocity > 0:

            key = (
                event.channel,
                event.note,
            )

            last_note_on.setdefault(
                key,
                []
            ).append(
                (
                    absolute_tick,
                    event.velocity,
                    current_program[
                        event.channel
                    ],
                )
            )

            continue

        if not (
            event.type == "note_off"
            or (
                event.type == "note_on"
                and event.velocity == 0
            )
        ):
            continue

        key = (
            event.channel,
            event.note,
        )

        if key not in last_note_on:
            continue

        open_notes = last_note_on[key]

        notes_to_close = [
            item
            for item in open_notes
            if item[0] != absolute_tick
        ]

        notes_to_keep = [
            item
            for item in open_notes
            if item[0] == absolute_tick
        ]

        for (
            start_tick,
            velocity,
            note_program,
        ) in notes_to_close:

            if (
                event.channel == channel
                and note_program == program
            ):
                notes.append(
                    [
                        "note",
                        start_tick,
                        absolute_tick - start_tick,
                        event.channel,
                        event.note,
                        velocity,
                    ]
                )

        if (
            notes_to_close
            and notes_to_keep
        ):
            last_note_on[key] = notes_to_keep
        else:
            del last_note_on[key]

    notes.sort(
        key=lambda event: event[1]
    )

    return notes

def resolve_stage3_instrument(
    midi_path,
    winning_instrument_index,
):
    """
    Reproduce UglyMIDI._load_instruments() and recover the exact
    (program, channel, raw_track) corresponding to a Stage 3
    instrument index.

    Stage 3 "track" is an index into UglyMIDI.instruments.
    It is NOT a raw MIDI track index.
    """

    midi_data = mido.MidiFile(
        filename=str(midi_path),
        clip=True,
    )

    # UglyMIDI converts every track from delta time to absolute
    # tick time before processing it.
    for track in midi_data.tracks:

        absolute_tick = 0

        for event in track:
            absolute_tick += event.time
            event.time = absolute_tick

    instrument_map = OrderedDict()
    stragglers = {}

    current_instrument = [0] * 16

    def get_instrument(
        program,
        channel,
        track_idx,
        create_new,
    ):
        key = (
            program,
            channel,
            track_idx,
        )

        if key in instrument_map:
            return key

        if (
            not create_new
            and (channel, track_idx) in stragglers
        ):
            return (
                channel,
                track_idx,
            )

        if create_new:

            if (
                channel,
                track_idx
            ) in stragglers:

                stragglers.pop(
                    (channel, track_idx)
                )

            instrument_map[key] = None

            return key

        stragglers[
            (channel, track_idx)
        ] = None

        return (
            channel,
            track_idx,
        )

    for track_idx, track in enumerate(
        midi_data.tracks
    ):

        last_note_on = {}

        for event in track:

            if event.type == "program_change":

                current_instrument[
                    event.channel
                ] = event.program

            elif (
                event.type == "note_on"
                and event.velocity > 0
            ):

                key = (
                    event.channel,
                    event.note,
                )

                last_note_on.setdefault(
                    key,
                    []
                ).append(
                    (
                        event.time,
                        event.velocity,
                    )
                )

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

                if key not in last_note_on:
                    continue

                open_notes = last_note_on[key]

                end_tick = event.time

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

                if notes_to_close:

                    program = current_instrument[
                        event.channel
                    ]

                    get_instrument(
                        program,
                        event.channel,
                        track_idx,
                        True,
                    )

                if (
                    notes_to_close
                    and notes_to_keep
                ):
                    last_note_on[key] = notes_to_keep
                else:
                    del last_note_on[key]

            elif event.type == "pitchwheel":

                program = current_instrument[
                    event.channel
                ]

                get_instrument(
                    program,
                    event.channel,
                    track_idx,
                    False,
                )

            elif event.type == "control_change":

                program = current_instrument[
                    event.channel
                ]

                get_instrument(
                    program,
                    event.channel,
                    track_idx,
                    False,
                )

    instrument_keys = list(
        instrument_map.keys()
    )

    if (
        winning_instrument_index < 0
        or winning_instrument_index
        >= len(instrument_keys)
    ):
        raise ValueError(
            f"Stage 3 instrument index "
            f"{winning_instrument_index} cannot be resolved. "
            f"UglyMIDI produced "
            f"{len(instrument_keys)} instruments."
        )

    return instrument_keys[
        winning_instrument_index
    ]

def read_midi(path):
    """
    Read a MIDI file and decode it into an opus.
    """

    with open(path, "rb") as f:
        midi_bytes = f.read()

    return MIDI.midi2opus(midi_bytes)


def absolute_score_track(opus_track):
    """
    Convert one opus track to score representation.

    Score representation uses absolute event times.
    """

    score = MIDI.opus2score([1000, opus_track])
    return score[1]


def remove_excluded_channel_notes(score_track):
    """
    Delete NOTE events on channels 9 and 15.

    This is deliberately done before any accompaniment
    merging, quantization, octave normalization, or duplicate
    suppression.

    Non-note events are retained.
    """

    filtered = []

    for event in score_track:
        if event[0] == "note":
            channel = event[3]

            if channel in EXCLUDED_ACCOMP_CHANNELS:
                continue

        filtered.append(event)

    return filtered


def merge_accompaniment_tracks(
    opus,
    winning_raw_track_index,
):
    merged = []

    for raw_track_index in range(
        len(opus) - 1
    ):

        if raw_track_index == winning_raw_track_index:
            continue

        score_track = MIDI.opus2score(
            [
                opus[0],
                opus[raw_track_index + 1],
            ]
        )[1]

        score_track = remove_excluded_channel_notes(
            score_track
        )

        merged.extend(
            score_track
        )

    merged.sort(
        key=lambda event: event[1]
    )

    return merged


def normalize_accompaniment_octaves(score_track):
    """
    Move every accompaniment note below C2 (MIDI 36) upward
    by octaves until it reaches MIDI 36 or higher.

    Only note pitch is modified.
    """

    normalized = []

    for event in score_track:
        event = list(event)

        if event[0] == "note":
            pitch = event[4]

            while pitch < MIN_ACCOMPANIMENT_PITCH:
                pitch += 12

            event[4] = pitch

        normalized.append(event)

    return normalized


def remove_duplicate_notes(score_track):
    """
    Remove identical note events after octave normalization.

    For note events, identity is:

        start time
        duration
        channel
        pitch
        velocity

    Non-note events are retained.

    This means two notes that become exactly identical after
    octave normalization collapse to one note.
    """

    result = []
    seen_notes = set()

    for event in score_track:
        if event[0] != "note":
            result.append(event)
            continue

        key = (
            event[1],  # absolute start time
            event[2],  # duration
            event[3],  # channel
            event[4],  # pitch
            event[5],  # velocity
        )

        if key in seen_notes:
            continue

        seen_notes.add(key)
        result.append(event)

    return result


def opus_track_to_score(opus_track, ticks_per_quarter):
    """
    Convert an opus track into score representation while
    preserving the actual MIDI ticks-per-quarter value.
    """

    score = MIDI.opus2score(
        [ticks_per_quarter, opus_track]
    )

    return score[1]


# ============================================================
# OUTPUT MIDI CONSTRUCTION
# ============================================================

def build_output_midi(
    midi_path,
    opus,
    winning_track_index,
):
    ticks_per_quarter = opus[0]

    (
        winning_program,
        winning_channel,
        winning_raw_track,
    ) = resolve_stage3_instrument(
        midi_path,
        winning_track_index,
    )

    track0 = extract_instrument_track(
        midi_path,
        winning_raw_track,
        winning_channel,
        winning_program,
    )

    accompaniment = merge_accompaniment_tracks(
        opus,
        winning_raw_track,
    )

    if APPLY_16TH_QUANTIZATION:
        accompaniment = apply_quantization(
            accompaniment,
            ticks_per_quarter,
        )

    accompaniment = normalize_accompaniment_octaves(
        accompaniment
    )

    accompaniment = remove_duplicate_notes(
        accompaniment
    )

    track0_opus = MIDI.score2opus(
        [
            ticks_per_quarter,
            track0,
        ]
    )[1]

    accompaniment_opus = MIDI.score2opus(
        [
            ticks_per_quarter,
            accompaniment,
        ]
    )[1]

    return [
        ticks_per_quarter,
        track0_opus,
        accompaniment_opus,
    ]
# ============================================================
# WORKER
# ============================================================

def process_one(item):
    """
    Worker function.

    Reads one source MIDI, determines whether it survives the
    Stage 4 criteria, constructs the output MIDI, and writes it.

    The worker returns only a compact result to the parent.
    """


    md5, winning_track_index, relative_path = item

    source_path = MIDI_ROOT / relative_path
    output_path = STAGE4_DIR / relative_path

    try:
        # ----------------------------------------------------
        # Read Stage 3 report
        #
        # The report is passed through the worker input only
        # as md5/path, so reports are loaded globally by the
        # parent and selection is resolved there.
        # ----------------------------------------------------

        opus = read_midi(source_path)

        output_opus = build_output_midi(
            source_path,
            opus,
            winning_track_index,
        )

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        midi_bytes = MIDI.opus2midi(output_opus)

        temporary_path = output_path.with_suffix(
            output_path.suffix + ".tmp"
        )

        with open(temporary_path, "wb") as f:
            f.write(midi_bytes)

        os.replace(
            temporary_path,
            output_path,
        )

        return {
            "status": "created",
            "md5": md5,
        }

    except Exception as exc:
        return {
            "status": "error",
            "md5": md5,
            "error": traceback.format_exc(),
        }


# ============================================================
# CHECKPOINTS
# ============================================================

def checkpoint_path(number):
    return CHECKPOINT_DIR / f"checkpoint{number}.json"


def write_checkpoint(
    checkpoint_number,
    completed_count,
    input_count,
    created_count,
    discarded_count,
    error_count,
    last_md5,
):
    """
    Write one compact checkpoint file.

    Checkpoints are independent files. Nothing is appended to
    one giant JSON document.
    """

    data = {
        "checkpoint_number": checkpoint_number,
        "completed_count": completed_count,
        "input_count": input_count,
        "created_count": created_count,
        "discarded_count": discarded_count,
        "error_count": error_count,
        "last_md5": last_md5,
        "score_threshold": SCORE_THRESHOLD,
        "mono_threshold": MONO_THRESHOLD,
        "pitch_mean_threshold": PITCH_MEAN_THRESHOLD,
        "apply_16th_quantization": APPLY_16TH_QUANTIZATION,
    }

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    path = checkpoint_path(checkpoint_number)
    temporary_path = path.with_suffix(".tmp")

    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=2,
        )

    os.replace(
        temporary_path,
        path,
    )


def find_latest_checkpoint():
    """
    Return the latest checkpoint number and its data.

    Returns None if no checkpoint exists.
    """

    if not CHECKPOINT_DIR.exists():
        return None

    checkpoints = []

    for path in CHECKPOINT_DIR.glob("checkpoint*.json"):
        name = path.stem

        try:
            number = int(name[len("checkpoint"):])
        except ValueError:
            continue

        checkpoints.append((number, path))

    if not checkpoints:
        return None

    number, path = max(
        checkpoints,
        key=lambda item: item[0],
    )

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    return number, data


def delete_checkpoints():
    """
    Delete all checkpoint<number>.json files.

    Called ONLY after successful completion.
    """

    if not CHECKPOINT_DIR.exists():
        return

    for path in CHECKPOINT_DIR.glob("checkpoint*.json"):
        try:
            path.unlink()
        except OSError:
            pass


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("STAGE 4 — MIDI CONSTRUCTION")
    print("=" * 72)

    print()
    print(f"Stage 3 reports : {STAGE3_REPORTS}")
    print(f"MIDI root       : {MIDI_ROOT}")
    print(f"Stage 4 output  : {STAGE4_DIR}")
    print()
    print("Selection:")
    print(f"  score >=      {SCORE_THRESHOLD}")
    print(f"  mono >=       {MONO_THRESHOLD}")
    print(f"  pitch_mean >= {PITCH_MEAN_THRESHOLD}")
    print()
    print("Accompaniment:")
    print(f"  excluded channels : {sorted(EXCLUDED_ACCOMP_CHANNELS)}")
    print(f"  min pitch         : C2 / {MIN_ACCOMPANIMENT_PITCH}")
    print(
        f"  16th quantization : "
        f"{'ENABLED' if APPLY_16TH_QUANTIZATION else 'DISABLED'}"
    )
    print()
    print(f"Workers         : {MAX_WORKERS}")
    print(f"Batch size      : {BATCH_SIZE}")
    print()

    # --------------------------------------------------------
    # Load Stage 3 reports
    # --------------------------------------------------------

    print("Loading Stage 3 reports...")

    with open(STAGE3_REPORTS, "r", encoding="utf-8") as f:
        reports = json.load(f)

    report_list = reports["reports"]

    print(f"Loaded reports: {len(report_list):,}")

    # --------------------------------------------------------
    # Build work list
    # --------------------------------------------------------

    work_items = []

    selected_count = 0
    discarded_count = 0

    for report in report_list:
        md5 = report.get("md5")
        relative_path = report.get("path")

        if not md5 or not relative_path:
            discarded_count += 1
            continue

        winner = select_winning_candidate(report)

        if winner is None:
            discarded_count += 1
            continue

        relative_path = Path(relative_path)

        winning_track_index = int(winner["track"])

        work_items.append(
            (
                md5,
                winning_track_index,
                relative_path,
            )
        )

        selected_count += 1

    input_count = len(work_items)

    print()
    print(f"Selected for Stage 4 : {selected_count:,}")
    print(f"Discarded            : {discarded_count:,}")
    print()

    if input_count == 0:
        print("Nothing to process.")
        return 0

    # --------------------------------------------------------
    # Resume handling
    # --------------------------------------------------------

    latest_checkpoint = find_latest_checkpoint()

    start_index = 0
    created_count = 0
    error_count = 0

    if latest_checkpoint is not None:

        checkpoint_number, checkpoint = latest_checkpoint

        completed_count = int(
            checkpoint.get("completed_count", 0)
        )

        checkpoint_input_count = int(
            checkpoint.get("input_count", -1)
        )

        last_md5 = checkpoint.get("last_md5")

        print(
            f"Found checkpoint{checkpoint_number}.json"
        )

        # The work list must be identical.
        if checkpoint_input_count != input_count:
            raise RuntimeError(
                "Checkpoint input_count does not match the "
                "current Stage 4 work list. Refusing to resume."
            )

        if completed_count > input_count:
            raise RuntimeError(
                "Checkpoint completed_count exceeds the "
                "current Stage 4 work list."
            )

        if completed_count > 0:
            expected_last_md5 = work_items[
                completed_count - 1
            ][0]

            if last_md5 != expected_last_md5:
                raise RuntimeError(
                    "Checkpoint last_md5 does not match the "
                    "current Stage 4 work-list ordering. "
                    "Refusing to resume."
                )

        start_index = completed_count

        created_count = int(
            checkpoint.get("created_count", 0)
        )

        error_count = int(
            checkpoint.get("error_count", 0)
        )

        print(
            f"Resuming from item "
            f"{start_index:,}/{input_count:,}"
        )
        print()

    else:
        print("No checkpoint found. Starting from beginning.")
        print()

    # --------------------------------------------------------
    # Process in bounded batches.
    #
    # A fresh ProcessPoolExecutor is created for every batch,
    # matching the memory-stability approach used in Stage 3.
    # --------------------------------------------------------

    completed_count = start_index

    try:

        while completed_count < input_count:

            batch_start = completed_count
            batch_end = min(
                batch_start + BATCH_SIZE,
                input_count,
            )

            batch = work_items[
                batch_start:batch_end
            ]

            checkpoint_number = (
                batch_end + BATCH_SIZE - 1
            ) // BATCH_SIZE

            print(
                f"Batch {checkpoint_number}: "
                f"{batch_start + 1:,}-{batch_end:,} "
                f"of {input_count:,}"
            )

            batch_created = 0
            batch_errors = 0

            # Fresh executor for every batch.
            with ProcessPoolExecutor(
                max_workers=MAX_WORKERS
            ) as executor:

                for result in executor.map(
                    process_one,
                    batch,
                ):

                    if result["status"] == "created":
                        batch_created += 1

                    elif result["status"] == "error":
                        batch_errors += 1

                        print(
                            f"\n  ERROR {result['md5']}\n"
                            f"{result['error']}"
                        )

            created_count += batch_created
            error_count += batch_errors

            completed_count = batch_end

            last_md5 = work_items[
                completed_count - 1
            ][0]

            write_checkpoint(
                checkpoint_number=checkpoint_number,
                completed_count=completed_count,
                input_count=input_count,
                created_count=created_count,
                discarded_count=discarded_count,
                error_count=error_count,
                last_md5=last_md5,
            )

            print(
                f"  created : {batch_created:,}"
            )
            print(
                f"  errors  : {batch_errors:,}"
            )
            print(
                f"  total   : {completed_count:,}/"
                f"{input_count:,}"
            )
            print()

    except KeyboardInterrupt:

        print()
        print("Interrupted.")
        print(
            "Checkpoint files have been preserved "
            "for resume."
        )
        print()

        return 1

    except Exception:

        print()
        print("Stage 4 failed.")
        print(
            "Checkpoint files have been preserved "
            "for resume."
        )
        print()

        traceback.print_exc()

        return 1

    # --------------------------------------------------------
    # Successful completion
    # --------------------------------------------------------

    print("=" * 72)
    print("STAGE 4 COMPLETE")
    print("=" * 72)

    print()
    print(f"Processed : {completed_count:,}")
    print(f"Created   : {created_count:,}")
    print(f"Discarded : {discarded_count:,}")
    print(f"Errors    : {error_count:,}")
    print()

    # ONLY NOW remove checkpoint files.
    delete_checkpoints()

    print("Checkpoint files removed.")
    print()
    print(f"Output: {STAGE4_DIR}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())