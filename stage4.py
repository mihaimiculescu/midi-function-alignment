#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Los Angeles MIDI Dataset
Stage 4 — Melody / Accompaniment Extraction

Stage 4 consumes Stage 3 output directly. No LAMDa metadata is used.

For every MIDI listed in candidates_<subdir>.json:

1. Find Stage-3 candidate tracks satisfying:
       pitch_mean >= 40
       melody_score >= 0.86
       monophonic_fraction >= 0.94
2. If none qualify, reject the MIDI.
3. Otherwise choose the qualifying track with the highest melody_score.
4. Write the WINNING track to output Track 1.
5. Write all NON-WINNING tracks, after channel-9 removal, merging,
   transposition, duplicate removal and A4 filtering, to output Track 0.
6. Copy tempo/key-signature/time-signature meta maps from all input
   tracks to output Track 0.

Output:
    Dataset/LAMDselection/selection_stage4/
        1/<file>.mid
        ...
        f/<file>.mid

    plus one independent manifest set per input subdirectory:
        candidates_1.txt
        candidates_1.json
        rejections_1.json
        summary_1.json
        ...

Checkpointing:
    checkpoint<number>.json

Each checkpoint contains only one newly completed contiguous batch.
Checkpoint files are deleted only after the COMPLETE Stage 4 run succeeds.

The parent process keeps at most WORKERS * 2 futures outstanding.
Workers return compact dictionaries only.
"""

import gc
import hashlib
import json
import os
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
MAX_PENDING = WORKERS * 2
CHECKPOINT_INTERVAL = 1000

INPUT_SUBDIRECTORIES = tuple("123456789abcdef")

PITCH_MEAN = 40.0
SCORE_THRESHOLD = 0.86
MONO_THRESHOLD = 0.94

GM_DRUM_CHANNEL = 9
C2 = 36
A4 = 69

QUANTIZE_OUTPUT = False


# ============================================================================
# JSON UTILITIES
# ============================================================================

def make_json_serializable(obj):
    if isinstance(obj, dict):
        return {
            make_json_serializable(k): make_json_serializable(v)
            for k, v in obj.items()
        }

    if isinstance(obj, (list, tuple)):
        return [make_json_serializable(v) for v in obj]

    if isinstance(obj, set):
        return [
            make_json_serializable(v)
            for v in sorted(obj)
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
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

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
                make_json_serializable(data),
                fh,
                indent=2,
                ensure_ascii=False,
            )
            fh.flush()
            os.fsync(fh.fileno())
            temporary = Path(fh.name)

        os.replace(temporary, path)

    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


# ============================================================================
# STAGE 3 INPUT
# ============================================================================

def load_stage3_candidates(subdir):
    path = STAGE3_DIR / f"candidates_{subdir}.json"

    if not path.is_file():
        raise FileNotFoundError(
            f"Stage 3 candidate file not found:\n{path}"
        )

    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    candidates = data.get("candidates")

    if not isinstance(candidates, list):
        raise RuntimeError(
            f"Invalid Stage 3 candidate file:\n{path}\n"
            "Expected a top-level 'candidates' list."
        )

    return candidates


def candidate_fingerprint(candidates):
    digest = hashlib.sha256()

    for candidate in candidates:
        md5 = str(candidate.get("md5", "")).lower()
        path = str(candidate.get("path", ""))

        digest.update(md5.encode("utf-8", errors="replace"))
        digest.update(b"\0")
        digest.update(path.encode("utf-8", errors="replace"))
        digest.update(b"\0")

    return digest.hexdigest()


# ============================================================================
# STAGE 4 QUALIFICATION
# ============================================================================

def get_stage3_analysis(candidate):
    stage3 = candidate.get("stage3")

    if not isinstance(stage3, dict):
        return []

    tracks = stage3.get("candidates")

    if not isinstance(tracks, list):
        return []

    return tracks


def qualifying_tracks(candidate):
    qualifying = []

    for track_report in get_stage3_analysis(candidate):
        try:
            pitch_mean = float(
                track_report.get("pitch_mean", 0.0)
            )
            melody_score = float(
                track_report.get("melody_score", 0.0)
            )
            monophonic_fraction = float(
                track_report.get("monophonic_fraction", 0.0)
            )
        except (TypeError, ValueError):
            continue

        if (
            pitch_mean >= PITCH_MEAN
            and melody_score >= SCORE_THRESHOLD
            and monophonic_fraction >= MONO_THRESHOLD
        ):
            qualifying.append(track_report)

    qualifying.sort(
        key=lambda item: float(
            item.get("melody_score", 0.0)
        ),
        reverse=True,
    )

    return qualifying


# ============================================================================
# MIDI
# ============================================================================

def import_mido():
    import mido
    return mido


def absolute_messages(track):
    result = []
    absolute_tick = 0

    for message in track:
        absolute_tick += int(message.time)
        result.append(
            (absolute_tick, message.copy())
        )

    return result


def note_message_type(message):
    return (
        not message.is_meta
        and message.type in ("note_on", "note_off")
    )


def is_note_on(message):
    return (
        not message.is_meta
        and message.type == "note_on"
        and int(message.velocity) > 0
    )


def is_note_off(message):
    if message.is_meta:
        return False

    if message.type == "note_off":
        return True

    return (
        message.type == "note_on"
        and int(message.velocity) == 0
    )


def extract_track_notes(track):
    """
    Extract complete physical-MIDI notes.

    Notes are paired FIFO per (channel, pitch).
    """

    absolute = absolute_messages(track)
    active = {}
    notes = []

    for tick, message in absolute:
        if not note_message_type(message):
            continue

        channel = int(getattr(message, "channel", 0))
        pitch = int(message.note)

        key = (channel, pitch)

        if is_note_on(message):
            active.setdefault(key, []).append(
                (tick, int(message.velocity))
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
# NON-WINNING TRACK TRANSFORMATION
# ============================================================================

def transpose_to_c2(pitch):
    pitch = int(pitch)

    while pitch < C2:
        pitch += 12

    return pitch


def merge_and_transform_notes(tracks, winning_track_index):
    """
    Merge every NON-winning physical track.

    Processing order:
      1. discard channel 9
      2. transpose every remaining pitch upward until >= C2
      3. for equal (start, pitch), retain only the longest-duration note
      4. remove every resulting pitch > A4

    Duplicate identity is deliberately:
        (start_tick, pitch)

    The longest-duration note wins. If durations tie, selection is
    deterministic by end tick, channel, velocity.
    """

    transformed = []

    for track_index, track in enumerate(tracks):
        if track_index == winning_track_index:
            continue

        notes = extract_track_notes(track)

        for note in notes:
            if note["channel"] == GM_DRUM_CHANNEL:
                continue

            transformed_pitch = transpose_to_c2(note["pitch"])

            transformed.append(
                {
                    "start": int(note["start"]),
                    "end": int(note["end"]),
                    "pitch": int(transformed_pitch),
                    "velocity": int(note["velocity"]),
                    "channel": int(note["channel"]),
                }
            )

    # ------------------------------------------------------------------
    # Duplicate removal AFTER transposition.
    #
    # Same pitch + same start:
    # retain longest duration.
    # ------------------------------------------------------------------
    best_by_identity = {}

    for note in transformed:
        identity = (
            note["start"],
            note["pitch"],
        )

        current = best_by_identity.get(identity)

        if current is None:
            best_by_identity[identity] = note
            continue

        current_duration = current["end"] - current["start"]
        new_duration = note["end"] - note["start"]

        if new_duration > current_duration:
            best_by_identity[identity] = note

        elif new_duration == current_duration:
            # Deterministic tie-breaking.
            current_key = (
                current["end"],
                current["channel"],
                current["velocity"],
            )
            new_key = (
                note["end"],
                note["channel"],
                note["velocity"],
            )

            if new_key > current_key:
                best_by_identity[identity] = note

    unique = list(best_by_identity.values())

    # ------------------------------------------------------------------
    # Remove everything above A4 AFTER duplicate removal.
    # A4 itself (69) is retained.
    # ------------------------------------------------------------------
    filtered = [
        note
        for note in unique
        if note["pitch"] <= A4
    ]

    filtered.sort(
        key=lambda note: (
            note["start"],
            note["end"],
            note["pitch"],
            note["channel"],
            note["velocity"],
        )
    )

    return filtered


# ============================================================================
# WINNING TRACK
# ============================================================================

def extract_winning_track_notes(track):
    """
    Extract the winning physical track.

    Channel 9 notes are removed.
    No transposition is performed.
    No duplicate removal is performed.
    No A4 filtering is performed.
    """

    notes = extract_track_notes(track)

    return [
        note
        for note in notes
        if note["channel"] != GM_DRUM_CHANNEL
    ]


# ============================================================================
# TEMPO / KEY / TIME-SIGNATURE MAPS
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
    Collect tempo, key-signature and time-signature meta messages from
    ALL physical input tracks, preserving their absolute tick positions.
    """

    collected = []

    for track_index, track in enumerate(tracks):
        for tick, message in absolute_messages(track):
            if (
                message.is_meta
                and message.type in MAP_MESSAGE_TYPES
            ):
                collected.append(
                    (
                        int(tick),
                        int(track_index),
                        message.copy(),
                    )
                )

    collected.sort(
        key=lambda item: (
            item[0],
            item[1],
        )
    )

    return collected


# ============================================================================
# OUTPUT MIDI CONSTRUCTION
# ============================================================================

def notes_to_messages(notes):
    """
    Convert note dictionaries to absolute-tick note events.

    At equal ticks note-offs precede note-ons.
    """

    messages = []

    for note in notes:
        start = int(note["start"])
        end = int(note["end"])

        if end <= start:
            continue

        pitch = int(note["pitch"])

        if not 0 <= pitch <= 127:
            continue

        velocity = max(
            0,
            min(127, int(note["velocity"]))
        )

        channel = max(
            0,
            min(15, int(note["channel"]))
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

    messages.sort(
        key=lambda item: (
            item[0],
            0 if item[1]["kind"] == "off" else 1,
        )
    )

    return messages


def build_note_track(mido, notes):
    track = mido.MidiTrack()
    previous_tick = 0

    for tick, data in notes_to_messages(notes):
        tick = int(tick)
        delta = tick - previous_tick

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

        track.append(message)
        previous_tick = tick

    return track


def build_track_zero(
    mido,
    accompaniment_notes,
    map_messages,
):
    """
    Track 0 contains:
      - tempo map
      - key-signature map
      - time-signature map
      - processed NON-winning/accompaniment notes

    This is intentional: the winning melody is Track 1.
    """

    events = []

    # Meta maps.
    for tick, source_track, message in map_messages:
        events.append(
            (
                int(tick),
                0,
                int(source_track),
                message.copy(),
            )
        )

    # Accompaniment notes.
    for tick, data in notes_to_messages(accompaniment_notes):
        if data["kind"] == "off":
            message = mido.Message(
                "note_off",
                channel=data["channel"],
                note=data["pitch"],
                velocity=0,
                time=0,
            )
            priority = 1
        else:
            message = mido.Message(
                "note_on",
                channel=data["channel"],
                note=data["pitch"],
                velocity=data["velocity"],
                time=0,
            )
            priority = 2

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
        tick = int(tick)
        delta = tick - previous_tick

        if delta < 0:
            raise RuntimeError(
                "Negative Track 0 delta time."
            )

        output.append(
            message.copy(time=delta)
        )

        previous_tick = tick

    return output


def build_track_one(mido, winning_notes):
    """
    Track 1 contains ONLY the winning melody notes.
    """

    return build_note_track(
        mido,
        winning_notes,
    )


# ============================================================================
# QUANTIZATION HOOK
# ============================================================================

def quantize_output(notes):
    """
    Future quantization hook.

    Intentionally does nothing for now.
    """

    return notes


# ============================================================================
# OUTPUT PATH
# ============================================================================

def output_midi_path(subdir, input_path):
    output_directory = STAGE4_DIR / str(subdir)
    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    return output_directory / Path(input_path).name


def atomic_midi_save(midi, output_path):
    output_path = Path(output_path)
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
            temporary = Path(fh.name)

        midi.save(filename=str(temporary))
        os.replace(temporary, output_path)

    finally:
        if temporary is not None and temporary.exists():
            try:
                temporary.unlink()
            except OSError:
                pass


# ============================================================================
# PROCESS ONE MIDI
# ============================================================================

def process_midi(candidate, subdir):
    md5 = str(
        candidate.get("md5", "")
    ).lower()

    input_path = Path(
        str(candidate.get("path", ""))
    )

    result_base = {
        "md5": md5,
        "path": str(input_path),
    }

    # ------------------------------------------------------------------
    # Qualification comes entirely from Stage 3 JSON.
    # ------------------------------------------------------------------
    qualifying = qualifying_tracks(candidate)

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
    except (KeyError, TypeError, ValueError):
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
                "reason": "midi_file_not_found",
            }
        )
        return result_base

    mido = import_mido()
    midi = None

    try:
        midi = mido.MidiFile(
            str(input_path)
        )

        tracks = midi.tracks

        if (
            winning_track_index < 0
            or winning_track_index >= len(tracks)
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

        # --------------------------------------------------------------
        # Winning melody -> OUTPUT TRACK 1
        # --------------------------------------------------------------
        winning_notes = extract_winning_track_notes(
            tracks[winning_track_index]
        )

        # --------------------------------------------------------------
        # Maps -> OUTPUT TRACK 0
        # --------------------------------------------------------------
        map_messages = collect_map_messages(tracks)

        # --------------------------------------------------------------
        # All non-winning tracks -> OUTPUT TRACK 0
        # --------------------------------------------------------------
        accompaniment_notes = merge_and_transform_notes(
            tracks,
            winning_track_index,
        )

        if QUANTIZE_OUTPUT:
            accompaniment_notes = quantize_output(
                accompaniment_notes
            )

        # --------------------------------------------------------------
        # Build output MIDI.
        # --------------------------------------------------------------
        output_midi = mido.MidiFile(
            type=1,
            ticks_per_beat=midi.ticks_per_beat,
        )

        # IMPORTANT:
        # Track 0 = maps + processed non-winning tracks.
        # Track 1 = winning melody.
        output_track_0 = build_track_zero(
            mido,
            accompaniment_notes,
            map_messages,
        )

        output_track_1 = build_track_one(
            mido,
            winning_notes,
        )

        output_midi.tracks.append(output_track_0)
        output_midi.tracks.append(output_track_1)

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
                "output_path": str(output_path),
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
                    int(len(qualifying)),
                "winning_notes":
                    int(len(winning_notes)),
                "track0_notes":
                    int(len(accompaniment_notes)),
                "input_tracks":
                    int(len(tracks)),
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
                    + type(exc).__name__,
                "error": str(exc),
            }
        )
        return result_base

    finally:
        midi = None
        gc.collect()


# ============================================================================
# WORKER
# ============================================================================

def stage4_worker(candidate, subdir):
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
                + type(exc).__name__,
            "error": str(exc),
        }
    finally:
        gc.collect()


# ============================================================================
# CHECKPOINTS
# ============================================================================

def checkpoint_paths():
    paths = []

    STAGE4_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    for path in STAGE4_DIR.glob("checkpoint*.json"):
        suffix = path.stem[len("checkpoint"):]

        if suffix.isdigit():
            paths.append(path)

    return sorted(
        paths,
        key=lambda path: int(
            path.stem[len("checkpoint"):]
        ),
    )


def next_checkpoint_number():
    paths = checkpoint_paths()

    if not paths:
        return 1

    return (
        int(
            paths[-1].stem[len("checkpoint"):]
        )
        + 1
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
    checkpoint_path = (
        STAGE4_DIR
        / f"checkpoint{checkpoint_number}.json"
    )

    checkpoint = {
        "stage": "4",
        "subdirectory": str(subdir),
        "input_count": int(input_count),
        "input_fingerprint": str(input_fingerprint),
        "completed_start": int(completed_start),
        "completed_end": int(completed_end),
        "result_count": int(len(results)),
        "results": results,
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
    relevant = []

    for path in checkpoint_paths():
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as fh:
            checkpoint = json.load(fh)

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
                f"Checkpoint input count mismatch:\n{path}"
            )

        if (
            checkpoint.get("input_fingerprint")
            != input_fingerprint
        ):
            raise RuntimeError(
                f"Checkpoint input fingerprint mismatch:\n{path}"
            )

        relevant.append(
            (path, checkpoint)
        )

    relevant.sort(
        key=lambda item: int(
            item[1]["completed_start"]
        )
    )

    reconstructed = []
    expected_start = 0

    for path, checkpoint in relevant:
        start = int(
            checkpoint["completed_start"]
        )
        end = int(
            checkpoint["completed_end"]
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

        if len(results) != end - start:
            raise RuntimeError(
                f"Checkpoint result count mismatch:\n{path}"
            )

        reconstructed.extend(results)
        expected_start = end

    # Verify checkpoint order against the Stage 3 input.
    for index, result in enumerate(reconstructed):
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


def delete_all_checkpoints():
    for path in checkpoint_paths():
        try:
            path.unlink()
        except OSError:
            pass


# ============================================================================
# FINAL OUTPUT MANIFESTS
# ============================================================================

def output_manifest_paths(subdir):
    return {
        "candidates_txt":
            STAGE4_DIR / f"candidates_{subdir}.txt",

        "candidates_json":
            STAGE4_DIR / f"candidates_{subdir}.json",

        "rejections_json":
            STAGE4_DIR / f"rejections_{subdir}.json",

        "summary_json":
            STAGE4_DIR / f"summary_{subdir}.json",
    }


def write_final_outputs(
    subdir,
    candidates,
    results,
):
    survivors = []
    rejections = []
    rejection_counts = {}

    for result in results:
        if result.get("kind") == "candidate":
            survivors.append(result)
        else:
            rejections.append(result)

            reason = str(
                result.get(
                    "reason",
                    "unknown",
                )
            )

            rejection_counts[reason] = (
                rejection_counts.get(reason, 0)
                + 1
            )

    paths = output_manifest_paths(subdir)

    # candidates_<subdir>.txt
    with open(
        paths["candidates_txt"],
        "w",
        encoding="utf-8",
    ) as fh:
        for record in survivors:
            fh.write(
                str(record["output_path"])
                + "\n"
            )

    # candidates_<subdir>.json
    atomic_json_write(
        paths["candidates_json"],
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",
            "stage": "4",
            "input_subdirectory": str(subdir),
            "input":
                str(
                    STAGE3_DIR
                    / f"candidates_{subdir}.json"
                ),
            "midi_root":
                str(MIDI_ROOT / str(subdir)),
            "output_root":
                str(STAGE4_DIR),
            "workers": int(WORKERS),
            "pitch_mean_threshold":
                float(PITCH_MEAN),
            "melody_score_threshold":
                float(SCORE_THRESHOLD),
            "monophonic_fraction_threshold":
                float(MONO_THRESHOLD),
            "quantize_output":
                bool(QUANTIZE_OUTPUT),
            "candidate_count":
                int(len(survivors)),
            "candidates":
                survivors,
        },
    )

    # rejections_<subdir>.json
    atomic_json_write(
        paths["rejections_json"],
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",
            "stage": "4",
            "input_subdirectory": str(subdir),
            "rejection_count":
                int(len(rejections)),
            "rejection_reasons":
                rejection_counts,
            "rejections":
                rejections,
        },
    )

    # summary_<subdir>.json
    atomic_json_write(
        paths["summary_json"],
        {
            "dataset":
                "Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA",
            "stage": "4",
            "input_subdirectory": str(subdir),
            "input_count":
                int(len(candidates)),
            "survivor_count":
                int(len(survivors)),
            "rejected_count":
                int(len(rejections)),
            "rejection_reasons":
                rejection_counts,
            "thresholds": {
                "pitch_mean":
                    float(PITCH_MEAN),
                "melody_score":
                    float(SCORE_THRESHOLD),
                "monophonic_fraction":
                    float(MONO_THRESHOLD),
            },
            "quantize_output":
                bool(QUANTIZE_OUTPUT),
        },
    )

    return len(survivors), len(rejections)


# ============================================================================
# PROCESS ONE INPUT SUBDIRECTORY
# ============================================================================

def process_subdirectory(subdir):
    candidates = load_stage3_candidates(subdir)

    input_count = len(candidates)
    fingerprint = candidate_fingerprint(candidates)

    results = load_checkpoints(
        subdir,
        candidates,
        fingerprint,
    )

    completed = len(results)

    print()
    print("=" * 78)
    print(f"STAGE 4 — SUBDIRECTORY {subdir}")
    print("=" * 78)
    print(f"Stage 3 input      : {input_count:,}")
    print(f"Already completed  : {completed:,}")
    print(f"Remaining          : {input_count - completed:,}")
    print(f"Workers            : {WORKERS}")
    print(f"Max pending        : {MAX_PENDING}")
    print(f"Checkpoint interval: {CHECKPOINT_INTERVAL}")
    print()

    if completed < input_count:
        pending = {}
        ready = {}

        next_submit = completed
        next_commit = completed

        checkpoint_number = next_checkpoint_number()
        uncheckpointed_results = []

        with ProcessPoolExecutor(
            max_workers=WORKERS
        ) as executor:

            # ----------------------------------------------------------
            # Initial bounded submission.
            # ----------------------------------------------------------
            while (
                next_submit < input_count
                and len(pending) < MAX_PENDING
            ):
                future = executor.submit(
                    stage4_worker,
                    candidates[next_submit],
                    subdir,
                )

                pending[future] = next_submit
                next_submit += 1

            # ----------------------------------------------------------
            # Consume and refill.
            # ----------------------------------------------------------
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
                                "future_error:"
                                + type(exc).__name__,
                            "error": str(exc),
                        }

                    ready[original_index] = result

                    # Refill immediately.
                    if next_submit < input_count:
                        future2 = executor.submit(
                            stage4_worker,
                            candidates[next_submit],
                            subdir,
                        )

                        pending[future2] = next_submit
                        next_submit += 1

                # ------------------------------------------------------
                # Commit only a contiguous completed prefix.
                # ------------------------------------------------------
                while next_commit in ready:
                    result = ready.pop(next_commit)

                    results.append(result)
                    uncheckpointed_results.append(result)

                    next_commit += 1

                    if (
                        len(uncheckpointed_results)
                        >= CHECKPOINT_INTERVAL
                    ):
                        start = (
                            next_commit
                            - len(uncheckpointed_results)
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
                - len(uncheckpointed_results)
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
    # Reconstruct from checkpoints / in-memory results.
    # --------------------------------------------------------------
    results = load_checkpoints(
        subdir,
        candidates,
        fingerprint,
    )

    if len(results) != input_count:
        raise RuntimeError(
            f"Stage 4 incomplete for subdirectory {subdir}: "
            f"{len(results)} / {input_count}"
        )

    survivors, rejections = write_final_outputs(
        subdir,
        candidates,
        results,
    )

    print()
    print(
        f"Subdirectory {subdir} complete: "
        f"{survivors:,} survivors, "
        f"{rejections:,} rejections."
    )

    return survivors, rejections


# ============================================================================
# MAIN
# ============================================================================

def main():
    STAGE4_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    total_input = 0
    total_survivors = 0
    total_rejections = 0

    try:
        for subdir in INPUT_SUBDIRECTORIES:
            candidates = load_stage3_candidates(subdir)

            total_input += len(candidates)

            survivors, rejections = process_subdirectory(
                subdir
            )

            total_survivors += survivors
            total_rejections += rejections

        # --------------------------------------------------------------
        # Only after EVERY subdirectory has completed successfully:
        # delete ALL checkpoint files.
        # --------------------------------------------------------------
        delete_all_checkpoints()

        print()
        print("=" * 78)
        print("STAGE 4 COMPLETE")
        print("=" * 78)
        print(f"Input files     : {total_input:,}")
        print(f"Survivors       : {total_survivors:,}")
        print(f"Rejections      : {total_rejections:,}")
        print(f"Workers         : {WORKERS}")
        print()
        print("All checkpoint files have been deleted.")
        print("=" * 78)

    except Exception:
        print()
        print("=" * 78)
        print("STAGE 4 FAILED")
        print("=" * 78)
        print("Checkpoint files have been retained.")
        print("=" * 78)
        raise


if __name__ == "__main__":
    main()
