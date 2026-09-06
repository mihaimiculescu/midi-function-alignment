#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Los Angeles MIDI Dataset
Stage 4 — melody-track selection + melodic-contamination-aware accompaniment.

Input:
    Dataset/LAMDselection/selection_stage3/candidates_<subdir>.json

For each Stage-3 survivor:
    1. Inspect the Stage-3 physical-track candidates.
    2. Keep the file iff at least one candidate satisfies:
           pitch_mean >= 40
           melody_score >= 0.86
           monophonic_fraction >= 0.94
    3. Select the qualifying candidate with the highest melody_score.
    4. Exclude OTHER Stage-3 candidates that also satisfy those melody
       criteria. They are treated as strongly melodic material (for example
       counter-melodies or melodic ostinatos), rather than accompaniment.
    5. Merge all remaining physical tracks into one accompaniment track.
    6. Write a two-track MIDI:
           track 0 = normalized accompaniment + tempo/key/time maps
           track 1 = winning physical track, with channel 9 notes removed

The new accompaniment selection deliberately does NOT reject short notes,
high note density, low polyphony, fast harmonic changes, arpeggios, or
rhythmic chord attacks. The purpose is to remove strongly melodic competing
tracks without sacrificing harmonic agility.

No LAMDa metadata is used.

Checkpointing:
    checkpoint<number>.json
    Each file contains only one completed contiguous batch.
    Checkpoints are deleted only after the entire Stage-4 run succeeds.

Memory/I/O:
    24 worker processes by default.
    At most WORKERS*2 futures are outstanding.
    Workers return compact result dictionaries; MIDI data is not returned
    to the parent process.
"""

import gc
import hashlib
import json
import os
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import mido


# ============================================================================
# PATHS
# ============================================================================

STAGE3_DIR = Path("Dataset/LAMDselection/selection_stage3")
STAGE4_DIR = Path("Dataset/LAMDselection/selection_stage4")
MIDI_ROOT = Path(
    "Dataset/Los-Angeles-MIDI-Dataset-Ver-4-0-CC-BY-NC-SA/MIDIs"
)

INPUT_SUBDIRECTORIES = tuple("123456789abcdef")

# ============================================================================
# CONFIGURATION
# ============================================================================

WORKERS = 24
MAX_PENDING = WORKERS * 2
CHECKPOINT_INTERVAL = 1000

PITCH_MEAN = 40
SCORE_THRESHOLD = 0.86
MONO_THRESHOLD = 0.94

GM_DRUM_CHANNEL = 9

# Deliberately does nothing for now. This is the extension point requested
# for future developments.
QUANTIZE_OUTPUT = False


def quantize_output(notes):
    """Future quantization hook. Deliberately a no-op for now."""
    return notes


# ============================================================================
# JSON / CHECKPOINT UTILITIES
# ============================================================================

def atomic_json_write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
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
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
            temporary_path = Path(fh.name)

        os.replace(temporary_path, path)

    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


def checkpoint_paths():
    paths = []

    for path in STAGE4_DIR.glob("checkpoint*.json"):
        suffix = path.stem[len("checkpoint"):]
        if suffix.isdigit():
            paths.append(path)

    return sorted(
        paths,
        key=lambda path: int(path.stem[len("checkpoint"):]),
    )


def next_checkpoint_number():
    paths = checkpoint_paths()

    if not paths:
        return 1

    return int(paths[-1].stem[len("checkpoint"):]) + 1


def input_fingerprint(candidates):
    digest = hashlib.sha256()

    for candidate in candidates:
        digest.update(
            str(candidate.get("md5", "")).lower().encode(
                "utf-8", errors="replace"
            )
        )
        digest.update(b"\0")

        digest.update(
            str(candidate.get("path", "")).encode(
                "utf-8", errors="replace"
            )
        )
        digest.update(b"\0")

        # The Stage-3 track candidates are part of the effective input to
        # Stage 4. Include the relevant values so a changed Stage-3 file
        # cannot accidentally reuse an old checkpoint.
        stage3 = candidate.get("stage3", {})
        for track in stage3.get("candidates", []):
            digest.update(str(track.get("track", "")).encode())
            digest.update(b"\0")
            digest.update(str(track.get("pitch_mean", "")).encode())
            digest.update(b"\0")
            digest.update(str(track.get("melody_score", "")).encode())
            digest.update(b"\0")
            digest.update(
                str(track.get("monophonic_fraction", "")).encode()
            )
            digest.update(b"\0")

    return digest.hexdigest()


def write_checkpoint(
    checkpoint_number,
    subdir,
    input_count,
    fingerprint,
    completed_start,
    completed_end,
    results,
):
    path = STAGE4_DIR / f"checkpoint{checkpoint_number}.json"

    payload = {
        "stage": "4",
        "subdirectory": str(subdir),
        "input_count": int(input_count),
        "input_fingerprint": fingerprint,
        "completed_start": int(completed_start),
        "completed_end": int(completed_end),
        "result_count": len(results),
        "results": results,
    }

    atomic_json_write(path, payload)

    print(
        f"  checkpoint{checkpoint_number}.json "
        f"[{completed_start + 1:,}..{completed_end:,}]"
    )


def load_checkpoints(subdir, candidates, fingerprint):
    relevant = []

    for path in checkpoint_paths():
        with open(path, "r", encoding="utf-8") as fh:
            checkpoint = json.load(fh)

        if str(checkpoint.get("subdirectory", "")) != str(subdir):
            continue

        if int(checkpoint.get("input_count", -1)) != len(candidates):
            raise RuntimeError(
                f"Checkpoint input-count mismatch:\n{path}"
            )

        if checkpoint.get("input_fingerprint") != fingerprint:
            raise RuntimeError(
                f"Checkpoint input fingerprint mismatch:\n{path}"
            )

        relevant.append((path, checkpoint))

    relevant.sort(
        key=lambda item: int(item[1]["completed_start"])
    )

    reconstructed = []
    expected_start = 0

    for path, checkpoint in relevant:
        start = int(checkpoint["completed_start"])
        end = int(checkpoint["completed_end"])
        results = checkpoint.get("results", [])

        if start != expected_start:
            raise RuntimeError(
                f"Checkpoint sequence gap/overlap for subdirectory "
                f"{subdir}: {path}"
            )

        if len(results) != end - start:
            raise RuntimeError(
                f"Checkpoint result-count mismatch:\n{path}"
            )

        for offset, result in enumerate(results):
            index = start + offset

            if index >= len(candidates):
                raise RuntimeError(
                    f"Checkpoint extends beyond input list:\n{path}"
                )

            expected_md5 = str(
                candidates[index].get("md5", "")
            ).lower()

            actual_md5 = str(
                result.get("md5", "")
            ).lower()

            if actual_md5 != expected_md5:
                raise RuntimeError(
                    "Checkpoint ordering/MD5 mismatch:\n"
                    f"  checkpoint: {path}\n"
                    f"  index: {index}\n"
                    f"  expected: {expected_md5}\n"
                    f"  actual:   {actual_md5}"
                )

        reconstructed.extend(results)
        expected_start = end

    return reconstructed


def delete_all_checkpoints():
    for path in checkpoint_paths():
        path.unlink()
        print(f"  deleted: {path}")


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
            "Expected top-level 'candidates' list."
        )

    normalized = []

    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise RuntimeError(
                f"Invalid Stage 3 candidate in {path}"
            )

        if not candidate.get("path"):
            raise RuntimeError(
                f"Stage 3 candidate has no path in {path}"
            )

        item = dict(candidate)
        item["path"] = str(item["path"])
        item["md5"] = str(
            item.get("md5", Path(item["path"]).stem)
        ).lower()

        normalized.append(item)

    return normalized


# ============================================================================
# MIDI LOW-LEVEL UTILITIES
# ============================================================================

META_TYPES_TO_TRACK0 = {
    "set_tempo",
    "key_signature",
    "time_signature",
}


def absolute_track_events(track):
    """Return (absolute_tick, sequence_number, message) tuples."""
    absolute = 0
    result = []

    for sequence_number, message in enumerate(track):
        absolute += int(message.time)
        result.append((absolute, sequence_number, message))

    return result


def note_events_from_track(track):
    """
    Extract complete notes from one mido track.

    Returns dictionaries:
        start, end, pitch, velocity, channel, order

    Channel 9 notes are omitted here because Stage 4 removes them.
    """
    absolute_events = absolute_track_events(track)

    active = {}
    notes = []
    order = 0

    for tick, _sequence_number, message in absolute_events:
        if message.type == "note_on" and message.velocity > 0:
            if message.channel == GM_DRUM_CHANNEL:
                continue

            key = (message.channel, message.note)
            active.setdefault(key, []).append(
                (tick, message.velocity, order)
            )
            order += 1

        elif message.type == "note_off" or (
            message.type == "note_on" and message.velocity == 0
        ):
            if message.channel == GM_DRUM_CHANNEL:
                continue

            key = (message.channel, message.note)
            starts = active.get(key)

            if not starts:
                continue

            start_tick, velocity, start_order = starts.pop()

            if tick <= start_tick:
                continue

            notes.append(
                {
                    "start": int(start_tick),
                    "end": int(tick),
                    "pitch": int(message.note),
                    "velocity": int(velocity),
                    "channel": int(message.channel),
                    "order": start_order,
                }
            )

    return notes


def note_events_from_tracks(midi, track_indexes):
    notes = []

    for track_index in track_indexes:
        notes.extend(note_events_from_track(midi.tracks[track_index]))

    return notes


def winning_track_notes(track):
    """
    Extract all complete non-channel-9 notes from the selected physical track.
    No other Stage-4 transformations are applied to these notes.
    """
    return note_events_from_track(track)


def normalize_accompaniment(notes):
    """
    Apply the Stage-4 accompaniment rules:

      1. transpose pitches below C2 upward by octaves until >= C2
      2. remove pitches above A4
      3. for duplicate (start, pitch), retain the longest note only
    """
    normalized = []

    for note in notes:
        pitch = int(note["pitch"])

        while pitch < 36:  # C2
            pitch += 12

        if pitch > 69:  # above A4
            continue

        item = dict(note)
        item["pitch"] = pitch
        normalized.append(item)

    # Longest duration wins for identical pitch + start time.
    # Stable tie handling preserves the first encountered note.
    best = {}

    for note in normalized:
        key = (note["start"], note["pitch"])
        duration = note["end"] - note["start"]

        previous = best.get(key)

        if previous is None:
            best[key] = note
        else:
            previous_duration = (
                previous["end"] - previous["start"]
            )

            if duration > previous_duration:
                best[key] = note

    result = list(best.values())

    result.sort(
        key=lambda note: (
            note["start"],
            note["pitch"],
            note["order"],
        )
    )

    return result


def make_note_message(note):
    return mido.Message(
        "note_on",
        channel=note["channel"],
        note=note["pitch"],
        velocity=note["velocity"],
        time=0,
    )


def make_note_off_message(note):
    return mido.Message(
        "note_off",
        channel=note["channel"],
        note=note["pitch"],
        velocity=0,
        time=0,
    )


def absolute_events_to_track(events):
    """
    Convert [(absolute_tick, order, mido_message), ...] to a MidiTrack
    with delta times.
    """
    events.sort(key=lambda item: (item[0], item[1]))

    track = mido.MidiTrack()
    previous_tick = 0

    for absolute_tick, _order, message in events:
        delta = int(absolute_tick) - previous_tick

        if delta < 0:
            raise RuntimeError("Negative MIDI delta encountered.")

        track.append(message.copy(time=delta))
        previous_tick = int(absolute_tick)

    track.append(mido.MetaMessage("end_of_track", time=0))

    return track


def notes_to_track(notes, extra_events=None):
    """
    Build a MIDI track from absolute-tick note events and optional meta events.
    """
    events = []
    order = 0

    if extra_events:
        for tick, sequence_number, message in extra_events:
            events.append(
                (int(tick), order, message.copy(time=0))
            )
            order += 1

    for note in notes:
        start = int(note["start"])
        end = int(note["end"])

        events.append(
            (
                start,
                order,
                make_note_message(note),
            )
        )
        order += 1

        events.append(
            (
                end,
                order,
                make_note_off_message(note),
            )
        )
        order += 1

    return absolute_events_to_track(events)


# ============================================================================
# MIDI TRANSFORMATION
# ============================================================================

def build_stage4_midi(input_path, winning_track_index, excluded_melodic_tracks):
    """
    Read one source MIDI and build the Stage-4 output.

    Output:
        track 0 = accompaniment + all tempo/key/time maps
        track 1 = winning melody track
    """
    midi = mido.MidiFile(filename=str(input_path), clip=True)

    if winning_track_index < 0 or winning_track_index >= len(midi.tracks):
        raise RuntimeError(
            f"Winning track {winning_track_index} does not exist in "
            f"{input_path}; MIDI has {len(midi.tracks)} tracks."
        )

    # Collect tempo/key/time signature maps from every input track.
    meta_events = []

    for track_index, track in enumerate(midi.tracks):
        for tick, sequence_number, message in absolute_track_events(track):
            if message.type in META_TYPES_TO_TRACK0:
                meta_events.append(
                    (
                        int(tick),
                        track_index,
                        sequence_number,
                        message.copy(time=0),
                    )
                )

    # Track 1: selected physical track, channel 9 removed.
    melody_notes = winning_track_notes(
        midi.tracks[winning_track_index]
    )

    # Merge all physical tracks except:
    #   1. the winning melody track
    #   2. other Stage-3 tracks that independently satisfy the melody
    #      criteria and are therefore treated as competing melodic material
    #
    # Channel 9 is discarded by note_events_from_track().
    accompaniment_track_indexes = [
        index
        for index in range(len(midi.tracks))
        if (
            index != winning_track_index
            and index not in excluded_melodic_tracks
        )
    ]

    accompaniment_notes = note_events_from_tracks(
        midi,
        accompaniment_track_indexes,
    )

    accompaniment_notes = normalize_accompaniment(
        accompaniment_notes
    )

    # Requested future quantization hook: after duplicate removal.
    if QUANTIZE_OUTPUT:
        accompaniment_notes = quantize_output(
            accompaniment_notes
        )

    track0_meta = [
        (tick, track_index, message)
        for tick, track_index, _sequence_number, message
        in meta_events
    ]

    track0 = notes_to_track(
        accompaniment_notes,
        extra_events=track0_meta,
    )

    track1 = notes_to_track(
        melody_notes
    )

    output = mido.MidiFile(
        type=1,
        ticks_per_beat=midi.ticks_per_beat,
    )
    output.tracks.append(track0)
    output.tracks.append(track1)

    return output, len(melody_notes), len(accompaniment_notes)


def atomic_midi_write(midi, output_path):
    """
    Atomically write a MIDI file.

    The temporary file is created in the destination directory so os.replace()
    remains on the same filesystem.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary_path = None

    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=output_path.parent,
            delete=False,
            prefix=f".{output_path.name}.",
            suffix=".tmp",
        ) as fh:
            temporary_path = Path(fh.name)

        midi.save(filename=str(temporary_path))
        os.replace(temporary_path, output_path)

    finally:
        if temporary_path is not None and temporary_path.exists():
            try:
                temporary_path.unlink()
            except OSError:
                pass


# ============================================================================
# STAGE 4 FILE DECISION
# ============================================================================

def is_melody_candidate(track):
    """
    Return True when a Stage-3 track independently satisfies the existing
    Stage-4 melody-selection criteria.
    """
    try:
        pitch_mean = float(track["pitch_mean"])
        melody_score = float(track["melody_score"])
        monophonic_fraction = float(
            track["monophonic_fraction"]
        )
    except (KeyError, TypeError, ValueError):
        return False

    return (
        pitch_mean >= PITCH_MEAN
        and melody_score >= SCORE_THRESHOLD
        and monophonic_fraction >= MONO_THRESHOLD
    )


def classify_stage3_tracks(candidate):
    """
    Return:
        (winning_track_index, excluded_melodic_track_indexes, reason)

    The winning track is the highest-melody-score qualifying Stage-3 track.
    Every OTHER qualifying Stage-3 track is excluded from the accompaniment.

    Stage 3 retains the top eight melody-like tracks per MIDI. Therefore these
    qualifying non-winners form a deliberate melodic-contamination watchlist.
    No duration, density, or polyphony rule is applied here.
    """
    stage3 = candidate.get("stage3")

    if not isinstance(stage3, dict):
        return None, set(), "missing_stage3_analysis"

    tracks = stage3.get("candidates")

    if not isinstance(tracks, list):
        return None, set(), "missing_stage3_candidates"

    qualifying = []

    for track in tracks:
        if not isinstance(track, dict):
            continue

        if not is_melody_candidate(track):
            continue

        try:
            track_index = int(track["track"])
            melody_score = float(track["melody_score"])
        except (KeyError, TypeError, ValueError):
            continue

        qualifying.append(track)

    if not qualifying:
        return None, set(), "no_qualifying_track"

    # Highest melody_score wins. Stage 4 deliberately performs the selection
    # itself rather than depending on Stage-3 ordering.
    qualifying.sort(
        key=lambda track: (
            float(track["melody_score"]),
            float(track.get("monophonic_fraction", 0.0)),
            float(track.get("pitch_mean", 0.0)),
            int(track.get("non_percussion_notes", 0)),
            -int(track["track"]),
        ),
        reverse=True,
    )

    winning_track = int(qualifying[0]["track"])

    excluded_melodic_tracks = {
        int(track["track"])
        for track in qualifying[1:]
    }

    return winning_track, excluded_melodic_tracks, None


def output_path_for(candidate, subdir):
    source = Path(candidate["path"])
    return STAGE4_DIR / str(subdir) / source.name


def stage4_worker(candidate, subdir):
    """
    Process one Stage-3 candidate.

    The worker writes the MIDI itself and returns only a compact status
    dictionary to the parent.
    """
    input_path = Path(candidate["path"])
    output_path = output_path_for(candidate, subdir)

    base_result = {
        "md5": str(candidate.get("md5", "")).lower(),
        "input_path": str(input_path),
        "output_path": str(output_path),
    }

    try:
        if not input_path.is_file():
            base_result["status"] = "rejected"
            base_result["reason"] = "input_file_not_found"
            return base_result

        (
            winning_track,
            excluded_melodic_tracks,
            reason,
        ) = classify_stage3_tracks(candidate)

        if winning_track is None:
            base_result["status"] = "rejected"
            base_result["reason"] = reason
            return base_result

        midi, melody_notes, accompaniment_notes = (
            build_stage4_midi(
                input_path,
                winning_track,
                excluded_melodic_tracks,
            )
        )

        atomic_midi_write(midi, output_path)

        base_result.update(
            {
                "status": "retained",
                "winning_track": int(winning_track),
                "excluded_melodic_tracks": sorted(
                    int(track)
                    for track in excluded_melodic_tracks
                ),
                "excluded_melodic_track_count": len(
                    excluded_melodic_tracks
                ),
                "melody_notes": int(melody_notes),
                "accompaniment_notes": int(
                    accompaniment_notes
                ),
            }
        )

        del midi
        gc.collect()

        return base_result

    except Exception as exc:
        base_result["status"] = "error"
        base_result["reason"] = (
            f"{type(exc).__name__}: {exc}"
        )
        return base_result

    finally:
        gc.collect()


# ============================================================================
# ONE SUBDIRECTORY
# ============================================================================

def process_subdirectory(subdir):
    candidates = load_stage3_candidates(subdir)

    input_count = len(candidates)
    fingerprint = input_fingerprint(candidates)

    completed_results = load_checkpoints(
        subdir,
        candidates,
        fingerprint,
    )

    completed = len(completed_results)

    print()
    print("=" * 78)
    print(f"STAGE 4 — SUBDIRECTORY {subdir}")
    print("=" * 78)
    print(f"Stage 3 input       : {input_count:,}")
    print(f"Already completed   : {completed:,}")
    print(f"Remaining           : {input_count - completed:,}")
    print(f"Workers             : {WORKERS}")
    print(f"Max pending         : {MAX_PENDING}")
    print(f"Checkpoint interval : {CHECKPOINT_INTERVAL}")
    print()

    if completed >= input_count:
        retained = sum(
            result.get("status") == "retained"
            for result in completed_results
        )
        rejected = sum(
            result.get("status") == "rejected"
            for result in completed_results
        )
        errors = sum(
            result.get("status") == "error"
            for result in completed_results
        )

        print("Already complete from checkpoints.")
        print(f"  retained : {retained:,}")
        print(f"  rejected : {rejected:,}")
        print(f"  errors   : {errors:,}")

        if errors:
            raise RuntimeError(
                f"Stage 4 has {errors} previously recorded worker errors "
                f"in subdirectory {subdir}."
            )

        return completed_results

    pending = {}
    ready = {}

    next_submit = completed
    next_commit = completed

    checkpoint_number = next_checkpoint_number()
    uncheckpointed = []

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:

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

        while pending:
            done, _ = wait(
                pending,
                return_when=FIRST_COMPLETED,
            )

            for future in done:
                original_index = pending.pop(future)

                candidate = candidates[original_index]

                try:
                    result = future.result()

                except Exception as exc:
                    result = {
                        "md5": str(
                            candidate.get("md5", "")
                        ).lower(),
                        "input_path": str(
                            candidate.get("path", "")
                        ),
                        "status": "error",
                        "reason": (
                            f"future_error:"
                            f"{type(exc).__name__}: {exc}"
                        ),
                    }

                ready[original_index] = result

                if next_submit < input_count:
                    future2 = executor.submit(
                        stage4_worker,
                        candidates[next_submit],
                        subdir,
                    )
                    pending[future2] = next_submit
                    next_submit += 1

            # Commit only a contiguous prefix, so every checkpoint is a
            # resumable prefix of the exact input order.
            while next_commit in ready:
                result = ready.pop(next_commit)

                uncheckpointed.append(result)
                next_commit += 1

                if len(uncheckpointed) >= CHECKPOINT_INTERVAL:
                    start = (
                        next_commit
                        - len(uncheckpointed)
                    )
                    end = next_commit

                    write_checkpoint(
                        checkpoint_number,
                        subdir,
                        input_count,
                        fingerprint,
                        start,
                        end,
                        uncheckpointed,
                    )

                    checkpoint_number += 1
                    uncheckpointed = []

            del done
            gc.collect()

    if uncheckpointed:
        start = next_commit - len(uncheckpointed)
        end = next_commit

        write_checkpoint(
            checkpoint_number,
            subdir,
            input_count,
            fingerprint,
            start,
            end,
            uncheckpointed,
        )

    if next_commit != input_count:
        raise RuntimeError(
            f"Internal Stage-4 completion error for {subdir}: "
            f"committed {next_commit} of {input_count}"
        )

    results = load_checkpoints(
        subdir,
        candidates,
        fingerprint,
    )

    retained = sum(
        result.get("status") == "retained"
        for result in results
    )
    rejected = sum(
        result.get("status") == "rejected"
        for result in results
    )
    errors = sum(
        result.get("status") == "error"
        for result in results
    )

    print()
    print(f"STAGE 4 COMPLETE — {subdir}")
    print(f"  Input    : {input_count:,}")
    print(f"  Retained : {retained:,}")
    print(f"  Rejected : {rejected:,}")
    print(f"  Errors   : {errors:,}")

    if errors:
        raise RuntimeError(
            f"Stage 4 encountered {errors} worker errors in "
            f"subdirectory {subdir}. Checkpoints retained."
        )

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print()
    print("=" * 78)
    print("Los Angeles MIDI Dataset")
    print("STAGE 4 — MELODY SELECTION + MELODIC-CONTAMINATION-AWARE ACCOMPANIMENT")
    print("=" * 78)
    print()
    print("Input = Stage 3 candidates_?.json directly.")
    print("LAMDa metadata = NOT USED.")
    print("Workers =", WORKERS)
    print("Max pending =", MAX_PENDING)
    print(
        "Thresholds:",
        f"pitch_mean >= {PITCH_MEAN},",
        f"melody_score >= {SCORE_THRESHOLD},",
        f"monophonic_fraction >= {MONO_THRESHOLD}",
    )
    print("Other qualifying Stage-3 melody-like tracks = EXCLUDED")
    print("Short-note / density / polyphony filtering = OFF")
    print("Quantization =", QUANTIZE_OUTPUT)
    print()

    if not STAGE3_DIR.is_dir():
        raise RuntimeError(
            f"Stage 3 directory does not exist:\n{STAGE3_DIR}"
        )

    if not MIDI_ROOT.is_dir():
        raise RuntimeError(
            f"MIDI root does not exist:\n{MIDI_ROOT}"
        )

    STAGE4_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    successful = False

    try:
        for subdir in INPUT_SUBDIRECTORIES:
            input_file = (
                STAGE3_DIR / f"candidates_{subdir}.json"
            )

            if not input_file.is_file():
                print(
                    f"Skipping {subdir}: "
                    f"{input_file} not found."
                )
                continue

            process_subdirectory(subdir)

        successful = True

    finally:
        if successful:
            print()
            print("=" * 78)
            print("REMOVING STAGE 4 CHECKPOINTS")
            print("=" * 78)
            delete_all_checkpoints()
        else:
            print()
            print("=" * 78)
            print(
                "STAGE 4 DID NOT COMPLETE SUCCESSFULLY — "
                "CHECKPOINTS RETAINED"
            )
            print("=" * 78)

    print()
    print("=" * 78)
    print("STAGE 4 COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()

