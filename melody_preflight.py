#!/usr/bin/env python3
"""
melody_preflight.py

Diagnostic-only melodic structure analysis for midi-function-alignment.

It detects:
  1. candidate structural boundaries,
  2. neutral segments between accepted boundaries,
  3. ornament-tolerant pairwise similarity,
  4. families of likely repeated / varied segments.

It does NOT infer chords, infer harmony, label verse/chorus/bridge,
modify the MIDI, or influence generation.

Grid: 1 step = 1/16 note = 1/4 quarter note.

Example:
    python melody_preflight.py test/inputs/ReconstruirePredestinati.mid

Useful diagnostics:
    python melody_preflight.py test/inputs/ReconstruirePredestinati.mid \
        --show-all-pairs --show-reduced-notes
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import mido
import numpy as np

#ALPHA SWEEP test
from dataclasses import replace

GRID_PER_QUARTER = 4.0
DEFAULT_NUMERATOR = 4
DEFAULT_DENOMINATOR = 4

DEFAULT_BOUNDARY_THRESHOLD = 0.50
DEFAULT_MIN_SEGMENT_STEPS = 16
DEFAULT_BOUNDARY_NMS_STEPS = 8

DEFAULT_PAIR_REPORT_THRESHOLD = 0.58
DEFAULT_FAMILY_THRESHOLD = 0.76
DEFAULT_MAX_BASE_BARS = 4
DEFAULT_MAX_MACRO_BARS = 8
DEFAULT_MACRO_THRESHOLD = 0.72
DEFAULT_LATTICE_TOLERANCE_STEPS = 4

# Piecewise structural-lattice decoding.
# v6 does NOT accumulate a reward per emitted span. Instead, at each accepted
# structural boundary it scores 1/2/3/4-bar LOCAL lattice hypotheses over a
# bounded look-ahead region, chooses the local fundamental, commits one span,
# and repeats.
DEFAULT_LOCAL_LATTICE_HORIZON_BARS = 8
DEFAULT_LOCAL_SWITCH_MARGIN = 0.035
DEFAULT_LOCAL_ONE_BAR_MARGIN = 0.050
GLOBAL_PRIOR_TIE_BONUS = 0.020
CONTINUITY_TIE_BONUS = 0.030

# Fundamental-vs-multiple correction.
# A local N-bar winner is demoted to a proper divisor when the divisor's own
# boundary grid is strongly and consistently supported. This is what prevents
# a 4-bar macro grid from masquerading as the 2-bar fundamental.
DIVISOR_SCORE_RATIO_MIN = 0.86
DIVISOR_BOUNDARY_RATIO_MIN = 0.82
DIVISOR_COVERAGE_MIN = 0.60
DIVISOR_SUBGRID_SUPPORT_MIN = 0.30
DIVISOR_SUBGRID_COVERAGE_MIN = 0.60
DIVISOR_MAX_RECURRENCE_DROP = 0.18

# One-bar is allowed as a true fundamental, but requires stricter evidence.
ONE_BAR_DIVISOR_SCORE_RATIO_MIN = 0.96
ONE_BAR_DIVISOR_BOUNDARY_RATIO_MIN = 0.92
ONE_BAR_DIVISOR_COVERAGE_MIN = 0.80
ONE_BAR_DIVISOR_SUBGRID_SUPPORT_MIN = 0.36
ONE_BAR_DIVISOR_SUBGRID_COVERAGE_MIN = 0.75

# v8 period-state logic with ammendments.
# Span length and fundamental period are deliberately separate:
# a 4-bar span may be emitted while the established 2-bar fundamental/phase
# remains active.
PERIOD_ESTABLISH_REPEATS = 2
PERIOD_ESTABLISH_CONFIRMATIONS = 1
PERIOD_SWITCH_SCORE_TOLERANCE = 0.020
PERIOD_SWITCH_CONFIRMATIONS = 2

# Long melody-free intervals are structural gaps, not phrase evidence.
DEFAULT_GAP_MIN_BARS = 2.0
GAP_RESTART_SEARCH_BARS = 1

# BasicPitch can fragment one sung terminal note into several short MIDI
# detections. Gap-start snapping therefore reasons about TEMPORAL ACTIVITY
# CLUSTERS rather than raw note-event count or velocity.
#
# Fragments whose sounding intervals overlap, or whose gap is <= this value,
# are treated as one continuous activity cluster.
GAP_TAIL_CLUSTER_MERGE_STEPS = 1.0

# A terminal cluster may extend this far from the candidate barline and still
# be treated as phrase-ending spill rather than the beginning of a new phrase.
# One beat = 4 steps, so this deliberately permits a little BasicPitch timing
# spill beyond beat 1 without swallowing substantial new melodic activity.
GAP_TAIL_CLUSTER_MAX_END_STEPS = 6.0

# Structural time is fixed by system contract:
# bars 1-2 are free/prompt context; bar 3 beat 1 is structural zero.
STRUCTURAL_ANCHOR_BAR = 3

LOCAL_WINDOW_STEPS = 16
LONG_NOTE_STEPS = 4.0
STRONG_REST_STEPS = 4.0
MEANINGFUL_REST_STEPS = 4.0

ORNAMENT_MAX_DURATION_STEPS = 2.0
ORNAMENT_MAX_GAP_STEPS = 1.0

EPS = 1e-9

# ALPHA SWEEP test
ENDPOINT_ALPHA_SWEEP = (
    0.00,
    0.05,
    0.10,
    0.15,
    0.175,
    0.20,
    0.25,
)

@dataclass
class Note:
    pitch: int
    start_tick: int
    end_tick: int
    start: float
    end: float
    duration: float
    velocity: int

    @property
    def pitch_class(self) -> int:
        return self.pitch % 12


@dataclass
class BoundaryCandidate:
    step: int
    score: float
    rest: float
    long_note: float
    barline: float
    register_reset: float
    contour_reset: float
    density_change: float
    rhythmic_break: float
    accepted: bool = False


@dataclass
class Segment:
    index: int
    start: int
    end: int
    notes: List[Note]
    reduced_notes: List[Note]
    duration_steps: float
    onset_positions: np.ndarray
    reduced_onset_positions: np.ndarray
    pitches: np.ndarray
    reduced_pitches: np.ndarray
    relative_pitches: np.ndarray
    reduced_relative_pitches: np.ndarray
    contour: np.ndarray
    reduced_contour: np.ndarray
    duration_pattern: np.ndarray
    reduced_duration_pattern: np.ndarray
    metrical_pitches: np.ndarray
    metrical_positions: np.ndarray
    pitch_class_histogram: np.ndarray

    @property
    def label(self) -> str:
        return f"S{self.index:02d}"


@dataclass
class PairSimilarity:
    left: int
    right: int
    overall: float
    reduced_melody: float
    contour: float
    rhythm: float
    metrical_anchor: float
    duration: float
    pitch_class: float
    absolute_register: float
    classification: str


@dataclass
class LatticeCandidate:
    origin: int
    unit_bars: int
    unit_steps: int
    score: float
    boundary_support: float
    coverage: float
    recurrence: float
    expected_boundaries: int


@dataclass
class StructuralSpan:
    index: int
    start: int
    end: int
    segment: Segment
    kind: str = "segment"
    pickup_start: Optional[float] = None
    pickup_steps: float = 0.0
    entrance_start: Optional[float] = None
    entrance_offset_steps: float = 0.0
    trailing_silence_steps: float = 0.0

    @property
    def label(self) -> str:
        if self.kind == "gap":
            return f"G{self.index:02d}"
        return f"S{self.index:02d}"


@dataclass
class MacroMatch:
    length_bars: int
    left_start: int
    left_end: int
    right_start: int
    right_end: int
    similarity: PairSimilarity


@dataclass
class AdaptiveSpanDecision:
    start: int
    end: int
    unit_bars: int
    local_score: float
    boundary_support: float
    coverage: float
    recurrence_support: float
    kind: str = "segment"
    runner_up_bars: Optional[int] = None
    runner_up_score: float = 0.0
    raw_unit_bars: Optional[int] = None
    established_period_bars: Optional[int] = None
    decision_reason: str = ""
    switch_probe_bars: Optional[int] = None


# ---------------------------------------------------------------------------
# MIDI reading
# ---------------------------------------------------------------------------

def tick_to_step(tick: int, ticks_per_beat: int) -> float:
    return float(tick) / float(ticks_per_beat) * GRID_PER_QUARTER


def find_single_note_track(midi: mido.MidiFile) -> int:
    tracks = []

    for i, track in enumerate(midi.tracks):
        if any(msg.type == "note_on" and msg.velocity > 0 for msg in track):
            tracks.append(i)

    if len(tracks) != 1:
        raise ValueError(
            "Input MIDI must contain exactly one non-empty melody track. "
            f"Found {len(tracks)}."
        )

    return tracks[0]


def read_time_signatures(midi: mido.MidiFile) -> List[Tuple[int, int, int]]:
    result = []

    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "time_signature":
                result.append((int(tick), int(msg.numerator), int(msg.denominator)))

    result.sort(key=lambda x: x[0])
    out = []

    for item in result:
        if out and out[-1][0] == item[0]:
            out[-1] = item
        else:
            out.append(item)

    return out


def read_tempos(midi: mido.MidiFile) -> List[Tuple[int, float]]:
    result = []

    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                result.append((int(tick), float(mido.tempo2bpm(msg.tempo))))

    result.sort(key=lambda x: x[0])
    out = []

    for item in result:
        if out and out[-1][0] == item[0]:
            out[-1] = item
        else:
            out.append(item)

    return out


def read_melody(path: str):
    midi = mido.MidiFile(path)
    track = midi.tracks[find_single_note_track(midi)]

    # Per-(channel,pitch) FIFO queue. This is robust to repeated same-pitch notes.
    active: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    notes: List[Note] = []
    absolute_tick = 0

    for msg in track:
        absolute_tick += msg.time
        channel = int(getattr(msg, "channel", 0))

        if msg.type == "note_on" and msg.velocity > 0:
            key = (channel, int(msg.note))
            active.setdefault(key, []).append((int(absolute_tick), int(msg.velocity)))

        elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
            key = (channel, int(msg.note))
            queue = active.get(key)
            if not queue:
                continue

            start_tick, velocity = queue.pop(0)
            if not queue:
                del active[key]

            end_tick = int(absolute_tick)
            if end_tick <= start_tick:
                continue

            start = tick_to_step(start_tick, midi.ticks_per_beat)
            end = tick_to_step(end_tick, midi.ticks_per_beat)
            notes.append(
                Note(
                    pitch=int(msg.note),
                    start_tick=start_tick,
                    end_tick=end_tick,
                    start=start,
                    end=end,
                    duration=end - start,
                    velocity=velocity,
                )
            )

    # Close malformed hanging notes at track end, diagnostically rather than crash.
    for (_, pitch), queue in active.items():
        for start_tick, velocity in queue:
            if absolute_tick <= start_tick:
                continue
            start = tick_to_step(start_tick, midi.ticks_per_beat)
            end = tick_to_step(absolute_tick, midi.ticks_per_beat)
            notes.append(
                Note(
                    pitch=pitch,
                    start_tick=start_tick,
                    end_tick=int(absolute_tick),
                    start=start,
                    end=end,
                    duration=end - start,
                    velocity=velocity,
                )
            )

    notes.sort(key=lambda n: (n.start, n.pitch, n.end))

    if not notes:
        raise ValueError("No melody notes found.")

    return midi, notes, read_time_signatures(midi), read_tempos(midi)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def clip01(x: float) -> float:
    return float(max(0.0, min(1.0, x)))


def normalized_positions(values: Sequence[float], start: float, end: float) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    return np.clip((arr - start) / max(EPS, end - start), 0.0, 1.0)


def relative_pitches(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) == 0:
        return arr
    # Median-relative survives transposition and is less pickup-sensitive than first-note-relative.
    return arr - float(np.median(arr))


def contour(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if len(arr) < 2:
        return np.zeros(0, dtype=np.float64)
    delta = np.diff(arr)
    return np.sign(delta) * np.log1p(np.abs(delta))


def duration_pattern(notes: Sequence[Note]) -> np.ndarray:
    if not notes:
        return np.zeros(0, dtype=np.float64)
    arr = np.asarray([n.duration for n in notes], dtype=np.float64)
    med = float(np.median(arr))
    return arr if med < EPS else arr / med


def pitch_class_histogram(notes: Sequence[Note]) -> np.ndarray:
    hist = np.zeros(12, dtype=np.float64)
    for note in notes:
        hist[note.pitch_class] += math.sqrt(max(0.25, note.duration))
    total = float(hist.sum())
    if total > EPS:
        hist /= total
    return hist


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) == 0 or len(b) == 0:
        return 0.0
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < EPS:
        return 1.0
    return clip01(float(np.dot(a, b)) / denom)


def is_strong_metric_position(step: float, bar_steps: float) -> bool:
    if bar_steps <= EPS:
        return False
    pos = step % bar_steps
    nearest_quarter = round(pos / GRID_PER_QUARTER) * GRID_PER_QUARTER
    return abs(pos - nearest_quarter) <= 0.20


# ---------------------------------------------------------------------------
# Ornament-tolerant reduction
# ---------------------------------------------------------------------------

def reduce_ornaments(notes: Sequence[Note], bar_steps: float) -> List[Note]:
    """
    Preserve first/last, strong-beat notes, long notes and local extrema.
    Short weak-position passing/neighbor notes may be removed.

    IMPORTANT: this reduced sequence is only ONE similarity cue.
    Original notes are retained for the rest of the analysis.
    """
    notes = list(notes)
    if len(notes) <= 2:
        return notes

    keep = np.zeros(len(notes), dtype=bool)
    keep[0] = True
    keep[-1] = True

    for i, note in enumerate(notes):
        if is_strong_metric_position(note.start, bar_steps):
            keep[i] = True
        if note.duration >= LONG_NOTE_STEPS:
            keep[i] = True

    for i in range(1, len(notes) - 1):
        p0, p1, p2 = notes[i - 1].pitch, notes[i].pitch, notes[i + 1].pitch
        if (p1 > p0 and p1 > p2) or (p1 < p0 and p1 < p2):
            keep[i] = True

    for i in range(1, len(notes) - 1):
        if keep[i]:
            continue

        prev_note = notes[i - 1]
        note = notes[i]
        next_note = notes[i + 1]

        gap_before = max(0.0, note.start - prev_note.end)
        gap_after = max(0.0, next_note.start - note.end)

        short = note.duration <= ORNAMENT_MAX_DURATION_STEPS
        connected = (
            gap_before <= ORNAMENT_MAX_GAP_STEPS
            and gap_after <= ORNAMENT_MAX_GAP_STEPS
        )
        passing = min(prev_note.pitch, next_note.pitch) <= note.pitch <= max(prev_note.pitch, next_note.pitch)
        neighborish = abs(note.pitch - prev_note.pitch) <= 2 or abs(note.pitch - next_note.pitch) <= 2

        if not (short and connected and (passing or neighborish)):
            keep[i] = True

    reduced = [note for note, yes in zip(notes, keep) if yes]

    # Collapse very close repeated pitches in the reduced representation only.
    collapsed: List[Note] = []
    for note in reduced:
        if not collapsed:
            collapsed.append(note)
            continue
        prev_note = collapsed[-1]
        if note.pitch == prev_note.pitch and note.start - prev_note.end <= ORNAMENT_MAX_GAP_STEPS:
            if note.duration > prev_note.duration:
                collapsed[-1] = note
            continue
        collapsed.append(note)

    return collapsed


# ---------------------------------------------------------------------------
# Boundary clues
# ---------------------------------------------------------------------------

def notes_in(notes: Sequence[Note], start: float, end: float) -> List[Note]:
    return [n for n in notes if start <= n.start < end]


def previous_note(notes: Sequence[Note], step: float) -> Optional[Note]:
    candidates = [n for n in notes if n.start < step]
    return max(candidates, key=lambda n: (n.start, n.end)) if candidates else None


def next_note(notes: Sequence[Note], step: float) -> Optional[Note]:
    candidates = [n for n in notes if n.start >= step]
    return min(candidates, key=lambda n: (n.start, n.end)) if candidates else None


def rest_score(notes: Sequence[Note], step: float) -> float:
    before = previous_note(notes, step)
    after = next_note(notes, step)
    if before is None or after is None:
        return 0.0
    gap = max(0.0, after.start - before.end)
    return 0.0 if gap < 0.5 else clip01(gap / STRONG_REST_STEPS)


def long_note_score(notes: Sequence[Note], step: float) -> float:
    before = previous_note(notes, step)
    if before is None:
        return 0.0
    distance = abs(step - before.end)
    if distance > STRONG_REST_STEPS:
        return 0.0
    return clip01(before.duration / LONG_NOTE_STEPS) * (1.0 - clip01(distance / STRONG_REST_STEPS))


def barline_score(step: float, bar_steps: float) -> float:
    if bar_steps <= EPS:
        return 0.0
    rem = step % bar_steps
    distance = min(rem, bar_steps - rem)
    return clip01(1.0 - distance)


def local_pitch_median(notes: Sequence[Note], start: float, end: float) -> Optional[float]:
    local = notes_in(notes, start, end)
    return float(np.median([n.pitch for n in local])) if local else None


def register_reset_score(notes: Sequence[Note], step: float) -> float:
    left = local_pitch_median(notes, step - LOCAL_WINDOW_STEPS, step)
    right = local_pitch_median(notes, step, step + LOCAL_WINDOW_STEPS)
    if left is None or right is None:
        return 0.0
    return clip01(abs(right - left) / 7.0)


def local_contour_direction(notes: Sequence[Note], start: float, end: float) -> Optional[float]:
    local = notes_in(notes, start, end)
    if len(local) < 2:
        return None

    x = np.asarray([n.start for n in local], dtype=np.float64)
    y = np.asarray([n.pitch for n in local], dtype=np.float64)
    x -= x.mean()
    denom = float(np.dot(x, x))
    if denom < EPS:
        return 0.0
    slope = float(np.dot(x, y - y.mean()) / denom)
    return math.tanh(slope * 2.0)


def contour_reset_score(notes: Sequence[Note], step: float) -> float:
    left = local_contour_direction(notes, step - LOCAL_WINDOW_STEPS, step)
    right = local_contour_direction(notes, step, step + LOCAL_WINDOW_STEPS)
    if left is None or right is None:
        return 0.0
    return clip01(abs(right - left) / 2.0)


def density_change_score(notes: Sequence[Note], step: float) -> float:
    left = len(notes_in(notes, step - LOCAL_WINDOW_STEPS, step)) / LOCAL_WINDOW_STEPS
    right = len(notes_in(notes, step, step + LOCAL_WINDOW_STEPS)) / LOCAL_WINDOW_STEPS
    denom = max(0.20, left, right)
    return clip01(abs(right - left) / denom)


def normalized_ioi(notes: Sequence[Note], start: float, end: float) -> np.ndarray:
    local = notes_in(notes, start, end)
    if len(local) < 3:
        return np.zeros(0, dtype=np.float64)
    onsets = np.asarray([n.start for n in local], dtype=np.float64)
    values = np.diff(onsets)
    med = float(np.median(values))
    return values if med < EPS else values / med


def rhythmic_break_score(notes: Sequence[Note], step: float) -> float:
    left = normalized_ioi(notes, step - LOCAL_WINDOW_STEPS, step)
    right = normalized_ioi(notes, step, step + LOCAL_WINDOW_STEPS)
    if len(left) == 0 or len(right) == 0:
        return 0.0
    delta = abs(float(np.median(left)) - float(np.median(right)))
    delta += 0.5 * abs(float(np.std(left)) - float(np.std(right)))
    return clip01(delta / 1.5)


def candidate_steps(notes: Sequence[Note], total_steps: int, bar_steps: float) -> List[int]:
    values = set()

    for note in notes:
        values.add(int(round(note.start)))
        values.add(int(round(note.end)))

    bar = bar_steps
    while bar_steps > EPS and bar < total_steps:
        values.add(int(round(bar)))
        bar += bar_steps

    return sorted(step for step in values if 0 < step < total_steps)


def score_boundaries(notes: Sequence[Note], total_steps: int, bar_steps: float) -> List[BoundaryCandidate]:
    result = []

    for step in candidate_steps(notes, total_steps, bar_steps):
        rest = rest_score(notes, step)
        long_note = long_note_score(notes, step)
        barline = barline_score(step, bar_steps)
        register = register_reset_score(notes, step)
        contour_reset = contour_reset_score(notes, step)
        density = density_change_score(notes, step)
        rhythm = rhythmic_break_score(notes, step)

        score = (
            0.30 * rest
            + 0.17 * long_note
            + 0.16 * barline
            + 0.11 * register
            + 0.09 * contour_reset
            + 0.10 * density
            + 0.07 * rhythm
        )

        strong_clues = sum(
            value >= 0.60
            for value in (rest, long_note, barline, register, contour_reset, density, rhythm)
        )
        if strong_clues >= 3:
            score += 0.08
        elif strong_clues == 2:
            score += 0.04

        result.append(
            BoundaryCandidate(
                step=step,
                score=clip01(score),
                rest=rest,
                long_note=long_note,
                barline=barline,
                register_reset=register,
                contour_reset=contour_reset,
                density_change=density,
                rhythmic_break=rhythm,
            )
        )

    return result


def select_boundaries(
    candidates: Sequence[BoundaryCandidate],
    total_steps: int,
    threshold: float,
    min_segment_steps: int,
    nms_steps: int,
) -> List[BoundaryCandidate]:
    # High confidence first.
    eligible = sorted(
        [c for c in candidates if c.score >= threshold],
        key=lambda c: (-c.score, c.step),
    )

    selected: List[BoundaryCandidate] = []

    for candidate in eligible:
        if candidate.step < min_segment_steps:
            continue
        if total_steps - candidate.step < min_segment_steps:
            continue
        if any(abs(candidate.step - other.step) < nms_steps for other in selected):
            continue
        selected.append(candidate)

    selected.sort(key=lambda c: c.step)

    # Remove weaker boundaries if they create a segment shorter than the requested minimum.
    changed = True
    while changed and selected:
        changed = False
        points = [0] + [c.step for c in selected] + [total_steps]

        for i in range(1, len(points)):
            if points[i] - points[i - 1] >= min_segment_steps:
                continue

            left_idx = i - 2
            right_idx = i - 1
            options = [idx for idx in (left_idx, right_idx) if 0 <= idx < len(selected)]
            if not options:
                continue

            remove_idx = min(options, key=lambda idx: (selected[idx].score, -selected[idx].step))
            del selected[remove_idx]
            changed = True
            break

    accepted_steps = {c.step for c in selected}
    for candidate in candidates:
        candidate.accepted = candidate.step in accepted_steps

    return selected


# ---------------------------------------------------------------------------
# Segment features
# ---------------------------------------------------------------------------

def metrical_anchors(notes: Sequence[Note], start: int, end: int, bar_steps: float):
    pitches = []
    positions = []

    for note in notes:
        local = note.start - start
        if is_strong_metric_position(local, bar_steps):
            pitches.append(note.pitch)
            positions.append(note.start)

    return (
        relative_pitches(pitches),
        normalized_positions(positions, start, end),
    )


def build_segment(index: int, start: int, end: int, all_notes: Sequence[Note], bar_steps: float) -> Segment:
    notes = [n for n in all_notes if n.start >= start - EPS and n.start < end - EPS]
    reduced = reduce_ornaments(notes, bar_steps)

    pitches = np.asarray([n.pitch for n in notes], dtype=np.float64)
    reduced_pitches = np.asarray([n.pitch for n in reduced], dtype=np.float64)
    metric_pitches, metric_positions = metrical_anchors(notes, start, end, bar_steps)

    return Segment(
        index=index,
        start=start,
        end=end,
        notes=notes,
        reduced_notes=reduced,
        duration_steps=float(end - start),
        onset_positions=normalized_positions([n.start for n in notes], start, end),
        reduced_onset_positions=normalized_positions([n.start for n in reduced], start, end),
        pitches=pitches,
        reduced_pitches=reduced_pitches,
        relative_pitches=relative_pitches(pitches),
        reduced_relative_pitches=relative_pitches(reduced_pitches),
        contour=contour(pitches),
        reduced_contour=contour(reduced_pitches),
        duration_pattern=duration_pattern(notes),
        reduced_duration_pattern=duration_pattern(reduced),
        metrical_pitches=metric_pitches,
        metrical_positions=metric_positions,
        pitch_class_histogram=pitch_class_histogram(notes),
    )


def make_segments(
    notes: Sequence[Note],
    boundaries: Sequence[BoundaryCandidate],
    total_steps: int,
    bar_steps: float,
) -> List[Segment]:
    points = [0] + sorted(c.step for c in boundaries) + [total_steps]
    return [
        build_segment(i + 1, points[i], points[i + 1], notes, bar_steps)
        for i in range(len(points) - 1)
    ]


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------

def dtw_distance(a: Sequence[float], b: Sequence[float], scale: float) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)

    if len(x) == 0 and len(y) == 0:
        return 0.0
    if len(x) == 0 or len(y) == 0:
        return float("inf")

    scale = max(EPS, float(scale))
    dp = np.full((len(x) + 1, len(y) + 1), np.inf, dtype=np.float64)
    dp[0, 0] = 0.0

    for i in range(1, len(x) + 1):
        for j in range(1, len(y) + 1):
            cost = abs(x[i - 1] - y[j - 1]) / scale
            dp[i, j] = cost + min(dp[i - 1, j], dp[i, j - 1], dp[i - 1, j - 1])

    return float(dp[-1, -1] / max(1, len(x) + len(y)))


def sequence_similarity(
    a: Sequence[float],
    b: Sequence[float],
    *,
    scale: float,
    sensitivity: float,
) -> float:
    distance = dtw_distance(a, b, scale)
    if not math.isfinite(distance):
        return 0.0
    return clip01(math.exp(-sensitivity * distance))


def metric_similarity(left: Segment, right: Segment) -> float:
    if len(left.metrical_pitches) == 0 or len(right.metrical_pitches) == 0:
        return 0.0

    pitch = sequence_similarity(
        left.metrical_pitches,
        right.metrical_pitches,
        scale=5.0,
        sensitivity=2.0,
    )
    position = sequence_similarity(
        left.metrical_positions,
        right.metrical_positions,
        scale=0.20,
        sensitivity=2.0,
    )
    return 0.70 * pitch + 0.30 * position


def compare_segments(left: Segment, right: Segment) -> PairSimilarity:
    reduced = sequence_similarity(
        left.reduced_relative_pitches,
        right.reduced_relative_pitches,
        scale=5.0,
        sensitivity=2.2,
    )
    cont = sequence_similarity(
        left.reduced_contour,
        right.reduced_contour,
        scale=1.5,
        sensitivity=2.0,
    )
    rhythm = sequence_similarity(
        left.reduced_onset_positions,
        right.reduced_onset_positions,
        scale=0.18,
        sensitivity=2.2,
    )
    metric = metric_similarity(left, right)
    durations = sequence_similarity(
        left.reduced_duration_pattern,
        right.reduced_duration_pattern,
        scale=1.0,
        sensitivity=1.7,
    )
    pc = cosine_similarity(left.pitch_class_histogram, right.pitch_class_histogram)
    abs_register = sequence_similarity(
        left.reduced_pitches,
        right.reduced_pitches,
        scale=12.0,
        sensitivity=1.8,
    )

    length_ratio = min(left.duration_steps, right.duration_steps) / max(EPS, left.duration_steps, right.duration_steps)

    # Absolute register is intentionally NOT in this aggregate.
    overall = (
        0.30 * reduced
        + 0.23 * cont
        + 0.22 * rhythm
        + 0.12 * metric
        + 0.08 * durations
        + 0.05 * pc
    )
    overall *= 0.75 + 0.25 * length_ratio
    overall = clip01(overall)

    if overall >= 0.88:
        classification = "VERY_STRONG_VARIANT"
    elif overall >= 0.80:
        classification = "STRONG_VARIANT"
    elif overall >= 0.72:
        classification = "LIKELY_VARIANT"
    elif overall >= 0.62:
        classification = "POSSIBLE_RELATION"
    else:
        classification = "WEAK"

    return PairSimilarity(
        left=left.index,
        right=right.index,
        overall=overall,
        reduced_melody=reduced,
        contour=cont,
        rhythm=rhythm,
        metrical_anchor=metric,
        duration=durations,
        pitch_class=pc,
        absolute_register=abs_register,
        classification=classification,
    )


def compare_all(segments: Sequence[Segment]) -> List[PairSimilarity]:
    pairs = []
    for i in range(len(segments)):
        for j in range(i + 1, len(segments)):
            pairs.append(compare_segments(segments[i], segments[j]))
    return sorted(pairs, key=lambda p: (-p.overall, p.left, p.right))


# ---------------------------------------------------------------------------
# Structural lattice and macro analysis
# ---------------------------------------------------------------------------

def step_to_bar_beat_subdiv(step: float, bar_steps: float) -> str:
    if bar_steps <= EPS:
        return f"step={step:.2f}"
    bar_index = int(math.floor(step / bar_steps))
    within_bar = step - bar_index * bar_steps
    beat_index = int(math.floor(within_bar / GRID_PER_QUARTER))
    within_beat = within_bar - beat_index * GRID_PER_QUARTER
    whole = int(math.floor(within_beat + 1e-9)) + 1
    frac = within_beat - math.floor(within_beat + 1e-9)
    if abs(frac) < 0.20:
        sub = str(whole)
    elif abs(frac - 0.5) < 0.20:
        sub = f"{whole}.5"
    else:
        sub = f"{whole}+{frac:.2f}"
    return f"{bar_index + 1}|{beat_index + 1}|{sub}"


def fixed_structural_origin(bar_steps: float) -> int:
    """
    System contract: bars 1-2 are free/prompt context and bar 3 beat 1
    is always structural zero. Melody onset never moves this anchor.
    """
    return int(round((STRUCTURAL_ANCHOR_BAR - 1) * bar_steps))


def _candidate_near_step(candidates, step, tolerance):
    nearby = [c for c in candidates if abs(c.step - step) <= tolerance]
    if not nearby:
        return None
    return max(nearby, key=lambda c: (c.score, -abs(c.step-step)))


def _window_recurrence_score(notes, origin, unit_steps, total_steps, bar_steps):
    windows=[]
    start=origin
    idx=1
    while start < total_steps:
        end=min(start+unit_steps,total_steps)
        if end-start < max(4,unit_steps//2):
            break
        seg=build_segment(idx,int(start),int(end),notes,bar_steps)
        if seg.notes:
            windows.append(seg)
        start += unit_steps
        idx += 1
    if len(windows) < 2:
        return 0.0
    best=[]
    for i,left in enumerate(windows):
        scores=[compare_segments(left,right).overall for j,right in enumerate(windows) if i!=j]
        if scores:
            best.append(max(scores))
    return float(np.median(best)) if best else 0.0


def score_lattice_candidates(notes, boundary_candidates, total_steps, bar_steps,
                             max_base_bars, tolerance_steps, origin):
    """Score only candidate UNIT LENGTHS. The origin is fixed externally."""
    if origin >= total_steps:
        return []

    out=[]
    for unit_bars in range(1,max_base_bars+1):
        unit_steps=int(round(bar_steps*unit_bars))
        expected=list(range(origin+unit_steps,total_steps,unit_steps))
        if not expected:
            continue
        supports=[]
        covered=0
        for e in expected:
            c=_candidate_near_step(boundary_candidates,e,tolerance_steps)
            if c is None:
                supports.append(0.0)
                continue
            distance=abs(c.step-e)
            proximity=max(0.0,1.0-distance/float(tolerance_steps+1))
            support=c.score*proximity
            supports.append(support)
            if support >= 0.30:
                covered += 1
        boundary_support=float(np.mean(supports))
        coverage=covered/float(len(expected))
        recurrence=_window_recurrence_score(notes,origin,unit_steps,total_steps,bar_steps)
        score=0.62*boundary_support + 0.23*coverage + 0.15*recurrence
        evidence_factor=min(1.0,len(expected)/4.0)
        score *= 0.85 + 0.15*evidence_factor
        out.append(LatticeCandidate(origin,unit_bars,unit_steps,clip01(score),
                                    boundary_support,coverage,recurrence,len(expected)))
    return sorted(out,key=lambda x:(-x.score,x.unit_bars))


def choose_lattice(candidates):
    """
    Choose the FUNDAMENTAL structural period, not merely the strongest
    higher-order periodicity.

    If the true base unit is 2 bars, a 4-bar grid is a subset of that lattice
    and can score even better because every second 2-bar boundary may be
    especially strong.

    We therefore start from the strongest raw candidate and test its integer
    divisors as possible fundamentals.
    """
    if not candidates:
        raise ValueError("Could not infer any structural lattice candidates.")

    by_bars = {c.unit_bars: c for c in candidates}

    strongest = max(
        candidates,
        key=lambda c: (
            c.score,
            c.boundary_support,
            c.coverage,
            -c.unit_bars,
        ),
    )

    MIN_SCORE_RATIO = 0.90
    MIN_BOUNDARY_RATIO = 0.85
    MIN_COVERAGE_RATIO = 0.75
    MAX_RECURRENCE_DROP = 0.12

    # 1-bar is treated more conservatively because barlines themselves create
    # strong local evidence and can otherwise collapse a genuine 2-bar phrase
    # lattice into a trivial bar grid.
    ONE_BAR_SCORE_RATIO = 0.985
    ONE_BAR_BOUNDARY_RATIO = 0.95
    ONE_BAR_COVERAGE_RATIO = 0.95

    chosen = strongest

    divisors = [
        bars
        for bars in sorted(by_bars)
        if (
            bars < strongest.unit_bars
            and strongest.unit_bars % bars == 0
        )
    ]

    for bars in divisors:
        candidate = by_bars[bars]

        score_ratio = candidate.score / max(EPS, strongest.score)
        boundary_ratio = (
            candidate.boundary_support
            / max(EPS, strongest.boundary_support)
        )
        coverage_ratio = (
            candidate.coverage
            / max(EPS, strongest.coverage)
        )
        recurrence_drop = strongest.recurrence - candidate.recurrence

        if bars == 1:
            fundamental_ok = (
                score_ratio >= ONE_BAR_SCORE_RATIO
                and boundary_ratio >= ONE_BAR_BOUNDARY_RATIO
                and coverage_ratio >= ONE_BAR_COVERAGE_RATIO
                and recurrence_drop <= MAX_RECURRENCE_DROP
            )
        else:
            fundamental_ok = (
                score_ratio >= MIN_SCORE_RATIO
                and boundary_ratio >= MIN_BOUNDARY_RATIO
                and coverage_ratio >= MIN_COVERAGE_RATIO
                and recurrence_drop <= MAX_RECURRENCE_DROP
            )

        if fundamental_ok:
            chosen = candidate
            break

    return chosen


def _last_note_before(notes, step):
    prior=[n for n in notes if n.start < step]
    return max(prior,key=lambda n:(n.start,n.end)) if prior else None


def _possible_pickup_start(notes, structural_start, bar_steps):
    if structural_start <= 0:
        return None
    lookback=max(0.0,structural_start-bar_steps)
    local=sorted([n for n in notes if lookback-EPS <= n.start < structural_start-EPS],
                 key=lambda n:n.start)
    if not local:
        return None
    candidates=[]
    for i,note in enumerate(local):
        prev = local[i-1] if i>0 else _last_note_before(notes,note.start)
        gap=float('inf') if prev is None else max(0.0,note.start-prev.end)
        if gap < MEANINGFUL_REST_STEPS:
            continue
        tail=sorted([n for n in notes if note.start-EPS <= n.start < structural_start+EPS],
                    key=lambda n:n.start)
        if not tail:
            continue
        max_gap=max([max(0.0,b.start-a.end) for a,b in zip(tail,tail[1:])] or [0.0])
        last=tail[-1]
        reaches = last.end >= structural_start-1.0 or structural_start-last.end <= MEANINGFUL_REST_STEPS
        if max_gap <= MEANINGFUL_REST_STEPS and reaches:
            candidates.append(note.start)
    return min(candidates) if candidates else None


def _boundary_support_at_step(boundary_candidates, step, tolerance_steps):
    """
    Return proximity-weighted boundary evidence around an exact structural
    grid point. The structural boundary itself remains on the grid; nearby
    melodic punctuation only contributes evidence.
    """
    candidate = _candidate_near_step(
        boundary_candidates,
        step,
        tolerance_steps,
    )
    if candidate is None:
        return 0.0

    distance = abs(candidate.step - step)
    proximity = max(
        0.0,
        1.0 - distance / float(tolerance_steps + 1),
    )
    return clip01(candidate.score * proximity)


def _candidate_segment_recurrence(
    notes,
    origin,
    start,
    end,
    bar_steps,
):
    """
    How strongly does this candidate structural span resemble an earlier
    BAR-ALIGNED span of the same length?

    This is deliberately local evidence for a proposed span length; it does not
    assign semantic form labels and does not move the fixed bar-3 anchor.
    """
    if end <= start:
        return 0.0

    candidate = build_segment(
        0,
        int(start),
        int(end),
        notes,
        bar_steps,
    )
    if not candidate.notes:
        return 0.0

    length_steps = end - start
    one_bar = int(round(bar_steps))
    earlier_scores = []

    previous_start = origin
    while previous_start + length_steps <= start:
        previous_end = previous_start + length_steps
        previous = build_segment(
            0,
            int(previous_start),
            int(previous_end),
            notes,
            bar_steps,
        )
        if previous.notes:
            earlier_scores.append(
                compare_segments(candidate, previous).overall
            )
        previous_start += one_bar

    if not earlier_scores:
        return 0.0

    # A single convincing earlier analogue is useful, but using the mean of the
    # best two avoids allowing one accidental match to dominate.
    earlier_scores.sort(reverse=True)
    best = earlier_scores[:2]
    return float(np.mean(best))


def _global_prior_by_bars(lattice_candidates):
    """
    Convert whole-song lattice scores into a weak relative prior in [0,1].
    These scores no longer dictate segmentation.
    """
    if not lattice_candidates:
        return {}

    raw = {
        c.unit_bars: c.score
        for c in lattice_candidates
    }
    maximum = max(raw.values())
    minimum = min(raw.values())

    if maximum - minimum <= EPS:
        return {
            bars: 0.5
            for bars in raw
        }

    return {
        bars: (score - minimum) / (maximum - minimum)
        for bars, score in raw.items()
    }


def _local_window_recurrence_score(
    notes,
    origin,
    unit_steps,
    local_end,
    bar_steps,
):
    """
    Recurrence score for one local lattice hypothesis.

    Windows are aligned to the CURRENT structural boundary and compared only
    inside the bounded local look-ahead region. This gives 1/2/3/4-bar
    hypotheses equal footing without paying the decoder for emitting more
    spans.
    """
    windows = []
    start = origin
    idx = 1

    while start + max(4, unit_steps // 2) <= local_end:
        end = min(
            start + unit_steps,
            local_end,
        )

        if end - start < max(
            4,
            unit_steps // 2,
        ):
            break

        seg = build_segment(
            idx,
            int(start),
            int(end),
            notes,
            bar_steps,
        )

        if seg.notes:
            windows.append(
                seg
            )

        start += unit_steps
        idx += 1

    if len(windows) < 2:
        return 0.0

    best_scores = []

    for i, left in enumerate(
        windows
    ):
        scores = [
            compare_segments(
                left,
                right,
            ).overall
            for j, right in enumerate(windows)
            if i != j
        ]

        if scores:
            best_scores.append(
                max(scores)
            )

    return (
        float(
            np.median(best_scores)
        )
        if best_scores
        else 0.0
    )


def score_local_lattice_candidates(
    notes,
    boundary_candidates,
    start,
    full_end,
    bar_steps,
    max_base_bars,
    tolerance_steps,
    horizon_bars,
):
    """
    Score candidate local periods beginning at ONE already accepted structural
    boundary.

    Critical difference from v5:
      * every 1/2/3/4-bar hypothesis is scored ONCE over the same local region;
      * there is no additive reward for producing more spans;
      * the chosen hypothesis only determines the NEXT structural boundary.
    """
    one_bar = int(
        round(bar_steps)
    )

    local_end = min(
        full_end,
        start + horizon_bars * one_bar,
    )

    out = []

    for unit_bars in range(
        1,
        max_base_bars + 1,
    ):
        unit_steps = (
            unit_bars * one_bar
        )

        if start + unit_steps > full_end:
            continue

        expected = list(
            range(
                start + unit_steps,
                local_end + 1,
                unit_steps,
            )
        )

        if not expected:
            continue

        supports = []
        covered = 0

        for e in expected:
            c = _candidate_near_step(
                boundary_candidates,
                e,
                tolerance_steps,
            )

            if c is None:
                supports.append(
                    0.0
                )
                continue

            distance = abs(
                c.step - e
            )
            proximity = max(
                0.0,
                1.0 - distance / float(
                    tolerance_steps + 1
                ),
            )
            support = (
                c.score * proximity
            )
            supports.append(
                support
            )

            if support >= 0.30:
                covered += 1

        boundary_support = float(
            np.mean(supports)
        )
        coverage = (
            covered
            / float(len(expected))
        )

        recurrence = _local_window_recurrence_score(
            notes,
            start,
            unit_steps,
            local_end,
            bar_steps,
        )

        score = (
            0.62 * boundary_support
            + 0.23 * coverage
            + 0.15 * recurrence
        )

        evidence_factor = min(
            1.0,
            len(expected) / 4.0,
        )
        score *= (
            0.85
            + 0.15 * evidence_factor
        )

        out.append(
            LatticeCandidate(
                origin=int(start),
                unit_bars=int(unit_bars),
                unit_steps=int(unit_steps),
                score=clip01(score),
                boundary_support=float(
                    boundary_support
                ),
                coverage=float(
                    coverage
                ),
                recurrence=float(
                    recurrence
                ),
                expected_boundaries=len(
                    expected
                ),
            )
        )

    return sorted(
        out,
        key=lambda x: (
            -x.score,
            x.unit_bars,
        ),
    )

def print_expected_boundary_trace(
    notes,
    boundary_candidates,
    start_bar,
    full_end,
    bar_steps,
    max_base_bars,
    tolerance_steps,
    horizon_bars,
):
    """
    Purely observational.

    Show exactly which boundary candidate is used at every expected grid point
    for each 1/2/3/4-bar hypothesis beginning at start_bar.

    No decoder state is consulted and no generation/segmentation is changed.
    """
    one_bar = int(round(bar_steps))

    start = int(
        round(
            (start_bar - 1)
            * bar_steps
        )
    )

    local_end = min(
        full_end,
        start + horizon_bars * one_bar,
    )

    print("\n=== EXPECTED BOUNDARY EVIDENCE TRACE ===")
    print(
        f"Start: {step_to_bar_beat_subdiv(start, bar_steps)} "
        f"(source bar {start_bar}, absolute step {start})"
    )
    print(
        f"Tolerance: +/-{tolerance_steps} step(s)"
    )
    print(
        f"Local horizon ends: "
        f"{step_to_bar_beat_subdiv(local_end, bar_steps)}"
    )
    print(
        "Coverage threshold: support >= 0.300"
    )

    for unit_bars in range(
        1,
        max_base_bars + 1,
    ):
        unit_steps = (
            unit_bars
            * one_bar
        )

        if start + unit_steps > full_end:
            continue

        expected = list(
            range(
                start + unit_steps,
                local_end + 1,
                unit_steps,
            )
        )

        if not expected:
            continue

        print()
        print(
            f"--- {unit_bars} BAR HYPOTHESIS ---"
        )

        print(
            " expected       matched        dist   bscore  prox   support  "
            "covered   rest  long  bar   reg   cont  dens  rhythm  clues"
        )
        print(
            " ------------   ------------   ----   ------  -----  -------  "
            "-------   ----  ----  ----  ----  ----  ----  ------  -----"
        )

        supports = []
        covered_count = 0

        for expected_step in expected:
            candidate = _candidate_near_step(
                boundary_candidates,
                expected_step,
                tolerance_steps,
            )

            expected_label = step_to_bar_beat_subdiv(
                expected_step,
                bar_steps,
            )

            if candidate is None:
                supports.append(0.0)

                print(
                    f" {expected_label:>12s}   "
                    f"{'NONE':>12s}   "
                    f"{'-':>4s}   "
                    f"{'-':>6s}  "
                    f"{'-':>5s}  "
                    f"{0.0:7.3f}  "
                    f"{'NO':>7s}   "
                    f"{'-':>4s}  "
                    f"{'-':>4s}  "
                    f"{'-':>4s}  "
                    f"{'-':>4s}  "
                    f"{'-':>4s}  "
                    f"{'-':>4s}  "
                    f"{'-':>6s}  "
                    f"{'-':>5s}"
                )
                continue

            distance = abs(
                candidate.step
                - expected_step
            )

            proximity = max(
                0.0,
                1.0
                - distance
                / float(
                    tolerance_steps + 1
                ),
            )

            support = (
                candidate.score
                * proximity
            )

            covered = (
                support >= 0.30
            )

            if covered:
                covered_count += 1

            supports.append(
                support
            )

            strong_clues = sum(
                value >= 0.60
                for value in (
                    candidate.rest,
                    candidate.long_note,
                    candidate.barline,
                    candidate.register_reset,
                    candidate.contour_reset,
                    candidate.density_change,
                    candidate.rhythmic_break,
                )
            )

            matched_label = step_to_bar_beat_subdiv(
                candidate.step,
                bar_steps,
            )

            print(
                f" {expected_label:>12s}   "
                f"{matched_label:>12s}   "
                f"{distance:4.1f}   "
                f"{candidate.score:6.3f}  "
                f"{proximity:5.3f}  "
                f"{support:7.3f}  "
                f"{('YES' if covered else 'NO'):>7s}   "
                f"{candidate.rest:4.2f}  "
                f"{candidate.long_note:4.2f}  "
                f"{candidate.barline:4.2f}  "
                f"{candidate.register_reset:4.2f}  "
                f"{candidate.contour_reset:4.2f}  "
                f"{candidate.density_change:4.2f}  "
                f"{candidate.rhythmic_break:6.2f}  "
                f"{strong_clues:5d}"
            )

        mean_support = (
            float(np.mean(supports))
            if supports
            else 0.0
        )

        coverage = (
            covered_count
            / float(len(expected))
            if expected
            else 0.0
        )

        print(
            f"  => mean boundary support: {mean_support:.3f}"
        )
        print(
            f"  => coverage: {covered_count}/{len(expected)} "
            f"= {coverage:.3f}"
        )

def _proper_divisors(unit_bars):
    """
    Proper integer divisors in ascending order.
    For 4 bars -> [1, 2]
    For 3 bars -> [1]
    For 2 bars -> [1]
    """
    return [
        d
        for d in range(1, unit_bars)
        if unit_bars % d == 0
    ]


def _divisor_subgrid_evidence(
    boundary_candidates,
    start,
    local_end,
    winner_bars,
    divisor_bars,
    bar_steps,
    tolerance_steps,
):
    """
    Measure the boundaries present on the divisor grid that the longer winner
    SKIPS.

    Example, winner=4 bars and divisor=2 bars:
        winner grid:   start+4, start+8, ...
        divisor extras:start+2, start+6, start+10, ...

    Strong evidence on those skipped midpoint boundaries means the 4-bar
    hypothesis is likely a macro multiple of a 2-bar fundamental.
    """
    one_bar = int(round(bar_steps))
    winner_steps = winner_bars * one_bar
    divisor_steps = divisor_bars * one_bar

    supports = []
    covered = 0

    step = start + divisor_steps
    while step <= local_end:
        # Ignore points already belonging to the winner grid. We only care
        # about the extra subdivision boundaries contributed by the divisor.
        relative = step - start
        if relative % winner_steps != 0:
            c = _candidate_near_step(
                boundary_candidates,
                step,
                tolerance_steps,
            )

            if c is None:
                supports.append(0.0)
            else:
                distance = abs(c.step - step)
                proximity = max(
                    0.0,
                    1.0 - distance / float(tolerance_steps + 1),
                )
                support = c.score * proximity
                supports.append(support)

                if support >= 0.30:
                    covered += 1

        step += divisor_steps

    if not supports:
        return 0.0, 0.0, 0

    return (
        float(np.mean(supports)),
        covered / float(len(supports)),
        len(supports),
    )


def _fundamental_correct_local_winner(
    candidates,
    winner,
    boundary_candidates,
    start,
    full_end,
    bar_steps,
    tolerance_steps,
    horizon_bars,
):
    """
    Demote a local higher-order winner to a supported proper divisor.

    This is deliberately conservative:
      - the divisor must already be a respectable local-lattice candidate;
      - its boundary/coverage evidence cannot collapse relative to the winner;
      - the extra subdivision boundaries skipped by the winner must themselves
        be supported;
      - recurrence may be somewhat weaker, but not dramatically so.

    The shortest qualifying divisor is chosen, because it is the candidate
    fundamental explaining the higher-order periodicity.
    """
    if winner.unit_bars <= 1:
        return winner, None

    by_bars = {
        c.unit_bars: c
        for c in candidates
    }

    one_bar = int(round(bar_steps))
    local_end = min(
        full_end,
        start + horizon_bars * one_bar,
    )

    qualifying = []

    for divisor_bars in _proper_divisors(
        winner.unit_bars
    ):
        divisor = by_bars.get(
            divisor_bars
        )
        if divisor is None:
            continue

        subgrid_support, subgrid_coverage, subgrid_count = _divisor_subgrid_evidence(
            boundary_candidates,
            start,
            local_end,
            winner.unit_bars,
            divisor_bars,
            bar_steps,
            tolerance_steps,
        )

        if subgrid_count == 0:
            continue

        score_ratio = divisor.score / max(
            EPS,
            winner.score,
        )
        boundary_ratio = divisor.boundary_support / max(
            EPS,
            winner.boundary_support,
        )
        recurrence_drop = (
            winner.recurrence
            - divisor.recurrence
        )

        if divisor_bars == 1:
            qualifies = (
                score_ratio >= ONE_BAR_DIVISOR_SCORE_RATIO_MIN
                and boundary_ratio >= ONE_BAR_DIVISOR_BOUNDARY_RATIO_MIN
                and divisor.coverage >= ONE_BAR_DIVISOR_COVERAGE_MIN
                and subgrid_support >= ONE_BAR_DIVISOR_SUBGRID_SUPPORT_MIN
                and subgrid_coverage >= ONE_BAR_DIVISOR_SUBGRID_COVERAGE_MIN
                and recurrence_drop <= DIVISOR_MAX_RECURRENCE_DROP
            )
        else:
            qualifies = (
                score_ratio >= DIVISOR_SCORE_RATIO_MIN
                and boundary_ratio >= DIVISOR_BOUNDARY_RATIO_MIN
                and divisor.coverage >= DIVISOR_COVERAGE_MIN
                and subgrid_support >= DIVISOR_SUBGRID_SUPPORT_MIN
                and subgrid_coverage >= DIVISOR_SUBGRID_COVERAGE_MIN
                and recurrence_drop <= DIVISOR_MAX_RECURRENCE_DROP
            )

        if qualifies:
            qualifying.append(
                (
                    divisor_bars,
                    divisor,
                    subgrid_support,
                    subgrid_coverage,
                )
            )

    if not qualifying:
        return winner, None

    qualifying.sort(
        key=lambda item: item[0]
    )

    divisor_bars, corrected, subgrid_support, subgrid_coverage = qualifying[0]

    diagnostic = {
        "from_bars": winner.unit_bars,
        "to_bars": corrected.unit_bars,
        "winner_score": winner.score,
        "divisor_score": corrected.score,
        "subgrid_support": subgrid_support,
        "subgrid_coverage": subgrid_coverage,
    }

    return corrected, diagnostic


def _select_local_lattice(
    candidates,
    global_prior_bars,
    one_bar_margin,
):
    """
    Select the locally preferred span length WITHOUT carrying the previous
    emitted span forward as state.

    v9 deliberately separates:
      - local span-length evidence;
      - established fundamental period / phase.

    A 4-bar span therefore does not automatically make "4 bars" the new base
    period for the next decision.
    """
    if not candidates:
        raise ValueError(
            "No local structural lattice candidates."
        )

    by_bars = {
        c.unit_bars: c
        for c in candidates
    }

    chosen = choose_lattice(
        candidates
    )

    # One bar is legal, but should not win trivially.
    if (
        chosen.unit_bars == 1
        and len(candidates) > 1
    ):
        non_one = max(
            (
                c
                for c in candidates
                if c.unit_bars != 1
            ),
            key=lambda c: c.score,
            default=None,
        )

        if (
            non_one is not None
            and chosen.score
            < non_one.score + one_bar_margin
        ):
            chosen = non_one

    # Whole-song prior remains only a close-tie preference.
    if (
        global_prior_bars is not None
        and global_prior_bars in by_bars
    ):
        global_candidate = by_bars[
            global_prior_bars
        ]

        if (
            global_candidate.score
            + GLOBAL_PRIOR_TIE_BONUS
            >= chosen.score
        ):
            chosen = global_candidate

    ranked = sorted(
        candidates,
        key=lambda c: (
            -c.score,
            c.unit_bars,
        ),
    )

    runner_up = next(
        (
            c
            for c in ranked
            if c.unit_bars
            != chosen.unit_bars
        ),
        None,
    )

    return chosen, runner_up

def print_local_candidate_trace(
    notes,
    boundary_candidates,
    trace_bars,
    structural_origin,
    full_end,
    bar_steps,
    max_base_bars,
    tolerance_steps,
    horizon_bars,
    global_prior_bars,
    one_bar_margin,
):
    """
    Purely observational diagnostic.

    For explicitly requested source-bar starts, print every local structural
    lattice candidate and show exactly how the existing selector resolves the
    winner.

    This does NOT affect segmentation.
    """
    if not trace_bars:
        return

    one_bar = int(round(bar_steps))

    print("\n=== LOCAL LATTICE CANDIDATE TRACE ===")
    print(
        "Observational only. Each requested bar is rescored independently; "
        "the adaptive decoder state is ignored."
    )
    print(
        "raw = candidate score from score_local_lattice_candidates()."
    )
    print(
        "selected = result of the existing _select_local_lattice() rules "
        "(1-bar protection + whole-song close-tie prior)."
    )

    for bar_number in trace_bars:
        # Source bar 1 begins at absolute step 0.
        start = int(
            round(
                (bar_number - 1)
                * bar_steps
            )
        )

        print()
        print(
            f"--- START {step_to_bar_beat_subdiv(start, bar_steps)} "
            f"(bar {bar_number}, absolute step {start}) ---"
        )

        if start < structural_origin:
            print(
                "SKIPPED: before fixed structural anchor "
                f"{step_to_bar_beat_subdiv(structural_origin, bar_steps)}"
            )
            continue

        if start >= full_end:
            print(
                f"SKIPPED: start is at/after structural end "
                f"{step_to_bar_beat_subdiv(full_end, bar_steps)}"
            )
            continue

        candidates = score_local_lattice_candidates(
            notes,
            boundary_candidates,
            start,
            full_end,
            bar_steps,
            max_base_bars,
            tolerance_steps,
            horizon_bars,
        )

        if not candidates:
            print("No legal local candidates.")
            continue

        # ---------------------------------------------------------------
        # Reconstruct the selector stages explicitly for diagnostics.
        # This mirrors _select_local_lattice() but changes no state.
        # ---------------------------------------------------------------

        # Stage 0: plain highest raw score.
        raw_winner = choose_lattice(
            candidates
        )

        # Stage 1: existing special protection against trivial 1-bar wins.
        after_one_bar = raw_winner
        one_bar_override = False

        if (
            after_one_bar.unit_bars == 1
            and len(candidates) > 1
        ):
            non_one = max(
                (
                    c
                    for c in candidates
                    if c.unit_bars != 1
                ),
                key=lambda c: c.score,
                default=None,
            )

            if (
                non_one is not None
                and after_one_bar.score
                < non_one.score + one_bar_margin
            ):
                after_one_bar = non_one
                one_bar_override = True

        # Stage 2: existing whole-song prior close-tie preference.
        final_selected = after_one_bar
        global_prior_override = False

        by_bars = {
            c.unit_bars: c
            for c in candidates
        }

        if (
            global_prior_bars is not None
            and global_prior_bars in by_bars
        ):
            global_candidate = by_bars[
                global_prior_bars
            ]

            if (
                global_candidate.score
                + GLOBAL_PRIOR_TIE_BONUS
                >= final_selected.score
            ):
                if (
                    global_candidate.unit_bars
                    != final_selected.unit_bars
                ):
                    global_prior_override = True

                final_selected = global_candidate

        # Sanity check against the actual production selector.
        production_selected, production_runner = _select_local_lattice(
            candidates,
            global_prior_bars,
            one_bar_margin,
        )

        if (
            production_selected.unit_bars
            != final_selected.unit_bars
        ):
            print(
                "WARNING: diagnostic selector reconstruction disagrees "
                "with _select_local_lattice()."
            )

        print(
            f"Global prior:       "
            f"{global_prior_bars if global_prior_bars is not None else '-'}b"
        )
        print(
            f"1-bar margin:       {one_bar_margin:.3f}"
        )
        print(
            f"Global tie bonus:   {GLOBAL_PRIOR_TIE_BONUS:.3f}"
        )
        print(
            f"Local horizon:      {horizon_bars} bars"
        )

        print()
        print(
            " bars  range                 raw     boundary  coverage  "
            "recurrence  expected   raw-rank   flags"
        )
        print(
            " ----  --------------------  ------  --------  --------  "
            "----------  --------   --------   -------------------------"
        )

        ranked = sorted(
            candidates,
            key=lambda c: (
                -c.score,
                c.unit_bars,
            ),
        )

        rank_by_bars = {
            c.unit_bars: i + 1
            for i, c in enumerate(ranked)
        }

        for candidate in sorted(
            candidates,
            key=lambda c: c.unit_bars,
        ):
            end = (
                start
                + candidate.unit_steps
            )

            flags = []

            if (
                candidate.unit_bars
                == raw_winner.unit_bars
            ):
                flags.append("RAW_WINNER")

            if (
                candidate.unit_bars
                == final_selected.unit_bars
            ):
                flags.append("SELECTED")

            if (
                global_prior_bars is not None
                and candidate.unit_bars
                == global_prior_bars
            ):
                flags.append("GLOBAL_PRIOR")

            if (
                production_runner is not None
                and candidate.unit_bars
                == production_runner.unit_bars
            ):
                flags.append("RUNNER_UP")

            print(
                f" {candidate.unit_bars:4d}  "
                f"{step_to_bar_beat_subdiv(start, bar_steps):>8s}"
                f" -> "
                f"{step_to_bar_beat_subdiv(end, bar_steps):<8s}  "
                f"{candidate.score:6.3f}  "
                f"{candidate.boundary_support:8.3f}  "
                f"{candidate.coverage:8.3f}  "
                f"{candidate.recurrence:10.3f}  "
                f"{candidate.expected_boundaries:8d}   "
                f"{rank_by_bars[candidate.unit_bars]:8d}   "
                f"{', '.join(flags) if flags else '-'}"
            )

        print()
        print(
            f"Raw winner:         {raw_winner.unit_bars}b "
            f"({raw_winner.score:.3f})"
        )

        if one_bar_override:
            print(
                f"1-bar rule:         changed winner "
                f"{raw_winner.unit_bars}b -> "
                f"{after_one_bar.unit_bars}b"
            )
        else:
            print(
                "1-bar rule:         no change"
            )

        if global_prior_override:
            print(
                f"Global-prior rule:  changed winner "
                f"{after_one_bar.unit_bars}b -> "
                f"{final_selected.unit_bars}b"
            )
        else:
            print(
                "Global-prior rule:  no change"
            )

        print(
            f"FINAL SELECTED:     {final_selected.unit_bars}b "
            f"({final_selected.score:.3f})"
        )

    print()

def print_endpoint_vs_lattice_trace(
    notes,
    boundary_candidates,
    adaptive_decisions,
    full_end,
    bar_steps,
    max_base_bars,
    tolerance_steps,
    horizon_bars,
):
    """
    Purely observational diagnostic.

    At every ACTUAL adaptive segment start, compare:

      A) evidence at the immediate proposed endpoint
      B) the existing multi-boundary local lattice score

    This does NOT influence segmentation.
    """

    one_bar = int(round(bar_steps))

    print("\n=== ENDPOINT VS LATTICE TRACE ===")
    print(
        "At each actual adaptive decision start, compare immediate endpoint "
        "boundary evidence with the existing periodic-grid lattice score."
    )
    print(
        "endpoint = matched boundary score * proximity within lattice tolerance."
    )
    print(
        "No scores or decoder decisions are modified."
    )

    for decision_index, decision in enumerate(adaptive_decisions, start=1):

        # Explicit structural gaps are not melodic span decisions.
        if decision.kind != "segment":
            continue

        start = int(decision.start)

        candidates = score_local_lattice_candidates(
            notes,
            boundary_candidates,
            start,
            full_end,
            bar_steps,
            max_base_bars,
            tolerance_steps,
            horizon_bars,
        )

        by_bars = {
            candidate.unit_bars: candidate
            for candidate in candidates
        }

        print()
        print(
            f"--- DECISION S{decision_index:02d} @ "
            f"{step_to_bar_beat_subdiv(start, bar_steps)} ---"
        )

        print(
            f"Actual emitted span: "
            f"{decision.unit_bars}b  "
            f"{step_to_bar_beat_subdiv(decision.start, bar_steps)}"
            f" -> "
            f"{step_to_bar_beat_subdiv(decision.end, bar_steps)}"
        )

        if decision.established_period_bars is None:
            print("Established base:    -")
        else:
            print(
                f"Established base:    "
                f"{decision.established_period_bars}b"
            )

        print()
        print(
            " bars  proposed end   matched bdry   dist   "
            "bscore   prox   ENDPOINT   LATTICE   "
            "boundary  coverage  recur    flags"
        )
        print(
            " ----  -------------  -------------  -----  "
            "-------  -----  --------   -------   "
            "--------  --------  -------  ----------------"
        )

        endpoint_rows = []

        for unit_bars in range(
            1,
            max_base_bars + 1,
        ):
            proposed_end = (
                start
                + unit_bars * one_bar
            )

            if proposed_end > full_end:
                continue

            candidate = by_bars.get(
                unit_bars
            )

            matched = _candidate_near_step(
                boundary_candidates,
                proposed_end,
                tolerance_steps,
            )

            if matched is None:
                distance = None
                proximity = 0.0
                endpoint_support = 0.0
                boundary_score = 0.0
                matched_label = "-"
            else:
                distance = abs(
                    matched.step
                    - proposed_end
                )

                proximity = max(
                    0.0,
                    1.0
                    - distance
                    / float(
                        tolerance_steps + 1
                    ),
                )

                boundary_score = float(
                    matched.score
                )

                endpoint_support = (
                    boundary_score
                    * proximity
                )

                matched_label = (
                    step_to_bar_beat_subdiv(
                        matched.step,
                        bar_steps,
                    )
                )

            proposed_label = (
                step_to_bar_beat_subdiv(
                    proposed_end,
                    bar_steps,
                )
            )

            lattice_score = (
                candidate.score
                if candidate is not None
                else 0.0
            )

            lattice_boundary = (
                candidate.boundary_support
                if candidate is not None
                else 0.0
            )

            lattice_coverage = (
                candidate.coverage
                if candidate is not None
                else 0.0
            )

            lattice_recurrence = (
                candidate.recurrence
                if candidate is not None
                else 0.0
            )

            endpoint_rows.append(
                (
                    unit_bars,
                    endpoint_support,
                    lattice_score,
                )
            )

            flags = []

            if unit_bars == decision.unit_bars:
                flags.append("EMITTED")

            if (
                decision.raw_unit_bars is not None
                and unit_bars
                == decision.raw_unit_bars
            ):
                flags.append("RAW")

            distance_text = (
                f"{distance:.1f}"
                if distance is not None
                else "-"
            )

            print(
                f" {unit_bars:4d}  "
                f"{proposed_label:>13s}  "
                f"{matched_label:>13s}  "
                f"{distance_text:>5s}  "
                f"{boundary_score:7.3f}  "
                f"{proximity:5.3f}  "
                f"{endpoint_support:8.3f}   "
                f"{lattice_score:7.3f}   "
                f"{lattice_boundary:8.3f}  "
                f"{lattice_coverage:8.3f}  "
                f"{lattice_recurrence:7.3f}  "
                f"{', '.join(flags) if flags else '-'}"
            )

        if not endpoint_rows:
            continue

        endpoint_winner = max(
            endpoint_rows,
            key=lambda row: (
                row[1],
                -row[0],
            ),
        )

        lattice_winner = max(
            endpoint_rows,
            key=lambda row: (
                row[2],
                -row[0],
            ),
        )

        print()
        print(
            f"Strongest endpoint: "
            f"{endpoint_winner[0]}b "
            f"({endpoint_winner[1]:.3f})"
        )
        print(
            f"Strongest numeric lattice score: "
            f"{lattice_winner[0]}b "
            f"({lattice_winner[2]:.3f})"
        )

        if (
            endpoint_winner[0]
            != lattice_winner[0]
        ):
            print(
                ">>> DISAGREEMENT: endpoint evidence and "
                "periodic lattice prefer different span lengths."
            )
        else:
            print(
                "Agreement: endpoint evidence and periodic lattice "
                "prefer the same span length."
            )

# ALPHA SWEEP test
def print_endpoint_alpha_sweep(
    notes,
    boundary_candidates,
    adaptive_decisions,
    full_end,
    bar_steps,
    max_base_bars,
    tolerance_steps,
    horizon_bars,
    global_prior_bars,
    one_bar_margin,
):
    """
    Diagnostic only.

    Re-scores each local lattice candidate as:

        original lattice score + alpha * immediate endpoint support

    Then replays the existing local selector rules:
      - choose_lattice()
      - one-bar protection
      - whole-song prior close-tie preference

    No production scores or decisions are modified.
    """

    one_bar = int(round(bar_steps))

    print("\n=== ENDPOINT ALPHA SWEEP ===")
    print(
        "hypothetical = lattice_score + alpha * endpoint_support"
    )
    print(
        "Existing selector rules are replayed on the hypothetical scores."
    )
    print(
        "No decoder state or segmentation is modified."
    )

    for decision_index, decision in enumerate(
        adaptive_decisions,
        start=1,
    ):
        if decision.kind != "segment":
            continue

        start = int(decision.start)

        candidates = score_local_lattice_candidates(
            notes,
            boundary_candidates,
            start,
            full_end,
            bar_steps,
            max_base_bars,
            tolerance_steps,
            horizon_bars,
        )

        if not candidates:
            continue

        endpoint_supports = {}

        for candidate in candidates:
            proposed_end = (
                start
                + candidate.unit_bars * one_bar
            )

            matched = _candidate_near_step(
                boundary_candidates,
                proposed_end,
                tolerance_steps,
            )

            if matched is None:
                endpoint_support = 0.0
            else:
                distance = abs(
                    matched.step - proposed_end
                )

                proximity = max(
                    0.0,
                    1.0
                    - distance
                    / float(tolerance_steps + 1),
                )

                endpoint_support = (
                    matched.score
                    * proximity
                )

            endpoint_supports[
                candidate.unit_bars
            ] = endpoint_support

        print()
        print(
            f"--- S{decision_index:02d} "
            f"@ {step_to_bar_beat_subdiv(start, bar_steps)} "
            f"(actual {decision.unit_bars}b) ---"
        )

        print(
            " alpha    1b       2b       3b       4b       selected"
        )
        print(
            " -----   -------  -------  -------  -------   --------"
        )

        for alpha in ENDPOINT_ALPHA_SWEEP:

            hypothetical_candidates = []

            for candidate in candidates:
                endpoint_support = (
                    endpoint_supports[
                        candidate.unit_bars
                    ]
                )

                hypothetical_score = clip01(
                    candidate.score
                    + alpha * endpoint_support
                )

                hypothetical_candidates.append(
                    replace(
                        candidate,
                        score=hypothetical_score,
                    )
                )

            selected, _ = _select_local_lattice(
                hypothetical_candidates,
                global_prior_bars,
                one_bar_margin,
            )

            by_bars = {
                candidate.unit_bars:
                    candidate.score
                for candidate
                in hypothetical_candidates
            }

            values = []

            for bars in range(
                1,
                max_base_bars + 1,
            ):
                if bars in by_bars:
                    values.append(
                        f"{by_bars[bars]:7.3f}"
                    )
                else:
                    values.append(
                        "      -"
                    )

            marker = ""

            if (
                selected.unit_bars
                != decision.unit_bars
            ):
                marker = "  *"

            print(
                f" {alpha:5.3f}   "
                + "  ".join(values)
                + f"   {selected.unit_bars}b"
                + marker
            )

        print(
            "* = hypothetical selection differs "
            "from current emitted span"
        )

# bounded two-decision path inspection
def print_two_decision_path_trace(
    notes,
    boundary_candidates,
    full_end,
    bar_steps,
    max_base_bars,
    tolerance_steps,
    horizon_bars,
    global_prior_bars,
    one_bar_margin,
    start_bars,
):
    """
    Purely observational two-decision path diagnostic.

    For every requested structural start:

      1. score all possible first spans (1..max_base_bars);
      2. pretend each one was committed;
      3. from that hypothetical endpoint, score a fresh local decision;
      4. print the full second-step candidate table and the selector's
         preferred second span.

    IMPORTANT:
      - no decoder state is modified;
      - no period is established;
      - no phase protection is applied;
      - no future-confirmation logic is applied;
      - no segmentation is changed.

    This specifically tests whether a locally attractive first decision leads
    to a structurally weaker or stronger immediate continuation.
    """

    one_bar = int(round(bar_steps))

    print("\n=== TWO-DECISION PATH TRACE ===")
    print(
        "Observational only: each possible first span is followed by "
        "one fresh hypothetical local decision."
    )
    print(
        "No period state, phase protection, establishment, switching, "
        "or decoder mutation is applied."
    )

    for start_bar in start_bars:
        start = int(
            round(
                (start_bar - 1)
                * bar_steps
            )
        )

        if start >= full_end:
            continue

        first_candidates = score_local_lattice_candidates(
            notes,
            boundary_candidates,
            start,
            full_end,
            bar_steps,
            max_base_bars,
            tolerance_steps,
            horizon_bars,
        )

        if not first_candidates:
            continue

        first_selected, _ = _select_local_lattice(
            first_candidates,
            global_prior_bars,
            one_bar_margin,
        )

        print()
        print(
            "================================================================"
        )
        print(
            f"START {step_to_bar_beat_subdiv(start, bar_steps)} "
            f"(source bar {start_bar})"
        )
        print(
            f"Current local selector at this start: "
            f"{first_selected.unit_bars}b "
            f"(score={first_selected.score:.3f})"
        )
        print(
            "================================================================"
        )

        ranked_first = sorted(
            first_candidates,
            key=lambda c: c.unit_bars,
        )

        for first in ranked_first:
            first_end = (
                start
                + first.unit_bars * one_bar
            )

            print()
            print(
                f"--- FIRST = {first.unit_bars}b: "
                f"{step_to_bar_beat_subdiv(start, bar_steps)}"
                f" -> "
                f"{step_to_bar_beat_subdiv(first_end, bar_steps)} ---"
            )

            print(
                f"first score={first.score:.3f} "
                f"boundary={first.boundary_support:.3f} "
                f"coverage={first.coverage:.3f} "
                f"recurrence={first.recurrence:.3f}"
            )

            if first_end >= full_end:
                print(
                    "second decision: unavailable "
                    "(first span reaches structural end)"
                )
                continue

            second_candidates = score_local_lattice_candidates(
                notes,
                boundary_candidates,
                first_end,
                full_end,
                bar_steps,
                max_base_bars,
                tolerance_steps,
                horizon_bars,
            )

            if not second_candidates:
                print(
                    "second decision: no valid candidates"
                )
                continue

            second_selected, second_runner_up = (
                _select_local_lattice(
                    second_candidates,
                    global_prior_bars,
                    one_bar_margin,
                )
            )

            print(
                " second   endpoint                 score   boundary  "
                "coverage  recurrence"
            )
            print(
                " ------   -----------------------  ------  --------  "
                "--------  ----------"
            )

            for second in sorted(
                second_candidates,
                key=lambda c: c.unit_bars,
            ):
                second_end = (
                    first_end
                    + second.unit_bars * one_bar
                )

                marker = (
                    "*"
                    if second.unit_bars
                    == second_selected.unit_bars
                    else " "
                )

                print(
                    f" {marker}{second.unit_bars:>4d}b   "
                    f"{step_to_bar_beat_subdiv(first_end, bar_steps):>9s}"
                    f" -> "
                    f"{step_to_bar_beat_subdiv(second_end, bar_steps):<9s}  "
                    f"{second.score:6.3f}  "
                    f"{second.boundary_support:8.3f}  "
                    f"{second.coverage:8.3f}  "
                    f"{second.recurrence:10.3f}"
                )

            second_end = (
                first_end
                + second_selected.unit_bars * one_bar
            )

            runner_text = (
                f"{second_runner_up.unit_bars}b/"
                f"{second_runner_up.score:.3f}"
                if second_runner_up is not None
                else "-"
            )

            print(
                f" => second selector: "
                f"{second_selected.unit_bars}b "
                f"{step_to_bar_beat_subdiv(first_end, bar_steps)}"
                f" -> "
                f"{step_to_bar_beat_subdiv(second_end, bar_steps)} "
                f"(score={second_selected.score:.3f}; "
                f"runner-up={runner_text})"
            )

def _period_switch_confirmed(
    notes,
    boundary_candidates,
    proposed_bars,
    proposed_end,
    full_end,
    bar_steps,
    max_base_bars,
    tolerance_steps,
    horizon_bars,
    global_prior_bars,
    one_bar_margin,
):
    """
    Confirm a genuinely incompatible period change with one future probe.

    Example:
        established base = 2 bars
        current proposal = 3 bars

    The 3-bar proposal is allowed to re-phase the song only if a local
    re-analysis from its hypothetical endpoint also supports 3 (or a multiple
    of 3). A single anomalous 3-bar decision therefore cannot contaminate all
    downstream boundaries.

    Multiples of the established period never come through this function:
    e.g. 2 -> 4 is allowed directly as a macro-sized span while the 2-bar base
    remains established.
    """
    if proposed_end >= full_end:
        return False, None

    confirmations = 0
    probe_start = int(proposed_end)
    last_probe_bars = None

    for _ in range(PERIOD_SWITCH_CONFIRMATIONS):
        probe_candidates = score_local_lattice_candidates(
            notes,
            boundary_candidates,
            probe_start,
            full_end,
            bar_steps,
            max_base_bars,
            tolerance_steps,
            horizon_bars,
        )

        if not probe_candidates:
            return False, last_probe_bars

        probe_choice, _ = _select_local_lattice(
            probe_candidates,
            None,  # no whole-song prior while confirming a local period change
            one_bar_margin,
        )

        last_probe_bars = int(
            probe_choice.unit_bars
        )

        if last_probe_bars == proposed_bars:
            confirmations += 1
            probe_start += (
                proposed_bars
                * int(round(bar_steps))
            )
        else:
            return False, last_probe_bars

    return (
        confirmations
        >= PERIOD_SWITCH_CONFIRMATIONS,
        last_probe_bars,
    )

def _period_establishment_confirmed(
    notes,
    boundary_candidates,
    proposed_bars,
    proposed_end,
    full_end,
    bar_steps,
    max_base_bars,
    tolerance_steps,
    horizon_bars,
    one_bar_margin,
):
    """
    Confirm a candidate fundamental period before establishing it.

    Repeated equal emitted spans are only a PERIOD CANDIDATE.

    Establishment requires future local evidence from the hypothetical endpoint
    of the most recent repeated span.

    Critical rules:
      * probes are unbiased by the whole-song/global prior;
      * the future local winner must EXACTLY equal proposed_bars;
      * an integer multiple does NOT confirm the proposed fundamental;
      * absence of future evidence does NOT confirm anything.

    Example:
        emitted 11->14 = 3 bars
        emitted 14->17 = 3 bars

        candidate fundamental = 3 bars

        unbiased probe at 17:
            if local winner == 3 bars:
                establishment may proceed
            if local winner == 2 or 4 bars:
                reject establishment
    """
    if proposed_end >= full_end:
        return False, None

    confirmations = 0
    probe_start = int(proposed_end)
    last_probe_bars = None
    one_bar = int(round(bar_steps))

    for _ in range(PERIOD_ESTABLISH_CONFIRMATIONS):
        # There must be enough room to test the proposed period itself.
        if (
            probe_start
            + proposed_bars * one_bar
            > full_end
        ):
            return False, last_probe_bars

        probe_candidates = score_local_lattice_candidates(
            notes,
            boundary_candidates,
            probe_start,
            full_end,
            bar_steps,
            max_base_bars,
            tolerance_steps,
            horizon_bars,
        )

        if not probe_candidates:
            return False, last_probe_bars

        probe_choice, _ = _select_local_lattice(
            probe_candidates,
            None,  # IMPORTANT: no whole-song prior during confirmation
            one_bar_margin,
        )

        last_probe_bars = int(
            probe_choice.unit_bars
        )

        # Exact match only.
        #
        # A 4-bar winner does NOT confirm a proposed 2-bar fundamental,
        # and a 2-bar winner certainly does not confirm a proposed 3-bar
        # fundamental.
        if last_probe_bars != proposed_bars:
            return False, last_probe_bars

        confirmations += 1
        probe_start += (
            proposed_bars
            * one_bar
        )

    return (
        confirmations
        >= PERIOD_ESTABLISH_CONFIRMATIONS,
        last_probe_bars,
    )


@dataclass
class StructuralGap:
    start: int
    end: int
    raw_silence_start: float
    raw_silence_end: float
    next_onset: float


def _ceil_barline(step: float, bar_steps: float) -> int:
    q = step / float(bar_steps)
    nearest = round(q)
    if abs(q - nearest) < 1e-7:
        return int(round(nearest * bar_steps))
    return int(math.ceil(q) * round(bar_steps))


def _floor_barline(step: float, bar_steps: float) -> int:
    return int(math.floor((step + 1e-7) / float(bar_steps)) * round(bar_steps))

def _merge_temporal_activity_clusters(
    notes,
    start,
    end,
    merge_gap_steps=GAP_TAIL_CLUSTER_MERGE_STEPS,
):
    """
    Merge raw note detections into temporal activity clusters.

    This is intentionally pitch- and velocity-agnostic.

    BasicPitch may represent one sung note / terminal gesture as several
    overlapping or closely spaced MIDI notes. For structural gap detection,
    those fragments should count as one continuous burst of melodic activity,
    not as several independent phrase events.

    Returns:
        List of (cluster_start, cluster_end, notes_in_cluster)
    """
    relevant = sorted(
        [
            n
            for n in notes
            if (
                n.end > start + EPS
                and n.start < end - EPS
            )
        ],
        key=lambda n: (
            n.start,
            n.end,
            n.pitch,
        ),
    )

    if not relevant:
        return []

    clusters = []

    current_notes = [relevant[0]]
    current_start = max(
        float(start),
        float(relevant[0].start),
    )
    current_end = float(
        relevant[0].end
    )

    for note in relevant[1:]:
        note_start = float(note.start)
        note_end = float(note.end)

        gap = max(
            0.0,
            note_start - current_end,
        )

        if gap <= merge_gap_steps + EPS:
            current_notes.append(note)
            current_end = max(
                current_end,
                note_end,
            )
            continue

        clusters.append(
            (
                current_start,
                current_end,
                current_notes,
            )
        )

        current_notes = [note]
        current_start = max(
            float(start),
            note_start,
        )
        current_end = note_end

    clusters.append(
        (
            current_start,
            current_end,
            current_notes,
        )
    )

    return clusters


def _choose_gap_start_boundary(
    notes,
    silence_start,
    structural_origin,
    bar_steps,
    merge_gap_steps=GAP_TAIL_CLUSTER_MERGE_STEPS,
    max_cluster_end_steps=GAP_TAIL_CLUSTER_MAX_END_STEPS,
):
    """
    Choose the STRUCTURAL start of a proven long melody-free gap.

    Literal silence and structural silence are deliberately separate.

    Normally:
        structural GAP start = first barline at/after raw silence start

    But raw BasicPitch MIDI may fragment one phrase-ending sung gesture into
    several short detections immediately after the preceding barline.

    We therefore inspect TEMPORAL ACTIVITY between the previous barline and
    the literal silence onset.

    Snap the structural GAP start backward to the previous barline only when:

      * all post-barline activity before the long silence forms ONE temporal
        activity cluster after merging overlaps / tiny detector gaps;

      * that cluster ends near the beginning of the bar;

      * no second separated activity cluster exists before the long silence.

    Pitch and velocity are deliberately ignored.

    raw_silence_start remains unchanged elsewhere, so diagnostics retain the
    literal end of detected vocal activity.
    """
    normal_start = _ceil_barline(
        silence_start,
        bar_steps,
    )

    previous_barline = _floor_barline(
        silence_start,
        bar_steps,
    )

    previous_barline = max(
        int(structural_origin),
        int(previous_barline),
    )

    # Nothing to snap if the literal silence already begins at the barline.
    if abs(
        float(silence_start)
        - float(previous_barline)
    ) <= EPS:
        return int(previous_barline)

    # Never inspect beyond the candidate bar itself.
    candidate_bar_end = (
        float(previous_barline)
        + float(bar_steps)
    )

    clusters = _merge_temporal_activity_clusters(
        notes,
        float(previous_barline),
        min(
            float(silence_start) + EPS,
            candidate_bar_end,
        ),
        merge_gap_steps=merge_gap_steps,
    )

    # No detected activity after the barline: ordinary snapping to that
    # barline is safe.
    if not clusters:
        return int(previous_barline)

    # More than one temporally distinct burst is evidence of actual melodic
    # activity inside the bar, not merely one fragmented terminal gesture.
    if len(clusters) != 1:
        return int(normal_start)

    cluster_start, cluster_end, cluster_notes = clusters[0]

    cluster_end_offset = (
        float(cluster_end)
        - float(previous_barline)
    )

    # The one activity burst must terminate near the beginning of the bar.
    # We deliberately constrain its END, not its raw note count.
    if cluster_end_offset > max_cluster_end_steps + EPS:
        return int(normal_start)

    # The cluster must actually account for the activity leading into the
    # detected long silence. This guards against accidentally snapping because
    # of an unrelated early-bar fragment followed by other unrepresented time.
    if (
        float(silence_start)
        - float(cluster_end)
        > merge_gap_steps + EPS
    ):
        return int(normal_start)

    return int(previous_barline)

def _choose_gap_restart_boundary(
    notes,
    boundary_candidates,
    next_onset,
    gap_start,
    full_end,
    bar_steps,
    max_base_bars,
    tolerance_steps,
    horizon_bars,
):
    """
    End a long structural gap at the first barline at/after the returning onset.

    This is deliberately conservative about pickups: after a long melody-free
    interval, the first returning melody note is allowed to anticipate the next
    structural boundary.  The note onset therefore does NOT pull the structural
    restart backward into its bar.

    If the returning onset is exactly on a barline, that barline is used.
    """
    restart = _ceil_barline(next_onset, bar_steps)
    restart = max(
        int(gap_start + round(bar_steps)),
        int(restart),
    )
    return min(int(full_end), int(restart))


def detect_structural_gaps(
    notes,
    boundary_candidates,
    structural_origin,
    full_end,
    bar_steps,
    max_base_bars,
    tolerance_steps,
    horizon_bars,
    gap_min_bars,
):
    """
    Detect long melody-free intervals and represent them explicitly.

    A gap is based on actual absence of melody notes, not low note density.

    Normally its structural start is the first barline at/after the last release.
    A very short phrase-ending tail immediately after the preceding barline may
    instead be absorbed into that structural boundary, while raw_silence_start
    continues to preserve the literal acoustic silence onset.

    Its end is the restart barline at/after the first returning onset, so a
    returning pickup may live before the structural restart boundary.
    """
    if not notes:
        return []

    min_steps = float(gap_min_bars) * float(bar_steps)
    ordered = sorted(notes, key=lambda n: (n.start, n.end, n.pitch))
    gaps = []

    active_end = max(structural_origin, ordered[0].end)
    for note in ordered[1:]:
        if note.start <= active_end + EPS:
            active_end = max(active_end, note.end)
            continue

        silence_start = float(active_end)
        silence_end = float(note.start)
        if silence_end - silence_start >= min_steps:
            gap_start = _choose_gap_start_boundary(
                notes,
                silence_start,
                structural_origin,
                bar_steps,
            )
            gap_start = max(
                int(structural_origin),
                int(gap_start),
            )
            if gap_start < full_end:
                gap_end = _choose_gap_restart_boundary(
                    notes,
                    boundary_candidates,
                    silence_end,
                    gap_start,
                    full_end,
                    bar_steps,
                    max_base_bars,
                    tolerance_steps,
                    horizon_bars,
                )
                if gap_end > gap_start:
                    gaps.append(
                        StructuralGap(
                            start=int(gap_start),
                            end=int(gap_end),
                            raw_silence_start=silence_start,
                            raw_silence_end=silence_end,
                            next_onset=silence_end,
                        )
                    )

        active_end = max(active_end, note.end)

    # Merge overlaps just in case malformed transcription creates adjacent gaps.
    merged = []
    for g in sorted(gaps, key=lambda x: (x.start, x.end)):
        if merged and g.start <= merged[-1].end:
            prev = merged[-1]
            prev.end = max(prev.end, g.end)
            prev.raw_silence_end = max(prev.raw_silence_end, g.raw_silence_end)
            prev.next_onset = max(prev.next_onset, g.next_onset)
        else:
            merged.append(g)
    return merged


def infer_adaptive_span_path(
    notes,
    boundary_candidates,
    lattice_candidates,
    global_lattice,
    origin,
    total_steps,
    bar_steps,
    max_base_bars,
    tolerance_steps,
    horizon_bars,
    switch_margin,
    one_bar_margin,
    gap_min_bars,
):
    """
    v9 piecewise structural decoding with explicit long-gap handling.

    The crucial state separation is:

        emitted span length != established fundamental period

    Once a period is established by repeated compatible decisions:
      * the same period is naturally allowed;
      * integer MULTIPLES are allowed as larger structural objects while the
        established fundamental and phase remain unchanged;
      * an incompatible proposal (e.g. established 2 bars -> proposed 3 bars)
        must be confirmed by future local evidence before it is allowed to
        re-phase the song;
      * a final shorter remainder is allowed when there is not enough complete
        material left for the established period.

    This preserves a legitimate 4-bar object such as 11->15 in Ochiitai while
    preventing an unconfirmed 15->18 three-bar proposal from shifting every
    subsequent boundary.
    """
    one_bar = int(
        round(bar_steps)
    )

    if one_bar <= 0:
        raise ValueError(
            "Invalid bar length."
        )

    full_end = origin
    while (
        full_end + one_bar
        <= total_steps
    ):
        full_end += one_bar

    if full_end <= origin:
        return [], full_end

    gaps = detect_structural_gaps(
        notes,
        boundary_candidates,
        origin,
        full_end,
        bar_steps,
        max_base_bars,
        tolerance_steps,
        horizon_bars,
        gap_min_bars,
    )

    decisions = []
    current = int(origin)

    global_prior_bars = (
        global_lattice.unit_bars
        if global_lattice is not None
        else None
    )

    established_period = None
    establishment_candidate = None
    establishment_count = 0

    while current < full_end:
        # A long melody-free interval terminates the current active melodic
        # region.  Do not allow a span hypothesis to cross it.
        gap_here = next(
            (g for g in gaps if g.start == current),
            None,
        )
        if gap_here is not None:
            gap_bars = max(1, int(round((gap_here.end - gap_here.start) / float(one_bar))))
            decisions.append(
                AdaptiveSpanDecision(
                    start=int(gap_here.start),
                    end=int(gap_here.end),
                    unit_bars=gap_bars,
                    kind="gap",
                    local_score=0.0,
                    boundary_support=0.0,
                    coverage=0.0,
                    recurrence_support=0.0,
                    raw_unit_bars=None,
                    established_period_bars=None,
                    decision_reason="structural-gap; period-state-reset",
                )
            )
            current = int(gap_here.end)
            established_period = None
            establishment_candidate = None
            establishment_count = 0
            continue

        next_gap = next(
            (g for g in gaps if g.start > current),
            None,
        )
        region_end = (
            int(next_gap.start)
            if next_gap is not None
            else int(full_end)
        )

        remaining_bars = int(
            round(
                (region_end - current)
                / float(one_bar)
            )
        )

        candidates = score_local_lattice_candidates(
            notes,
            boundary_candidates,
            current,
            region_end,
            bar_steps,
            max_base_bars,
            tolerance_steps,
            horizon_bars,
        )

        if not candidates:
            # Consume any complete-bar remainder up to a structural gap without
            # inventing a period change.
            if region_end > current:
                rem_bars = int(round((region_end - current) / float(one_bar)))
                if rem_bars > 0:
                    decisions.append(
                        AdaptiveSpanDecision(
                            start=int(current),
                            end=int(region_end),
                            unit_bars=rem_bars,
                            kind="segment",
                            local_score=0.0,
                            boundary_support=0.0,
                            coverage=0.0,
                            recurrence_support=0.0,
                            raw_unit_bars=None,
                            established_period_bars=(
                                int(established_period) if established_period is not None else None
                            ),
                            decision_reason="pre-gap-remainder",
                        )
                    )
                    current = int(region_end)
                    continue
            break

        by_bars = {
            c.unit_bars: c
            for c in candidates
        }

        raw_chosen, raw_runner = _select_local_lattice(
            candidates,
            global_prior_bars,
            one_bar_margin,
        )

        chosen = raw_chosen
        reason = "local"
        switch_probe_bars = None

        if established_period is not None:
            established_candidate_local = by_bars.get(
                established_period
            )

            # Not enough complete bars remain to emit the established period:
            # allow the terminal complete-bar remainder without interpreting it
            # as a period change.
            if remaining_bars < established_period:
                legal_bars = min(
                    remaining_bars,
                    max_base_bars,
                )
                chosen = by_bars.get(
                    legal_bars,
                    raw_chosen,
                )
                reason = "terminal-remainder"

            elif raw_chosen.unit_bars == established_period:
                chosen = raw_chosen
                reason = "established-base"

            elif (
                raw_chosen.unit_bars > established_period
                and raw_chosen.unit_bars
                % established_period == 0
            ):
                # Important v8 behaviour:
                # a 4-bar span may be a perfectly valid larger object in a
                # 2-bar-based region. Emit it, but DO NOT change the base.
                chosen = raw_chosen
                reason = (
                    f"macro-multiple-of-{established_period}b"
                )

            else:
                # Incompatible candidate would move the structural phase.
                # It must first be competitive with the established base.
                competitive = (
                    established_candidate_local is None
                    or raw_chosen.score
                    >= (
                        established_candidate_local.score
                        - PERIOD_SWITCH_SCORE_TOLERANCE
                    )
                )

                proposed_end = (
                    current
                    + raw_chosen.unit_steps
                )

                confirmed = False
                if competitive:
                    confirmed, switch_probe_bars = _period_switch_confirmed(
                        notes,
                        boundary_candidates,
                        int(raw_chosen.unit_bars),
                        int(proposed_end),
                        region_end,
                        bar_steps,
                        max_base_bars,
                        tolerance_steps,
                        horizon_bars,
                        global_prior_bars,
                        one_bar_margin,
                    )

                if confirmed:
                    chosen = raw_chosen
                    reason = (
                        f"confirmed-period-change-"
                        f"{established_period}b->{raw_chosen.unit_bars}b"
                    )
                    established_period = int(
                        raw_chosen.unit_bars
                    )
                    establishment_candidate = (
                        established_period
                    )
                    establishment_count = (
                        PERIOD_ESTABLISH_REPEATS
                    )
                elif established_candidate_local is not None:
                    chosen = established_candidate_local
                    reason = (
                        f"phase-protected-{established_period}b"
                    )
                else:
                    chosen = raw_chosen
                    reason = "local-no-base-candidate"

        # Before a base is established, repeated identical emitted spans create
        # only a CANDIDATE period.
        #
        # The candidate must then be confirmed by an unbiased future local probe.
        # This prevents two accidentally equal spans from crystallising into a
        # false fundamental and phase-locking the rest of the region.
        if established_period is None:
            observed = int(
                chosen.unit_bars
            )

            if (
                establishment_candidate
                == observed
            ):
                establishment_count += 1
            else:
                establishment_candidate = observed
                establishment_count = 1

            if (
                establishment_count
                >= PERIOD_ESTABLISH_REPEATS
            ):
                proposed_end = (
                    current
                    + chosen.unit_steps
                )

                establishment_confirmed, establishment_probe_bars = (
                    _period_establishment_confirmed(
                        notes,
                        boundary_candidates,
                        observed,
                        int(proposed_end),
                        region_end,
                        bar_steps,
                        max_base_bars,
                        tolerance_steps,
                        horizon_bars,
                        one_bar_margin,
                    )
                )

                # Reuse the existing diagnostic probe field. It already means
                # "future local period observed by a confirmation probe".
                if establishment_probe_bars is not None:
                    switch_probe_bars = int(
                        establishment_probe_bars
                    )

                if establishment_confirmed:
                    established_period = observed
                    reason += (
                        f"; establish-{observed}b-confirmed"
                    )
                else:
                    reason += (
                        f"; candidate-{observed}b-unconfirmed"
                    )

                    # Do not allow the same already-rejected pair to trigger an
                    # establishment attempt forever. Keep the most recent span as
                    # the first observation of a fresh candidate run.
                    establishment_candidate = observed
                    establishment_count = 1


        # If a macro multiple was emitted, established_period deliberately
        # remains unchanged.
        end = (
            current
            + chosen.unit_steps
        )

        if end > region_end:
            break

        ranked_after = sorted(
            (
                c
                for c in candidates
                if c.unit_bars
                != chosen.unit_bars
            ),
            key=lambda c: (
                -c.score,
                c.unit_bars,
            ),
        )
        runner_up = (
            ranked_after[0]
            if ranked_after
            else None
        )

        decisions.append(
            AdaptiveSpanDecision(
                start=int(current),
                end=int(end),
                unit_bars=int(
                    chosen.unit_bars
                ),
                kind="segment",
                local_score=float(
                    chosen.score
                ),
                boundary_support=float(
                    chosen.boundary_support
                ),
                coverage=float(
                    chosen.coverage
                ),
                recurrence_support=float(
                    chosen.recurrence
                ),
                runner_up_bars=(
                    int(
                        runner_up.unit_bars
                    )
                    if runner_up is not None
                    else None
                ),
                runner_up_score=(
                    float(
                        runner_up.score
                    )
                    if runner_up is not None
                    else 0.0
                ),
                raw_unit_bars=int(
                    raw_chosen.unit_bars
                ),
                established_period_bars=(
                    int(established_period)
                    if established_period is not None
                    else None
                ),
                decision_reason=reason,
                switch_probe_bars=(
                    int(switch_probe_bars)
                    if switch_probe_bars is not None
                    else None
                ),
            )
        )

        current = int(
            end
        )

    return (
        decisions,
        full_end,
    )




def _build_structural_span(
    notes,
    index,
    start,
    end,
    bar_steps,
    kind="segment",
):
    seg = build_segment(
        index,
        int(start),
        int(end),
        notes,
        bar_steps,
    )

    # A gap is an explicit absence interval; it has no pickup/entrance of its
    # own. For melodic spans, a pickup may begin before the immutable boundary.
    pickup = (
        None
        if kind == "gap"
        else _possible_pickup_start(
            notes,
            int(start),
            bar_steps,
        )
    )
    pickup_steps = (
        float(start) - pickup
        if pickup is not None
        else 0.0
    )

    # Entrance timing is descriptive only. It NEVER moves the structure.
    if pickup is not None:
        entrance_start = float(pickup)
    elif seg.notes:
        entrance_start = float(
            min(n.start for n in seg.notes)
        )
    else:
        entrance_start = None

    entrance_offset = (
        entrance_start - float(start)
        if entrance_start is not None
        else 0.0
    )

    if seg.notes:
        last_release = max(
            n.end
            for n in seg.notes
        )
        trailing = max(
            0.0,
            float(end) - last_release,
        )
    else:
        trailing = float(
            end - start
        )

    return StructuralSpan(
        index,
        int(start),
        int(end),
        seg,
        kind,
        pickup,
        pickup_steps,
        entrance_start,
        entrance_offset,
        trailing,
    )


def make_structural_spans_adaptive(
    notes,
    decisions,
    full_end,
    total_steps,
    bar_steps,
):
    """Materialize melodic spans plus explicit long structural gaps."""
    spans = []
    melodic_index = 0
    gap_index = 0

    for decision in decisions:
        if decision.kind == "gap":
            gap_index += 1
            # Use no melody notes for a GAP even if the restart pickup begins
            # before its structural end.
            gap_span = _build_structural_span(
                [],
                gap_index,
                decision.start,
                decision.end,
                bar_steps,
                kind="gap",
            )
            spans.append(gap_span)
        else:
            melodic_index += 1
            spans.append(
                _build_structural_span(
                    notes,
                    melodic_index,
                    decision.start,
                    decision.end,
                    bar_steps,
                    kind="segment",
                )
            )

    if full_end < total_steps:
        melodic_index += 1
        spans.append(
            _build_structural_span(
                notes,
                melodic_index,
                int(full_end),
                int(total_steps),
                bar_steps,
                kind="segment",
            )
        )

    return spans



def print_adaptive_lattice_path(
    decisions,
    bar_steps,
):
    print(
        "\n=== ADAPTIVE STRUCTURAL LATTICE ==="
    )
    print(
        "Bar 3 beat 1 remains fixed. Local span length and established "
        "fundamental period are tracked separately."
    )
    print(
        "Integer multiples may form larger structural objects without changing "
        "the established base or phase."
    )
    print(
        "An incompatible period change must be confirmed by future local "
        "evidence before it is allowed to re-phase the song."
    )

    if not decisions:
        print(
            "(no complete post-anchor structural bars)"
        )
        return

    print(
        "\n span  structural range                  bars  raw  base  score   boundary  coverage  recurrence  runner-up  decision"
    )
    print(
        " ----  --------------------------------  ----  ---  ----  ------  --------  --------  ----------  ---------  -----------------------------"
    )

    melodic_counter = 0
    gap_counter = 0

    for d in decisions:
        if d.kind == "gap":
            gap_counter += 1
            print(
                f" G{gap_counter:02d}   "
                f"{step_to_bar_beat_subdiv(d.start, bar_steps):>12s} -> "
                f"{step_to_bar_beat_subdiv(d.end, bar_steps):<12s}   "
                f"{d.unit_bars:4d}  "
                f"  -     -       -         -         -           -          -  "
                f"{d.decision_reason}"
            )
            continue

        melodic_counter += 1
        runner = (
            f"{d.runner_up_bars}b/{d.runner_up_score:.3f}"
            if d.runner_up_bars is not None
            else "-"
        )
        base = (
            f"{d.established_period_bars}b"
            if d.established_period_bars is not None
            else "-"
        )
        raw = (
            f"{d.raw_unit_bars}b"
            if d.raw_unit_bars is not None
            else "-"
        )
        probe = (
            f"; probe={d.switch_probe_bars}b"
            if d.switch_probe_bars is not None
            else ""
        )

        print(
            f" S{melodic_counter:02d}   "
            f"{step_to_bar_beat_subdiv(d.start, bar_steps):>12s} -> "
            f"{step_to_bar_beat_subdiv(d.end, bar_steps):<12s}   "
            f"{d.unit_bars:4d}  "
            f"{raw:>3s}  "
            f"{base:>4s}  "
            f"{d.local_score:6.3f}  "
            f"{d.boundary_support:8.3f}  "
            f"{d.coverage:8.3f}  "
            f"{d.recurrence_support:10.3f}  "
            f"{runner:>9s}  "
            f"{d.decision_reason}{probe}"
        )





def _macro_windows_from_spans(
    spans,
    max_macro_bars,
    bar_steps,
):
    """
    Build every macro made from TWO OR MORE COMPLETE ADJACENT structural spans,
    up to max_macro_bars total length.

    Variable base-span lengths are allowed. The final truncated residual tail is
    excluded automatically because its duration is not an integer number of
    bars.
    """
    windows_by_length = {}

    for i in range(len(spans)):
        start = spans[i].start

        if spans[i].kind == "gap":
            continue

        for j in range(i + 1, len(spans)):
            group = spans[i:j + 1]

            if any(s.kind == "gap" for s in group):
                break

            if any(
                group[k].end != group[k + 1].start
                for k in range(len(group) - 1)
            ):
                break

            end = group[-1].end
            length_steps = end - start
            length_bars_float = (
                length_steps / bar_steps
            )

            # Residual partial tails must not become macro evidence.
            nearest = round(
                length_bars_float
            )
            if abs(
                length_bars_float - nearest
            ) > 1e-6:
                break

            length_bars = int(nearest)

            if length_bars > max_macro_bars:
                break

            all_notes = []
            for s in group:
                all_notes.extend(
                    s.segment.notes
                )

            unique = {}
            for n in all_notes:
                key = (
                    n.pitch,
                    n.start_tick,
                    n.end_tick,
                    n.velocity,
                )
                unique[key] = n

            macro_notes = sorted(
                unique.values(),
                key=lambda n: (
                    n.start,
                    n.pitch,
                    n.end,
                ),
            )

            segment = build_segment(
                i + 1,
                int(start),
                int(end),
                macro_notes,
                bar_steps,
            )

            windows_by_length.setdefault(
                length_bars,
                [],
            ).append(
                segment
            )

    return windows_by_length


def find_macro_matches(
    spans,
    bar_steps,
    max_macro_bars,
    threshold,
):
    """
    Compare only macros made from complete adjacent structural spans.

    Unlike v4, adaptive structural spans may have different lengths. Macro
    candidates therefore compete only with other complete-span macros having
    the same TOTAL bar length.
    """
    if not spans:
        return []

    windows_by_length = _macro_windows_from_spans(
        spans,
        max_macro_bars,
        bar_steps,
    )

    matches = []

    for length_bars, windows in sorted(
        windows_by_length.items()
    ):
        for i, left in enumerate(
            windows
        ):
            for right in windows[i + 1:]:
                if not (
                    left.end <= right.start
                    or right.end <= left.start
                ):
                    continue

                sim = compare_segments(
                    left,
                    right,
                )

                if sim.overall >= threshold:
                    matches.append(
                        MacroMatch(
                            length_bars,
                            left.start,
                            left.end,
                            right.start,
                            right.end,
                            sim,
                        )
                    )

    return sorted(
        matches,
        key=lambda m: (
            -m.similarity.overall,
            -m.length_bars,
            m.left_start,
            m.right_start,
        ),
    )



def _span_labels_for_range(spans,start,end):
    labels=[s.label for s in spans if s.kind != "gap" and s.start >= start and s.end <= end]
    return "+".join(labels) if labels else "-"


def print_lattice_candidates(candidates,chosen,bar_steps):
    print("\n=== STRUCTURAL LATTICE CANDIDATES ===")
    print(f"Structural anchor: {step_to_bar_beat_subdiv(chosen.origin,bar_steps)} "
          f"(FIXED; local step 0 for mel_to_chord)")
    print(" bars  steps  score  boundary  coverage  recurrence  expected")
    print(" ----  -----  -----  --------  --------  ----------  --------")
    for c in candidates:
        mark='*' if c.unit_bars==chosen.unit_bars else ' '
        print(f"{mark}{c.unit_bars:4d}  {c.unit_steps:5d}  "
              f"{c.score:5.3f}  {c.boundary_support:8.3f}  {c.coverage:8.3f}  "
              f"{c.recurrence:10.3f}  {c.expected_boundaries:8d}")
    raw_best = max(candidates, key=lambda c: c.score)

    print(f"\nGlobal prior unit: {chosen.unit_bars} bar(s) / {chosen.unit_steps} steps")

    if chosen.unit_bars != raw_best.unit_bars:
        print(
            f"Raw strongest periodicity: {raw_best.unit_bars} bar(s) "
            f"/ {raw_best.unit_steps} steps, score={raw_best.score:.3f}"
        )
        print(
            "Fundamental-period correction selected the shorter divisor "
            "because it explains the stronger higher-order periodicity."
        )

    print("The structural origin is NOT inferred. Bar 3 beat 1 is law.")
    print(
        "This whole-song result is now only a PRIOR. It does NOT force one "
        "unit length through the entire song."
    )


def print_structural_spans(spans,origin,bar_steps):
    print("\n=== STRUCTURAL SPANS ===")
    if origin > 0:
        print(f"S00  {step_to_bar_beat_subdiv(0,bar_steps)} -> "
              f"{step_to_bar_beat_subdiv(origin,bar_steps)}   "
              "FREE / PROMPT AREA (bars 1-2; may contain anticipation)")

    print("\n seg   structural range                  local range      notes  reduced  entrance relative to boundary                 trailing silence")
    print(" ---   --------------------------------  ---------------  -----  -------  --------------------------------------------  ----------------")
    for s in spans:
        local_start=s.start-origin
        local_end=s.end-origin

        if s.kind == "gap":
            print(
                f" {s.label:>3s}   {step_to_bar_beat_subdiv(s.start,bar_steps):>12s} -> "
                f"{step_to_bar_beat_subdiv(s.end,bar_steps):<12s}   "
                f"{local_start:5d}:{local_end:<5d}       -        -  "
                f"{'STRUCTURAL GAP / NO MELODY EVIDENCE':<44s}  "
                f"{float(s.end-s.start):6.1f} steps"
            )
            continue

        if s.entrance_start is None:
            entrance='-'
        elif s.entrance_offset_steps < -EPS:
            entrance=(
                f"{step_to_bar_beat_subdiv(s.entrance_start,bar_steps)} "
                f"({abs(s.entrance_offset_steps):.1f} steps early)"
            )
        elif s.entrance_offset_steps > EPS:
            entrance=(
                f"{step_to_bar_beat_subdiv(s.entrance_start,bar_steps)} "
                f"({s.entrance_offset_steps:.1f} steps late)"
            )
        else:
            entrance=f"{step_to_bar_beat_subdiv(s.entrance_start,bar_steps)} (on boundary)"

        print(f" {s.label:>3s}   {step_to_bar_beat_subdiv(s.start,bar_steps):>12s} -> "
              f"{step_to_bar_beat_subdiv(s.end,bar_steps):<12s}   "
              f"{local_start:5d}:{local_end:<5d}   {len(s.segment.notes):5d}  "
              f"{len(s.segment.reduced_notes):7d}  {entrance:<44s}  "
              f"{s.trailing_silence_steps:6.1f} steps")


def print_structural_similarity(spans,threshold):
    segments=[s.segment for s in spans if s.kind != "gap" and s.segment.notes]
    pairs=compare_all(segments)
    print("\n=== STRUCTURAL-SPAN SIMILARITY ===")
    selected=[p for p in pairs if p.overall >= threshold]
    if not selected:
        print(f"No non-empty structural spans reached {threshold:.3f}.")
        return
    print(" pair       overall  reduced  contour  rhythm  metric  duration  pc     abs-reg   class")
    for p in selected:
        print(f" S{p.left:02d}<->S{p.right:02d}   {p.overall:7.3f}  {p.reduced_melody:7.3f}  "
              f"{p.contour:7.3f}  {p.rhythm:6.3f}  {p.metrical_anchor:6.3f}  {p.duration:8.3f}  "
              f"{p.pitch_class:5.3f}  {p.absolute_register:7.3f}   {p.classification}")


def print_macro_matches(matches,spans,bar_steps,threshold,max_per_length=8):
    print("\n=== MACRO RECURRENCE ===")
    print(f"Threshold: {threshold:.3f}")
    print(
        "Macro candidates are built only from complete adjacent melodic spans and never cross a structural GAP."
    )
    if not matches:
        print("No macro matches reached the threshold.")
        return
    groups={}
    for m in matches:
        groups.setdefault(m.length_bars,[]).append(m)
    for length in sorted(groups):
        print(f"\n{length}-BAR OBJECTS")
        for m in groups[length][:max_per_length]:
            ll=_span_labels_for_range(spans,m.left_start,m.left_end)
            rr=_span_labels_for_range(spans,m.right_start,m.right_end)
            print(f"  {step_to_bar_beat_subdiv(m.left_start,bar_steps)} -> {step_to_bar_beat_subdiv(m.left_end,bar_steps)} [{ll}]"
                  f"   <->   {step_to_bar_beat_subdiv(m.right_start,bar_steps)} -> {step_to_bar_beat_subdiv(m.right_end,bar_steps)} [{rr}]"
                  f"   score={m.similarity.overall:.3f}")
            print(f"      reduced={m.similarity.reduced_melody:.3f}  contour={m.similarity.contour:.3f}  "
                  f"rhythm={m.similarity.rhythm:.3f}  metric={m.similarity.metrical_anchor:.3f}")


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

class UnionFind:
    def __init__(self, values: Iterable[int]):
        self.parent = {value: value for value in values}

    def find(self, value: int) -> int:
        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])
        return self.parent[value]

    def union(self, a: int, b: int) -> None:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if ra < rb:
            self.parent[rb] = ra
        else:
            self.parent[ra] = rb


def build_families(
    segments: Sequence[Segment],
    pairs: Sequence[PairSimilarity],
    threshold: float,
) -> List[List[int]]:
    uf = UnionFind(segment.index for segment in segments)

    for pair in pairs:
        if pair.overall >= threshold:
            uf.union(pair.left, pair.right)

    groups: Dict[int, List[int]] = {}
    for segment in segments:
        groups.setdefault(uf.find(segment.index), []).append(segment.index)

    return sorted((sorted(v) for v in groups.values()), key=lambda v: v[0])


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def print_metadata(
    path: str,
    midi: mido.MidiFile,
    notes: Sequence[Note],
    time_signatures: Sequence[Tuple[int, int, int]],
    tempos: Sequence[Tuple[int, float]],
    numerator: int,
    denominator: int,
    bar_steps: float,
    total_steps: int,
) -> None:
    print("\n=== MELODY STRUCTURE PREFLIGHT ===")
    print(f"Input:              {path}")
    print(f"MIDI type:          {midi.type}")
    print(f"Ticks per beat:     {midi.ticks_per_beat}")
    print(f"Melody notes:       {len(notes)}")
    print(f"Length:             {total_steps} 16th-note steps")
    print(f"Time signature:     {numerator}/{denominator}")
    print(f"Bar length:         {bar_steps:.2f} steps")
    structural_origin = fixed_structural_origin(bar_steps)
      
    print(f"Structural anchor:  {step_to_bar_beat_subdiv(structural_origin, bar_steps)} "
          f"(absolute step {structural_origin}; mel_to_chord local step 0)")

    if tempos:
        print(f"Initial tempo:      {tempos[0][1]:.3f} BPM")
    else:
        print("Initial tempo:      not present (not needed for structural step analysis)")

    if len(time_signatures) > 1:
        print("\nWARNING: multiple time signatures found.")
        print("v0 uses the FIRST time signature for metric/bar features.")
        for tick, n, d in time_signatures:
            step = tick_to_step(tick, midi.ticks_per_beat)
            print(f"  step {step:8.2f}: {n}/{d}")


def print_boundaries(
    candidates: Sequence[BoundaryCandidate],
    accepted: Sequence[BoundaryCandidate],
    threshold: float,
) -> None:
    print("\n=== BOUNDARY CANDIDATES ===")
    print(f"Acceptance threshold: {threshold:.3f}\n")
    print(" step   score   accepted   rest  long  bar   reg   cont  dens  rhythm")
    print(" ----   -----   --------   ----  ----  ----  ----  ----  ----  ------")

    accepted_steps = {c.step for c in accepted}
    floor = max(0.20, threshold - 0.15)

    report = [
        c for c in candidates
        if c.step in accepted_steps or c.score >= floor
    ]
    report.sort(key=lambda c: c.step)

    for c in report:
        print(
            f"{c.step:5d}   {c.score:5.3f}   {'YES' if c.accepted else '-':>8s}   "
            f"{c.rest:4.2f}  {c.long_note:4.2f}  {c.barline:4.2f}  "
            f"{c.register_reset:4.2f}  {c.contour_reset:4.2f}  "
            f"{c.density_change:4.2f}  {c.rhythmic_break:6.2f}"
        )

    print("\nAccepted boundaries:", " ".join(str(c.step) for c in accepted) if accepted else "(none)")


def print_segments(segments: Sequence[Segment]) -> None:
    print("\n=== STRUCTURAL SEGMENTS ===")
    print(" seg   range         len   notes  reduced  first-last   median-pitch")
    print(" ---   ------------  ----  -----  -------  ----------   ------------")

    for segment in segments:
        if segment.notes:
            first_pitch = segment.notes[0].pitch
            last_pitch = segment.notes[-1].pitch
            median_pitch = float(np.median([n.pitch for n in segment.notes]))
            first_last = f"{first_pitch:3d}->{last_pitch:3d}"
            median_text = f"{median_pitch:6.1f}"
        else:
            first_last = "empty"
            median_text = "-"

        print(
            f" {segment.label:>3s}   {segment.start:4d}:{segment.end:<4d}   "
            f"{segment.end - segment.start:4d}  {len(segment.notes):5d}  "
            f"{len(segment.reduced_notes):7d}  {first_last:>10s}   {median_text:>12s}"
        )


def print_pairs(pairs: Sequence[PairSimilarity], threshold: float, show_all: bool) -> None:
    print("\n=== SEGMENT SIMILARITY ===")
    selected = list(pairs) if show_all else [p for p in pairs if p.overall >= threshold]

    if not selected:
        print(f"No pairs reached report threshold {threshold:.3f}.")
        return

    print(" pair       overall  reduced  contour  rhythm  metric  duration  pc     abs-reg   class")
    print(" --------   -------  -------  -------  ------  ------  --------  -----  -------   -------------------")

    for p in selected:
        print(
            f" S{p.left:02d}<->S{p.right:02d}   {p.overall:7.3f}  {p.reduced_melody:7.3f}  "
            f"{p.contour:7.3f}  {p.rhythm:6.3f}  {p.metrical_anchor:6.3f}  "
            f"{p.duration:8.3f}  {p.pitch_class:5.3f}  {p.absolute_register:7.3f}   "
            f"{p.classification}"
        )


def print_families(families: Sequence[Sequence[int]], threshold: float) -> None:
    print("\n=== SIMILARITY FAMILIES ===")
    print(f"Family threshold: {threshold:.3f}\n")

    repeated = [f for f in families if len(f) >= 2]
    unique = [f[0] for f in families if len(f) == 1]

    if repeated:
        for i, family in enumerate(repeated, 1):
            print(f"F{i:02d}: " + " ".join(f"S{x:02d}" for x in family))
    else:
        print("No repeated families at the current threshold.")

    print("\nUnique:", " ".join(f"S{x:02d}" for x in unique) if unique else "(none)")
    print("\nNOTE: family labels are structural only. They do NOT mean verse/chorus/bridge.")


def print_reduced_notes(segments: Sequence[Segment]) -> None:
    print("\n=== REDUCED MELODIC BACKBONES ===")

    for segment in segments:
        print(f"\n{segment.label} {segment.start}:{segment.end}")
        if not segment.reduced_notes:
            print("  (empty)")
            continue

        for note in segment.reduced_notes:
            print(
                f"  local={note.start - segment.start:6.2f}  "
                f"global={note.start:7.2f}  pitch={note.pitch:3d}  dur={note.duration:5.2f}"
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_time_signature(value: Optional[str], time_signatures: Sequence[Tuple[int, int, int]]):
    if time_signatures:
        _, n, d = time_signatures[0]
        return n, d

    if value:
        try:
            n_text, d_text = value.split("/", 1)
            n, d = int(n_text), int(d_text)
        except Exception as exc:
            raise ValueError("--time-signature must look like 4/4") from exc
        if n <= 0 or d <= 0:
            raise ValueError("Time-signature values must be positive.")
        return n, d

    return DEFAULT_NUMERATOR, DEFAULT_DENOMINATOR


def analyze(args: argparse.Namespace) -> None:
    midi, notes, time_signatures, tempos = read_melody(args.input_midi)
    numerator, denominator = parse_time_signature(args.time_signature, time_signatures)
    bar_steps = numerator * 16.0 / denominator
    total_steps = int(math.ceil(max(note.end for note in notes)))

    print_metadata(args.input_midi, midi, notes, time_signatures, tempos,
                   numerator, denominator, bar_steps, total_steps)

    boundary_candidates = score_boundaries(notes, total_steps, bar_steps)

    structural_origin = fixed_structural_origin(bar_steps)

    trace_local_bars = []

    if args.trace_local_bars:
        try:
            trace_local_bars = [
                int(value.strip())
                for value in args.trace_local_bars.split(",")
                if value.strip()
            ]
        except ValueError as exc:
            raise ValueError(
                "--trace-local-bars must be a comma-separated list "
                "of integer source bar numbers, e.g. 11,14,17,19"
            ) from exc

        if any(
            bar_number < 1
            for bar_number in trace_local_bars
        ):
            raise ValueError(
                "--trace-local-bars values must be >= 1"
            )

    if structural_origin >= total_steps:
        raise ValueError(
            f"Melody ends at step {total_steps}, before the fixed structural anchor "
            f"at {step_to_bar_beat_subdiv(structural_origin, bar_steps)} "
            f"(absolute step {structural_origin})."
        )

    lattice_candidates = score_lattice_candidates(
        notes, boundary_candidates, total_steps, bar_steps,
        args.max_base_bars, args.lattice_tolerance_steps, structural_origin,
    )
    lattice = choose_lattice(lattice_candidates)
    print_lattice_candidates(lattice_candidates, lattice, bar_steps)

    adaptive_decisions, full_structural_end = infer_adaptive_span_path(
        notes,
        boundary_candidates,
        lattice_candidates,
        lattice,
        structural_origin,
        total_steps,
        bar_steps,
        args.max_base_bars,
        args.lattice_tolerance_steps,
        args.local_lattice_horizon_bars,
        args.local_switch_margin,
        args.local_one_bar_margin,
        args.gap_min_bars,
    )
    print_adaptive_lattice_path(
        adaptive_decisions,
        bar_steps,
    )
    if args.trace_endpoint_vs_lattice:
        print_endpoint_vs_lattice_trace(
            notes,
            boundary_candidates,
            adaptive_decisions,
            full_structural_end,
            bar_steps,
            args.max_base_bars,
            args.lattice_tolerance_steps,
            args.local_lattice_horizon_bars,
        )
    # ALPHA SWEEP test
    if args.trace_endpoint_alpha_sweep:
        print_endpoint_alpha_sweep(
            notes,
            boundary_candidates,
            adaptive_decisions,
            full_structural_end,
            bar_steps,
            args.max_base_bars,
            args.lattice_tolerance_steps,
            args.local_lattice_horizon_bars,
            (
                lattice.unit_bars
                if lattice is not None
                else None
            ),
            args.local_one_bar_margin,
        )
    # bounded two-decision path inspection
    if args.trace_two_decision_bars:
        two_decision_start_bars = [
            int(part.strip())
            for part in args.trace_two_decision_bars.split(",")
            if part.strip()
        ]

        print_two_decision_path_trace(
            notes,
            boundary_candidates,
            full_structural_end,
            bar_steps,
            args.max_base_bars,
            args.lattice_tolerance_steps,
            args.local_lattice_horizon_bars,
            (
                lattice.unit_bars
                if lattice is not None
                else None
            ),
            args.local_one_bar_margin,
            two_decision_start_bars,
        )

    print_local_candidate_trace(
        notes,
        boundary_candidates,
        trace_local_bars,
        structural_origin,
        full_structural_end,
        bar_steps,
        args.max_base_bars,
        args.lattice_tolerance_steps,
        args.local_lattice_horizon_bars,
        (
            lattice.unit_bars
            if lattice is not None
            else None
        ),
        args.local_one_bar_margin,
    )
    print_expected_boundary_trace(
        notes,
        boundary_candidates,
        11,
        full_structural_end,
        bar_steps,
        args.max_base_bars,
        args.lattice_tolerance_steps,
        args.local_lattice_horizon_bars,
    )

    spans = make_structural_spans_adaptive(
        notes,
        adaptive_decisions,
        full_structural_end,
        total_steps,
        bar_steps,
    )
    print_structural_spans(
        spans,
        structural_origin,
        bar_steps,
    )
    print_structural_similarity(spans, args.pair_report_threshold)

    macro_matches = find_macro_matches(
        spans,
        bar_steps,
        args.max_macro_bars,
        args.macro_threshold,
    )
    print_macro_matches(
        macro_matches, spans, bar_steps, args.macro_threshold,
        max_per_length=args.max_macro_matches_per_length,
    )

    if args.show_boundary_evidence:
        print_boundaries(boundary_candidates, [], args.boundary_threshold)

    if args.show_reduced_notes:
        print_reduced_notes([span.segment for span in spans])

    print("\n=== PREFLIGHT COMPLETE ===")
    print("Diagnostic only: no MIDI was changed and no generation was influenced.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Structural preflight for a monophonic melody MIDI. "
            "Uses fixed structural zero at bar 3 beat 1, rescoring local "
            "structural periodicity after every accepted boundary, tracking established period/phase separately from emitted span length, "
            "and searches multi-bar macro recurrence without semantic "
            "song-form labels."
        )
    )
    parser.add_argument("input_midi")
    parser.add_argument("--time-signature", default=None,
                        help="Fallback if MIDI has no time-signature event. Example: 4/4.")
    parser.add_argument("--boundary-threshold", type=float, default=DEFAULT_BOUNDARY_THRESHOLD)
    parser.add_argument("--max-base-bars", type=int, default=DEFAULT_MAX_BASE_BARS)
    parser.add_argument("--lattice-tolerance-steps", type=int, default=DEFAULT_LATTICE_TOLERANCE_STEPS)
    parser.add_argument(
        "--local-lattice-horizon-bars",
        type=int,
        default=DEFAULT_LOCAL_LATTICE_HORIZON_BARS,
        help="Look-ahead region used to compare local 1/2/3/4-bar periodicities.",
    )
    parser.add_argument(
        "--local-switch-margin",
        type=float,
        default=DEFAULT_LOCAL_SWITCH_MARGIN,
        help="Keep the previous local period when it remains within this score margin.",
    )
    parser.add_argument(
        "--local-one-bar-margin",
        type=float,
        default=DEFAULT_LOCAL_ONE_BAR_MARGIN,
        help="A 1-bar local period must beat the strongest non-1-bar alternative by this margin.",
    )
    parser.add_argument(
        "--gap-min-bars",
        type=float,
        default=DEFAULT_GAP_MIN_BARS,
        help="Minimum actual melody-free duration, in bars, to become an explicit structural GAP.",
    )
    parser.add_argument("--pair-report-threshold", type=float, default=DEFAULT_PAIR_REPORT_THRESHOLD)
    parser.add_argument("--max-macro-bars", type=int, default=DEFAULT_MAX_MACRO_BARS)
    parser.add_argument("--macro-threshold", type=float, default=DEFAULT_MACRO_THRESHOLD)
    parser.add_argument("--max-macro-matches-per-length", type=int, default=8)
    parser.add_argument(
        "--trace-local-bars",
        default=None,
        help=(
            "Comma-separated SOURCE bar numbers at which to print the complete "
            "local 1/2/3/4-bar candidate table. Diagnostic only. "
            "Example: --trace-local-bars 11,14,17,19"
        ),
    )
    parser.add_argument("--show-boundary-evidence", action="store_true")
    parser.add_argument("--show-reduced-notes", action="store_true")
    parser.add_argument(
        "--trace-endpoint-vs-lattice",
        action="store_true",
        help=(
            "Diagnostic: at every adaptive segment start, compare immediate "
            "endpoint boundary evidence against the current periodic lattice score."
        ),
    )
    # ALPHA SWEEP test
    parser.add_argument(
        "--trace-endpoint-alpha-sweep",
        action="store_true",
        help=(
            "Diagnostic sweep of endpoint-support weights added "
            "to existing local lattice scores."
        ),
    )    
    # bounded two-decision path inspection
    parser.add_argument(
        "--trace-two-decision-bars",
        default=None,
        help=(
            "Comma-separated SOURCE bar numbers for observational "
            "two-decision path inspection. "
            "Example: --trace-two-decision-bars 7,11"
        ),
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.local_lattice_horizon_bars < 2:
        parser.error("--local-lattice-horizon-bars must be at least 2")
    if args.local_switch_margin < 0.0:
        parser.error("--local-switch-margin must be >= 0")
    if args.local_one_bar_margin < 0.0:
        parser.error("--local-one-bar-margin must be >= 0")

    for name in ("boundary_threshold", "pair_report_threshold", "macro_threshold"):
        value = getattr(args, name)
        if not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be in [0, 1]")

    if args.max_base_bars < 1:
        parser.error("--max-base-bars must be >= 1")
    if args.max_macro_bars < 2:
        parser.error("--max-macro-bars must be >= 2")
    if args.lattice_tolerance_steps < 0:
        parser.error("--lattice-tolerance-steps must be >= 0")
    if args.max_macro_matches_per_length < 1:
        parser.error("--max-macro-matches-per-length must be >= 1")

    analyze(args)


if __name__ == "__main__":
    main()
