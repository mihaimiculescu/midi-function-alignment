#!/usr/bin/env python3

import json
import math
import statistics
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    "Dataset/LAMDselection/selection_stage1/stage3"
)

REPORTS_JSON = BASE / "reports.json"

ANALYSIS_DIR = BASE / "analysis"
ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

ANALYSIS_REPORT = (
    ANALYSIS_DIR / "analysis_report.txt"
)

CANDIDATE_DATA_JSON = (
    ANALYSIS_DIR / "candidate_data.json"
)

MIDI_SUMMARY_JSON = (
    ANALYSIS_DIR / "midi_summary.json"
)


# ============================================================
# ANALYSIS PARAMETERS
# ============================================================

SCORE_THRESHOLDS = [
    0.75,
    0.80,
    0.85,
    0.90,
]

MONO_THRESHOLDS = [
    0.80,
    0.85,
    0.90,
    0.95,
]

MIN_NOTE_COUNTS = [
    20,
    30,
    50,
    75,
    100,
]


# Pitch bands are deliberately fairly fine-grained.
# MIDI pitch is used directly.
PITCH_BANDS = [
    ("<30", None, 30),
    ("30-34", 30, 35),
    ("35-39", 35, 40),
    ("40-44", 40, 45),
    ("45-49", 45, 50),
    ("50-54", 50, 55),
    ("55-59", 55, 60),
    ("60-64", 60, 65),
    ("65-69", 65, 70),
    ("70-74", 70, 75),
    ("75-79", 75, 80),
    ("80+", 80, None),
]


# ============================================================
# HELPERS
# ============================================================

def percentile(values, p):
    if not values:
        return None

    values = sorted(values)

    if len(values) == 1:
        return values[0]

    k = (len(values) - 1) * (p / 100.0)

    lower = math.floor(k)
    upper = math.ceil(k)

    if lower == upper:
        return values[lower]

    return (
        values[lower]
        + (values[upper] - values[lower])
        * (k - lower)
    )


def stats(values):
    values = [
        float(v)
        for v in values
        if v is not None
        and math.isfinite(float(v))
    ]

    if not values:
        return {
            "count": 0,
            "min": None,
            "p05": None,
            "p10": None,
            "p25": None,
            "median": None,
            "p75": None,
            "p90": None,
            "p95": None,
            "max": None,
            "mean": None,
        }

    return {
        "count": len(values),
        "min": min(values),
        "p05": percentile(values, 5),
        "p10": percentile(values, 10),
        "p25": percentile(values, 25),
        "median": statistics.median(values),
        "p75": percentile(values, 75),
        "p90": percentile(values, 90),
        "p95": percentile(values, 95),
        "max": max(values),
        "mean": statistics.mean(values),
    }


def fmt(value, digits=4):
    if value is None:
        return "n/a"

    if isinstance(value, float):
        return f"{value:.{digits}f}"

    return str(value)


def get_pitch_band(pitch):
    for label, low, high in PITCH_BANDS:

        if low is None:
            if pitch < high:
                return label

        elif high is None:
            if pitch >= low:
                return label

        else:
            if low <= pitch < high:
                return label

    return "UNKNOWN"


def candidate_valid(candidate):
    required = (
        "melody_score",
        "monophonic_fraction",
        "pitch_mean",
        "pitch_range",
        "raw_notes",
        "note_density",
        "activity_span",
    )

    return all(
        key in candidate
        for key in required
    )


# ============================================================
# LOAD REPORTS
# ============================================================

print()
print("=" * 72)
print("LOADING STAGE 3 REPORTS")
print("=" * 72)

with open(REPORTS_JSON, "r") as f:
    data = json.load(f)

reports = data["reports"]

print(
    f"Reports loaded : {len(reports):,}"
)
print()


# ============================================================
# BUILD MIDI/CANDIDATE DATA
# ============================================================

all_candidates = []
midi_records = []

invalid_candidates = 0


for report in reports:

    md5 = report["md5"]
    path = report["path"]

    stage3 = report.get("stage3", {})

    raw_candidates = stage3.get(
        "candidates",
        []
    )

    candidates = []

    for candidate in raw_candidates:

        if not candidate_valid(candidate):
            invalid_candidates += 1
            continue

        c = {
            "md5": md5,
            "path": path,

            "track": candidate["track"],
            "channels": candidate["channels"],
            "program": candidate["program"],
            "is_drum": candidate["is_drum"],

            "raw_notes": candidate["raw_notes"],
            "non_percussion_notes":
                candidate["non_percussion_notes"],

            "polyphony": candidate["polyphony"],
            "monophonic_fraction":
                candidate["monophonic_fraction"],

            "pitch_range": candidate["pitch_range"],
            "pitch_mean": candidate["pitch_mean"],

            "note_density": candidate["note_density"],
            "activity_span": candidate["activity_span"],

            "duration_mean":
                candidate["duration"]["mean"],

            "duration_median":
                candidate["duration"]["median"],

            "duration_min":
                candidate["duration"]["min"],

            "duration_max":
                candidate["duration"]["max"],

            "melody_score":
                candidate["melody_score"],
        }

        candidates.append(c)
        all_candidates.append(c)

    # --------------------------------------------------------
    # RAW TOP SCORE
    # --------------------------------------------------------

    top_candidate = None

    if candidates:
        top_candidate = max(
            candidates,
            key=lambda c: c["melody_score"]
        )

    midi_records.append({
        "md5": md5,
        "path": path,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "top_candidate": top_candidate,
    })


print(
    f"Candidate tracks : {len(all_candidates):,}"
)

print(
    f"Invalid candidates: {invalid_candidates:,}"
)

print()


# ============================================================
# BASIC DISTRIBUTIONS
# ============================================================

def collect(field):
    return [
        c[field]
        for c in all_candidates
    ]


basic_distributions = {
    "melody_score":
        stats(collect("melody_score")),

    "monophonic_fraction":
        stats(collect("monophonic_fraction")),

    "pitch_mean":
        stats(collect("pitch_mean")),

    "pitch_range":
        stats(collect("pitch_range")),

    "raw_notes":
        stats(collect("raw_notes")),

    "note_density":
        stats(collect("note_density")),

    "activity_span":
        stats(collect("activity_span")),
}


# ============================================================
# RAW TOP-CANDIDATE DISTRIBUTIONS
# ============================================================

top_candidates = [
    m["top_candidate"]
    for m in midi_records
    if m["top_candidate"] is not None
]

top_distributions = {
    "melody_score":
        stats([
            c["melody_score"]
            for c in top_candidates
        ]),

    "monophonic_fraction":
        stats([
            c["monophonic_fraction"]
            for c in top_candidates
        ]),

    "pitch_mean":
        stats([
            c["pitch_mean"]
            for c in top_candidates
        ]),

    "pitch_range":
        stats([
            c["pitch_range"]
            for c in top_candidates
        ]),

    "raw_notes":
        stats([
            c["raw_notes"]
            for c in top_candidates
        ]),
}


# ============================================================
# QUALIFYING CANDIDATE FUNCTION
# ============================================================

def qualifying_candidates(
    candidates,
    score_threshold,
    mono_threshold,
    min_notes=0,
):
    return [
        c
        for c in candidates
        if (
            c["melody_score"]
            >= score_threshold
            and
            c["monophonic_fraction"]
            >= mono_threshold
            and
            c["raw_notes"]
            >= min_notes
        )
    ]


# ============================================================
# MIDI-LEVEL ANALYSIS
# ============================================================

#
# This is the important distinction:
#
# We don't ask:
#
#   "Is the highest-scoring candidate good?"
#
# We ask:
#
#   "Does this MIDI contain ANY candidate that satisfies
#    the criteria?"
#
# And then choose the highest-scoring qualifying candidate.
#

midi_level_results = {}


for score_threshold in SCORE_THRESHOLDS:

    midi_level_results[
        str(score_threshold)
    ] = {}

    for mono_threshold in MONO_THRESHOLDS:

        key = (
            f"score_{score_threshold:.2f}"
            f"_mono_{mono_threshold:.2f}"
        )

        records = []

        for midi in midi_records:

            eligible = qualifying_candidates(
                midi["candidates"],
                score_threshold,
                mono_threshold,
                0,
            )

            if not eligible:
                continue

            best = max(
                eligible,
                key=lambda c: c["melody_score"]
            )

            records.append({
                "md5": midi["md5"],
                "path": midi["path"],
                "candidate": best,
            })

        midi_level_results[
            str(score_threshold)
        ][str(mono_threshold)] = records


# ============================================================
# PITCH-BAND ANALYSIS
# ============================================================

#
# For every score/mono combination:
#
#   1. Find every MIDI with >= 1 qualifying candidate.
#   2. Select the highest-scoring qualifying candidate.
#   3. Put THAT candidate into a pitch band.
#
# This is MIDI-level, not candidate-level.
#

pitch_band_analysis = {}


for score_threshold in SCORE_THRESHOLDS:

    pitch_band_analysis[
        str(score_threshold)
    ] = {}

    for mono_threshold in MONO_THRESHOLDS:

        records = midi_level_results[
            str(score_threshold)
        ][str(mono_threshold)]

        counts = {
            label: 0
            for label, _, _ in PITCH_BANDS
        }

        for record in records:

            pitch = record[
                "candidate"
            ]["pitch_mean"]

            band = get_pitch_band(pitch)

            if band in counts:
                counts[band] += 1

        pitch_band_analysis[
            str(score_threshold)
        ][str(mono_threshold)] = {
            "total_midis": len(records),
            "bands": counts,
        }


# ============================================================
# NOTE COUNT ANALYSIS
# ============================================================

#
# Same procedure, but adding minimum note counts.
#

note_count_analysis = {}


for score_threshold in SCORE_THRESHOLDS:

    note_count_analysis[
        str(score_threshold)
    ] = {}

    for mono_threshold in MONO_THRESHOLDS:

        note_count_analysis[
            str(score_threshold)
        ][str(mono_threshold)] = {}

        for min_notes in MIN_NOTE_COUNTS:

            records = []

            for midi in midi_records:

                eligible = qualifying_candidates(
                    midi["candidates"],
                    score_threshold,
                    mono_threshold,
                    min_notes,
                )

                if not eligible:
                    continue

                best = max(
                    eligible,
                    key=lambda c:
                        c["melody_score"]
                )

                records.append({
                    "md5": midi["md5"],
                    "path": midi["path"],
                    "candidate": best,
                })

            note_count_analysis[
                str(score_threshold)
            ][str(mono_threshold)][
                str(min_notes)
            ] = {
                "midi_count": len(records),
                "records": records,
            }


# ============================================================
# LOW-REGISTER TOP CANDIDATE VS ALTERNATIVE
# ============================================================

#
# This is specifically designed to detect the situation
# represented by 7a00...:
#
#   raw top candidate = bass
#   another candidate = actual melody
#
# We define "low register" as pitch_mean < 50.
#
# This is NOT a proposed final threshold.
# It is purely diagnostic.
#

LOW_REGISTER_THRESHOLD = 50.0

alternative_analysis = {}


for score_threshold in SCORE_THRESHOLDS:

    alternative_analysis[
        str(score_threshold)
    ] = {}

    for mono_threshold in MONO_THRESHOLDS:

        total = 0
        low_top = 0
        low_top_with_higher_alternative = 0
        low_top_without_higher_alternative = 0

        examples = []

        for midi in midi_records:

            top = midi["top_candidate"]

            if top is None:
                continue

            eligible = qualifying_candidates(
                midi["candidates"],
                score_threshold,
                mono_threshold,
                0,
            )

            if not eligible:
                continue

            total += 1

            if (
                top["pitch_mean"]
                < LOW_REGISTER_THRESHOLD
            ):

                low_top += 1

                higher = [
                    c
                    for c in eligible
                    if (
                        c["pitch_mean"]
                        >= LOW_REGISTER_THRESHOLD
                    )
                ]

                if higher:

                    low_top_with_higher_alternative += 1

                    best_higher = max(
                        higher,
                        key=lambda c:
                            c["melody_score"]
                    )

                    if len(examples) < 20:
                        examples.append({
                            "md5": midi["md5"],
                            "top": top,
                            "higher_alternative":
                                best_higher,
                        })

                else:

                    low_top_without_higher_alternative += 1

        alternative_analysis[
            str(score_threshold)
        ][str(mono_threshold)] = {
            "total_qualifying_midis": total,
            "low_register_top": low_top,
            "low_register_top_percentage": (
                low_top / total * 100
                if total else 0
            ),
            "low_top_with_higher_alternative":
                low_top_with_higher_alternative,
            "low_top_without_higher_alternative":
                low_top_without_higher_alternative,
            "examples": examples,
        }


# ============================================================
# KNOWN-GOOD FILE
# ============================================================

KNOWN_GOOD = (
    "7a00abfe35bc33d491af476f62406c4d"
)

known_good = next(
    (
        m
        for m in midi_records
        if m["md5"] == KNOWN_GOOD
    ),
    None,
)


# ============================================================
# WRITE CANDIDATE DATA
# ============================================================

with open(
    CANDIDATE_DATA_JSON,
    "w"
) as f:

    json.dump(
        {
            "source": str(REPORTS_JSON),
            "candidate_count":
                len(all_candidates),
            "candidates":
                all_candidates,
        },
        f,
        indent=2,
    )


# ============================================================
# WRITE MIDI SUMMARY
# ============================================================

with open(
    MIDI_SUMMARY_JSON,
    "w"
) as f:

    json.dump(
        {
            "source": str(REPORTS_JSON),
            "midi_count":
                len(midi_records),
            "summaries":
                midi_records,
        },
        f,
        indent=2,
    )


# ============================================================
# HUMAN-READABLE REPORT
# ============================================================

lines = []

lines.append("=" * 72)
lines.append("STAGE 3 CANDIDATE / REGISTER ANALYSIS")
lines.append("=" * 72)
lines.append("")

lines.append(
    f"Input reports      : {len(reports):,}"
)

lines.append(
    f"Candidate tracks   : {len(all_candidates):,}"
)

lines.append(
    f"Top candidates     : {len(top_candidates):,}"
)

lines.append(
    f"Invalid candidates : {invalid_candidates:,}"
)

lines.append("")


# ============================================================
# BASIC DISTRIBUTIONS
# ============================================================

lines.append("=" * 72)
lines.append("ALL CANDIDATE DISTRIBUTIONS")
lines.append("=" * 72)
lines.append("")

for name, s in basic_distributions.items():

    lines.append(name)
    lines.append("-" * len(name))

    for key in (
        "count",
        "min",
        "p05",
        "p10",
        "p25",
        "median",
        "p75",
        "p90",
        "p95",
        "max",
        "mean",
    ):

        lines.append(
            f"{key:>7} : {fmt(s[key])}"
        )

    lines.append("")


# ============================================================
# TOP CANDIDATE DISTRIBUTIONS
# ============================================================

lines.append("=" * 72)
lines.append("RAW TOP-CANDIDATE DISTRIBUTIONS")
lines.append("=" * 72)
lines.append("")

for name, s in top_distributions.items():

    lines.append(name)
    lines.append("-" * len(name))

    for key in (
        "count",
        "min",
        "p05",
        "p10",
        "p25",
        "median",
        "p75",
        "p90",
        "p95",
        "max",
        "mean",
    ):

        lines.append(
            f"{key:>7} : {fmt(s[key])}"
        )

    lines.append("")


# ============================================================
# PITCH BAND TABLES
# ============================================================

lines.append("=" * 72)
lines.append(
    "MIDI-LEVEL PITCH BANDS OF BEST QUALIFYING CANDIDATE"
)
lines.append("=" * 72)
lines.append("")

lines.append(
    "For each MIDI, the highest-scoring candidate satisfying"
)
lines.append(
    "the score + monophony thresholds is selected."
)
lines.append("")

for score_threshold in SCORE_THRESHOLDS:

    for mono_threshold in MONO_THRESHOLDS:

        result = pitch_band_analysis[
            str(score_threshold)
        ][str(mono_threshold)]

        lines.append(
            f"Score >= {score_threshold:.2f}, "
            f"Mono >= {mono_threshold:.2f} "
            f"({result['total_midis']:,} MIDIs)"
        )

        lines.append("")

        lines.append(
            f"{'Pitch mean':>12} {'MIDIs':>12} {'%':>10}"
        )

        lines.append("-" * 38)

        total = result["total_midis"]

        for label, _, _ in PITCH_BANDS:

            count = result["bands"][label]

            pct = (
                count / total * 100
                if total
                else 0
            )

            lines.append(
                f"{label:>12} "
                f"{count:>12,} "
                f"{pct:>9.2f}%"
            )

        lines.append("")
        lines.append("")


# ============================================================
# NOTE COUNT TABLES
# ============================================================

lines.append("=" * 72)
lines.append(
    "MIDI COUNT VS MINIMUM NOTE COUNT"
)
lines.append("=" * 72)
lines.append("")

for score_threshold in SCORE_THRESHOLDS:

    lines.append(
        f"Score >= {score_threshold:.2f}"
    )

    lines.append("")

    for mono_threshold in MONO_THRESHOLDS:

        lines.append(
            f"Mono >= {mono_threshold:.2f}"
        )

        header = (
            f"{'Min notes':>12}"
            f"{'MIDIs':>12}"
        )

        lines.append(header)
        lines.append("-" * 26)

        for min_notes in MIN_NOTE_COUNTS:

            count = note_count_analysis[
                str(score_threshold)
            ][str(mono_threshold)][
                str(min_notes)
            ]["midi_count"]

            lines.append(
                f"{min_notes:>12,}"
                f"{count:>12,}"
            )

        lines.append("")

    lines.append("")


# ============================================================
# LOW TOP / HIGHER ALTERNATIVE
# ============================================================

lines.append("=" * 72)
lines.append(
    "LOW-REGISTER TOP CANDIDATE VS HIGHER ALTERNATIVE"
)
lines.append("=" * 72)
lines.append("")

lines.append(
    f"Diagnostic low-register boundary: "
    f"pitch_mean < {LOW_REGISTER_THRESHOLD:.1f}"
)
lines.append("")

for score_threshold in SCORE_THRESHOLDS:

    for mono_threshold in MONO_THRESHOLDS:

        x = alternative_analysis[
            str(score_threshold)
        ][str(mono_threshold)]

        lines.append(
            f"Score >= {score_threshold:.2f}, "
            f"Mono >= {mono_threshold:.2f}"
        )

        lines.append(
            f"  qualifying MIDIs             : "
            f"{x['total_qualifying_midis']:,}"
        )

        lines.append(
            f"  low-register top             : "
            f"{x['low_register_top']:,} "
            f"({x['low_register_top_percentage']:.2f}%)"
        )

        lines.append(
            f"  low top + higher alternative : "
            f"{x['low_top_with_higher_alternative']:,}"
        )

        lines.append(
            f"  low top only                 : "
            f"{x['low_top_without_higher_alternative']:,}"
        )

        lines.append("")

        if x["examples"]:

            lines.append(
                "  Examples:"
            )

            for example in x["examples"]:

                top = example["top"]
                alt = example[
                    "higher_alternative"
                ]

                lines.append(
                    f"    {example['md5']}"
                )

                lines.append(
                    f"      TOP: "
                    f"score={top['melody_score']:.4f} "
                    f"mono={top['monophonic_fraction']:.4f} "
                    f"pitch={top['pitch_mean']:.2f} "
                    f"notes={top['raw_notes']}"
                )

                lines.append(
                    f"      ALT: "
                    f"score={alt['melody_score']:.4f} "
                    f"mono={alt['monophonic_fraction']:.4f} "
                    f"pitch={alt['pitch_mean']:.2f} "
                    f"notes={alt['raw_notes']}"
                )

            lines.append("")

        lines.append("")


# ============================================================
# KNOWN-GOOD
# ============================================================

lines.append("=" * 72)
lines.append("KNOWN-GOOD FILE")
lines.append("=" * 72)
lines.append("")

if known_good is None:

    lines.append(
        f"{KNOWN_GOOD}: NOT FOUND"
    )

else:

    lines.append(
        f"MD5: {known_good['md5']}"
    )

    lines.append(
        f"Path: {known_good['path']}"
    )

    lines.append("")

    lines.append(
        f"Candidate count: "
        f"{known_good['candidate_count']}"
    )

    lines.append("")

    lines.append(
        "Candidates ranked by melody_score:"
    )

    lines.append("")

    for rank, c in enumerate(
        known_good["candidates"],
        start=1,
    ):

        lines.append(
            f"{rank:>2}. "
            f"track={c['track']:>2} "
            f"program={c['program']:>3} "
            f"channels={c['channels']} "
            f"score={c['melody_score']:.4f} "
            f"mono={c['monophonic_fraction']:.4f} "
            f"pitch={c['pitch_mean']:.2f} "
            f"range={c['pitch_range']:>3} "
            f"notes={c['raw_notes']:>4}"
        )

    lines.append("")

    lines.append(
        "Known-good file under threshold combinations:"
    )

    lines.append("")

    for score_threshold in SCORE_THRESHOLDS:

        for mono_threshold in MONO_THRESHOLDS:

            eligible = qualifying_candidates(
                known_good["candidates"],
                score_threshold,
                mono_threshold,
                0,
            )

            if eligible:

                best = max(
                    eligible,
                    key=lambda c:
                        c["melody_score"]
                )

                lines.append(
                    f"Score >= {score_threshold:.2f}, "
                    f"Mono >= {mono_threshold:.2f}: "
                    f"track={best['track']} "
                    f"score={best['melody_score']:.4f} "
                    f"mono={best['monophonic_fraction']:.4f} "
                    f"pitch={best['pitch_mean']:.2f} "
                    f"notes={best['raw_notes']}"
                )

            else:

                lines.append(
                    f"Score >= {score_threshold:.2f}, "
                    f"Mono >= {mono_threshold:.2f}: "
                    f"NONE"
                )

    lines.append("")


# ============================================================
# WRITE REPORT
# ============================================================

with open(
    ANALYSIS_REPORT,
    "w"
) as f:

    f.write(
        "\n".join(lines)
    )


# ============================================================
# FINAL CONSOLE OUTPUT
# ============================================================

print("=" * 72)
print("ANALYSIS COMPLETE")
print("=" * 72)

print(
    f"Reports                 : "
    f"{len(reports):,}"
)

print(
    f"Candidate tracks        : "
    f"{len(all_candidates):,}"
)

print(
    f"MIDIs with candidates   : "
    f"{len(top_candidates):,}"
)

print()

print("Output:")
print(
    f"  {ANALYSIS_REPORT}"
)
print(
    f"  {CANDIDATE_DATA_JSON}"
)
print(
    f"  {MIDI_SUMMARY_JSON}"
)

print()

if known_good:

    top = known_good["top_candidate"]

    print("Known-good file:")

    print(
        f"  {KNOWN_GOOD}"
    )

    print(
        f"  top score={top['melody_score']:.4f}, "
        f"mono={top['monophonic_fraction']:.4f}, "
        f"pitch_mean={top['pitch_mean']:.2f}, "
        f"range={top['pitch_range']}, "
        f"notes={top['raw_notes']}"
    )

print()