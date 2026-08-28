import argparse
import copy
import os
import re

import mido
import numpy as np
import pretty_midi
import torch

from cp_transformer_yinyang import RoformerYinyang, PreprocessingParameters
from cp_transformer_inference import decode_output
from preprocess_large_midi_dataset import preprocess_midi


DEFAULT_BPM = 120.0
DEFAULT_NUMERATOR = 4
DEFAULT_DENOMINATOR = 4
DEFAULT_QUANTIZATION = 16 #for 16ths

# DEFAULT_GENERATION_LENGTH = 384
DEFAULT_TEMPERATURE = 1.0
DEFAULT_SAMPLES = 1
DEFAULT_SEED = 0

# Nottingham-style melody -> chord configuration.
MELODY_PROGRAM = 64
CHORD_PROGRAM = 0

# Chord notes are placed in octave 3.
CHORD_OCTAVE = 3


# ---------------------------------------------------------------------------
# MIDI metadata
# ---------------------------------------------------------------------------

def read_tempo_map(midi_path):
    """
    Read the actual MIDI tempo map using mido.

    Returns:
        list of (absolute_tick, bpm)
    """
    midi = mido.MidiFile(midi_path)

    tempo_map = []

    for track in midi.tracks:
        absolute_tick = 0

        for msg in track:
            absolute_tick += msg.time

            if msg.type == "set_tempo":
                bpm = mido.tempo2bpm(msg.tempo)
                tempo_map.append((absolute_tick, float(bpm)))

    tempo_map.sort(key=lambda x: x[0])

    # Avoid duplicate tempo events at the same tick.
    result = []
    for tick, bpm in tempo_map:
        if result and result[-1][0] == tick:
            result[-1] = (tick, bpm)
        else:
            result.append((tick, bpm))

    return result


def read_time_signature_map(midi_path):
    """
    Read the actual MIDI time-signature map using mido.

    Returns:
        list of (absolute_tick, numerator, denominator)
    """
    midi = mido.MidiFile(midi_path)

    time_signature_map = []

    for track in midi.tracks:
        absolute_tick = 0

        for msg in track:
            absolute_tick += msg.time

            if msg.type == "time_signature":
                time_signature_map.append(
                    (
                        absolute_tick,
                        int(msg.numerator),
                        int(msg.denominator),
                    )
                )

    time_signature_map.sort(key=lambda x: x[0])

    result = []
    for tick, numerator, denominator in time_signature_map:
        if result and result[-1][0] == tick:
            result[-1] = (tick, numerator, denominator)
        else:
            result.append((tick, numerator, denominator))

    return result


def parse_time_signature(value):
    match = re.fullmatch(r"\s*(\d+)\s*/\s*(\d+)\s*", value)

    if not match:
        raise ValueError(
            f"Invalid time signature '{value}'. Expected e.g. 4/4 or 3/4."
        )

    numerator = int(match.group(1))
    denominator = int(match.group(2))

    if numerator <= 0 or denominator <= 0:
        raise ValueError("Time-signature values must be positive.")

    return numerator, denominator


def get_midi_metadata(midi_path, cli_bpm=None, cli_time_signature=None):
    """
    Determine BPM and time signature.

    Priority:
        1. MIDI tempo map / time-signature map
        2. user supplied values
        3. defaults

    The resolved BPM is normalized to 2 decimal places immediately.
    The complete MIDI tempo/time-signature maps are preserved so they
    can be copied to the output MIDI.
    """

    tempo_map = read_tempo_map(midi_path)
    time_signature_map = read_time_signature_map(midi_path)

    # Tempo
    if tempo_map:
        bpm = round(float(tempo_map[0][1]), 2)
        bpm_source = "MIDI tempo map"
    elif cli_bpm is not None:
        bpm = round(float(cli_bpm), 2)
        bpm_source = "command line"
    else:
        bpm = round(float(DEFAULT_BPM), 2)
        bpm_source = "default"

    # Normalize EVERY tempo-map entry immediately.
    # This prevents values such as 108.000108000108 from
    # propagating into the output MIDI tempo map.
    normalized_tempo_map = [
        (tick, round(float(event_bpm), 2))
        for tick, event_bpm in tempo_map
    ]

    # Time signature
    if time_signature_map:
        _, numerator, denominator = time_signature_map[0]
        time_signature_source = "MIDI time-signature map"
    elif cli_time_signature is not None:
        numerator, denominator = parse_time_signature(cli_time_signature)
        time_signature_source = "command line"
    else:
        numerator = DEFAULT_NUMERATOR
        denominator = DEFAULT_DENOMINATOR
        time_signature_source = "default"

    return {
        "bpm": bpm,
        "numerator": numerator,
        "denominator": denominator,
        "bpm_source": bpm_source,
        "time_signature_source": time_signature_source,

        # Preserve the COMPLETE maps.
        "tempo_map": normalized_tempo_map,
        "time_signature_map": time_signature_map,
    }

# ---------------------------------------------------------------------------
# INPUT MIDI FILE LENGTH
# ---------------------------------------------------------------------------

def read_input_length(midi_path):
    midi = mido.MidiFile(midi_path)

    max_tick = 0

    for track in midi.tracks:
        absolute_tick = 0

        for msg in track:
            absolute_tick += msg.time

        max_tick = max(max_tick, absolute_tick)

    return int(round((max_tick / midi.ticks_per_beat) * DEFAULT_QUANTIZATION / 4))




# ---------------------------------------------------------------------------
# Key / chord handling
# ---------------------------------------------------------------------------

PITCH_CLASSES = {
    "C": 0,
    "B#": 0,
    "C#": 1,
    "Db": 1,
    "D": 2,
    "D#": 3,
    "Eb": 3,
    "E": 4,
    "Fb": 4,
    "E#": 5,
    "F": 5,
    "F#": 6,
    "Gb": 6,
    "G": 7,
    "G#": 8,
    "Ab": 8,
    "A": 9,
    "A#": 10,
    "Bb": 10,
    "B": 11,
    "Cb": 11,
}


def parse_key(key):
    """
    Parse keys such as:

        C minor
        Cm
        Eb major
        Eb
        F# minor
        Bb major
    """

    value = key.strip()

    match = re.fullmatch(
        r"([A-Ga-g](?:#|b)?)(?:\s*)(major|minor|maj|min|M|m)?",
        value,
    )

    if not match:
        raise ValueError(
            f"Invalid key '{key}'. Examples: 'C minor', 'Cm', 'Eb major'."
        )

    root = match.group(1)
    mode = match.group(2)

    # Normalize accidental spelling.
    root = root[0].upper() + root[1:]

    if root not in PITCH_CLASSES:
        raise ValueError(f"Unsupported key root: {root}")

    if mode is None:
        mode = "major"

    mode = mode.lower()

    if mode in ("m", "min", "minor"):
        mode = "minor"
    else:
        mode = "major"

    return root, mode


def make_tonic_triad(key):
    """
    Return MIDI pitches for the tonic triad.

    Examples:

        Cm      -> C3 Eb3 G3
        Eb major -> Eb3 G3 Bb3
    """

    root_name, mode = parse_key(key)

    root_pc = PITCH_CLASSES[root_name]
    third = 3 if mode == "minor" else 4
    fifth = 7

    root_pitch = 12 * (CHORD_OCTAVE + 1) + root_pc

    return [
        root_pitch,
        root_pitch + third,
        root_pitch + fifth,
    ]


# ---------------------------------------------------------------------------
# Synthetic melody -> chord input
# ---------------------------------------------------------------------------

def calculate_bar_ticks(resolution, numerator, denominator):
    """
    Length of one bar in MIDI ticks.

    MIDI resolution is ticks per quarter note.
    """

    quarter_notes_per_bar = numerator * (4.0 / denominator)

    return int(round(resolution * quarter_notes_per_bar))


def create_melody_chord_input(
    input_midi_path,
    output_midi_path,
    key,
):
    """
    Create the exact two-track structure required by the
    melody -> chord model.

    Track 0:
        original melody

    Track 1:
        tonic triad, whole-bar notes, for the first two bars
    """

    source = pretty_midi.PrettyMIDI(input_midi_path)

    # Only keep non-empty instruments.
    non_empty = [
        ins for ins in source.instruments
        if len(ins.notes) > 0
    ]

    if len(non_empty) != 1:
        raise ValueError(
            "Melody → chord input must contain exactly one non-empty "
            f"melody track. Found {len(non_empty)}."
        )

    melody_source = non_empty[0]

    # Metadata has already been determined from the original file.
    metadata = get_midi_metadata(
        input_midi_path,
        cli_bpm=None,
        cli_time_signature=None,
    )

    bpm = metadata["bpm"]
    numerator = metadata["numerator"]
    denominator = metadata["denominator"]

    # Construct a clean MIDI with exactly two instruments.
    result = pretty_midi.PrettyMIDI(
        resolution=source.resolution,
        initial_tempo=bpm,
    )

    # Preserve the original time-signature map.
    result.time_signature_changes = copy.deepcopy(
        source.time_signature_changes
    )

    # Preserve key signatures if present.
    result.key_signature_changes = copy.deepcopy(
        source.key_signature_changes
    )

    # Track 0 = melody.
    melody = pretty_midi.Instrument(
        program=melody_source.program,
        is_drum=melody_source.is_drum,
        name="Melody",
    )

    melody.notes = [
        pretty_midi.Note(
            velocity=note.velocity,
            pitch=note.pitch,
            start=note.start,
            end=note.end,
        )
        for note in melody_source.notes
    ]

    result.instruments.append(melody)

    # Track 1 = tonic chord prompt.
    chord = pretty_midi.Instrument(
        program=CHORD_PROGRAM,
        is_drum=False,
        name=f"{key} prompt",
    )

    chord_pitches = make_tonic_triad(key)

    bar_ticks = calculate_bar_ticks(
        source.resolution,
        numerator,
        denominator,
    )

    # Two bars.
    for bar in range(2):
        start_tick = bar * bar_ticks
        end_tick = (bar + 1) * bar_ticks

        start_time = source.tick_to_time(start_tick)
        end_time = source.tick_to_time(end_tick)

        for pitch in chord_pitches:
            chord.notes.append(
                pretty_midi.Note(
                    velocity=100,
                    pitch=pitch,
                    start=start_time,
                    end=end_time,
                )
            )

    result.instruments.append(chord)

    os.makedirs(os.path.dirname(output_midi_path), exist_ok=True)
    result.write(output_midi_path)

    return metadata, chord_pitches


# ---------------------------------------------------------------------------
# Model inference
# ---------------------------------------------------------------------------

def decompress(model, byte_arr):
    x = torch.tensor(byte_arr).unsqueeze(0)
    x = x.cuda()

    return model.preprocess(
        x,
        pitch_shift=torch.zeros(
            1,
            dtype=torch.int8,
            device="cuda",
        ),
        preprocess_args=PreprocessingParameters(""),
    )[:2]


def get_unquantized_melody(
    midi_path,
    generation_length,
    fixed_tempo,
):
    """
    Preserve the melody as an unquantized track for the final output.
    """

    results = preprocess_midi(
        midi_path,
        16,
        ins_ids=["track-0"],
        filter=False,
        fixed_length=generation_length,
        return_midi_ins=True,
    )

    if results is None:
        raise RuntimeError(
            "preprocess_midi() returned None while extracting the melody."
        )

    concat_result = []

    scale_ratio = 60.0 / fixed_tempo / 4.0

    for result in results:
        for ins in result:
            ins.notes = [
                pretty_midi.Note(
                    start=note.start * scale_ratio,
                    end=note.end * scale_ratio,
                    pitch=note.pitch,
                    velocity=100,
                )
                for note in ins.notes
            ]

            ins.program = MELODY_PROGRAM
            concat_result.append(ins)

    return concat_result

# ---------------------------------------------------------------------------
# Final cleanup - remove the synthetic prompt initially injected at 0-32
# used just before writing the output files
# ---------------------------------------------------------------------------
def remove_initial_chord_notes(midi_path, cutoff_steps):
    """
    Remove chord note events from the first `cutoff_steps` 16th-note
    positions while preserving the MIDI timeline.
    """

    midi = mido.MidiFile(midi_path)

    cutoff_tick = int(
        cutoff_steps * midi.ticks_per_beat / 4
    )

    for track in midi.tracks:

        absolute_tick = 0
        retained = []

        for msg in track:
            absolute_tick += msg.time

            if msg.type in ("note_on", "note_off"):
                if absolute_tick < cutoff_tick:
                    continue

            retained.append((absolute_tick, msg.copy()))

        track.clear()

        previous_tick = 0

        for absolute_tick, msg in retained:
            msg.time = absolute_tick - previous_tick
            track.append(msg)
            previous_tick = absolute_tick

    midi.save(midi_path)

def generate(
    model,
    input_midi,
    output_dir,
    bpm,
    prompt_length,
    generation_length,
    temperature,
    samples,
    seed,
):
    os.makedirs(output_dir, exist_ok=True)

    print()
    print("=== Melody → Chord ===")
    print(f"Input:          {input_midi}")
    print(f"BPM:            {bpm}")
    print(f"Prompt length:  {prompt_length}")
    print(f"Generation:     {generation_length}")
    print(f"Temperature:    {temperature}")
    print(f"Samples:        {samples}")
    print()

    ins_ids = ["track-0", "track-1"]

    fixed_program = [MELODY_PROGRAM, CHORD_PROGRAM]
    fixed_velocity = [100, 100]

    print("Preprocessing MIDI...")

    result = preprocess_midi(
        input_midi,
        16,
        ins_ids=ins_ids,
        filter=False,
        fixed_length=generation_length,
    )

    if result is None:
        raise RuntimeError(
            "preprocess_midi() returned None.\n"
            "The synthetic MIDI must contain notes in both track-0 "
            "and track-1."
        )

    byte_arr = result[0]

    x1, x2 = decompress(model, byte_arr)

    print(f"x1 shape: {x1.shape}")
    print(f"x2 shape: {x2.shape}")

    # x1 = conditioning melody
    # x2 = chord-generation stream
    x1 = x1[:, :generation_length]
    x2 = x2[:, :prompt_length]

    print("Conditioning stream:", x1.shape)
    print("Generation prompt:", x2.shape)

    extra_melody = get_unquantized_melody(
        input_midi,
        generation_length=generation_length,
        fixed_tempo=bpm,
    )

# #OLD START
#     with torch.no_grad():

#         x1 = x1.repeat(samples, 1, 1)
#         x2 = x2.repeat(samples, 1, 1)

#         torch.manual_seed(seed)
#         torch.cuda.manual_seed_all(seed)
#         np.random.seed(seed)

#         print("Generating...")

#         output = model.global_sampling(
#             x1,
#             x2,
#             temperature=temperature,
#         )

#     for i in range(samples):

#         output_i = [
#             output[j][i:i + 1, :]
#             for j in range(len(output))
#         ]

#         output_file = os.path.join(
#             output_dir,
#             f"proposal_{i + 1:02d}.mid",
#         )

#         print(f"Writing: {output_file}")

#         decode_output(
#             (output_i,),
#             output_file,
#             ratio=(
#                 model.compress_ratio_l,
#                 model.compress_ratio_r,
#             ),
#             tempo=bpm,
#             velocity=100,
#             fixed_program=CHORD_PROGRAM,
#             extra_instruments=extra_melody,
#         )
# #OLD END

#NEW START
    # ------------------------------------------------------------------
    # Chord generation
    #
    # The model was trained with a 384-step sequence length.
    #
    # For longer inputs, generate overlapping chunks:
    #
    #   Chunk 1:  0       -> 384
    #   Chunk 2:  352     -> 736
    #   Chunk 3:  704     -> 1088
    #   ...
    #
    # Each new chunk uses the previous 32 generated chord steps as
    # its prompt.  The complete new chunk is generated, but its first
    # 32 steps are the prompt overlap and are therefore discarded when
    # stitching the final result.
    # ------------------------------------------------------------------

    MODEL_MAX_LENGTH = 384
    overlap = prompt_length

    if overlap >= MODEL_MAX_LENGTH:
        raise ValueError(
            f"Prompt length ({overlap}) must be smaller than "
            f"model maximum length ({MODEL_MAX_LENGTH})."
        )

    # One output list per requested sample.
    final_outputs = [[] for _ in range(samples)]

    chunk_start = 0
    chunk_number = 1

    print("Generating...")

    while chunk_start < generation_length:

        # The first chunk starts at zero.
        #
        # Every subsequent chunk starts `overlap` steps before the
        # previous chunk ended.
        if chunk_number == 1:
            chunk_start = 0
        else:
            chunk_start = previous_chunk_end - overlap

        chunk_end = min(
            chunk_start + MODEL_MAX_LENGTH,
            generation_length,
        )

        chunk_length = chunk_end - chunk_start

        print()
        print(
            f"CHUNK {chunk_number}: "
            f"{chunk_start}:{chunk_end} "
            f"({chunk_length} steps)"
        )

        # Melody conditioning for this chunk.
        melody_chunk = x1[:, chunk_start:chunk_end]

        # --------------------------------------------------------------
        # First chunk:
        # use the artificial two-bar prompt exactly as before.
        # --------------------------------------------------------------
        if chunk_number == 1:

            chord_prompt = x2

            print(
                f"  Melody: {chunk_start}:{chunk_end}"
            )
            print(
                f"  Chord prompt: artificial "
                f"{prompt_length} steps"
            )

            # Generate all samples together.
            with torch.no_grad():

                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                np.random.seed(seed)

                melody_batch = melody_chunk.repeat(
                    samples, 1, 1
                )

                prompt_batch = chord_prompt.repeat(
                    samples, 1, 1
                )

                output = model.global_sampling(
                    melody_batch,
                    prompt_batch,
                    temperature=temperature,
                )

            for i in range(samples):

                output_i = [
                    output[j][i:i + 1, :]
                    for j in range(len(output))
                ]

                final_outputs[i].extend(output_i)

        # --------------------------------------------------------------
        # Subsequent chunks:
        #
        # Use the last `overlap` generated chord steps from the
        # previous chunk as the new prompt.
        #
        # Example:
        #
        #   previous chunk = 0:384
        #   prompt         = 352:384
        #   new chunk      = 352:483
        #
        # The returned 352:483 chunk contains the 32-step prompt at
        # its beginning.  We discard those 32 steps when stitching.
        # --------------------------------------------------------------
        else:

            prompt_start = chunk_start
            prompt_end = chunk_start + overlap

            print(
                f"  Melody: {chunk_start}:{chunk_end}"
            )
            print(
                f"  Chord prompt: "
                f"{prompt_start}:{prompt_end}"
            )
            print(
                f"  Generated chunk: "
                f"{chunk_start}:{chunk_end}"
            )
            print(
                f"  Retained: "
                f"{prompt_end}:{chunk_end}"
            )

            # Each sample has its own generated chord history,
            # therefore continuation chunks must be generated
            # separately per sample.
#EXCLUDE            
            for i in range(samples):

                previous_output = final_outputs[i]

                # The previous output already contains the complete
                # chord timeline generated so far.
                chord_prompt = torch.stack(
                    previous_output[
                        prompt_start:prompt_end
                    ],
                    dim=1,
                )

                # previous_output entries have shape [1, 32].
                # stack(..., dim=1) therefore gives [1, 32, 32].
                assert chord_prompt.shape[1] == overlap, (
                    f"Expected {overlap} prompt steps, got "
                    f"{chord_prompt.shape[1]}"
                )

                with torch.no_grad():

                    output = model.global_sampling(
                        melody_chunk,
                        chord_prompt,
                        temperature=temperature,
                    )

                output_i = [
                    output[j][0:1, :]
                    for j in range(len(output))
                ]

                # The first `overlap` positions are the prompt we
                # supplied to this generation.  They overlap the
                # previous chunk, so do NOT append them again.
                new_output = output_i[overlap:]

                final_outputs[i].extend(new_output)
#END EXCLUDE

        previous_chunk_end = chunk_end
        if previous_chunk_end >= generation_length:
            break
        chunk_number += 1

    # ------------------------------------------------------------------
    # Sanity check
    # ------------------------------------------------------------------

    for i, output_i in enumerate(final_outputs):

        if len(output_i) != generation_length:
            raise RuntimeError(
                f"Sample {i + 1}: expected "
                f"{generation_length} chord steps, got "
                f"{len(output_i)}"
            )

    # ------------------------------------------------------------------
    # Write final MIDI files.
    # ------------------------------------------------------------------

    for i, output_i in enumerate(final_outputs):

        output_file = os.path.join(
            output_dir,
            f"proposal_{i + 1:02d}.mid",
        )

        print(
            f"Writing: {output_file}"
        )

        decode_output(
            (output_i,),
            output_file,
            ratio=(
                model.compress_ratio_l,
                model.compress_ratio_r,
            ),
            tempo=bpm,
            velocity=100,
            fixed_program=CHORD_PROGRAM,
            extra_instruments=extra_melody,
        )
        remove_initial_chord_notes(
            output_file,
            cutoff_steps=prompt_length,
        )
#NEW END

    print()
    print("DONE.")
    print(f"Output directory: {output_dir}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Melody → Chord inference for MIDI Function Alignment."
    )

    parser.add_argument(
        "input",
        help="Input melody MIDI.",
    )

    parser.add_argument(
        "--key",
        required=True,
        help="Key used for the two-bar tonic prompt, e.g. 'C minor' or 'Eb major'.",
    )

    parser.add_argument(
        "--bpm",
        type=float,
        default=None,
        help="Fallback BPM if the MIDI contains no tempo map.",
    )

    parser.add_argument(
        "--time-signature",
        default=None,
        help="Fallback time signature if the MIDI contains none, e.g. 4/4.",
    )

    parser.add_argument(
        "--model",
        default=(
            "ckpt/mel_to_chord/"
            "cp_transformer_yinyang_v5.1_lora_batch_8_"
            "nottingham_cp8_v2_chord_mel_rev_mask0.0-10-step1."
            "epoch=last.ckpt"
        ),
        help="Melody → chord checkpoint.",
    )

    parser.add_argument(
        "--output-dir",
        default="./test/outputs",
        help="Output directory.",
    )

    parser.add_argument(
        "--generation-length",
        type=int,
        default=None,
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_TEMPERATURE,
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=DEFAULT_SAMPLES,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
    )

    args = parser.parse_args()
    if args.generation_length is None:
        args.generation_length = read_input_length(args.input)

    # ---------------------------------------------------------------
    # Read metadata BEFORE creating the synthetic prompt MIDI.
    # ---------------------------------------------------------------

    metadata = get_midi_metadata(
        args.input,
        cli_bpm=args.bpm,
        cli_time_signature=args.time_signature,
    )

    bpm = metadata["bpm"]
    numerator = metadata["numerator"]
    denominator = metadata["denominator"]

    print()
    print("=== MIDI METADATA ===")
    print(f"Tempo:          {bpm:.3f} BPM ({metadata['bpm_source']})")
    print(
        f"Time signature: {numerator}/{denominator} "
        f"({metadata['time_signature_source']})"
    )

    if metadata["tempo_map"]:
        print("Tempo map:")
        for tick, tempo in metadata["tempo_map"]:
            print(f"  tick {tick}: {tempo:.3f} BPM")
    else:
        print("Tempo map:       none")

    if metadata["time_signature_map"]:
        print("Time-signature map:")
        for tick, num, den in metadata["time_signature_map"]:
            print(f"  tick {tick}: {num}/{den}")
    else:
        print("Time-signature map: none")

    # Two bars × quarter-note units × four 16th notes.
    quarter_notes_per_bar = numerator * (4.0 / denominator)
    prompt_length = int(round(
        2 * quarter_notes_per_bar * 4
    ))

    print(f"Prompt length:   {prompt_length} 16th-note steps")
    print()

    # ---------------------------------------------------------------
    # Create synthetic two-track MIDI.
    # ---------------------------------------------------------------

    base_name = os.path.splitext(
        os.path.basename(args.input)
    )[0]

    safe_key = args.key.replace(" ", "_")

    synthetic_input = os.path.join(
        args.output_dir,
        f"{base_name}_{safe_key}_prompt.mid",
    )

    print("Creating melody + two-bar chord prompt...")
    print(f"Key:             {args.key}")
    print(f"Chord pitches:   {make_tonic_triad(args.key)}")
    print(f"Prompt MIDI:     {synthetic_input}")

    create_melody_chord_input(
        args.input,
        synthetic_input,
        args.key,
    )

    # ---------------------------------------------------------------
    # Load downstream model.
    # ---------------------------------------------------------------

    print()
    print("Loading model...")
    print(args.model)

    model = RoformerYinyang.load_from_checkpoint(
        args.model,
        strict=False,
    )

    model.save_name = os.path.basename(args.model)

    model.cuda()
    model.eval()

    # ---------------------------------------------------------------
    # Generate.
    # ---------------------------------------------------------------

    generate(
        model=model,
        input_midi=synthetic_input,
        output_dir=args.output_dir,
        bpm=bpm,
        prompt_length=prompt_length,
        generation_length=args.generation_length,
        temperature=args.temperature,
        samples=args.samples,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()

    