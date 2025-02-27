from ..data.pianoroll import Pianoroll
from ..processor import Processor
from ..data_access import Song

scale = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
quality = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "7": [0, 4, 7, 10],
    "m7": [0, 3, 7, 10],
    "M7": [0, 4, 7, 11],
}
chord_to_chroma = {}
chord_to_chroma_one_hot = {}
for i, s in enumerate(scale):
    for q in quality:
        chord_to_chroma[f"{s}{q}"] = [(x + i) % 12 for x in quality[q]]
        chord_to_chroma_one_hot[f"{s}{q}"] = [0] * 12
        for x in quality[q]:
            chord_to_chroma_one_hot[f"{s}{q}"][(x + i) % 12] = 1

chord_prior = {
    "C": 1,
    "G": 0.8,
    "F": 0.7,
    "Am": 0.9,
    "Em": 0.7,
    "Dm": 0.65,
    "D": 0.3,
    "A": 0.3,
    "E": 0.3,
    "A#": 0.3,
}

# fill other natural chords with 0.2
for chord in chord_to_chroma.keys():
    if chord not in chord_prior:
        chord_prior[chord] = 0.2

# fill m, 7, m7 with 0.8 of the original chord
for chord, prior in chord_prior.copy().items():
    for q in ["m", "7", "m7", "M7"]:
        new_chord = chord + q
        if new_chord in chord_to_chroma:
            continue
        chord_to_chroma[new_chord] = chord_to_chroma[chord]
        chord_to_chroma_one_hot[new_chord] = chord_to_chroma_one_hot[chord]
        chord_prior[new_chord] = prior * 0.8


def sigmoid(x):
    return 1 / (1 + 2.718281828459045**-x)

def chroma_to_chord(query):
    scores: dict[str, float] = {}
    for chord, chroma in chord_to_chroma_one_hot.items():
        score = 1.0

        # bayesian like stuff
        # asume p(pitch1, pitch2, ..., pitch12 | chord) = p(pitch1 | chord) * p(pitch2 | chord) * ... * p(pitch12 | chord)
        for i in range(12):
            pitch_is_present = sigmoid(query[i] - 5)
            this_prob = ((chroma[i] + 0.1) ** pitch_is_present) * (
                (1 - chroma[i] + 0.1) ** (1 - pitch_is_present)
            )
            score *= this_prob
        scores[chord] = score * chord_prior[chord]
    argmax = max(scores, key=scores.__getitem__)
    return argmax


def pitch_chroma_weight(pitch):
    """
    Lower pitches have more weight. An octave lower is 1.8 times more important.
    """
    return 1.4 ** (-pitch / 12)


def get_chord_sequence(pr: Pianoroll, granularity):
    """
    Get the chord sequence of the pianoroll (heuristic).
    """
    chords = []
    last_segment_chroma = [0] * 12
    for bar in pr.iter_over_bars_unpack(granularity):
        already_pressed_pitches = set()
        chroma = [value * 0.05 for value in last_segment_chroma]
        for time, pitch, vel, offset in bar:
            if pitch in already_pressed_pitches:
                continue
            chroma[pitch % 12] += (
                pitch_chroma_weight(pitch)
                * (granularity - (time % granularity))
                / granularity
            ) * 30
            already_pressed_pitches.add(pitch)
        chord = chroma_to_chord(chroma)
        chords.append(chord)
        last_segment_chroma = chord_to_chroma_one_hot[chord]
    return chords


class ChordProcessor(Processor):
    input_props = ["pianoroll"]
    output_props = ["chords"]

    def process_impl(self, song: Song):
        pianoroll = song.read_pianoroll('pianoroll')
        chords = get_chord_sequence(pianoroll, granularity=16)
        song.write_json("chords", chords)
