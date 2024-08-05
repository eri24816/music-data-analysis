from ..data.pianoroll import PianoRoll
from ..processor import Processor
from ..data_access import Song

scale = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
quality = {
    "": [0, 4, 7],
    "m": [0, 3, 7],
    "7": [0, 4, 7, 10, 10],  # the 7th note is important
    "m7": [0, 3, 7, 10, 10],
}
chord_to_chroma = {}
chord_to_chroma_one_hot = {}
for i, s in enumerate(scale):
    for q in quality:
        chord_to_chroma[f"{s}{q}"] = [(x + i) % 12 for x in quality[q]]
        chord_to_chroma_one_hot[f"{s}{q}"] = [0] * 12
        for x in quality[q]:
            chord_to_chroma_one_hot[f"{s}{q}"][x] = 1


def chroma_to_chord(query):
    scores: dict[str, float] = {}
    for chord, chroma in chord_to_chroma.items():
        score = 0
        for i in chroma:
            score += query[i]
        scores[chord] = score / len(chroma)
    argmax = max(scores, key=scores.__getitem__)
    return argmax


def pitch_chroma_weight(pitch):
    """
    Lower pitches have more weight. An octave lower is 1.8 times more important.
    """
    return 2 ** (-pitch / 12)


def get_chord_sequence(pr: PianoRoll, granularity):
    """
    Get the chord sequence of the pianoroll (heuristic).
    """
    chords = []
    last_segment_chroma = [0] * 12
    for bar in pr.iter_over_bars_unpack(granularity):
        already_pressed_pitches = set()
        chroma = [value * 0.005 for value in last_segment_chroma]
        for time, pitch, vel, offset in bar:
            if pitch in already_pressed_pitches:
                continue
            chroma[pitch % 12] += (
                pitch_chroma_weight(pitch)
                * (granularity - (time % granularity))
                / granularity
            )
            print(time, (granularity - (time % granularity)), pitch)
            print(chroma)
            already_pressed_pitches.add(pitch)
        chord = chroma_to_chord(chroma)
        chords.append(chord)
        last_segment_chroma = chord_to_chroma_one_hot[chord]
        if time > 16:
            a
    return chords


class ChordProcessor(Processor):
    def process(self, song: Song):
        pianoroll = PianoRoll.load(song.get_old_path("pianoroll"))
        chords = get_chord_sequence(pianoroll, granularity=16)
        song.write_json("chords", chords)
