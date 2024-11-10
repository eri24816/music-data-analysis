from ..processor import Processor
from ..data_access import Song
from chorder import Dechorder, Chord

note_unify_map = {
    'C': 'C',
    'C#': 'C#',
    'Db': 'C#',
    'D': 'D',
    'D#': 'D#',
    'Eb': 'D#',
    'E': 'E',
    'Fb': 'E',
    'F': 'F',
    'E#': 'F',
    'F#': 'F#',
    'Gb': 'F#',
    'G': 'G',
    'G#': 'G#',
    'Ab': 'G#',
    'A': 'A',
    'A#': 'A#',
    'Bb': 'A#',
    'B': 'B',
    'Cb': 'B',
}

def chord_to_name(chord: Chord):
    root = chord.root()
    quality = chord.quality
    if root is None or quality is None:
        return 'None'
    return note_unify_map[root] + '_' + quality

class ChordProcessor(Processor):
    def process(self, song: Song):
        midi = song.read_midi('synced_midi')
        chords = Dechorder.dechord(midi, scale=None)
        chords: list[Chord]
        data_to_write = {
            'name': [chord_to_name(chord) for chord in chords],
            'root': [chord.root() if chord.root() is not None else 'None' for chord in chords],
            'quality': [chord.quality if chord.quality is not None else 'None' for chord in chords],
            'bass': [chord.bass() if chord.bass() is not None else 'None' for chord in chords],
        }
        song.write_json("chords", data_to_write)
