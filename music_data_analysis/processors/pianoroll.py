from ..data.pianoroll import Pianoroll
from ..processor import Processor
from ..data_access import Song


class PianoRollProcessor(Processor):
    """
    Converts a MIDI file to a piano roll.
    """

    def process(self, song: Song):
        midi = song.get_old_path("synced_midi")
        Pianoroll.from_midi(midi).save(song.get_new_path("pianoroll", "json"))
