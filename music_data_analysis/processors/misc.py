from ..processor import Processor
from ..data_access import Song


class ReconstructProcessor(Processor):
    """
    Converts a MIDI file to a piano roll.
    """

    def process(self, song: Song):
        pr = song.read_pianoroll("pianoroll")
        pr.to_midi(song.get_new_path("recon_midi", "mid"))