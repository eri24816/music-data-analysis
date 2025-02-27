from ..processor import Processor
from ..data_access import Song


class ReconstructProcessor(Processor):
    """
    Converts a MIDI file to a piano roll.
    """

    input_props = ["pianoroll"]
    output_props = ["recon_midi"]

    def process_impl(self, song: Song):
        pr = song.read_pianoroll("pianoroll")
        pr.to_midi(song.get_new_path("recon_midi", "mid"))