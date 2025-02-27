from ..processor import Processor
from ..data_access import Song


class DurationProcessor(Processor):
    input_props = ["synced_midi"]
    output_props = ["duration"]

    def process_impl(self, song: Song):
        pianoroll = song.read_pianoroll("pianoroll")
        song.write_json("duration", pianoroll.duration)
