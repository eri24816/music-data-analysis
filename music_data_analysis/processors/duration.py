from ..processor import Processor
from ..data_access import Song


class DurationProcessor(Processor):
    def process(self, song: Song):
        pianoroll = song.read_pianoroll("pianoroll")
        song.write_json("duration", pianoroll.duration)
