from ..processor import Processor
from ..data_access import Song
from ..data.pianoroll import PianoRoll


class NoteDensityProcessor(Processor):
    def process(self, song: Song):
        pianoroll = PianoRoll.load(song.get_old_path("pianoroll"))
        note_density = pianoroll.get_density()
        song.write_json("note_density", note_density)
