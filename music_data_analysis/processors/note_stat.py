from ..processor import Processor
from ..data_access import Song
from ..data.pianoroll import PianoRoll


class NoteDensityProcessor(Processor):
    def process(self, song: Song):
        pianoroll = PianoRoll.load(song.get_old_path("pianoroll"))
        note_density = pianoroll.get_density()
        song.write_json("note_density", note_density)


class PolyphonyProcessor(Processor):
    def process(self, song: Song):
        pianoroll = PianoRoll.load(song.get_old_path("pianoroll"))
        polyphony = pianoroll.get_polyphony()
        song.write_json("polyphony", polyphony)


class VelocityProcessor(Processor):
    def process(self, song: Song):
        pianoroll = PianoRoll.load(song.get_old_path("pianoroll"))
        velocity = pianoroll.get_velocity()
        song.write_json("velocity", velocity)


class HighestPitchProcessor(Processor):
    def process(self, song: Song):
        pianoroll = PianoRoll.load(song.get_old_path("pianoroll"))
        highest_pitch = pianoroll.get_highest_pitch()
        song.write_json("highest_pitch", highest_pitch)


class LowestPitchProcessor(Processor):
    def process(self, song: Song):
        pianoroll = PianoRoll.load(song.get_old_path("pianoroll"))
        lowest_pitch = pianoroll.get_lowest_pitch()
        song.write_json("lowest_pitch", lowest_pitch)
