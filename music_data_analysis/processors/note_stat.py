from ..processor import Processor
from ..data_access import Song
from ..data.pianoroll import Pianoroll


class DensityProcessor(Processor):
    input_props = ["pianoroll"]
    output_props = ["density"]

    def process_impl(self, song: Song):
        pianoroll = Pianoroll.load(song.get_old_path("pianoroll"))
        density = pianoroll.get_density()
        song.write_json("density", density)


class PolyphonyProcessor(Processor):
    input_props = ["pianoroll"]
    output_props = ["polyphony"]

    def process_impl(self, song: Song):
        pianoroll = Pianoroll.load(song.get_old_path("pianoroll"))
        polyphony = pianoroll.get_polyphony()
        song.write_json("polyphony", polyphony)


class VelocityProcessor(Processor):
    input_props = ["pianoroll"]
    output_props = ["velocity"]

    def process_impl(self, song: Song):
        pianoroll = Pianoroll.load(song.get_old_path("pianoroll"))
        velocity = pianoroll.get_velocity()
        song.write_json("velocity", velocity)


class HighestPitchProcessor(Processor):
    input_props = ["pianoroll"]
    output_props = ["highest_pitch"]

    def process_impl(self, song: Song):
        pianoroll = Pianoroll.load(song.get_old_path("pianoroll"))
        highest_pitch = pianoroll.get_highest_pitch()
        song.write_json("highest_pitch", highest_pitch)


class LowestPitchProcessor(Processor):
    input_props = ["pianoroll"]
    output_props = ["lowest_pitch"]

    def process_impl(self, song: Song):
        pianoroll = Pianoroll.load(song.get_old_path("pianoroll"))
        lowest_pitch = pianoroll.get_lowest_pitch()
        song.write_json("lowest_pitch", lowest_pitch)
