from typing import Literal
from ..data.pianoroll import Pianoroll
from ..processor import Processor
from ..data_access import Song


class PianoRollProcessor(Processor):
    """
    Converts a MIDI file to a Pianoroll.
    """

    input_props = ["synced_midi"]
    output_props = ["pianoroll"]

    def __init__(self, beats_per_bar: int = 4, frames_per_beat: int = 16, save_format: Literal["torch", "json"] = "torch"):
        self.beats_per_bar = beats_per_bar
        self.frames_per_beat = frames_per_beat
        self.save_format: Literal["torch", "json"] = save_format
        self.ext = "pt" if save_format == "torch" else "json"

    def process_impl(self, song: Song):
        midi = song.get_old_path("synced_midi")
        Pianoroll.from_midi(
            midi,
            beats_per_bar=self.beats_per_bar,
            frames_per_beat=self.frames_per_beat
        ).save(
            song.get_new_path("pianoroll", self.ext),
            format=self.save_format
        )
