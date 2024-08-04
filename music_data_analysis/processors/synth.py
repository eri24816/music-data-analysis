from pathlib import Path
from ..processor import Processor
from ..data_access import Song
from ..utils import run_command


class SynthProcessor(Processor):
    def __init__(self, sound_font_path: Path, bit_rate: int = 96):
        self.sound_font_path = sound_font_path
        self.bit_rate = bit_rate

    def process(self, song: Song):
        midi_file = song.get_old_path("midi")
        output_file = song.get_new_path("synth", "mp3")
        run_command(
            f"fluidsynth -T raw -F - {self.sound_font_path} {midi_file} | ffmpeg -y -f s32le -i - -b:a {self.bit_rate}k {output_file}"
        )
