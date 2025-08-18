from pathlib import Path
from ..processor import Processor
from ..data_access import Song
from ..utils import run_command


class SynthProcessor(Processor):
    input_props = ["midi"]
    output_props = ["synth"]

    def __init__(self, sound_font_path: Path, bit_rate: int = 96):
        self.sound_font_path = sound_font_path
        self.bit_rate = bit_rate

    def process_impl(self, song: Song):
        midi_file = song.get_old_path("midi")
        output_file = song.get_new_path("synth", "wav")

        output_file.parent.mkdir(parents=True, exist_ok=True)

        # A mysterious 0.05s trim needed to make timing correct.
        # print(
        #     f'fluidsynth -T raw -F - {self.sound_font_path} {midi_file} | ffmpeg -hide_banner -loglevel error -y -f s32le -i - -af "atrim=0.05" -b:a {self.bit_rate}k {output_file}'
        # )

        run_command(
            f'fluidsynth -F {output_file}  {self.sound_font_path} {midi_file}' 
        )
