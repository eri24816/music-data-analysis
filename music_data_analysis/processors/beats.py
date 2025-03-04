from ..processor import Processor
from ..data_access import Song
from beat_this.inference import File2Beats, load_checkpoint


class BeatsProcessor(Processor):

    input_props = ["synth"]
    output_props = ["beats"]

    # It hangs if instantiated in workers.
    max_num_processes = 1

    def prepare_main_process(self):
        # in the main process, make sure the model is downloaded
        print("Downloading beats model if needed...")
        File2Beats(device="cpu")

    def prepare(self):
        self.file2beats = File2Beats(device="cuda", dbn=True)

    def process_impl(self, song: Song):
        beats, downbeats = self.file2beats(song.get_old_path("synth"))
        
        if len(beats) < 4 or len(downbeats) == 0:
            song.write_json("beats", beats=list(beats), start_beat=1)
            return

        for i in range(4):
            if downbeats[0] == beats[i]:
                start_beat = 4 - i
        song.write_json("beats", beats=list(beats), start_beat=start_beat)
