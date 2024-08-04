from ..processor import Processor
from ..data_access import Song
from beat_this.inference import File2Beats


class BeatsProcessor(Processor):
    def prepare(self):
        self.file2beats = File2Beats(device="cuda", dbn=True)

    def process(self, song: Song):
        beats, downbeats = self.file2beats(song.get_old_path("mp3"))
        for i in range(4):
            if downbeats[0] == beats[i]:
                start_beat = 4 - i
        song.write_json("beats", beats=list(beats), start_beat=start_beat)
