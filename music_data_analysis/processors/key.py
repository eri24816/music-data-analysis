from ..data_access import Song
from ..processor import Processor

from madmom.features import CNNKeyRecognitionProcessor


class KeyProcessor(Processor):
    def prepare(self):
        self.key_processor = CNNKeyRecognitionProcessor()

    def process(self, song: Song):
        if song.exists("key"):
            return
        prediction = self.key_processor(str(song.get_old_path("synth")))
        key = prediction.argmax()

        song.write_json("key", {"key": int(key), "prediction": prediction[0].tolist()})
