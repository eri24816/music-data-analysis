from ..data_access import Song
from ..processor import Processor

from madmom.features import CNNKeyRecognitionProcessor


class KeyProcessor(Processor):
    input_props = ["synth"]
    output_props = ["key"]

    def prepare(self):
        self.key_processor = CNNKeyRecognitionProcessor()

    def process_impl(self, song: Song):
        prediction = self.key_processor(str(song.get_old_path("synth")))
        key = prediction.argmax()

        song.write_json("key", {"key": int(key), "prediction": prediction[0].tolist()})
