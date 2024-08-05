from pathlib import Path
from music_data_analysis.apply import apply_to_dataset
from music_data_analysis.data_access import Dataset
from music_data_analysis.processors.key import KeyProcessor


if __name__ == "__main__":
    dataset = Dataset(Path("./data"))
    # proc = SynthProcessor(Path("W:/music/FluidR3_GM/FluidR3_GM.sf2"))
    # apply_to_dataset(dataset, proc, num_processes=4)

    # proc = BeatsProcessor()
    # apply_to_dataset(dataset, proc, num_processes=4)

    # proc = AlignAndSyncProcessor()
    # apply_to_dataset(dataset, proc, num_processes=8)

    # proc = PianoRollProcessor()
    # apply_to_dataset(dataset, proc, num_processes=8)

    # proc = ChordProcessor()

    apply_to_dataset(dataset, KeyProcessor(), num_processes=4)
