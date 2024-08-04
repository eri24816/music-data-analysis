from pathlib import Path
from music_data_analysis.apply import apply_to_dataset
from music_data_analysis.data_access import Dataset
from music_data_analysis.processors.align_sync import AlignAndSyncProcessor


if __name__ == "__main__":
    # proc = SynthProcessor(Path("W:/music/FluidR3_GM/FluidR3_GM.sf2"))
    # dataset = Dataset(Path("./data"))
    # apply_to_dataset(dataset, proc, num_processes=4)

    # proc = BeatsProcessor()
    # dataset = Dataset(Path("./data"))
    # apply_to_dataset(dataset, proc, num_processes=4)

    proc = AlignAndSyncProcessor()
    dataset = Dataset(Path("./data"))
    apply_to_dataset(dataset, proc, num_processes=8)
