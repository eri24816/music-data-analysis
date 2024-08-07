from pathlib import Path
from music_data_analysis.apply import apply_to_dataset
from music_data_analysis.data_access import Dataset
from music_data_analysis.processors.note_stat import (
    PolyphonyProcessor,
    VelocityProcessor,
    HighestPitchProcessor,
    LowestPitchProcessor,
)


if __name__ == "__main__":
    dataset = Dataset(Path("./data"))
    # proc = SynthProcessor(Path("W:/music/FluidR3_GM/FluidR3_GM.sf2"))
    # apply_to_dataset(dataset, proc, num_processes=4)

    # proc = BeatsProcessor()
    # apply_to_dataset(dataset, proc, num_processes=4)

    # proc = AlignAndSyncProcessor()
    # apply_to_dataset(dataset, proc, num_processes=8)
    # apply_to_dataset(dataset, AlignAndSyncProcessor(), num_processes=4)

    # apply_to_dataset(dataset, PianoRollProcessor(), num_processes=8)

    # apply_to_dataset(dataset, ChordProcessor(), num_processes=4)

    # apply_to_dataset(dataset, NoteDensityProcessor(), num_processes=4)
    apply_to_dataset(dataset, PolyphonyProcessor(), num_processes=4)
    apply_to_dataset(dataset, VelocityProcessor(), num_processes=4)
    apply_to_dataset(dataset, HighestPitchProcessor(), num_processes=4)
    apply_to_dataset(dataset, LowestPitchProcessor(), num_processes=4)
