from pathlib import Path
from shutil import copytree
import shutil
import tempfile
from music_data_analysis.apply import apply_to_dataset
from music_data_analysis.data_access import Dataset

from music_data_analysis.processors.pianoroll import PianoRollProcessor
from music_data_analysis.processors.duration import DurationProcessor
from music_data_analysis.processors.pitch import PitchProcessor
from music_data_analysis.processors.note_stat import NoteDensityProcessor
from music_data_analysis.processors.note_stat import PolyphonyProcessor
from music_data_analysis.processors.note_stat import VelocityProcessor
from music_data_analysis.processors.chord import ChordProcessor

def sync_midi(dataset: Dataset, verbose=True):
    # lazy import here because they have optional dependencies
    from music_data_analysis.processors.synth import SynthProcessor
    from music_data_analysis.processors.align_sync import AlignAndSyncProcessor
    from music_data_analysis.processors.beats import BeatsProcessor
    from music_data_analysis.processors.key import KeyProcessor
    procs = [
        SynthProcessor(Path("W:/music/FluidR3_GM/FluidR3_GM.sf2")),
        BeatsProcessor(),
        KeyProcessor(),
        AlignAndSyncProcessor()
    ]

    for proc in procs:
        apply_to_dataset(dataset, proc, num_processes=1, verbose=verbose)

def to_pianoroll(dataset: Dataset, verbose=True):
    procs = [
        PianoRollProcessor()
    ]

    for proc in procs:
        apply_to_dataset(dataset, proc, num_processes=1, verbose=verbose)

def extract_features(dataset: Dataset, verbose=True):
    procs = [
        DurationProcessor(),
        PitchProcessor(),
        NoteDensityProcessor(),
        PolyphonyProcessor(),
        VelocityProcessor(),
        ChordProcessor()
    ]

    for proc in procs:
        apply_to_dataset(dataset, proc, num_processes=1, verbose=verbose)

def run(src_path:str|Path, sync:bool=False, verbose=True) -> Dataset:
    src_path = Path(src_path)
    dataset_path = Path(tempfile.mkdtemp())
    while Path(dataset_path).exists():
        dataset_path = dataset_path.with_name(dataset_path.name + '_1')

    dataset_path.mkdir()
    if sync:
        if src_path.is_dir():
            # copy all files from src_path to dataset_path/midi/
            copytree(src_path, dataset_path / 'midi')
        else:
            # copy src_path to dataset_path/midi/
            (dataset_path / 'midi').mkdir()
            shutil.copy(src_path, dataset_path / 'midi')

            dataset = Dataset(dataset_path, "midi")  
            
    else:
        if src_path.is_dir():
            # copy all files from src_path to dataset_path/midi/
            copytree(src_path, dataset_path / 'synced_midi')
        else:
            # copy src_path to dataset_path/midi/
            (dataset_path / 'synced_midi').mkdir()
            shutil.copy(src_path, dataset_path / 'synced_midi')

        dataset = Dataset(dataset_path, "synced_midi")

    if sync:
        sync_midi(dataset, verbose=verbose)
    to_pianoroll(dataset, verbose=verbose)
    extract_features(dataset, verbose=verbose)
    return dataset


if __name__ == "__main__":
    dataset = Dataset(Path("./data"), "synced_midi")
    to_pianoroll(dataset)
    extract_features(dataset)
