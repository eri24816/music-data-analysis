import multiprocessing
from pathlib import Path
from shutil import copytree
import shutil
import tempfile
from typing import Literal

from music_data_analysis.processors.misc import ReconstructProcessor
from .apply import apply_to_dataset
from .data_access import Dataset

from .processors.pianoroll import PianoRollProcessor
from .processors.duration import DurationProcessor
from .processors.pitch import PitchProcessor
from .processors.note_stat import NoteDensityProcessor
from .processors.note_stat import PolyphonyProcessor
from .processors.note_stat import VelocityProcessor
from .processors.chord import ChordProcessor

def align_and_sync_midi(dataset: Dataset, soundfont_path: Path, align=True, sync=True, frames_per_beat: int|None=None, verbose=True, num_processes: int = 1, num_shards: int = 1, shard_id: int = 0, overwrite_existing: bool = False):
    # lazy import here because they contain optional dependencies
    from .processors.synth import SynthProcessor
    from .processors.align_sync import AlignAndSyncProcessor
    from .processors.beats import BeatsProcessor
    from .processors.key import KeyProcessor
    procs = [
        SynthProcessor(soundfont_path),
        BeatsProcessor(),
        KeyProcessor(),
        AlignAndSyncProcessor(align_midi=align, sync_midi=sync, frames_per_beat=frames_per_beat)
    ]

    for proc in procs:
        apply_to_dataset(dataset, proc, num_processes=num_processes, verbose=verbose, num_shards=num_shards, shard_id=shard_id, overwrite_existing=overwrite_existing)

def to_pianoroll(dataset: Dataset, verbose=True, beats_per_bar: int = 4, frames_per_beat: int = 16, save_format: Literal["torch", "json"] = "torch", num_processes: int = 1, num_shards: int = 1, shard_id: int = 0, overwrite_existing: bool = False):
    procs = [
        PianoRollProcessor(beats_per_bar=beats_per_bar, frames_per_beat=frames_per_beat, save_format=save_format)
    ]

    for proc in procs:
        apply_to_dataset(dataset, proc, num_processes=num_processes, verbose=verbose, num_shards=num_shards, shard_id=shard_id, overwrite_existing=overwrite_existing)

def extract_features(dataset: Dataset, verbose=True, num_processes: int = 1, num_shards: int = 1, shard_id: int = 0, overwrite_existing: bool = False):
    procs = [
        # DurationProcessor(),
        # PitchProcessor(),
        # NoteDensityProcessor(),
        # PolyphonyProcessor(),
        VelocityProcessor(),
        # ChordProcessor()
    ]

    for proc in procs:
        apply_to_dataset(dataset, proc, num_processes=num_processes, verbose=verbose, num_shards=num_shards, shard_id=shard_id, overwrite_existing=overwrite_existing)

def recon_midi(dataset: Dataset, verbose=True, num_processes: int = 1, num_shards: int = 1, shard_id: int = 0, overwrite_existing: bool = False):
    procs = [
        ReconstructProcessor()
    ]

    for proc in procs:
        apply_to_dataset(dataset, proc, num_processes=num_processes, verbose=verbose, num_shards=num_shards, shard_id=shard_id, overwrite_existing=overwrite_existing)

def run(src_path:str|Path, sync:bool=False, verbose=True, soundfont_path:Path|None=None, frames_per_beat: int|None=None, overwrite_existing: bool = False) -> Dataset:
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
        assert soundfont_path is not None, "soundfont_path is required when sync is True"
        align_and_sync_midi(dataset, soundfont_path, verbose=verbose, frames_per_beat=frames_per_beat, overwrite_existing=overwrite_existing)
    to_pianoroll(dataset, verbose=verbose, overwrite_existing=overwrite_existing)
    extract_features(dataset, verbose=verbose, overwrite_existing=overwrite_existing)
    return dataset


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", "--path", type=Path, required=True, help="Path to the dataset directory to be created, where all the processed data will be stored")
    parser.add_argument("--src_midi_dir", "--src", type=Path, default=None, help="Path to the source MIDI directory from which the content will be copied to the {dataset_path}/midi/ directory. If not provided, it will assume the {dataset_path}/midi/ directory is already present")
    parser.add_argument("--symlink_to_src", "-l", action="store_true", help="Create a symlink to the source MIDI directory instead of copying the content from it")
    parser.add_argument("--num_processes", "-n", type=int, default=None, help="Number of processes to use for the processing")
    parser.add_argument("--num_shards", "-ns", type=int, default=1, help="Number of shards to use for the processing")
    parser.add_argument("--start_shard", "-ss", type=int, default=0, help="Start shard to use for the processing")
    parser.add_argument("--search_index", "-i", type=str, default='midi', help="The property to search songs in the dataset")
    parser.add_argument("--overwrite_existing", "-o", action="store_true", help="Overwrite existing files even if they are already present. If not present, the program will skip the files that already exist.")
    
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    parser.add_argument("--sync", "-s", action="store_true", help="Sync the MIDI files")
    parser.add_argument("--align", "-a", action="store_true", help="Align the MIDI files")
    parser.add_argument("--to_pianoroll", "-p", action="store_true", help="Convert the MIDI files to pianoroll")
    parser.add_argument("--extract_features", "-e", action="store_true", help="Extract features from the MIDI files")
    parser.add_argument("--reconstruct_midi", "-r", action="store_true", help="Reconstruct the MIDI files from the pianoroll, for debugging purposes")

    parser.add_argument("--soundfont_path", "-sf", type=Path, default=None, help="Path to the soundfont file to be used for the MIDI files. Required when --sync or --align is True.")
    parser.add_argument("--beats_per_bar", type=int, default=4, help="Beats per bar")
    parser.add_argument("--frames_per_beat", type=int, default=16, help="Frames per beat")
    parser.add_argument("--save_format", type=str, default="torch", help="Save format for the pianorolls. Either 'torch' or 'json'")
    args = parser.parse_args()

    if args.sync or args.align:
        assert args.soundfont_path is not None, "--soundfont_path is required when --sync or --align is True"

    if args.to_pianoroll:
        assert args.save_format in ["torch", "json"], "--save_format must be either 'torch' or 'json'"

    if args.num_processes is None:
        args.num_processes = multiprocessing.cpu_count() // 2


    # copy all files from src_midi_dir to dataset_path/midi/
    if args.src_midi_dir:
        if args.dataset_path.exists():
            respond = input(f"{args.dataset_path} already exists. Do you want to overwrite it? (y/n)")
            if respond == "y":
                shutil.rmtree(args.dataset_path)
            else:
                exit()
        
        args.dataset_path.mkdir(parents=True, exist_ok=False)

        if args.symlink_to_src:
            print(f"Creating symlink to {args.src_midi_dir} at {args.dataset_path / 'midi'}")
            (args.dataset_path / "midi").symlink_to(args.src_midi_dir) 
        else:
            print(f"Copying {args.src_midi_dir} to {args.dataset_path / 'midi'}")
            copytree(args.src_midi_dir, args.dataset_path / "midi")
    else:
        args.dataset_path.mkdir(parents=True, exist_ok=True)

    dataset = Dataset(Path(args.dataset_path), args.search_index)

    print(f"Dataset initialized at {args.dataset_path}")

    def run_shard(shard_id: int):
        if args.sync or args.align:
            align_and_sync_midi(dataset, args.soundfont_path, align=args.align, sync=args.sync, frames_per_beat=args.frames_per_beat, verbose=args.verbose, num_processes=args.num_processes, num_shards=args.num_shards, shard_id=shard_id, overwrite_existing=args.overwrite_existing)

        if args.to_pianoroll:
            to_pianoroll(dataset, verbose=args.verbose, beats_per_bar=args.beats_per_bar, frames_per_beat=args.frames_per_beat, save_format=args.save_format, num_processes=args.num_processes, num_shards=args.num_shards, shard_id=shard_id, overwrite_existing=args.overwrite_existing)

        if args.extract_features:
            extract_features(dataset, verbose=args.verbose, num_processes=args.num_processes, num_shards=args.num_shards, shard_id=shard_id, overwrite_existing=args.overwrite_existing)

        if args.reconstruct_midi:
            recon_midi(dataset, verbose=args.verbose, num_processes=args.num_processes, num_shards=args.num_shards, shard_id=shard_id, overwrite_existing=args.overwrite_existing)

        # clean up files in synth/ for this shard, so disk doesn't blow up
        print(f"Cleaning up files in synth/ for shard {shard_id}")
        if (args.dataset_path / "synth").exists():
            shutil.rmtree(args.dataset_path / "synth")


    for shard_id in range(args.start_shard, args.num_shards):
        run_shard(shard_id)

    print(f"Finished processing dataset at {args.dataset_path}")


if __name__ == "__main__":
    main()