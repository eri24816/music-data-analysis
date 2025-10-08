
from pathlib import Path
from music_data_analysis.processors.align_sync import AlignAndSyncProcessor
from music_data_analysis.processors.segmentation import SegmentationProcessor
from music_data_analysis.processors.synth import SynthProcessor
from music_data_analysis.processors.duration import DurationProcessor

processor_cls_dict = {
    "segmentation": SegmentationProcessor,
    "synth": SynthProcessor,
    "align_sync": AlignAndSyncProcessor,
    "duration": DurationProcessor,
}

def main():
    from music_data_analysis.data_access import Dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("processor", type=str)
    parser.add_argument("num_processes", type=int, default=1)

    # processor specific arguments (kwargs)
    parser.add_argument("processor_kwargs", nargs="*", default=[])

    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--overwrite_existing", '-o', action='store_true')

    parser.add_argument("--num_shards", "--ns", type=int, default=1, help="Number of shards to use for the processing")
    parser.add_argument("--shard_id", "--sid", type=int, default=0, help="Shard ID to use for the processing")

    args = parser.parse_args()
    
    # Process kwargs into a dictionary
    kwargs = {}
    for kwarg in args.processor_kwargs:
        if "=" in kwarg:
            key, value = kwarg.split("=", 1)
            # Try to convert value to appropriate type
            try:
                # Try as int
                kwargs[key] = int(value)
            except ValueError:
                try:
                    # Try as float
                    kwargs[key] = float(value)
                except ValueError:
                    # Keep as string if not numeric
                    kwargs[key] = value
    print('processor kwargs:', kwargs)
    dataset = Dataset(args.dataset_path, song_search_index=processor_cls_dict[args.processor].input_props[0])
    processor = processor_cls_dict[args.processor](**kwargs)
    dataset.apply_processor(
        processor,
        num_processes=args.num_processes,
        verbose=args.verbose,
        overwrite_existing=args.overwrite_existing,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )

if __name__ == "__main__":
    main()
