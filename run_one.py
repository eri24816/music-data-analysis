
from pathlib import Path
from music_data_analysis.processors.segmentation import SegmentationProcessor
from music_data_analysis.processors.synth import SynthProcessor

processor_cls_dict = {
    "segmentation": SegmentationProcessor,
    "synth": SynthProcessor,
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
    dataset = Dataset(args.dataset_path)
    processor = processor_cls_dict[args.processor](**kwargs)
    dataset.apply_processor(
        processor,
        num_processes=args.num_processes,
        verbose=args.verbose,
        overwrite_existing=args.overwrite_existing,
    )

if __name__ == "__main__":
    main()
