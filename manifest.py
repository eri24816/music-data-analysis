# generate manifest.json for a dataset

import json
from pathlib import Path
from tqdm import tqdm

from music_data_analysis import Dataset

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=Path, help="Path to the dataset directory")
    parser.add_argument("--props", type=str, nargs="+", default=[], help="properties to include in the manifest. Only json formatted properties are supported.")
    args = parser.parse_args()

    ds = Dataset(args.dataset_path, args.props[0] if len(args.props) > 0 else 'midi')

    manifest = {
        "num_songs": len(ds),
        "properties": {}
    }

    for prop in args.props:
        print(f"Processing property: {prop}")
        manifest["properties"][prop] = {}
        for song in tqdm(ds.songs(), desc=f"Processing property: {prop}"):
            manifest["properties"][prop][song.song_name] = song.read_json(prop)

    with open(args.dataset_path / "manifest.json", "w") as f:
        json.dump(manifest, f)

    print(f"Manifest saved to {args.dataset_path / 'manifest.json'}")