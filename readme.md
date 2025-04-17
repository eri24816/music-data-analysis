This package contains a pipeline for analyzing/processing a set of midi files and building a dataset from the analysis.

It also provides a compact interface for accessing such a dataset.


## Analysis pipeline

The analysis pipeline requires a source directory containing MIDI files as input and outputs a dataset directory containing analysis results.

Before running the pipeline, you need to:
1. Install this package (for local installation, `pip install -e .`)
2. Prepare a soundfont file on your disk
3. Prepare the source folder 

Run the pipeline:

```bash
python run.py --path <path_to_dataset_dir> --src <path_to_src_dir> --verbose\
 -sape\
 --soundfont <path_to_soundfont>
```

The pipeline provides 4 actions:

1. `--sync`, `-s`: Sync the notes to the beat
    - Input: `midi`
    - Output: `synced_midi`
2. `--align`, `-a`: Align the MIDI files to C major/A minor
    - Input: `midi`
    - Output: `synced_midi` (same as -s. If -a is used together with -s, their effects will both be applied to `synced_midi`)
3. `--to_pianoroll`, `-p`: Convert the processed MIDI files to pianoroll format
    - Input: `synced_midi`
    - Output: `pianoroll`
4. `--extract_features`, `-e`: Extract features from the pianorolls
    - Input: `pianoroll`
    - Output: many

Combinations of actions:

- `-sape` full pipeline
- `-spe` full pipeline without alignment
- `-sa` outputs synced and aligned midi. No conversion to pianorolls

## Dataset interface

In the dataset directory, each subdirectory contains one attribute of all pieces.

```
dataset_dir/
    midi/
        piece1.mid
        piece2.mid
    pianoroll/
        piece1.npz
        piece2.npz
    key/
        piece1.json
        piece2.json
    density/
        piece1.json
        piece2.json
    ...
```

The `music_data_analysis.Dataset` class provides an interface for accessing attributes of the pieces.

```python
from music_data_analysis import Dataset

dataset = Dataset('path_to_dataset')
songs = dataset.songs() # list[music_data_analysis.Song]

# Get midi of the first song
midi = songs[0].read_midi("synced_midi") # miditoolkit.MidiFile

# Get pianoroll of the first song
pianoroll = songs[0].read_pianoroll("pianoroll") # music_data_analysis.Pianoroll

# Get key of the first song
key = songs[0].read_json("key") # dict

# Get note density of the first song
density = songs[0].read_json("density") # list[float], one value per bar

# Get song by name
song = dataset.get_song("piece1") # music_data_analysis.Song
```

