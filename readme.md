## Usage

Run the pipeline end to end:

```bash
python run.py --dataset_path <path_to_dataset> --src_midi_dir <path_to_midi_dir>\
 --sync --align --to_pianoroll --extract_features\ # actions
 --soundfont <path_to_soundfont> # required for -s and -a
```

Simplified arguments:

```bash
python run.py --path <path_to_dataset> --src <path_to_midi_dir> --verbose\
 -sape\
 --soundfont <path_to_soundfont>
```

```bash
python run.py --path ignore/debug_ds --src ignore/debug_midi -v\
 -sape\
 --soundfont ignore/FluidR3_GM.sf2
```

Actions:

- `-s` syncs the MIDI files to the beat
- `-a` aligns the MIDI files to C major/A minor
- `-p` converts the MIDI files to pianorolls
- `-e` extracts features from the pianorolls

Combinations of actions:

- `-sape` full pipeline
- `-spe` full pipeline without alignment
- `-sa` outputs synced and aligned midi. No conversion to pianorolls
