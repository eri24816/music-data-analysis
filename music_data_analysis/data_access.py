import json
from pathlib import Path
from typing import Any

from mido import MidiFile


def get_old_file_path(dataset_path: Path, song_name: str, prop_name: str) -> Path:
    return (dataset_path / prop_name).glob(f"{song_name}.*").__next__()


def get_new_file_path(
    dataset_path: Path, song_name: str, prop_name: str, ext: str, create_dir=True
) -> Path:
    dir_path = dataset_path / prop_name
    if create_dir:
        dir_path.mkdir(exist_ok=True, parents=True)
    return dataset_path / prop_name / f"{song_name}.{ext}"


def read_json(dataset_path: Path, song_name: str, prop_name: str) -> Any:
    prop_file = get_old_file_path(dataset_path, song_name, prop_name)
    return json.loads(prop_file.read_text())


EMPTY = object()


def write_json(
    dataset_path: Path, song_name: str, prop_name: str, data_: Any = EMPTY, **kwargs
):
    if data_ is EMPTY:
        data = kwargs
    else:
        data = data_

    prop_file = get_new_file_path(dataset_path, song_name, prop_name, "json")
    prop_file.write_text(json.dumps(data, ensure_ascii=False))


def read_midi(dataset_path: Path, song_name: str, prop_name: str) -> MidiFile:
    prop_file = get_old_file_path(dataset_path, song_name, prop_name)
    return MidiFile(prop_file)


def write_midi(dataset_path: Path, song_name: str, prop_name: str, midi: MidiFile):
    prop_file = get_new_file_path(dataset_path, song_name, prop_name, "mid")
    midi.save(prop_file)


class Dataset:
    def __init__(self, dataset_path: Path, song_search_index: str = "midi"):
        self.dataset_path = dataset_path
        self.song_search_index = song_search_index

    def songs(self) -> list["Song"]:
        songs = []
        for file in (self.dataset_path / self.song_search_index).glob("*"):
            song_name = file.stem
            songs.append(Song(self, song_name))
        return songs

    def get_song(self, song_name: str):
        return Song(self, song_name)

    def get_old_file_path(self, song_name: str, prop_name: str) -> Path:
        return get_old_file_path(self.dataset_path, song_name, prop_name)

    def get_new_file_path(
        self, song_name: str, prop_name: str, ext: str, create_dir=True
    ):
        return get_new_file_path(
            self.dataset_path, song_name, prop_name, ext, create_dir
        )

    def read_json(self, song_name: str, prop_name: str) -> Any:
        return read_json(self.dataset_path, song_name, prop_name)

    def write_json(self, song_name: str, prop_name: str, data_: Any = EMPTY, **kwargs):
        write_json(self.dataset_path, song_name, prop_name, data_, **kwargs)

    def read_midi(self, song_name: str, prop_name: str) -> MidiFile:
        return read_midi(self.dataset_path, song_name, prop_name)

    def write_midi(self, song_name: str, prop_name: str, midi: MidiFile):
        write_midi(self.dataset_path, song_name, prop_name, midi)


class Song:
    def __init__(self, dataset: Dataset, song_name: str):
        self.dataset = dataset
        self.song_name = song_name

    def get_old_path(self, prop_name: str) -> Path:
        return self.dataset.get_old_file_path(self.song_name, prop_name)

    def get_new_path(self, prop_name: str, ext: str, create_dir=True):
        return self.dataset.get_new_file_path(
            self.song_name, prop_name, ext, create_dir
        )

    def read_json(self, prop_name: str) -> Any:
        return self.dataset.read_json(self.song_name, prop_name)

    def write_json(self, prop_name: str, data_: Any = EMPTY, **kwargs):
        self.dataset.write_json(self.song_name, prop_name, data_, **kwargs)

    def read_midi(self, prop_name: str) -> MidiFile:
        return self.dataset.read_midi(self.song_name, prop_name)

    def write_midi(self, prop_name: str, midi: MidiFile):
        self.dataset.write_midi(self.song_name, prop_name, midi)