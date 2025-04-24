import json
import hashlib
from pathlib import Path
from typing import TYPE_CHECKING, Any
from functools import lru_cache
from miditoolkit import MidiFile

if TYPE_CHECKING:
    from music_data_analysis.processor import Processor

from .data.pianoroll import Pianoroll


def exists(dataset_path: Path, song_name: str, prop_name: str) -> bool:
    return len(list((dataset_path / prop_name).glob(f"{song_name}.*"))) > 0


def get_old_file_path(dataset_path: Path, song_name: str, prop_name: str) -> Path:
    try:
        # get extension of the first leaf file found
        if "/" in song_name:
            song_path = song_name.rsplit("/", 1)[0]
        else:
            song_path = ""
        ext = next(
            f.suffix
            for f in (dataset_path / prop_name / song_path).glob("*")
            if f.is_file()
        )
        return (dataset_path / prop_name / song_name).with_suffix(f"{ext}")
    except StopIteration:
        raise FileNotFoundError(
            f"File not found for property {prop_name} of song {song_name}"
        )


def get_new_file_path(
    dataset_path: Path,
    song_name: str,
    prop_name: str,
    ext: str | None = None,
    create_dir=True,
) -> Path:
    dir_path = dataset_path / prop_name
    if create_dir:
        dir_path.mkdir(exist_ok=True, parents=True)
    result = dataset_path / prop_name / f"{song_name}"
    if ext is not None:
        result = result.with_suffix(f".{ext}")
    return result


def read_json(dataset_path: Path, song_name: str, prop_name: str) -> Any:
    prop_file = get_old_file_path(dataset_path, song_name, prop_name)
    return json.loads(prop_file.read_text())


UNSET = object()


def write_json(
    dataset_path: Path, song_name: str, prop_name: str, data_: Any = UNSET, **kwargs
):
    if data_ is UNSET:
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
    midi.dump(prop_file)


def read_pianoroll(
    dataset_path: Path,
    song_name: str,
    prop_name: str,
    frames_per_beat: int = 8,
) -> Pianoroll:
    prop_file = get_old_file_path(dataset_path, song_name, prop_name)
    pr = Pianoroll.load(prop_file, frames_per_beat=frames_per_beat)
    pr.metadata.name = song_name
    return pr


def write_pianoroll(
    dataset_path: Path, song_name: str, prop_name: str, pianoroll: Pianoroll
):
    prop_file = get_new_file_path(dataset_path, song_name, prop_name, "json")
    pianoroll.save(prop_file)


def read_pt(dataset_path: Path, song_name: str, prop_name: str) -> Any:
    import torch

    prop_file = get_old_file_path(dataset_path, song_name, prop_name)
    return torch.load(prop_file, map_location=torch.device("cpu"))


def write_pt(dataset_path: Path, song_name: str, prop_name: str, data: Any):
    import torch

    prop_file = get_new_file_path(dataset_path, song_name, prop_name, "pt")
    torch.save(data, prop_file)


def hash_consistent(song_name: str) -> int:
    return int(hashlib.md5(song_name.encode(), usedforsecurity=False).hexdigest(), 16)


def is_in_shard(song_name: str, num_shards: int, shard_id: int) -> bool:
    if num_shards == 1:
        return True
    return hash_consistent(song_name) % num_shards == shard_id


class Dataset:
    def __init__(
        self,
        dataset_path: Path,
        song_search_index: str | None = None,
        delete_when_destruct=False,
    ):
        self.dataset_path = dataset_path
        self.song_search_index = song_search_index
        self.delete_when_destruct = delete_when_destruct

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} not found")

        if (dataset_path / "manifest.json").exists():
            self.manifest = json.load(open(dataset_path / "manifest.json"))
            self.length = self.manifest["num_songs"]
        else:
            self.manifest = None
            if song_search_index is None:
                song_search_index = "midi"
            self.length = len(
                [
                    f
                    for f in list((dataset_path / song_search_index).glob("**/*"))
                    if f.is_file()
                ]
            )

    @lru_cache(maxsize=16)
    def songs(self, num_shards: int = 1, shard_id: int = 0) -> list["Song"]:
        songs = []

        if (
            self.manifest is not None
            and len(self.manifest["properties"]) > 0
            and self.song_search_index is None
        ):
            song_search_index = list(self.manifest["properties"].keys())[0]
        elif self.song_search_index is not None:
            song_search_index = self.song_search_index
        else:
            song_search_index = "midi"

        if (
            self.manifest is not None
            and song_search_index in self.manifest["properties"]
        ):
            for song_name in self.manifest["properties"][song_search_index]:
                songs.append(Song(self, song_name))
        else:
            for file in (self.dataset_path / song_search_index).glob("**/*"):
                if not file.is_file():
                    continue
                song_name = str(
                    file.relative_to(self.dataset_path / song_search_index).with_suffix(
                        ""
                    )
                )
                songs.append(Song(self, song_name))

        songs = [
            song for song in songs if is_in_shard(song.song_name, num_shards, shard_id)
        ]

        try:
            songs.sort(key=lambda song: int(song.song_name))
        except ValueError:
            songs.sort(key=lambda song: song.song_name)
        return songs

    def get_song(self, song_name: str):
        return Song(self, song_name)

    def exists(self, song_name: str, prop_name: str) -> bool:
        return exists(self.dataset_path, song_name, prop_name)

    def get_old_file_path(self, song_name: str, prop_name: str) -> Path:
        return get_old_file_path(self.dataset_path, song_name, prop_name)

    def get_new_file_path(
        self, song_name: str, prop_name: str, ext: str | None = None, create_dir=True
    ):
        return get_new_file_path(
            self.dataset_path, song_name, prop_name, ext, create_dir
        )

    def read_json(self, song_name: str, prop_name: str) -> Any:
        if self.manifest is not None and prop_name in self.manifest["properties"]:
            return self.manifest["properties"][prop_name][song_name]
        return read_json(self.dataset_path, song_name, prop_name)

    def write_json(self, song_name: str, prop_name: str, data_: Any = UNSET, **kwargs):
        write_json(self.dataset_path, song_name, prop_name, data_, **kwargs)

    def read_midi(self, song_name: str, prop_name: str) -> MidiFile:
        return read_midi(self.dataset_path, song_name, prop_name)

    def write_midi(self, song_name: str, prop_name: str, midi: MidiFile):
        write_midi(self.dataset_path, song_name, prop_name, midi)

    def read_pianoroll(
        self, song_name: str, prop_name: str, frames_per_beat: int = 8
    ) -> Pianoroll:
        return read_pianoroll(self.dataset_path, song_name, prop_name, frames_per_beat)

    def write_pianoroll(self, song_name: str, prop_name: str, pianoroll: Pianoroll):
        write_pianoroll(self.dataset_path, song_name, prop_name, pianoroll)

    def read_pt(self, song_name: str, prop_name: str) -> Any:
        return read_pt(self.dataset_path, song_name, prop_name)

    def write_pt(self, song_name: str, prop_name: str, data: Any):
        write_pt(self.dataset_path, song_name, prop_name, data)

    def __len__(self):
        return self.length
    
    def apply_processor(self, processor: 'Processor', num_processes: int = 1, verbose=True, num_shards: int = 1, shard_id: int = 0, overwrite_existing: bool = False):
        '''
        Apply a `Processor` to each song in the `Dataset`. Single process by default.
        '''
        from music_data_analysis.apply import apply_to_dataset
        apply_to_dataset(self, processor, num_processes=num_processes, verbose=verbose, num_shards=num_shards, shard_id=shard_id, overwrite_existing=overwrite_existing)
        

class Song:
    def __init__(self, dataset: Dataset, song_name: str):
        self.dataset = dataset
        self.song_name = song_name

    def get_segment(self, start: int, end: int, frames_per_beat: int = 8):
        return SongSegment(self, start, end, frames_per_beat)

    def exists(self, prop_name: str) -> bool:
        return self.dataset.exists(self.song_name, prop_name)

    def get_old_path(self, prop_name: str) -> Path:
        return self.dataset.get_old_file_path(self.song_name, prop_name)

    def get_new_path(self, prop_name: str, ext: str | None = None, create_dir=True):
        return self.dataset.get_new_file_path(
            self.song_name, prop_name, ext, create_dir
        )

    def read_json(self, prop_name: str) -> Any:
        return self.dataset.read_json(self.song_name, prop_name)

    def write_json(self, prop_name: str, data_: Any = UNSET, **kwargs):
        self.dataset.write_json(self.song_name, prop_name, data_, **kwargs)

    def read_midi(self, prop_name: str) -> MidiFile:
        return self.dataset.read_midi(self.song_name, prop_name)

    def write_midi(self, prop_name: str, midi: MidiFile):
        self.dataset.write_midi(self.song_name, prop_name, midi)

    def read_pianoroll(self, prop_name: str, frames_per_beat: int = 8) -> Pianoroll:
        return self.dataset.read_pianoroll(self.song_name, prop_name, frames_per_beat)

    def write_pianoroll(self, prop_name: str, pianoroll: Pianoroll):
        self.dataset.write_pianoroll(self.song_name, prop_name, pianoroll)


    def read_pt(self, prop_name: str) -> Any:
        return self.dataset.read_pt(self.song_name, prop_name)

    def write_pt(self, prop_name: str, data: Any):
        self.dataset.write_pt(self.song_name, prop_name, data)


class SongSegment:
    # TODO: unhardcode this
    _hardcoded_granularity_in_beats = {
        "chords": 1,
        "density": 4,
        "pitch": 4,
        "polyphony": 4,
        "velocity": 4,
    }

    _hardcoded_pad_value = {
        "chords": "None",
        "density": 0,
        "pitch": 0,
        "polyphony": 0,
        "velocity": 0,
    }

    def __init__(self, song: Song, start: int, end: int, frames_per_beat: int = 8):
        self.song = song
        self.start = start
        self.end = end
        self.frames_per_beat = frames_per_beat

    def __len__(self):
        return self.end - self.start

    def exists(self, prop_name: str) -> bool:
        return self.song.exists(prop_name)

    def get_old_path(self, prop_name: str) -> Path:
        return self.song.get_old_path(prop_name)

    def get_new_path(self, prop_name: str, ext: str | None = None, create_dir=True):
        return self.song.get_new_path(prop_name, ext, create_dir)

    def read_json(
        self,
        prop_name: str,
        granularity: int | None = None,
        pad_value: Any = UNSET,
    )-> Any:
        if pad_value is UNSET:
            if prop_name in self._hardcoded_pad_value:
                pad_value = self._hardcoded_pad_value[prop_name]
            else:
                raise ValueError(
                    f"Please provide pad_value for this property: {prop_name}"
                )

        if granularity is None:
            granularity = (
                self._hardcoded_granularity_in_beats[prop_name] * self.frames_per_beat
            )

        def get_prop_slice(prop_list: list[Any]) -> list[Any]:
            start_idx = self.start // granularity
            end_idx = self.end // granularity

            left_pad = 0
            right_pad = 0

            if start_idx < 0:
                left_pad = -start_idx
                start_idx = 0

            if end_idx > len(prop_list):
                right_pad = end_idx - len(prop_list)
                end_idx = len(prop_list)

            return (
                [pad_value] * left_pad
                + prop_list[start_idx:end_idx]
                + [pad_value] * right_pad
            )

        j = self.song.read_json(prop_name)
        if isinstance(j, list):
            result = get_prop_slice(j)
            return result
        else:
            assert isinstance(j, dict)
            result = {}
            for k, v in j.items():
                result[k] = get_prop_slice(v)
            return result

    def read_pianoroll(self, prop_name: str, frames_per_beat: int = 8) -> Pianoroll:
        return self.song.read_pianoroll(prop_name, frames_per_beat).slice(
            self.start, self.end
        )

    def read_pt(self, prop_name: str, granularity: int = 32) -> Any:
        return self.song.read_pt(prop_name)[
            self.start // granularity : self.end // granularity
        ]