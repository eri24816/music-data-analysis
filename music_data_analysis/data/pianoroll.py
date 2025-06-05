from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
import dataclasses
from io import BytesIO
import itertools
from pathlib import Path
import random
from typing import Any, Generator, Iterator, Literal, Sequence, Tuple, TypeVar, Generic
from matplotlib import pyplot as plt
import numpy as np
from math import ceil
import miditoolkit.midi.parser
import json

INF = 2147483647

HAS_TORCH = False
try:
    import torch

    HAS_TORCH = True
except ImportError:
    pass


def get_origin(type):
    if hasattr(type, "__origin__"):
        return type.__origin__
    return type

def dict_to_dataclass_nested(obj, target_type: Any):
    target_type = get_origin(target_type)
    if dataclasses.is_dataclass(target_type) and isinstance(target_type, type):
        fields = dataclasses.fields(target_type)
        d = {}
        for field in fields:
            d[field.name] = dict_to_dataclass_nested(obj[field.name], field.type)
        return target_type(**d)
    else:
        return obj

T = TypeVar("T")
def json_load(f, dataclass_type: None|type[T]=None) -> T:
    obj = json.load(open(f, "r"))
    if dataclass_type is not None:
        # assert dataclasses.is_dataclass(dataclass_type)
        return dict_to_dataclass_nested(obj, dataclass_type)
    return obj



def json_dump(obj, f):
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        obj = dataclasses.asdict(obj)
    json.dump(obj, open(f, "w"))


class Note:
    def __init__(self, onset: int, pitch: int, velocity: int, offset: int|None=None) -> None:
        self.onset = onset
        self.pitch = pitch
        self.velocity = velocity
        self.offset = offset

    def __repr__(self) -> str:
        return f"Note({self.onset},{self.pitch},{self.velocity},{self.offset})"

    def __gt__(self, other):
        if self.onset == other.onset:
            return self.pitch > other.pitch
        return self.onset > other.onset

    def copy(self):
        return Note(self.onset, self.pitch, self.velocity, self.offset)

    def __eq__(self, other):
        return self.onset == other.onset and self.pitch == other.pitch and self.velocity == other.velocity and self.offset == other.offset

    def __hash__(self):
        return hash((self.onset, self.pitch, self.velocity, self.offset))


@dataclass
class PRMetadata:
    name: str = "(unnamed)"
    start_time: int = 0
    end_time: int = 0

    def copy(self):
        return PRMetadata(self.name, self.start_time, self.end_time)


NotesType = TypeVar("NotesType", bound=Sequence[tuple[int,int,int,int|None]]|torch.Tensor)
@dataclass
class PianorollSerialized(Generic[NotesType]):
    notes: NotesType
    pedal: list[int]|None
    metadata: PRMetadata
    beats_per_bar: int
    frames_per_beat: int
    duration: int


if HAS_TORCH and hasattr(torch.serialization, 'add_safe_globals'):
    torch.serialization.add_safe_globals([PianorollSerialized, PRMetadata])

def clear_duplicate_notes(notes: list[Note]) -> list[Note]:
    # often happens when loading data with larger quantization
    current_onset = -1
    current_pitch = -1

    note_idx_to_remove = []
    for note_idx, note in enumerate(notes):
        if note.onset == current_onset and note.pitch == current_pitch:
            note_idx_to_remove.append(note_idx)
        else:
            current_onset = note.onset
            current_pitch = note.pitch

    # we do it lazily because it rarely happens
    if len(note_idx_to_remove) > 0:
        return [note for note_idx, note in enumerate(notes) if note_idx not in note_idx_to_remove]
    return notes

def sort_notes(notes: list[Note]) -> list[Note]:
    return sorted(notes, key=lambda x: x.onset*1000000 + x.pitch)

class Pianoroll:
    """
    Pianoroll class is a representation of music, which is a sequence of notes. Each note has onset, pitch, velocity, and offset (offset is optional).
    Pianoroll is preferred over midi in case for minimal file size and easy to manipulate.
    """


    @classmethod
    def cat(cls, pianorolls: list["Pianoroll"]) -> "Pianoroll":
        result = pianorolls[0]
        for pr in pianorolls[1:]:
            result |= pr
        return result


    def __init__(self, notes: list[Note], pedal: list[int]|None=None, beats_per_bar: int=4, frames_per_beat: int=8, duration: int|None=None, metadata: PRMetadata|None=None):
        # [onset time, pitch, velocity, (offset time)]

        self.notes = notes
        self.notes = sorted(self.notes)  # ensure the event is sorted by time
        self.beats_per_bar = beats_per_bar
        self.frames_per_beat = frames_per_beat
        self.frames_per_bar = self.beats_per_bar * self.frames_per_beat

        self.pedal = pedal
        if self.pedal is not None:
            self.pedal = sorted(self.pedal)

        # calculate duration
        if duration is not None:
            self.duration = duration
        else:
            if len(self.notes):
                if self.notes[-1].offset is None:
                    self.duration = ceil((self.notes[-1].onset + 1) / self.frames_per_bar) * self.frames_per_bar
                else:
                    self.duration = ceil((self.notes[-1].offset) / self.frames_per_bar) * self.frames_per_bar
            else:
                self.duration = 0

        self._have_offset = len(self.notes) == 0 or self.notes[0].offset is not None

        if metadata is None:
            self.metadata = PRMetadata(name="(unnamed)", start_time=0, end_time=self.duration)
        else:
            self.metadata = metadata

    def __repr__(self) -> str:
        return f"Pianoroll Bar {self.metadata.start_time//self.frames_per_bar:03d} - {ceil(self.metadata.end_time/self.frames_per_bar):03d} of {self.metadata.name}"

    """
    ==================
    Utils
    ==================
    """

    def set_metadata(self, name=None, start_time=None, end_time=None):
        if name is not None:
            self.metadata.name = name
        if start_time is not None:
            self.metadata.start_time = start_time
        if end_time is not None:
            self.metadata.end_time = end_time

    def iter_over_notes_unpack(self, notes=None):
        """
        generator that yields (onset, pitch, velocity, offset iterator)
        """
        if notes is None:
            notes = self.notes
        for note in notes:
            yield note.onset, note.pitch, note.velocity, note.offset

    def iter_over_bars_unpack(
        self, bar_length: int|None=None, relative_time = False
    ) -> Generator[list[Tuple[int, int, int, int]], None, None]:
        """
        generator that yields (onset, pitch, velocity, offset iterator)
        """

        if bar_length is None:
            bar_length = self.frames_per_bar

        iterator = iter(self.notes)
        for bar_start in range(0, self.duration, bar_length):
            list_of_notes = []
            shift = - bar_start if relative_time else 0
            try:
                while True:
                    note = next(iterator)
                    if note.onset >= bar_start + bar_length:
                        # put the note back
                        iterator = itertools.chain([note], iterator)
                        break
                    list_of_notes.append(
                        (note.onset + shift, note.pitch, note.velocity, None if note.offset is None else note.offset + shift)
                    )
            except StopIteration:
                pass
            yield list_of_notes

    def iter_over_bars(self, bar_length: int|None=None, relative_time = False) -> Generator[list[Note], None, None]:
        """
        generator that yields list of notes in each bar
        """
        if bar_length is None:
            bar_length = self.frames_per_bar

        iterator = iter(self.notes)
        for bar_start in range(0, self.duration, bar_length):
            list_of_notes = []
            try:
                while True:
                    note = next(iterator)
                    if note.onset >= bar_start + bar_length:
                        # put the note back
                        iterator = itertools.chain([note], iterator)
                        break
                    if relative_time:
                        note = Note(note.onset - bar_start, note.pitch, note.velocity, note.offset - bar_start)
                    list_of_notes.append(note)
            except StopIteration:
                pass
            yield list_of_notes

    def iter_over_bars_pr(self, bar_length: int|None=None) -> Generator["Pianoroll", None, None]:
        """
        generator that yields Pianoroll of each bar
        """
        if bar_length is None:
            bar_length = self.frames_per_bar

        for notes in self.iter_over_bars(bar_length, relative_time=True):
            pr = Pianoroll(notes, beats_per_bar=self.beats_per_bar, frames_per_beat=self.frames_per_beat, duration = bar_length)
            pr.set_metadata(
                self.metadata.name, self.metadata.start_time, self.metadata.end_time
            )
            yield pr

    def get_offsets_with_pedal(self, pedal, hold_beat_threshold: int = 1, hold_pitch_threshold: int = 12) -> list[int]:
        offsets = []
        next_onset = [INF] * 88
        i = len(pedal)
        def get_pedal_up(i):
            if i >= len(pedal):
                return self.duration
            else:
                return pedal[i]
        for onset, pitch, vel, _ in reversed(
            list(self.iter_over_notes_unpack())
        ):
            pitch -= 21  # midi number to piano
            while i > 0 and pedal[i - 1] > onset:
                i -= 1
            next_pedal_up = get_pedal_up(i)

            # if onset is close to next pedal up, hold the note over the pedal up conditionally
            hold_offset = False
            if next_pedal_up - onset < 1*self.frames_per_beat:
                for pitch_to_check in range(pitch - hold_pitch_threshold, pitch + hold_pitch_threshold):
                    if pitch_to_check < 0 or pitch_to_check >= 88:
                        continue
                    if next_onset[pitch_to_check] - next_pedal_up < hold_beat_threshold*(1-abs(pitch_to_check-pitch)/hold_pitch_threshold)*self.frames_per_beat:
                        break
                else:
                    hold_offset = True

            if hold_offset:
                offset = min(next_onset[pitch],get_pedal_up(i+1))
            else:
                offset = min(next_onset[pitch], next_pedal_up)

            offsets.append(offset)
            next_onset[pitch] = onset
        offsets = list(reversed(offsets))
        return offsets

    '''
    ==================
    Save and load
    ==================
    '''

    def to_dict(self) -> PianorollSerialized[list[tuple[int,int,int,int|None]]]:
        '''
        For json.dump
        '''
        notes = [
            (note.onset, note.pitch, note.velocity, note.offset)
            for note in self.notes
        ]
        return PianorollSerialized(notes, self.pedal, self.metadata, self.beats_per_bar, self.frames_per_beat, self.duration)

    def to_dict_torch(self) -> PianorollSerialized[torch.Tensor]:
        '''
        For torch.save
        '''
        # make notes compact with torch.Tensor [num_notes, 4]
        neg_one_if_none = lambda x: -1 if x is None else x
        notes = torch.tensor(
            [[note.onset, note.pitch, note.velocity, neg_one_if_none(note.offset)] for note in self.notes],
            dtype=torch.int32
        )
        return PianorollSerialized(notes, self.pedal, self.metadata, self.beats_per_bar, self.frames_per_beat, self.duration)


    def save(self, path: Path, format: Literal["json", "torch"] = "json"):
        if format == "json":
            json_dump(self.to_dict(), path)
        elif format == "torch":
            torch.save(self.to_dict_torch(), path)
        else:
            raise ValueError(f"Invalid format: {format}")

    @staticmethod
    def load(path: Path, frames_per_beat: int = 8) -> "Pianoroll":
        """
        Load a pianoroll from a json file
        """
        if path.suffix == ".json":
            pr = Pianoroll.load_json(path, frames_per_beat)
        elif path.suffix == ".pt":
            pr = Pianoroll.load_torch(path, frames_per_beat)
        else:
            raise ValueError(f"Invalid file extension: {path.suffix}")

        # ensure offset > onset
        for note in pr.notes:
            if note.offset is not None and note.offset <= note.onset:
                note.offset = note.onset + 1

        return pr

    @staticmethod
    def load_json(path: Path, frames_per_beat: int|None=None) -> "Pianoroll":
        """
        Load a pianoroll from a json file
        """
        serialized = json_load(path, PianorollSerialized[list[tuple[int,int,int,int|None]]])
        notes = [Note(*note) for note in serialized.notes]
        if frames_per_beat is not None:
            frames_per_beat_scale = frames_per_beat / serialized.frames_per_beat
            for note in notes:
                note.onset = int(round(note.onset * frames_per_beat_scale))
                note.offset = int(round(note.offset * frames_per_beat_scale)) if note.offset is not None else None
        duration = int(ceil(serialized.duration * frames_per_beat_scale))
        notes = sort_notes(notes)
        notes = clear_duplicate_notes(notes)
        return Pianoroll(notes, serialized.pedal, serialized.beats_per_bar, frames_per_beat, duration, serialized.metadata)

    @staticmethod
    def load_torch(path: Path, frames_per_beat: int|None=None) -> "Pianoroll":
        """
        Load a pianoroll from a torch file
        """
        serialized = torch.load(path)
        # convert serialized.notes to list[Note]
        notes = []

        serialized_notes = (
            serialized.notes.cpu().tolist()
        )  # without this, the iteration will be slow
        for onset, pitch, velocity, offset in serialized_notes:
            onset = int(onset)
            pitch = int(pitch)
            velocity = int(velocity)
            if offset == -1:
                offset = None
            else:
                offset = int(offset)
            notes.append(Note(onset, pitch, velocity, offset))

        if frames_per_beat is not None:
            frames_per_beat_scale = frames_per_beat / serialized.frames_per_beat
            for note in notes:
                note.onset = int(round(note.onset * frames_per_beat_scale))
                note.offset = int(round(note.offset * frames_per_beat_scale)) if note.offset is not None else None
            duration = int(ceil(serialized.duration * frames_per_beat_scale))
            metadata = serialized.metadata
            metadata.start_time = int(round(metadata.start_time * frames_per_beat_scale))
            metadata.end_time = int(round(metadata.end_time * frames_per_beat_scale))
        else:
            duration = serialized.duration
            frames_per_beat = serialized.frames_per_beat
            metadata = serialized.metadata

        notes = sort_notes(notes)
        notes = clear_duplicate_notes(notes)


        return Pianoroll(notes, serialized.pedal, serialized.beats_per_bar, frames_per_beat, duration, metadata)

    """
    ==================
    Conversion between pianoroll and tensor
    ==================
    """

    @staticmethod
    def from_tensor(tens: "torch.Tensor", thres=5, normalized=False, binary=False, beats_per_bar: int=4, frames_per_beat: int=8):
        """
        Convert a tensor to a pianoroll
        """
        if not HAS_TORCH:
            raise ImportError(
                "Pianoroll.from_tensor requires torch. Please install torch."
            )
        if normalized and not binary:
            tens = (tens + 1) * 64
        elif binary:
            tens = tens * 100
        tens = tens.cpu().to(torch.int32).clamp(0, 127)

        notes: list[Note] = []
        for t in range(tens.shape[0]):
            for p in range(tens.shape[1]):
                if tens[t, p] > thres:
                    notes.append(Note(t, p + 21, int(tens[t, p])))

        return Pianoroll(notes, beats_per_bar=beats_per_bar, frames_per_beat=frames_per_beat)


    def to_tensor(
        self,
        start_time: int = 0,
        end_time: int = INF,
        padding=False,
        normalized=False,
        binary=False,
        chromagram=False,
    ) -> "torch.Tensor":
        """
        Convert the pianoroll to a tensor
        """
        if not HAS_TORCH:
            raise RuntimeError(
                "Pianoroll.to_tensor requires torch. Please install torch."
            )

        n_features = 88 if not chromagram else 12

        if padding:
            # zero pad to end_time
            assert end_time != INF
            length = end_time - start_time
        else:
            length = min(self.duration, end_time) - start_time

        size = [length, n_features]
        piano_roll = torch.zeros(size)

        for time, pitch, vel, _ in self.iter_over_notes_unpack():
            rel_time = time - start_time
            # only contain notes between start_time and end_time
            if rel_time < 0:
                continue
            if rel_time >= length:
                break
            pitch -= 21  # midi to piano
            if chromagram:
                pitch = (pitch + 9) % 12
            if binary:
                piano_roll[rel_time, pitch] = 1
            else:
                piano_roll[rel_time, pitch] = vel

        if normalized and not binary:
            piano_roll = piano_roll / 64 - 1
        return piano_roll

    '''
    ==================
    Conversion between pianoroll and miditoolkit.MidiFile
    ==================
    '''

    @staticmethod
    def from_midi(path_or_file: Path | BytesIO | miditoolkit.midi.parser.MidiFile, name: str|None=None, beats_per_bar: int=4, frames_per_beat: int=8) -> "Pianoroll":
        """
        Load a pianoroll from a midi file given a path
        """
        if isinstance(path_or_file, Path):
            midi = miditoolkit.midi.parser.MidiFile(path_or_file)
        elif isinstance(path_or_file, BytesIO):
            midi = miditoolkit.midi.parser.MidiFile(file=path_or_file)
        else:
            midi = path_or_file

        notes: list[Note] = []
        miditookit_notes: Iterator[miditoolkit.Note] = itertools.chain(*[i.notes for i in midi.instruments])
        for note in miditookit_notes:

            new_note = Note(
                int(round(note.start * frames_per_beat / midi.ticks_per_beat)),
                note.pitch,
                note.velocity,
                int(round(note.end * frames_per_beat / midi.ticks_per_beat)),
            )
            if new_note.offset == new_note.onset:
                new_note.offset = new_note.onset + 1
            notes.append(
                new_note
            )

        notes = sort_notes(notes)
        notes = clear_duplicate_notes(notes)
        pr = Pianoroll(notes, beats_per_bar=beats_per_bar, frames_per_beat=frames_per_beat)
        if name is None and isinstance(path_or_file, Path):
            name = path_or_file.stem
        pr.set_metadata(name=name)
        return pr

    def to_midi(
        self, path=None, apply_pedal=False, bpm=105, markers: list[tuple[int, str]] = []
    ) -> miditoolkit.MidiFile:
        """
        Convert the pianoroll to a midi file. Returns a miditoolkit.MidiFile object
        If path is specified, the midi file will be saved to the path.
        """
        notes = deepcopy(self.notes)

        if apply_pedal:
            if self.pedal:
                pedal = self.pedal
            else:
                pedal = list(range(0, self.duration, self.frames_per_bar))
            offsets = self.get_offsets_with_pedal(pedal)
            checking_notes_pitch_group: defaultdict[int, list[Note]] = defaultdict(list)
            checking_notes_time_group: defaultdict[int, list[Note]] = defaultdict(list)
            for note in notes:
                checking_notes_pitch_group[note.pitch].append(note)
                checking_notes_time_group[note.onset].append(note)

            for i, note in enumerate(notes):
                note.offset = offsets[i]
        else:
            assert self._have_offset, "Offset not found"
        return self._save_to_midi([notes], path, bpm, markers)

    def _save_to_midi(self, instrs, path, bpm=105, markers: list[tuple[int, str]] = []):
        midi = miditoolkit.MidiFile()
        midi.instruments = [
            miditoolkit.Instrument(program=0, is_drum=False, name=f"Piano{i}")
            for i in range(len(instrs))
        ]
        midi.tempo_changes.append(miditoolkit.TempoChange(bpm, 0))
        for i, notes in enumerate(instrs):
            for onset, pitch, vel, offset in self.iter_over_notes_unpack(notes):
                assert offset is not None, "Offset not found"
                midi.instruments[i].notes.append(
                    miditoolkit.Note(
                        vel,
                        pitch,
                        int(onset * midi.ticks_per_beat / self.frames_per_beat),
                        int(offset * midi.ticks_per_beat / self.frames_per_beat),
                    )
                )
        for time, text in markers:
            midi.markers.append(miditoolkit.Marker(text, int(time * midi.ticks_per_beat / self.frames_per_beat)))
        if path:
            midi.dump(path)
        return midi

    def save_to_pretty_score(
        self,
        path,
        separate_point=60,
        position_weight=3,
        mode="combine",
        make_pretty_voice=True,
    ):
        notes = deepcopy(self.notes)
        # separate left and right hand
        left_hand: list[Note] = []
        right_hand: list[Note] = []

        def loss(
            note,
            prev_notes,
            which_hand,
            max_dist=16,
            separate_point=60,
            position_weight=3,
        ):
            res = 0

            for prev_note in reversed(prev_notes):
                dt = note.onset - prev_note.onset
                dp = note.pitch - prev_note.pitch
                if dt > max_dist:
                    break
                loss = max(0, abs(dp) - 5 - 8 * dt)
                res += loss

            if which_hand == "l":
                res += (note.pitch - separate_point) * position_weight
            elif which_hand == "r":
                res -= (note.pitch - separate_point) * position_weight
            else:
                raise ValueError("which_hand must be 'l' or 'r'")
            return res

        # recursively search for min loss
        def cummulative_loss(
            past_notes_l, past_notes_r, future_notes, max_depth=4, discount_factor=0.9
        ):
            future_notes = future_notes[:max_depth]
            if len(future_notes) == 0:
                return 0, "l"
            else:
                future_loss_l = cummulative_loss(
                    past_notes_l + [future_notes[0]], past_notes_r, future_notes[1:]
                )[0]
                future_loss_r = cummulative_loss(
                    past_notes_l, past_notes_r + [future_notes[0]], future_notes[1:]
                )[0]
                loss_l = future_loss_l * discount_factor + loss(
                    future_notes[0],
                    past_notes_l,
                    "l",
                    16,
                    separate_point,
                    position_weight,
                )
                loss_r = future_loss_r * discount_factor + loss(
                    future_notes[0],
                    past_notes_r,
                    "r",
                    16,
                    separate_point,
                    position_weight,
                )
                if loss_l < loss_r:
                    return loss_l, "l"
                else:
                    return loss_r, "r"

        while len(notes):
            _, hand = cummulative_loss(left_hand, right_hand, notes)
            if hand == "l":
                left_hand.append(notes.pop(0))
            else:
                right_hand.append(notes.pop(0))

        def pretty_voice(voice: list[Note]):
            current = []
            for note in voice:
                if len(current) == 0:
                    current.append(note)
                else:
                    if note.onset == current[-1].onset:
                        current.append(note)
                    else:
                        stop_time = note.onset
                        for c in current:
                            c.offset = stop_time
                        current = [note]

        if make_pretty_voice:
            pretty_voice(left_hand)
            pretty_voice(right_hand)
        res = [right_hand, left_hand]
        print("left hand notes:", len(left_hand))
        print("right hand notes:", len(right_hand))
        if mode == "combine":
            self._save_to_midi(res, path)
        elif mode == "separate":
            self._save_to_midi([left_hand], path + "_left.mid")
            self._save_to_midi([right_hand], path + "_right.mid")

    def to_img(self, path, size_factor: int = 1, annotations: list[tuple[int, str]] = []):
        """
        Convert the pianoroll to a image
        """

        def create_fig_with_size(w:int, h:int, dpi:int = 100):
            fig = plt.figure(figsize=(w/dpi, h/dpi), dpi=dpi)
            ax = plt.Axes(fig, [0, 0, 1, 1])  # use full figure area
            ax.set_axis_off()
            fig.add_axes(ax)
            return fig, ax

        img = np.zeros((88, self.duration))
        for time, pitch, vel, offset in self.iter_over_notes_unpack():
            if time >= self.duration:
                print("Warning: time >= duration") #TODO: fix this
                continue
            img[pitch - 21, time] = vel
        # enlarge the image
        img = np.repeat(img, size_factor, axis=0)
        img = np.repeat(img, size_factor, axis=1)

        # add bar lines
        for t in range(self.duration):
            if t % self.frames_per_bar == 0:
                img[:, t * size_factor] += 20

        # inverse y
        img = np.flip(img, axis=0)

        fig, ax = create_fig_with_size(img.shape[1], img.shape[0])
        ax.imshow(img, vmin=0, vmax=127)
        for t, text in annotations:
            ax.text(t * size_factor+3, 3, text, ha='left', va='top', fontsize=12)
        fig.savefig(path)
        plt.close()

    def to_img_tensor(self, size_factor: int = 1, plot_sustain: bool = False):
        """
        Convert the pianoroll to a image tensor
        """
        img = torch.zeros((88, self.duration))
        if plot_sustain:
            for time, pitch, vel, offset in self.iter_over_notes_unpack():
                img[pitch - 21, time:offset] = vel
        else:
            for time, pitch, vel, offset in self.iter_over_notes_unpack():
                img[pitch - 21, time] = vel
        # enlarge the image
        img = img.repeat_interleave(size_factor, dim=0).repeat_interleave(size_factor, dim=1)

        # add bar lines
        for t in range(self.duration):
            if t % self.frames_per_bar == 0:
                img[:, t * size_factor] += 20

        # inverse y
        img = torch.flip(img, dims=(0,))

        return img

    def show(self):
        """
        Show the pianoroll
        """
        plt.imshow(self.to_img_tensor())
        plt.show()

    """
    ==================
    Basic operations
    ==================
    """

    def __getitem__(self, slice: slice) -> "Pianoroll":
        """
        Slice a pianoroll from start_time to end_time
        """
        assert slice.step is None, "Step is not supported"
        return self.slice(slice.start, slice.stop)

    def slice(self, start_time: int | None = None, end_time: int | None = None, allow_out_of_range: bool = False) -> "Pianoroll":
        """
        Slice a pianoroll from start_time to end_time
        """
        if start_time is None:
            start_time = 0
        if end_time is None:
            end_time = self.duration
        if end_time < start_time:
            raise ValueError("end_time must be greater than start_time")
        if not allow_out_of_range:
            if start_time < 0:
                raise ValueError("start_time must be greater than 0")
            if end_time > self.duration:
                raise ValueError("end_time must be less than or equal to duration")
        length = end_time - start_time
        sliced_notes: list[Note] = []
        for time, pitch, vel, offset in self.iter_over_notes_unpack():
            rel_time = time - start_time
            if rel_time < 0:
                continue
            if rel_time >= length:
                break

            if offset is not None:
                rel_offset = offset - start_time
                rel_offset = min(rel_offset, length)
            else:
                rel_offset = None
            # only contain notes between start_time and end_time
            sliced_notes.append(Note(rel_time, pitch, vel, rel_offset))

        if self.pedal:
            sliced_pedal: list[int] = []
            for pedal in self.pedal:
                time = pedal
                rel_time = time - start_time
                # only contain pedal between start_time and end_time
                if rel_time < 0:
                    continue
                if rel_time >= length:
                    break
                sliced_pedal.append(rel_time)
            new_pr = Pianoroll(
                sliced_notes, sliced_pedal, self.beats_per_bar, self.frames_per_beat, duration = length
            )
        else:
            new_pr = Pianoroll(sliced_notes, self.pedal.copy() if self.pedal else None, self.beats_per_bar, self.frames_per_beat, duration = length)

        new_pr.set_metadata(
            self.metadata.name,
            self.metadata.start_time + start_time,
            self.metadata.start_time + end_time,
        )
        return new_pr

    def random_slice(self, length: int = 128, snap_to_bar: bool = True) -> "Pianoroll":
        """
        Randomly slice a pianoroll with length
        """
        snap = self.frames_per_bar if snap_to_bar else 1
        start_time = random.randint(0, max(0, (self.duration - length) // snap)) * snap
        return self.slice(start_time, start_time + length)

    def get_random_tensor_clip(self, duration, normalized=False, snap_to_bar: bool = True):
        """
        Get a random clip of the pianoroll
        """
        snap = self.frames_per_bar if snap_to_bar else 1
        start_time = (
            random.randint(0, (self.duration - duration) // snap) * snap
        )  # snap to bar
        return self.to_tensor(start_time, start_time + duration, normalized=normalized)

    def get_polyphony(self, granularity: int|None=None):
        """
        Get the polyphony of the pianoroll
        """
        if granularity is None:
            granularity = self.frames_per_bar

        polyphony = []
        for bar in self.iter_over_bars_unpack(granularity):
            to_be_reduced = []
            last_note_frame = 0
            poly = 0
            for frame, pitch, vel, offset in bar:
                if frame > last_note_frame:
                    if poly > 0:
                        to_be_reduced.append(poly)
                    last_note_frame = frame
                    poly = 0
                poly += 1
            if poly > 0:
                to_be_reduced.append(poly)
            max_3 = sorted(to_be_reduced, reverse=True)[:3]
            if len(max_3) == 0:
                polyphony.append(0)
            else:
                polyphony.append(sum(max_3) / len(max_3))



        return polyphony

    def get_density(self, granularity: int|None=None):
        """
        Get the density of the pianoroll
        """
        if granularity is None:
            granularity = self.frames_per_bar

        density = []
        for bar in self.iter_over_bars_unpack(granularity):
            frames = set()
            for frame, pitch, vel, offset in bar:
                frames.add(frame)
            density.append(len(frames))
        return density

    def get_velocity(self, granularity: int|None=None):
        """
        Get the velocity of the pianoroll. Average velocity of each bar
        """
        if granularity is None:
            granularity = self.frames_per_bar

        velocity = []
        for bar in self.iter_over_bars_unpack(granularity):
            vel_sum = 0
            count = 0
            for frame, pitch, vel, offset in bar:
                vel_sum += vel
                count += 1
            if count == 0:
                count = 1
            velocity.append(vel_sum / count)
        return velocity

    def get_highest_pitch(self, granularity: int|None=None):
        """
        Get the highest pitchs of each bar
        """
        if granularity is None:
            granularity = self.frames_per_bar

        highest_pitch = []
        for bar in self.iter_over_bars_unpack(granularity):
            if len(bar) == 0:
                if len(highest_pitch) > 0:
                    highest_pitch.append(highest_pitch[-1])
                else:
                    highest_pitch.append(0)
            else:
                highest_pitch.append(max([note[1] for note in bar]))
        return highest_pitch

    def get_lowest_pitch(self, granularity: int|None=None):
        """
        Get the lowest pitchs of each bar
        """
        if granularity is None:
            granularity = self.frames_per_bar

        lowest_pitch = []
        for bar in self.iter_over_bars_unpack(granularity):
            if len(bar) == 0:
                if len(lowest_pitch) > 0:
                    lowest_pitch.append(lowest_pitch[-1])
                else:
                    lowest_pitch.append(0)
            else:
                lowest_pitch.append(min([note[1] for note in bar]))
        return lowest_pitch

    def copy(self):
        return Pianoroll(deepcopy(self.notes), self.pedal.copy() if self.pedal else None, self.beats_per_bar, self.frames_per_beat, self.duration, self.metadata.copy())

    def __add__(self, other: "Pianoroll"):
        '''
        Overlap two pianorolls
        '''
        assert self.beats_per_bar == other.beats_per_bar, "Beats per bar must be the same when adding two pianorolls"
        assert self.frames_per_beat == other.frames_per_beat, "Frames per beat must be the same when adding two pianorolls"
        notes = self.notes.copy() + other.notes.copy()
        notes = sorted(notes, key=lambda x: x.onset) # sort by onset
        duration = max(self.duration, other.duration)
        return Pianoroll(notes, self.pedal.copy() if self.pedal else None, self.beats_per_bar, self.frames_per_beat, duration)

    def __irshift__(self, frames: int):
        '''
        Shift the pianoroll to the right
        '''
        for note in self.notes:
            note.onset += frames
        if self.pedal:
            for i, pedal in enumerate(self.pedal):
                self.pedal[i] += frames
        self.duration += frames
        return self

    def __rshift__(self, frames: int):
        '''
        Shift the pianoroll to the right
        '''
        new_pr = self.copy()
        new_pr.__irshift__(frames)
        return new_pr

    def __or__(self, b):
        '''
        Concatenate two pianorolls. a | b is equivalent to a + (b >> a.duration)
        '''
        return self + (b >> self.duration)

