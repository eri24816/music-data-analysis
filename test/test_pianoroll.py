import tempfile
from music_data_analysis.data.pianoroll import Pianoroll, Note
from pathlib import Path
from typing import Iterator
import pytest
import torch
MIDI_SAMPLES_PATH = Path(__file__).parent / "midi_samples"

def assert_notes_equal_ignoring_offset(notes1: list[Note], notes2: list[Note]):
    for note1, note2 in zip(notes1, notes2):
        assert note1.pitch == note2.pitch
        assert note1.velocity == note2.velocity
        assert note1.onset == note2.onset

def iter_samples(path: Path = MIDI_SAMPLES_PATH, frames_per_beat_list: list[int] = [8,17]) -> Iterator[Pianoroll]:
    for file in path.glob("*.mid"):
        for frames_per_beat in frames_per_beat_list:
            yield Pianoroll.from_midi(file, frames_per_beat=frames_per_beat)

iter_samples_decor = pytest.mark.parametrize("pr",  list(iter_samples()))

@iter_samples_decor
def test_from_to_midi(pr: Pianoroll):
    midi = pr.to_midi(apply_pedal=False)
    pr2 = Pianoroll.from_midi(midi)
    assert pr.notes == pr2.notes


@iter_samples_decor
def test_from_to_tensor(pr: Pianoroll):
    tensor = pr.to_tensor()
    pr2 = Pianoroll.from_tensor(tensor) # this throws away offset information
    assert_notes_equal_ignoring_offset(pr.notes, pr2.notes)

@iter_samples_decor
@pytest.mark.parametrize("start", [0, 25])
@pytest.mark.parametrize("end", [25,56])
def test_slice_tensor(pr: Pianoroll, start: int, end: int):
    tensor1 = pr.to_tensor(start, end)
    tensor2 = pr.to_tensor()[start:end]
    tensor3 = pr.slice(start, end).to_tensor()
    assert torch.allclose(tensor1, tensor2)
    assert torch.allclose(tensor1, tensor3)

@iter_samples_decor
def test_slice_and_concat(pr: Pianoroll):
    a = pr.slice(0, 64)
    b = pr.slice(64, 96)
    c = pr.slice(96)
    reconst = a | b | c # concatenate them
    assert_notes_equal_ignoring_offset(reconst.notes, pr.notes)

@iter_samples_decor
def test_slice_and_concat2(pr: Pianoroll):
    a = pr.slice(0, 51)
    b = pr.slice(51, 52)
    c = pr.slice(52, 128)
    d = pr.slice(128)
    reconst = a | b | c | d # concatenate them
    assert_notes_equal_ignoring_offset(reconst.notes, pr.notes)

@iter_samples_decor
def test_add(pr: Pianoroll):
    a = pr.slice(0, 64)
    b = pr.slice(64, 96) >> 64 # move right by 64
    c = pr.slice(96) >> 96 # move right by 96
    reconst = a + b + c # add them together
    assert_notes_equal_ignoring_offset(reconst.notes, pr.notes)

@iter_samples_decor
def test_add_2(pr: Pianoroll):
    a = pr.slice(0, 67)
    b = pr.slice(67, 68) >> 67 # move right by 67
    c = pr.slice(68) >> 68 # move right by 68
    reconst = a + b + c # add them together
    assert_notes_equal_ignoring_offset(reconst.notes, pr.notes)

@iter_samples_decor
def test_save_and_load_json(pr: Pianoroll):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "test.json"
        pr.save(path, format="json")
        pr2 = Pianoroll.load(path)
        assert pr.notes == pr2.notes
        assert pr.pedal == pr2.pedal
        assert pr.beats_per_bar == pr2.beats_per_bar
        assert pr.frames_per_beat == pr2.frames_per_beat
        assert pr.duration == pr2.duration
        assert pr.metadata == pr2.metadata

@iter_samples_decor
def test_save_and_load_torch(pr: Pianoroll):
    with tempfile.TemporaryDirectory() as temp_dir:
        path = Path(temp_dir) / "test.pt"
        pr.save(path, format="torch")
        pr2 = Pianoroll.load(path)
        assert pr.notes == pr2.notes
        assert pr.pedal == pr2.pedal
        assert pr.beats_per_bar == pr2.beats_per_bar
        assert pr.frames_per_beat == pr2.frames_per_beat
        assert pr.duration == pr2.duration
        assert pr.metadata == pr2.metadata
