from math import floor
import miditoolkit
from ..data_access import Song
from ..processor import Processor


class AlignAndSyncProcessor(Processor):

    def process(self, song: Song):
        if song.exists("synced_midi"):
            return

        midi = song.read_midi("midi")

        beats_info = song.read_json("beats")
        midi = sync_midi(midi, beats_info["beats"], beats_info["start_beat"])

        key = song.read_json("key")["key"]
        midi = align_midi(midi, key)

        song.write_midi("synced_midi", midi)


def time2beat(beat_time, t):
    if t < beat_time[0] - 0.5:
        return None
    if t <= beat_time[0]:
        return 0
    beat_time = list(beat_time) + [beat_time[-1] * 2 - beat_time[-2]]
    for beat, time in enumerate(beat_time):
        if time > t:
            start = beat_time[beat - 1]
            end = time
            res = beat - 1 + (t - start) / (end - start)
            return res


def align_midi(midi: miditoolkit.MidiFile, key: int) -> miditoolkit.MidiFile:
    pitches = []
    for note in midi.instruments[0].notes:
        pitches.append(note.pitch)
    min_pitch, max_pitch = min(pitches), max(pitches)  # in midi number
    min_piano, max_piano = 21, 108  # in midi number

    key_shift = [
        3,
        2,
        1,
        0,
        -1,
        -2,
        -3,
        -4,
        -5,
        -6,
        5,
        4,
        0,
        -1,
        -2,
        -3,
        -4,
        -5,
        -6,
        5,
        4,
        3,
        2,
        1,
    ][key]
    left_margin = (min_pitch + key_shift) - min_piano
    right_margin = max_piano - (max_pitch + key_shift)
    if left_margin < 0 or right_margin < 0:
        if left_margin < right_margin - 12 and key_shift < 0:
            key_shift += 12
        if left_margin - 12 > right_margin and key_shift > 0:
            key_shift -= 12

    result_notes = []
    for note in midi.instruments[0].notes:
        if note.pitch + key_shift < min_piano:
            continue
        if note.pitch + key_shift > max_piano:
            continue
        new_note = miditoolkit.Note(
            velocity=note.velocity,
            pitch=note.pitch + key_shift,
            start=note.start,
            end=note.end,
        )
        result_notes.append(new_note)

    midi.instruments[0].notes = result_notes
    
    return midi

def snap_beats(beats: list, onsets: list[float], time_res: int) -> list[float]:
    """
    make beat track prediction more accurate by using heurictic
    snap beats to the nearest onsets inside one grid size left or right
    """

    if len(beats) <= 1:
        return beats
    if len(onsets) == 0:
        return beats

    beats_dif = []  # The difference between two beats
    for i in range(len(beats) - 1):
        beats_dif.append(beats[i + 1] - beats[i])
    beats_dif.insert(0, beats[0] - 0)
    beats_dif.append(beats[-1])

    result = []
    onset_cursor = 0

    # for debug
    deltas = []

    for i in range(len(beats)):
        candidates = []
        threshold_left = beats[i] - beats_dif[i] / time_res
        threshold_right = beats[i] + beats_dif[i + 1] / time_res

        # collect candidates (onsets inside threshold)
        while onset_cursor < len(onsets):
            if onsets[onset_cursor] < threshold_left:
                onset_cursor += 1
                continue
            if onsets[onset_cursor] > threshold_right:
                break
            candidates.append(onsets[onset_cursor])
            onset_cursor += 1

        if len(candidates) == 0:
            result.append(beats[i])
            continue

        # find the nearest candidate
        nearest = min(candidates, key=lambda x: abs(x - beats[i]))

        deltas.append(beats[i] - nearest)

        result.append(nearest)
    # plt.hist(deltas, bins=20)
    # plt.show()
    return result


def sync_midi(midi: miditoolkit.MidiFile, beats: list, start_beat: int) -> miditoolkit.MidiFile:
    time_res = 8
    notes = []  # onset, pitch, velocity, offset
    discarded_note_count = 0

    beats = [
        beat + 0.035 for beat in beats
    ]  # the beat track seems to be slightly off. Heuristics to fix it
    # snap beats
    onsets: list[float] = []
    for note in midi.instruments[0].notes:
        onsets.append(note.start)

    beats = snap_beats(beats, onsets, time_res)
    sec_per_tick = midi.get_tick_to_time_mapping()[1]
    for note in midi.instruments[0].notes:
        onset_beat = time2beat(beats, note.start * sec_per_tick)
        offset_beat = time2beat(beats, note.end * sec_per_tick)
        if onset_beat is None:
            discarded_note_count += 1
            continue
        if offset_beat is None:
            offset_beat = onset_beat

        onset_tick = round((start_beat % 4 + onset_beat) * time_res)  # quantize
        offset_tick = round((start_beat % 4 + offset_beat) * time_res)

        if offset_tick <= onset_tick:
            offset_tick = onset_tick + 1

        notes.append(
            [onset_tick, note.pitch, note.velocity, offset_tick]
        )

    notes.sort(key=lambda x: (x[0], x[1]))  # sort by onset, pitch

    # remove empty bars at the beginning
    first_note_tick = notes[0][0]
    shift_ticks = floor(first_note_tick / (time_res*4)) * (time_res*4)
    if shift_ticks > 0:
        for note in notes:
            note[0] -= shift_ticks
            note[3] -= shift_ticks

    if discarded_note_count > 0:
        print(f"Warning: discarded {discarded_note_count} notes")

    # create new midi object
    synced_midi = miditoolkit.MidiFile()
    instrument = miditoolkit.Instrument(0)
    synced_midi.instruments.append(instrument)

    for onset, pitch, velocity, offset in notes:
        instrument.notes.append(
            miditoolkit.Note(
                velocity=velocity,
                pitch=pitch,
                start=onset,
                end=offset,
            )
        )

    synced_midi.ticks_per_beat = time_res

    return synced_midi
