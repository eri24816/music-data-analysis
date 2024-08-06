from ..data_access import Song
from ..processor import Processor
import mido


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


def align_midi(midi: mido.MidiFile, key: int) -> mido.MidiFile:
    pitches = []
    for event in midi:
        if event.type == "note_on" and event.velocity > 0:
            pitches.append(event.note)
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

    new_midi = mido.MidiFile()
    new_midi.ticks_per_beat = midi.ticks_per_beat
    new_midi.tracks.append(mido.MidiTrack())

    for event in midi.tracks[0]:
        if event.type == "note_on" or event.type == "note_off":
            if event.note + key_shift < min_piano:
                continue
            if event.note + key_shift > max_piano:
                continue
            new_event = event
            new_event.note += key_shift
            new_midi.tracks[0].append(new_event)
        else:
            new_midi.tracks[0].append(event)
    return new_midi

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


def sync_midi(midi: mido.MidiFile, beats: list, start_beat: int) -> mido.MidiFile:
    time_res = 8
    accumulated_time = 0
    onset_events = []  # onset, pitch, velocity, offset
    waiting_for_offset = {}
    discarded_note_count = 0

    beats = [
        beat + 0.035 for beat in beats
    ]  # the beat track seems to be slightly off. Heuristics to fix it

    # snap beats
    onsets: list[float] = []
    cumulated_time = 0
    for event in midi:
        cumulated_time += event.time
        if event.type == "note_on" and event.velocity > 0:
            onsets.append(cumulated_time)

    beats = snap_beats(beats, onsets, time_res)

    for event in midi:
        accumulated_time += event.time
        if event.type == "note_on" and event.velocity > 0:
            note = {
                "onset_time": accumulated_time,
                "pitch": event.note,
                "velocity": event.velocity,
            }
            waiting_for_offset[event.note] = note
        elif event.type == "note_on" and event.velocity == 0:
            if event.note not in waiting_for_offset:
                discarded_note_count += 1
                continue
            note = waiting_for_offset.pop(event.note)
            note["offset_time"] = accumulated_time
            onset_beat = time2beat(beats, note["onset_time"])
            offset_beat = time2beat(beats, note["offset_time"])
            if onset_beat == None:
                discarded_note_count += 1
                continue
            if offset_beat == None:
                offset_beat = onset_beat

            onset_beat = round((start_beat % 4 + onset_beat) * time_res)  # quantize
            offset_beat = round((start_beat % 4 + offset_beat) * time_res)

            if offset_beat <= onset_beat:
                offset_beat = onset_beat + 1

            onset_events.append(
                [onset_beat, note["pitch"], note["velocity"], offset_beat]
            )

    if discarded_note_count > 0:
        print(f"Warning: discarded {discarded_note_count} notes in {midi.filename}")
    # sort by time
    onset_events = sorted(onset_events)

    # create new midi object
    synced_midi = mido.MidiFile()
    track = mido.MidiTrack()
    synced_midi.tracks.append(track)
    track.append(mido.Message("program_change", program=0))
    pseudo_midi_events = []
    for onset, pitch, velocity, offset in onset_events:
        onset_time = onset
        offset_time = offset
        pseudo_midi_events.append([onset_time, "note_on", pitch, velocity])
        pseudo_midi_events.append([offset_time, "note_off", pitch, 0])

    pseudo_midi_events = sorted(pseudo_midi_events, key=lambda x: x[0])

    accumulated_time = 0
    for time, type, pitch, velocity in pseudo_midi_events:
        track.append(
            mido.Message(
                type, note=pitch, velocity=velocity, time=time - accumulated_time
            )
        )
        accumulated_time = time

    synced_midi.ticks_per_beat = time_res

    return synced_midi
