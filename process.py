from pathlib import Path
import mido
import subprocess
from beat_this.inference import File2Beats


def run_command(command: str):
    subprocess.run(
        command,
        shell=True,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )


def midi_to_mp3(midi_file: Path, sound_font_file: Path, output_file: Path):
    run_command(f"fluidsynth -F {output_file} {sound_font_file} {midi_file}")


def get_beats(audio_file: Path, file2beats: File2Beats):
    beats, downbeats = file2beats(audio_file)
    for i in range(4):
        if downbeats[0] == beats[i]:
            start_beat = 4 - i
    return list(beats), start_beat


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


def sync_midi(midi: mido.MidiFile, beats: list, start_beat: int) -> mido.MidiFile:
    time_res = 8
    tick_shift = 0.3
    accumulated_time = 0
    onset_events = []  # onset, pitch, velocity, offset
    waiting_for_offset = {}
    discarded_note_count = 0
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
            note = waiting_for_offset.pop(event.note)
            note["offset_time"] = accumulated_time
            onset_beat = time2beat(beats, note["onset_time"])
            offset_beat = time2beat(beats, note["offset_time"])
            if onset_beat == None:
                discarded_note_count += 1
                continue
            if offset_beat == None:
                offset_beat = onset_beat

            onset_beat = round(
                (start_beat + onset_beat) * time_res + tick_shift
            )  # quantize
            offset_beat = round((start_beat + offset_beat) * time_res + tick_shift)

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


from madmom.features import CNNKeyRecognitionProcessor

key_processor = CNNKeyRecognitionProcessor()


def get_key(audio_file: Path):
    return key_processor(audio_file).argmax()


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


def process_song(
    input_midi_file: Path,
    output_midi_file: Path,
    file2beats: File2Beats,
    audio_dir: Path = Path("./audio"),
):
    mp3_file = audio_dir / f"{input_midi_file.stem}.mp3"
    midi_to_mp3(input_midi_file, Path("W:/music/FluidR3_GM/FluidR3_GM.sf2"), mp3_file)

    midi = mido.MidiFile(input_midi_file)

    beats, start_beat = get_beats(mp3_file, file2beats)
    midi = sync_midi(midi, beats, start_beat)

    key = get_key(mp3_file)
    midi = align_midi(midi, key)

    midi.save(output_midi_file)


def mp_init():
    global file2beats
    file2beats = File2Beats(device="cuda", dbn=True)


def mp_task(args):
    global file2beats
    process_song(**args, file2beats=file2beats)


def process_songs_multi_process(input_dir: Path, output_dir: Path):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    args_list = []
    for input_midi_file in list(input_dir.glob("*.mid")) + list(
        input_dir.glob("*.midi")
    ):
        output_midi_file = output_dir / input_midi_file.name
        args_list.append(
            {
                "input_midi_file": input_midi_file,
                "output_midi_file": output_midi_file,
            }
        )
    from multiprocessing import Pool
    from tqdm import tqdm

    with Pool(4, initializer=mp_init) as p:
        for _ in tqdm(p.imap_unordered(mp_task, args_list), total=len(args_list)):
            pass


if __name__ == "__main__":
    process_songs_multi_process(Path("./midi"), Path("./processed_midi"))
