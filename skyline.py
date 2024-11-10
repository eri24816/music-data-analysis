import os
import copy
import numpy as np

import miditoolkit 
from miditoolkit.midi.containers import Instrument



num2pitch = {
    0: 'C',
    1: 'C#',
    2: 'D',
    3: 'D#',
    4: 'E',
    5: 'F',
    6: 'F#',
    7: 'G',
    8: 'G#',
    9: 'A',
    10: 'A#',
    11: 'B',
}

def traverse_dir(
        root_dir,
        extension=('mid', 'MID', 'midi'),
        amount=None,
        str_=None,
        is_pure=False,
        verbose=False,
        is_sort=False,
        is_ext=True):
    if verbose:
        print('[*] Scanning...')
    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.endswith(extension):
                if (amount is not None) and (cnt == amount):
                    break
                if str_ is not None:
                    if str_ not in file:
                        continue
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                if verbose:
                    print(pure_path)
                file_list.append(pure_path)
                cnt += 1
    if verbose:
        print('Total: %d files' % len(file_list))
        print('Done!!!')
    if is_sort:
        file_list.sort()
    return file_list


def quantize_melody(notes, tick_resol=240):
    melody_notes = []
    for note in notes:
        # cut too long notes
        if note.end - note.start > tick_resol * 8:
            note.end = note.start + tick_resol * 4

        # quantize
        note.start = int(np.round(note.start / tick_resol) * tick_resol)
        note.end = int(np.round(note.end / tick_resol) * tick_resol)

        # append
        melody_notes.append(note) 
    return melody_notes


def extract_melody(notes):
    # quantize
    #melody_notes = quantize_melody(notes)
    melody_notes = notes

    # sort by start, pitch from high to low
    melody_notes.sort(key=lambda x: (x.start, -x.pitch))
    
    # exclude notes < 60 
    bins = []
    prev = None
    tmp_list = []
    for nidx in range(len(melody_notes)):
        note = melody_notes[nidx]
        if note.pitch >= 60:
            if note.start != prev:
                if tmp_list:
                    bins.append(tmp_list)
                tmp_list = [note]
            else:
                tmp_list.append(note)
            prev = note.start  
    
    # preserve only highest one at each step
    notes_out = []
    for b in bins:
        notes_out.append(b[0])

    # avoid overlapping
    notes_out.sort(key=lambda x:x.start)
    for idx in range(len(notes_out) - 1):
        if notes_out[idx].end >= notes_out[idx+1].start:
            notes_out[idx].end = notes_out[idx+1].start
    
    # delete note having no duration
    notes_clean = []
    for note in notes_out:
        if note.start != note.end:
            notes_clean.append(note)  

    # filtered by interval
    notes_final = [notes_clean[0]]
    for i in range(1, len(notes_clean) -1):
        if ((notes_clean[i].pitch - notes_clean[i-1].pitch) <= -9) and \
        ((notes_clean[i].pitch - notes_clean[i+1].pitch) <= -9):
            continue
        else:
            notes_final.append(notes_clean[i])
    notes_final += [notes_clean[-1]]
    return notes_final


def proc_one(path_infile, path_outfile):
    # load
    midi_obj = miditoolkit.midi.parser.MidiFile(path_infile)
    midi_obj_out = copy.deepcopy(midi_obj)
    midi_obj_out.instruments = []
    notes = midi_obj.instruments[0].notes
    notes = sorted(notes, key=lambda x: (x.start, x.pitch))

    # --- melody --- #
    melody_notes = extract_melody(notes)
    #melody_notes = quantize_melody(melody_notes)
    
    # === save === #
    # mkdir
    fn = os.path.basename(path_outfile)
    os.makedirs(path_outfile[:-len(fn)], exist_ok=True)

    # save piano (0) and melody (1)
    melody_track = Instrument(program=0, is_drum=False, name='melody')
    melody_track.notes = melody_notes
    midi_obj_out.instruments.append(melody_track)

    # save
    midi_obj_out.instruments[0].name = 'piano'
    midi_obj_out.dump(path_outfile)

if __name__ == '__main__':
    # paths
    path_indir = './midi_synchronized'
    path_outdir = './midi_analyzed'
    os.makedirs(path_outdir, exist_ok=True)

    # list files
    midifiles = traverse_dir(
        path_indir,
        is_pure=True,
        is_sort=True)
    n_files = len(midifiles)
    print('num fiels:', n_files)

    # run
    for fidx in range(n_files): 
        path_midi = midifiles[fidx]
        print('{}/{}'.format(fidx, n_files))

        # paths
        path_infile = os.path.join(path_indir, path_midi)
        path_outfile = os.path.join(path_outdir, path_midi)
        
        # proc
        proc_one(path_infile, path_outfile)