from numpy import mean
import numpy as np

from music_data_analysis.data_access import Song
from music_data_analysis.processor import Processor


def calculate_loss(x:list[int]):
    loss = 0
    for i in x:
        for j in x:
            loss += (i-j)**2
    return loss / len(x)**2

def optimal_two_groups(x:list[int]):
    '''
    Separate x into two groups, minimize the sqare difference between the sum of each group
    '''
    n = len(x)
    x = sorted(x)

    min_loss = np.inf
    best_sep = 0
    for sep in range(n-1):
        left = x[:sep+1]
        right = x[sep+1:]
        loss = calculate_loss(left) + calculate_loss(right)
        if loss < 0:
            loss = -loss
        if loss < min_loss:
            min_loss = loss

            best_sep = sep

    return x[:best_sep+1], x[best_sep+1:]


def reduce(pitches):
    if len(pitches) == 0:
        return None
    return mean(pitches)
    
def fill_out(pitches):
    first_non_nulls = []

    for pitch in pitches:
        if pitch is not None:
            first_non_nulls.append(pitch)
            if len(first_non_nulls) == 2:
                break

    if len(first_non_nulls) == 0:
        latest_pitch = 60
    else:
        latest_pitch = mean(first_non_nulls)

    for i, pitch in enumerate(pitches):
        if pitch is None:
            pitches[i] = latest_pitch
        else:
            latest_pitch = pitch

    return pitches

class PitchProcessor(Processor):
    input_props = ["pianoroll"]
    output_props = ["pitch"]

    def process_impl(self, song: Song):

        pr = song.read_pianoroll('pianoroll')

        pitch_low = []
        pitch_high = []

        for bar in pr.iter_over_bars():
            pitches = [note.pitch for note in bar]
            low, high = optimal_two_groups(pitches)

            pitch_low.append(reduce(low))
            pitch_high.append(reduce(high))

        # fill out the Nones
        pitch_low = fill_out(pitch_low)
        pitch_high = fill_out(pitch_high)

        song.write_json('pitch', {'low': pitch_low, 'high': pitch_high})

