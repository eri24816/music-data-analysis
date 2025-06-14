from pathlib import Path
from music_data_analysis.data.pianoroll import Note, Pianoroll
from music_data_analysis.data_access import Song
from music_data_analysis.processor import Processor
from collections import defaultdict
import torch

import numpy as np
from sklearn.cluster import KMeans


def get_num_overlaps2(a: Pianoroll, b: Pianoroll, pitch_shift: int):
    num_overlaps = 0
    i = 0
    j = 0
    a_pitch_to_notes: dict[int, list[Note]] = defaultdict(list)
    for note in a.notes:
        a_pitch_to_notes[note.pitch].append(note)
    b_pitch_to_notes: dict[int, list[Note]] = defaultdict(list)
    for note in b.notes:
        b_pitch_to_notes[note.pitch + pitch_shift].append(note)

    all_pitches = set(a_pitch_to_notes.keys()) | set(b_pitch_to_notes.keys())
    for pitch in all_pitches:
        a_notes = a_pitch_to_notes[pitch]
        b_notes = b_pitch_to_notes[pitch]
        i = 0
        j = 0
        while i < len(a_notes) and j < len(b_notes):
            if abs(a_notes[i].onset - b_notes[j].onset) <= 1:
                num_overlaps += 1
                i += 1
                j += 1
            elif a_notes[i].onset < b_notes[j].onset:
                i += 1
            else:
                j += 1

    denom = max(len(a.notes), len(b.notes))
    if denom == 0:
        return 0
    else:
        return num_overlaps / denom


def get_overlap_sim(a: Pianoroll, b: Pianoroll):
    pitch_shift_search = [0, -12, 12]
    num_overlaps_list = []
    for pitch_shift in pitch_shift_search:
        num_overlaps = get_num_overlaps2(a, b, pitch_shift)
        num_overlaps_list.append(num_overlaps)
    return max(num_overlaps_list)

def get_skyline(pr: Pianoroll, max_slope:float=1, intercept:float=0):
    '''
    max_slope: octaves per beat
    '''
    notes = pr.notes

    max_slope_semitones_per_frame = max_slope * 12 / pr.frames_per_beat

    # filter notes that on top of each frame

    result1 = []
    for i in range(len(notes)-1):
        if notes[i].onset != notes[i+1].onset:
            result1.append(notes[i])
    result1.append(notes[-1])

    result2: list[Note] = []
    last_onset = -2147483648
    last_pitch = 0
    for note in result1:
        if (note.pitch - last_pitch + intercept) / (note.onset - last_onset) >= -max_slope_semitones_per_frame:
            result2.append(note.copy())
            last_onset = note.onset
            last_pitch = note.pitch

    result3: list[Note] = []
    last_onset = 2147483647
    last_pitch = 0
    for note in reversed(result2):
        if (note.pitch - last_pitch + intercept) / (last_onset - note.onset) >= -max_slope_semitones_per_frame:
            result3.append(note.copy())
            last_onset = note.onset
            last_pitch = note.pitch

    result3.reverse()
    
    return Pianoroll(result3, beats_per_bar=pr.beats_per_bar, frames_per_beat=pr.frames_per_beat, duration=pr.duration)
    
class SegmentationProcessor(Processor):
    input_props = ["pianoroll"]
    output_props = ["segmentation"]

    def process_impl(self, song: Song):
        pr = song.read_pianoroll("pianoroll", frames_per_beat=8)
        skyline = get_skyline(pr)
        mat = torch.zeros((pr.duration // 32, pr.duration // 32))
        for i in range(pr.duration // 32):
            for j in range(i, pr.duration // 32):
                sim = (
                    get_overlap_sim(
                        skyline.slice(i * 32, (i + 1) * 32), skyline.slice(j * 32, (j + 1) * 32)
                    )
                    * 0.5
                    + get_overlap_sim(
                        pr.slice(i * 32, (i + 1) * 32), pr.slice(j * 32, (j + 1) * 32)
                    )
                    * 0.5
                )
                mat[i, j] = sim
                mat[j, i] = sim

        ignores = []
        # A is similarity matrix add adjacency matrix so we favor more connected bars
        adj = torch.zeros_like(mat)
        adj.diagonal(0).fill_(1)  # Main diagonal
        adj.diagonal(1).fill_(1)  # Diagonal +1
        adj.diagonal(-1).fill_(1)  # Diagonal -1
        adj_weight = 0.7
        mat_clamped = torch.clamp(mat, min=0.2)
        A = mat_clamped * (1 - adj_weight) + adj_weight * adj
        D = torch.diag(A.sum(dim=1))
        L = D - A

        # Extract the first k eigenvectors of the Laplacian (smallest eigenvalues):
        k = 5
        eigvals, eigvecs = torch.linalg.eig(L)
        eigvecs = eigvecs[:, torch.argsort(eigvals.real)[:k]]

        if len(eigvals) < 16:
            # too short. one segment
            split_points = []
            labels = [0] * len(eigvals)
        else:
            # remove the ignores
            eigvecs = eigvecs[[i not in ignores for i in range(len(eigvecs))], :]

            labels: np.ndarray | list[int] = KMeans(n_clusters=k).fit_predict(
                eigvecs.real
            )

            label_order = []
            seen_labels = set()
            for i in range(len(labels)):
                if labels[i] not in seen_labels:
                    label_order.append(labels[i])
                    seen_labels.add(labels[i])
            labels = [label_order.index(i) for i in labels]
            split_points = []
            for i in range(1, len(labels)):
                if labels[i] != labels[i - 1]:
                    split_points.append(i)

        result = []
        for split, next_split in zip([0] + split_points, split_points + [mat.shape[0]]):
            result.append(
                {"start": split * 32, "end": next_split * 32, "label": labels[split]}
            )
        song.write_json("segmentation", result)


def main():
    from music_data_analysis.data_access import Dataset
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_path", type=Path)
    parser.add_argument("--num_processes", type=int, default=1)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--overwrite_existing", type=bool, default=False)
    args = parser.parse_args()
    dataset = Dataset(args.dataset_path)
    processor = SegmentationProcessor()
    dataset.apply_processor(
        processor,
        num_processes=args.num_processes,
        verbose=args.verbose,
        overwrite_existing=args.overwrite_existing,
    )


if __name__ == "__main__":
    main()
