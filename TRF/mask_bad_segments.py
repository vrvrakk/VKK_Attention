import os
from pathlib import Path
import mne
import numpy as np
import pandas as pd

from TRF.overlap_ratios import load_eeg_files
from TRF.predictors_run import sub, condition, sfreq, stim_dur, results_path, predictors_path


def extract_bad_segments():
    all_bad_segments = []
    for eeg_files in eeg_files_list:
        bad_segments_block = []
        for description, onset, duration in zip(eeg_files.annotations.description,
                                                eeg_files.annotations.onset,
                                                eeg_files.annotations.duration):
            if description.lower().startswith('bad'):
                offset = onset + duration
                bad_segments_block.append((onset, offset))
        all_bad_segments.append(bad_segments_block)
    return all_bad_segments

def set_bad_series(all_bad_segments):
    bad_series_all = []
    for eeg_len, blocks in zip(eeg_lens, all_bad_segments):
        bad_series = np.zeros(eeg_len)
        for block in blocks:
            onset_samples = int(np.round(block[0] * sfreq))
            offset_samples = int(np.round(block[1] * sfreq))
            bad_series[onset_samples:offset_samples] = -999
        bad_series_all.append(bad_series)
    return bad_series_all


def save_bad_series(bad_series_all, bad_series_concat):
    save_path = predictors_path / 'bad_segments' / sub / condition
    save_path.mkdir(parents=True, exist_ok=True)
    for i, series in enumerate(bad_series_all):
        np.savez(save_path/f'{sub}_{condition}_{i}_bad_series.npz',
                 bad_series=series,
                 sfreq=sfreq,
                 stim_duration_samples=int(stim_dur * sfreq),
                 stream_label=f'bad_series_{i}'
                 )
    # save concatenated series:
    np.savez(save_path/f'{sub}_{condition}_bad_series_concat.npy',
             bad_series=bad_series_concat,
             sfreq=sfreq,
             stim_duration_samples=int(stim_dur * sfreq),
             stream_label='bad_series_concat')


if __name__ == '__main__':
    eeg_files_list, eeg_events_list = load_eeg_files(sub=sub, condition=condition, sfreq=sfreq, results_path=results_path)
    eeg_concat = mne.concatenate_raws(eeg_files_list)

    # set masking series:
    eeg_lens = [eeg_files.n_times for eeg_files in eeg_files_list]
    all_bad_segments = extract_bad_segments()
    bad_series_all = set_bad_series(all_bad_segments)
    bad_series_concat = np.concatenate(bad_series_all)

    save_bad_series(bad_series_all, bad_series_concat)



