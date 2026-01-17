import os
from pathlib import Path
import mne
import numpy as np
import pandas as pd

from EEG.extract_events import load_eeg_files, extract_eeg_files
from TRF_predictors.config import sub, sfreq, stim_dur, predictors_path


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
    sub_list = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19', 'sub20',
                'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29']
    condition = 'a2'
    eeg_header_files = extract_eeg_files(condition=condition)
    eeg_files_list = load_eeg_files(eeg_header_files)
    # Capture individual lengths BEFORE concatenation
    eeg_sesamp = [eeg_files.resample(sfreq=500) for eeg_files in eeg_files_list]
    eeg_lens = [eeg_files.n_times for eeg_files in eeg_files_list]
    eeg_concat = mne.concatenate_raws(eeg_files_list)
    # set masking series
    all_bad_segments = extract_bad_segments()
    bad_series_all = set_bad_series(all_bad_segments)
    bad_series_concat = np.concatenate(bad_series_all)
    # get len of samplepoints in bad_series_all that are != 0 (so noisy)
    percentages = []
    for idx, eeg_len in enumerate(eeg_lens):
        sub_bads = bad_series_all[idx]
        eeg_len = eeg_len
        bads_len = len(sub_bads[sub_bads != 0])
        if bads_len > 0:
            percentage_bads = (bads_len * 100) / eeg_len
        else:
            percentage_bads = 0
        percentages.append(percentage_bads)
    if len(percentages) > 0:
        max_perc_per_cond = np.max(percentages)
        print(f'Condition {condition} max. bad segments % removed: {max_perc_per_cond}')
    else:
        print(f'Condition {condition} max. bad segments % removed: 0 %')
    # save bad segments masking arrays
    save_bad_series(bad_series_all, bad_series_concat)



