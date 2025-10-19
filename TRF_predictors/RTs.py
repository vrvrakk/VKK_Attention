# import libraries:
import mne
import numpy as np
import pandas as pd

import os
from pathlib import Path

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def get_response_events():
    responses_dict = {}
    for sub_folder in eeg_dir.iterdir():
        sub_responses = []
        if 'sub' not in sub_folder.name:
            continue
        for eeg_folder in sub_folder.iterdir():
            if 'ica' in eeg_folder.name:
                for files in eeg_folder.iterdir():
                    if condition in files.name:
                        eeg = mne.io.read_raw_fif(files, preload=True)
                        eeg_resamp = eeg.resample(sfreq)
                        eeg_len = eeg_resamp.get_data().shape[-1]
                        response_array = np.zeros(eeg_len)
                        eeg_events, event_ids = mne.events_from_annotations(eeg_resamp)
                        response_value = [value for key, value in event_ids.items() if
                                          key in list(response_nums.keys())]
                        response_events = np.array([event[0] for event in eeg_events if event[2] in response_value])
                        # add ones at response samplepoints, in the response array:
                        for index, value in enumerate(response_array):
                            if index in response_events:
                                response_array[index] = 1
                        sub_responses.append(response_array)
            # only add if not empty
            if sub_responses:
                responses_dict[sub_folder.name] = sub_responses
    return responses_dict


if __name__ == '__main__':

    # define paths:
    base_dir = Path.cwd()
    data_dir = base_dir / 'data' / 'eeg'
    predictor_dir = data_dir / 'predictors'
    bad_segments_dir = predictor_dir / 'bad_segments'
    stream_events_dir = predictor_dir / 'streams_events'
    eeg_dir = Path('D:/VKK_Attention/data/eeg/preprocessed/results')

    stim_type = 'non_targets'
    sfreq = 125

    response_nums = {'Stimulus/S129': 1, 'Stimulus/S130': 2, 'Stimulus/S131': 3, 'Stimulus/S132': 4,
                     'Stimulus/S133': 5, 'Stimulus/S134': 6, 'Stimulus/S136': 8, 'Stimulus/S137': 9}

    conditions = {
        'a1': {'target': 'stream1', 'distractor': 'stream2'},
        'e1': {'target': 'stream1', 'distractor': 'stream2'},
        'a2': {'target': 'stream2', 'distractor': 'stream1'},
        'e2': {'target': 'stream2', 'distractor': 'stream1'}}

    for condition in list(conditions.keys()):
        target_stream = conditions[condition]['target']
        distractor_stream = conditions[condition]['distractor']

        # mask bads:
        bads_dict = {}

        for sub_folders in bad_segments_dir.iterdir():
            bads = []
            for cond_folders in sub_folders.iterdir():
                if condition in cond_folders.name:
                    for files in cond_folders.iterdir():
                        if 'concat.npy.npz' in files.name:
                            bad_array = np.load(files, allow_pickle=True)
                            bad_array = bad_array['bad_series']
                            bads.append(bad_array)
            bads_dict[sub_folders.name] = bads

        # # get response stream events and  concatenate response arrays:
        responses_dict = get_response_events()

        responses = {}
        for sub in responses_dict.keys():
            arrays = responses_dict[sub]
            sub_array = np.concatenate(arrays)
            bad_array = bads_dict[sub][0]
            sub_masked = sub_array[bad_array == 0]
            responses[sub] = sub_masked

        # save RT arrays:
        save_dir = predictor_dir / 'responses' / condition / stim_type
        save_dir.mkdir(parents=True, exist_ok=True)

        for sub, array in responses.items():
            filename = f'{sub}_{condition}_responses_array.npz'
            np.savez(save_dir/filename, responses=array)
            print(f'{sub} responses array for condition {condition} saved'
                  f' as {save_dir/filename}')






