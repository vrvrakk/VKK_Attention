import os
from pathlib import Path
import mne
from mtrf import TRF
from TRF_test.TRF_test_config import temporal_roi, frontal_roi, occipitoparietal_right, occipitoparietal_left
import pandas as pd
import numpy
import os
import random
import pandas
import numpy as np
from mtrf.model import TRF
from mtrf.stats import pearsonr
import pickle
import json
import mne
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from scipy.signal import welch
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from mtrf.stats import pearsonr, crossval


def get_eeg_files(condition=''):
    eeg_files = {}
    for folders in eeg_results_path.iterdir():
        if 'sub' in folders.name:
            sub_data = []
            for files in folders.iterdir():
                if 'ica' in files.name:
                    for data in files.iterdir():
                        if condition in data.name:
                            eeg = mne.io.read_raw_fif(data, preload=True)
                            eeg.set_eeg_reference('average')
                            eeg.resample(sfreq=sfreq)
                            sub_data.append(eeg)
            eeg_files[folders.name] = sub_data
    return eeg_files


def pick_channels(eeg_files):
    eeg_concat_list = {}

    for sub, sub_list in eeg_files.items():
        if len(sub_list) > 0:
            eeg_concat = mne.concatenate_raws(sub_list)
            eeg_concat.resample(sfreq)
            eeg_concat.pick(frontal_roi)
            eeg_concat.filter(l_freq=None, h_freq=30)
            eeg_concat_list[sub] = eeg_concat
    return eeg_concat_list


def mask_bad_segmets(eeg_concat_list, condition):
    eeg_clean_list = {}
    for sub in eeg_concat_list:
        eeg_concat = eeg_concat_list[sub]
        eeg_data = eeg_concat.get_data()

        sub_bad_segments_path = predictors_path / 'bad_segments' / sub / condition

        if sub_bad_segments_path.exists():
            for file in sub_bad_segments_path.iterdir():
                if 'concat.npy' in file.name:
                    bad_segments = np.load(file)
                    bad_series = bad_segments['bad_series']
                    good_samples = bad_series == 0  # Boolean mask
                    print(f"Loaded bad segments for {sub} {condition}.")
                    eeg_clean = eeg_data[:, good_samples]
                    # z-scoring..
                    eeg_clean = (eeg_clean - eeg_clean.mean(axis=1, keepdims=True)) / eeg_clean.std(axis=1, keepdims=True)
                    print(f"{sub}: Clean EEG shape = {eeg_clean.shape}")
                    eeg_clean_list[sub] = eeg_clean
                    break
        else:
            print(f"No bad segments found for {sub} {condition}, assuming all samples are good.")
            eeg_len = eeg_concat.n_times
            good_samples = np.ones(eeg_len, dtype=bool)
            eeg_clean = eeg_data[:, good_samples]
            # z-scoring..
            eeg_clean = (eeg_clean - eeg_clean.mean(axis=1, keepdims=True)) / eeg_clean.std(axis=1, keepdims=True)
            print(f"{sub}: Clean EEG shape = {eeg_clean.shape}")
            eeg_clean_list[sub] = eeg_clean
    return eeg_clean_list


def centering_predictor_array(predictor_array, min_std=1e-6, pred_type=''):
    """
    Normalize predictor arrays for TRF modeling.

    Rules:
    - 'envelopes' → z-score if dense and std > min_std.
    - All other predictors → left unchanged.
    - Fully zero predictors → returned unchanged.
    """

    if pred_type == 'envelopes':
        std = predictor_array.std()
        nonzero_ratio = np.count_nonzero(predictor_array) / len(predictor_array)

        if nonzero_ratio > 0.5:
            print(f'{pred_type}: Dense predictor, applying z-score.')
            mean = predictor_array.mean()
            return (predictor_array - mean) / std if std > min_std else predictor_array - mean

        elif nonzero_ratio > 0:
            print(f'{pred_type}: Sparse predictor, mean-centering non-zero entries.')
            mask = predictor_array != 0
            mean = predictor_array[mask].mean()
            predictor_array = predictor_array.copy()
            predictor_array[mask] -= mean
            return predictor_array

        else:
            print(f'{pred_type}: All zeros, returning unchanged.')
            return predictor_array

    else:
        print(f'{pred_type}: No normalization applied.')
        return predictor_array

def get_predictor_dict(condition='', pred_type=''):
    predictor_dict = {}
    for files in predictor.iterdir():
        if 'sub' in files.name:
            sub_name = files.name  # e.g., "sub01"
            stream1_data, stream2_data = None, None

            for file in files.iterdir():
                if condition in file.name:
                    for stim_type in file.iterdir():
                        if stim_type.name == stream_type1:
                            for array in stim_type.iterdir():
                                if 'concat' in array.name:
                                    stream1_data = np.load(array)
                                    stream1_data = stream1_data[f'{pred_type}']
                        elif stim_type.name == stream_type2:
                            for array in stim_type.iterdir():
                                if 'concat' in array.name:
                                    stream2_data = np.load(array)
                                    stream2_data = stream2_data[f'{pred_type}']

            if stream1_data is not None and stream2_data is not None:
                predictor_dict[sub_name] = {
                    'stream1': stream1_data,
                    'stream2': stream2_data
                }
                print(f"Loaded predictors for {sub_name}: {stream1_data.shape}, {stream2_data.shape}")
            else:
                print(f"Missing predictor(s) for {sub_name} {condition}")
    return predictor_dict


def predictor_mask_bads(predictor_dict, condition, pred_type=''):
    predictor_dict_masked = {}
    predictor_dict_masked_only = {}

    for sub, sub_dict in predictor_dict.items():
        sub_bad_segments_path = predictors_path / 'bad_segments' / sub / condition
        good_samples = None  # default fallback

        # --- Load bad segment mask if available
        if sub_bad_segments_path.exists():
            for file in sub_bad_segments_path.iterdir():
                if 'concat.npy' in file.name:
                    bad_segments = np.load(file)
                    bad_series = bad_segments['bad_series']
                    good_samples = bad_series == 0  # Boolean mask
                    print(f"Loaded bad segments for {sub} - {condition}.")
                    break  # Stop after finding the correct file

        sub_masked = {}
        sub_maked_only = {}

        for stream_name, stream_array in sub_dict.items():
            if good_samples is not None and len(good_samples) == len(stream_array):
                stream_array_masked = stream_array[good_samples]
                stream_array_clean = centering_predictor_array(stream_array_masked, min_std=1e-6, pred_type=pred_type)
            else:
                stream_array_masked = stream_array[good_samples]
                stream_array_clean = centering_predictor_array(stream_array_masked, min_std=1e-6, pred_type=pred_type)
            sub_masked[stream_name] = stream_array_clean
            sub_maked_only[stream_name] = stream_array_masked

        predictor_dict_masked[sub] = sub_masked
        predictor_dict_masked_only[sub] = sub_maked_only
    return predictor_dict_masked, predictor_dict_masked_only


# concat all subs eegs together in order,
# same for envelopes stream1 and stream2 respectively.
# then vstack stream1 and stream2
# Clean EEG
def arrays_lists(eeg_clean_list_masked, predictor_dict_masked, s1_key='', s2_key=''):
    all_eeg_clean = []
    all_stream1 = []
    all_stream2 = []

    for sub in eeg_clean_list_masked.keys():
        eeg = eeg_clean_list_masked[sub]
        env = predictor_dict_masked[sub]
        stream1 = env[f'{s1_key}']
        stream2 = env[f'{s2_key}']

        all_eeg_clean.append(eeg)
        all_stream1.append(stream1)
        all_stream2.append(stream2)
    return all_eeg_clean, all_stream1, all_stream2



if __name__ == '__main__':

    default_path = Path.cwd()
    predictors_path = default_path / 'data/eeg/predictors'
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'
    sfreq = 125

    eeg_files1 = get_eeg_files(condition='e1')
    eeg_files2 = get_eeg_files(condition='e2')
    plane = 'elevation'

    eeg_concat_list1 = pick_channels(eeg_files1)
    eeg_concat_list2 = pick_channels(eeg_files2)

    eeg_clean_list_masked1 = mask_bad_segmets(eeg_concat_list1, condition='e1')
    eeg_clean_list_masked2 = mask_bad_segmets(eeg_concat_list2, condition='e2')

    n = 6

    predictors_list = ['binary_weights', 'envelopes', 'overlap_ratios', 'events_proximity', 'events_proximity', 'RTs', 'RTs']
    predictor_name = predictors_list[n]
    pred_types = ['onsets', 'envelopes', 'overlap_ratios', 'events_proximity_pre', 'events_proximity_post', 'RT_labels', 'RTs']
    pred_type = pred_types[n]
    predictor = default_path/f'data/eeg/predictors/{predictor_name}'
    stream_type1 = 'stream1'
    stream_type2 = 'stream2'

    predictor_dict1 = get_predictor_dict(condition='e1', pred_type=pred_type)
    predictor_dict2 = get_predictor_dict(condition='e2', pred_type=pred_type)

    predictor_dict_masked1, predictor_dict_masked_only1 = predictor_mask_bads(predictor_dict1, condition='e1', pred_type=pred_type)
    predictor_dict_masked2, predictor_dict_masked_only2 = predictor_mask_bads(predictor_dict2, condition='e2', pred_type=pred_type)

    stim1 = 'target_stream'
    stim2 = 'distractor_stream'
    for sub, sub_dict in predictor_dict_masked1.items():
        sub_dict[f'{stim1}'] = sub_dict.pop('stream1')  # pop to replace OG array, not add extra array with new key
        sub_dict[f'{stim2}'] = sub_dict.pop('stream2')

    for sub, sub_dict in predictor_dict_masked2.items():
        sub_dict[f'{stim2}'] = sub_dict.pop('stream1')
        sub_dict[f'{stim1}'] = sub_dict.pop('stream2')

    if n == 6:

        RTs_target = {}
        RTs_mean_target = {}
        for sub, sub_dict in predictor_dict_masked1.items():
            target_dict = sub_dict[stim1]
            response_rts = []
            for index, rt_values in enumerate(range(len(target_dict) - 1)):
                if index == 0:
                    prev_val = 0
                else:
                    prev_val = target_dict[index - 1]
                response = False
                current_val = target_dict[index]
                if prev_val == 0 and current_val != 0:
                    response = True
                    response_rts.append(current_val)
                    response_mean = np.mean(response_rts)
            RTs_target[sub] = response_rts
            RTs_mean_target[sub] = response_mean

        title = f'Mean RTs {stim1} - {plane}'
        y = list(RTs_mean_target.values())
        x = list(range(1, len(y) + 1))  # 1-based subject indices

        plt.figure(figsize=(12, 8))
        plt.title(title)
        plt.bar(x, y)
        plt.xlabel('Subjects')
        plt.ylabel('Average RTs (s)')
        rt_path = default_path / 'data/performance/aggregated_results/RTs/figures'
        rt_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(rt_path / f'{title}.png', dpi=300)





