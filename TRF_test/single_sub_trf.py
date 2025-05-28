from pathlib import Path
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import mtrf
from mtrf import TRF
from mtrf.stats import crossval
from TRF_predictors.overlap_ratios import load_eeg_files
from scipy.signal import welch
import copy
from TRF_test.TRF_test_config import frontal_roi
from joblib import Parallel, delayed
from tqdm import tqdm
import psutil
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression

''' A script to get optimal lambda per condition (azimuth vs elevation) - with stacked predictors - all 8'''


def get_eeg_files(condition=''):
    eeg_files = {}
    for folders in eeg_results_path.iterdir():
        if 'sub' in folders.name:
            if condition in ['a1', 'a2'] and stream_type2 == 'deviants' and folders.name == 'sub01':
                continue
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
    eeg_masked_list = {}
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
                    eeg_masked = eeg_data[:, good_samples]
                    # z-scoring..
                    eeg_clean = (eeg_masked - eeg_masked.mean(axis=1, keepdims=True)) / eeg_masked.std(axis=1, keepdims=True)
                    print(f"{sub}: Clean EEG shape = {eeg_clean.shape}")
                    eeg_clean_list[sub] = eeg_clean
                    eeg_masked_list[sub] = eeg_masked
                    break
        else:
            print(f"No bad segments found for {sub} {condition}, assuming all samples are good.")
            eeg_len = eeg_concat.n_times
            good_samples = np.ones(eeg_len, dtype=bool)
            eeg_masked = eeg_data[:, good_samples]
            # z-scoring..
            eeg_clean = (eeg_masked - eeg_masked.mean(axis=1, keepdims=True)) / eeg_masked.std(axis=1, keepdims=True)
            print(f"{sub}: Clean EEG shape = {eeg_clean.shape}")
            eeg_masked_list[sub] = eeg_masked
            eeg_clean_list[sub] = eeg_clean
    return eeg_clean_list, eeg_masked_list

def centering_predictor_array(predictor_array, min_std=1e-6, predictor_name=''):
    """
    Normalize predictor arrays for TRF modeling.

    Rules:
    - 'envelopes' → z-score if dense and std > min_std.
    - All other predictors → left unchanged.
    - Fully zero predictors → returned unchanged.
    """

    if predictor_name == 'envelopes':
        std = predictor_array.std()
        nonzero_ratio = np.count_nonzero(predictor_array) / len(predictor_array)

        if nonzero_ratio > 0.5:
            print(f'{predictor_name}: Dense predictor, applying z-score.')
            mean = predictor_array.mean()
            return (predictor_array - mean) / std if std > min_std else predictor_array - mean

        elif nonzero_ratio > 0:
            print(f'{predictor_name}: Sparse predictor, mean-centering non-zero entries.')
            mask = predictor_array != 0
            mean = predictor_array[mask].mean()
            predictor_array = predictor_array.copy()
            predictor_array[mask] -= mean
            return predictor_array

        else:
            print(f'{predictor_name}: All zeros, returning unchanged.')
            return predictor_array

    else:
        print(f'{predictor_name}: No normalization applied.')
        return predictor_array


def get_predictor_dict(condition='', pred_type=''):
    predictor_dict = {}
    for files in predictor.iterdir():
        if 'sub' in files.name:
            sub_name = files.name  # e.g., "sub01"
            if condition in ['a1', 'a2'] and sub_name == 'sub01' and stream_type2 == 'deviants':
                continue
            stream1_data, stream2_data = None, None

            for file in files.iterdir():
                if condition in file.name:
                    for stim_type in file.iterdir():
                        if stim_type.name == stream_type1:
                            for array in stim_type.iterdir():
                                if 'concat' in array.name:
                                    stream1_data = np.load(array)
                                    print(list(stream1_data.keys()))
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

def define_streams_dict(predictors1, predictors2):
    for pred_type1, pred_dict1 in predictors1.items():
        for sub, sub_dict in pred_dict1.items():
            sub_dict[f'{stim1}'] = sub_dict.pop('stream1')  # pop to replace OG array, not add extra array with new key
            sub_dict[f'{stim2}'] = sub_dict.pop('stream2')

    for pred_type2, pred_dict2 in predictors2.items():
        for sub, sub_dict in pred_dict2.items():
            sub_dict[f'{stim1}'] = sub_dict.pop('stream2')
            sub_dict[f'{stim2}'] = sub_dict.pop('stream1')
    return predictors1, predictors2


def predictor_mask_bads(predictor_dict, condition, predictor_name=''):
    predictor_dict_masked = {}
    predictor_dict_masked_raw = {}

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
        sub_masked_raw = {}

        for stream_name, stream_array in sub_dict.items():
            if good_samples is not None and len(good_samples) == len(stream_array):
                stream_array_masked = stream_array[good_samples]
                stream_array_clean = centering_predictor_array(stream_array_masked, min_std=1e-6, predictor_name=predictor_name)
            else:
                stream_array_masked = stream_array  #full array if no mask found or mismatched
                stream_array_clean = centering_predictor_array(stream_array_masked, min_std=1e-6, predictor_name=predictor_name)
            sub_masked[stream_name] = stream_array_clean
            sub_masked_raw[stream_name] = stream_array_masked

        predictor_dict_masked[sub] = sub_masked
        predictor_dict_masked_raw[sub] = sub_masked_raw
    return predictor_dict_masked, predictor_dict_masked_raw


def separate_arrays(predictors1, predictors2, eeg_list1, eeg_list2):
        s1_all_stream_targets = {}
        s1_all_stream_distractors = {}
        s1_all_eeg_clean = {}
        for pred_type1, pred_dict1 in predictors1.items():
            all_eeg_clean1, a1_all_stream_target, a1_all_stream_distractor = arrays_lists(eeg_list1,
                                                                                          pred_dict1,
                                                                                          s1_key=f'{stim1}',
                                                                                          s2_key=f'{stim2}')
            s1_all_stream_targets[pred_type1] = a1_all_stream_target
            s1_all_stream_distractors[pred_type1] = a1_all_stream_distractor
            s1_all_eeg_clean[pred_type1] = all_eeg_clean1

        s2_all_stream_targets = {}
        s2_all_stream_distractors = {}
        s2_all_eeg_clean = {}
        for pred_type2, pred_dict2 in predictors2.items():
            all_eeg_clean2, a2_all_stream_distractor, a2_all_stream_target = arrays_lists(eeg_list2,
                                                                                          pred_dict2,
                                                                                          s1_key=f'{stim2}',
                                                                                          s2_key=f'{stim1}')
            s2_all_stream_targets[pred_type2] = a2_all_stream_target
            s2_all_stream_distractors[pred_type2] = a2_all_stream_distractor
            s2_all_eeg_clean[pred_type2] = all_eeg_clean2
        return (s1_all_stream_targets, s1_all_stream_distractors, all_eeg_clean1,
                s2_all_stream_targets, s2_all_stream_distractors, all_eeg_clean2)


def get_stream_arrays_all(s1_all_targets, s1_all_distractors, s2_all_targets, s2_all_distractors):
        all_pred_target_stream_arrays = {}
        all_pred_distractor_stream_arrays = {}
        for pred_type in pred_types:
            s1_array_target = s1_all_targets[pred_type]
            s2_array_target = s2_all_targets[pred_type]
            all_target_stream_arrays = s1_array_target + s2_array_target
            target_stream_all = np.concatenate(all_target_stream_arrays, axis=0)  # shape: (total_samples,)
            all_pred_target_stream_arrays[pred_type] = target_stream_all

            s1_array_distractor = s1_all_distractors[pred_type]
            s2_array_distractor = s2_all_distractors[pred_type]
            all_distractor_stream_arrays = s1_array_distractor + s2_array_distractor
            distractor_stream_all = np.concatenate(all_distractor_stream_arrays, axis=0)  # shape: (total_samples,)
            all_pred_distractor_stream_arrays[pred_type] = distractor_stream_all
        return all_pred_target_stream_arrays, all_pred_distractor_stream_arrays


# concat all subs eegs together in order,
# same for envelopes stream1 and stream2 respectively.
# then vstack stream1 and stream2
# Clean EEH
def arrays_lists(eeg_clean_list_masked, predictor_dict_masked, s1_key='', s2_key=''):
    all_eeg_clean = []
    all_stream1 = []
    all_stream2 = []

    for sub in eeg_clean_list_masked.keys():
        eeg = eeg_clean_list_masked[sub]
        pred = predictor_dict_masked[sub]
        stream1 = pred[f'{s1_key}']
        stream2 = pred[f'{s2_key}']

        all_eeg_clean.append(eeg)
        all_stream1.append(stream1)
        all_stream2.append(stream2)
    return all_eeg_clean, all_stream1, all_stream2



if __name__ == '__main__':

    # best lambda based on investigation of data and model testing:
    best_lambda = 1.0

    print("Available CPUs:", os.cpu_count())
    print(f"Free RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")

    default_path = Path.cwd()
    predictors_path = default_path / 'data/eeg/predictors'
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'
    sfreq = 125

    stream_type1 = 'stream1'
    stream_type2 = 'stream2'

    plane = 'elevation'
    condition1 = 'e1'
    condition2 = 'e2'

    eeg_files1 = get_eeg_files(condition=condition1)
    eeg_files2 = get_eeg_files(condition=condition2)


    eeg_concat_list1 = pick_channels(eeg_files1)
    eeg_concat_list2 = pick_channels(eeg_files2)

    eeg_clean_list_masked1, eeg_masked_list1 = mask_bad_segmets(eeg_concat_list1, condition=condition1)
    eeg_clean_list_masked2, eeg_masked_list2 = mask_bad_segmets(eeg_concat_list2, condition=condition2)

    predictors_list = ['binary_weights', 'envelopes', 'overlap_ratios', 'events_proximity', 'events_proximity', 'RTs']
    pred_types = ['onsets', 'envelopes', 'overlap_ratios', 'events_proximity_pre', 'events_proximity_post', 'RT_labels']



    folder_types = ['all_stims', 'target_nums', 'non_targets', 'distractors_x_deviants']
    if stream_type1 == 'stream1':
        folder_type = folder_types[0]
    elif stream_type1 == 'targets':
        folder_type = folder_types[1]
    elif stream_type1 == 'nt_target':
        folder_type = folder_types[2]
    elif stream_type1 == 'distractors':
        folder_type = folder_types[3]

    stim1 = 'target_stream'
    stim2 = 'distractor_stream'

    s1_predictors = {}
    s2_predictors = {}

    s1_predictors_raw = {}
    s2_predictors_raw = {}

    # Mapping semantic weights

    semantic_mapping = {
        5.0: 1.0,
        4.0: 0.85,
        3.0: 0.65,
        2.0: 0.45,
        1.0: 0.25
    }


    def remap_onsets_nested(predictor_dict):
        remapped_dict = {}
        for subj, stream_dict in predictor_dict.items():
            remapped_streams = {}
            for stream_key, arr in stream_dict.items():
                remapped_arr = arr.copy()
                for orig_val, new_val in semantic_mapping.items():
                    remapped_arr[arr == orig_val] = new_val
                remapped_streams[stream_key] = remapped_arr
            remapped_dict[subj] = remapped_streams
        return remapped_dict


    for predictor_name, pred_type in zip(predictors_list, pred_types):
        predictor = default_path/ f'data/eeg/predictors/{predictor_name}'
        predictor_dict1 = get_predictor_dict(condition='a1', pred_type=pred_type)
        predictor_dict2 = get_predictor_dict(condition='a2', pred_type=pred_type)
        predictor_dict_masked1, predictor_dict_masked_raw1 = predictor_mask_bads(predictor_dict1, condition='a1', predictor_name=pred_type)
        predictor_dict_masked2, predictor_dict_masked_raw2 = predictor_mask_bads(predictor_dict2, condition='a2', predictor_name=pred_type)

        # Remap onsets (semantic weights) before storing
        if pred_type == 'onsets':
            predictor_dict_masked1 = remap_onsets_nested(predictor_dict_masked1)
            predictor_dict_masked2 = remap_onsets_nested(predictor_dict_masked2)
            predictor_dict_masked_raw1 =  remap_onsets_nested(predictor_dict_masked_raw1)
            predictor_dict_masked_raw2  = remap_onsets_nested(predictor_dict_masked_raw2)

        s1_predictors[pred_type] = predictor_dict_masked1
        s1_predictors_raw[pred_type] = predictor_dict_masked_raw1
        s2_predictors[pred_type] = predictor_dict_masked2
        s2_predictors_raw[pred_type] = predictor_dict_masked_raw2


    s1_predictors, s2_predictors = define_streams_dict(s1_predictors, s2_predictors)
    s1_predictors_raw, s2_predictors_raw = define_streams_dict(s1_predictors_raw, s2_predictors_raw)


    # Configuration
    fs = 125
    model_type = 1  # forward model
    lag_start = -0.1
    lag_end = 1.0
    lambda_val = 1.0
    selected_stream = 'distractor_stream'

    # Select which s_predictors to use
    s_predictors = s1_predictors  # change to s2_predictors for elevation if needed

    # Save R-values
    output_dir = default_path / f'data/eeg/trf/trf_testing/composite_model/single_sub/{plane}/{folder_type}'
    output_dir.mkdir(parents=True, exist_ok=True)


    weights_dir = output_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    # Initialize storage for R-values
    subject_rvals = {}

    # Extract subject list from one of the predictors
    subjects = list(next(iter(s_predictors.values())).keys())

    # Run model per subject
    for subject in subjects:
        if plane == 'elevation':
            if subject in ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub08']:
                continue
        print(f"Running composite TRF for {subject}, {selected_stream}, {plane}...")

        predictor_arrays = []
        for pred_type in ['onsets', 'envelopes', 'overlap_ratios', 'events_proximity_pre',
                          'events_proximity_post', 'RT_labels']:
            arr = s_predictors[pred_type][subject][selected_stream]
            predictor_arrays.append(arr)

        # Stack predictors
        X = np.column_stack(predictor_arrays)

        # Z-score only the envelope predictor (index 1)
        X[:, 1] = (X[:, 1] - np.mean(X[:, 1])) / np.std(X[:, 1])

        # Get EEG data for this subject
        eeg = eeg_clean_list_masked1[subject].mean(axis=0)  # average over ROI channels

        # Ensure EEG and predictor array lengths match
        min_len = min(len(eeg), len(X))
        X = X[:min_len]
        eeg = eeg[:min_len]

        # Run TRF model
        trf = TRF(direction=model_type)
        trf.train(X, eeg, fs=fs, tmin=lag_start, tmax=lag_end, regularization=best_lambda, seed=42)
        prediction, r = trf.predict(X, eeg)

        subject_rvals[subject] = r
        # Save TRF weights (time × predictors)
        np.save(weights_dir / f"{subject}_weights_{selected_stream}.npy", trf.weights)

        # Save time lags once (same for all)
        if subject == subjects[0]:
            np.save(weights_dir / "trf_time_lags.npy", trf.times)

        # Optionally: Save metadata as a CSV row
        metadata_path = weights_dir / f"metadata_{selected_stream}.csv"
        metadata_row = {
            "subject": subject,
            "stream": selected_stream,
            "plane": plane,
            "r_value": r,
            "num_predictors": trf.weights.shape[0],
            "num_lags": trf.weights.shape[1]
        }
        if metadata_path.exists():
            df_meta = pd.read_csv(metadata_path)
            df_meta = pd.concat([df_meta, pd.DataFrame([metadata_row])], ignore_index=True)
        else:
            df_meta = pd.DataFrame([metadata_row])
        df_meta.to_csv(metadata_path, index=False)

    np.save(output_dir / f"subjectwise_rvals_{plane}_{selected_stream}_{folder_type}.npy", subject_rvals)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.bar(subject_rvals.keys(), subject_rvals.values(), color='slateblue')
    plt.xticks(rotation=45)
    plt.ylabel('Correlation (r)')
    plt.title(f'TRF Composite Model: {plane.capitalize()} - {selected_stream} - {folder_type}')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_dir / f"subjectwise_trf_{plane}_{selected_stream}.png", dpi=300)
    plt.show()
