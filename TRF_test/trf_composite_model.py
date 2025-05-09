from pathlib import Path
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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


''' A script to get optimal lambda per condition (azimuth vs elevation) - with stacked predictors - all 8'''


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


def centering_predictor_array(predictor_array, min_std=1e-6, predictor_name=''):
    """
    Normalize predictor arrays for TRF modeling.

    Rules:
    - 'envelopes', 'overlap_ratios', 'events_proximity', 'RTs' → z-score if std > min_std.
    - 'binary_weights' → leave unchanged (categorical codes: 0,1,2,3,4).
    - Sparse arrays (<50% non-zero values) → mean-center non-zeros only.
    """

    if predictor_name == 'binary_weights': # do not normalize semantic weights arrays
        print("Predictor type is categorical (binary_weights): skipping transformation.")
        return predictor_array

    std = predictor_array.std() # otherwise estimate STD and the non-zero vals ratio
    nonzero_ratio = np.count_nonzero(predictor_array) / len(predictor_array)

    if nonzero_ratio > 0.5:  # if non-zeros exceed 50% -> z-score
        # however, if std is close to 0: center the mean only
        # Dense predictor → full z-score
        print(f'{predictor_name}: Dense predictor, applying z-score.')
        mean = predictor_array.mean()
        return (predictor_array - mean) / std if std > min_std else predictor_array - mean

    elif nonzero_ratio > 0:
        # Sparse → mean-center only non-zero values
        print(f'{predictor_name}: Sparse predictor, mean-centering non-zero entries.')
        mask = predictor_array != 0
        mean = predictor_array[mask].mean()
        predictor_array[mask] -= mean
        return predictor_array

    else:
        # if all values are just zero: first of all, something is wrong with the array.
        print(f'{predictor_name}: All zeros, returning unchanged.')
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


def predictor_mask_bads(predictor_dict, condition, predictor_name=''):
    predictor_dict_masked = {}

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

        for stream_name, stream_array in sub_dict.items():
            if good_samples is not None and len(good_samples) == len(stream_array):
                stream_array_masked = stream_array[good_samples]
                stream_array_clean = centering_predictor_array(stream_array_masked, min_std=1e-6, predictor_name=predictor_name)
            else:
                stream_array_masked = stream_array  # use full array if no mask found or mismatched
                stream_array_clean = centering_predictor_array(stream_array_masked, min_std=1e-6, predictor_name=predictor_name)
            sub_masked[stream_name] = stream_array_clean

        predictor_dict_masked[sub] = sub_masked
    return predictor_dict_masked


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
        env = predictor_dict_masked[sub]
        stream1 = env[f'{s1_key}']
        stream2 = env[f'{s2_key}']

        all_eeg_clean.append(eeg)
        all_stream1.append(stream1)
        all_stream2.append(stream2)
    return all_eeg_clean, all_stream1, all_stream2


def optimize_lambda(X_folds, Y_folds, fs, tmin, tmax, lambdas, n_jobs=-1):
    def test_lambda(lmbda):
        fwd_trf = TRF(direction=1)
        r = crossval(fwd_trf, X_folds, Y_folds, fs, tmin, tmax, lmbda)
        return lmbda, r.mean()

    print(f"Running lambda optimization across {len(lambdas)} values...")
    results = Parallel(n_jobs=n_jobs)(
        delayed(test_lambda)(lmbda) for lmbda in tqdm(lambdas, desc=f'Lambda testing')
    )

    # Find best
    best_lambda, best_score = max(results, key=lambda x: x[1])
    print(f'Best lambda: {best_lambda:.2e} (mean r = {best_score:.3f})')
    return best_lambda


if __name__ == '__main__':

    print("Available CPUs:", os.cpu_count())
    print(f"Free RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")

    default_path = Path.cwd()
    predictors_path = default_path / 'data/eeg/predictors'
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'
    sfreq = 125

    eeg_files1 = get_eeg_files(condition='a1')
    eeg_files2 = get_eeg_files(condition='a2')
    plane = 'azimuth'

    eeg_concat_list1 = pick_channels(eeg_files1)
    eeg_concat_list2 = pick_channels(eeg_files2)

    eeg_clean_list_masked1 = mask_bad_segmets(eeg_concat_list1, condition='a1')
    eeg_clean_list_masked2 = mask_bad_segmets(eeg_concat_list2, condition='a2')

    predictors_list = ['binary_weights', 'envelopes', 'overlap_ratios', 'events_proximity', 'events_proximity', 'RTs']
    pred_types = ['onsets', 'envelopes', 'overlap_ratios', 'events_proximity_pre', 'events_proximity_post', 'RTs']
    stream_type1 = 'stream1'
    stream_type2 = 'stream2'

    stim1 = 'target_stream'
    stim2 = 'distractor_stream'
    s1_predictors = {}
    s2_predictors = {}
    for predictor_name, pred_type in zip(predictors_list, pred_types):
        predictor = Path(f'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/predictors/{predictor_name}')
        if predictor_name == 'RTs' and stream_type1 != 'targets':
            continue
        else:
            predictor_dict1 = get_predictor_dict(condition='a1', pred_type=pred_type)
            predictor_dict2 = get_predictor_dict(condition='a2', pred_type=pred_type)
            predictor_dict_masked1 = predictor_mask_bads(predictor_dict1, condition='a1', predictor_name=predictor_name)
            predictor_dict_masked2 = predictor_mask_bads(predictor_dict2, condition='a2', predictor_name=predictor_name)
            s1_predictors[pred_type] = predictor_dict_masked1
            s2_predictors[pred_type] = predictor_dict_masked2

    for pred_type1, pred_dict1 in s1_predictors.items():
        for sub, sub_dict in pred_dict1.items():
            sub_dict[f'{stim1}'] = sub_dict.pop('stream1')  # pop to replace OG array, not add extra array with new key
            sub_dict[f'{stim2}'] = sub_dict.pop('stream2')
            
    for pred_type2, pred_dict2 in s2_predictors.items():
        for sub, sub_dict in pred_dict2.items():
            sub_dict[f'{stim2}'] = sub_dict.pop('stream2')
            sub_dict[f'{stim1}'] = sub_dict.pop('stream1')
    
    s1_all_stream_targets = {}
    s1_all_stream_distractors = {}
    s1_all_eeg_clean = {}
    for pred_type1, pred_dict1 in s1_predictors.items():
        all_eeg_clean1, a1_all_stream_target, a1_all_stream_distractor = arrays_lists(eeg_clean_list_masked1,
                                                                                      pred_dict1,
                                                                                      s1_key=f'{stim1}',
                                                                                      s2_key=f'{stim2}')
        s1_all_stream_targets[pred_type1] = a1_all_stream_target
        s1_all_stream_distractors[pred_type1] = a1_all_stream_distractor
        s1_all_eeg_clean[pred_type1] = all_eeg_clean1

    s2_all_stream_targets = {}
    s2_all_stream_distractors = {}
    s2_all_eeg_clean = {}
    for pred_type2, pred_dict2 in s2_predictors.items():
        all_eeg_clean2, a2_all_stream_distractor, a2_all_stream_target = arrays_lists(eeg_clean_list_masked2,
                                                                                      pred_dict2,
                                                                                      s1_key=f'{stim2}',
                                                                                      s2_key=f'{stim1}')
        s2_all_stream_targets[pred_type2] = a2_all_stream_target
        s2_all_stream_distractors[pred_type2] = a2_all_stream_distractor
        s2_all_eeg_clean[pred_type2] = all_eeg_clean2

    all_eeg_clean = all_eeg_clean1 + all_eeg_clean2 # concat as is

    all_pred_target_stream_arrays = {}
    all_pred_distractor_stream_arrays = {}
    for pred_type in pred_types:
        if pred_type == 'RTs' and stream_type1 != 'targets':
            continue
        s1_array_target = s1_all_stream_targets[pred_type]
        s2_array_target = s2_all_stream_targets[pred_type]
        all_target_stream_arrays = s1_array_target + s2_array_target
        target_stream_all = np.concatenate(all_target_stream_arrays, axis=0)  # shape: (total_samples,)
        all_pred_target_stream_arrays[pred_type] = target_stream_all

        s1_array_distractor = s1_all_stream_distractors[pred_type]
        s2_array_distractor = s2_all_stream_distractors[pred_type]
        all_distractor_stream_arrays = s1_array_distractor + s2_array_distractor
        distractor_stream_all = np.concatenate(all_distractor_stream_arrays, axis=0)  # shape: (total_samples,)
        all_pred_distractor_stream_arrays[pred_type] = distractor_stream_all

    # Concatenate across subjects
    eeg_all = np.concatenate(all_eeg_clean, axis=1)  # shape: (total_samples, channels)

    # Define order to ensure consistency
    if stream_type1 != 'targets':
        ordered_keys = ['onsets', 'envelopes', 'overlap_ratios',
                        'events_proximity_pre', 'events_proximity_post']
    else:
        ordered_keys = ['onsets', 'envelopes', 'overlap_ratios',
                        'events_proximity_pre', 'events_proximity_post', 'RTs']

    # Stack predictors for the target stream
    X_target = np.column_stack([all_pred_target_stream_arrays[k] for k in ordered_keys])

    # Stack predictors for the distractor stream
    X_distractor = np.column_stack([all_pred_distractor_stream_arrays[k] for k in ordered_keys])

    print("X_target shape:", X_target.shape)
    print("X_distractor shape:", X_distractor.shape)

    eeg_all = eeg_all.T

    # checking collinearity:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Build combined DataFrame
    X = pd.DataFrame(
        np.column_stack([
            X_target,  # target predictors
            X_distractor  # distractor predictors
        ]),
        columns=[f'{k}_target' for k in ordered_keys] + [f'{k}_distractor' for k in ordered_keys]
    )

    # Add constant for VIF calculation
    X = sm.add_constant(X)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    print(vif)

    # split into trials:
    predictors_stacked = X.values  # ← ready for modeling
    n_samples = sfreq * 60
    total_samples = len(predictors_stacked)
    n_folds = total_samples // n_samples
    # Split predictors and EEG into subject chunks
    X_folds = np.array_split(predictors_stacked, n_folds)
    Y_folds = np.array_split(eeg_all, n_folds)

    lambdas = np.logspace(-2, 2, 20)  # based on prev literature

    best_regularization = optimize_lambda(X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, lambdas=lambdas, n_jobs=2)
    # Each CPU core handles one λ — so if you have 8 cores, you test 8 lambdas in parallel.
    print(f'Best lambda for {plane} is {best_regularization}')
    save_path = default_path / f'data/eeg/trf/trf_testing/{predictor_name}/{plane}'
    save_path.mkdir(parents=True, exist_ok=True)
    data_path = save_path / 'data'
    data_path.mkdir(parents=True, exist_ok=True)

    np.savez(data_path / f'{plane}_TRF_best_lambda_all.npz',
             best_lambda=best_regularization,
             plane=plane)