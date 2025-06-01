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


def save_model_inputs(eeg_all, all_pred_target_stream_arrays, all_pred_distractor_stream_arrays, plane='', folder_type=''):
    """
    eeg_all (np.ndarray): Concatenated EEG data.
    all_pred_distractor_stream_arrays (list of np.ndarray): Predictors for distractor stream.
    all_pred_target_stream_arrays (list of np.ndarray): Predictors for target stream.
    plane (str): Either "azimuth" or "elevation", used in filename.
    save_dir (str): Directory to save the files in (default: 'model_inputs').
    """
    save_dir = default_path / f'data/eeg/trf/model_inputs/{plane}/{folder_type}'
    os.makedirs(save_dir, exist_ok=True)
    np.save(os.path.join(save_dir/f'{plane}_eeg_all.npy'), eeg_all)
    np.savez(os.path.join(save_dir/f'{plane}_{folder_type}_pred_{stim1}_arrays.npz'), **all_pred_target_stream_arrays)
    np.savez(os.path.join(save_dir/f'{plane}_{folder_type}_pred_{stim2}_arrays.npz'), **all_pred_distractor_stream_arrays)
    # ** -> "unpack a dictionary into keyword arguments."

if __name__ == '__main__':

    # best lambda based on investigation of data and model testing:
    best_lambda = 1.0

    print("Available CPUs:", os.cpu_count())
    print(f"Free RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")

    default_path = Path.cwd()
    predictors_path = default_path / 'data/eeg/predictors'
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'
    sfreq = 125

    stream_type1 = 'stream1'  # used to select correct folder within sub's folder
    stream_type2 = 'stream2'

    plane = 'elevation'
    if plane == 'azimuth':
        condition1 = 'a1'
        condition2 = 'a2'
    elif plane == 'elevation':
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
        predictor_dict1 = get_predictor_dict(condition=condition1, pred_type=pred_type)
        predictor_dict2 = get_predictor_dict(condition=condition2, pred_type=pred_type)
        predictor_dict_masked1, predictor_dict_masked_raw1 = predictor_mask_bads(predictor_dict1, condition=condition1, predictor_name=pred_type)
        predictor_dict_masked2, predictor_dict_masked_raw2 = predictor_mask_bads(predictor_dict2, condition=condition2, predictor_name=pred_type)

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

    (s1_all_stream_targets_raw, s1_all_stream_distractors_raw, all_eeg_clean1_raw,
     s2_all_stream_targets_raw, s2_all_stream_distractors_raw, all_eeg_clean2_raw) = separate_arrays(s1_predictors_raw, s2_predictors_raw, eeg_masked_list1, eeg_masked_list2)

    (s1_all_stream_targets, s1_all_stream_distractors, all_eeg_clean1,
     s2_all_stream_targets, s2_all_stream_distractors, all_eeg_clean2) = separate_arrays(s1_predictors, s2_predictors, eeg_clean_list_masked1, eeg_clean_list_masked2)
    all_eeg_clean = all_eeg_clean1 + all_eeg_clean2 # concat as is, only once
    eeg_all_raw = all_eeg_clean1_raw + all_eeg_clean2_raw

    # Concatenate across subjects

    all_pred_target_stream_arrays_raw, all_pred_distractor_stream_arrays_raw = get_stream_arrays_all(s1_all_stream_targets_raw, s1_all_stream_distractors_raw, s2_all_stream_targets_raw, s2_all_stream_distractors_raw)
    eeg_raw = np.concatenate(eeg_all_raw, axis=1)
    save_model_inputs(eeg_raw, all_pred_target_stream_arrays_raw, all_pred_distractor_stream_arrays_raw, plane=f'{plane}_raw')

    eeg_all = np.concatenate(all_eeg_clean, axis=1)  # shape: (total_samples, channels)
    all_pred_target_stream_arrays, all_pred_distractor_stream_arrays = get_stream_arrays_all(s1_all_stream_targets, s1_all_stream_distractors, s2_all_stream_targets, s2_all_stream_distractors)
    save_model_inputs(eeg_all, all_pred_target_stream_arrays, all_pred_distractor_stream_arrays, plane=plane, folder_type=folder_type)

    # Define order to ensure consistency
    ordered_keys = ['onsets', 'envelopes', 'RT_labels', 'overlap_ratios']
    folder_names = ['onsets', 'envs', 'onsets_envs', 'onsets_envs_RTs', 'onsets_envs_RTs_overlaps', 'all']
    model_name = folder_names[-1]

    # exclude proximity predictors from composite model, as standalone they do not increase predictive power
    X_target = np.column_stack([all_pred_target_stream_arrays[k] for k in ordered_keys])

    # Stack predictors for the distractor stream
    X_distractor = np.column_stack([all_pred_distractor_stream_arrays[k] for k in ordered_keys])

    print("X_target shape:", X_target.shape)
    print("X_distractor shape:", X_distractor.shape)

    eeg_all_T = eeg_all.T

    # 3. Combine predictors into final matrix
    def orthogonalize_predictor(X, target_col, regressor_cols):
        """Orthogonalize `target_col` in X w.r.t. `regressor_cols`."""
        model = LinearRegression().fit(X[regressor_cols], X[target_col])
        X_orth = X[target_col] - model.predict(X[regressor_cols])
        return X_orth


    def X_df(X_stream, stim):
        # 1. Build design matrix
        X = pd.DataFrame(
            np.column_stack([X_stream]),
            columns=[f'{k}_{stim}' for k in ordered_keys]
        )

        # 2. Add constant for VIF
        X = sm.add_constant(X)
        X = sm.add_constant(X)  # Just once is enough; double add doesn't hurt but isn't needed

        # 3. Compute initial VIF
        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns
        )
        print("\nInitial VIFs:")
        print(vif)

        # 4. Orthogonalize predictors with VIF > 3
        for col, vif_val in vif.items():
            if col == 'const' or vif_val <= 3:
                continue

            print(f'\nHigh collinearity detected for: {col} (VIF={vif_val:.2f})')

            # Pick regressors: all other non-constant, non-target columns
            regressors = [c for c in X.columns if c != col and c != 'const']

            # Orthogonalize
            X[col] = orthogonalize_predictor(X, target_col=col, regressor_cols=regressors)

            print(f' → Orthogonalized {col} w.r.t.: {regressors}')

        # 5. Recompute final VIFs
        vif_final = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns
        )
        print("\nFinal VIFs after orthogonalization:")
        print(vif_final)

        return X, vif_final


    # Ask for stream input
    stream = input('target, distractor, deviants? ')
    if stream == 'target':
        X, vif = X_df(X_target, stim=stim1)
    elif stream == 'distractor':
        X, vif = X_df(X_distractor, stim=stim2)
    elif stream == 'deviants':
        X, vif = X_df(X_distractor, stim=stim2)


    # split into trials:
    predictors_stacked = X.values  # ← ready for modeling
    n_samples = sfreq * 60
    total_samples = len(predictors_stacked)
    n_folds = total_samples // n_samples
    # Split predictors and EEG into subject chunks
    X_folds = np.array_split(predictors_stacked, n_folds)
    Y_folds = np.array_split(eeg_all_T, n_folds)

    # Multiply that by 482 folds × 2 streams × 2 jobs, and the savings are huge.
    # Set reproducible seed
    import random
    random.seed(42)

    trf = TRF(direction=1)
    trf.train(X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=best_lambda, seed=42)

    predictions = []
    r_vals = []

    for i in tqdm(range(n_folds)):
        start = i * n_samples
        end = start + n_samples
        X_chunk = predictors_stacked[start:end]
        Y_chunk = eeg_all_T[start:end]

        pred_chunk, r_chunk = trf.predict(X_chunk, Y_chunk)
        predictions.append(pred_chunk)
        r_vals.append(r_chunk)

    predicted_full = np.vstack(predictions)
    # Average across the 3rd dimension (channels)
    predicted_avg_channels = np.mean(predicted_full, axis=2)  # shape: (361, 7500)

    r_mean = np.mean(r_vals, axis=0)
    print("Avg r across all chunks:", np.round(r_mean, 3))

    n_total = len(X_folds)
    n_split = n_total // 3  # or any chunk size you want

    X_subsample = X_folds[:n_split]
    Y_subsample = Y_folds[:n_split]
    X_subsample2 = X_folds[n_split:2 * n_split]
    Y_subsample2 = Y_folds[n_split:2 * n_split]
    X_subsample3 = X_folds[2 * n_split:]
    Y_subsample3 = Y_folds[2 * n_split:]
    X_subsamples = [X_subsample, X_subsample2, X_subsample3]
    Y_subsamples = [Y_subsample, Y_subsample2, Y_subsample3]

    r_crossvals = []
    for x_subsamples, y_subsamples in zip(X_subsamples, Y_subsamples):
        r_crossval = crossval(
            trf,
            x_subsamples,
            y_subsamples,
            fs=sfreq,
            tmin=-0.1,
            tmax=1.0,
            regularization=best_lambda
        )
        r_crossvals.append(r_crossval)

    r_crossval_mean = np.mean(r_crossvals)
    print("Avg r_crossval across all chunks:", np.round(r_crossval_mean, 3))

    weights = trf.weights  # shape: (n_features, n_lags, n_channels)
    weights_avg = np.mean(weights, axis=-1)
    time_lags = np.linspace(-0.1, 1.0, weights.shape[1])  # time axis

    # Loop and plot
    # Define your lag window of interest
    tmin_plot = 0.0
    tmax_plot = 1.0

    # Create a mask for valid time lags
    lag_mask = (time_lags >= tmin_plot) & (time_lags <= tmax_plot)
    time_lags_trimmed = time_lags[lag_mask]

    # Loop and plot
    save_path = default_path / f'data/eeg/trf/trf_testing/composite_model/{plane}/{folder_type}'
    save_path.mkdir(parents=True, exist_ok=True)
    data_path = save_path / 'data'
    data_path.mkdir(parents=True, exist_ok=True)

    trf_preds = list(X.columns)


    # Save TRF results for this condition
    np.savez(
        data_path / f'{plane}_{model_name}_{stream}_TRF_results.npz',
        vif=vif,
        results=predicted_full,
        results_avg=predicted_avg_channels,
        preds=list(X.columns),
        weights=weights,  # raw TRF weights (n_predictors, n_lags, n_channels)
        weights_avg=weights_avg,
        r=r_mean,
        r_crossval=r_crossval_mean,
        best_lambda=best_lambda,
        time_lags=time_lags,
        time_lags_trimmed=time_lags_trimmed,
        condition=plane
    )

    # Plot each predictor
    for i, name in enumerate(trf_preds):
        if i == 0:
            continue
        filename = model_name + '_'  + name
        plt.ion()
        plt.figure(figsize=(8, 4))

        trf_weights = weights[i].T[:, lag_mask]  # shape: (n_channels, n_lags_selected)

        # Smooth for aesthetics
        window_len = 11
        hamming_win = np.hamming(window_len)
        hamming_win /= hamming_win.sum()
        smoothed_weights = np.array([
            np.convolve(trf_weights[ch], hamming_win, mode='same')
            for ch in range(trf_weights.shape[0])
        ])

        # Plot per channel
        for ch in range(smoothed_weights.shape[0]):
            plt.plot(time_lags_trimmed, smoothed_weights[ch], alpha=0.4)

        plt.title(f'TRF for {name}')
        plt.xlabel('Time lag (s)')
        plt.ylabel('Amplitude (a.u.)')
        plt.plot([], [], ' ', label=f'λ = {best_lambda:.2f}, r = {r_crossval_mean:.2f}')
        plt.legend(loc='upper right', fontsize=8)
        plt.tight_layout()

        # Save and show
        plt.savefig(save_path / f'{filename}.png', dpi=300)
        plt.show()

    plt.close('all')

