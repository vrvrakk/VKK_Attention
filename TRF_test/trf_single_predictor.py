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
from scipy.signal import welch
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from mtrf.stats import pearsonr, neg_mse, nested_crossval, crossval


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
    - 'envelopes', 'overlap_ratios', 'events_proximity', 'RTs' → z-score if std > min_std.
    - 'binary_weights' → leave unchanged (categorical codes: 0,1,2,3,4).
    - Sparse arrays (<50% non-zero values) → mean-center non-zeros only.
    """

    if pred_type == 'onsets' or pred_type == 'RT_labels': # do not normalize semantic weights arrays
        print("Predictor type is categorical (semantic_weights/RTs): skipping transformation.")
        return predictor_array

    std = predictor_array.std() # otherwise estimate STD and the non-zero vals ratio
    nonzero_ratio = np.count_nonzero(predictor_array) / len(predictor_array)

    if nonzero_ratio > 0.5:  # if non-zeros exceed 50% -> z-score
        # however, if std is close to 0: center the mean only
        # Dense predictor → full z-score
        print(f'{pred_type}: Dense predictor, applying z-score.')
        mean = predictor_array.mean()
        return (predictor_array - mean) / std if std > min_std else predictor_array - mean

    elif nonzero_ratio > 0:
        # Sparse → mean-center only non-zero values
        print(f'{pred_type}: Sparse predictor, mean-centering non-zero entries.')

        predictor_array = predictor_array.copy()  # <- this is critical
        mask = predictor_array != 0
        mean = predictor_array[mask].mean()
        print(f"{pred_type}: Non-zero entries = {np.count_nonzero(predictor_array)}, Mean = {mean}")
        predictor_array[mask] -= mean
        return predictor_array

    else:
        # if all values are just zero: first of all, something is wrong with the array.
        print(f'{pred_type}: All zeros, returning unchanged.')
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
                stream_array_masked = stream_array[good_samples].copy()
                stream_array_clean = centering_predictor_array(stream_array_masked, min_std=1e-6, pred_type=pred_type)
            else:
                stream_array_masked = stream_array[good_samples].copy()
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


def optimize_lambda(predictor, eeg, fs, tmin, tmax, lambdas):
    scores = {}
    fwd_trf = TRF(direction=1)
    for l in lambdas:
        r = crossval(fwd_trf, predictor, eeg, fs=fs, tmin=tmin, tmax=tmax, regularization=l)
        scores[l] = r

    best_r = np.max(list(scores.values()))
    best_lambda = [l for l, r in scores.items() if r == best_r][0]
    return best_lambda, best_r, scores


def plot_lambda_scores(scores):
    lambdas = list(scores.keys())
    performance = list(scores.values())
    plt.plot(lambdas, performance, marker='o')
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Mean r')
    plt.title('TRF Performance vs. Lambda')
    plt.grid(True)
    plt.show()
    plt.savefig(save_path / f'{plane}_lambda_scores_{predictor_name}.png')
    plt.close()


if __name__ == '__main__':

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
    n = 2
    predictors_list = ['binary_weights', 'envelopes', 'overlap_ratios', 'events_proximity', 'events_proximity', 'RTs']
    predictor_name = predictors_list[n]
    pred_types = ['onsets', 'envelopes', 'overlap_ratios', 'events_proximity_pre', 'events_proximity_post', 'RT_labels']
    pred_type = pred_types[n]
    predictor = Path(f'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/predictors/{predictor_name}')
    stream_type1 = 'targets'
    stream_type2 = 'distractors'

    predictor_dict1 = get_predictor_dict(condition='a1', pred_type=pred_type)
    predictor_dict2 = get_predictor_dict(condition='a2', pred_type=pred_type)

    predictor_dict_masked1, predictor_dict_masked_only1 = predictor_mask_bads(predictor_dict1, condition='a1', pred_type=pred_type)
    for sub, streams in predictor_dict_masked_only1.items():
        stream1_nz = np.count_nonzero(streams['stream1'])
        stream2_nz = np.count_nonzero(streams['stream2'])
        print(f"{sub}: stream1 nonzeros = {stream1_nz}, stream2 nonzeros = {stream2_nz}")
    predictor_dict_masked2, predictor_dict_masked_only2 = predictor_mask_bads(predictor_dict2, condition='a2', pred_type=pred_type)
    # todo: check overlap ratios predictor
    stim1 = 'target_stream'
    stim2 = 'distractor_stream'
    for sub, sub_dict in predictor_dict_masked1.items():
        sub_dict[f'{stim1}'] = sub_dict.pop('stream1')  # pop to replace OG array, not add extra array with new key
        sub_dict[f'{stim2}'] = sub_dict.pop('stream2')

    for sub, sub_dict in predictor_dict_masked2.items():
        sub_dict[f'{stim1}'] = sub_dict.pop('stream1')
        sub_dict[f'{stim2}'] = sub_dict.pop('stream2')

    all_eeg_clean1, a1_all_stream_target, a1_all_stream_distractor = arrays_lists(eeg_clean_list_masked1,
                                                                                  predictor_dict_masked1,
                                                                                  s1_key=f'{stim1}',
                                                                                  s2_key=f'{stim2}')
    all_eeg_clean2, a2_all_stream_distractor, a2_all_stream_target = arrays_lists(eeg_clean_list_masked2,
                                                                                  predictor_dict_masked2,
                                                                                  s1_key=f'{stim2}',
                                                                                  s2_key=f'{stim1}')

    all_eeg_clean = all_eeg_clean1 + all_eeg_clean2
    all_target_stream_arrays = a1_all_stream_target + a2_all_stream_target
    all_distractor_stream_arrays = a1_all_stream_distractor + a2_all_stream_distractor
    # Concatenate across subjects
    eeg_all = np.concatenate(all_eeg_clean, axis=1)  # shape: (total_samples, channels)
    target_stream_all = np.concatenate(all_target_stream_arrays, axis=0)  # shape: (total_samples,)
    distractor_stream_all = np.concatenate(all_distractor_stream_arrays, axis=0)  # shape: (total_samples,)

    def count_nonzero_events(arr, stim_len=93):
        """Count how many non-overlapping non-zero events exist in the array."""
        i = 0
        count = 0
        while i < len(arr) - stim_len:
            if arr[i] != 0:
                count += 1
                i += stim_len  # skip full event
            else:
                i += 1
        return count


    # Run for both
    n_target_events = count_nonzero_events(target_stream_all)
    n_distractor_events = count_nonzero_events(distractor_stream_all)
    print(f"Target events: {n_target_events}")
    print(f"Distractor events: {n_distractor_events}")

    # Make stream2 orthogonal to stream1
    model = LinearRegression().fit(target_stream_all.reshape(-1, 1), distractor_stream_all)
    distractor_stream_ortho = distractor_stream_all - model.predict(target_stream_all.reshape(-1, 1))

    # Stack predictors (final TRF design matrix)
    predictors_stacked = np.vstack([target_stream_all, distractor_stream_ortho]).T  # shape: (samples, 2)

    eeg_data_all = eeg_all.T
    print(f"EEG shape: {eeg_data_all.shape}, Predictors shape: {predictors_stacked.shape}")

    # checking collinearity:
    X = pd.DataFrame(predictors_stacked, columns=[f'{stim1}', f'{stim2}'])
    X = sm.add_constant(X)  # Add intercept for VIF calc
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    print(vif)

    # split into trials:
    # 1 min long blocks
    n_samples = sfreq * 60
    total_samples = len(predictors_stacked)
    n_folds = total_samples // n_samples
    # Split predictors and EEG into subject chunks
    X_folds = np.array_split(predictors_stacked, n_folds)
    Y_folds = np.array_split(eeg_data_all, n_folds)

    # lambdas = np.logspace(-2, 2, 20)  # based on prev literature

    # best_regularization, best_r, scores = optimize_lambda(X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, lambdas=lambdas)

    save_path = default_path / f'data/eeg/trf/trf_testing/{predictor_name}/{plane}'
    save_path.mkdir(parents=True, exist_ok=True)

    # plot_lambda_scores(scores)
    best_regularization = 1.0

    trf = TRF(direction=1, metric=pearsonr)
    trf.train(X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=best_regularization, seed=42)
    prediction, r = trf.predict(predictors_stacked, eeg_data_all)
    print(f"Full model correlation: {r.round(3)}")

    predictor_names = [f'{stim1}', f'{stim2}']  # or however many you have
    weights = trf.weights  # shape: (n_features, n_lags, n_channels)
    time_lags = np.linspace(-0.1, 1.0, weights.shape[1])  # time axis

    # Loop and plot
    # Define your lag window of interest
    tmin_plot = 0.0
    tmax_plot = 1.0

    # Create a mask for valid time lags
    lag_mask = (time_lags >= tmin_plot) & (time_lags <= tmax_plot)
    time_lags_trimmed = time_lags[lag_mask]

    # Loop and plot
    data_path = save_path / 'data'
    data_path.mkdir(parents=True, exist_ok=True)
    # Save TRF results for this condition
    np.savez(
        data_path / f'{plane}_{pred_type}_{stream_type1}_{stream_type2}_TRF_results.npz',
        # scores=scores,
        weights=weights,  # raw TRF weights (n_predictors, n_lags, n_channels)
        r=r,
        # r_crossval=best_r,
        best_lambda=best_regularization,
        time_lags=time_lags,
        time_lags_trimmed=time_lags_trimmed,
        predictor_names=np.array(predictor_names),
        condition=plane
    )

    for i, name in enumerate(predictor_names):
        filename = name + '_' + pred_type + '_' + stream_type1 + '_' + stream_type2
        plt.figure(figsize=(8, 4))
        trf_weights = weights[i].T[:, lag_mask]  # shape: (n_channels, selected_lags)
        # Smoothing with Hamming window for aesthetic purposes..
        window_len = 11
        hamming_win = np.hamming(window_len)
        hamming_win /= hamming_win.sum()
        smoothed_weights = np.array([
            np.convolve(trf_weights[ch], hamming_win, mode='same')
            for ch in range(trf_weights.shape[0])
        ])

        for ch in range(trf_weights.shape[0]):
            plt.plot(time_lags_trimmed, smoothed_weights[ch], alpha=0.4)

        plt.title(f'TRF for {name}')
        plt.xlabel('Time lag (s)')
        plt.ylabel('Amplitude')
        plt.plot([], [], ' ', label=f'λ = {best_regularization:.2f}, r = {r:.2f}')
        plt.legend(loc='upper right', fontsize=8, ncol=2)
        plt.tight_layout()
        plt.show()
        plt.savefig(save_path / f'{filename}.png', dpi=300)
    plt.close('all')

