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
from scipy.stats import sem
from scipy.fft import fft, fftfreq

from mtrf.stats import crossval
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
            eeg_concat.pick('all')
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


def remap_onsets_nested(predictor_dict, mapping):
    remapped_dict = {}
    for subj, stream_dict in predictor_dict.items():
        remapped_streams = {}
        for stream_key, arr in stream_dict.items():
            remapped_arr = arr.copy()
            for orig_val, new_val in mapping.items():
                remapped_arr[arr == orig_val] = new_val
            remapped_streams[stream_key] = remapped_arr
        remapped_dict[subj] = remapped_streams
    return remapped_dict



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

def X_df(X_stream, column_names, min_std=1e-6):
    """
    Returns the original DataFrame and VIFs for inspection only.
    Does not alter or drop any predictors — purely diagnostic.
    """
    # Build DataFrame
    X = pd.DataFrame(X_stream, columns=column_names)

    # Identify low-variance predictors (informational only)
    stds = X.std()
    low_var = stds[stds <= min_std].index.tolist()
    if low_var:
        print(f"[Warning] Low-variance columns (kept in model, noted for info only): {low_var}")

    # Add constant only for VIF computation
    X_for_vif = sm.add_constant(X)
    vif = pd.Series(
        [variance_inflation_factor(X_for_vif.values, i) for i in range(X_for_vif.shape[1])],
        index=X_for_vif.columns
    )

    print("\n[INFO] VIFs (diagnostic only, no effect on model):\n", vif)

    # Return full unaltered X and VIFs (excluding 'const' for readability)
    return vif

def design_matrices(s_predictors):
    all_subject_design_matrices = {}
    # Expected predictor column names
    predictor_keys = s_predictors.keys()
    expected_columns = [
        f"{key}_{stream}" for key in predictor_keys for stream in ['target', 'distractor']
    ]

    for sub in subs:
        X_data = {}
        for key in predictor_keys:
            for stream in ['target', 'distractor']:
                colname = f"{key}_{stream}"
                try:
                    X_data[colname] = s_predictors[key][sub][f"{stream}_stream"]
                except KeyError:
                    print(f"[Warning] Missing {colname} for {sub} — filling with zeros")
                    shape_like = next(iter(X_data.values()), np.zeros(75000))
                    X_data[colname] = np.zeros_like(shape_like)

        X_df = pd.DataFrame(X_data)[expected_columns]  # enforce order
        all_subject_design_matrices[sub] = X_df
        print(f"[{sub}] Design matrix shape: {X_df.shape}")
    return all_subject_design_matrices


def run_trf(all_subject_design_matrices, eeg_clean_list_masked):
    all_subject_rvals = {}
    all_subject_crossvals = {}
    all_subject_predictions = {}
    all_subject_weights = {}
    all_subject_vifs = {}
    all_subject_preds = {}

    for sub in subs:
        print(f"\n[INFO] Running TRF for {sub}...")
        X_raw = all_subject_design_matrices[sub]
        vif_final = X_df(X_raw.values, X_raw.columns)
        print(f"{sub} predictors: {list(X_raw.columns)}")

        actual_cols_used = X_raw.columns
        all_subject_vifs[sub] = vif_final

        eeg = eeg_clean_list_masked[sub]
        min_len = min(len(eeg[-1]), len(X_raw))
        X_clean = X_raw[:min_len]
        eeg = eeg[:, :min_len]

        n_samples = sfreq * 60
        X_folds = []
        Y_folds = []

        for start in range(0, len(X_clean), n_samples):
            end = min(start + n_samples, len(X_clean))
            X_folds.append(X_clean.values[start:end])
            Y_folds.append(eeg[:, start:end].T)  # Now shape is (n_samples, n_channels)


        # Check if last fold is too short
        min_required_samples = int((1.0 - -0.1) * sfreq)

        if len(Y_folds[-1][1]) < min_required_samples:
            print(f"[Info] Last fold too short ({len(Y_folds[-1][1])} samples). Merging with previous.")

            # Merge last with second-last
            X_folds[-2] = np.vstack([X_folds[-2], X_folds[-1]])
            Y_folds[-2] = np.vstack([Y_folds[-2], Y_folds[-1]])  # shape: (n_samples_total, n_channels)

            # Remove the last fold
            X_folds = X_folds[:-1]
            Y_folds = Y_folds[:-1]

        import random
        random.seed(42)

        trf = TRF(direction=1)
        trf.train(X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=best_lambda, seed=42)
        print(trf.weights.shape)

        predictions = []
        r_vals = []

        for i in tqdm(range(len(X_folds))):
            X_chunk = X_folds[i]
            Y_chunk = Y_folds[i]

            pred_chunk, r_chunk = trf.predict(X_chunk, Y_chunk)
            predictions.append(pred_chunk[0])
            r_vals.append(r_chunk)

        predicted_full = np.concatenate(predictions, axis=0)
        predicted_avg = predicted_full.squeeze()  # shape: (7500,)

        r_mean = np.mean(r_vals, axis=0)
        print("Avg r across all chunks:", np.round(r_mean, 3))

        # Cross-validation
        n_total = len(X_folds)
        n_split = n_total // 3

        X_subsamples = [X_folds[:n_split], X_folds[n_split:2 * n_split], X_folds[2 * n_split:]]
        Y_subsamples = [Y_folds[:n_split], Y_folds[n_split:2 * n_split], Y_folds[2 * n_split:]]

        r_crossvals = []
        for x_sub, y_sub in zip(X_subsamples, Y_subsamples):
            r_crossval = crossval(
                trf, x_sub, y_sub, fs=sfreq,
                tmin=-0.1, tmax=1.0, regularization=best_lambda
            )
            r_crossvals.append(r_crossval)

        r_crossval_mean = np.mean(r_crossvals)
        print("Avg r_crossval across all chunks:", np.round(r_crossval_mean, 3))

        all_subject_predictions[sub] = predicted_avg
        all_subject_rvals[sub] = r_mean
        all_subject_crossvals[sub] = r_crossval_mean
        all_subject_weights[sub] = trf.weights
        all_subject_preds[sub] = list(actual_cols_used)
        print(f'Subject {sub} completed.')

    return (
        all_subject_predictions,
        all_subject_rvals,
        all_subject_crossvals,
        all_subject_weights,
        all_subject_vifs,
        all_subject_preds
    )


def save_and_plot(all_subject_weights, all_subject_predictions, all_subject_rvals, all_subject_crossvals,
                  all_subject_vifs, all_subject_preds, cond):
    for sub in subs:
        weights = all_subject_weights[sub]  # shape: (n_predictors, n_lags, n_channels)
        predicted_full = all_subject_predictions[sub]  # shape: (n_samples, n_channels)
        r_mean = all_subject_rvals[sub]
        r_crossval_mean = all_subject_crossvals[sub]
        vif_final = all_subject_vifs[sub]
        preds = all_subject_preds[sub]

        time_lags = np.linspace(-0.1, 1.0, weights.shape[1])  # time axis
        lag_mask = (time_lags >= 0.0) & (time_lags <= 1.0)
        time_lags_trimmed = time_lags[lag_mask]

        save_path = default_path / f'data/eeg/trf/trf_testing/composite_model/{plane}/{cond}/{folder_type}'
        save_path.mkdir(parents=True, exist_ok=True)
        data_path = save_path / 'data'
        data_path.mkdir(parents=True, exist_ok=True)

        # Save everything
        np.savez(
            data_path / f'{sub}_{plane}_{folder_type}_both_streams_TRF_results.npz',
            vif=vif_final,
            results=predicted_full,
            preds=preds,
            weights=weights,
            r=r_mean,
            r_crossval=r_crossval_mean,
            best_lambda=best_lambda,
            time_lags=time_lags,
            time_lags_trimmed=time_lags_trimmed,
            condition=plane
        )

        # Plot TRF per predictor
        # Plot TRF per predictor (averaging over channels)
        for i, name in enumerate(preds):
            plt.figure(figsize=(8, 4))

            # Average across channels if >1 channel
            if weights.shape[2] > 1:
                trf_weights = weights[i, :, :].mean(axis=1)  # (n_lags,)
            else:
                trf_weights = weights[i, :, 0]  # (n_lags,)

            trf_weights = trf_weights[lag_mask]  # Apply time lag mask

            # Smooth
            window_len = 11
            hamming_win = np.hamming(window_len)
            hamming_win /= hamming_win.sum()
            smoothed_weights = np.convolve(trf_weights, hamming_win, mode='same')

            # Plot
            plt.plot(time_lags_trimmed, smoothed_weights, alpha=0.8)
            plt.title(f'TRF for {name} - Both Streams')
            plt.xlabel('Time lag (s)')
            plt.ylabel('Amplitude (a.u.)')
            plt.plot([], [], ' ', label=f'λ = {best_lambda:.2f}, r = {r_crossval_mean:.2f}')
            plt.legend(loc='upper right', fontsize=8)
            plt.tight_layout()
            plt.savefig(save_path / f'{sub}_{folder_type}_{name}.png', dpi=300)
            plt.close()

def plot_avg_psd_predictions(all_subject_predictions, sfreq=125, title='', show_individuals=False):
    """
    Plots the average PSD across subjects from predicted EEG, limited to 1–8 Hz.

    Parameters:
    - all_subject_predictions: dict {sub_id: 1D np.array}, predicted EEG per subject
    - sfreq: float, sampling frequency
    - title: str, title for the plot
    - show_individuals: bool, whether to plot individual subject PSDs
    """
    fft_all_subjects = []

    # Determine minimum time length across all subjects
    min_len = min(pred.shape[0] for pred in all_subject_predictions.values() if pred.ndim == 2)
    if min_len < 2:
        print("Too short time series for FFT.")
        return

    for sub, pred in all_subject_predictions.items():
        if pred.ndim != 2:
            print(f"[Skipping] {sub}: expected 2D prediction (time x channels), got shape {pred.shape}")
            continue

        pred_trimmed = pred[:min_len, :]  # Shape: (time, channels)

        # FFT along time axis for each channel
        fft_vals = np.abs(fft(pred_trimmed, axis=0)) ** 2  # Power spectrum per channel
        psd = fft_vals[:min_len // 2, :]  # Take positive freqs only

        # Average PSD across channels
        psd_mean = psd.mean(axis=1)  # Shape: (n_freqs,)
        fft_all_subjects.append(psd_mean)

    if not fft_all_subjects:
        print("No valid PSD data to plot.")
        return

    fft_matrix = np.stack(fft_all_subjects)  # Shape: (n_subjects, n_freqs)
    freqs = fftfreq(min_len, d=1 / sfreq)[:min_len // 2]

    # Limit to 1–8 Hz
    freq_mask = (freqs >= 1) & (freqs <= 8)
    freqs_limited = freqs[freq_mask]
    fft_matrix_limited = fft_matrix[:, freq_mask]

    mean_psd = fft_matrix_limited.mean(axis=0)
    sem_psd = fft_matrix_limited.std(axis=0, ddof=1) / np.sqrt(fft_matrix_limited.shape[0])
    peak_idx = np.argmax(mean_psd)

    plt.figure(figsize=(10, 5))

    if show_individuals:
        for psd in fft_matrix_limited:
            plt.plot(freqs_limited, psd, alpha=0.3, color='gray')

    plt.plot(freqs_limited, mean_psd, color='blue', label='Mean PSD')
    plt.fill_between(freqs_limited, mean_psd - sem_psd, mean_psd + sem_psd, alpha=0.3, color='blue')

    # Annotate peak
    plt.scatter(freqs_limited[peak_idx], mean_psd[peak_idx], color='red', zorder=5)
    plt.text(freqs_limited[peak_idx] + 0.1, mean_psd[peak_idx] + 0.05,
             f"Peak: {freqs_limited[peak_idx]:.2f} Hz\nPower = {mean_psd[peak_idx]:.2f}",
             fontsize=9, color='red')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (a.u.)')
    plt.title(f'Average PSD of Predicted EEG {title} (1–8 Hz)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_crossval_r_distribution(crossval_dict, title='Cross-validated r Distribution', color='cornflowerblue'):
    """
    Plots the distribution of r values across subjects from a cross-validation result.

    Parameters:
    - crossval_dict: dict {sub_id: r_value}, correlation results per subject
    - title: str, title of the plot
    - color: str, matplotlib color for histogram and stats
    """
    r_vals = np.array(list(crossval_dict.values()))
    mean_r = np.mean(r_vals)
    std_r = np.std(r_vals, ddof=1)

    print(f"Mean r: {mean_r:.4f}")
    print(f"SD r:   {std_r:.4f}")

    plt.figure(figsize=(8, 5))
    plt.hist(r_vals, bins=10, color=color, edgecolor='black', alpha=0.75)
    plt.axvline(mean_r, color='red', linestyle='--', label=f'Mean = {mean_r:.3f}')
    plt.axvline(mean_r + std_r, color='gray', linestyle=':', label=f'±1 SD = {std_r:.3f}')
    plt.axvline(mean_r - std_r, color='gray', linestyle=':')

    plt.title(title)
    plt.xlabel('Cross-validated r')
    plt.ylabel('Number of Subjects')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

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

    plane = 'azimuth'
    if plane == 'azimuth':
        target1 = 'Right'
        target2 = 'Left'
        condition1 = 'a1'
        condition2 = 'a2'
    elif plane == 'elevation':
        target1 = 'Bottom'
        target2 = 'Top'
        condition1 = 'e1'
        condition2 = 'e2'

    eeg_files1 = get_eeg_files(condition=condition1)
    eeg_files2 = get_eeg_files(condition=condition2)


    eeg_concat_list1 = pick_channels(eeg_files1)
    eeg_concat_list2 = pick_channels(eeg_files2)

    eeg_clean_list_masked1, eeg_masked_list1 = mask_bad_segmets(eeg_concat_list1, condition=condition1)
    eeg_clean_list_masked2, eeg_masked_list2 = mask_bad_segmets(eeg_concat_list2, condition=condition2)

    predictors_list = ['binary_weights', 'envelopes']
    pred_types = ['onsets', 'envelopes']

    folder_types = ['all_stims', 'target_nums', 'non_targets', 'targets_x_deviants']

    if stream_type1 == 'stream1':
        folder_type = folder_types[0]
    elif stream_type1 == 'targets':
        folder_type = folder_types[1]
    elif stream_type1 == 'nt_target':
        folder_type = folder_types[2]
    elif stream_type2 == 'deviants':
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
        4.0: 1,
        3.0: 1,
        2.0: 1,
        1.0: 1
    }


    for predictor_name, pred_type in zip(predictors_list, pred_types):
        predictor = default_path/ f'data/eeg/predictors/{predictor_name}'
        predictor_dict1 = get_predictor_dict(condition=condition1, pred_type=pred_type)
        predictor_dict2 = get_predictor_dict(condition=condition2, pred_type=pred_type)
        predictor_dict_masked1, predictor_dict_masked_raw1 = predictor_mask_bads(predictor_dict1, condition=condition1, predictor_name=pred_type)
        predictor_dict_masked2, predictor_dict_masked_raw2 = predictor_mask_bads(predictor_dict2, condition=condition2, predictor_name=pred_type)

        # Remap onsets (semantic weights) before storing
        if pred_type == 'onsets':
            predictor_dict_masked1 = remap_onsets_nested(predictor_dict_masked1, semantic_mapping)
            predictor_dict_masked2 = remap_onsets_nested(predictor_dict_masked2, semantic_mapping)
            predictor_dict_masked_raw1 =  remap_onsets_nested(predictor_dict_masked_raw1, semantic_mapping)
            predictor_dict_masked_raw2  = remap_onsets_nested(predictor_dict_masked_raw2, semantic_mapping)


        s1_predictors[pred_type] = predictor_dict_masked1
        s1_predictors_raw[pred_type] = predictor_dict_masked_raw1
        s2_predictors[pred_type] = predictor_dict_masked2
        s2_predictors_raw[pred_type] = predictor_dict_masked_raw2


    s1_predictors, s2_predictors = define_streams_dict(s1_predictors, s2_predictors)
    s1_predictors_raw, s2_predictors_raw = define_streams_dict(s1_predictors_raw, s2_predictors_raw)

    from collections import defaultdict
    subs = list(s1_predictors['onsets'].keys())
    # Define predictor order and get subject list
    
    # condition 1:
    all_subject_design_matrices1 = design_matrices(s1_predictors)
    
    all_subject_predictions1, all_subject_rvals1, all_subject_crossvals1, all_subject_weights1, all_subject_vifs1, all_subject_preds1 = (
        run_trf(all_subject_design_matrices1, eeg_clean_list_masked1))

    save_and_plot(all_subject_weights1, all_subject_predictions1, all_subject_rvals1, all_subject_crossvals1,
                  all_subject_vifs1, all_subject_preds1, cond=condition1)


    # condition 2:
    all_subject_design_matrices2 = design_matrices(s2_predictors)

    all_subject_predictions2, all_subject_rvals2, all_subject_crossvals2, all_subject_weights2, all_subject_vifs2, subject_preds2 = (
        run_trf(all_subject_design_matrices2, eeg_clean_list_masked2))

    save_and_plot(all_subject_weights2, all_subject_predictions2, all_subject_rvals2, all_subject_crossvals2,
                  all_subject_vifs2,subject_preds2, cond=condition2)

    plot_avg_psd_predictions(all_subject_predictions1, sfreq=125, title=f'(Composite Model Target {target1})', show_individuals=False)
    plot_avg_psd_predictions(all_subject_predictions2, sfreq=125, title=f'(Separate Model Target {target2})', show_individuals=False)

    plot_crossval_r_distribution(all_subject_crossvals1, title=f'Target {target1} Cross-Validation R')
    plot_crossval_r_distribution(all_subject_crossvals2, title=f'Target {target2} Cross-Validation R')

