import copy
# 1: Import Libraries
# for defining directories and loading/saving
import os
from pathlib import Path
import pickle as pkl

# for designing the matrix
import numpy as np
import pandas as pd
import mne
from mtrf import TRF
from mtrf.stats import crossval
import random

# troubleshooting
import logging
from copy import deepcopy

# analysis
from scipy.stats import ttest_rel, wilcoxon, shapiro
import statsmodels.stats.multitest as smm
from mne.stats import permutation_cluster_test
from scipy.stats import zscore

# for plotting
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')


# vif function:
def matrix_vif(matrix):
    X = sm.add_constant(matrix)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    print(vif)
    return vif


def get_weight_avg(smoothed_weights, n, ch_mask):
    weights = smoothed_weights[n, :, ch_mask]
    if weights.size > 0:
        weights_avg = np.mean(weights, axis=0)
    else:
        # fallback: either skip subject or use all channels
        weights_avg = np.mean(smoothed_weights[n, :, :], axis=1)
        print(f"Warning: {sub} had no significant channels, using all channels instead.")
    return weights_avg


def run_model(X_folds, Y_folds, sub_list):
    if condition in ['a1', 'a2']:
        X_folds_filt = X_folds[6:]  # only filter for the conditions that are affected
        Y_folds_filt = Y_folds[6:]
        sub_list = sub_list[6:]
    else:
        X_folds_filt = X_folds
        Y_folds_filt = Y_folds
        sub_list = sub_list

    predictions_dict = {}
    for sub, pred_fold, eeg_fold in zip(sub_list, X_folds_filt, Y_folds_filt):
        trf = TRF(direction=1, method='ridge')  # forward model
        trf.train(stimulus=pred_fold, response=eeg_fold, fs=sfreq, tmin=tmin, tmax=tmax, regularization=best_lambda, average=True, seed=42)
        # Do I want one TRF across all the data? → average=True
        predictions, r = trf.predict(stimulus=pred_fold, response=eeg_fold, average=True)
        weights = trf.weights
        predictions_dict[sub] = {'predictions': predictions, 'r': r, 'weights': weights}

    time = trf.times

    return time, predictions_dict


def get_pred_idx(stream):
    phoneme_idx = np.where(col_names == f'phonemes_{stream}')[0][0]
    onset_idx = np.where(col_names == f'onsets_{stream}')[0][0]
    env_idx = np.where(col_names == f'envelopes_{stream}')[0][0]
    alpha_idx = np.where(col_names == f'alpha')[0][0]
    if stream == 'target':
        response_idx = np.where(col_names == f'responses_{stream}')[0][0]
        return phoneme_idx, onset_idx, env_idx, response_idx, alpha_idx
    else:
        return phoneme_idx, onset_idx, env_idx, alpha_idx


def extract_trfs(predictions_dict_updated, stream=''):
    phoneme_trfs = {}
    onset_trfs = {}
    env_trfs = {}
    response_trfs = {}
    alpha_trfs = {}
    for sub, rows in predictions_dict_updated.items():
        # predictions = rows['predictions']
        weights = rows['weights']

        # smooth weights across channels and predictors:
        window_len = 11
        hamming_win = np.hamming(window_len)
        hamming_win /= hamming_win.sum()

        # smooth along timepoints axis (axis=1)
        smoothed_weights = np.empty_like(weights)
        for p in range(weights.shape[0]):        # predictors
            for ch in range(weights.shape[2]):   # channels
                smoothed_weights[p, :, ch] = np.convolve(
                    weights[p, :, ch], hamming_win, mode='same'
                )
        if stream == 'target':
            phoneme_idx, onset_idx, env_idx, response_idx, alpha_idx = get_pred_idx(stream)
            response_avg = get_weight_avg(smoothed_weights, response_idx, ch_mask)
            response_trfs[sub] = response_avg
        else:
            phoneme_idx, onset_idx, env_idx, alpha_idx = get_pred_idx(stream)

        # common predictors for both target and distractor
        phoneme_avg = get_weight_avg(smoothed_weights, phoneme_idx, ch_mask)
        onset_avg = get_weight_avg(smoothed_weights, onset_idx, ch_mask)
        env_avg = get_weight_avg(smoothed_weights, env_idx, ch_mask)
        alpha_avg = get_weight_avg(smoothed_weights, alpha_idx, ch_mask)

        phoneme_trfs[sub] = phoneme_avg
        onset_trfs[sub] = onset_avg
        env_trfs[sub] = env_avg
        alpha_trfs[sub] = alpha_avg
    return phoneme_trfs, onset_trfs, env_trfs, response_trfs, alpha_trfs


def cluster_perm(target_trfs, distractor_trfs, predictor):
    # stack into arrays
    target_data = np.vstack(list(target_trfs.values()))     # shape (n_subjects, n_times)
    distractor_data = np.vstack(list(distractor_trfs.values()))

    target_std = np.std(target_data, axis=0)
    target_mean = np.mean(target_data, axis=0)
    distractor_std = np.std(distractor_data, axis=0)
    distractor_mean = np.mean(distractor_data, axis=0)
    target_sem = target_std / np.sqrt(len(sub_list))
    distractor_sem = distractor_std / np.sqrt(len(sub_list))

    # run cluster permutation test
    X = [target_data, distractor_data]
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(X, n_permutations=10000, tail=1, n_jobs=1)

    # plot grand averages
    plt.plot(time, target_data.mean(axis=0), 'b-', linewidth=2, label='Target')
    plt.fill_between(time,
                     target_mean - target_sem,
                     target_mean + target_sem,
                     color='b', alpha=0.3)
    plt.plot(time, distractor_data.mean(axis=0), 'r-', linewidth=2, label='Distractor')
    plt.fill_between(time,
                     distractor_mean - distractor_sem,
                     distractor_mean + distractor_sem,
                     color='r', alpha=0.3)

    # highlight significant clusters
    for cl, pval in zip(clusters, cluster_p_values):
        if pval < 0.05:
            time_inds = cl[0]
            plt.axvspan(time[time_inds[0]], time[time_inds[-1]],
                        color='gray', alpha=0.3)

    plt.title(f'TRF Comparison - {condition} - {predictor}')
    plt.xlim([time[0], 0.6])
    plt.legend()
    fig_path = data_dir / 'journal' / 'figures' / 'TRF' / condition / stim_type
    fig_path.mkdir(parents=True, exist_ok=True)
    filename = f'{predictor}_{stim_type}_{condition}.png'
    plt.savefig(fig_path/filename, dpi=300)
    plt.show()
    plt.close()


def get_components(arr, components):
    """Return mean amplitude per component window as a dict."""
    res = {}
    for name, (start, end) in components.items():
        res[name] = arr[start:end].mean()
    return res


def compare_time_windows(target_trfs, distractor_trfs):
    """
    Compare subject-wise TRF responses between target and distractor.
    Input: predictions_dict: dictionary with the predictions and r-values of the
    composite model + weights of each predictor (target & distractor included)
    Goal: cluster target and distractor TRF responses of each sub, separate into time-window
    components: P1 (0-50ms), N1(50-150), P2(150-250), N2(250-400), late (400-600)
    - Run paired t-test  - across subjects, in the diff time-windows with then FDR correction applied

    """

    results = {comp: {'target': [], 'distractor': []} for comp in components.keys()}

    for sub in target_trfs.keys():
        target_arr = target_trfs[sub]
        distractor_arr = distractor_trfs[sub]

        target_vals = get_components(target_arr, components)
        distractor_vals = get_components(distractor_arr, components)

        for comp in components.keys():
            results[comp]['target'].append(target_vals[comp])
            results[comp]['distractor'].append(distractor_vals[comp])

    all_comps = list(components.keys())
    all_p = []
    stats = {}
    for comp in all_comps:
        t_vals = results[comp]['target']
        d_vals = results[comp]['distractor']
        # normality test:
        _, target_p = shapiro(t_vals)
        _, distractor_p = shapiro(d_vals)
        if target_p and distractor_p > 0.05:
            # normally distributed:
            print('Data is normally distributed, running t-test')
            t_stat, p_val = ttest_rel(t_vals, d_vals)
            stats[comp] = (t_stat, p_val)
            all_p.append(p_val)
        else:
            print('Data non-parametric, runnig Wilcoxon test.')
            t_stat, p_val = wilcoxon(x=t_vals, y=d_vals, zero_method='wilcox', alternative='two-sided')
            stats[comp] = (t_stat, p_val)
            all_p.append(p_val)

    # FDR correction
    reject, p_fdr, _, _ = smm.multipletests(all_p, method='fdr_bh')

    for comp, (t_stat, p_val), p_corr, sig in zip(all_comps, stats.values(), p_fdr, reject):
        print(f"{comp}: t={t_stat:.2f}, p={p_val:.3f}, FDR={p_corr:.3f}, sig={sig}")
    return stats, p_fdr, all_p


def detect_trf_outliers(predictions_dict, method="zscore", threshold=3.0):
    # stack data
    data_list = []
    for sub in predictions_dict.keys():
        data = predictions_dict[sub]['r']
        data_list.append(data)

    outliers = {}

    subs = list(predictions_dict.keys())
    if method == "zscore":
        data_z = zscore(data_list)
        for i, sub in enumerate(subs):
            outliers[sub] = np.abs(data_z[i]) > threshold

    elif method == "iqr":
        def iqr_outlier_flags(values):
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            return (values < lower) | (values > upper)

        data_flags = iqr_outlier_flags(data_list)
        for i, sub in enumerate(subs):
            outliers[sub] = data_flags[i]

    # print results
    print(f"\n=== Outlier detection ({method}) ===")
    for sub, flags in outliers.items():
        if outliers[sub]:
            print(f"Sub {sub} flagged: {flags}. Subject removed from further analysis.")
            predictions_dict_updated = predictions_dict.copy()
            predictions_dict_updated.pop(sub, None)  # removes key 'sub' if it exists

    # quick scatterplot
    plt.figure()
    plt.plot(data_list, c='k')
    for i, (sub, val) in enumerate(zip(subs, data_list)):
        plt.text(i, val + 0.02, sub, ha='center', va='bottom', fontsize=9, color='red')
    plt.xlabel("Subjects r")
    plt.title(f"Outlier check")
    plt.show()

    return outliers, predictions_dict_updated


if __name__ == '__main__':

    stim_type = 'target_nums'
    all_trfs = {}
    for condition in ['a1', 'a2', 'e1', 'e2']:
        trfs_dict = {}

        # directories:
        base_dir = Path.cwd()
        data_dir = base_dir / 'data' / 'eeg'
        predictor_dir = data_dir / 'predictors'
        bad_segments_dir = predictor_dir / 'bad_segments'
        eeg_dir = Path('D:/VKK_Attention/data/eeg/preprocessed/results')
        alpha_dir = data_dir / 'journal' / 'alpha' / condition

        # load matrices of chosen stimulus and condition:
        dict_dir = data_dir / 'journal' / 'TRF' / 'matrix' / condition / stim_type
        with open(dict_dir / f'{condition}_matrix_target.pkl', 'rb') as f:
            target_dict = pkl.load(f)

        with open(dict_dir / f'{condition}_matrix_distractor.pkl', 'rb') as f:
            distractor_dict = pkl.load(f)

        sub_list = list(distractor_dict.keys())

        # load alpha:
        for files in alpha_dir.iterdir():
            if condition in files.name:
                with open(files, 'rb') as f:
                    alpha_dict = pkl.load(f)
        # keep occ_alpha:
        alpha_arrays = {}
        for sub, rows in alpha_dict.items():
            alpha_arr = rows['occ_alpha']
            if sub in sub_list:
                alpha_arrays[sub] = alpha_arr

        X_folds = []
        Y_folds = []
        # Stack predictors for the target stream
        for sub, target_data, distractor_data, alpha_arr in zip(sub_list, target_dict.values(), distractor_dict.values(),
                                                                alpha_arrays.values()):
            eeg = target_data['eeg']

            X_target = np.column_stack(
                [target_data['onsets'], target_data['envelopes'], target_data['phonemes'], target_data['responses']])
            X_distractor = np.column_stack(
                [distractor_data['onsets'], distractor_data['envelopes'], distractor_data['phonemes']])

            Y_eeg = eeg
            print("X_target shape:", X_target.shape)
            print("X_distractor shape:", X_distractor.shape)
            print("EEG shape:", Y_eeg.shape)
            # checking collinearity:
            import statsmodels.api as sm
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            # Build combined DataFrame
            col_names_target = ['onsets_target', 'envelopes_target', 'phonemes_target', 'responses_target']
            col_names_distr = ['onsets_distractor', 'envelopes_distractor', 'phonemes_distractor']

            # Build combined DataFrame
            X = pd.DataFrame(
                np.column_stack([X_target, X_distractor, alpha_arr]),
                columns=col_names_target + col_names_distr + ['alpha'])

            # Add constant for VIF calculation
            vif = matrix_vif(X)

            # split into trials:
            predictors_stacked = X.values  # ← ready for modeling
            X_folds.append(predictors_stacked)
            Y_folds.append(Y_eeg)

        col_names = np.array(X.columns)

        random.seed(42)

        best_lambda = 0.01

        tmin = - 0.1
        tmax = 1.0
        sfreq = 125

        threshold = 0.1  # e.g., keep channels with r >= 0.05

        all_ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
                  'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10',
                  'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1',
                  'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                  'FCz']

        common_roi = np.array(['Fp1', 'F3', 'Fz', 'F4', 'FC1', 'C3', 'Cz', 'C4', 'C1', 'C2', 'CP5',
                               'CP1', 'CP3', 'P7', 'P3', 'Pz', 'P4', 'P5', 'P1', 'F1', 'F2', 'AF3', 'FCz'])
        # these do be the electrodes that have high r vals in all subs and conditions

        ch_mask = np.isin(all_ch, common_roi)

        time, predictions_dict = run_model(X_folds, Y_folds, sub_list)

        # outliers, predictions_dict_updated = detect_trf_outliers(predictions_dict, method="iqr", threshold=3.0)

        target_phoneme_trfs, target_onset_trfs, target_env_trfs, target_response_trfs, alpha_trfs = \
            extract_trfs(predictions_dict, stream='target')

        distractor_phoneme_trfs, distractor_onset_trfs, distractor_env_trfs, _, _ = \
            extract_trfs(predictions_dict, stream='distractor')

        cluster_perm(target_phoneme_trfs, distractor_phoneme_trfs, predictor='phonemes')
        cluster_perm(target_onset_trfs, distractor_onset_trfs, predictor='onsets')
        cluster_perm(target_env_trfs, distractor_env_trfs, predictor='envelopes')
        # skip responses and alpha nuisance

        # define windows in ms
        win_defs = {
            'P1': (0, 50),
            'N1': (50, 150),
            'P2': (150, 250),
            'N2': (250, 400),
            'Late': (400, 600)
        }

        # convert to sample indices
        components = {}
        for name, (tmin, tmax) in win_defs.items():
            comp = {name: (int(tmin * sfreq / 1000),
                           int(tmax * sfreq / 1000))}
            components.update(comp)

        onset_stats, onset_p_fdr, onset_all_p = compare_time_windows(target_onset_trfs, distractor_onset_trfs)
        env_stats, env_p_fdr, env_all_p = compare_time_windows(target_env_trfs, distractor_env_trfs)
        phoneme_stats, phoneme_p_fdr, phoneme_all_p = compare_time_windows(target_phoneme_trfs, distractor_phoneme_trfs)

