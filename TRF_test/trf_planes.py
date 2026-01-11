
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
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
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
import seaborn as sns
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12


def run_model(X_folds, Y_folds, sub_list):
    predictions_dict = {}
    time = None
    for sub, pred_fold, eeg_fold in zip(sub_list, X_folds, Y_folds):
        trf = TRF(direction=1, method='ridge')  # forward model
        trf.train(stimulus=pred_fold, response=eeg_fold, fs=sfreq, tmin=tmin, tmax=tmax, regularization=best_lambda, average=True, seed=42)
        # Do I want one TRF across all the data? → average=True
        predictions, r = trf.predict(stimulus=pred_fold, response=eeg_fold, average=False)
        weights = trf.weights
        predictions_dict[sub] = {'predictions': predictions, 'r': r, 'weights': weights}
        if time is None:
            time = trf.times

    return time, predictions_dict


def get_pred_idx(stream):
    phoneme_idx = np.where(col_names == f'phonemes_{stream}')[0][0]
    env_idx = np.where(col_names == f'envelopes_{stream}')[0][0]
    # onset_idx = np.where(col_names == f'onsets_{stream}')[0][0]
    if stream in ['target', 'target_x_ele']:
        response_idx = np.where(col_names == f'responses_{stream}')[0][0]
        return phoneme_idx, env_idx, response_idx
    else:
        return phoneme_idx, env_idx


def get_weight_avg(smoothed_weights, n, masking):
    weights = smoothed_weights[n, :, masking]
    if weights.size > 0:
        weights_avg = np.mean(weights, axis=0)
    else:
        # fallback: either skip subject or use all channels
        weights_avg = np.mean(smoothed_weights[n, :, :], axis=1)
    return weights_avg


def extract_trfs(predictions_dict, stream='', plane_name='', ch_selection=None):
    phoneme_trfs = {}
    env_trfs = {}
    response_trfs = {}

    # map streams valid per plane
    # this makes sure we ALWAYS pick the correct predictor indices
    stream_map = {
        'azimuth': {
            'target': ('phonemes_target', 'envelopes_target', 'responses_target'),
            'distractor': ('phonemes_distractor', 'envelopes_distractor', None)},
        'elevation': {
            'target_x_ele': ('phonemes_target_x_ele', 'envelopes_target_x_ele', 'responses_target_x_ele'),
            'distractor_x_ele': ('phonemes_distractor_x_ele', 'envelopes_distractor_x_ele', None)}}

    if plane_name not in stream_map or stream not in stream_map[plane_name]:
        raise ValueError(f"Invalid combination plane={plane_name}, stream={stream}")

    phoneme_name, env_name, response_name = stream_map[plane_name][stream]

    # get predictor indices once
    phoneme_idx = np.where(col_names == phoneme_name)[0][0]
    env_idx = np.where(col_names == env_name)[0][0]
    response_idx = None
    if response_name is not None:
        response_idx = np.where(col_names == response_name)[0][0]

    # loop subjects
    for sub, rows in predictions_dict.items():

        weights = rows['weights']
        masking = np.isin(all_ch, ch_selection)

        # smooth weights
        window_len = 11
        hamming_win = np.hamming(window_len)
        hamming_win /= hamming_win.sum()

        smoothed_weights = np.empty_like(weights)
        for p in range(weights.shape[0]):
            for ch in range(weights.shape[2]):
                smoothed_weights[p, :, ch] = np.convolve(
                    weights[p, :, ch], hamming_win, mode='same'
                )

        # extract phoneme + envelope weights
        phoneme_avg = get_weight_avg(smoothed_weights, phoneme_idx, masking)
        env_avg = get_weight_avg(smoothed_weights, env_idx, masking)

        phoneme_trfs[sub] = phoneme_avg
        env_trfs[sub] = env_avg

        # extract response predictor only for target streams
        if response_idx is not None:
            response_avg = get_weight_avg(smoothed_weights, response_idx, masking)
            response_trfs[sub] = response_avg
    return phoneme_trfs, env_trfs, response_trfs


def cluster_effect_size(target_data, distractor_data, time, time_sel, cl):
    """
    Compute Cohen's dz and Hedges' gz for a given cluster.

    target_data, distractor_data : arrays (n_subjects, n_times)
    time : full time vector
    time_sel : the time points used in this component window
    cl : cluster indices relative to time_sel (from MNE)
    """
    # cluster indices relative to time_sel
    ti = cl[0]
    cluster_times = time_sel[ti]  # actual time values

    # build mask for the full time axis
    cluster_mask = np.isin(time, cluster_times)

    # subject-wise averages
    T_vals = target_data[:, cluster_mask].mean(axis=1)
    D_vals = distractor_data[:, cluster_mask].mean(axis=1)
    delta = T_vals - D_vals

    # effect sizes
    mean_diff = delta.mean()
    sd_diff = delta.std(ddof=1)
    dz = mean_diff / sd_diff
    n = len(delta)
    J = 1 - (3 / (4*n - 9))
    gz = J * dz

    return mean_diff, dz, gz


def count_non_zeros(X_folds_all, sub_list, phoneme_trfs, stream=''):
    if stream in ['target', 'target_x_ele']:
        phoneme_idx = 1
    else:
        phoneme_idx = -1

    phoneme_trfs_standardized = {}

    for sub_idx, sub_matrix in enumerate(X_folds_all):
        sub_name = sub_list[sub_idx]
        if sub_name not in phoneme_trfs:
            continue

        phoneme_weights = phoneme_trfs[sub_name].copy()
        phonemes = sub_matrix[:, phoneme_idx]
        p = np.count_nonzero(phonemes) / len(phonemes)
        std = np.sqrt(p * (1 - p))

        phoneme_weights *= std
        phoneme_trfs_standardized[sub_name] = phoneme_weights

    return phoneme_trfs_standardized


def compute_diff_waves(target_dict, distractor_dict):
    """
    Compute subject-wise difference waves: target - distractor.

    Both inputs are dicts: {sub: 1D array of TRF weights}.
    Returns a dict with the same structure for subs present in both.
    """
    diff = {}
    common_subs = set(target_dict.keys()).intersection(distractor_dict.keys())
    for sub in common_subs:
        diff[sub] = target_dict[sub] - distractor_dict[sub]
    return diff


def cluster_perm(az_trfs, ele_trfs, predictor):
    # stack into arrays (n_subjects, n_times)
    from mne.stats import fdr_correction
    az_data = np.vstack(list(az_trfs.values()))
    ele_data = np.vstack(list(ele_trfs.values()))

    # compute means/SEMs for plotting full time
    az_mean = az_data.mean(axis=0)
    ele_mean = ele_data.mean(axis=0)
    az_sem = az_data.std(axis=0) / np.sqrt(ele_data.shape[0])
    ele_sem = ele_data.std(axis=0) / np.sqrt(ele_data.shape[0])

    # plot full responses
    plt.plot(time, az_mean, 'b-', linewidth=2, label='Target - Azimuth')
    plt.fill_between(time, az_mean - az_sem, az_mean + az_sem,
                     color='b', alpha=0.3)
    plt.plot(time, ele_mean, 'r-', linewidth=2, label='Target - Elevation')
    plt.fill_between(time, ele_mean - ele_sem, ele_mean + ele_sem,
                     color='r', alpha=0.3)

    all_pvals = []
    all_clusters = []
    all_labels = []
    all_times = []

    # loop windows
    for comp, (tmin, tmax) in component_windows.items():
        tmask = (time >= tmin) & (time <= tmax)
        if not tmask.any():
            continue
        time_sel = time[tmask]
        X = [az_data[:, tmask], ele_data[:, tmask]]
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
            X, n_permutations=5000, tail=1, n_jobs=1)

        for cl, pval in zip(clusters, cluster_p_values):
            all_pvals.append(pval)
            all_labels.append(comp)
            all_clusters.append(cl)
            all_times.append(time_sel)

    # apply FDR once across all windows
    reject, pvals_fdr = fdr_correction(all_pvals, alpha=0.05)

    # highlight significant clusters after correction
    for comp, cl, pval, pval_corr, rej, time_sel in zip(
            all_labels, all_clusters, all_pvals, pvals_fdr, reject, all_times):
        if rej:
            ti = cl[0]  # time indices relative to time_sel
            mean_diff, dz, gz = cluster_effect_size(az_data, ele_data, time, time_sel, cl)
            plt.axvspan(time_sel[ti[0]], time_sel[ti[-1]],
                        color='gray', alpha=0.2)
            plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            t_start, t_end = time_sel[ti[0]], time_sel[ti[-1]]
            print(f"{comp}: {t_start * 1000:.0f}-{t_end * 1000:.0f} ms, g={gz:.3f}, pFDR={pval_corr:.3f}")

    # plt.title(f'TRF Comparison - {plane} - {predictor}')
    plt.xlim([time[0], 0.6])
    if predictor == 'phonemes':
        plt.ylim([-0.6, 0.7])
    else:
        plt.ylim([-1, 1.5])
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('TRF amplitude (a.u.)')
    sns.despine(top=True, right=True)
    # fig_path = data_dir / 'journal' / 'figures' / 'TRF' / plane / stim_type
    # fig_path.mkdir(parents=True, exist_ok=True)
    # filename = f'{predictor}_{stim_type}_{condition}_{roi_type}_roi.png'
    # plt.savefig(fig_path / filename, dpi=300)
    # plt.savefig(fig_path / f'{predictor}_{stim_type}_{condition}_{roi_type}_roi.pdf', dpi=300)
    plt.show()


if __name__ == '__main__':

    stim_type = 'non_targets'

    azimuth = ['a1', 'a2']
    elevation = ['e1', 'e2']
    all_conditions = azimuth + elevation

    base_dir = Path.cwd()
    data_dir = base_dir / 'data' / 'eeg'
    predictor_dir = data_dir / 'predictors'
    eeg_dir = Path('D:/VKK_Attention/data/eeg/preprocessed/results')

    # this will hold *all* trials from all conditions
    X_folds_all = []
    Y_folds_all = []

    # (optional) keep info about which plane each trial came from
    trial_plane_labels = []  # 'azimuth' or 'elevation'
    trial_condition_labels = []  # 'a1', 'a2', 'e1', 'e2'


    def build_plane_trf_design():
        # subjects to keep
        sub_list = [
            'sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19', 'sub20',
            'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29']

        # per-subject containers
        X_by_sub = {sub: [] for sub in sub_list}
        Y_by_sub = {sub: [] for sub in sub_list}

        trial_plane_labels = []
        trial_condition_labels = []

        for condition in all_conditions:  # ['a1','a2','e1','e2']
            dict_dir = data_dir / 'journal' / 'TRF' / 'matrix' / condition / stim_type

            # load stimulus matrices
            with open(dict_dir / f'{condition}_matrix_target.pkl', 'rb') as f:
                target_dict = pkl.load(f)
            with open(dict_dir / f'{condition}_matrix_distractor.pkl', 'rb') as f:
                distractor_dict = pkl.load(f)

            # restrict to subjects we care about
            target_dict = {k: v for k, v in target_dict.items() if k in sub_list}
            distractor_dict = {k: v for k, v in distractor_dict.items() if k in sub_list}

            for sub in sub_list:
                if sub not in target_dict:
                    continue  # subject missing in this condition

                target_data = target_dict[sub]
                distractor_data = distractor_dict[sub]

                eeg = target_data['eeg']  # (n_samples, n_channels) or (n_samples,)

                # target and distractor predictors
                X_target = np.column_stack([
                    target_data['onsets'],
                    target_data['envelopes'],
                    target_data['phonemes'],
                    target_data['responses']
                ])
                X_distractor = np.column_stack([
                    distractor_data['onsets'],
                    distractor_data['envelopes'],
                    distractor_data['phonemes']
                ])

                col_names_target = ['onsets_target', 'envelopes_target', 'phonemes_target', 'responses_target']
                col_names_distr = ['onsets_distractor', 'envelopes_distractor', 'phonemes_distractor']

                X_base = pd.DataFrame(
                    np.column_stack([X_target, X_distractor]),
                    columns=col_names_target + col_names_distr
                )

                # drop onsets if you don't want them
                X_base = X_base.drop(columns=['onsets_target', 'onsets_distractor'])

                # plane dummy
                plane_ele = 1.0 if condition in elevation else 0.0

                # interaction block (only non-zero for elevation)
                X_interaction = X_base.values * plane_ele
                inter_cols = [c + '_x_ele' for c in X_base.columns]

                X_full = pd.DataFrame(
                    np.column_stack([X_base.values, X_interaction]),
                    columns=list(X_base.columns) + inter_cols
                )

                # accumulate per subject
                X_by_sub[sub].append(X_full.values)
                Y_by_sub[sub].append(eeg)

                trial_plane_labels.append('azimuth' if condition in azimuth else 'elevation')
                trial_condition_labels.append(condition)

        # now stack across conditions for each subject
        X_folds_all = []
        Y_folds_all = []

        for sub in sub_list:
            if len(X_by_sub[sub]) == 0:
                continue
            X_folds_all.append(np.vstack(X_by_sub[sub]))
            Y_folds_all.append(np.vstack(Y_by_sub[sub]))

        # X_full only used to grab column names later
        example_sub = sub_list[0]
        X_full_example = pd.DataFrame(
            np.vstack(X_by_sub[example_sub]),
            columns=list(X_base.columns) + inter_cols)

        return X_folds_all, Y_folds_all, trial_plane_labels, trial_condition_labels, X_full_example, sub_list

    X_folds_all, Y_folds_all, trial_plane_labels, trial_condition_labels, X_full, sub_list = build_plane_trf_design()
    col_names = np.array(X_full.columns)

    random.seed(42)

    best_lambda = 0.01

    tmin = - 0.1
    tmax = 1.0
    sfreq = 125

    all_ch = np.array(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
              'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz',
              'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6',
              'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1',
              'Oz', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5',
              'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8',
              'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz',
              'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3',
              'POz', 'PO4', 'PO8', 'FCz'])

    sub_list = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19', 'sub20',
                'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29']

    time, predictions_dict = run_model(X_folds_all, Y_folds_all, sub_list)

    phoneme_roi = np.array(['F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                            'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8'])  # supposedly phoneme electrodes

    env_roi = np.array(['Cz'])

    component_windows = {
        "P1": (0.05, 0.15),  # early sensory
        "N1": (0.15, 0.25),  # robust first attention effects; frontocentral and temporal
        "P2": (0.25, 0.35),  # conflict monitoring / categorization of stimulus
        "N2": (0.35, 0.50)}  # late attention-driven decision making

    # phonemes - azimuth
    az_target_phoneme_trfs, _, _,\
         = extract_trfs(predictions_dict, stream='target', plane_name='azimuth', ch_selection=phoneme_roi)

    az_distractor_phoneme_trfs, _, _, \
         = extract_trfs(predictions_dict, stream='distractor', plane_name='azimuth', ch_selection=phoneme_roi)

    az_target_phoneme_trfs_standardized = count_non_zeros(X_folds_all,
                                                       sub_list, az_target_phoneme_trfs, stream='target')
    az_distractor_phoneme_trfs_standardized = count_non_zeros(X_folds_all,
                                                           sub_list, az_distractor_phoneme_trfs, stream='distractor')

    # phonemes - elevation
    ele_target_phoneme_trfs, _, _, \
        = extract_trfs(predictions_dict, stream='target_x_ele', plane_name='elevation', ch_selection=phoneme_roi)

    ele_distractor_phoneme_trfs, _, _, \
        = extract_trfs(predictions_dict, stream='distractor_x_ele', plane_name='elevation', ch_selection=phoneme_roi)

    ele_target_phoneme_trfs_standardized = count_non_zeros(X_folds_all,
                                                          sub_list, ele_target_phoneme_trfs, stream='target_x_ele')

    ele_distractor_phoneme_trfs_standardized = count_non_zeros(X_folds_all,
                                                              sub_list, ele_distractor_phoneme_trfs, stream='distractor_x_ele')

    # phoneme diff waves (target - distractor) per plane
    az_phoneme_diff_waves_standardized = compute_diff_waves(
        az_target_phoneme_trfs_standardized,
        az_distractor_phoneme_trfs_standardized)

    ele_diff_waves_phoneme_trfs_standardized = compute_diff_waves(
        ele_target_phoneme_trfs_standardized,
        ele_distractor_phoneme_trfs_standardized)

    cluster_perm(
        az_phoneme_diff_waves_standardized,
        ele_diff_waves_phoneme_trfs_standardized,
        predictor='phonemes')

    # repeat for envelopes
    _, az_target_env_trfs, _, \
        = extract_trfs(predictions_dict, stream='target', plane_name='azimuth', ch_selection=env_roi)

    _, ele_target_env_trfs, _, \
        = extract_trfs(predictions_dict, stream='target_x_ele', plane_name='elevation', ch_selection=env_roi)
    # distractor - envelopes
    _, az_distractor_env_trfs, _, \
        = extract_trfs(predictions_dict, stream='distractor', plane_name='azimuth', ch_selection=env_roi)
    _, ele_distractor_env_trfs, _, \
        = extract_trfs(predictions_dict, stream='distractor_x_ele', plane_name='elevation', ch_selection=env_roi)

    # envelope diff waves (target - distractor) per plane
    az_env_diff_waves = compute_diff_waves(
        az_target_env_trfs,
        az_distractor_env_trfs)
    ele_env_diff_waves = compute_diff_waves(
        ele_target_env_trfs,
        ele_distractor_env_trfs)

    # compare envelope diff waves across planes
    cluster_perm(
        az_env_diff_waves,
        ele_env_diff_waves,
        predictor='envelopes')