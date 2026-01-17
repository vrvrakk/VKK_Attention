
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


def get_pred_idx(stream, plane_name=''):
    phoneme_idx = np.where(col_names == f'{plane_name}_phonemes_{stream}')[0][0]
    env_idx = np.where(col_names == f'envelopes_{stream}')[0][0]
    if stream in ['target']:
        response_idx = np.where(col_names == f'{plane_name}_responses_{stream}')[0][0]
        return phoneme_idx, env_idx, response_idx
    else:
        return phoneme_idx, env_idx


def get_weight_avg(smoothed_weights, predictor_idx, ch_selection=None, all_ch=None):
    """
    smoothed_weights: array (n_predictors, n_times, n_channels)
    predictor_idx: index of predictor to extract
    ch_selection: list/array of channel names to keep (or None for all)
    all_ch: array of all channel names, same order as weights' 3rd dim
    """
    # shape: (n_times, n_channels)
    w = smoothed_weights[predictor_idx, :, :]

    if ch_selection is not None and all_ch is not None:
        mask = np.isin(all_ch, ch_selection)
        w = w[:, mask]

    # return as (n_channels, n_times) – nicer for MNE-style spatio-temporal data
    return np.transpose(w, (1, 0))  # (n_channels, n_times)


def extract_trfs(predictions_dict, stream='', plane_name='', ch_selection=None):
    phoneme_trfs = {}
    env_trfs = {}
    response_trfs = {}

    # map streams valid per plane
    stream_map = {
        'azimuth': {
            'target': ('azimuth_phonemes_target', 'azimuth_envelopes_target', 'azimuth_responses_target'),
            'distractor': ('azimuth_phonemes_distractor', 'azimuth_envelopes_distractor', None)},
        'elevation': {
            'target': ('elevation_phonemes_target', 'elevation_envelopes_target', 'elevation_responses_target'),
            'distractor': ('elevation_phonemes_distractor', 'elevation_envelopes_distractor', None)}
    }

    if plane_name not in stream_map or stream not in stream_map[plane_name]:
        raise ValueError(f"Invalid combination plane={plane_name}, stream={stream}")

    phoneme_name, env_name, response_name = stream_map[plane_name][stream]

    # get predictor indices once
    phoneme_idx = np.where(col_names == phoneme_name)[0][0]
    env_idx = np.where(col_names == env_name)[0][0]
    response_idx = None
    if response_name is not None:
        response_idx = np.where(col_names == response_name)[0][0]

    for sub, rows in predictions_dict.items():
        weights = rows['weights']  # (n_predictors, n_times, n_channels)
        n_pred, n_times, n_ch = weights.shape

        # smooth **all predictors and channels** over time
        window_len = 11
        hamming_win = np.hamming(window_len)
        hamming_win /= hamming_win.sum()

        smoothed_weights = np.empty_like(weights)
        for p in range(n_pred):
            for ch in range(n_ch):
                smoothed_weights[p, :, ch] = np.convolve(
                    weights[p, :, ch], hamming_win, mode='same'
                )

        # extract phoneme + envelope weights, keep ALL channels
        phoneme_trfs[sub] = get_weight_avg(
            smoothed_weights, phoneme_idx,
            ch_selection=ch_selection, all_ch=all_ch
        )  # (n_channels, n_times)

        env_trfs[sub] = get_weight_avg(
            smoothed_weights, env_idx,
            ch_selection=ch_selection, all_ch=all_ch
        )  # (n_channels, n_times)

        # response predictor only for target streams
        if response_idx is not None:
            response_trfs[sub] = get_weight_avg(
                smoothed_weights, response_idx,
                ch_selection=ch_selection, all_ch=all_ch
            )  # (n_channels, n_times)

    return phoneme_trfs, env_trfs, response_trfs

def cluster_effect_size(az_data, ele_data, full_mask):
    """
    az_data, ele_data: (n_subjects, n_channels, n_times)
    full_mask: boolean mask (n_channels, n_times) for the significant cluster
    """
    # Average within the cluster per subject
    T_vals = az_data[:, full_mask].mean(axis=1)
    D_vals = ele_data[:, full_mask].mean(axis=1)
    delta = T_vals - D_vals

    mean_diff = delta.mean()
    sd_diff = delta.std(ddof=1)
    dz = mean_diff / sd_diff
    n = len(delta)
    J = 1 - (3 / (4*n - 9))
    gz = J * dz
    return mean_diff, dz, gz

def count_non_zeros(X_folds_all, sub_list, phoneme_trfs, column=''):
    if 'azimuth_phonemes_target' in column:
        phoneme_idx = 1
    elif 'azimuth_phonemes_distractor' in column:
        phoneme_idx = 4
    elif 'elevation_phonemes_target' in column:
        phoneme_idx = 6
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


def compute_diff_waves(target_dict, distractor_dict, predictor=''):
    """
    Compute subject-wise difference waves: target - distractor.

    Both inputs are dicts: {sub: 1D array of TRF weights}.
    Returns a dict with the same structure for subs present in both.
    """
    diff = {}
    gfp = {}  # or rms of the diff
    common_subs = set(target_dict.keys()).intersection(distractor_dict.keys())

    for sub in common_subs:
        # 1. channel-wise difference (target – distractor), shape: (channels × time)
        diff[sub] = target_dict[sub][predictor] - distractor_dict[sub][predictor]
    return diff


from mne.stats import spatio_temporal_cluster_1samp_test, fdr_correction, combine_adjacency

def cluster_perm(az_trfs, ele_trfs, predictor, plane_name=''):
    """
    az_trfs, ele_trfs: dict[sub_id] -> array (n_channels, n_times)
    predictor: 'phonemes' or 'envelopes'
    plane_name: 'azimuth' or 'elevation' label for plotting/filenames
    """
    # ensure same subject order
    subs = sorted(set(az_trfs.keys()) & set(ele_trfs.keys()))
    if len(subs) == 0:
        raise ValueError("No overlapping subjects between az_trfs and ele_trfs")

    az_data = np.stack([az_trfs[s] for s in subs], axis=0)   # (n_sub, n_ch, n_times)
    ele_data = np.stack([ele_trfs[s] for s in subs], axis=0) # (n_sub, n_ch, n_times)

    n_sub, n_ch, n_times = az_data.shape

    # global channel-averaged TRFs for plotting (no stats yet)
    az_mean = az_data.mean(axis=(0, 1))   # (n_times,)
    ele_mean = ele_data.mean(axis=(0, 1)) # (n_times,)
    az_sem = az_data.mean(axis=1).std(axis=0) / np.sqrt(n_sub)
    ele_sem = ele_data.mean(axis=1).std(axis=0) / np.sqrt(n_sub)

    plt.figure(figsize=(7, 4))
    plt.plot(time, az_mean, linewidth=2, label=f'{plane_name} Target')
    plt.fill_between(time, az_mean - az_sem, az_mean + az_sem, alpha=0.3)

    plt.plot(time, ele_mean, linewidth=2, label=f'{plane_name} Distractor')
    plt.fill_between(time, ele_mean - ele_sem, ele_mean + ele_sem, alpha=0.3)

    all_pvals = []
    all_clusters = []
    all_labels = []
    all_masks = []

    # difference data for paired test
    diff_data = az_data - ele_data  # (n_sub, n_ch, n_times)

    # loop over your component windows (e.g. N1, P2, ...)
    for comp, (tmin, tmax) in component_windows.items():
        tmask = (time >= tmin) & (time <= tmax)
        if not tmask.any():
            continue

        # data in this window: (n_sub, n_ch, n_t_window)
        X = diff_data[:, :, tmask]
        _, _, n_t_win = X.shape

        # adjacency: channels x time
        # simple adjacency in time dimension; spatial adjacency is just "neighbors" in index
        adjacency = combine_adjacency(n_ch, n_t_win)

        T_obs, clusters, cluster_p_values, H0 = spatio_temporal_cluster_1samp_test(
            X, n_permutations=5000, tail=0, adjacency=adjacency,
            n_jobs=1, out_type='mask'
        )

        for cl, pval in zip(clusters, cluster_p_values):
            # cl is (n_ch, n_t_win); expand to full-time mask
            full_mask = np.zeros((n_ch, n_times), dtype=bool)
            full_mask[:, tmask] = cl

            all_pvals.append(pval)
            all_clusters.append(full_mask)
            all_labels.append(comp)
            all_masks.append(full_mask)

    # FDR across all clusters/windows
    if len(all_pvals) == 0:
        print("No clusters found in any window.")
        return

    reject, pvals_fdr = fdr_correction(all_pvals, alpha=0.05)

    # highlight significant clusters and report effect sizes
    for comp, full_mask, p_uncorr, p_corr, rej in zip(
            all_labels, all_clusters, all_pvals, pvals_fdr, reject):

        if not rej:
            continue

        # effect size
        mean_diff, dz, gz = cluster_effect_size(az_data, ele_data, full_mask)

        # time range of this cluster (any channel)
        time_mask_cluster = np.any(full_mask, axis=0)
        cluster_times = time[time_mask_cluster]
        t_start, t_end = cluster_times[0], cluster_times[-1]

        # shade cluster time span on the global plot
        plt.axvspan(t_start, t_end, color='gray', alpha=0.2)

        print(
            f"{comp}: {t_start*1000:.0f}-{t_end*1000:.0f} ms, "
            f"g={gz:.3f}, p_uncorr={p_uncorr:.4f}, pFDR={p_corr:.4f}"
        )

    plt.axvline(x=0, linestyle='--', linewidth=0.8, alpha=0.7)
    plt.xlim([time[0], 0.6])

    if predictor == 'phonemes':
        plt.ylim([-0.6, 0.7])
    elif predictor == 'envelopes' and stim_type in ['all', 'non_targets']:
        plt.ylim([-1, 1.5])
    elif predictor == 'envelopes' and stim_type == 'target_nums':
        plt.ylim([-12, 12])

    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('TRF amplitude (a.u.)')
    sns.despine(top=True, right=True)

    fig_path = data_dir / 'journal' / 'figures' / 'TRF' / 'ultimate_model' / stim_type
    fig_path.mkdir(parents=True, exist_ok=True)
    filename = f'{predictor}_{stim_type}_{plane_name}.png'
    plt.savefig(fig_path / filename, dpi=300)
    plt.savefig(fig_path / f'{predictor}_{stim_type}_{plane_name}_roi.pdf', dpi=300)
    plt.show()

def build_plane_trf_design():
    # subjects to keep
    sub_list = [
        'sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19', 'sub20',
        'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29']

    azimuth = ['a1', 'a2']
    elevation = ['e1', 'e2']
    all_conditions = azimuth + elevation

    # container for final per-subject design matrices and EEG
    X_folds_all = []
    Y_folds_all = []

    for sub in sub_list:
        az_X_blocks = []
        az_Y_blocks = []
        ele_X_blocks = []
        ele_Y_blocks = []

        # 1) build separate azimuth & elevation blocks for this subject
        for condition in all_conditions:
            print(sub, condition)
            dict_dir = data_dir / 'journal' / 'TRF' / 'matrix' / condition / stim_type

            # load stimulus matrices
            with open(dict_dir / f'{condition}_matrix_target.pkl', 'rb') as f:
                target_dict = pkl.load(f)
            with open(dict_dir / f'{condition}_matrix_distractor.pkl', 'rb') as f:
                distractor_dict = pkl.load(f)

            target_dict = {k: v for k, v in target_dict.items() if k in sub_list}
            distractor_dict = {k: v for k, v in distractor_dict.items() if k in sub_list}

            target_data = target_dict[sub]
            distractor_data = distractor_dict[sub]

            eeg = target_data['eeg']  # (n_samples, n_channels)

            # base predictors for this block
            env_t = np.asarray(target_data['envelopes']).ravel()
            phon_t = np.asarray(target_data['phonemes']).ravel()
            resp_t = np.asarray(target_data['responses']).ravel()
            env_d = np.asarray(distractor_data['envelopes']).ravel()
            phon_d = np.asarray(distractor_data['phonemes']).ravel()

            # 5-column block: [env_t, phon_t, resp_t, env_d, phon_d]
            X_block = np.column_stack([env_t, phon_t, resp_t, env_d, phon_d])

            if condition in azimuth:
                az_X_blocks.append(X_block)
                az_Y_blocks.append(eeg)
            elif condition in elevation:
                ele_X_blocks.append(X_block)
                ele_Y_blocks.append(eeg)

        # stack A1+A2 and E1+E2 per subject
        X_az = np.vstack(az_X_blocks)  # (N_az, 5)
        Y_az = np.vstack(az_Y_blocks)  # (N_az, n_channels)

        X_ele = np.vstack(ele_X_blocks)  # (N_ele, 5)
        Y_ele = np.vstack(ele_Y_blocks)  # (N_ele, n_channels)

        N_az = X_az.shape[0]
        N_ele = X_ele.shape[0]

        # --- 2) zero-padding structure you described ---

        # azimuth predictors active for az part, zero for elevation part
        az_block_full = np.vstack([
            np.hstack([X_az, np.zeros((N_az, 5))]),  # [az predictors, zeros] during azimuth EEG
            np.hstack([np.zeros((N_ele, 5)), np.zeros((N_ele, 5))])  # zeros during elevation EEG
        ])

        # elevation predictors active for ele part, zero for az part
        ele_block_full = np.vstack([
            np.hstack([np.zeros((N_az, 5)), np.zeros((N_az, 5))]),  # zeros during azimuth EEG
            np.hstack([np.zeros((N_ele, 5)), X_ele])  # [zeros, ele predictors] during elevation EEG
        ])

        # combine into final (N_az+N_ele, 10) design
        X_full = az_block_full + ele_block_full  # equivalent to stacking, but clearer logically

        # combine EEG in the same order: first azimuth then elevation
        Y_full = np.vstack([Y_az, Y_ele])

        X_folds_all.append(X_full)
        Y_folds_all.append(Y_full)

    # final column names (10 predictors)
    col_names = np.array([
        'azimuth_envelopes_target', 'azimuth_phonemes_target', 'azimuth_responses_target',
        'azimuth_envelopes_distractor', 'azimuth_phonemes_distractor',
        'elevation_envelopes_target', 'elevation_phonemes_target', 'elevation_responses_target',
        'elevation_envelopes_distractor', 'elevation_phonemes_distractor'])

    return X_folds_all, Y_folds_all, col_names, sub_list


if __name__ == '__main__':

    stim_type = 'target_nums'

    azimuth = ['a1', 'a2']
    elevation = ['e1', 'e2']
    all_conditions = azimuth + elevation

    base_dir = Path.cwd()
    data_dir = base_dir / 'data' / 'eeg'
    predictor_dir = data_dir / 'predictors'
    eeg_dir = Path('D:/VKK_Attention/data/eeg/preprocessed/results')

    X_folds_all, Y_folds_all, col_names, sub_list = build_plane_trf_design()

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
    #
    # phoneme_roi = np.array(['F3', 'F4', 'F5', 'F6', 'F7', 'F8',
    #                         'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8'])  # supposedly phoneme electrodes
    phoneme_roi = [ch for ch in list(all_ch) if not ch.startswith(('O', 'PO', 'P'))]

    env_roi = phoneme_roi

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
                                                       sub_list, az_target_phoneme_trfs, column='azimuth_phonemes_target')
    az_distractor_phoneme_trfs_standardized = count_non_zeros(X_folds_all,
                                                           sub_list, az_distractor_phoneme_trfs, column='azimuth_phonemes_distractor')

    # phonemes - elevation
    ele_target_phoneme_trfs, _, _, \
        = extract_trfs(predictions_dict, stream='target', plane_name='elevation', ch_selection=phoneme_roi)

    ele_distractor_phoneme_trfs, _, _, \
        = extract_trfs(predictions_dict, stream='distractor', plane_name='elevation', ch_selection=phoneme_roi)

    ele_target_phoneme_trfs_standardized = count_non_zeros(X_folds_all,
                                                          sub_list, ele_target_phoneme_trfs, column='elevation_phonemes_target')

    ele_distractor_phoneme_trfs_standardized = count_non_zeros(X_folds_all,
                                                              sub_list, ele_distractor_phoneme_trfs, column='elevation_phonemed_distractor')

    # first test phonemes target-distractor azimuth:
    cluster_perm(
        az_target_phoneme_trfs_standardized,
        az_distractor_phoneme_trfs_standardized,
        predictor='phonemes')
    # then phonemes target-distractor elevation:
    cluster_perm(
        ele_target_phoneme_trfs_standardized,
        ele_distractor_phoneme_trfs_standardized,
        predictor='phonemes')

    # phoneme diff waves (target - distractor) per plane
    az_phoneme_diff_waves_standardized = compute_diff_waves(
        az_target_phoneme_trfs_standardized,
        az_distractor_phoneme_trfs_standardized, predictor='phonemes')

    ele_diff_waves_phoneme_trfs_standardized = compute_diff_waves(
        ele_target_phoneme_trfs_standardized,
        ele_distractor_phoneme_trfs_standardized, predictor='phonemes')

    # and then the diff waves of phonemes across planes:
    cluster_perm(
        az_phoneme_diff_waves_standardized,
        ele_diff_waves_phoneme_trfs_standardized,
        predictor='phonemes')

    # repeat for envelopes
    _, az_target_env_trfs, _, \
        = extract_trfs(predictions_dict, stream='target', plane_name='azimuth', ch_selection=env_roi)

    _, ele_target_env_trfs, _, \
        = extract_trfs(predictions_dict, stream='target', plane_name='elevation', ch_selection=env_roi)
    # distractor - envelopes
    _, az_distractor_env_trfs, _, \
        = extract_trfs(predictions_dict, stream='distractor', plane_name='azimuth', ch_selection=env_roi)
    _, ele_distractor_env_trfs, _, \
        = extract_trfs(predictions_dict, stream='distractor', plane_name='elevation', ch_selection=env_roi)

    # first test envelopes target-distractor azimuth:
    cluster_perm(
        az_target_env_trfs,
        az_distractor_env_trfs,
        predictor='envelopes')

    # then envelopes target-distractor elevation:
    cluster_perm(
        ele_target_env_trfs,
        ele_distractor_env_trfs,
        predictor='envelopes')

    # envelope diff waves (target - distractor) per plane
    az_env_diff_waves = compute_diff_waves(
        az_target_env_trfs,
        az_distractor_env_trfs, predictor='envelopes')
    ele_env_diff_waves = compute_diff_waves(
        ele_target_env_trfs,
        ele_distractor_env_trfs, predictor='envelopes')

    # compare envelope diff waves across planes
    cluster_perm(
        az_env_diff_waves,
        ele_env_diff_waves,
        predictor='envelopes')