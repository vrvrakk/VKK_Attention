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

# troubleshooting
import logging
from copy import deepcopy

# for plotting
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

# directories:
base_dir = Path.cwd()
data_dir = base_dir / 'data' / 'eeg'
predictor_dir = data_dir / 'predictors'
bad_segments_dir = predictor_dir / 'bad_segments'
eeg_dir = Path('D:/VKK_Attention/data/eeg/preprocessed/results')

# specify condition:
condition = 'e2'
stim_type = 'all'

# load matrices of chosen stimulus and condition:
dict_dir = data_dir / 'journal' / 'TRF' / 'matrix' / condition / stim_type
with open(dict_dir / f'{condition}_matrix_target.pkl', 'rb') as f:
    target_dict = pkl.load(f)

with open(dict_dir / f'{condition}_matrix_distractor.pkl', 'rb') as f:
    distractor_dict = pkl.load(f)

import random

sub_list = list(distractor_dict.keys())


# vif function:
def matrix_vif(matrix):
    X = sm.add_constant(matrix)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    print(vif)
    return vif


target_X_folds = []
distractor_X_folds = []

Y_folds = []
# Stack predictors for the target stream
for sub, target_data, distractor_data in zip(sub_list, target_dict.values(), distractor_dict.values()):
    eeg = target_data['eeg']

    X_target = np.column_stack([target_data['onsets'], target_data['envelopes'], target_data['phonemes'], target_data['responses']])
    X_distractor = np.column_stack(
        [distractor_data['onsets'], distractor_data['envelopes'], distractor_data['phonemes'], distractor_data['responses']])

    Y_eeg = eeg
    col_names = ['onsets', 'envelopes', 'phonemes', 'responses']
    print("X_target shape:", X_target.shape)
    print("X_distractor shape:", X_distractor.shape)
    print("EEG shape:", Y_eeg.shape)
    # checking collinearity:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Build combined DataFrame
    X_target_df = pd.DataFrame(np.column_stack([X_target]),columns=[k for k in col_names])
    X_distractor_df = pd.DataFrame(np.column_stack([X_distractor]), columns=[k for k in col_names])
    # Add constant for VIF calculation
    target_vif = matrix_vif(X_target_df)
    distractor_vif = matrix_vif(X_distractor_df)
    # split into trials:
    target_predictors_stacked = X_target_df.values  # ← ready for modeling
    distractor_predictors_stacked = X_distractor_df.values
    target_X_folds.append(target_predictors_stacked)
    distractor_X_folds.append(distractor_predictors_stacked)
    Y_folds.append(Y_eeg)

random.seed(42)


best_lambda = 0.01

tmin = - 0.1
tmax = 1.0
sfreq = 125

threshold = 0.1   # e.g., keep channels with r >= 0.05

all_ch = ['Fp1', 'Fp2', 'F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8',
          'TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10',
          'AF7','AF3','AF4','AF8','F5','F1','F2','F6','FT9','FT7','FC3','FC4','FT8','FT10','C5','C1',
          'C2','C6','TP7','CP3','CPz','CP4','TP8','P5','P1','P2','P6','PO7','PO3','POz','PO4','PO8','FCz']


eeg_ch = np.array(['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8', 'TP9','CP5',
                   'CP1', 'CP2','CP6','TP10','P7','P3','Pz','P4','P8','AF7','AF3','AF4','AF8', 'F5','F1','F2','F6',
                   'FT9','FT7','FC3', 'FC4','FT8','FT10','C5','C1','C2','C6','TP7','CP3', 'CPz','CP4',
                   'TP8','P5','P1','P2','P6','FCz'])

roi = np.array(['Fp1', 'F3', 'Fz', 'F4', 'FC1', 'C3', 'Cz', 'C4', 'C1', 'C2', 'CP5',
                   'CP1', 'CP3', 'P7', 'P3', 'Pz', 'P4', 'P5', 'P1', 'F1', 'F2', 'AF3', 'FCz'])


ch_mask = np.isin(eeg_ch, roi)


def get_weight_avg(smoothed_weights, n, sig_mask):
    weights = smoothed_weights[n, :, sig_mask]
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
        predictions, r = trf.predict(stimulus=pred_fold, response=eeg_fold, average=False)
        weights = trf.weights
        predictions_dict[sub] = {'predictions': predictions, 'r': r, 'weights': weights}

    time = trf.times

    phoneme_trfs = {}
    onset_trfs = {}
    env_trfs = {}
    response_trfs = {}
    sig_channels = {}
    for sub, rows in predictions_dict.items():
        r_values = rows['r']
        # predictions = rows['predictions']
        weights = rows['weights']
        sig_mask = r_values >= threshold

        # # Selected channels
        # roi_channels = np.array(eeg_ch)[sig_mask]
        # roi_r_values = r[sig_mask]

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

        phoneme_avg = get_weight_avg(smoothed_weights, 2, sig_mask)
        onset_avg = get_weight_avg(smoothed_weights, 0, sig_mask)
        env_avg = get_weight_avg(smoothed_weights, 1, sig_mask)
        response_avg = get_weight_avg(smoothed_weights, 3, sig_mask)

        phoneme_trfs[sub] = phoneme_avg
        onset_trfs[sub] = onset_avg
        env_trfs[sub] = env_avg
        response_trfs[sub] = response_avg

    return predictions_dict, phoneme_trfs, onset_trfs, env_trfs, response_trfs, time, sig_channels


target_predictions_dict, target_phoneme_trfs, target_onset_trfs, target_env_trfs, target_response_trfs, time,\
    target_sig_channels = run_model(target_X_folds, Y_folds, sub_list)

distractor_predictions_dict, distractor_phoneme_trfs, distractor_onset_trfs, distractor_env_trfs, \
    distractor_response_trfs, _, distractor_sig_channels = run_model(distractor_X_folds, Y_folds, sub_list)

# further isolate common channels:
# get intersection of all subject channel sets

# intersection within each dict
# common_target = set.intersection(*(set(chs) for chs in target_sig_channels.values()))
# common_distractor = set.intersection(*(set(chs) for chs in distractor_sig_channels.values()))
#
# # intersection across both dicts
# common_both = common_target & common_distractor
#
# print("Common across both:", sorted(common_both))
#
# import pickle as pkl
# channel_dir = data_dir / 'journal' / 'common_channels'
# channel_dir.mkdir(parents=True, exist_ok=True)
# with open(channel_dir/f'{condition}_{stim_type}_common_chs.pkl', 'wb') as f:
#     print(f'Saved common channels in {channel_dir}')
#     pkl.dump(common_both, f)


# run cluster-based non-parametric permutation across time
# target weights vs distractor
from mne.stats import permutation_cluster_test


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
    plt.legend()
    fig_path = data_dir / 'journal' / 'figures' / 'TRF' / condition / stim_type
    fig_path.mkdir(parents=True, exist_ok=True)
    filename = f'{predictor}_{stim_type}_{condition}.png'
    plt.savefig(fig_path/filename, dpi=300)
    plt.show()


cluster_perm(target_phoneme_trfs, distractor_phoneme_trfs, predictor='phonemes')
cluster_perm(target_onset_trfs, distractor_onset_trfs, predictor='onsets')
cluster_perm(target_env_trfs, distractor_env_trfs, predictor='envelopes')


from scipy.stats import ttest_rel, wilcoxon


def compare_r_values(predictions_target, predictions_distractor, test='ttest'):
    """
    Compare subject-wise r values between target and distractor.

    Parameters
    ----------
    predictions_target : dict
        Dictionary {sub: {'r': r_values, ...}} from run_model for target.
    predictions_distractor : dict
        Same as above but for distractor.
    test : str
        'ttest' or 'wilcoxon' for paired comparison.
    alpha : float
        Significance threshold.

    Returns
    -------
    stats : dict
        Contains means, test results, effect size, etc.
    """

    # match subjects
    subs = sorted(set(predictions_target).intersection(predictions_distractor))
    target_r = np.array([np.mean(predictions_target[s]['r']) for s in subs])
    distractor_r = np.array([np.mean(predictions_distractor[s]['r']) for s in subs])

    # run paired test
    if test == 'ttest':
        stat, pval = ttest_rel(target_r, distractor_r)
        # effect size: Cohen's d for paired samples
        diff = target_r - distractor_r
        cohen_d = np.mean(diff) / np.std(diff, ddof=1)
    elif test == 'wilcoxon':
        stat, pval = wilcoxon(target_r, distractor_r)
        diff = target_r - distractor_r
        # effect size: rank-biserial correlation
        cohen_d = (np.sum(diff > 0) - np.sum(diff < 0)) / len(diff)
    else:
        raise ValueError("test must be 'ttest' or 'wilcoxon'")

    # summary
    stats = {
        'target_mean': target_r.mean(),
        'distractor_mean': distractor_r.mean(),
        'target_std': target_r.std(ddof=1),
        'distractor_std': distractor_r.std(ddof=1),
        'statistic': stat,
        'pval': pval,
        'effect_size': cohen_d,
        'n': len(subs)
    }

    # plot
    fig, ax = plt.subplots()
    x = [0, 1]
    ax.bar(x, [target_r.mean(), distractor_r.mean()],
           yerr=[target_r.std(ddof=1) / np.sqrt(len(subs)),
                 distractor_r.std(ddof=1) / np.sqrt(len(subs))],
           color=['blue', 'red'], alpha=0.7, capsize=5)
    ax.set_xticks(x)
    ax.set_xticklabels(['Target', 'Distractor'])
    ax.set_ylabel('Mean r-value')
    ax.set_title(f'Prediction Accuracy Comparison (n={len(subs)})\n'
                 f'p={pval:.3g}, effect={cohen_d:.2f}')

    # paired scatter
    for i, (t, d) in enumerate(zip(target_r, distractor_r)):
        ax.plot([0, 1], [t, d], 'k-', alpha=0.5)

    plt.show()

    return stats


stats = compare_r_values(target_predictions_dict, distractor_predictions_dict, test='ttest')
print(stats)