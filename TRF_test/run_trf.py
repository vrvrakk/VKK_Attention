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

X_folds = []
Y_folds = []

sub_list = list(distractor_dict.keys())
# Stack predictors for the target stream
scores_dict = {}
for sub, target_data, distractor_data in zip(sub_list, target_dict.values(), distractor_dict.values()):
    eeg = target_data['eeg']
    X_target = np.column_stack([target_data['onsets'], target_data['envelopes'], target_data['phonemes']])
    X_distractor = np.column_stack(
        [distractor_data['onsets'], distractor_data['envelopes'], distractor_data['phonemes']])
    Y_eeg = eeg
    col_names = ['onsets', 'envelopes', 'phonemes']
    print("X_target shape:", X_target.shape)
    print("X_distractor shape:", X_distractor.shape)
    print("EEG shape:", Y_eeg.shape)
    # checking collinearity:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    # Build combined DataFrame
    X = pd.DataFrame(
        np.column_stack([
            X_target,  # target predictors
            X_distractor  # distractor predictors
        ]),
        columns=[f'{k}_target' for k in col_names] + [f'{k}_distractor' for k in col_names])
    # Add constant for VIF calculation
    X = sm.add_constant(X)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    print(vif)

    # split into trials:
    predictors_stacked = X.values  # ← ready for modeling
    X_folds.append(predictors_stacked)
    Y_folds.append(Y_eeg)

random.seed(42)

reg_lambda = 14.4
tmin = - 0.1
tmax = 1.0
sfreq = 125

X_folds_filt = X_folds[6:]
Y_folds_filt = Y_folds[6:]

predictions_dict = {}
for sub, pred_fold, eeg_fold in zip(sub_list, X_folds_filt, Y_folds_filt):
    trf = TRF(direction=1, method='ridge')  # forward model
    trf.train(stimulus=pred_fold, response=eeg_fold, fs=sfreq, tmin=tmin, tmax=tmax, regularization=reg_lambda, average=True, seed=42)
    # Do I want one TRF across all the data? → average=True
    predictions, r = trf.predict(stimulus=pred_fold, response=eeg_fold, average=False)
    weights = trf.weights
    predictions_dict[sub] = {'predictions': predictions, 'r': r, 'weights': weights}


threshold = 0.1   # e.g., keep channels with r >= 0.05
eeg_ch = ['Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8','TP9','CP5','CP1',
          'CP2','CP6','TP10','P7','P3','Pz','P4','P8','AF7','AF3','AF4','AF8','F5','F1','F2','F6','FT9','FT7','FC3',
          'FC4','FT8','FT10','C5','C1','C2','C6','TP7','CP3','CPz','CP4','TP8','P5','P1','P2','P6','FCz']

predictors = list(X.keys())
time = trf.times

target_trfs = {}
distractor_trfs = {}

for sub, rows in predictions_dict.items():
    r_values = rows['r']
    predictions = rows['predictions']
    weights = rows['weights']
    sig_mask = r_values >= threshold

    # Selected channels
    roi_channels = np.array(eeg_ch)[sig_mask]
    roi_r_values = r[sig_mask]

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

    target_weights_phonemes = smoothed_weights[3, :, sig_mask]
    target_phoneme_avg = np.average(target_weights_phonemes, axis=0)

    distractor_weights_phonemes = smoothed_weights[6, :, sig_mask]
    distractor_phoneme_avg = np.average(distractor_weights_phonemes, axis=0)
    target_trfs[sub] = target_phoneme_avg
    distractor_trfs[sub] = distractor_phoneme_avg


# run cluster-based non-parametric permutation across time
# target weights vs distractor
from mne.stats import permutation_cluster_test

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
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    X, n_permutations=10000, tail=1, n_jobs=1
)

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
        plt.axvspan(time[time_inds.start], time[time_inds.stop-1],
                    color='gray', alpha=0.3)

plt.title(f'TRF Comparison - {condition}')
plt.legend()
plt.show()