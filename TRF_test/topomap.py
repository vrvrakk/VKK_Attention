import copy
# 1: Import Libraries
# for defining directories and loading/saving
import os
from pathlib import Path
import pickle as pkl

# for designing the matrix
import numpy as np
import pandas as pd
from mtrf import TRF
import random

# eeg:
import mne

# plotting:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

# stats
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from scipy.stats import ttest_1samp

'''
a script to get topomaps of each predictor (azimuth and elevation respectively), avged across subjects, we get:
- avg mV difference in each channel: TRF response amplitude of target - distractor
- repeat for each predictor separately
- plot with colorbar included for extra transparency
- avg sub-conds of each plane
- save
'''


def matrix_vif(matrix):
    X = sm.add_constant(matrix)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    print(vif)
    return vif


def run_model(X_folds, Y_folds, sub_list):
    if condition in ['a1', 'a2']:
        X_folds_filt = X_folds[6:]  # only filter for the conditions that are from the old design (elevation)
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
        # get r vals for each channel bish
        weights = trf.weights
        predictions_dict[sub] = {'predictions': predictions, 'r': r, 'weights': weights}

    time = trf.times

    return time, predictions_dict


if __name__ == '__main__':

    # define channels and conditions
    all_ch = np.array(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                       'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz',
                       'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6',
                       'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1',
                       'Oz', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5',
                       'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8',
                       'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz',
                       'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3',
                       'POz', 'PO4', 'PO8', 'FCz'])

    planes = {'azimuth': ['a1', 'a2'],
              'elevation': ['e1', 'e2']}

    plane_name = 'azimuth'
    conditions = planes[plane_name]
    stim_type = 'all'

    # initialize combined arrays
    plane_X_target_folds = {cond: {} for cond in conditions}
    plane_X_distractor_folds = {cond: {} for cond in conditions}
    plane_Y_folds = {cond: {} for cond in conditions}

    for condition in conditions:
        print(f'Processing condition: {condition.upper()}')

        base_dir = Path.cwd()
        data_dir = base_dir / 'data' / 'eeg'
        predictor_dir = data_dir / 'predictors'
        bad_segments_dir = predictor_dir / 'bad_segments'
        eeg_dir = Path('D:/VKK_Attention/data/eeg/preprocessed/results')

        dict_dir = data_dir / 'journal' / 'TRF' / 'matrix' / condition / stim_type

        with open(dict_dir / f'{condition}_matrix_target.pkl', 'rb') as f:
            target_dict = pkl.load(f)
        with open(dict_dir / f'{condition}_matrix_distractor.pkl', 'rb') as f:
            distractor_dict = pkl.load(f)

        sub_list = list(distractor_dict.keys())

        X_target_folds = []
        X_distractor_folds = []
        Y_folds = []
        for sub, target_data, distractor_data in zip(sub_list, target_dict.values(), distractor_dict.values()):
            eeg = target_data['eeg']
            X_target = np.column_stack(
                [target_data['envelopes'], target_data['phonemes'], target_data['responses']])
            X_distractor = np.column_stack(
                [distractor_data['envelopes'], distractor_data['phonemes'], distractor_data['responses']])

            # quick sanity check
            print(f"{sub} ({condition}): X_target {X_target.shape}, EEG {eeg.shape}")

            # Build DataFrames for VIF
            col_names = ['envelopes', 'phonemes', 'responses']
            target_df = pd.DataFrame(X_target, columns=[f'{name}_target' for name in col_names])
            distractor_df = pd.DataFrame(X_distractor, columns=[f'{name}_distractor' for name in col_names])

            # compute VIFs (optional)
            target_vif = matrix_vif(target_df)
            distractor_vif = matrix_vif(distractor_df)

            # split into trials:
            target_predictors_stacked = target_df.values  # ← ready for modeling
            distractor_predictors_stacked = distractor_df.values
            X_target_folds.append(target_predictors_stacked)
            X_distractor_folds.append(distractor_predictors_stacked)
            Y_folds.append(eeg)
            plane_X_target_folds[condition] = X_target_folds
            plane_X_distractor_folds[condition] = X_distractor_folds
            plane_Y_folds[condition] = Y_folds

        # concatenate predictor arrays of conditions per subject
    def concat_conds(plane_X_folds):
        X_cond1 = plane_X_folds[conditions[0]]
        X_cond2 = plane_X_folds[conditions[1]]
        X_folds_concat = []
        for sub_arrays1, sub_arrays2 in zip(X_cond1, X_cond2):
            sub_arr_concat = np.concatenate((sub_arrays1, sub_arrays2), axis=0)
            X_folds_concat.append(sub_arr_concat)
        return X_folds_concat

    X_target_folds_concat = concat_conds(plane_X_target_folds)
    X_distractor_folds_concat = concat_conds(plane_X_distractor_folds)

    Y_cond1 = plane_Y_folds[conditions[0]]
    Y_cond2 = plane_Y_folds[conditions[1]]
    Y_folds_concat = []
    for eeg_arr1, eeg_arr2 in zip(Y_cond1, Y_cond2):
        sub_eeg_concat = np.concatenate((eeg_arr1, eeg_arr2), axis=0)
        Y_folds_concat.append(sub_eeg_concat)

    random.seed(42)

    best_lambda = 0.01

    tmin = - 0.1
    tmax = 1.0
    sfreq = 125

    time, target_predictions_dict = run_model(X_target_folds_concat, Y_folds_concat, sub_list)
    _, distractor_predictions_dict = run_model(X_distractor_folds_concat, Y_folds_concat, sub_list)

    def plot_trf_diffs(target_predictions_dict, distractor_predictions_dict, predictor='', roi=all_ch):
        if predictor == 'envelopes':
            pred_idx = 0
        elif predictor == 'phonemes':
            pred_idx = 1

        sub_trf_diffs = {}
        for sub in target_predictions_dict.keys():
            target_pred = target_predictions_dict[sub]['weights'][pred_idx]  # (time, channels)
            distractor_pred = distractor_predictions_dict[sub]['weights'][pred_idx]
            pred_diff = target_pred - distractor_pred
            avg_mV = np.mean(pred_diff, axis=0)  # average over time → one value per channel
            sub_trf_diffs[sub] = avg_mV

        # Stack all subjects (subjects × channels)
        all_diffs = np.stack(list(sub_trf_diffs.values()))  # (n_subs, n_channels)

        # --- Compute mean and consistency ---
        mean_diff_per_channel = np.mean(all_diffs, axis=0)  # group mean per channel
        std_diff_per_channel = np.std(all_diffs, axis=0)
        sem_diff_per_channel = std_diff_per_channel / np.sqrt(all_diffs.shape[0])  # standard error

        # Optionally compute t-values vs. 0 (null hypothesis: no diff)
        t_vals, p_vals = ttest_1samp(all_diffs, popmean=0, axis=0)
        # (t-values show how consistent the difference is across subs)

        # --- Create MNE info for topomap ---
        info = mne.create_info(ch_names=list(roi), sfreq=sfreq, ch_types='eeg')
        montage = mne.channels.make_standard_montage('standard_1020')
        info.set_montage(montage)

        # --- Plot topomaps side by side: mean and consistency ---
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))

        # Mean amplitude map
        im1, _ = mne.viz.plot_topomap(
            mean_diff_per_channel, info, axes=axes[0], show=False,
            cmap='RdBu_r', contours=0,
            vlim=(-np.max(np.abs(mean_diff_per_channel)),
                  np.max(np.abs(mean_diff_per_channel)))
        )
        axes[0].set_title(f'{predictor.capitalize()} Mean TRF Difference (mV)')
        cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.6)
        cbar1.set_label('Mean amplitude (mV)')

        # Consistency map (t-values)
        im2, _ = mne.viz.plot_topomap(
            t_vals, info, axes=axes[1], show=False,
            cmap='plasma', contours=0
        )
        axes[1].set_title(f'{predictor.capitalize()} TRF Consistency (t-values)')
        cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.6)
        cbar2.set_label('t-statistic across subjects')

        plt.suptitle(f"Plane: {plane_name.capitalize()} — {predictor.capitalize()}", fontsize=14)
        fig_path = data_dir / 'journal' / 'figures' / 'TRF' / 'topomap' / plane_name / 'all'
        fig_path.mkdir(parents=True, exist_ok=True)
        filename = f'topomaps_{predictor}_{plane_name}.png'
        plt.tight_layout()
        plt.show()
        plt.savefig(fig_path / filename, dpi=300)

    plot_trf_diffs(target_predictions_dict, distractor_predictions_dict, predictor='envelopes', roi=all_ch)
    plot_trf_diffs(target_predictions_dict, distractor_predictions_dict, predictor='phonemes', roi=all_ch)
