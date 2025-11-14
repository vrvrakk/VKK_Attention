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
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8

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


def standardize_phoneme_trfs(X_target_folds_concat, X_distractor_folds_concat,
                             sub_list, target_trfs, distractor_trfs):
    """
    Standardize phoneme TRF weights using √p(1−p),
    where p is the proportion of non-zero impulses in the phoneme predictor (column index = 1).

    Parameters
    ----------
    X_target_folds_concat : list of np.ndarray
        Each element = subject design matrix for target stream (time × predictors)
    X_distractor_folds_concat : list of np.ndarray
        Each element = subject design matrix for distractor stream (time × predictors)
    sub_list : list of str
        Subject IDs, same order as X_folds_concat
    target_trfs, distractor_trfs : dict
        Each dict: {sub: {'weights': [env_TRF, phoneme_TRF]}}

    Returns
    -------
    target_trfs_std, distractor_trfs_std : dict
        With standardized phoneme weights
    """
    phoneme_idx = 1
    target_trfs_std = {}
    distractor_trfs_std = {}

    for sub_idx, sub_name in enumerate(sub_list):
        if sub_name not in target_trfs or sub_name not in distractor_trfs:
            continue

        # target stream
        phonemes_target = X_target_folds_concat[sub_idx][:, phoneme_idx]
        p_t = np.count_nonzero(phonemes_target) / len(phonemes_target)
        std_t = np.sqrt(p_t * (1 - p_t))
        target_trfs_std[sub_name] = target_trfs[sub_name].copy()
        target_trfs_std[sub_name]['weights'][1] = target_trfs[sub_name]['weights'][1] * std_t

        # distractor stream
        phonemes_distractor = X_distractor_folds_concat[sub_idx][:, phoneme_idx]
        p_d = np.count_nonzero(phonemes_distractor) / len(phonemes_distractor)
        std_d = np.sqrt(p_d * (1 - p_d))
        distractor_trfs_std[sub_name] = distractor_trfs[sub_name].copy()
        distractor_trfs_std[sub_name]['weights'][1] = distractor_trfs[sub_name]['weights'][1] * std_d

    return target_trfs_std, distractor_trfs_std


def concat_conds(plane_X_folds):
    X_cond1 = plane_X_folds[conditions[0]]
    X_cond2 = plane_X_folds[conditions[1]]
    X_folds_concat = []
    for sub_arrays1, sub_arrays2 in zip(X_cond1, X_cond2):
        sub_arr_concat = np.concatenate((sub_arrays1, sub_arrays2), axis=0)
        X_folds_concat.append(sub_arr_concat)
    return X_folds_concat


def select_time_windows(tw_dict, plane_name, stim_type, predictor=''):
    time_windows = tw_dict[plane_name][predictor][stim_type]
    tmin = time_windows[0]
    tmax = time_windows[1]
    return tmin, tmax


def plot_trf_components(target_predictions_dict, distractor_predictions_dict, predictor='', roi=None, times=None,
                        sfreq=125, plane_name='', data_dir=None, stim_type=''):

    if predictor == 'envelopes':
        pred_idx = 0
    elif predictor == 'phonemes':
        pred_idx = 1
        target_predictions_dict, distractor_predictions_dict = standardize_phoneme_trfs(
            X_target_folds_concat, X_distractor_folds_concat,
            sub_list,
            target_predictions_dict,
            distractor_predictions_dict)
    else:
        raise ValueError("Predictor must be 'envelopes' or 'phonemes'.")

    #  create MNE info for topomap
    info = mne.create_info(ch_names=list(roi), sfreq=sfreq, ch_types='eeg')
    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    # collect subject TRF differences
    sub_diffs = []
    for sub in target_predictions_dict.keys():
        target_pred = target_predictions_dict[sub]['weights'][pred_idx]  # (time, channels)
        distractor_pred = distractor_predictions_dict[sub]['weights'][pred_idx]
        pred_diff = target_pred - distractor_pred
        sub_diffs.append(pred_diff)
    sub_diffs = np.stack(sub_diffs)  # (n_subs, n_times, n_channels)

    # helper to extract indices for component windows
    tmin, tmax = select_time_windows(tw_dict, plane_name, stim_type, predictor)

    def window_idx(tmin, tmax):
        return np.where((times >= tmin) & (times <= tmax))[0]

    # prepare figure
    n_comp = 1
    fig, axes = plt.subplots(1, n_comp, figsize=(4 * n_comp, 4))

    # Handle case of only one component
    if n_comp == 1:
        axes = np.array([axes])

    # loop over components
    tidx = window_idx(tmin, tmax)
    comp_vals = np.mean(sub_diffs[:, tidx, :], axis=1)  # (n_subs, n_channels)
    mean_map = np.mean(comp_vals, axis=0)  # average across subjects

    im, _ = mne.viz.plot_topomap(
        mean_map, info, axes=axes[0], show=False,
        cmap='magma', contours=0)
    axes[0].set_title(f'({tmin * 1e3:.0f}–{tmax * 1e3:.0f} ms)')
    cbar = plt.colorbar(im, ax=axes[0], shrink=0.6)
    cbar.set_label('Mean $\\Delta$TRF Amplitude (a.u.)')

    # plt.suptitle(f'{predictor.capitalize()} TRF components — {plane_name.capitalize()}', fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    # --- save figure ---
    if data_dir is not None:
        fig_path = data_dir / 'journal' / 'figures' / 'TRF' / 'topomap' / plane_name / 'components' / stim_type
        fig_path.mkdir(parents=True, exist_ok=True)
        filename = f'trf_components_{predictor}_{plane_name}_seismic.png'
        plt.savefig(fig_path / filename, dpi=300)
        plt.savefig(fig_path / f'trf_components_{predictor}_{plane_name}_seismic.pdf', dpi=300)
        print(f"Saved to: {fig_path / filename}")
    plt.show()


if __name__ == '__main__':
    # time windows with sig clusters, based on cluster-based perm analysis:
    tw_main = {'azimuth': {'envelopes': {'all': (0.152, 0.200)},
                           'phonemes': {'all': (0.056, 0.088)}},
               'elevation': {'envelopes': {'all': (0.296, 0.344)},
                             'phonemes': {'all': (0.056, 0.112)}}}

    tw_phonemes = {'elevation': {'phonemes': {'non_targets': (0.056, 0.096),
                                              'target_nums': (0.256, 0.336)}},
                   'azimuth': {'phonemes': {'target_nums': (0.056, 0.144)}}}

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

    plane_name = 'elevation'
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

    # select time-windows of interest, based on plane and stimulus type:
    if stim_type == 'all':
        tw_dict = tw_main
    else:
        tw_dict = tw_phonemes

    plot_trf_components(
        target_predictions_dict, distractor_predictions_dict,
        predictor='envelopes',
        roi=all_ch,
        times=time,
        sfreq=125,
        plane_name=plane_name,
        data_dir=data_dir,
        stim_type=stim_type)

    plot_trf_components(
        target_predictions_dict, distractor_predictions_dict,
        predictor='phonemes',
        roi=all_ch,
        times=time,
        sfreq=125,
        plane_name=plane_name,
        data_dir=data_dir,
        stim_type=stim_type)
