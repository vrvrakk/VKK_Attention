'''
A script to run a forward TRF model, using 2 audio features as regressors (envelopes and phoneme binary impulses)
+ motor responses as binary impulses.
- Envelopes arrays are z-scored, phonemes without initial phoneme of each word
- for each plane, concatenate the sub-condition regressor and EEG data arrays. subject-level
- Run TRF prediction with a priori regularization parameter (0.01), across all channels
- Select channels of significance / based on literature (Di Liberto for phonemes + envelopes)
'''


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


def run_model(X_folds, Y_folds, sub_list):
    if plane == ['a1', 'a2']:
        X_folds_filt = X_folds[6:]  # only filter for the conditions that are affected
        Y_folds_filt = Y_folds[6:]
        sub_list = sub_list[6:]
    else:
        X_folds_filt = X_folds
        Y_folds_filt = Y_folds
        sub_list = sub_list

    predictions_dict = {}
    time = None
    for sub, pred_fold, eeg_fold in zip(sub_list, X_folds_filt, Y_folds_filt):

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
    if stream == 'target':
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
        print(f"Warning: {sub} had no significant channels, using all channels instead.")
    return weights_avg


def extract_trfs(predictions_dict, stream='', ch_selection=None):
    phoneme_trfs = {}
    env_trfs = {}
    response_trfs = {}
    # onset_trfs = {}
    # sig_chs = {}
    for sub, rows in predictions_dict.items():
        r_vals = rows['r']
        weights = rows['weights']
        # sig_mask = r_vals >= 0.1
        # masking = np.isin(all_ch, ch_selection) & sig_mask
        # sig_ch = all_ch[sig_mask]
        # sig_chs[sub] = sig_ch
        masking = np.isin(all_ch, ch_selection)
        # keep channels from all channels that are in the ROI
        # (channel selection) and mask the rest as False

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
            phoneme_idx, env_idx, response_idx = get_pred_idx(stream)
            response_avg = get_weight_avg(smoothed_weights, response_idx, masking)
            response_trfs[sub] = response_avg
        else:
            phoneme_idx, env_idx = get_pred_idx(stream)

        # common predictors for both target and distractor
        phoneme_avg = get_weight_avg(smoothed_weights, phoneme_idx, masking)
        env_avg = get_weight_avg(smoothed_weights, env_idx, masking)
        # onset_avg = get_weight_avg(smoothed_weights, masking)
        phoneme_trfs[sub] = phoneme_avg
        env_trfs[sub] = env_avg
        # onset_trfs[sub] = onset_avg
    return phoneme_trfs, env_trfs, response_trfs #, sig_chs


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


def count_non_zeros(X_folds_concat, sub_list, phoneme_trfs, stream=''):
    '''
    a helper function to count the binary impulses / ones in the phoneme predictor array
    - calculate the p rate of 1s in the array pred_std = (len(ones) / len(array)
    - then standardize phoneme weights:
         beta_std(t) = beta(t) * (σ_x / σ_y). With z-scored EEG, σ_y = 1
         therefore: b_std = pred_weight(t) * pred_std
    '''
    if stream == 'target':
        phoneme_idx = 1
    else:
        phoneme_idx = -1
    phoneme_trfs_standardized = {}
    for sub_idx, sub_matrix in enumerate(X_folds_concat):
        sub_name = sub_list[sub_idx]
        if sub_name not in list(phoneme_trfs.keys()):
            continue
        phoneme_weights = phoneme_trfs[sub_name]
        phoneme_weights_copy = phoneme_weights.copy()
        phonemes = sub_matrix[:, phoneme_idx]
        p = np.count_nonzero(phonemes) / len(phonemes)
        std = np.sqrt(p * (1 - p))
        for idx, beta in enumerate(phoneme_weights_copy):
            phoneme_weights_copy[idx] = beta * std
        phoneme_trfs_standardized[sub_name] = phoneme_weights_copy
    return phoneme_trfs_standardized


def cluster_perm(target_trfs, distractor_trfs, predictor, plane='', roi_type=''):
    # stack into arrays (n_subjects, n_times)
    from mne.stats import fdr_correction
    target_data = np.vstack(list(target_trfs.values()))
    distractor_data = np.vstack(list(distractor_trfs.values()))

    # compute means/SEMs for plotting full time
    target_mean = target_data.mean(axis=0)
    distractor_mean = distractor_data.mean(axis=0)
    target_sem = target_data.std(axis=0) / np.sqrt(target_data.shape[0])
    distractor_sem = distractor_data.std(axis=0) / np.sqrt(distractor_data.shape[0])

    # plot full responses
    plt.plot(time, target_mean, 'b-', linewidth=2, label='Target')
    plt.fill_between(time, target_mean - target_sem, target_mean + target_sem,
                     color='b', alpha=0.3)
    plt.plot(time, distractor_mean, 'r-', linewidth=2, label='Distractor')
    plt.fill_between(time, distractor_mean - distractor_sem, distractor_mean + distractor_sem,
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
        X = [target_data[:, tmask], distractor_data[:, tmask]]
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
            X, n_permutations=5000, tail=1, n_jobs=1
        )

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
            mean_diff, dz, gz = cluster_effect_size(target_data, distractor_data, time, time_sel, cl)
            plt.axvspan(time_sel[ti[0]], time_sel[ti[-1]],
                        color='gray', alpha=0.2, label=f'{comp} (p={pval_corr:.3f}\nHedges $g_z$ = {np.round(gz, 3)})')
            print(f"Cluster in {comp}: raw p={pval:.3f}, FDR-corrected p={pval_corr:.3f}, g = {gz:.3f}")

    plt.title(f'TRF Comparison - {plane} - {predictor}')
    plt.xlim([time[0], 0.6])
    if predictor == 'phonemes':
        plt.ylim([-0.6, 0.65])
    plt.legend(loc='upper right', fontsize='small')
    fig_path = data_dir / 'journal' / 'figures' / 'TRF' / plane / stim_type
    fig_path.mkdir(parents=True, exist_ok=True)
    filename = f'{predictor}_{stim_type}_{condition}_{roi_type}_roi.png'
    plt.savefig(fig_path / filename, dpi=300)
    plt.show()


def detect_trf_outliers(predictions_dict, method="iqr", threshold=3.0):
    # stack data
    data_list = []
    for sub in predictions_dict.keys():
        data = np.mean(predictions_dict[sub]['r'])
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
            return (values < lower)

        data_flags = iqr_outlier_flags(data_list)
        for i, sub in enumerate(subs):
            outliers[sub] = data_flags[i]

    # print results
    print(f"\n=== Outlier detection ({method}) ===")
    # always initialize
    predictions_dict_updated = predictions_dict.copy()
    for sub, flags in outliers.items():
        if outliers[sub]:
            print(f"Sub {sub} flagged: {flags}. Subject removed from further analysis.")
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
        else:
            print('No outliers detected')
            return None, None


def get_prediction_accuracy(predictions_dict, sub_list, predictor='phonemes',
                            roi=None, save_dir=None, metric='mean_r'):
    """
    Compute TRF model prediction accuracy per subject based on ROI-averaged r-values.

    Parameters
    ----------
    predictions_dict : dict
        Output from run_model(). Expected format:
        {sub: {'predictions': array, 'r': array (n_channels,), 'weights': array}}
    sub_list : list
        Subject IDs corresponding to predictions_dict keys.
    predictor : str
        Which predictor to use ('phonemes' or 'envelopes').
    roi : electrodes for selected predictor
    save_dir : str or Path, optional
        Directory to save results as CSV.
    metric : str
        Aggregation method across ROI channels: 'mean_r' or 'max_r'.

    Returns
    -------
    acc_df : pd.DataFrame
        DataFrame with subject IDs and ROI-averaged accuracy.
    """

    results = []
    for sub in sub_list:
        if sub not in predictions_dict:
            print(f"Skipping {sub} (no data found)")
            continue

        r_vals = predictions_dict[sub]['r']

        # If r is scalar, just take it
        if np.ndim(r_vals) == 0:
            r_metric = r_vals
        else:
            # Restrict to ROI channels
            roi_mask = np.isin(all_ch, roi)
            roi_r = np.array(r_vals)[roi_mask]
            if metric == 'mean_r':
                r_metric = np.nanmean(roi_r)
            elif metric == 'max_r':
                r_metric = np.nanmax(roi_r)
            else:
                raise ValueError("metric must be 'mean_r' or 'max_r'")

        results.append({'subject': sub, 'roi_mean_r': r_metric})

    acc_df = pd.DataFrame(results).sort_values('subject').reset_index(drop=True)

    print(f"\n=== Prediction Accuracy ({predictor.upper()}, {metric}) ===")
    print(acc_df)

    # Save results
    if save_dir is not None:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        csv_path = save_dir / f"prediction_accuracy_{predictor}_{metric}.csv"
        acc_df.to_csv(csv_path, sep=';', encoding='utf-8', index=False)
        print(f"Saved accuracy table to: {csv_path}")

    return acc_df


def plot_topomaps(env_roi, phoneme_roi):
    phoneme_masking = np.isin(all_ch, phoneme_roi)
    env_masking = np.isin(all_ch, env_roi)
    subs_r_vals = {}
    for sub_name in predictions_dict.keys():
        phoneme_roi_r_vals = np.mean(predictions_dict[sub_name]['r'][phoneme_masking], axis=-1)
        env_roi_r_vals = np.mean(predictions_dict[sub_name]['r'][env_masking], axis=-1)
        sub_vals = {'phonemes': phoneme_roi_r_vals, 'envelopes': env_roi_r_vals}
        subs_r_vals[sub_name] = sub_vals
    # average across subs:
    phonemes_arr = []
    envs_arr = []
    for sub_names in subs_r_vals.keys():
        phoneme = subs_r_vals[sub_names]['phonemes']
        phonemes_arr.append(phoneme)
        env = subs_r_vals[sub_names]['envelopes']
        envs_arr.append(env)
    phonemes_r_avg = np.mean(phonemes_arr)
    envs_r_avg = np.mean(envs_arr)
    info = mne.create_info(ch_names=list(all_ch), sfreq=125, ch_types='eeg')
    info.set_montage('standard_1020')

    plt.style.use('seaborn-white')
    fig, ax = plt.subplots(1, 2, figsize=(10, 6))
    phoneme_values = np.full(len(all_ch), 0.3)
    env_values = np.full(len(all_ch), 0.3)
    for i, ch in enumerate(all_ch):
        if ch in phoneme_roi:
            phoneme_values[i] = 0.9  # highlight ROI electrodes
        elif ch in env_roi:
            env_values[i] = 0.9
    im, _ = mne.viz.plot_topomap(phoneme_values, info, cmap='magma', contours=6, sensors=False, show=False, axes=ax[0],
                                 vlim=(0.2, 1.0), extrapolate='head')
    im, _ = mne.viz.plot_topomap(env_values, info, cmap='magma', contours=6, sensors=False, show=False, axes=ax[1],
                                 vlim=(0.2, 1.0), extrapolate='head')
    ax[0].set_title(f'Phoneme TRF (mean r={phonemes_r_avg:.3f})', fontsize=11, fontweight='bold')
    ax[1].set_title(f'Envelopes TRF (mean r={envs_r_avg:.3f})', fontsize=11, fontweight='bold')
    plt.tight_layout()
    save_dir = data_dir / 'journal' / 'figures' / 'TRF' / 'topomap' / plane_name / stim_type
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f"topomaps_mask.png", dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':

    stim_type = 'target_nums'
    all_trfs = {}
    azimuth = ['a1', 'a2']
    elevation = ['e1', 'e2']
    planes = [azimuth, elevation]
    plane = planes[1]

    plane_X_folds = {cond: {} for cond in plane}
    plane_Y_folds = {cond: {} for cond in plane}
    for condition in plane:
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
            X = X.drop(columns=([col for col in list(X.columns) if col in ['onsets_target', 'onsets_distractor', 'alpha']]))
            # Add constant for VIF calculation
            vif = matrix_vif(X)

            # split into trials:
            predictors_stacked = X.values  # ← ready for modeling
            X_folds.append(predictors_stacked)
            Y_folds.append(Y_eeg)
            plane_X_folds[condition] = X_folds
            plane_Y_folds[condition] = Y_folds

    col_names = np.array(X.columns)

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

    # concatenate predictor arrays of conditions per subject
    X_cond1 = plane_X_folds[plane[0]]
    X_cond2 = plane_X_folds[plane[1]]
    X_folds_concat = []
    for sub_arrays1, sub_arrays2 in zip(X_cond1, X_cond2):
        sub_arr_concat = np.concatenate((sub_arrays1, sub_arrays2), axis=0)
        X_folds_concat.append(sub_arr_concat)
    Y_cond1 = plane_Y_folds[plane[0]]
    Y_cond2 = plane_Y_folds[plane[1]]
    Y_folds_concat = []
    for eeg_arr1, eeg_arr2 in zip(Y_cond1, Y_cond2):
        sub_eeg_concat = np.concatenate((eeg_arr1, eeg_arr2), axis=0)
        Y_folds_concat.append(sub_eeg_concat)

    time, predictions_dict = run_model(X_folds_concat, Y_folds_concat, sub_list)

    outliers, predictions_dict_updated = detect_trf_outliers(predictions_dict, method="iqr", threshold=3.0)

    roi_type = input('Choose an roi (main/test1/test2/viz/all/topo): ')

    if roi_type == 'main':
        phoneme_roi = np.array(['F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                                'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8'])  # supposedly phoneme electrodes
        env_roi = np.array(['Cz'])
    elif roi_type == 'topo':
        phoneme_roi = np.array(['FCz', 'Fz', 'F1', 'F2', 'AF3', 'AF4', 'AF7', 'AF8', 'Fp1', 'Fp2', 'Fpz'])
        env_roi = np.array(['Cz', 'FCz', 'CPz', 'Fz'])  # no AFz nor FPz available
    elif roi_type == 'test1':  # all channels but occipital
        phoneme_roi = [ch for ch in list(all_ch) if not ch.startswith(('O', 'PO'))]
        env_roi = phoneme_roi
    elif roi_type == 'test2':  # a more wide roi for each predictor
        phoneme_roi = np.array([
            'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6',
            'FT7', 'FT8', 'F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
            'AF3', 'AF4', 'AF7', 'AF8', 'Fp1', 'Fp2'])
        env_roi = np.array(['Cz', 'FCz', 'CPz'])  # no AFz nor FPz available
    elif roi_type == 'viz':
        phoneme_roi = np.array(['O1, O2', 'PO3', 'PO4', 'PO7', 'PO8', 'PO9', 'PO10', 'POz', 'Oz'])
        env_roi = phoneme_roi
    else:
        phoneme_roi = all_ch
        env_roi = all_ch

    if ['e1', 'e2'] == plane:
        plane_name = 'elevation'
    else:
        plane_name = 'azimuth'

    component_windows = {
        "P1": (0.05, 0.15),  # early sensory
        "N1": (0.15, 0.25),  # robust first attention effects; frontocentral and temporal
        "P2": (0.25, 0.35),  # conflict monitoring / categorization of stimulus
        "N2": (0.35, 0.50)}  # late attention-driven decision making

    # phonemes
    target_phoneme_trfs, _, _,\
         = extract_trfs(predictions_dict, stream='target', ch_selection=phoneme_roi)

    distractor_phoneme_trfs, _, _, \
         = extract_trfs(predictions_dict, stream='distractor', ch_selection=phoneme_roi)

    target_phoneme_trfs_standardized = count_non_zeros(X_folds_concat,
                                                       sub_list, target_phoneme_trfs, stream='target')

    distractor_phoneme_trfs_standardized = count_non_zeros(X_folds_concat,
                                                           sub_list, distractor_phoneme_trfs, stream='distractor')

    # cluster-based non-parametric permutation of target-distractor TRF responses
    cluster_perm(target_phoneme_trfs_standardized, distractor_phoneme_trfs_standardized,
                 predictor='phonemes', plane=plane_name, roi_type=roi_type)

    # repeat for envelopes
    _, target_env_trfs, _, \
        = extract_trfs(predictions_dict, stream='target', ch_selection=env_roi)

    _, distractor_env_trfs, _, \
        = extract_trfs(predictions_dict, stream='distractor', ch_selection=env_roi)

    cluster_perm(target_env_trfs, distractor_env_trfs, predictor='envelopes', plane=plane_name, roi_type=roi_type)

    # compute per-predictor accuracy
    # Compute and save model accuracy
    save_dir = data_dir / 'journal' / 'TRF' / 'results' / 'diagnostics' / 'main' / plane_name / stim_type

    acc_phonemes = get_prediction_accuracy(predictions_dict, sub_list,
                                           predictor='phonemes',
                                           roi=phoneme_roi,
                                           save_dir=save_dir)

    acc_envelopes = get_prediction_accuracy(predictions_dict, sub_list,
                                            predictor='envelopes',
                                            roi=env_roi,
                                            save_dir=save_dir)
    plot_topomaps(env_roi, phoneme_roi)



