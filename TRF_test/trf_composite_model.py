'''
Script for lambda optimization and ROI selection for TRF modelling
Using a sub-section of the data
Dropping occipitoparietal channels
Filtering up to 8 Hz (based on DiLiberto's work with phoneme models)
Predictors usage: binary weights, envelopes, phonemes

# steps:
1. load EEG, drop occipitoparietal channels, filter up to 8 Hz
2. mask EEG bad segments
3. load predictors -> mask as well if not done already
4. stack in a design matrix, 3 predictors per stream
'''
import copy
# 1: Import Libraries
# for defining directories and loading/saving
import os
from pathlib import Path

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

# 2: define directories
base_dir = Path.cwd()
data_dir = base_dir / 'data' / 'eeg'
predictor_dir = data_dir / 'predictors'
bad_segments_dir = predictor_dir / 'bad_segments'
eeg_dir = Path('D:/VKK_Attention/data/eeg/preprocessed/results')

# define some parameters:
conditions = {
    'a1': {'target': 'stream1', 'distractor': 'stream2'},
    'e1': {'target': 'stream1', 'distractor': 'stream2'},
    'a2': {'target': 'stream2', 'distractor': 'stream1'},
    'e2': {'target': 'stream2', 'distractor': 'stream1'}}

condition = 'a1'
predictors = ['binary_weights', 'envelopes', 'phonemes']
stim_types = ['all', 'non_targets', 'target_nums']
sfreq = 125


# now define functions
# 3: Load EEG:
def load_eeg(condition):
    eeg_list = []
    for sub_folders in eeg_dir.iterdir():
        if 'sub' in sub_folders.name:
            eeg_folder = sub_folders / 'ica'
            for eeg_files in eeg_folder.iterdir():
                if condition in eeg_files.name:
                    eeg = mne.io.read_raw_fif(eeg_files, preload=True)
                    # filter up to 8 Hz, resample if necessary and drop occipitoparietal channels + avg ref
                    eeg_filt = eeg.filter(l_freq=None, h_freq=8)
                    eeg_resamp = eeg_filt.resample(sfreq)
                    eeg_avg = eeg_resamp.set_eeg_reference('average')
                    eeg_ch = eeg_filt.pick([ch for ch in eeg_avg.ch_names if not ch.startswith(('O', 'PO'))])
                    eeg_list.append(eeg_ch)
    return eeg_list


eeg_list = load_eeg(condition=condition)
sub_list = []
for index, eeg_file in enumerate(eeg_list):
    filename = eeg_list[index].filenames[0]
    subject = os.path.basename(filename).split('_')[0]  # get base name of fif file (without path), split by _,
    # and get first value (sub)
    if subject not in sub_list:
        sub_list.append(subject)

# placeholder for EEG data / subject
eeg_dict = {}
for subject in sub_list:
    eeg_arrays = []
    for eeg_file in eeg_list:
        eeg_name = eeg_file.filenames[0]
        if subject in eeg_name:
            eeg_arrays.append(eeg_file)
        eeg_dict[subject] = eeg_arrays

# 4: mask bad segments
# import bad segment arrays, mask concatenated EEG data and add safety check
bads = []
for sub_folders in bad_segments_dir.iterdir():
    for cond_folders in sub_folders.iterdir():
        if condition in cond_folders.name:
            for files in cond_folders.iterdir():
                if 'concat.npy.npz' in files.name:
                    bad_array = np.load(files, allow_pickle=True)
                    bad_array = bad_array['bad_series']
                    bads.append(bad_array)


def mask_eeg():
    eeg_masked_dict = {}
    for (sub, eeg_arr), bad_series in zip(eeg_dict.items(), bads):
        # concatenate EEG data per subject:
        eeg_concat = deepcopy(eeg_arr)
        eeg_concat = mne.concatenate_raws(eeg_concat)
        # length check prior masking:
        logging.basicConfig(level=logging.INFO)
        logging.info(f"EEG length: {len(eeg_concat)}, bads length: {len(bad_series)}")
        assert len(eeg_concat) == len(bad_series), "Mismatch between EEG and bad segments length!"

        # mask along the array of corresponding sub:
        eeg_data = eeg_concat.get_data()
        eeg_masked = eeg_data[:, bad_series == 0]
        # z-score EEG data:
        eeg_clean = (eeg_masked - eeg_masked.mean(axis=1, keepdims=True)) / eeg_masked.std(axis=1, keepdims=True)
        print(eeg_clean.shape)  # (55, n_good_samples)
        eeg_masked_dict[sub] = eeg_clean
    return eeg_masked_dict


eeg_masked_dict = mask_eeg()
# 6: load predictors (ugh)
stim_type = 'all'


# I guess this is a function only for envs and onsets:
def select_folder(stim_type):
    if stim_type == 'all':
        target_stream = conditions[condition]['target']
        distractor_stream = conditions[condition]['distractor']
    elif stim_type == 'target_nums':
        target_stream = 'targets'
        distractor_stream = 'distractors'
    elif stim_type == 'non_targets':
        target_stream = 'nt_target'
        distractor_stream = 'nt_distractor'
    return target_stream, distractor_stream


def envs_onsets(predictor, key):
    target_pred_arrays = []
    distractor_pred_arrays = []
    envelopes_dir = predictor_dir / predictor
    for sub_folder in envelopes_dir.iterdir():
        for cond_folder in sub_folder.iterdir():
            if condition not in cond_folder.name:
                continue  # skip wrong condition
            target_stream, distractor_stream = select_folder(stim_type)
            for stream_folder in cond_folder.iterdir():
                if target_stream in stream_folder.name:
                    for f in stream_folder.glob('*concat.npz'):
                        target_array = np.load(f, allow_pickle=True)[key]
                        target_pred_arrays.append(target_array)
                        print("Loaded target:", f)
                elif distractor_stream in stream_folder.name:
                    for f in stream_folder.glob('*concat.npz'):
                        distractor_array = np.load(f, allow_pickle=True)[key]
                        distractor_pred_arrays.append(distractor_array)
                        print("Loaded distractor:", f)
    return target_pred_arrays, distractor_pred_arrays, target_stream, distractor_stream


target_env_arrays, distractor_env_arrays, target_stream, distractor_stream = envs_onsets(predictor='envelopes', key='envelopes')
target_onsets_arrays, distractor_onsets_arrays, _, _= envs_onsets(predictor='binary_weights', key='onsets')

# normalizing envelope arrays, as they are continuous:

def normalize_envelopes(predictor_array):
    normalized = []
    min_std = 1e-6

    for array in predictor_array:
        std = array.std()
        nonzero_ratio = np.count_nonzero(array) / len(array)

        if nonzero_ratio > 0.5:
            print('Envelopes: Dense predictor, applying z-score.')
            mean = array.mean()
            normed = (array - mean) / std if std > min_std else array - mean

        elif nonzero_ratio > 0:
            print('Envelopes: Sparse predictor, mean-centering non-zero entries.')
            mask = array != 0
            mean = array[mask].mean()
            normed = array.copy()
            normed[mask] -= mean

        elif nonzero_ratio == 0:
            print('Envelopes: All zeros, returning unchanged.')
            normed = array

        else:
            print('Envelopes: No normalization applied.')
            normed = array

        normalized.append(normed)

    return normalized


target_env_arrays_norm = normalize_envelopes(target_env_arrays)
distractor_env_arrays_norm = normalize_envelopes(distractor_env_arrays)

# phonemes:
def phonemes():
    target_arr_list = []
    distractor_arr_list = []
    phonemes_dir = predictor_dir / 'phonemes'
    for cond_folders in phonemes_dir.iterdir():
        if condition in cond_folders.name:
            for sub_folders in cond_folders.iterdir():
                for stim_folders in sub_folders.iterdir():
                    if stim_type in stim_folders.name:
                        for files in stim_folders.iterdir():
                            phonemes = np.load(files, allow_pickle=True)
                            target_arr = phonemes['target']
                            distractor_arr = phonemes['distractor']
                            target_arr_list.append(target_arr)
                            distractor_arr_list.append(distractor_arr)
    return target_arr_list, distractor_arr_list


target_phonemes, distractor_phonemes = phonemes()


# mask predictors:
def mask_predictors(target_predictor_list, distractor_predictor_list):
    masked_predictor_dict = {}
    for target_array, distractor_array, bad_series, (sub, eeg_masked) in zip(target_predictor_list, distractor_predictor_list,
                                                               bads, eeg_masked_dict.items()):
        print(target_array, sub)
        target_masked = target_array[bad_series == 0]
        distractor_masked = distractor_array[bad_series == 0]

        # check if len matches with masked eeg
        logging.info(f"EEG length: {len(eeg_masked[0, :])}, target pred length: {len(target_masked)}")
        logging.info(f"EEG length: {len(eeg_masked[0, :])}, distractor pred length: {len(distractor_masked)}")
        masked_predictor_dict[sub] = {'target': target_masked, 'distractor':distractor_masked}
    return masked_predictor_dict


# masked_phonemes_dict = mask_predictors(target_phonemes, distractor_phonemes) # already masked
masked_phonemes_dict = {}
for sub, arr_target, arr_distractor in zip(eeg_dict.keys(), target_phonemes, distractor_phonemes):
    masked_phonemes_dict[sub] = {'target': arr_target, 'distractor': arr_distractor}

masked_env_dict = mask_predictors(target_env_arrays_norm, distractor_env_arrays_norm)
masked_onsets_dict = mask_predictors(target_onsets_arrays, distractor_onsets_arrays)

# Design Matrices:
# Target:
target_dict = {}
distractor_dict = {}

for sub, onsets, envelopes, phonemes, eeg_data in zip(sub_list, masked_onsets_dict.values(),
                                            masked_env_dict.values(), masked_phonemes_dict.values(), eeg_masked_dict.values()):

    target_dict[sub] = {'eeg': eeg_data.T, 'onsets': onsets['target'], 'envelopes': envelopes['target'],
                        'phonemes': phonemes['target']}
    distractor_dict[sub] = {'eeg': eeg_data.T, 'onsets': onsets['distractor'], 'envelopes': envelopes['distractor'],
                            'phonemes': phonemes['distractor']}


# now optimize regularization parameter:
lambdas = np.logspace(-2, 2, 20)  # based on prev literature

# Stack predictors for the target stream
scores_dict = {}
for sub, target_data, distractor_data in zip(sub_list, target_dict.values(), distractor_dict.values()):
    eeg = target_data['eeg']
    X_target = np.column_stack([target_data['onsets'], target_data['envelopes'], target_data['phonemes']])
    X_distractor = np.column_stack([distractor_data['onsets'], distractor_data['envelopes'], distractor_data['phonemes']])
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
    n_samples = sfreq * 60
    total_samples = len(predictors_stacked)
    n_folds = total_samples // n_samples
    # Split predictors and EEG into subject chunks
    X_folds = np.array_split(predictors_stacked, n_folds)
    Y_folds = np.array_split(Y_eeg, n_folds)

    import random

    random.seed(42)

    # Choose 50% of folds for lambda optimization
    subset_fraction = 0.5
    n_subset = int(n_folds * subset_fraction)
    subset_indices = random.sample(range(n_folds), n_subset)

    X_folds_subset = [X_folds[i] for i in subset_indices]
    Y_folds_subset = [Y_folds[i] for i in subset_indices]

    lambdas = np.logspace(-2, 2, 20)  # based on prev literature
    tmin = - 0.1
    tmax = 1.0

    def optimize_lambda(X_folds, Y_folds, sfreq, tmin, tmax, lambdas):
        best_lambda, best_score = None, -np.inf
        scores = []
        print(f"Running lambda optimization across {len(lambdas)} values...")
        for lmbda in lambdas:
            fwd_trf = TRF(direction=1)
            r = crossval(fwd_trf, X_folds, Y_folds, sfreq, tmin, tmax, lmbda)
            mean_r = r.mean()
            scores.append(mean_r)
            print(f"lambda={lmbda:.2e}, mean r={mean_r:.3f}")

            if mean_r > best_score:
                best_lambda, best_score = lmbda, mean_r

        print(f"Best lambda: {best_lambda:.2e} (mean r = {best_score:.3f})")
        # plt.semilogx(lambdas, scores, marker="o")
        # plt.xlabel("lambda")
        # plt.ylabel("mean r")
        # plt.show()
        return best_lambda, scores

    best_regularization, scores = optimize_lambda(X_folds_subset, Y_folds_subset, sfreq=sfreq, tmin=-0.1,
                                          tmax=1.0, lambdas=lambdas)

    scores_dict[sub] = {'scores': scores, 'best_regularization': best_regularization}



