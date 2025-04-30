from pathlib import Path
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import mtrf
from mtrf import TRF
from mtrf.stats import crossval
from TRF_predictors.overlap_ratios import load_eeg_files

default_path = Path.cwd()
sub = 'sub29'
condition = 'a1'
sfreq = 125
predictors_list = ['binary_weights', 'envelopes', 'events_proximity', 'overlap_ratios']
predictors_path = default_path / 'data' / 'eeg' / 'predictors'
eeg_path = default_path / 'data/eeg/preprocessed/results'
stim_type = 'stream1'


# load eeg_files:
frontal_roi = ['F3', 'Fz', 'F4', 'FC1', 'FC2']
temporal_roi = ['T7', 'T8', 'TP7', 'TP8', 'FT7', 'FT8', 'C5', 'C6', 'P7', 'P8']


eeg_files_list, _ = load_eeg_files(sub=sub, condition=condition, results_path=eeg_path, sfreq=sfreq)
eeg_concat = mne.concatenate_raws(eeg_files_list)
eeg_concat.pick(temporal_roi)
eeg_data = eeg_concat.get_data()

# --- Load bad segments ---
sub_bad_segments_path = predictors_path / 'bad_segments' / sub / condition
bad_segments_found = False
if sub_bad_segments_path.exists():
    for file in sub_bad_segments_path.iterdir():
        if 'concat' in file.name:
            bad_segments = np.load(file)
            bad_segments_found = True
            bad_series = bad_segments['bad_series']
            good_samples = bad_series == 0  # good samples only
            print(f"Loaded bad segments for {sub} {condition}.")
            eeg_clean = eeg_data[:, good_samples]  # still 2D: (n_channels, good_samples)
            print(eeg_clean.shape)
            signal_power = np.var(eeg_clean.mean(axis=0))  # average across channels
            total_power = np.var(eeg_clean)
            snr_ratio = signal_power / (total_power - signal_power)
            print("SNR ratio:", snr_ratio)
            break  # stop after finding the file
else:
    print(f"No bad segments found for {sub} {condition}, assuming all samples are good.")
    # Create "fake good samples" (all good)
    eeg_len = eeg_concat.n_times
    good_samples = np.ones(eeg_len, dtype=bool)
    eeg_clean = eeg_data[:, good_samples]  # still 2D: (n_channels, good_samples)
    print(eeg_clean.shape)
    signal_power = np.var(eeg_clean.mean(axis=0))  # average across channels
    total_power = np.var(eeg_clean)
    snr_ratio = signal_power / (total_power - signal_power)
    print("SNR ratio:", snr_ratio)

# load all predictors for chosen stim_type and sub:
all_predictors = {}

for current_predictor in predictors_list:
    predictor_path = predictors_path / current_predictor / sub / condition / stim_type
    for files in predictor_path.iterdir():
        if 'concat' in files.name:
            predictor_loaded = np.load(files, allow_pickle=True)
            predictor_keys = list(predictor_loaded.keys())

            if current_predictor == 'events_proximity':
                # Store both pre and post separately
                predictor_array_pre = predictor_loaded[predictor_keys[0]]
                predictor_clean_pre = (predictor_array_pre - predictor_array_pre.mean()) / predictor_array_pre.std()
                predictor_bad_clean_pre = predictor_clean_pre[good_samples]  # now 1D: (good_samples,)

                predictor_array_post = predictor_loaded[predictor_keys[1]]
                predictor_clean_post = (predictor_array_post - predictor_array_post.mean()) / predictor_array_post.std()
                predictor_bad_clean_post = predictor_clean_post[good_samples]  # now 1D: (good_samples,)

                all_predictors[f'{current_predictor}_pre'] = predictor_bad_clean_pre
                all_predictors[f'{current_predictor}_post'] = predictor_bad_clean_post
            else:
                predictor_array = predictor_loaded[predictor_keys[0]]
                predictor_clean = (predictor_array - predictor_array.mean()) / predictor_array.std()
                predictor_clean_bad = predictor_clean[good_samples]
                all_predictors[current_predictor] = predictor_clean_bad

# vstack predictors:
# Stack all cleaned and masked predictors into a composite design matrix
design_matrix = np.column_stack(list(all_predictors.values()))
print("Composite design matrix shape:", design_matrix.shape)
eeg_clean = eeg_clean.T

# === SANITY CHECK: PLOT EEG VS PREDICTORS ===
plt.figure(figsize=(15, 5))
plt.plot(eeg_clean.T[:, 0], label='EEG (channel 0)', alpha=0.6)
if 'envelopes' in all_predictors:
    plt.plot(all_predictors['envelopes'], label='Envelope', alpha=0.6)
# if 'binary_weights' in all_predictors:
#     plt.plot(all_predictors['binary_weights'], label='Semantic Weights', alpha=0.6)
plt.legend()
plt.title(f'Sanity check: EEG vs Predictors (Sub {sub})')
plt.xlabel('Samples')
plt.tight_layout()
plt.show()

# split into trials:
n_folds = 5
X_folds = np.array_split(design_matrix, n_folds)
Y_folds = np.array_split(eeg_clean, n_folds)
lambdas = np.logspace(1, 8, 20)


def optimize_lambda(predictor, eeg, fs, tmin, tmax, lambdas):
    scores = []
    fwd_trf = TRF(direction=1)
    for l in lambdas:
        r = crossval(fwd_trf, predictor, eeg, fs, tmin, tmax, l)
        scores.append(r.mean())
    best_idx = np.argmax(scores)
    best_lambda = lambdas[best_idx]
    print(f"Best lambda: {best_lambda:.2e} (mean r = {scores[best_idx]:.3f})")
    return best_lambda


best_lambda = optimize_lambda(X_folds, Y_folds, fs=sfreq, tmin=0, tmax=0.8, lambdas=lambdas)
trf = TRF(direction=1)
trf.train(design_matrix, eeg_clean, fs=sfreq, tmin=0, tmax=0.8, regularization=best_lambda)
prediction, r = trf.predict(design_matrix, eeg_clean)
print(f"Full model correlation: {r.round(3)}")

r_crossval = crossval(trf, X_folds, Y_folds, fs=sfreq, tmin=0, tmax=0.8, regularization=best_lambda)
print(f"mean correlation between actual and predicted response: {r_crossval.mean().round(3)}")

for i, name in enumerate(all_predictors.keys()):
    trf.plot(feature=i, kind='line')
    plt.title(f'TRF for {name}')
    plt.show()

# checking collinearity:
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = pd.DataFrame(design_matrix, columns=all_predictors.keys())
X = sm.add_constant(X)  # Add intercept for VIF calc
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
print(vif)

# todo: collinearity is supposedly ok, SNR okay. so what is the issue?