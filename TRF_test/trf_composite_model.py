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
from scipy.signal import welch
import copy
from TRF_test.TRF_test_config import occipitoparietal_left, occipitoparietal_right, frontal_roi, temporal_roi


default_path = Path.cwd()
sub = 'sub29'
condition = 'a1'
sfreq = 125
predictors_list = ['binary_weights', 'envelopes', 'events_proximity', 'overlap_ratios']
predictors_path = default_path / 'data' / 'eeg' / 'predictors'
eeg_path = default_path / 'data/eeg/preprocessed/results'
stim_type = 'stream1'


# load eeg_files:
eeg_files_list, _ = load_eeg_files(sub=sub, condition=condition, results_path=eeg_path, sfreq=sfreq)
eeg_concat = mne.concatenate_raws(eeg_files_list)
eeg_concat.pick(frontal_roi)
eeg_concat_copy = copy.deepcopy(eeg_concat)
eeg_concat_copy.filter(l_freq=4, h_freq=7)
eeg_data = eeg_concat_copy.get_data()


# === COMPUTE SNR ===
def compute_snr(data, eeg):
    # Signal = variance of the mean signal across time (averaged across channels)
    fs = eeg.info['sfreq']

    f, psd = welch(data[0], fs=fs)
    signal_band = (f > 8) & (f < 13)
    noise_band = (f > 20) & (f < 40)
    snr = psd[signal_band].mean() / psd[noise_band].mean()
    print(f"SNR ratio: {snr}")
    return snr


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
            snr = compute_snr(eeg_clean, eeg_concat)
            break  # stop after finding the file
else:
    print(f"No bad segments found for {sub} {condition}, assuming all samples are good.")
    # Create "fake good samples" (all good)
    eeg_len = eeg_concat.n_times
    good_samples = np.ones(eeg_len, dtype=bool)
    eeg_clean = eeg_data[:, good_samples]  # still 2D: (n_channels, good_samples)
    print(eeg_clean.shape)
    snr = compute_snr(eeg_clean, eeg_concat)

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

# checking collinearity:
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = pd.DataFrame(design_matrix, columns=all_predictors.keys())
X = sm.add_constant(X)  # Add intercept for VIF calc
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
print(vif)

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
    trf.plot(feature=i, kind='line',)
    plt.title(f'TRF for {name}')
    plt.plot([], [], ' ', label=f'λ = {best_lambda:.2f}')
    plt.legend(loc='upper right', fontsize=10)
    plt.show()

#
time_windows = [(-0.1, 0.6), (0, 0.5), (0, 0.8), (-0.2, 0.8), (0.1, 0.8), (0.2, 0.8), (0.2, 0.9), (0.2, 1.2)]
best_score = -np.inf
best_window = None

for tmin, tmax in time_windows:
    r_score = crossval(trf, X_folds, Y_folds, fs=sfreq, tmin=tmin, tmax=tmax, regularization=best_lambda).mean()
    print(f"tmin={tmin}, tmax={tmax}, r={r_score:.4f}")
    if r_score > best_score:
        best_score = r_score
        best_window = (tmin, tmax)

print(f"Best lag window: {best_window}, r = {best_score:.4f}")

# todo: collinearity is supposedly ok, SNR okay. so what is the issue?
#  -> design too complex, linear model insufficient most likely

envelope_raw = all_predictors['envelopes']
import scipy.signal as signal

# Smooth with a low-pass filter (e.g., 15 Hz)
nyq = 0.5 * sfreq
cutoff = 15
b, a = signal.butter(4, cutoff / nyq, btype='low')
envelope_smooth = signal.filtfilt(b, a, envelope_raw)
time = np.arange(len(envelope_raw)) / sfreq  # time in seconds

plt.figure(figsize=(10, 4))
plt.plot(time, envelope_raw, label='Raw Envelope', alpha=0.7)
plt.plot(time, envelope_smooth, label='Smoothed Envelope', linewidth=2)
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.title('Speech Envelope Over Time')
plt.legend()
plt.tight_layout()
plt.show()


# todo: idea -> get optimal lambda based on all subs data, and then use this lambda on single sub level
# todo: combine stream 1 and stream 2 envelopes predictors and run a simple TRF test
# todo: can add lines to find best ROI -> extract the R squared for each of the channels