from pathlib import Path
import os
import mne
import numpy as np
import pandas as pd

from TRF_predictors.overlap_ratios import load_eeg_files
from TRF_predictors.config import sub, sfreq, condition

frontal_roi = ['F3', 'Fz', 'F4', 'FC1', 'FC2']
default_path = Path.cwd()
eeg_path = default_path / 'data/eeg/preprocessed/results'
predictors_path = default_path / 'data/eeg/predictors'

predictors_list = [predictor.name for predictor in predictors_path.iterdir()]
# choose a predictor
predictor_name = predictors_list[0] # for bad segments


eeg_files_list, eeg_events_list = load_eeg_files(sub=sub, condition=condition, sfreq=sfreq, results_path=eeg_path)
eeg_concat = mne.concatenate_raws(eeg_files_list)
eeg_concat.pick(frontal_roi)
eeg_data = eeg_concat.get_data()
type = 'stream1'
sub_bad_segments_path = predictors_path / predictor_name / sub / condition

for files in sub_bad_segments_path.iterdir():
    if 'concat' in files.name:
        bad_segments = np.load(files)

bad_series = bad_segments['bad_series']
good_samples = bad_series == 0  # or use .astype(bool) if needed


selected_predictor = predictors_list[1]
chosen_predictor_path = predictors_path / selected_predictor / sub / condition / f'{sub}_{condition}_weights_series_concat.npz'
weight_series = np.load(chosen_predictor_path)
weight_keys = list(weight_series.keys())


def filter_bad_segments(weight_series, predictor_key=None):
    weight_series_data = weight_series[predictor_key]

    eeg_clean = eeg_data[:, good_samples]       # still 2D: (n_channels, good_samples)
    predictor_clean = weight_series_data[good_samples]   # now 1D: (good_samples,)
    predictor_clean = (predictor_clean - predictor_clean.mean()) / predictor_clean.std()
    # z-scoring data...
    print(eeg_clean.shape)
    print(predictor_clean.shape)
    return eeg_clean, predictor_clean


eeg_clean1, predictor_clean_onsets1 = filter_bad_segments(weight_series, predictor_key=weight_keys[0]) # onsets1
eeg_clean2, predictor_clean_onsets2 = filter_bad_segments(weight_series, predictor_key=weight_keys[1]) # onsets2

tmin = -0.1
tmax = 0.8

def set_lags(tmin, tmax, predictor_clean, eeg_clean):
    lags = np.arange(round(tmin * sfreq), round(tmax * sfreq) + 1)
    time_lags_ms = lags * 1000 / sfreq  # convert to milliseconds
    n_lags = len(lags)
    n_samples = len(predictor_clean)
    X = np.zeros((n_samples, n_lags))

    for i, lag in enumerate(lags):
        if lag < 0:
            X[-lag:, i] = predictor_clean[:n_samples + lag]
        elif lag > 0:
            X[:n_samples - lag, i] = predictor_clean[lag:]
        else:
            X[:, i] = predictor_clean

    Y = eeg_clean.T  # shape = (samples, channels)
    return X, Y, time_lags_ms


X1, Y1, time_lags_ms1 = set_lags(tmin, tmax, predictor_clean_onsets1, eeg_clean1)
X2, Y2, time_lags_ms2 = set_lags(tmin, tmax, predictor_clean_onsets2, eeg_clean2)


from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

alpha = 0.1


def ridge_reg(alpha, X, Y):
    model = Ridge(alpha=alpha)
    trf_weights = []
    r2_scores = []

    for ch in range(Y.shape[1]):
        model.fit(X, Y[:, ch])
        trf_weights.append(model.coef_)
        r2_scores.append(r2_score(Y[:, ch], model.predict(X)))



    print("X shape:", X.shape)
    print("X mean:", X.mean())
    print("X std:", X.std())
    print("Unique values in X:", np.unique(X))
    return trf_weights, r2_scores


trf_weights1, r2_scores1 = ridge_reg(alpha, X1, Y1)
trf_weights2, r2_scores2 = ridge_reg(alpha, X2, Y2)

import matplotlib.pyplot as plt

# Convert to NumPy arrays
trf_weights1 = np.array(trf_weights1)
trf_weights2 = np.array(trf_weights2)

# Average across channels
trf_mean1 = trf_weights1.mean(axis=0)
trf_mean2 = trf_weights2.mean(axis=0)

plt.figure(figsize=(10, 5))
plt.plot(time_lags_ms1, trf_mean1, label='Stream 1 (target)', linewidth=2)
plt.plot(time_lags_ms2, trf_mean2, label='Stream 2 (distractor)', linewidth=2)
plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
plt.axvline(0, color='black', linestyle='--', linewidth=0.7)
plt.xlabel('Time Lag (ms)')
plt.ylabel('TRF Weight')
plt.title(f'TRF: {sub} - {condition}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()