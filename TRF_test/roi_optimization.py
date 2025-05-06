import os
from pathlib import Path
import mne
from mtrf import TRF
from TRF_test.TRF_test_config import temporal_roi, frontal_roi, occipitoparietal_right, occipitoparietal_left
import pandas as pd
import numpy
import os
import random
import pandas
import numpy as np
from mtrf.model import TRF
from mtrf.stats import pearsonr
import pickle
import json
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch



"""
Using this script we are going to determine the roi, by checking the Pearson's r for each sensor. 
"""

# directories
condition = 'a2'
from TRF_test.TRF_test_config import azimuth_subs
default_path = Path.cwd()
predictors_path = default_path / 'data/eeg/predictors'
eeg_results_path = default_path / 'data/eeg/preprocessed/results'
sfreq = 125

eeg_files = {}
for folders in eeg_results_path.iterdir():
    if 'sub' in folders.name:
        sub_data = []
        for files in folders.iterdir():
            if 'ica' in files.name:
                for data in files.iterdir():
                    if condition in data.name:
                        eeg = mne.io.read_raw_fif(data, preload=True)
                        eeg.set_eeg_reference('average')
                        eeg.resample(sfreq=sfreq)
                        sub_data.append(eeg)
        eeg_files[folders.name] = sub_data

eeg_concat_list = {}

for sub, sub_list in eeg_files.items():
    eeg_concat = mne.concatenate_raws(sub_list)
    eeg_concat.resample(sfreq)
    eeg_concat.pick(frontal_roi)
    eeg_concat.filter(l_freq=None, h_freq=15)
    eeg_concat_list[sub] = eeg_concat

for sub in eeg_concat_list:
    eeg_concat = eeg_concat_list[sub]
    eeg_data = eeg_concat.get_data()

    sub_bad_segments_path = predictors_path / 'bad_segments' / sub / condition
    bad_segments_found = False

    if sub_bad_segments_path.exists():
        for file in sub_bad_segments_path.iterdir():
            if 'concat.npy' in file.name:
                bad_segments = np.load(file)
                bad_segments_found = True
                bad_series = bad_segments['bad_series']
                good_samples = bad_series == 0  # Boolean mask
                print(f"Loaded bad segments for {sub} {condition}.")
                eeg_clean = eeg_data[:, good_samples]
                # z-scoring..
                eeg_clean = (eeg_clean - eeg_clean.mean(axis=1, keepdims=True)) / eeg_clean.std(axis=1, keepdims=True)
                print(f"{sub}: Clean EEG shape = {eeg_clean.shape}")
                break
    else:
        print(f"No bad segments found for {sub} {condition}, assuming all samples are good.")
        eeg_len = eeg_concat.n_times
        good_samples = np.ones(eeg_len, dtype=bool)
        eeg_clean = eeg_data[:, good_samples]
        # z-scoring..
        eeg_clean = (eeg_clean - eeg_clean.mean(axis=1, keepdims=True)) / eeg_clean.std(axis=1, keepdims=True)
        print(f"{sub}: Clean EEG shape = {eeg_clean.shape}")


envelope_predictor = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/predictors/envelopes')
stim1 = 'stream1'
stim2 = 'stream2'

predictor_dict = {}

for files in envelope_predictor.iterdir():
    if 'sub' in files.name:
        sub_name = files.name  # e.g., "sub01"
        stream1_data, stream2_data = None, None

        for file in files.iterdir():
            if condition in file.name:
                for stim_type in file.iterdir():
                    if stim_type.name == stim1:
                        for array in stim_type.iterdir():
                            if 'concat' in array.name:
                                stream1_data = np.load(array)
                                stream1_data = stream1_data['envelopes']
                    elif stim_type.name == stim2:
                        for array in stim_type.iterdir():
                            if 'concat' in array.name:
                                stream2_data = np.load(array)
                                stream2_data = stream2_data['envelopes']

        if stream1_data is not None and stream2_data is not None:
            predictor_dict[sub_name] = {
                'stream1': stream1_data,
                'stream2': stream2_data
            }
            print(f"Loaded envelopes for {sub_name}: {stream1_data.shape}, {stream2_data.shape}")
        else:
            print(f"Missing envelope(s) for {sub_name} {condition}")

predictor_dict_masked = {}

for sub, sub_dict in predictor_dict.items():
    sub_bad_segments_path = predictors_path / 'bad_segments' / sub / condition
    bad_segments_found = False
    good_samples = None  # default fallback

    # --- Load bad segment mask if available
    if sub_bad_segments_path.exists():
        for file in sub_bad_segments_path.iterdir():
            if 'concat.npy' in file.name:
                bad_segments = np.load(file)
                bad_segments_found = True
                bad_series = bad_segments['bad_series']
                good_samples = bad_series == 0  # Boolean mask
                print(f"Loaded bad segments for {sub} - {condition}.")
                break  # Stop after finding the correct file

    sub_masked = {}

    for stream_name, stream_array in sub_dict.items():
        if good_samples is not None and len(good_samples) == len(stream_array):
            stream_array_masked = stream_array[good_samples]
            stram_array_clean = (stream_array_masked - stream_array_masked.mean()) / stream_array_masked.std()
        else:
            stream_array_masked = stream_array  # use full array if no mask found or mismatched
            stram_array_clean = (stream_array_masked - stream_array_masked.mean()) / stream_array_masked.std()
        sub_masked[stream_name] = stram_array_clean

    predictor_dict_masked[sub] = sub_masked

# concat all subs eegs together in order,
# same for envelopes stream1 and stream2 respectively.
# then vstack stream1 and stream2
# Clean EEG

all_eeg_clean = []
all_stream1 = []
all_stream2 = []

for sub in eeg_concat_list.keys():
    # Mask envelopes
    eeg = eeg_concat_list[sub]
    env = predictor_dict[sub]
    stream1 = env['stream1']
    stream2 = env['stream2']

    all_eeg_clean.append(eeg)
    all_stream1.append(stream1)
    all_stream2.append(stream2)

# Concatenate across subjects
eeg_all = mne.concatenate_raws(all_eeg_clean)       # shape: (total_samples, channels)
stream1_all = np.concatenate(all_stream1, axis=0)     # shape: (total_samples,)
stream2_all = np.concatenate(all_stream2, axis=0)     # shape: (total_samples,)

from sklearn.linear_model import LinearRegression

# Make stream2 orthogonal to stream1
model = LinearRegression().fit(stream1_all.reshape(-1, 1), stream2_all)
stream2_ortho = stream2_all - model.predict(stream1_all.reshape(-1, 1))
# stream1_ortho = stream1_all - model.predict(stream2_all.reshape(-1, 1))
# Stack predictors (final TRF design matrix)
predictors_stacked = np.vstack([stream1_all, stream2_ortho]).T  # shape: (samples, 2)
# predictors_stacked = np.vstack([stream1_ortho, stream2_all]).T  # shape: (samples, 2)
eeg_data_all = eeg_all._data.T
print(f"EEG shape: {eeg_data_all.shape}, Predictors shape: {predictors_stacked.shape}")

# checking collinearity:
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

X = pd.DataFrame(predictors_stacked, columns=['stream1', 'stream2'])
X = sm.add_constant(X)  # Add intercept for VIF calc
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
print(vif)

# split into trials:
n_folds = 5
X_folds = np.array_split(predictors_stacked, n_folds)
Y_folds = np.array_split(eeg_data_all, n_folds)
lambdas = np.logspace(-6, 2, 20)
scores = []
fwd_trf = TRF(direction=1)
from mtrf.stats import crossval

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


best_lambda = optimize_lambda(X_folds, Y_folds, fs=sfreq, tmin=0.1, tmax=1.0, lambdas=lambdas)

trf = TRF(direction=1)
trf.train(predictors_stacked, eeg_data_all, fs=sfreq, tmin=0.1, tmax=1.0, regularization=best_lambda)
prediction, r = trf.predict(predictors_stacked, eeg_data_all)
print(f"Full model correlation: {r.round(3)}")

r_crossval = crossval(trf, X_folds, Y_folds, fs=sfreq, tmin=0.1, tmax=1.0, regularization=best_lambda)
print(f"mean correlation between actual and predicted response: {r_crossval.mean().round(3)}")

predictor_names = ['stream1', 'stream2']  # or however many you have
weights = trf.weights  # shape: (n_features, n_lags, n_channels)
time_lags = np.linspace(0.1, 1.0, weights.shape[1])  # time axis

# Loop and plot
for i, name in enumerate(predictor_names):
    plt.figure(figsize=(8, 4))
    trf_weights = weights[i].T  # shape: (n_channels, n_lags)

    for ch in range(trf_weights.shape[0]):
        plt.plot(time_lags, trf_weights[ch], alpha=0.4)

    plt.title(f'TRF for {name}')
    plt.xlabel('Time lag (s)')
    plt.ylabel('Amplitude')
    plt.plot([], [], ' ', label=f'λ = {best_lambda:.2f} (CV) = {r_crossval:.3f}')  # show both
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()

plt.close('all')