# TRF modeling script for a **single condition** (e.g., A1, A2, E1, E2)
# To run: just set `condition = 'a1'` or another valid label

import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mtrf.model import TRF
from mtrf.stats import crossval, pearsonr
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

# --------------- CONFIG -----------------
default_path = Path.cwd()
predictors_path = default_path / 'data/eeg/predictors'
eeg_results_path = default_path / 'data/eeg/preprocessed/results'
sfreq = 125
condition = 'a1'  # Change to 'a2', 'e1', or 'e2'
plane = 'azimuth' if 'a' in condition else 'elevation'
n = 0
stream_type1 = 'stream1'
stream_type2 = 'stream2'
stim1 = 'target_stream' if condition in ['a1', 'e1'] else 'distractor_stream'
stim2 = 'distractor_stream' if stim1 == 'target_stream' else 'target_stream'
predictor_names = ['binary_weights', 'envelopes', 'overlap_ratios', 'events_proximity', 'events_proximity']
predictor_name = predictor_names[n]  # Change if needed
pred_types = ['onsets', 'envelopes', 'overlap_ratios', 'events_proximity_pre', 'events_proximity_post']
pred_type = pred_types[n]
predictor_dir = predictors_path / predictor_name

# ---------- LOAD EEG + PREDICTORS ------------
def get_eeg_files():
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
                            eeg.resample(sfreq)
                            eeg.pick_types(eeg=True)
                            eeg.filter(None, 30)
                            eeg_files[folders.name] = eeg
    return eeg_files

def mask_bad_segments(eeg_files):
    eeg_clean = {}
    for sub, raw in eeg_files.items():
        data = raw.get_data()
        mask_file = predictors_path / 'bad_segments' / sub / condition / 'concat.npy'
        if mask_file.exists():
            bad_series = np.load(mask_file)['bad_series']
            good = bad_series == 0
            data = data[:, good]
        # Z-score
        data = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
        eeg_clean[sub] = data
    return eeg_clean

def load_predictors():
    pred_dict = {}
    for sub_dir in predictor_dir.iterdir():
        if condition in sub_dir.name:
            sub = sub_dir.parent.name
            for stim_type in sub_dir.iterdir():
                if stim_type.name in [stream_type1, stream_type2]:
                    for file in stim_type.iterdir():
                        if 'concat' in file.name:
                            data = np.load(file)[pred_type]
                            if sub not in pred_dict:
                                pred_dict[sub] = {}
                            pred_dict[sub][stim_type.name] = data
    return pred_dict

def apply_mask(pred_dict):
    masked = {}
    for sub, streams in pred_dict.items():
        mask_file = predictors_path / 'bad_segments' / sub / condition / 'concat.npy'
        if mask_file.exists():
            good = np.load(mask_file)['bad_series'] == 0
        else:
            good = np.ones_like(next(iter(streams.values())), dtype=bool)
        masked[sub] = {
            stim1: (streams[stream_type1] if stim1 == 'target_stream' else streams[stream_type2])[good],
            stim2: (streams[stream_type2] if stim1 == 'target_stream' else streams[stream_type1])[good]
        }
    return masked

def collect_arrays(eeg_clean, predictor_masked):
    eeg_all = []
    s1_all = []
    s2_all = []
    for sub in eeg_clean:
        eeg_all.append(eeg_clean[sub])
        s1_all.append(predictor_masked[sub][stim1])
        s2_all.append(predictor_masked[sub][stim2])
    eeg = np.concatenate(eeg_all, axis=1).T
    s1 = np.concatenate(s1_all)
    s2 = np.concatenate(s2_all)
    return eeg, s1, s2

# ---------- RUN -------------------
eeg_files = get_eeg_files()
eeg_clean = mask_bad_segments(eeg_files)
predictors_raw = load_predictors()
predictors_masked = apply_mask(predictors_raw)
eeg_all, s1_array, s2_array = collect_arrays(eeg_clean, predictors_masked)

# Make stream2 orthogonal to stream1
model = LinearRegression().fit(s1_array.reshape(-1, 1), s2_array)
s2_ortho = s2_array - model.predict(s1_array.reshape(-1, 1))

# Stack predictors
predictors_stacked = np.vstack([s1_array, s2_ortho]).T

# Collinearity check + ortho
X_df = pd.DataFrame(predictors_stacked, columns=[stim1, stim2])
X_df = sm.add_constant(X_df)
vif = pd.Series([variance_inflation_factor(X_df.values, i) for i in range(X_df.shape[1])], index=X_df.columns)
print(vif)

# ---------- TRF MODEL ------------
n_samples = sfreq * 60
X_folds = np.array_split(predictors_stacked, len(eeg_all) // n_samples)
Y_folds = np.array_split(eeg_all, len(eeg_all) // n_samples)

lambdas = np.logspace(-2, 2, 20)


def optimize_lambda(X_folds, Y_folds):
    best_lambda, best_score = 0, -np.inf
    for lmbda in lambdas:
        score = crossval(TRF(direction=1), X_folds, Y_folds, sfreq, -0.1, 1.0, lmbda).mean()
        if score > best_score:
            best_lambda, best_score = lmbda, score
    print(f'Best lambda: {best_lambda:.3f}, mean r: {best_score:.3f}')
    return best_lambda


best_lambda = optimize_lambda(X_folds, Y_folds)

trf = TRF(direction=1)
trf.train(X_folds, Y_folds, sfreq, -0.1, 1.0, best_lambda, seed=42)

prediction, r = trf.predict(predictors_stacked, eeg_all)
print(f'Full model correlation: {r.mean():.3f}')

# ---------- SAVE & PLOT ----------
weights = trf.weights
lags = np.linspace(-0.1, 1.0, weights.shape[1])
save_dir = default_path / 'data/eeg/trf/trf_testing' / condition / 'data'
plot_dir = default_path / 'data/eeg/trf/trf_testing' / condition / 'figures'
save_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)


np.savez(save_dir / f'{condition}_{predictor_name}_TRF_results.npz',
         weights=weights,
         r=r,
         best_lambda=best_lambda,
         time_lags=lags,
         predictor_names=np.array([stim1, stim2]),
         condition=condition)

for i, name in enumerate([stim1, stim2]):
    trf_weights = weights[i]
    plt.figure(figsize=(8, 4))
    for ch in range(trf_weights.shape[0]):
        plt.plot(lags, trf_weights[ch], alpha=0.3)
    plt.title(f'TRF: {name}')
    plt.xlabel('Lag (s)')
    plt.ylabel('Weight')
    plt.tight_layout()
    plt.savefig(plot_dir / f'{condition}_TRF_{name}.png', dpi=300)
    plt.close()