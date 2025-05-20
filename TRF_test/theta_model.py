import numpy as np
from scipy.signal import resample
from mtrf.model import TRF
from mtrf.stats import pearsonr
import mne
import os
from pathlib import Path
from TRF_test.TRF_test_config import frontal_roi
# ----------------------
# CONFIGURATION
# ----------------------
sfreq = 125   # Target sampling frequency
n_phase_steps = 20  # Number of phase offsets (0–2π)
lag_start, lag_end = -100, 500  # ms for TRF

# ----------------------
# LOAD DATA BLOCK-WISE
# ----------------------

def load_eeg_blocks(condition, subject_folder):
    block_data = {}

    for folder in subject_folder.iterdir():
        if 'ica' in folder.name:
            for fif_file in folder.iterdir():
                if condition in fif_file.name:
                    sub = fif_file.name[0:5]  # Adjust if needed
                    raw = mne.io.read_raw_fif(fif_file, preload=True)
                    raw.filter(l_freq=None, h_freq=30)
                    raw.set_eeg_reference('average')
                    raw.resample(sfreq)
                    raw.pick(frontal_roi)
                    if sub not in block_data:
                        block_data[sub] = []

                    block_data[sub].append(raw)

    return block_data  # {sub_id: [block1, block2, ..., block5]}

# ----------------------
# CREATE THETA WAVES
# ----------------------

def generate_theta_candidates(block_len, sfreq, freq_range, n_phase_steps):
    t = np.arange(block_len) / sfreq
    frequencies = np.arange(freq_range[0], freq_range[1] + 0.001, 0.25)  # e.g., 4.0 to 8.0 Hz
    phases = np.linspace(0, 2 * np.pi, n_phase_steps, endpoint=False)

    theta_candidates = []
    for f in frequencies:
        for p in phases:
            theta_wave = np.sin(2 * np.pi * f * t + p)
            theta_candidates.append((theta_wave, f, p))
    return theta_candidates


# ----------------------
# FIT TRF FOR EACH PHASE
# ----------------------

def find_best_theta_predictor(eeg_array, theta_candidates, lags, sfreq):
    best_r = -np.inf
    best_theta = None
    best_f = None
    best_p = None

    for theta, f, p in theta_candidates:
        X = theta[np.newaxis, :].T  # Shape: 1 x time
        Y = eeg_array.T  # Shape: n_channels x time
        for l in lambdas:
            model = TRF(direction=1)
            model.train(X, Y, sfreq, lags[0], lags[1], l)
            prediction, r = model.predict(theta[np.newaxis, :], eeg_array)
            if r > best_r:
                best_r = r
                best_theta = theta
                best_f = f
                best_p = p
                best_lambda = l

    return best_theta, best_f, best_p, best_r, best_lambda


# ----------------------
# MAIN LOOP ACROSS SUBJECTS + CONDITIONS
# ----------------------

all_best_theta = []
all_eeg = []

default_path = Path.cwd()
predictors_path = default_path / 'data/eeg/predictors'
eeg_results_path = default_path / 'data/eeg/preprocessed/results'
lambdas = np.logspace(-2, 2, 20)  # based on prev literature
condition = 'a1'

for subject_folder in eeg_results_path.iterdir():
    sub = subject_folder.name
    if 'sub' not in sub:
        continue
    blocks = load_eeg_blocks(condition, subject_folder)
    eeg_concat = mne.concatenate_raws(blocks[sub])
    eeg_data = eeg_concat.get_data()
    block_len = eeg_data.shape[1]
    theta_candidates = generate_theta_candidates(block_len, sfreq, freq_range=(4, 8), n_phase_steps=20)
    best_theta, best_f, best_p, best_r, best_lambda = find_best_theta_predictor(eeg_data, theta_candidates, (lag_start, lag_end), sfreq)
    all_best_theta.append(best_theta)

# ----------------------
# CONCATENATE AND FINAL TRF
# ----------------------

X = np.concatenate([t[np.newaxis, :] for t in all_best_theta], axis=1)  # 1 x total_time
Y = np.concatenate(all_eeg, axis=1)  # n_channels x total_time

final_model = TRF(direction=1)
final_model.train(X, Y, sfreq, lag_start, lag_end, regularization=best_lambda, seed=42)
Y_pred = final_model.predict(X)
final_r = pearsonr(Y_pred, Y)

print(f"Final TRF model r = {final_r:.3f}")
