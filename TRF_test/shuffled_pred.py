import numpy as np
import pandas as pd
import random
import mne
from mne.io import RawArray
from mne.channels import make_standard_montage
from mne.filter import filter_data
import matplotlib.pyplot as plt
from tqdm import tqdm


def run_trf_with_shuffle_and_plot(all_subject_design_matrices, eeg_clean_list_masked, subs, sfreq, best_lambda):
    all_preds_by_sub = []

    for sub in subs:
        print(f"\n[INFO] Running TRF for {sub} with shuffled predictors...")
        X_raw = all_subject_design_matrices1[sub]

        # Shuffle predictor columns
        X_shuffled = X_raw.copy()
        for col in X_shuffled.columns:
            X_shuffled[col] = np.random.permutation(X_shuffled[col].values)

        eeg = eeg_clean_list_masked1[sub]
        min_len = min(eeg.shape[1], len(X_shuffled))
        X_clean = X_shuffled[:min_len]
        eeg = eeg[:, :min_len]

        n_samples = sfreq * 60
        X_folds = []
        Y_folds = []

        for start in range(0, len(X_clean), n_samples):
            end = min(start + n_samples, len(X_clean))
            X_folds.append(X_clean.values[start:end])
            Y_folds.append(eeg[:, start:end].T)

        if Y_folds[-1].shape[0] < int((1.0 - -0.1) * sfreq):
            X_folds[-2] = np.vstack([X_folds[-2], X_folds[-1]])
            Y_folds[-2] = np.vstack([Y_folds[-2], Y_folds[-1]])
            X_folds.pop()
            Y_folds.pop()

        trf = TRF(direction=1)
        trf.train(X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=best_lambda, seed=42)

        predictions = []
        for i in tqdm(range(len(X_folds))):
            X_chunk = X_folds[i]
            Y_chunk = Y_folds[i]
            pred_chunk, _ = trf.predict(X_chunk, Y_chunk)
            predictions.append(pred_chunk[0])

        predicted_full = np.concatenate(predictions, axis=0)  # shape: (time, n_channels)
        all_preds_by_sub.append(predicted_full.T)  # shape: (n_channels, time)

    # Find minimum time length across all predicted EEGs
    min_time_len = min(pred.shape[1] for pred in all_preds_by_sub)

    # Truncate all to the same length
    truncated_preds = [pred[:, :min_time_len] for pred in all_preds_by_sub]

    # Stack and average
    avg_predicted = np.mean(np.stack(truncated_preds), axis=0)  # shape: (n_channels, time)

    # Create MNE Raw object
    ch_names = [f"EEG{i:03}" for i in range(avg_predicted.shape[0])]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = RawArray(avg_predicted, info)

    # Bandpass filter 1–30 Hz
    raw_filtered = raw.copy().filter(l_freq=1.0, h_freq=30.0, fir_design='firwin', verbose=False)

    # Apply standard montage
    if raw_filtered.get_montage() is None:
        raw_filtered.set_montage('standard_1020')

    # Plot PSD
    print("[INFO] Plotting PSD of average predicted EEG (1–30 Hz filtered)...")
    raw_filtered.plot_psd(fmax=30, average=True, spatial_colors=False, show=True)

    return raw_filtered

raw_filtered = run_trf_with_shuffle_and_plot(all_subject_design_matrices1, eeg_clean_list_masked1, subs, sfreq, best_lambda)