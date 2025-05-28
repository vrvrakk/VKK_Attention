import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
from mtrf import TRF


default_path = Path.cwd()
plane = 'azimuth'
stims='all'

stream1='target'
stream2='distractor'

data_path1 = default_path / f'data/eeg/trf/trf_testing/composite_model/{plane}/{stims}_stims/data/{plane}_{stims}_{stream1}_TRF_results.npz'
data_path2 = default_path / f'data/eeg/trf/trf_testing/composite_model/{plane}/{stims}_stims/data/{plane}_{stims}_{stream2}_TRF_results.npz'

# Example: Load TRF results for azimuth plane
target_data = np.load(data_path1, allow_pickle=True)
distractor_data = np.load(data_path2, allow_pickle=True)
list(target_data.keys())

# --- Extract ---
preds = target_data['preds'].tolist()
time_lags = target_data['time_lags']  # shape: (n_lags,)
tmin_plot = 0.0
tmax_plot = 0.6
lag_mask = (time_lags >= tmin_plot) & (time_lags <= tmax_plot)
time_lags_trimmed = time_lags[lag_mask]

# --- Envelope TRFs ---
pred = 'RT_labels'
pred_idx = preds.index(f'{pred}_target_stream')
target_weights = target_data['weights'][pred_idx].T[:, lag_mask]  # (channels, lags)
distractor_weights = distractor_data['weights'][pred_idx].T[:, lag_mask]

# --- Smooth ---
def smooth_channels(weights, window_len=11):
    hamming_win = np.hamming(window_len)
    hamming_win /= hamming_win.sum()
    return np.array([
        np.convolve(weights[ch], hamming_win, mode='same')
        for ch in range(weights.shape[0])
    ])

target_smoothed = smooth_channels(target_weights)
distractor_smoothed = smooth_channels(distractor_weights)

# --- Average across channels ---
target_avg = target_smoothed.mean(axis=0)
distractor_avg = distractor_smoothed.mean(axis=0)
diff_wave = target_avg - distractor_avg

# --- Plot ---
plt.figure(figsize=(10, 6))
plt.plot(time_lags_trimmed, target_avg, label='Target', linewidth=2)
plt.plot(time_lags_trimmed, distractor_avg, label='Distractor', linewidth=2)
plt.plot(time_lags_trimmed, diff_wave, label='Target - Distractor', linestyle='--', color='black')
plt.axhline(0, color='gray', linestyle=':')
plt.xlabel('Time Lag (s)')
plt.ylabel('TRF Amplitude')
plt.title(f'TRF Comparison for {preds[pred_idx]} Predictor ({plane.capitalize()})')
plt.legend()
plt.tight_layout()
plt.show()