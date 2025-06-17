import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from scipy.integrate import trapezoid
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection
from itertools import combinations
from pathlib import Path

# === Config ===
conditions = ['a1', 'a2', 'e1', 'e2']
planes = {'a1': 'azimuth', 'a2': 'azimuth', 'e1': 'elevation', 'e2': 'elevation'}
selected_streams = ['target_stream', 'distractor_stream']
folder_types_dict = {
    'target_stream': ['all_stims', 'non_targets', 'target_nums'],
    'distractor_stream': ['all_stims', 'non_targets', 'target_nums', 'deviants']
}
sfreq = 125
time_lags = np.linspace(-0.1, 1.0, 139)
time_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_trimmed = time_lags[time_mask]
predictor_idx = 1  # envelope

colors = {
    'a1': 'royalblue',
    'a2': 'firebrick',
    'e1': 'seagreen',
    'e2': 'goldenrod'
}
labels = {
    'a1': 'Azimuth A1',
    'a2': 'Azimuth A2',
    'e1': 'Elevation E1',
    'e2': 'Elevation E2'
}

# === Paths ===
default_path = Path.cwd()
base_dir = default_path / 'data/eeg/trf/trf_testing/results/single_sub'

# === Functions ===
def smooth_weights(w, window_len=11):
    h = np.hamming(window_len)
    h /= h.sum()
    return np.array([np.convolve(w[:, i], h, mode='same') for i in range(w.shape[1])]).T

def load_trfs(base_dir, plane, cond, folder, stream):
    weights_dir = base_dir / plane / cond / folder / "on_en_ov_RT" / "weights"
    if not weights_dir.exists():
        return None
    files = sorted([f for f in weights_dir.iterdir() if stream in f.name and 'npy' in f.name])
    trfs = []
    for f in files:
        data = np.load(f, allow_pickle=True).squeeze().T
        smoothed = smooth_weights(data)
        trfs.append(smoothed[:, predictor_idx])
    return np.stack(trfs, axis=0)[:, time_mask] if trfs else None

time_windows = {
    'full': (0.0, 0.5),
    'early': (0.0, 0.2),
    'late': (0.2, 0.5)
}

def get_window_mask(start, end):
    return (time_trimmed >= start) & (time_trimmed <= end)

def compute_window_stats(trfs_dict):
    stats = {}
    for (cond1, trf1), (cond2, trf2) in combinations(trfs_dict.items(), 2):
        for win_name, (start, end) in time_windows.items():
            mask = get_window_mask(start, end)
            t_stat, p_val = ttest_rel(trf1[:, mask].mean(axis=1), trf2[:, mask].mean(axis=1))
            stats[(cond1, cond2, win_name)] = (t_stat, p_val)
    return stats

def plot_with_significance(folder_type, stream, trfs_dict):
    plt.figure(figsize=(10, 5))
    time_len = len(time_trimmed)

    # Plot means + SEM
    for cond, trfs in trfs_dict.items():
        mean = trfs.mean(axis=0)
        sem = trfs.std(axis=0) / np.sqrt(trfs.shape[0])
        plt.plot(time_trimmed, mean, label=labels[cond], color=colors[cond])
        plt.fill_between(time_trimmed, mean - sem, mean + sem, alpha=0.3, color=colors[cond])

    # Pairwise stats across time
    for (cond1, cond2) in combinations(trfs_dict.keys(), 2):
        trf1, trf2 = trfs_dict[cond1], trfs_dict[cond2]
        p_vals = np.array([
            ttest_rel(trf1[:, i], trf2[:, i]).pvalue
            for i in range(time_len)
        ])
        _, p_fdr = fdrcorrection(p_vals)
        sig_mask = p_fdr < 0.05

        for i in range(time_len - 1):
            if sig_mask[i]:
                plt.axvspan(time_trimmed[i], time_trimmed[i+1], color='gray', alpha=0.2)

        # Window stats
        win_stats = compute_window_stats({cond1: trf1, cond2: trf2})
        for win_name in time_windows:
            t_val, p_val = win_stats[(cond1, cond2, win_name)]
            print(f"{folder_type} | {stream} | {cond1} vs {cond2} | {win_name}: t={t_val:.2f}, p={p_val:.4f}")

    plt.title(f'{stream.replace("_", " ").title()} | {folder_type.replace("_", " ").title()} | All Conditions')
    plt.xlabel('Time lag (s)')
    plt.ylabel('Amplitude (a.u.)')
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Main Loop ===
for stream in selected_streams:
    folder_types = folder_types_dict[stream]
    for folder_type in folder_types:
        if folder_type == 'deviants' and stream != 'distractor_stream':
            continue

        trfs_dict = {}
        for cond in conditions:
            plane = planes[cond]
            trfs = load_trfs(base_dir, plane, cond, folder_type, stream)
            if trfs is not None:
                trfs_dict[cond] = trfs

        if len(trfs_dict) >= 2:
            plot_with_significance(folder_type, stream, trfs_dict)

