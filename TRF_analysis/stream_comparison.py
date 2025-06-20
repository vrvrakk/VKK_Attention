import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from scipy.integrate import trapezoid
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import permutation_cluster_test
from pathlib import Path

# === Config ===
planes = {
    'azimuth': ['a1'],
    'elevation': ['e1']}

selected_streams = ['target_stream', 'distractor_stream']

folder_types_dict = {
    'target_stream': ['non_targets', 'target_nums'],
    'distractor_stream': ['non_targets', 'target_nums', 'deviants']}

sfreq = 125
time_lags = np.linspace(-0.1, 1.0, 139)
time_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_trimmed = time_lags[time_mask]
predictor_idx = 1  # envelope

# === Components ===
component_peaks = {'P1': 0.07, 'N1': 0.15, 'P2': 0.25}

# === Functions ===
def smooth_weights(w, window_len=11):
    h = np.hamming(window_len)
    h /= h.sum()
    smoothed_w = np.array([np.convolve(wi, h, mode='same') for wi in w])
    return smoothed_w

def load_trfs(base_dir, plane, cond, folder, stream):
    weights_dir = base_dir / plane / cond / folder / "on_en_ov_RT" / "weights"
    file = weights_dir / f"avg_trf_weights_{stream}.npy"
    if not file.exists():
        print(f"Missing file: {file}")
        return None
    data = np.load(file, allow_pickle=True)
    smoothed = smooth_weights(data)
    return smoothed

def bootstrap_sem(data, n_boot=1000):
    """Bootstrap SEM over axis 0 (subjects)."""
    rng = np.random.default_rng(seed=42)
    boot_means = np.array([
        np.mean(rng.choice(data, size=len(data), replace=True), axis=0)
        for _ in range(n_boot)
    ])
    return np.std(boot_means, axis=0)

# === Main ===
default_path = Path.cwd()
base_dir = default_path / 'data/eeg/trf/trf_testing/results/averaged'  # changed from single_sub
from mne.stats import permutation_cluster_test



component_windows = {
    'P1': (0.05, 0.09),
    'N1': (0.11, 0.18),
    'P2': (0.19, 0.29),
    'N2': (0.25, 0.35)
}

for plane, conditions in planes.items():
    for cond in conditions:
        for stream in selected_streams:
            folder_types = folder_types_dict[stream]
            all_trfs = []

            for folder in folder_types:
                folder_path = base_dir / plane / cond / folder / "on_en_ov_RT" / "weights"
                if not folder_path.is_dir():
                    print(f"Missing: {folder_path}")
                    continue

                trfs = load_trfs(base_dir, plane, cond, folder, stream)
                if trfs is not None:
                    all_trfs.append(trfs)  # [n_lags, ]
                else:
                    print(f"No TRFs loaded for {folder}, {stream}")

            if len(all_trfs) < 2:
                print(f"Skipping {plane}-{cond}-{stream}: not enough data.")
                continue

            X = [trf[:, time_mask] for trf in all_trfs]
            # X is a list of (n_conditions,) arrays, each of shape (n_subjects, n_lags)
            T_obs, clusters, p_values, _ = permutation_cluster_test(
                X,
                n_permutations=10000,
                tail=0,
                out_type='mask',
                verbose=False,
                seed=42
            )
            labels = folder_types
            colors = ['royalblue', 'firebrick', 'seagreen'][:len(X)]

            # === Plot ===
            # === Plotting (mean over subjects per condition) ===
            plt.figure(figsize=(9, 5))
            for i, trf in enumerate(X):
                mean_trf = trf.mean(axis=0)  # shape: [n_lags]
                sem_trf = trf.std(axis=0) / np.sqrt(trf.shape[0])  # SEM

                plt.plot(time_trimmed, mean_trf, label=labels[i], color=colors[i])
                plt.fill_between(time_trimmed, mean_trf - sem_trf, mean_trf + sem_trf,
                                 color=colors[i], alpha=0.15)

            # === Component peak labels ===
            t_mean = np.mean([trf.mean(axis=0) for trf in X], axis=0)  # avg over all conditions

            for label, (start, end) in component_windows.items():
                idx_range = (time_trimmed >= start) & (time_trimmed <= end)
                sub_time = time_trimmed[idx_range]
                sub_ampl = t_mean[idx_range]

                if label.startswith('N'):
                    peak_idx = np.argmin(sub_ampl)
                else:
                    peak_idx = np.argmax(sub_ampl)

                peak_time = sub_time[peak_idx]
                peak_amp = sub_ampl[peak_idx]

                plt.text(peak_time,
                         peak_amp + 0.09 * np.sign(peak_amp),
                         label,
                         ha='center',
                         va='bottom' if peak_amp > 0 else 'top',
                         fontsize=10,
                         fontweight='bold',
                         color='black')

            # === Highlight significant clusters ===
            for i_c, cluster_mask in enumerate(clusters):
                if p_values[i_c] < 0.05:
                    t = time_lags[cluster_mask]
                    plt.axvspan(t[0], t[-1], color='red', alpha=0.1)

            plt.axhline(0, color='gray', lw=0.8, linestyle='--')
            plt.xlabel('Time lag (s)')
            plt.ylabel('TRF amplitude (a.u.)')
            plt.title(f'TRF Response Comparison across Stimulus Types (Concatenated)\n{plane.capitalize()}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            plt.show()