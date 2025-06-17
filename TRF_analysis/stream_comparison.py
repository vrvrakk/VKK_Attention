# Re-import necessary packages after code state reset
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
    'azimuth': ['a1', 'a2'],
    'elevation': ['e1', 'e2']
}
selected_streams = ['target_stream', 'distractor_stream']
folder_types_dict = {
    'target_stream': ['non_targets', 'target_nums'],
    'distractor_stream': ['non_targets', 'target_nums', 'deviants']
}
sfreq = 125
time_lags = np.linspace(-0.1, 1.0, 139)
time_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_trimmed = time_lags[time_mask]
predictor_idx = 1  # envelope

def smooth_weights(w, window_len=11):
    h = np.hamming(window_len)
    h /= h.sum()
    return np.array([np.convolve(w[:, i], h, mode='same') for i in range(w.shape[1])]).T

def load_trfs(base_dir, plane, cond, folder, stream):
    weights_dir = base_dir/plane/cond/folder/"on_en_ov_RT"/"weights"
    files = sorted([f for f in weights_dir.iterdir() if stream in f.name and 'npy' in f.name])
    trfs = []
    for f in files:
        data = np.load(os.path.join(weights_dir, f), allow_pickle=True).squeeze().T
        smoothed = smooth_weights(data)
        trfs.append(smoothed[:, predictor_idx])
    return np.stack(trfs, axis=0)[:, time_mask]

def extract_metrics(data, start, end):
    mask = (time_trimmed >= start) & (time_trimmed <= end)
    return {
        'Mean': np.mean(data[:, mask], axis=1),
        'RMS': np.sqrt(np.mean(data[:, mask]**2, axis=1)),
        'AUC': trapezoid(data[:, mask], dx=1/sfreq, axis=1)
    }

default_path = Path.cwd()
base_dir = default_path /'data/eeg/trf/trf_testing/results/single_sub'


# Plot and analyze
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
                all_trfs.append(trfs)

            if len(all_trfs) < 2:
                print(f"Skipping {plane}-{cond}-{stream}: not enough data.")
                continue

            # Cluster-based permutation test
            X = all_trfs
            T_obs, clusters, cluster_p_vals, _ = permutation_cluster_test(
                X, n_permutations=10000, tail=0, out_type='mask', verbose=False
            )

            # FDR-corrected timepoints (pairwise t-test across time)
            p_vals = np.array([
                ttest_rel(all_trfs[0][:, i], all_trfs[1][:, i]).pvalue
                for i in range(all_trfs[0].shape[1])
            ])
            _, p_fdr = fdrcorrection(p_vals)
            sig_mask = p_fdr < 0.05

            # === Plotting ===
            plt.figure(figsize=(10, 5))
            colors = ['royalblue', 'firebrick', 'seagreen']
            for i, trfs in enumerate(all_trfs):
                mean = trfs.mean(axis=0)
                sem = trfs.std(axis=0) / np.sqrt(trfs.shape[0])
                plt.plot(time_trimmed, mean, label=folder_types[i], color=colors[i])
                plt.fill_between(time_trimmed, mean - sem, mean + sem, alpha=0.3, color=colors[i])

            # Highlight FDR-significant timepoints
            for i in range(len(sig_mask)):
                if sig_mask[i]:
                    plt.axvspan(time_trimmed[i], time_trimmed[i] + 0.004, color='gray', alpha=0.3)

            # Highlight clusters from permutation test
            for i, cluster_mask in enumerate(clusters):
                if cluster_p_vals[i] < 0.05:
                    t = time_trimmed[cluster_mask]
                    plt.axvspan(t[0], t[-1], color='red', alpha=0.1)

            plt.title(f'{stream.replace("_", " ").title()} | {plane.upper()}-{cond.upper()} | TRF Envelope')
            plt.xlabel('Time lag (s)')
            plt.ylabel('Amplitude (a.u.)')
            plt.legend()
            plt.tight_layout()
            plt.show()
