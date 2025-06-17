import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from statsmodels.stats.multitest import fdrcorrection
from pathlib import Path
from itertools import combinations

# === Config ===
conditions = ['a1', 'a2', 'e1', 'e2']
stim_type = 'target_nums'
sfreq = 125
time_lags = np.linspace(-0.1, 1.0, 139)
time_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_trimmed = time_lags[time_mask]
predictor_idx = 1

def plane_for_cond(cond):
    return 'azimuth' if cond in ['a1', 'a2'] else 'elevation'

def smooth_weights(w, window_len=11):
    h = np.hamming(window_len)
    h /= h.sum()
    return np.array([np.convolve(w[:, i], h, mode='same') for i in range(w.shape[1])]).T

def load_trfs(base_dir, plane, cond, folder, stream):
    path = base_dir / plane / cond / folder / "on_en_ov_RT" / "weights"
    files = sorted([f for f in path.iterdir() if stream in f.name and 'npy' in f.name])
    trfs = []
    for f in files:
        data = np.load(f, allow_pickle=True).squeeze().T
        smoothed = smooth_weights(data)
        trfs.append(smoothed[:, predictor_idx])
    return np.stack(trfs, axis=0)[:, time_mask]

# === Base directory
default_path = Path.cwd()
base_dir = default_path / 'data/eeg/trf/trf_testing/results/single_sub'

# === Load TRF differences (Target - Distractor)
diff_trfs = {}
for cond in conditions:
    plane = plane_for_cond(cond)
    target = load_trfs(base_dir, plane, cond, stim_type, 'target_stream')
    distractor = load_trfs(base_dir, plane, cond, stim_type, 'distractor_stream')
    diff = target - distractor
    diff_trfs[cond.upper()] = diff  # Uppercase for consistent labels

# === Pairwise comparisons across TRF differences
all_pairs = list(combinations(diff_trfs.keys(), 2))
sig_masks = {}
for c1, c2 in all_pairs:
    data1, data2 = diff_trfs[c1], diff_trfs[c2]
    tvals, pvals = ttest_rel(data1, data2)
    _, pvals_fdr = fdrcorrection(pvals)
    sig_masks[(c1, c2)] = pvals_fdr < 0.05

# === Timepoints significant in all pairwise comparisons
consistent_mask = np.ones_like(time_trimmed, dtype=bool)
for mask in sig_masks.values():
    consistent_mask &= mask

# === Plot TRF Difference with Highlighted Significance
def plot_trf_differences(diffs, time_trimmed, consistent_mask, stim_type='Non Targets'):
    plt.figure(figsize=(10, 5))

    condition_colors = {
        'A1': '#1f77b4',  # Medium blue
        'A2': '#ff7f0e',  # Orange
        'E1': '#2ca02c',  # Green
        'E2': '#d62728'  # Red
    }

    for cond, data in diffs.items():
        mean_diff = np.mean(data, axis=0)
        sem_diff = np.std(data, axis=0) / np.sqrt(data.shape[0])
        plt.plot(time_trimmed, mean_diff, label=cond, color=condition_colors[cond], linewidth=2)
        plt.fill_between(time_trimmed, mean_diff - sem_diff, mean_diff + sem_diff,
                         color=condition_colors[cond], alpha=0.25)

    # --- Highlight cognitive windows
    plt.axvspan(0.0, 0.15, color='gray', alpha=0.08, label='Early Window')
    plt.axvspan(0.15, 0.35, color='gray', alpha=0.05, label='Late Window')

    # --- Highlight consistently significant timepoints
    if consistent_mask.any():
        sig_regions = np.where(consistent_mask)[0]
        for i in range(len(sig_regions) - 1):
            if sig_regions[i + 1] != sig_regions[i] + 1:
                plt.axvspan(time_trimmed[sig_regions[i]], time_trimmed[sig_regions[i]], color='gold', alpha=0.2)
        plt.axvspan(time_trimmed[sig_regions[0]], time_trimmed[sig_regions[-1]], color='gold', alpha=0.2, label='Consistent Difference')

    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.title(f"TRF Difference (Target − Distractor) | {stim_type}", fontsize=12)
    plt.xlabel("Time lag (s)", fontsize=11)
    plt.ylabel("Amplitude Difference (a.u.)", fontsize=11)
    plt.legend(title="Condition", fontsize=9)
    plt.tight_layout()
    plt.grid(False)
    plt.show()

# === Plot
plot_trf_differences(diff_trfs, time_trimmed, consistent_mask, stim_type=stim_type)
