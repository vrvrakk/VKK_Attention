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
conditions = ['a1', 'e1']
planes = {'a1': 'azimuth', 'e1': 'elevation'}
selected_streams = ['target_stream', 'distractor_stream']
folder_types_dict = {
    'target_stream': ['non_targets', 'target_nums'],
    'distractor_stream': ['non_targets', 'target_nums', 'deviants']}
sfreq = 125
time_lags = np.linspace(-0.1, 1.0, 139)
time_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_trimmed = time_lags[time_mask]
predictor_idx = 1  # envelope

colors = {
    'a1': 'royalblue',
    'e1': 'seagreen'}

labels = {
    'a1': 'Azimuth',
    'e1': 'Elevation'}

# === Paths ===
default_path = Path.cwd()
base_dir = default_path / 'data/eeg/trf/trf_testing/results/averaged'  # changed from single_sub

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

time_windows = {
    'early': (0.1, 0.2),
    'late': (0.2, 0.4)}

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


from mne.stats import permutation_cluster_test

component_windows = {
    'P1': (0.05, 0.09),
    'N1': (0.11, 0.18),
    'P2': (0.19, 0.29),
    'N2': (0.25, 0.35),
    'P3': (0.35, 0.45)
}

def plot_with_significance(folder_type, stream, trfs_dict):
    plt.figure(figsize=(10, 5))
    time_len = len(time_trimmed)

    # Assume you have exactly 2 conditions
    conditions = list(trfs_dict.keys())
    assert len(conditions) == 2, "Exactly two conditions required for pairwise cluster test."

    cond1, cond2 = conditions
    data1 = trfs_dict[cond1][:, time_mask]  # shape: (n_subjects, n_lags)
    data2 = trfs_dict[cond2][:, time_mask]

    # === Cluster-based permutation test ===
    X = [data1, data2]  # List of two arrays
    T_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
        X,
        n_permutations=100000,
        tail=1,
        out_type='mask',
        seed=42)
    # === Plot mean ± SEM for both conditions ===
    for cond, trfs in trfs_dict.items():
        mean = trfs.mean(axis=0)[time_mask]
        sem = trfs.std(axis=0) / np.sqrt(trfs.shape[0])
        sem = sem[time_mask]
        plt.plot(time_trimmed, mean, label=labels[cond], color=colors[cond])
        plt.fill_between(time_trimmed, mean - sem, mean + sem, alpha=0.3, color=colors[cond])

    # === Highlight significant clusters ===
    for i_c, cluster_mask in enumerate(clusters):
        if cluster_p_values[i_c] < 0.05:
            t_sig = time_trimmed[cluster_mask]
            plt.axvspan(t_sig[0], t_sig[-1], color='red', alpha=0.2)

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

    plt.axhline(0, color='gray', lw=0.8, linestyle='--')
    plt.xlabel('Time lag (s)')
    plt.ylabel('TRF amplitude (a.u.)')
    plt.title(f'{folder_type.capitalize().replace('_', ' ')} – {stream.capitalize().replace('_', ' ')} | Cluster Permutation Test')
    plt.legend()
    plt.tight_layout()
    plt.grid(alpha=0.3)
    save_dir = base_dir / 'figures' / 'plane_comparison'
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f'planes_comparison_{stream}_{folder_type}.png'
    plt.show()
    plt.savefig(save_dir / filename)


# === Main Loop ===

for stream in selected_streams:
    folder_types = folder_types_dict[stream]
    for folder_type in folder_types:
        if folder_type == 'deviants' and stream == 'target_stream':
            continue

        trfs_dict = {}
        for cond in conditions:
            plane = planes[cond]
            trfs = load_trfs(base_dir, plane, cond, folder_type, stream)
            if trfs is not None:
                trfs_dict[cond] = trfs

        if len(trfs_dict) >= 2:
            plot_with_significance(folder_type, stream, trfs_dict)
            from scipy.stats import ttest_rel
            from pathlib import Path

            # === RMS comparison for early vs late ===
            rms_windows = {
                'Early (0.10–0.20s)': (0.10, 0.20),
                'Late (0.20–0.40s)': (0.20, 0.40)
            }

            for win_label, (t_start, t_end) in rms_windows.items():
                idx_range = (time_trimmed >= t_start) & (time_trimmed <= t_end)
                if not np.any(idx_range):
                    print(f"\nSkipping {win_label} – no data in range.")
                    continue

                print(f"\n=== RMS Paired t-test ({win_label}) for {stream} | {folder_type} ===")
                rms_vals = []
                cond_names = list(trfs_dict.keys())

                for cond in cond_names:
                    trfs = trfs_dict[cond][:, time_mask]  # shape: (n_subjects, n_lags)
                    rms = np.sqrt(np.mean(trfs[:, idx_range] ** 2, axis=1))  # shape: (n_subjects,)
                    rms_vals.append(rms)

                # Paired t-test between two conditions
                if len(rms_vals) == 2:
                    t_val, p_val = ttest_rel(rms_vals[0], rms_vals[1])

                    # Effect size: Cohen's d for paired samples
                    diff = rms_vals[0] - rms_vals[1]
                    cohen_d = diff.mean() / diff.std(ddof=1)

                    print(f"t = {t_val:.3f}, p = {p_val:.4f}, d = {cohen_d:.3f}")

                    # Save to file
                    report_dir = base_dir / 'plane_comparison' / "data"
                    report_dir.mkdir(parents=True, exist_ok=True)
                    report_file = report_dir / f"plane_comparison_rms_{stream}_{folder_type}.txt"

                    with open(report_file, "a", encoding="utf-8") as f:
                        f.write(f"\n=== RMS Paired t-test ({win_label}) for {stream} | {folder_type} ===\n")
                        f.write(f"{cond_names[0]} vs {cond_names[1]}:\n")
                        f.write(f"t = {t_val:.3f}, p = {p_val:.4f}, d = {cohen_d:.3f}\n")

