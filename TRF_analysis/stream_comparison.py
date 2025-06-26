import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
from scipy.integrate import trapezoid
from scipy.stats import ttest_rel, ttest_ind
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import permutation_cluster_test
from pathlib import Path
from itertools import combinations

# === Config ===
planes = {'azimuth': ['a1'], 'elevation': ['e1']}
selected_streams = ['target_stream', 'distractor_stream']
folder_types_dict = {
    'target_stream': ['non_targets', 'target_nums'],
    'distractor_stream': ['non_targets', 'target_nums', 'deviants']}
sfreq = 125
time_lags = np.linspace(-0.1, 1.0, 139)
time_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_trimmed = time_lags[time_mask]
pred_idx = 1
pred = 'onsets' if pred_idx == 0 else 'envelopes'

# === Functions ===
def smooth_weights(w, window_len=11):
    h = np.hamming(window_len)
    h /= h.sum()
    smoothed_w = np.array([np.convolve(wi, h, mode='same') for wi in w])
    return smoothed_w

def load_trfs(base_dir, plane, cond, folder, stream, pred_idx):
    weights_dir = base_dir / plane / cond / folder / "on_en_ov_RT" / "weights"
    file = weights_dir / f"avg_trf_weights_{stream}.npy"
    if not file.exists():
        print(f"Missing file: {file}")
        return None
    data = np.load(file, allow_pickle=True)
    data = data[:, pred_idx, :]
    smoothed = smooth_weights(data)
    return smoothed

# === Main ===
default_path = Path.cwd()
base_dir = default_path / 'data/eeg/trf/trf_testing/results/averaged'

component_windows = {
    'P1': (0.05, 0.09),
    'N1': (0.11, 0.18),
    'P2': (0.19, 0.29),
    'N2': (0.25, 0.35),
    'P3': (0.35, 0.45)
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

                trfs = load_trfs(base_dir, plane, cond, folder, stream, pred_idx)
                if trfs is not None:
                    all_trfs.append(trfs)  # shape: (n_subjects, n_lags)
                else:
                    print(f"No TRFs loaded for {folder}, {stream}")

            if len(all_trfs) < 2:
                print(f"Skipping {plane}-{cond}-{stream}: not enough data.")
                continue

            # === FDR-corrected t-tests across time points ===
            for (i, j) in combinations(range(len(all_trfs)), 2):
                label_i, label_j = folder_types[i], folder_types[j]
                data_i, data_j = all_trfs[i][:, time_mask], all_trfs[j][:, time_mask]

                p_vals = np.array([ttest_rel(data_i[:, t], data_j[:, t]).pvalue for t in range(data_i.shape[1])])
                _, p_fdr = fdrcorrection(p_vals)
                sig_mask = p_fdr < 0.05
                print(f"FDR t-test: {label_i} vs {label_j} → {np.sum(sig_mask)} significant points")

            # === Permutation cluster test ===
            X = [trf[:, time_mask] for trf in all_trfs]
            T_obs, clusters, p_values, _ = permutation_cluster_test(
                X, n_permutations=10000, tail=0,
                out_type='mask', verbose=False, seed=42
            )

            labels = folder_types
            colors = ['royalblue', 'firebrick', 'seagreen'][:len(X)]

            # === Plot ===
            plt.figure(figsize=(9, 5))
            for i, trf in enumerate(X):
                mean_trf = trf.mean(axis=0)
                sem_trf = trf.std(axis=0) / np.sqrt(trf.shape[0])
                plt.plot(time_trimmed, mean_trf, label=labels[i], color=colors[i])
                plt.fill_between(time_trimmed, mean_trf - sem_trf, mean_trf + sem_trf,
                                 color=colors[i], alpha=0.15)

            # === Component peak labels ===
            t_mean = np.mean([trf.mean(axis=0) for trf in X], axis=0)
            for label, (start, end) in component_windows.items():
                idx_range = (time_trimmed >= start) & (time_trimmed <= end)
                sub_time = time_trimmed[idx_range]
                sub_ampl = t_mean[idx_range]
                peak_idx = np.argmin(sub_ampl) if label.startswith('N') else np.argmax(sub_ampl)
                peak_time = sub_time[peak_idx]
                peak_amp = sub_ampl[peak_idx]
                plt.text(peak_time, peak_amp + 0.09 * np.sign(peak_amp), label,
                         ha='center', va='bottom' if peak_amp > 0 else 'top',
                         fontsize=10, fontweight='bold', color='black')

            # === Highlight significant clusters ===
            for i_c, cluster_mask in enumerate(clusters):
                if p_values[i_c] < 0.05:
                    t = time_lags[cluster_mask]
                    plt.axvspan(t[0], t[-1], color='red', alpha=0.1)

            plt.axhline(0, color='gray', lw=0.8, linestyle='--')
            plt.xlabel('Time lag (s)')
            plt.ylabel('TRF amplitude (a.u.)')
            plt.title(f'TRF Comparison across Stimulus Types | {stream.capitalize()} | {plane}')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()

            save_dir = base_dir / 'figures'
            save_dir.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_dir / f'{plane}-{cond}-{stream}_all_responses_{pred}.png', dpi=300)
            plt.show()


    from scipy.stats import f_oneway
    import numpy as np

    # === Define analysis windows ===
    rms_windows = {
        'Early (0.10–0.20s)': (0.10, 0.20),
        'Late (0.20–0.40s)': (0.20, 0.40)
    }

    for win_label, (t_start, t_end) in rms_windows.items():
        idx_range = (time_trimmed >= t_start) & (time_trimmed <= t_end)
        if not np.any(idx_range):
            print(f"\n Skipping {win_label} – no data in range.")
            continue

        print(f"\n=== RMS ANOVA ({win_label}) for {stream} | {plane} | {cond} ===")

        # === Get RMS per subject per condition in the window ===
        rms_vals = []
        for trf in X:  # shape: (n_subjects, n_lags)
            rms = np.sqrt(np.mean(trf[:, idx_range] ** 2, axis=1))  # per subject
            rms_vals.append(rms)

        # === One-way ANOVA ===
        f_val, p_val = f_oneway(*rms_vals)

        # === Effect size (η²) ===
        all_vals = np.concatenate(rms_vals)
        group_means = [np.mean(r) for r in rms_vals]
        grand_mean = np.mean(all_vals)

        ss_between = sum(len(r) * (m - grand_mean) ** 2 for r, m in zip(rms_vals, group_means))
        ss_total = sum((x - grand_mean) ** 2 for x in all_vals)
        eta_squared = ss_between / ss_total if ss_total > 0 else np.nan

        print(f"RMS ANOVA: F = {f_val:.3f}, p = {p_val:.4f}, η² = {eta_squared:.3f}")

        # === Optional: pairwise post-hoc t-tests ===
        if len(rms_vals) == 3:
            from itertools import combinations
            from scipy.stats import ttest_ind

            print("Post-hoc t-tests (uncorrected):")
            for (i, j) in combinations(range(3), 2):
                t_stat, p_pair = ttest_ind(rms_vals[i], rms_vals[j])
                print(f"  {labels[i]} vs {labels[j]}: t = {t_stat:.2f}, p = {p_pair:.4f}")
        # === Save report to file ===
        report_dir = base_dir / "data"
        report_dir.mkdir(parents=True, exist_ok=True)

        report_file = report_dir / f"rms_{stream}_{plane}_{cond}_{pred}.txt"

        with open(report_file, "a", encoding="utf-8") as f:
            f.write(f"\n=== RMS ANOVA ({win_label}) for {stream} | {plane} | {cond} ===\n")
            f.write(f"RMS ANOVA: F = {f_val:.3f}, p = {p_val:.4f}, η² = {eta_squared:.3f}\n")

            if len(rms_vals) == 3:
                f.write("Post-hoc t-tests (uncorrected):\n")
                for (i, j) in combinations(range(3), 2):
                    t_stat, p_pair = ttest_ind(rms_vals[i], rms_vals[j])
                    f.write(f"  {labels[i]} vs {labels[j]}: t = {t_stat:.2f}, p = {p_pair:.4f}\n")