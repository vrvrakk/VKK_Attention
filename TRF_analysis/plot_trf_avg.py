import os
import numpy as np
from scipy.stats import shapiro, wilcoxon
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, sem
from statsmodels.stats.multitest import fdrcorrection
from mne.stats import permutation_cluster_test
import matplotlib
matplotlib.use('TkAgg')
plt.ion()

# === CONFIG ===
plane = 'azimuth'
cond = 'a1' if plane == 'azimuth' else 'e1'
target_label = 'Right' if cond == 'a1' else 'Bottom'
folder_type = 'all_stims'

sfreq = 125
time_lags = np.linspace(-0.1, 1.0, 139)
plot_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_plot = time_lags[plot_mask]
weights_dir = fr"C:/Users/pppar/PycharmProjects/VKK_Attention/data/eeg/trf/trf_testing/results/single_sub/{plane}/{cond}/{folder_type}/on_en_ov_RT/weights"
window_len = 11
fig_dir = Path.cwd()/ f'data/eeg/trf/trf_testing/composite_model/single_sub/figures/{plane}/{cond}/{folder_type}'

# === FUNCTIONS ===

def smooth_weights(weights, win_len):
    """Smooth predictors over time using Hamming window."""
    win = np.hamming(win_len)
    win /= win.sum()
    return np.stack([np.convolve(weights[:, i], win, mode='same') for i in range(weights.shape[1])], axis=1)

def load_smoothed_weights(stream_type):
    """Load and smooth predictor weights for one stream."""
    exclude = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub08']
    files = [f for f in sorted(os.listdir(weights_dir)) if stream_type in f and 'weights' in f and not any(sub in f for sub in exclude)]
    all_weights = []
    for f in files:
        w = np.load(os.path.join(weights_dir, f), allow_pickle=True).squeeze().T
        smoothed = smooth_weights(w, window_len)
        all_weights.append(smoothed)
    return np.stack(all_weights, axis=0)[:, :, 1]  # Keep only envelope predictor (index 1)

def compute_cohens_d(x, y):
    diff = np.array(x) - np.array(y)
    return np.mean(diff) / np.std(diff, ddof=1)


def plot_with_significance_and_components(time, target, distractor, sig_mask, clusters, cluster_p_vals, d_val, highlight_components=True):

    plt.figure(figsize=(8, 4))

    # Mean & SEM
    t_mean, t_sem = target.mean(axis=0), sem(target, axis=0)
    d_mean, d_sem = distractor.mean(axis=0), sem(distractor, axis=0)

    # Plot TRFs
    plt.plot(time, t_mean, label='Target', color='#4C72B0', lw=2)
    plt.fill_between(time, t_mean - t_sem, t_mean + t_sem, color='#4C72B0', alpha=0.3)

    plt.plot(time, d_mean, label='Distractor', color='#DD5C5C', lw=2)
    plt.fill_between(time, d_mean - d_sem, d_mean + d_sem, color='#DD5C5C', alpha=0.3)

    # === Optional component highlights ===
    # === Floating component labels near true peaks ===
    if highlight_components:
        component_windows = {
            'P1': (0.05, 0.08),
            'N1': (0.08, 0.13),
            'P2': (0.15, 0.25),
            'N2': (0.25, 0.35)
        }

        for label, (start, end) in component_windows.items():
            idx_range = (time >= start) & (time <= end)
            sub_time = time[idx_range]
            sub_ampl = t_mean[idx_range]

            if label.startswith('N'):  # negative peak
                peak_idx = np.argmin(sub_ampl)
            else:  # positive peak
                peak_idx = np.argmax(sub_ampl)

            peak_time = sub_time[peak_idx]
            peak_amp = sub_ampl[peak_idx]

            plt.text(peak_time, peak_amp + 0.09 * np.sign(peak_amp),
                     label, ha='center', va='bottom' if peak_amp > 0 else 'top',
                     fontsize=9, color='black', fontweight='bold')
    # FDR significant spans
    in_sig = False
    for i, sig in enumerate(sig_mask):
        if sig and not in_sig:
            start = time[i]
            in_sig = True
        elif not sig and in_sig:
            end = time[i]
            in_sig = False
            plt.axvspan(start, end, color='gray', alpha=0.15, label='FDR p < 0.05' if i == 0 else None)
            plt.text((start + end)/2, max(t_mean[i], d_mean[i]) + 0.05, '*', ha='center', fontsize=11)

    # Cluster significance
    cluster_labeled = False
    for i, p in enumerate(cluster_p_vals):
        if p < 0.05:
            cluster_time = time[clusters[i]]
            start, end = cluster_time[0], cluster_time[-1]
            plt.axvspan(start, end, color='red', alpha=0.12,
                        label='Permutation p < 0.05' if not cluster_labeled else 'Permutation p > 0.05')
            cluster_labeled = True

    # Final formatting
    plt.axhline(0, color='k', linestyle='--', lw=0.8)
    plt.xlabel('Time lag (s)')
    plt.ylabel('TRF amplitude (a.u.)')
    plt.title(f'TRFs by Stream | Cohen\'s d = {d_val:.2f}')

    # Deduplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), fontsize=9)

    plt.tight_layout()
    save_dir = fig_dir / f'{plane}_trf_comparison_{folder_type}.png'
    plt.savefig(save_dir, dpi=300, bbox_inches='tight')
    plt.show()

def plot_trf_magnitude(time, target, distractor):
    target_gfp = np.mean(np.abs(target), axis=0)
    distractor_gfp = np.mean(np.abs(distractor), axis=0)
    target_sem = sem(np.abs(target), axis=0)
    distractor_sem = sem(np.abs(distractor), axis=0)

    plt.figure(figsize=(8, 5))
    plt.plot(time, target_gfp, color='#0072B2', lw=2.2, label='Target |TRF|')
    plt.fill_between(time, target_gfp - target_sem, target_gfp + target_sem, alpha=0.25, color='#0072B2')
    plt.plot(time, distractor_gfp, color='#D55E00', lw=2.2, label='Distractor |TRF|')
    plt.fill_between(time, distractor_gfp - distractor_sem, distractor_gfp + distractor_sem, alpha=0.25, color='#D55E00')

    plt.axhline(0, color='gray', linestyle='--', lw=0.8)
    plt.xlabel('Time lag (s)', fontsize=13)
    plt.ylabel(r'Mean $|$TRF amplitude$|$ (a.u.)', fontsize=13)
    plt.title('TRF Magnitude Over Time', fontsize=14)

    plt.grid(axis='y', alpha=0.2)
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.legend(fontsize=10)
    plt.tight_layout()
    save_dir = fig_dir / f'{plane}_trf_GFP_{folder_type}.png'
    plt.savefig(save_dir, dpi=300, bbox_inches='tight')
    plt.show()


def extract_cluster_metrics(cluster_windows, target, distractor, time_axis):
    for i, (start, end) in enumerate(cluster_windows):
        mask = (time_axis >= start) & (time_axis <= end)
        t_rms = np.sqrt(np.mean(target[:, mask] ** 2, axis=1))
        d_rms = np.sqrt(np.mean(distractor[:, mask] ** 2, axis=1))
        print(f"\nCluster {i + 1}: {start:.3f}–{end:.3f}s")
        print(f"Target RMS: {t_rms.mean():.4f}, Distractor RMS: {d_rms.mean():.4f}")
        print(f"Cohen's d: {compute_cohens_d(t_rms, d_rms):.2f}")

def plot_individual_trfs_with_sd(trfs, time_lags, title='', color='royalblue'):
    """
    Plot each subject's TRF along with mean ± SD.

    Parameters
    ----------
    trfs : np.ndarray
        TRFs of shape (n_subjects, n_times)
    time_lags : np.ndarray
        Time axis (same length as TRF time dimension)
    title : str
        Title for the plot
    color : str
        Base color for lines and shading
    """
    import matplotlib.pyplot as plt
    from numpy import mean, std

    mean_trf = trfs.mean(axis=0)
    std_trf = trfs.std(axis=0)

    plt.figure(figsize=(8, 5))

    # Individual TRFs
    for trf in trfs:
        plt.plot(time_lags, trf, alpha=0.3, lw=1, color=color)

    # Mean ± SD
    plt.plot(time_lags, mean_trf, color=color, lw=2.5, label='Mean')
    plt.fill_between(time_lags, mean_trf - std_trf, mean_trf + std_trf,
                     color=color, alpha=0.25, label='±1 SD')

    plt.axhline(0, color='gray', linestyle='--', lw=0.8)
    plt.xlabel('Time lag (s)')
    plt.ylabel('TRF amplitude (a.u.)')
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.show()


# === New: Extract RMS and PTP in early/late windows and compare ===

def extract_trf_metrics_by_window(target, distractor, time_axis, save_path=None, label=''):
    """
    Extract RMS and PTP for early (0–0.2s) and late (0.2–0.4s) windows.
    Returns subject-level metrics + group-level stats.
    """
    windows = {'early': (0.0, 0.2), 'late': (0.2, 0.4)}
    metric_dict = {}
    stats_summary = []

    for win_label, (start, end) in windows.items():
        mask = (time_axis >= start) & (time_axis <= end)

        # RMS
        target_rms = np.sqrt(np.mean(target[:, mask] ** 2, axis=1))
        distractor_rms = np.sqrt(np.mean(distractor[:, mask] ** 2, axis=1))

        # PTP
        target_ptp = np.ptp(target[:, mask], axis=1)
        distractor_ptp = np.ptp(distractor[:, mask], axis=1)

        # Store for CSV
        metric_dict[f'target_rms_{win_label}'] = target_rms
        metric_dict[f'distractor_rms_{win_label}'] = distractor_rms
        metric_dict[f'target_ptp_{win_label}'] = target_ptp
        metric_dict[f'distractor_ptp_{win_label}'] = distractor_ptp

        # === Compare streams
        for metric_name, tvals, dvals in zip(
            [f'RMS {win_label}', f'PTP {win_label}'],
            [target_rms, target_ptp],
            [distractor_rms, distractor_ptp]
        ):
            # Normality
            t_norm = shapiro(tvals).pvalue > 0.05
            d_norm = shapiro(dvals).pvalue > 0.05
            if t_norm and d_norm:
                stat_type = 't-test'
                p_val = ttest_rel(tvals, dvals).pvalue
            else:
                stat_type = 'Wilcoxon'
                try:
                    p_val = wilcoxon(tvals, dvals).pvalue
                except Exception:
                    p_val = np.nan

            d = compute_cohens_d(tvals, dvals)
            stats_summary.append({
                'Metric': metric_name,
                'Test': stat_type,
                'p-value': p_val,
                "Cohen's d": d
            })

    # === Save subject-level metrics
    df_subjects = pd.DataFrame(metric_dict)
    if save_path:
        df_subjects.to_csv(os.path.join(save_path, f'subject_metrics_{label}.csv'), index=False)

    # === Save group-level stats
    df_stats = pd.DataFrame(stats_summary)
    if save_path:
        df_stats.to_csv(os.path.join(save_path, f'metric_stats_{label}.csv'), index=False)

    return df_subjects, df_stats


# === MAIN ===

# Load TRFs
target_trfs = load_smoothed_weights("target_stream")[:, plot_mask]
distractor_trfs = load_smoothed_weights("distractor_stream")[:, plot_mask]

# Pointwise t-tests with FDR
p_vals = np.array([ttest_rel(target_trfs[:, i], distractor_trfs[:, i]).pvalue for i in range(target_trfs.shape[1])])
_, p_fdr = fdrcorrection(p_vals)
sig_mask = p_fdr < 0.05

# Permutation cluster test
X = [target_trfs, distractor_trfs]
T_obs, clusters, cluster_p_vals, _ = permutation_cluster_test(
    X, n_permutations=100000, tail=0, threshold=2.1, out_type='mask', verbose=True, seed=42
)

# Compute effect size
d_val = compute_cohens_d(
    target_trfs[:, sig_mask].mean(axis=1) if sig_mask.any() else target_trfs.mean(axis=1),
    distractor_trfs[:, sig_mask].mean(axis=1) if sig_mask.any() else distractor_trfs.mean(axis=1)
)

# Plot
plot_with_significance_and_components(time_plot, target_trfs, distractor_trfs, sig_mask, clusters, cluster_p_vals, d_val, highlight_components=True)
plot_trf_magnitude(time_plot, target_trfs, distractor_trfs)

# Extract cluster-defined TRF metrics
sig_clusters = [(time_plot[mask][0], time_plot[mask][-1]) for i, mask in enumerate(clusters) if cluster_p_vals[i] < 0.05]
extract_cluster_metrics(sig_clusters, target_trfs, distractor_trfs, time_plot)

plot_individual_trfs_with_sd(target_trfs, time_plot, title='Target Number TRFs (per subject)', color='royalblue')
plot_individual_trfs_with_sd(distractor_trfs, time_plot, title='Distractor Number TRFs (per subject)', color='darkred')



# === Extract RMS and PTP in early/late windows and save
output_dir = os.path.join(weights_dir, "metrics_summary")
os.makedirs(output_dir, exist_ok=True)

df_subs, df_stats = extract_trf_metrics_by_window(
    target_trfs, distractor_trfs,
    time_plot, save_path=output_dir,
    label=f'{plane}_{cond}_{folder_type}'
)

print("\nSaved subject metrics and group stats to:", output_dir)
print(df_stats)


