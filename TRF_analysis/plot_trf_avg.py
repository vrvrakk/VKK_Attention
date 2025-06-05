import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, zscore, sem
from scipy.integrate import trapezoid
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
matplotlib.use('TkAgg')
plt.ion()

# === Configuration ===

plane = 'azimuth'
folder_types = ['all_stims', 'non_targets', 'target_nums', 'deviants']

folder_type = folder_types[0]
weights_dir = rf"C:/Users/pppar/PycharmProjects/VKK_Attention/data/eeg/trf/trf_testing/composite_model/single_sub/{plane}/{folder_type}/on_en_RT_ov/weights"
window_len = 11  # Hamming window length
sfreq = 125  # Sampling rate (Hz)
time_lags = np.linspace(-0.1, 1.0, 139)  # time axis

# === Helper: Smooth with Hamming window ===
def smooth_weights(weights):
    """
    Smooths each predictor (column) over time using a Hamming window.
    Expects weights of shape (n_lags, n_predictors)
    Returns smoothed weights of the same shape.
    """
    hamming_win = np.hamming(window_len)
    hamming_win /= hamming_win.sum()
    return np.array([
        np.convolve(weights[:, i], hamming_win, mode='same')
        for i in range(weights.shape[1])
    ]).T  # transpose back to (n_lags, n_predictors)

# === Load, smooth per subject, and stack all weights ===
def load_and_smooth_weights(stream_type):
    subject_ids = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub08']
    files = sorted([f for f in os.listdir(weights_dir) if stream_type in f and 'weights' in f])
    smoothed_all = []
    for f in files:
        if any(sub in f for sub in subject_ids):
            continue
        w = np.load(os.path.join(weights_dir, f), allow_pickle=True).squeeze().T  # (139, 4)
        smoothed = smooth_weights(w)
        smoothed_all.append(smoothed)
    smoothed_all = np.stack(smoothed_all, axis=0)  # (n_subjects, 139, 4)
    avg = np.mean(smoothed_all, axis=0).T  # (4, 139)
    return avg, smoothed_all  # average, per-subject

# Load data
smoothed_target, smoothed_target_all = load_and_smooth_weights("target_stream")
smoothed_target_all = smoothed_target_all[:, :, 1]
smoothed_distractor, smoothed_distractor_all = load_and_smooth_weights("distractor_stream")
smoothed_distractor_all = smoothed_distractor_all[:, :, 1]

# === Z-score data over time (axis=1) ===
smoothed_target_all_z = zscore(smoothed_target_all, axis=1)
smoothed_distractor_all_z = zscore(smoothed_distractor_all, axis=1)

# === Plotting ===
def plot_smoothed_response(data, title_prefix, colors, filename_prefix, predictor_labels=None):
    # Restrict to time lags between 0.1 and 0.4 seconds
    time_mask = (time_lags >= 0.0) & (time_lags <= 0.4)
    time_trimmed = time_lags[time_mask]

    for i in range(data.shape[0]):  # i = predictor index
        label = predictor_labels[i] if predictor_labels else f'Predictor {i+1}'
        plt.figure(figsize=(6, 3))
        plt.plot(time_trimmed, data[i, time_mask], color=colors[i], linewidth=2)
        plt.title(f'{title_prefix}: {label}')
        plt.xlabel('Time lag (s)')
        plt.ylabel('Amplitude (a.u.)')
        plt.tight_layout()
        plt.savefig(os.path.join(weights_dir, f'{filename_prefix}_{label}.png'), dpi=300)
        plt.show()

predictor_labels = ['Onsets', 'Envelopes', 'RTs', 'Overlap']
colors = ['royalblue', 'seagreen', 'firebrick', 'goldenrod']

# plot_smoothed_response(smoothed_target, "Target Stream", colors, 'avg_trf_target', predictor_labels)
# plot_smoothed_response(smoothed_distractor, "Distractor Stream", colors, 'avg_trf_distractor', predictor_labels)

######################

def compare_trf_metrics(start, end, time_lags, sfreq, data_target_all, distractor_data_all, label='envelope'):
    """
    Compare TRF metrics between target and distractor within a specific time window.

    Parameters
    ----------
    start : float
        Start time of the window (in seconds).
    end : float
        End time of the window (in seconds).
    time_lags : np.ndarray
        Time vector (e.g., from TRF model).
    sfreq : float
        EEG sampling frequency.
    smoothed_target_all : ndarray (n_subjects x n_times x n_channels)
    smoothed_distractor_all : ndarray (n_subjects x n_times x n_channels)
    label : str
        Label for the predictor (e.g., 'envelope', 'semantic').

    Returns
    -------
    result_dict : dict
        Dictionary of computed stats and p-values.
    """
    # === Time window mask ===
    window_mask = (time_lags >= start) & (time_lags <= end)
    time_resolution = 1 / sfreq

    # === Extract window data ===
    t_full = data_target_all[:, window_mask]
    d_full = distractor_data_all[:, window_mask]


    # === Compute metrics ===
    t_mean, d_mean = np.mean(t_full, axis=1), np.mean(d_full, axis=1)
    t_rms, d_rms = np.sqrt(np.mean(t_full**2, axis=1)), np.sqrt(np.mean(d_full**2, axis=1))
    t_auc, d_auc = trapezoid(t_full, dx=time_resolution), trapezoid(d_full, dx=time_resolution)

    pos_peak_t = np.max(t_full, axis=1)
    pos_peak_idx_t = np.argmax(t_full, axis=1)
    pos_peak_latency_t = time_lags[window_mask][pos_peak_idx_t]

    neg_peak_t = np.min(t_full, axis=1)
    neg_peak_idx_t = np.argmin(t_full, axis=1)
    neg_peak_latency_t = time_lags[window_mask][neg_peak_idx_t]

    pos_amp_d = np.array([d_full[i, pos_peak_idx_t[i]] for i in range(len(d_full))])
    neg_amp_d = np.array([d_full[i, neg_peak_idx_t[i]] for i in range(len(d_full))])

    pos_peak_d = np.max(d_full, axis=1)
    neg_peak_d = np.min(d_full, axis=1)

    ptp_t = pos_peak_t - neg_peak_t
    ptp_d = pos_peak_d - neg_peak_d

    pos_peak_latency_d = time_lags[window_mask][np.argmax(d_full, axis=1)]
    neg_peak_latency_d = time_lags[window_mask][np.argmin(d_full, axis=1)]

    # create metrics dictionary:
    metrics_dict = {
        'target_mean': t_mean,
        'distractor_mean': d_mean,
        'target_rms': t_rms,
        'distractor_rms': d_rms,
        'target_auc': t_auc,
        'distractor_auc': d_auc,
        'target_pos_peak': pos_peak_t,
        'target_pos_peak_latency': pos_peak_latency_t,
        'target_neg_peak': neg_peak_t,
        'target_neg_peak_latency': neg_peak_latency_t,
        'distractor_pos_peak': pos_peak_d,
        'distractor_pos_peak_latency': pos_peak_latency_d,
        'distractor_neg_peak': neg_peak_d,
        'distractor_neg_peak_latency': neg_peak_latency_d,
        'target_ptp': ptp_t,
        'distractor_ptp': ptp_d,
        'distractor_pos_amp_at_target_peak': pos_amp_d,
        'distractor_neg_amp_at_target_peak': neg_amp_d
    }

    # === Stat helpers ===
    def safe_wilcoxon(x, y):
        try:
            return wilcoxon(x, y).pvalue
        except:
            return np.nan

    def cohen_d(x, y):
        return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x) ** 2 + np.std(y) ** 2) / 2)

    # === Package results ===
    result = {
        'Window': f'{start*1000:.0f}–{end*1000:.0f} ms',
        'Predictor': label,

        # Descriptives
        'Target Mean ± SD': f'{np.mean(t_mean):.4f} ± {np.std(t_mean):.4f}',
        'Distractor Mean ± SD': f'{np.mean(d_mean):.4f} ± {np.std(d_mean):.4f}',
        'Target RMS ± SD': f'{np.mean(t_rms):.4f} ± {np.std(t_rms):.4f}',
        'Distractor RMS ± SD': f'{np.mean(d_rms):.4f} ± {np.std(d_rms):.4f}',
        'Target AUC ± SD': f'{np.mean(t_auc):.4f} ± {np.std(t_auc):.4f}',
        'Distractor AUC ± SD': f'{np.mean(d_auc):.4f} ± {np.std(d_auc):.4f}',
        'Target Pos Peak ± SD': f'{np.mean(pos_peak_t):.4f} ± {np.std(pos_peak_t):.4f}',
        'Target Neg Peak ± SD': f'{np.mean(neg_peak_t):.4f} ± {np.std(neg_peak_t):.4f}',
        'Distractor@Target Pos Peak ± SD': f'{np.mean(pos_amp_d):.4f} ± {np.std(pos_amp_d):.4f}',
        'Distractor@Target Neg Peak ± SD': f'{np.mean(neg_amp_d):.4f} ± {np.std(neg_amp_d):.4f}',
        'Target PTP ± SD': f'{np.mean(ptp_t):.4f} ± {np.std(ptp_t):.4f}',
        'Distractor PTP ± SD': f'{np.mean(ptp_d):.4f} ± {np.std(ptp_d):.4f}',

        # Tests

        't-test p (Mean)': ttest_rel(t_mean, d_mean).pvalue,
        'Wilcoxon p (Mean)': safe_wilcoxon(t_mean, d_mean),
        'Cohen d (Mean)': cohen_d(t_mean, d_mean),

        't-test p (RMS)': ttest_rel(t_rms, d_rms).pvalue,
        'Wilcoxon p (RMS)': safe_wilcoxon(t_rms, d_rms),
        'Cohen d (RMS)': cohen_d(t_rms, d_rms),

        't-test p (Pos Peak)': ttest_rel(pos_peak_t, pos_amp_d).pvalue,
        'Wilcoxon p (Pos Peak)': safe_wilcoxon(pos_peak_t, pos_amp_d),
        'Cohen d (Pos Peak)': cohen_d(pos_peak_t, pos_amp_d),

        't-test p (Neg Peak)': ttest_rel(neg_peak_t, neg_amp_d).pvalue,
        'Wilcoxon p (Neg Peak)': safe_wilcoxon(neg_peak_t, neg_amp_d),
        'Cohen d (Neg Peak)': cohen_d(neg_peak_t, neg_amp_d),

        't-test p (PTP)': ttest_rel(ptp_t, ptp_d).pvalue,
        'Wilcoxon p (PTP)': safe_wilcoxon(ptp_t, ptp_d),
        'Cohen d (PTP)': cohen_d(ptp_t, ptp_d),

        't-test p (Pos t)': ttest_rel(pos_peak_latency_t, pos_peak_latency_d).pvalue,
        'Wilcoxon p (Pos t)': safe_wilcoxon(pos_peak_latency_t, pos_peak_latency_d),
        'Cohen d (Pos t)': cohen_d(pos_peak_latency_t, pos_peak_latency_d),

        't-test p (Neg t)': ttest_rel(neg_peak_latency_t, neg_peak_latency_d).pvalue,
        'Wilcoxon p (Neg t)': safe_wilcoxon(neg_peak_latency_t, neg_peak_latency_d),
        'Cohen d (Neg t)': cohen_d(neg_peak_latency_t, neg_peak_latency_d),

        't-test p (AUC)': ttest_rel(t_auc, d_auc).pvalue,
        'Wilcoxon p (AUC)': safe_wilcoxon(t_auc, d_auc),
        'Cohen d (AUC)': cohen_d(t_auc, d_auc),
    }

    return result, metrics_dict

# === Save and report ===
# Full window
result_full_raw, metrics_full_raw = compare_trf_metrics(0.0, 0.5, time_lags, sfreq, smoothed_target_all, smoothed_distractor_all)
result_full_z, metrics_full_z = compare_trf_metrics(0.0, 0.5, time_lags, sfreq, smoothed_target_all_z, smoothed_distractor_all_z)
# Early window
result_early_raw, metrics_early_raw = compare_trf_metrics(0.0, 0.2, time_lags, sfreq, smoothed_target_all, smoothed_distractor_all)
result_early_z, metrics_early_z = compare_trf_metrics(0.0, 0.2, time_lags, sfreq, smoothed_target_all_z, smoothed_distractor_all_z)

# Late window
result_late_raw, metrics_late_raw = compare_trf_metrics(0.2, 0.5, time_lags, sfreq, smoothed_target_all, smoothed_distractor_all)
result_late_z, metrics_late_z = compare_trf_metrics(0.2, 0.5, time_lags, sfreq, smoothed_target_all_z, smoothed_distractor_all_z)

# Combine into DataFrame
result_dict_raw_all = {
    'full': result_full_raw,
    'early': result_early_raw,
    'late': result_late_raw
}

result_dict_z_all = {
    'full': result_full_z,
    'early': result_early_z,
    'late': result_late_z
}

metrics_dict_raw_all = {
    'full': metrics_full_raw,
    'early': metrics_early_raw,
    'late': metrics_late_raw
}

metrics_dict_z_all = {
    'full': metrics_full_z,
    'early': metrics_early_z,
    'late': metrics_late_z
}

# === Testing for normality ===
from scipy.stats import shapiro
# === Define all metric arrays ===


def run_normality_tests(metrics_dict_all, window='full', save_path=None, label=''):
    """
    Run Shapiro-Wilk normality test on each metric from a specific time window.

    Parameters
    ----------
    metrics_dict_all : dict
        Dictionary containing 'full', 'early', 'late' keys, each mapping to a metrics dictionary.
    window : str
        Which time window to test ('full', 'early', or 'late').
    save_path : str or None
        Optional path to save the results as a CSV.
    label : str
        Optional label to include in filename (e.g., 'z', 'raw').

    Returns
    -------
    pd.DataFrame
        DataFrame with metric name, p-value, and normality boolean.
    """
    assert window in ['full', 'early', 'late'], "Window must be 'full', 'early', or 'late'."

    metrics_dict = metrics_dict_all[window]

    normality_results = {
        'Metric': [],
        'Shapiro p-value': [],
        'Normally Distributed': []
    }

    for name, values in metrics_dict.items():
        try:
            stat, p = shapiro(values)
            normal = p > 0.05
        except Exception as e:
            p, normal = np.nan, False
        normality_results['Metric'].append(name)
        normality_results['Shapiro p-value'].append(p)
        normality_results['Normally Distributed'].append(normal)

    df = pd.DataFrame(normality_results)

    if save_path:
        filename = f'normality_test_results_{label}_{window}.csv'
        df.to_csv(os.path.join(save_path, filename), index=False)

    return df

normality_df_full_raw = run_normality_tests(metrics_dict_raw_all, window='full', save_path=weights_dir, label='raw')
normality_df_early_z = run_normality_tests(metrics_dict_z_all, window='early', save_path=weights_dir, label='z')



# === FDR correction on p-values of all metrics ===

def fdr_corr(results_dict_all, window='full', test_type='wilcoxon'):
    """
    Apply FDR correction to p-values from a result dictionary for a given time window.

    Parameters
    ----------
    results_dict_all : dict
        Dictionary with 'full', 'early', 'late' result dicts.
    window : str
        One of 'full', 'early', 'late'.
    test_type : str
        't-test' or 'wilcoxon'. Determines which test's p-values to use.

    Returns
    -------
    fdr_results : dict
        Dictionary of FDR-corrected p-values.
    """
    assert test_type in ['t-test', 'Wilcoxon'], "Choose 't-test' or 'wilcoxon'"
    assert window in ['full', 'early', 'late']

    # Metrics to correct
    metric_keys = ['Mean', 'RMS', 'Pos Peak', 'Neg Peak', 'PTP', 'Pos t', 'Neg t', 'AUC']
    pval_key_template = f'{test_type} p ({{}})'

    # Extract relevant p-values
    pvals = []
    for m in metric_keys:
        col_name = pval_key_template.format(m)
        pvals.append(results_dict_all[window][col_name])

    # Run FDR correction
    rejected, pvals_corrected = fdrcorrection(pvals, alpha=0.05)

    # Save back corrected values
    fdr_results = {}
    for i, m in enumerate(metric_keys):
        label = f'FDR-corrected p ({m})'
        results_dict_all[window][label] = pvals_corrected[i]
        fdr_results[label] = pvals_corrected[i]

    return fdr_results

fdr_results_full_raw = fdr_corr(result_dict_raw_all, window='full', test_type='Wilcoxon')


################

# === Plot Comparison: Target vs Distractor for Each Predictor ===
comparison_dir = os.path.join(weights_dir, 'comparison_plots')
os.makedirs(comparison_dir, exist_ok=True)

import seaborn as sns

def get_sig_star(p):
    if p < 0.001:
        return '***'
    elif p < 0.01:
        return '**'
    elif p < 0.05:
        return '*'
    else:
        return 'n.s.'


def parse_pval(p):
    """
    Convert a p-value to float for comparison:
    - If already a float, returns it.
    - If string like '< 0.01', returns 0.01.
    - Otherwise returns np.nan.
    """
    if isinstance(p, str):
        p = p.strip()
        if p.startswith('<'):
            try:
                return float(p[1:].strip())
            except ValueError:
                return np.nan
        try:
            return float(p)
        except ValueError:
            return np.nan
    elif isinstance(p, (float, int, np.floating)):
        return float(p)
    return np.nan


def plot_metrics(metrics_df_dict, results_dict, window='full', alpha=0.05):
    metrics_dict = metrics_df_dict[window]
    result_row = results_dict[window]

    # === Identify matching metric name stems ===
    metric_names = set()
    for key in metrics_dict:
        if key.startswith('target_'):
            base = key.replace('target_', '')
            if f'distractor_{base}' in metrics_dict:
                metric_names.add(base)

    # === Filter based on p-value < alpha ===
    selected_metrics = []
    for base in metric_names:
        pkey = f't-test p ({base.replace("_", " ").title()})'
        if pkey in result_row:
            p_val = parse_pval(result_row[pkey])
            if not np.isnan(p_val) and p_val < alpha:
                selected_metrics.append((base, pkey))

    if not selected_metrics:
        print(f"No significant metrics (p < {alpha}) for window: {window}")
        return

    # === Plotting ===
    cols = len(selected_metrics)
    rows = int(np.ceil(len(selected_metrics) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for i, (base, pkey) in enumerate(selected_metrics):
        tkey = f'target_{base}'
        dkey = f'distractor_{base}'
        tvals, dvals = metrics_dict[tkey], metrics_dict[dkey]

        df = pd.DataFrame({
            'Value': list(tvals) + list(dvals),
            'Stream': ['Target'] * len(tvals) + ['Distractor'] * len(dvals)
        })

        ax = axes[i]
        sns.boxplot(x='Stream', y='Value', data=df, ax=ax)
        sns.stripplot(x='Stream', y='Value', data=df, color='black', size=3, alpha=0.4, ax=ax)
        raw_p = result_row[pkey]
        parsed_p = parse_pval(raw_p)
        ax.set_title(f'{base.replace("_", " ").title()}')
        y_max = max(np.max(tvals), np.max(dvals))
        y_range = y_max - min(np.min(tvals), np.min(dvals))
        y_text = y_max + 0.1 * y_range

        ax.plot([0, 1], [y_text] * 2, color='black')
        ax.text(0.5, y_text + 0.02 * y_range, get_sig_star(parsed_p),
                ha='center', va='bottom', fontsize=12)
    for j in range(len(selected_metrics), len(axes)):
        fig.delaxes(axes[j])

    fig.tight_layout()
    fig.savefig(os.path.join(comparison_dir, f'trf_sig_metrics_{window}.png'), dpi=300)
    plt.show()


plot_metrics(metrics_dict_z_all, result_dict_z_all, window='full')
plot_metrics(metrics_dict_raw_all, result_dict_raw_all, window='early')
plot_metrics(metrics_dict_raw_all, result_dict_raw_all, window='late')


# --- Plot TRF Responses --- #
from scipy.stats import sem

# === Extract envelope predictor (index 1) and apply time mask ===
plot_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_plot = time_lags[plot_mask]

target_trfs = smoothed_target_all_z[:, plot_mask]
distractor_trfs = smoothed_distractor_all_z[:, plot_mask]

from statsmodels.stats.multitest import fdrcorrection

# Run paired t-test for each time point
p_vals = np.array([ttest_rel(target_trfs[:, i], distractor_trfs[:, i]).pvalue for i in range(target_trfs.shape[1])])

_, p_fdr = fdrcorrection(p_vals)
sig_mask = p_fdr < 0.05

# === Compute Mean and SEM ===
target_mean = target_trfs.mean(axis=0)
target_sem = sem(target_trfs, axis=0)

distractor_mean = distractor_trfs.mean(axis=0)
distractor_sem = sem(distractor_trfs, axis=0)

# === Plotting ===
plt.figure(figsize=(8, 4))
plt.plot(time_plot, target_mean, color='royalblue', label='Target', linewidth=2)
plt.fill_between(time_plot, target_mean - target_sem, target_mean + target_sem, color='royalblue', alpha=0.3)

plt.plot(time_plot, distractor_mean, color='darkorange', label='Distractor', linewidth=2)
plt.fill_between(time_plot, distractor_mean - distractor_sem, distractor_mean + distractor_sem, color='darkorange', alpha=0.3)

in_sig = False
for i in range(len(sig_mask)):
    if sig_mask[i] and not in_sig:
        in_sig = True
        start_idx = i
        start = time_plot[i]
    elif not sig_mask[i] and in_sig:
        in_sig = False
        end_idx = i
        end = time_plot[i]

        # Draw shaded region
        plt.axvspan(start, end, color='gray', alpha=0.2)

        # Optional: Label with asterisk or p-value
        center_time = (start + end) / 2
        y_max = max(np.max(target_mean[start_idx:end_idx]), np.max(distractor_mean[start_idx:end_idx]))
        y_text = y_max + 0.05  # small gap above the wave

        # Get minimum corrected p-value in this cluster
        min_p = p_fdr[start_idx:end_idx].min()
        label = get_sig_star(min_p)

        plt.text(center_time, y_text, label, ha='center', va='bottom', fontsize=12, color='black')

plt.xlabel('Time lag (s)')
plt.ylabel('TRF Amplitude (a.u.)')
plt.title('Envelope TRF: Target vs Distractor (with significance)')
plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(comparison_dir, 'envelope_trf_target_vs_distractor_with_sig.png'), dpi=300)
plt.show()

# == Temporal Cluster Permutation ===
from mne.stats import permutation_cluster_test

# Prepare the data (shape: [n_subjects, n_timepoints])
X = [smoothed_target_all[:, :], smoothed_distractor_all[:, :]]  # list of arrays

# Run cluster test
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    X,
    n_permutations=1000,
    tail=0,
    threshold=None,
    out_type='mask',  # <- ensure it's a boolean mask
    verbose=True
)

# Plot TRF curves
# Define cutoff index for 0.5s
cutoff_idx = np.where(time_lags <= 0.5)[0]

# Restrict time and data to 0–0.5s
time_lags_plot = time_lags[cutoff_idx]
target_plot = X[0].mean(axis=0)[cutoff_idx]
distractor_plot = X[1].mean(axis=0)[cutoff_idx]

plt.figure(figsize=(10, 4))
plt.plot(time_lags_plot, target_plot, label='Target')
plt.plot(time_lags_plot, distractor_plot, label='Distractor')

# Highlight significant clusters (only if within 0–0.5s)
for i_c, cluster in enumerate(clusters):
    if cluster_p_values[i_c] < 0.05:
        cluster_mask = np.zeros_like(time_lags, dtype=bool)
        cluster_mask[cluster] = True  # works for both slices and indices
        cluster_mask = cluster_mask & (time_lags <= 0.5)
        if cluster_mask.any():
            plt.axvspan(
                time_lags[cluster_mask][0],
                time_lags[cluster_mask][-1],
                color='red', alpha=0.3
            )


significant_clusters = np.where(cluster_p_values < 0.05)[0]

for idx in significant_clusters:
    cluster_mask = np.zeros_like(time_lags, dtype=bool)
    cluster_mask[clusters[idx]] = True
    cluster_mask = cluster_mask & (time_lags <= 0.5)
    if cluster_mask.any():
        start = time_lags[cluster_mask][0]
        end = time_lags[cluster_mask][-1]
        p_val = cluster_p_values[idx]
        print(f"Cluster {idx}: p = {p_val:.4f}, Time = {start:.3f}s to {end:.3f}s")


plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.title('TRF Cluster Permutation Result (0–0.5s, ROI)')
plt.xlim([time_lags_plot[0], time_lags_plot[-1]])
plt.tight_layout()
plt.show()









