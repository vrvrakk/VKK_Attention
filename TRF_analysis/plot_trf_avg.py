import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import pandas as pd
matplotlib.use('TkAgg')
plt.ion()

# === Configuration ===

plane = 'elevation'
folder_types = ['all_stims', 'non_targets', 'target_nums', 'deviants']

folder_type = folder_types[3]
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
smoothed_distractor, smoothed_distractor_all = load_and_smooth_weights("distractor_stream")


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

# === Comparison Plots & Stats ===

from scipy.stats import ttest_rel, wilcoxon
import pandas as pd

# === Time window for stat analysis ===
# Define full window for mean/RMS (e.g., 0–0.4s), and peak window (e.g., 0.1–0.3s)
full_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
peak_mask = (time_lags >= 0.1) & (time_lags <= 0.5)
peak_lag_indices = np.where(peak_mask)[0]

results = []


    # === Extract data ===
t_full = smoothed_target_all[:, full_mask, 1]    # For mean/RMS
d_full = smoothed_distractor_all[:, full_mask, 1]

# === Compute mean & RMS from full window ===
t_mean = np.mean(t_full, axis=1)
d_mean = np.mean(d_full, axis=1)
t_rms = np.sqrt((t_full ** 2).mean(axis=1))
d_rms = np.sqrt((d_full ** 2).mean(axis=1))


pos_peak_t = np.max(t_full, axis=1)
pos_peak_idx_t = np.argmax(t_full, axis=1)              # index of max for each subject
pos_peak_latency_t = time_lags[pos_peak_idx_t]            # corresponding time in seconds
neg_peak_t = np.min(t_full, axis=1)
neg_peak_idx_t = np.argmin(t_full, axis=1)
neg_peak_latency_t = time_lags[neg_peak_idx_t]
# Distractor amplitude at each subject's target peak time
pos_amp_d = np.array([d_full[i, pos_peak_idx_t[i]] for i in range(len(d_full))])
neg_amp_d = np.array([d_full[i, neg_peak_idx_t[i]] for i in range(len(d_full))])
### true positive and negative peaks of distractor:
pos_peak_d = np.max(d_full, axis=1)
pos_peak_idx_d = np.argmax(d_full, axis=1)              # index of max for each subject
pos_peak_latency_d = time_lags[pos_peak_idx_d]            # corresponding time in seconds
neg_peak_d = np.min(d_full, axis=1)
neg_peak_idx_d = np.argmin(d_full, axis=1)
neg_peak_latency_d = time_lags[neg_peak_idx_d]

# === Peak-to-Peak Dynamic Range (per subject) ===
ptp_t = pos_peak_t - neg_peak_t     # For target stream
ptp_d = pos_peak_d - neg_peak_d     # For distractor stream

# === Peak Latency Jitter (standard deviation across subjects) ===
pos_peak_latency_jitter_t = np.std(pos_peak_latency_t)  # seconds
neg_peak_latency_jitter_t = np.std(neg_peak_latency_t)

pos_peak_latency_jitter_d = np.std(pos_peak_latency_d)
neg_peak_latency_jitter_d = np.std(neg_peak_latency_d)

# You could also compute SEM:
from scipy.stats import sem
pos_peak_latency_sem_t = sem(pos_peak_latency_t)


# === Utility functions ===
def safe_wilcoxon(x, y):
    try:
        return wilcoxon(x, y).pvalue
    except:
        return np.nan

def cohen_d(x, y):
    return (np.mean(x) - np.mean(y)) / np.sqrt(((np.std(x) ** 2 + np.std(y) ** 2) / 2))

results.append({
    'Predictor': 'envelope',

    # --- Descriptive stats ---
    'Target Mean ± SD': f'{np.mean(t_mean):.4f} ± {np.std(t_mean):.4f}',
    'Distractor Mean ± SD': f'{np.mean(d_mean):.4f} ± {np.std(d_mean):.4f}',
    'Target RMS ± SD': f'{np.mean(t_rms):.4f} ± {np.std(t_rms):.4f}',
    'Distractor RMS ± SD': f'{np.mean(d_rms):.4f} ± {np.std(d_rms):.4f}',
    'Target Pos Peak ± SD': f'{np.mean(pos_peak_t):.4f} ± {np.std(pos_peak_t):.4f}',
    'Target Neg Peak ± SD': f'{np.mean(neg_peak_t):.4f} ± {np.std(neg_peak_t):.4f}',
    'Distractor@Target Pos Peak ± SD': f'{np.mean(pos_amp_d):.4f} ± {np.std(pos_amp_d):.4f}',
    'Distractor@Target Neg Peak ± SD': f'{np.mean(neg_amp_d):.4f} ± {np.std(neg_amp_d):.4f}',
    'Target PTP ± SD': f'{np.mean(ptp_t):.4f} ± {np.std(ptp_t):.4f}',
    'Distractor PTP ± SD': f'{np.mean(ptp_d):.4f} ± {np.std(ptp_d):.4f}',

    # --- Mean amplitude comparison ---
    't-test p (Mean)': ttest_rel(t_mean, d_mean).pvalue,
    'Wilcoxon p (Mean)': safe_wilcoxon(t_mean, d_mean),
    'Cohen d (Mean)': cohen_d(t_mean, d_mean),

    # --- RMS comparison ---
    't-test p (RMS)': ttest_rel(t_rms, d_rms).pvalue,
    'Wilcoxon p (RMS)': safe_wilcoxon(t_rms, d_rms),
    'Cohen d (RMS)': cohen_d(t_rms, d_rms),

    # --- Positive peak comparison (true peak) ---
    't-test p (Pos Peak)': ttest_rel(pos_peak_t, pos_amp_d).pvalue,
    'Wilcoxon p (Pos Peak)': safe_wilcoxon(pos_peak_t, pos_amp_d),
    'Cohen d (Pos Peak)': cohen_d(pos_peak_t, pos_amp_d),

    # --- Negative peak comparison (true peak) ---
    't-test p (Neg Peak)': ttest_rel(neg_peak_t, neg_amp_d).pvalue,
    'Wilcoxon p (Neg Peak)': safe_wilcoxon(neg_peak_t, neg_amp_d),
    'Cohen d (Neg Peak)': cohen_d(neg_peak_t, neg_amp_d),

    # --- Peak-to-peak comparison ---
    't-test p (PTP)': ttest_rel(ptp_t, ptp_d).pvalue,
    'Wilcoxon p (PTP)': safe_wilcoxon(ptp_t, ptp_d),
    'Cohen d (PTP)': cohen_d(ptp_t, ptp_d),
    # --- Pos Peak Latency comparison ---
    't-test p (Pos t)': ttest_rel(pos_peak_latency_t, pos_peak_latency_d).pvalue,
    'Wilcoxon p (Pos t)': safe_wilcoxon(pos_peak_latency_t, pos_peak_latency_d),
    'Cohen d (Pos t)': cohen_d(pos_peak_latency_t, pos_peak_latency_d),

    # --- Neg Peak Latency comparison ---
    't-test p (Neg t)': ttest_rel(neg_peak_latency_t, neg_peak_latency_d).pvalue,
    'Wilcoxon p (Neg t)': safe_wilcoxon(neg_peak_latency_t, neg_peak_latency_d),
    'Cohen d (Neg t)': cohen_d(neg_peak_latency_t, neg_peak_latency_d),
})

# === Save and report ===
stats_df = pd.DataFrame(results)
stats_df.to_csv(os.path.join(weights_dir, 'trf_stats_comparison.csv'), index=False)
print(stats_df)

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


metrics = {
    'Mean Amplitude': (t_mean, d_mean),
    'RMS': (t_rms, d_rms),
    'Positive Peak': (pos_peak_t, pos_amp_d),
    'Negative Peak': (neg_peak_t, neg_amp_d),
    'Peak-to-Peak': (ptp_t, ptp_d),
    'Pos Peak Latency': (pos_peak_latency_t, pos_peak_latency_d),
    'Neg Peak Latency': (neg_peak_latency_t, neg_peak_latency_d),
}

metric_to_pkey = {
    'Mean Amplitude': 'Wilcoxon p (Mean)',
    'RMS': 'Wilcoxon p (RMS)',
    'Positive Peak': 'Wilcoxon p (Pos Peak)',
    'Negative Peak': 'Wilcoxon p (Neg Peak)',
    'Peak-to-Peak': 'Wilcoxon p (PTP)',
    'Pos Peak Latency': 'Wilcoxon p (Pos t)',
    'Neg Peak Latency': 'Wilcoxon p (Neg t)',}


num_metrics = len(metrics)
cols = 4
rows = int(np.ceil(num_metrics / cols))

fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
axes = axes.flatten()

for i, (label, (target_vals, distractor_vals)) in enumerate(metrics.items()):
    data = pd.DataFrame({
        'Value': list(target_vals) + list(distractor_vals),
        'Stream': ['Target'] * len(target_vals) + ['Distractor'] * len(distractor_vals)
    })
    ax = axes[i]
    sns.boxplot(x='Stream', y='Value', data=data, ax=ax)
    sns.stripplot(x='Stream', y='Value', data=data, color='black', size=3, alpha=0.4, ax=ax)
    ax.set_title(label)

    # === Add significance annotation ===
    pkey = metric_to_pkey.get(label)
    if pkey and pkey in stats_df.columns:
        p_val = stats_df.loc[0, pkey]
        star = get_sig_star(p_val)

        # Position: above top point
        y_max = max(np.max(target_vals), np.max(distractor_vals))
        y_min = min(np.min(target_vals), np.min(distractor_vals))
        y_range = y_max - y_min
        y_text = y_max + 0.1 * y_range

        ax.plot([0, 1], [y_text] * 2, color='black')
        ax.text(0.5, y_text + 0.02 * y_range, star, ha='center', va='bottom', fontsize=12)

# Remove unused axes
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

fig.tight_layout()
fig.savefig(os.path.join(comparison_dir, 'trf_boxplot_comparison.png'), dpi=300)

# --- Plot TRF Responses --- #
from scipy.stats import sem

# === Extract envelope predictor (index 1) and apply time mask ===
plot_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_plot = time_lags[plot_mask]

target_trfs = smoothed_target_all[:, plot_mask, 1]
distractor_trfs = smoothed_distractor_all[:, plot_mask, 1]

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













