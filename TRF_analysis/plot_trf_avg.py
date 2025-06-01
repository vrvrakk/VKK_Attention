import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')
plt.ion()

# === Configuration ===
plane = 'elevation'
weights_dir = rf"C:\Users\pppar\PycharmProjects\VKK_Attention\data\eeg\trf\trf_testing\composite_model\single_sub\{plane}\all_stims\on_en_RT_ov\weights"
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

plot_smoothed_response(smoothed_target, "Target Stream", colors, 'avg_trf_target', predictor_labels)
plot_smoothed_response(smoothed_distractor, "Distractor Stream", colors, 'avg_trf_distractor', predictor_labels)


###############

# === Comparison Plots & Stats ===

from scipy.stats import ttest_rel, wilcoxon
import pandas as pd

# Use absolute max for these predictors (e.g., RTs), else use positive max
use_abs_peak = ['RTs']  # add others if needed

# === Time window for stat analysis ===
# Define full window for mean/RMS (e.g., 0–0.4s), and peak window (e.g., 0.1–0.3s)
full_mask = (time_lags >= 0.0) & (time_lags <= 0.4)
peak_mask = (time_lags >= 0.1) & (time_lags <= 0.3)
peak_lag_indices = np.where(peak_mask)[0]

results = []

for i, label in enumerate(predictor_labels):
    # === Extract data ===
    t_full = smoothed_target_all[:, full_mask, i]    # For mean/RMS
    d_full = smoothed_distractor_all[:, full_mask, i]

    t_peakwin = smoothed_target_all[:, peak_mask, i]  # For peak
    d_peakwin = smoothed_distractor_all[:, peak_mask, i]

    # === Compute mean & RMS from full window ===
    t_mean = np.mean(np.abs(t_full), axis=1)
    d_mean = np.mean(np.abs(d_full), axis=1)
    t_rms = np.sqrt((t_full ** 2).mean(axis=1))
    d_rms = np.sqrt((d_full ** 2).mean(axis=1))

    # === Compute peak from peak window ===
    if label in use_abs_peak:
        t_peak_indices = np.argmax(t_peakwin, axis=1)
        t_max = np.max(t_peakwin, axis=1)
    else:
        t_peak_indices = np.argmax(t_peakwin, axis=1)
        t_max = np.max(t_peakwin, axis=1)

    # Get distractor amplitude at target-peak positions
    d_at_target_peak = np.take_along_axis(d_peakwin, t_peak_indices[:, np.newaxis], axis=1).flatten()
    peak_diff = t_max - d_at_target_peak

    # === Utility functions ===
    def safe_wilcoxon(x, y):
        try:
            return wilcoxon(x, y).pvalue
        except:
            return np.nan

    def cohen_d(x, y):
        return (np.mean(x) - np.mean(y)) / np.sqrt(((np.std(x) ** 2 + np.std(y) ** 2) / 2))

    # === Append results ===
    results.append({
        'Predictor': label,
        'Target Mean ± SD': f'{np.mean(t_mean):.4f} ± {np.std(t_mean):.4f}',
        'Distractor Mean ± SD': f'{np.mean(d_mean):.4f} ± {np.std(d_mean):.4f}',
        'Target RMS': np.mean(t_rms),
        'Distractor RMS': np.mean(d_rms),
        'Target Max': np.mean(t_max),
        'Distractor@Target-Peak': np.mean(d_at_target_peak),
        'Peak Diff': np.mean(peak_diff),
        't-test p (Mean)': ttest_rel(t_mean, d_mean).pvalue,
        'Wilcoxon p (Mean)': safe_wilcoxon(t_mean, d_mean),
        'Cohen d (Mean)': cohen_d(t_mean, d_mean),
        't-test p (RMS)': ttest_rel(t_rms, d_rms).pvalue,
        'Wilcoxon p (RMS)': safe_wilcoxon(t_rms, d_rms),
        'Cohen d (RMS)': cohen_d(t_rms, d_rms),
        't-test p (Max)': ttest_rel(t_max, d_at_target_peak).pvalue,
        'Wilcoxon p (Max)': safe_wilcoxon(t_max, d_at_target_peak),
        'Cohen d (Max)': cohen_d(t_max, d_at_target_peak)
    })

# === Save and report ===
stats_df = pd.DataFrame(results)
stats_df.to_csv(os.path.join(weights_dir, 'trf_stats_comparison.csv'), index=False)
print(stats_df)

################

# === Plot Comparison: Target vs Distractor for Each Predictor ===
comparison_dir = os.path.join(weights_dir, 'comparison_plots')
os.makedirs(comparison_dir, exist_ok=True)

for i, label in enumerate(predictor_labels):
    # === Extract TRF response trimmed to 0–0.4 s for plotting ===
    full_mask = (time_lags >= 0.0) & (time_lags <= 0.4)
    time_trimmed = time_lags[full_mask]
    t_vals = smoothed_target[i][full_mask]
    d_vals = smoothed_distractor[i][full_mask]

    # === Trimmed peak window: 0.1–0.3 s ===
    peak_mask = (time_trimmed >= 0.1) & (time_trimmed <= 0.3)
    lags_peakwin = time_trimmed[peak_mask]
    t_trim = t_vals[peak_mask]
    d_trim = d_vals[peak_mask]

    # === Choose peak logic ===
    use_abs = label in ['RTs']  # Modify as needed

    if use_abs:
        t_peak_idx = np.argmax(t_trim)
    else:
        t_peak_idx = np.argmax(t_trim)

    # === Get peak time and amplitudes ===
    t_peak_time = lags_peakwin[t_peak_idx]
    t_peak_amp = t_trim[t_peak_idx]
    d_peak_amp = d_trim[t_peak_idx]  # value at same time in distractor

    # === Extract stats ===
    stats = stats_df.iloc[i]
    p_val = stats['t-test p (Max)']
    if p_val < 0.001:
        p_display = "0.001"
    else:
        p_display = f"{p_val:.4f}"
    max_diff = stats['Cohen d (Max)']
    peak_note = "abs" if use_abs else "pos"

    # === Plot ===
    plt.figure(figsize=(8, 4))
    plt.plot(time_trimmed, t_vals, label='Target', color='royalblue', linewidth=2)
    plt.plot(time_trimmed, d_vals, label='Distractor', color='darkorange', linewidth=2)
    plt.axvline(t_peak_time, linestyle='--', color='royalblue', alpha=0.5)
    plt.axvline(t_peak_time, linestyle='--', color='darkorange', alpha=0.5)
    plt.title(f'{label} [{peak_note} peak], p={p_display}, Max d={max_diff:.2f}')
    plt.xlabel('Time lag (s)')
    plt.ylabel('Amplitude (a.u.)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(comparison_dir, f'{label}_target_vs_distractor.png'), dpi=300)
    plt.show()