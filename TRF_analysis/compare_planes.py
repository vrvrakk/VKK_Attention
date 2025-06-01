import os
import numpy as np
import pandas as pd
from scipy.stats import ttest_rel, wilcoxon

# === Settings ===
predictor_labels = ['Onsets', 'Envelopes', 'RTs', 'Overlap']
use_abs_peak = ['RTs']  # Only use abs peaks for specific predictors
time_lags = np.linspace(-0.1, 1.0, 139)
full_mask = (time_lags >= 0.0) & (time_lags <= 0.4)
peak_mask = (time_lags >= 0.1) & (time_lags <= 0.3)

# === Helper: Load data per stream and plane ===
def load_weights(base_dir, plane, stream):
    sublist = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub08']
    path = os.path.join(base_dir, plane, 'all_stims', 'on_en_RT_ov', 'weights')
    files = [f for f in os.listdir(path) if stream in f and 'weights' in f and not any(sub in f for sub in sublist)]
    data = [np.load(os.path.join(path, f), allow_pickle=True).squeeze().T for f in sorted(files)]
    return np.stack(data)  # shape: (n_subjects, 139, n_predictors)

# === Comparison function ===
def compare_planes(data_az, data_el, label_idx, stream_name):
    az_full = data_az[:, full_mask, label_idx]
    el_full = data_el[:, full_mask, label_idx]
    az_peak = data_az[:, peak_mask, label_idx]
    el_peak = data_el[:, peak_mask, label_idx]

    az_mean = np.mean(np.abs(az_full), axis=1)
    el_mean = np.mean(np.abs(el_full), axis=1)
    az_rms = np.sqrt(np.mean(az_full ** 2, axis=1))
    el_rms = np.sqrt(np.mean(el_full ** 2, axis=1))

    if predictor_labels[label_idx] in use_abs_peak:
        az_idx = np.argmax(np.abs(az_peak), axis=1)
        az_max = np.max(np.abs(az_peak), axis=1)
    else:
        az_idx = np.argmax(az_peak, axis=1)
        az_max = np.max(az_peak, axis=1)

    el_at_az_peak = np.take_along_axis(el_peak, az_idx[:, None], axis=1).flatten()
    peak_diff = az_max - el_at_az_peak

    def cohen_d(x, y):
        return (np.mean(x) - np.mean(y)) / np.sqrt(((np.std(x)**2 + np.std(y)**2) / 2))

    def safe_wilcoxon(x, y):
        try: return wilcoxon(x, y).pvalue
        except: return np.nan

    return {
        'Stream': stream_name,
        'Predictor': predictor_labels[label_idx],
        'Az Mean ± SD': f'{np.mean(az_mean):.4f} ± {np.std(az_mean):.4f}',
        'El Mean ± SD': f'{np.mean(el_mean):.4f} ± {np.std(el_mean):.4f}',
        'Az RMS': np.mean(az_rms),
        'El RMS': np.mean(el_rms),
        'Az Max': np.mean(az_max),
        'El@Az-Peak': np.mean(el_at_az_peak),
        'Peak Diff': np.mean(peak_diff),
        't-test p (Mean)': ttest_rel(az_mean, el_mean).pvalue,
        'Wilcoxon p (Mean)': safe_wilcoxon(az_mean, el_mean),
        'Cohen d (Mean)': cohen_d(az_mean, el_mean),
        't-test p (RMS)': ttest_rel(az_rms, el_rms).pvalue,
        'Wilcoxon p (RMS)': safe_wilcoxon(az_rms, el_rms),
        'Cohen d (RMS)': cohen_d(az_rms, el_rms),
        't-test p (Max)': ttest_rel(az_max, el_at_az_peak).pvalue,
        'Wilcoxon p (Max)': safe_wilcoxon(az_max, el_at_az_peak),
        'Cohen d (Max)': cohen_d(az_max, el_at_az_peak)
    }

# === Run ===
base_dir = r"C:\Users\pppar\PycharmProjects\VKK_Attention\data\eeg\trf\trf_testing\composite_model\single_sub"
az_target = load_weights(base_dir, 'azimuth', 'target_stream')
el_target = load_weights(base_dir, 'elevation', 'target_stream')
az_distractor = load_weights(base_dir, 'azimuth', 'distractor_stream')
el_distractor = load_weights(base_dir, 'elevation', 'distractor_stream')

results = []
for i in range(len(predictor_labels)):
    results.append(compare_planes(az_target, el_target, i, 'Target'))
    results.append(compare_planes(az_distractor, el_distractor, i, 'Distractor'))

# === Save and print ===
df = pd.DataFrame(results)
out_path = os.path.join(base_dir, 'az_vs_el_stats')
os.makedirs(out_path, exist_ok=True)
df.to_csv(os.path.join(out_path, 'az_vs_el_TRF_comparison.csv'), index=False)
print(df)

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
comparison_dir = os.path.join(base_dir, 'az_vs_el_stats', 'comparison_plots')
os.makedirs(comparison_dir, exist_ok=True)


# First mask full window (0–0.4s), then within that: peak window (0.1–0.25s or 0.3s)
full_mask = (time_lags >= 0.0) & (time_lags <= 0.4)
peak_mask_within_trimmed = (time_lags[full_mask] >= 0.1) & (time_lags[full_mask] <= 0.3)

time_trimmed = time_lags[full_mask]
lags_peakwin = time_trimmed[peak_mask_within_trimmed]  # For x-axis

window_len = 11
hamming_win = np.hamming(window_len)
hamming_win /= hamming_win.sum()

for i, label in enumerate(predictor_labels):
    for stream_type, az_data, el_data in zip(
            ['Target', 'Distractor'],
            [az_target, az_distractor],
            [el_target, el_distractor]
    ):
        # === Average over subjects ===
        az_avg = az_data[:, :, i].mean(axis=0)  # (139,)
        el_avg = el_data[:, :, i].mean(axis=0)

        # === Full plot window (0–0.4s) ===
        az_avg_smooth = np.convolve(az_avg, hamming_win, mode='same')
        el_avg_smooth = np.convolve(el_avg, hamming_win, mode='same')

        # === Full plot window (0–0.4s) ===
        az_vals = az_avg_smooth[full_mask]
        el_vals = el_avg_smooth[full_mask]

        # === Peak window (0.1–0.3s) within trimmed range ===
        az_trim = az_vals[peak_mask_within_trimmed]
        el_trim = el_vals[peak_mask_within_trimmed]

        use_abs = label in ['RTs']  # Modify this as needed
        peak_idx = np.argmax(np.abs(az_trim)) if use_abs else np.argmax(az_trim)

        peak_time = lags_peakwin[peak_idx]
        az_peak_amp = az_trim[peak_idx]
        el_peak_amp = el_trim[peak_idx]

        stats = df[(df['Predictor'] == label) & (df['Stream'] == stream_type)].iloc[0]
        p_val = stats['t-test p (Max)']
        p_display = "0.001" if p_val < 0.001 else f"{p_val:.4f}"
        coh_d = stats['Cohen d (Max)']
        peak_note = "abs" if use_abs else "pos"

        # === Plot ===
        plt.figure(figsize=(8, 4))
        plt.plot(time_trimmed, az_vals, label='Azimuth', color='royalblue', linewidth=2)
        plt.plot(time_trimmed, el_vals, label='Elevation', color='darkorange', linewidth=2)
        plt.axvline(peak_time, linestyle='--', color='royalblue', alpha=0.5)
        plt.axvline(peak_time, linestyle='--', color='darkorange', alpha=0.5)
        plt.title(f'{stream_type} Stream – {label} [{peak_note} peak], p={p_display}, d={coh_d:.2f}')
        plt.xlabel('Time lag (s)')
        plt.ylabel('Amplitude (a.u.)')
        plt.legend()
        plt.tight_layout()
        filename = f'{stream_type}_{label}_az_vs_el.png'
        plt.savefig(os.path.join(comparison_dir, filename), dpi=300)
        plt.show()