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

weights_dir_list = []
for folder_type in folder_types:
    weights_dir = rf"C:/Users/pppar/PycharmProjects/VKK_Attention/data/eeg/trf/trf_testing/composite_model/single_sub/{plane}/{folder_type}/on_en_RT_ov/weights"
    weights_dir_list.append(weights_dir)

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
smoothed_targets = {}
smoothed_targets_all = {}
smoothed_distractors = {}
smoothed_distractors_all = {}
for folder_type, weights_dir in zip(folder_types, weights_dir_list):
    smoothed_target, smoothed_target_all = load_and_smooth_weights("target_stream")
    smoothed_targets[folder_type] = smoothed_target
    smoothed_targets_all[folder_type] = smoothed_target_all
    smoothed_distractor, smoothed_distractor_all = load_and_smooth_weights("distractor_stream")
    smoothed_distractors[folder_type] = smoothed_distractor
    smoothed_distractors_all[folder_type] = smoothed_distractor_all

# === Plot all target stream responses across folder_types ===
from scipy.stats import sem, ttest_rel
from statsmodels.stats.multitest import fdrcorrection

predictor_idx = 1  # envelope
time_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_plot = time_lags[time_mask]


def plot_stream_responses(stream, smoothed_data_all):
    if stream == 'distractor_stream':
        colors = {
            'non_targets': 'royalblue',
            'target_nums': 'firebrick',
            'deviants': 'darkorange'
        }
    else:
        colors = {'non_targets': 'royalblue',
                  'target_nums': 'firebrick'}

    ref = 'non_targets'  # baseline
    zscored_data = {}  # store z-scored data per condition

    # === Z-score all datasets over time per subject ===
    for key in colors:
        raw = smoothed_data_all[key][:, time_mask, predictor_idx]
        zscored = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, keepdims=True)
        zscored_data[key] = zscored

    # === Compute global y-limits based on all z-scored data ===
    all_vals = np.concatenate([zscored_data[key] for key in colors])
    y_min, y_max = np.min(all_vals), np.max(all_vals)
    y_margin = 0.15 * (y_max - y_min)

    plt.figure(figsize=(10, 5))

    for key in colors:
        data = zscored_data[key]
        mean = data.mean(axis=0)
        sem_vals = sem(data, axis=0)

        plt.plot(time_plot, mean, label=key, color=colors[key], linewidth=2)
        plt.fill_between(time_plot, mean - sem_vals, mean + sem_vals, color=colors[key], alpha=0.3)

        if key != ref:
            ref_data = zscored_data[ref]
            p_vals = np.array([ttest_rel(ref_data[:, i], data[:, i]).pvalue for i in range(data.shape[1])])
            _, p_fdr = fdrcorrection(p_vals)
            sig_mask = p_fdr < 0.05

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

                    plt.axvspan(start, end, color=colors[key], alpha=0.08)

                    center_time = (start + end) / 2
                    segment = np.concatenate([ref_data[:, start_idx:end_idx], data[:, start_idx:end_idx]])
                    local_y_max = np.max(segment)
                    local_y_min = np.min(segment)
                    y_text = local_y_max + 0.1 * (local_y_max - local_y_min)

                    min_p = p_fdr[start_idx:end_idx].min()
                    if min_p < 0.001:
                        label = '***'
                    elif min_p < 0.01:
                        label = '**'
                    elif min_p < 0.05:
                        label = '*'
                    else:
                        label = f"p={min_p:.3f}"

                    plt.text(center_time, y_text, label, ha='center', va='bottom', fontsize=10, color=colors[key])

    # Finalize and save
    plt.ylim(y_min - y_margin, y_max + y_margin)
    plt.axhline(0, color='gray', linestyle='--')
    plt.xlabel('Time lag (s)')
    plt.ylabel('TRF Amplitude (z-scored)')
    plt.title(f'{stream.replace('_', ' ').capitalize()} TRF: Stimulus Type Comparison (baseline = non-targets)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(weights_dir_list[0], '..', 'target_trf_vs_non_targets.png'), dpi=300)
    plt.show()

plot_stream_responses(stream='target_stream', smoothed_data_all=smoothed_targets_all)
plot_stream_responses(stream='distractor_stream', smoothed_data_all=smoothed_distractors_all)

# --- Plot all in one --- #
# Load data
smoothed_targets = {}
smoothed_targets_all = {}
smoothed_distractors = {}
smoothed_distractors_all = {}
for folder_type, weights_dir in zip(folder_types, weights_dir_list):
    smoothed_target, smoothed_target_all = load_and_smooth_weights("target_stream")
    smoothed_targets[folder_type] = smoothed_target
    smoothed_targets_all[folder_type] = smoothed_target_all
    smoothed_distractor, smoothed_distractor_all = load_and_smooth_weights("distractor_stream")
    smoothed_distractors[folder_type] = smoothed_distractor
    smoothed_distractors_all[folder_type] = smoothed_distractor_all

# === Plot all target stream responses across folder_types ===
from scipy.stats import sem, ttest_rel
from statsmodels.stats.multitest import fdrcorrection

predictor_idx = 1  # envelope
time_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_plot = time_lags[time_mask]
stream = 'distractor_stream'

if stream == 'distractor_stream':
    colors = {
        'non_targets': 'royalblue',
        'target_nums': 'firebrick',
        'deviants': 'darkorange'
    }
else:
    colors = {'non_targets': 'royalblue',
              'target_nums': 'firebrick'}

ref = 'non_targets'  # baseline
zscored_data = {}  # store z-scored data per condition

# === Z-score all datasets over time per subject ===
for key in colors:
    raw = smoothed_distractors_all[key][:, time_mask, predictor_idx]
    zscored = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, keepdims=True)
    zscored_data[key] = zscored

# === Compute global y-limits based on all z-scored data ===
all_vals = np.concatenate([zscored_data[key] for key in colors])
y_min, y_max = np.min(all_vals), np.max(all_vals)
y_margin = 0.15 * (y_max - y_min)

plt.figure(figsize=(10, 5))

for key in colors:
    data = zscored_data[key]
    mean = data.mean(axis=0)
    sem_vals = sem(data, axis=0)

    plt.plot(time_plot, mean, label=key, color=colors[key], linewidth=2)
    plt.fill_between(time_plot, mean - sem_vals, mean + sem_vals, color=colors[key], alpha=0.3)

    if key != ref:
        ref_data = zscored_data[ref]
        p_vals = np.array([ttest_rel(ref_data[:, i], data[:, i]).pvalue for i in range(data.shape[1])])
        _, p_fdr = fdrcorrection(p_vals)
        sig_mask = p_fdr < 0.05

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

                plt.axvspan(start, end, color=colors[key], alpha=0.08)

                center_time = (start + end) / 2
                segment = np.concatenate([ref_data[:, start_idx:end_idx], data[:, start_idx:end_idx]])
                local_y_max = np.max(segment)
                local_y_min = np.min(segment)
                y_text = local_y_max + 0.1 * (local_y_max - local_y_min)

                min_p = p_fdr[start_idx:end_idx].min()
                if min_p < 0.001:
                    label = '***'
                elif min_p < 0.01:
                    label = '**'
                elif min_p < 0.05:
                    label = '*'
                else:
                    label = f"p={min_p:.3f}"

                plt.text(center_time, y_text, label, ha='center', va='bottom', fontsize=10, color=colors[key])

# Finalize and save
plt.ylim(y_min - y_margin, y_max + y_margin)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Time lag (s)')
plt.ylabel('TRF Amplitude (z-scored)')
plt.title(f'{stream.replace('_', ' ').capitalize()} TRF: Stimulus Type Comparison (baseline = non-targets)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(weights_dir_list[0], '..', 'target_trf_vs_non_targets.png'), dpi=300)
plt.show()

# --- Plot all responses in one plot --- #

from scipy.stats import sem, ttest_rel
from statsmodels.stats.multitest import fdrcorrection

predictor_idx = 1  # envelope
time_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_plot = time_lags[time_mask]

# === Define keys and color map ===
conditions = {
    'target_stream': {
        'non_targets': '#1f77b4',   # Blue
        'target_nums': '#d62728'    # Red
    },
    'distractor_stream': {
        'non_targets': '#17becf',   # Teal
        'target_nums': '#ff7f0e',   # Orange
        'deviants': '#9467bd'       # Purple
    }
}

# === Initialize storage ===
zscored_data = {}
y_vals = []

# === Z-score and organize data from both target and distractor stream dicts ===
for stream, conds in conditions.items():
    for key, color in conds.items():
        label = f"{key} ({stream.split('_')[0]})"

        # Select correct source dictionary
        source = smoothed_targets_all if stream == 'target_stream' else smoothed_distractors_all
        raw = source[key][:, time_mask, predictor_idx]

        # Z-score per subject
        z = (raw - raw.mean(axis=1, keepdims=True)) / raw.std(axis=1, keepdims=True)
        zscored_data[label] = {
            'data': z,
            'color': color,
            'stream': stream,
            'base_key': key
        }
        y_vals.append(z)

# === Compute global y-limits ===
all_z = np.concatenate(y_vals)
y_min, y_max = np.min(all_z), np.max(all_z)
y_margin = 0.15 * (y_max - y_min)

plt.figure(figsize=(12, 6))

# === Plot each condition ===
for label, info in zscored_data.items():
    data = info['data']
    color = info['color']
    stream = info['stream']
    base_key = info['base_key']

    mean = data.mean(axis=0)
    sem_vals = sem(data, axis=0)

    plt.plot(time_plot, mean, label=label, color=color, linewidth=2)
    plt.fill_between(time_plot, mean - sem_vals, mean + sem_vals, color=color, alpha=0.25)

    # Significance vs non_targets baseline (within same stream)
    if base_key != 'non_targets':
        ref_label = f"non_targets ({stream.split('_')[0]})"
        ref_data = zscored_data[ref_label]['data']

        p_vals = np.array([ttest_rel(ref_data[:, i], data[:, i]).pvalue for i in range(data.shape[1])])
        _, p_fdr = fdrcorrection(p_vals)
        sig_mask = p_fdr < 0.05

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

                plt.axvspan(start, end, color=color, alpha=0.06)

                center_time = (start + end) / 2
                segment = np.concatenate([ref_data[:, start_idx:end_idx], data[:, start_idx:end_idx]])
                local_y_max = np.max(segment)
                local_y_min = np.min(segment)
                y_text = local_y_max + 0.1 * (local_y_max - local_y_min)

                min_p = p_fdr[start_idx:end_idx].min()
                if min_p < 0.001:
                    label_txt = '***'
                elif min_p < 0.01:
                    label_txt = '**'
                elif min_p < 0.05:
                    label_txt = '*'
                else:
                    label_txt = f"p={min_p:.3f}"

                plt.text(center_time, y_text, label_txt, ha='center', va='bottom', fontsize=10, color=color)

# === Finalize ===
plt.ylim(y_min - y_margin, y_max + y_margin)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Time lag (s)')
plt.ylabel('TRF Amplitude (z-scored)')
plt.title('TRF Comparison: Target and Distractor Streams')
plt.legend()
plt.tight_layout()

# === Save ===
plt.savefig(os.path.join(weights_dir_list[0], '..', 'all_trf_target_and_distractor.png'), dpi=300)
plt.show()
