import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
from scipy.stats import sem, ttest_rel
from statsmodels.stats.multitest import fdrcorrection

# === Configuration ===
folder_type = 'all_stims'  # Choose one: 'all_stims', 'non_targets', 'target_nums', 'deviants'
planes = ['azimuth', 'elevation']
subject_ids = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub08']
predictor_idx = 1  # Envelope
sfreq = 125
time_lags = np.linspace(-0.1, 1.0, 139)
time_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_plot = time_lags[time_mask]
window_len = 11  # Hamming window length

# === Paths ===
base_path = "C:/Users/pppar/PycharmProjects/VKK_Attention/data/eeg/trf/trf_testing/composite_model/single_sub"
colors = {'azimuth': 'mediumseagreen', 'elevation': 'steelblue'}

# === Helper Functions ===
def smooth_weights(weights):
    hamming_win = np.hamming(window_len)
    hamming_win /= hamming_win.sum()
    return np.array([
        np.convolve(weights[:, i], hamming_win, mode='same')
        for i in range(weights.shape[1])
    ]).T

def load_and_smooth_weights(plane, stream_type):
    weights_dir = os.path.join(base_path, plane, folder_type, "on_en_RT_ov", "weights")
    files = sorted([f for f in os.listdir(weights_dir) if stream_type in f and 'weights' in f])
    smoothed_all = []
    for f in files:
        if any(sub in f for sub in subject_ids):
            continue
        w = np.load(os.path.join(weights_dir, f), allow_pickle=True).squeeze().T
        smoothed = smooth_weights(w)
        smoothed_all.append(smoothed)
    return np.stack(smoothed_all, axis=0)  # (n_subjects, 139, n_predictors)

# === Load and prepare data for both planes ===
zscored_data = {}
all_vals = []

for plane in planes:
    data = load_and_smooth_weights(plane, stream_type="distractor_stream")
    data = data[:, time_mask, predictor_idx]
    z = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
    zscored_data[plane] = z
    all_vals.append(z)

# === Global y-limits ===
all_vals = np.concatenate(all_vals)
y_min, y_max = np.min(all_vals), np.max(all_vals)
y_margin = 0.15 * (y_max - y_min)

# === Plot ===
plt.figure(figsize=(10, 5))

for plane in planes:
    data = zscored_data[plane]
    mean = data.mean(axis=0)
    sem_vals = sem(data, axis=0)
    plt.plot(time_plot, mean, label=plane, color=colors[plane], linewidth=2)
    plt.fill_between(time_plot, mean - sem_vals, mean + sem_vals, color=colors[plane], alpha=0.3)

# === Paired t-test and significance shading ===
p_vals = np.array([ttest_rel(zscored_data['azimuth'][:, i], zscored_data['elevation'][:, i]).pvalue for i in range(zscored_data['azimuth'].shape[1])])
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
        plt.axvspan(start, end, color='gray', alpha=0.1)
        center_time = (start + end) / 2
        segment = np.concatenate([zscored_data['azimuth'][:, start_idx:end_idx], zscored_data['elevation'][:, start_idx:end_idx]])
        y_text = segment.max() + 0.1 * (segment.max() - segment.min())
        min_p = p_fdr[start_idx:end_idx].min()
        label = '***' if min_p < 0.001 else '**' if min_p < 0.01 else '*' if min_p < 0.05 else f"p={min_p:.3f}"
        plt.text(center_time, y_text, label, ha='center', va='bottom', fontsize=10, color='gray')

# === Finalize ===
plt.ylim(y_min - y_margin, y_max + y_margin)
plt.axhline(0, color='gray', linestyle='--')
plt.xlabel('Time lag (s)')
plt.ylabel('TRF Amplitude (z-scored)')
plt.title(f"TRF Comparison Across Planes for {folder_type.replace('_', ' ').capitalize()}")
plt.legend()
plt.tight_layout()

# === Save ===
save_path = os.path.join(base_path, 'comparison_across_planes')
os.makedirs(save_path, exist_ok=True)
plt.savefig(os.path.join(save_path, f"plane_comparison_{folder_type}.png"), dpi=300)
plt.show()

# --- Plot and compare all responses from both planes --- #

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem, ttest_rel
from statsmodels.stats.multitest import fdrcorrection

# === Configuration ===
folder_types = ['all_stims', 'non_targets', 'target_nums', 'deviants']
planes = ['azimuth', 'elevation']
streams = ['target_stream', 'distractor_stream']
predictor_idx = 1  # envelope
sfreq = 125
time_lags = np.linspace(-0.1, 1.0, 139)
time_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_plot = time_lags[time_mask]
subject_ids = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub08']
window_len = 11
base_path = "C:/Users/pppar/PycharmProjects/VKK_Attention/data/eeg/trf/trf_testing/composite_model/single_sub"

# === Helper: Smooth with Hamming window ===
def smooth_weights(weights):
    hamming_win = np.hamming(window_len)
    hamming_win /= hamming_win.sum()
    return np.array([
        np.convolve(weights[:, i], hamming_win, mode='same')
        for i in range(weights.shape[1])
    ]).T

# === Load, smooth, and stack weights ===
def load_smoothed_weights(plane, folder_type, stream_type):
    weights_dir = os.path.join(base_path, plane, folder_type, 'on_en_RT_ov', 'weights')
    files = sorted([f for f in os.listdir(weights_dir) if stream_type in f and 'weights' in f])
    smoothed_all = []
    for f in files:
        if any(sub in f for sub in subject_ids):
            continue
        w = np.load(os.path.join(weights_dir, f), allow_pickle=True).squeeze().T
        smoothed = smooth_weights(w)
        smoothed_all.append(smoothed)
    return np.stack(smoothed_all, axis=0)

# === Organize data ===
data_dict = {}
for stream in streams:
    for folder in folder_types:
        key = f"{stream}_{folder}"
        data_dict[key] = {}
        for plane in planes:
            smoothed = load_smoothed_weights(plane, folder, stream)
            data = smoothed[:, time_mask, predictor_idx]
            z = (data - data.mean(axis=1, keepdims=True)) / data.std(axis=1, keepdims=True)
            data_dict[key][plane] = z

# === Plot ===
import matplotlib.cm as cm
cmap = cm.get_cmap('tab10')
colors = {plane: cmap(i) for i, plane in enumerate(planes)}

fig, axs = plt.subplots(len(folder_types), len(streams), figsize=(14, 10), sharex=True, sharey=True)

for i, folder in enumerate(folder_types):
    for j, stream in enumerate(streams):
        key = f"{stream}_{folder}"
        ax = axs[i, j]
        for plane in planes:
            z = data_dict[key][plane]
            mean = z.mean(axis=0)
            sem_vals = sem(z, axis=0)
            ax.plot(time_plot, mean, label=f"{plane}", color=colors[plane], linewidth=2)
            ax.fill_between(time_plot, mean - sem_vals, mean + sem_vals, color=colors[plane], alpha=0.25)

        # Statistical comparison
        az = data_dict[key]['azimuth']
        el = data_dict[key]['elevation']
        p_vals = np.array([ttest_rel(az[:, i], el[:, i]).pvalue for i in range(az.shape[1])])
        _, p_fdr = fdrcorrection(p_vals)
        sig_mask = p_fdr < 0.05

        in_sig = False
        for k in range(len(sig_mask)):
            if sig_mask[k] and not in_sig:
                in_sig = True
                start_idx = k
                start = time_plot[k]
            elif not sig_mask[k] and in_sig:
                in_sig = False
                end_idx = k
                end = time_plot[k]
                ax.axvspan(start, end, color='gray', alpha=0.1)

        ax.axhline(0, color='gray', linestyle='--', linewidth=1)
        ax.set_title(f"{stream.replace('_stream','')} - {folder}")
        if i == len(folder_types) - 1:
            ax.set_xlabel('Time lag (s)')
        if j == 0:
            ax.set_ylabel('TRF (z-scored)')

handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper right')
fig.suptitle("TRF Comparison: Azimuth vs Elevation per Stimulus Type and Stream", fontsize=14)
fig.tight_layout(rect=[0, 0, 1, 0.97])

output_path = os.path.join(base_path, "trf_plane_comparison_all_stims.png")
plt.savefig(output_path, dpi=300)
plt.show()