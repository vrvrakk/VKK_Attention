import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# === Constants ===
sfreq = 125
time_lags = np.linspace(-0.1, 1.0, 139)
time_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
time_trimmed = time_lags[time_mask]
predictor_idx = 1  # Envelope predictor

# === Helper functions ===
def smooth_weights(w, window_len=11):
    h = np.hamming(window_len)
    h /= h.sum()
    return np.array([np.convolve(w[:, i], h, mode='same') for i in range(w.shape[1])]).T

def load_trfs(base_dir, plane, cond, folder, stream):
    weights_dir = base_dir / plane / cond / folder / "on_en_ov_RT" / "weights"
    files = sorted([f for f in weights_dir.iterdir() if stream in f.name and 'npy' in f.name])
    trfs = []
    for f in files:
        data = np.load(weights_dir / f, allow_pickle=True).squeeze().T
        smoothed = smooth_weights(data)
        trfs.append(smoothed[:, predictor_idx])
    return np.stack(trfs, axis=0)[:, time_mask]

# === Setup paths ===
default_path = Path.cwd()
base_dir = default_path / 'data/eeg/trf/trf_testing/results/single_sub'
stream = 'distractor_stream'
colors = {'a2': 'darkorange', 'e2': 'teal'}
cond_labels = {'a2': 'Left Target (A2)', 'e2': 'Top Target (E2)'}

# === Plotting ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

for i, folder in enumerate(['target_nums', 'deviants']):
    ax = axes[i]
    for cond in ['a2', 'e2']:
        plane = 'azimuth' if cond.startswith('a') else 'elevation'
        folder_path = base_dir / plane / cond / folder / "on_en_ov_RT" / "weights"
        if not folder_path.is_dir():
            print(f"Missing folder: {folder_path}")
            continue

        trfs = load_trfs(base_dir, plane, cond, folder, stream)
        mean = trfs.mean(axis=0)
        sem = trfs.std(axis=0) / np.sqrt(trfs.shape[0])
        ax.plot(time_trimmed, mean, label=cond_labels[cond], color=colors[cond])
        ax.fill_between(time_trimmed, mean - sem, mean + sem, alpha=0.3, color=colors[cond])

    ax.set_title(f'Distractor: {folder.replace("_", " ").title()}', fontsize=11)
    ax.set_xlabel('Time lag (s)')
    if i == 0:
        ax.set_ylabel('Amplitude (a.u.)')
    ax.legend(fontsize=9)
    ax.grid(True)

plt.suptitle('TRF Envelope: Distractor Stream Responses\nTarget_Nums vs Deviants (A2/E2)', fontsize=13)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
