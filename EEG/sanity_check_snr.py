import os
from pathlib import Path
import mne
import numpy as np
from scipy.signal import welch

# === PARAMETERS ===
results_path = Path("C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results")
condition = "a1"
snr_threshold = 1.0  # You can adjust this threshold
output_dir = results_path / 'snr_reports'
output_dir.mkdir(exist_ok=True)

# === STORAGE ===
high_snr = {}
low_snr = {}
high_files = []
low_files = []

# === PROCESSING ===
for sub_dir in results_path.iterdir():
    if sub_dir.is_dir() and sub_dir.name.startswith("sub"):
        sub = sub_dir.name
        eeg_path = sub_dir / "ica"
        snrs = []

        for sub_file in eeg_path.glob(f"*{condition}*.fif"):
            eeg_file = mne.io.read_raw_fif(sub_file, preload=True, verbose=False)
            data = eeg_file.get_data()
            fs = eeg_file.info['sfreq']

            f, psd = welch(data[0], fs=fs)
            signal_band = (f > 8) & (f < 13)
            noise_band = (f > 20) & (f < 40)

            snr = psd[signal_band].mean() / psd[noise_band].mean()
            snrs.append(snr)

            if snr >= snr_threshold:
                high_files.append(f"{sub_file} | SNR: {snr:.3f}")
            else:
                low_files.append(f"{sub_file} | SNR: {snr:.3f}")

        if snrs:
            mean_snr = np.mean(snrs)
            if mean_snr >= snr_threshold:
                high_snr[sub] = mean_snr
            else:
                low_snr[sub] = mean_snr

# === OUTPUT ===
print("\n--- High SNR Subjects ---")
for sub, snr in high_snr.items():
    print(f"{sub}: {snr:.3f}")

print("\n--- Low SNR Subjects ---")
for sub, snr in low_snr.items():
    print(f"{sub}: {snr:.3f}")

# === SAVE FILE LISTS ===
with open(output_dir / "high_snr_files.txt", "w") as f:
    f.write("\n".join(high_files))

with open(output_dir / "low_snr_files.txt", "w") as f:
    f.write("\n".join(low_files))

print(f"\nSaved file lists to: {output_dir}")
