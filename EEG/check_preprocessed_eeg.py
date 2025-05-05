import os
from pathlib import Path
import mne
import numpy as np

default_path = Path.cwd()
eeg_path = default_path / 'data/eeg/preprocessed/results'
condition = 'e1'

eeg_header_files = {}
for folders in eeg_path.iterdir():
    if condition in ['e1', 'e2'] and folders.name in ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub08']:
        continue
    else:
        sub_eeg = []
        for sub_folder in folders.iterdir():
            if 'ica' in sub_folder.name:
                for files in sub_folder.iterdir():
                    if files.is_file() and 'fif' in files.name:
                        if condition in files.name:
                            eeg = mne.io.read_raw_fif(files, preload=True)
                            sub_eeg.append(eeg)
    eeg_header_files[folders.name] = sub_eeg

sub = 'sub08'
selected_list = eeg_header_files[sub]

for files in selected_list:
    files.plot()


def compute_bad_channels(raw):
    data = raw.get_data() * 1e6  # to µV
    chan_std = data.std(axis=1)

    # Flatline check
    flat_chs = np.where(chan_std < 1.0)[0]

    # Robust Z-score for std
    median_std = np.median(chan_std)
    mad_std = np.median(np.abs(chan_std - median_std))
    robust_z = 0.6745 * (chan_std - median_std) / mad_std
    bad_by_deviation = np.where(np.abs(robust_z) > 5.0)[0]

    # HF noise ratio (>50Hz)
    if raw.info['sfreq'] > 100:
        raw_high = raw.copy().filter(l_freq=50., h_freq=None)
        hf_power = raw_high.get_data().std(axis=1) * 1e6
        noise_fraction = np.divide(hf_power, chan_std, out=np.zeros_like(hf_power), where=chan_std != 0)
        med_nf = np.median(noise_fraction)
        mad_nf = np.median(np.abs(noise_fraction - med_nf))
        robust_z_hf = 0.6745 * (noise_fraction - med_nf) / mad_nf
        bad_by_hf = np.where(robust_z_hf > 5.0)[0]
    else:
        bad_by_hf = np.array([], dtype=int)

    # Combine all bads
    bad_channels = np.unique(np.concatenate([flat_chs, bad_by_deviation, bad_by_hf])).astype(int)
    bad_names = [raw.ch_names[i] for i in bad_channels]
    return bad_names

n = 0
ica = mne.preprocessing.ICA(n_components=0.99, method='picard', random_state=99)
ica.fit(selected_list[n])
# ica.plot_components()
ica.plot_sources(selected_list[n])
ica.apply(selected_list[n])

# bad_channels = {sub: [[], [], [], [], []] for sub in eeg_header_files.keys()}

for i, files in enumerate(selected_list):
    block = []
    bads = compute_bad_channels(files)
    print(f"\n File {i}: {len(bads)} bad channels: {bads}")
    block.append(bads)
    bad_channels[sub][i] = bads
    files.interpolate_bads()


for i, files in enumerate(selected_list):
    save_path = eeg_path / sub / 'ica'
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f'{sub}_{condition}_{i}_ica-raw.fif'
    files.save(save_path/filename, overwrite=True)