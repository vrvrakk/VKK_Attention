import os
from pathlib import Path
import mne
from mtrf import TRF


import numpy
import os
import random
import pandas
import numpy as np
from mtrf.model import TRF
from mtrf.stats import pearsonr
import pickle
import json
import mne
import matplotlib.pyplot as plt
from scipy.signal import welch



"""
Using this script we are going to determine the roi, by checking the Pearson's r for each sensor. 
"""
def compute_snr(data, eeg):
    # Signal = variance of the mean signal across time (averaged across channels)
    fs = eeg.info['sfreq']

    f, psd = welch(data[0], fs=fs)
    signal_band = (f > 8) & (f < 13)
    noise_band = (f > 20) & (f < 40)
    snr = psd[signal_band].mean() / psd[noise_band].mean()
    print(f"SNR ratio: {snr}")
    return snr



# directories
condition = 'a1'
from TRF_test.TRF_test_config import azimuth_subs
default_path = Path.cwd()
predictors_path = default_path / 'data/eeg/predictors'
eeg_results_path = default_path / 'data/eeg/preprocessed/results'

eeg_files = {}
for folders in eeg_results_path.iterdir():
    if 'sub' in folders.name:
        sub_data = []
        for files in folders.iterdir():
            if 'ica' in files.name:
                for data in files.iterdir():
                    if condition in data.name:
                        eeg = mne.io.read_raw_fif(data, preload=True)
                        sub_data.append(eeg)
        eeg_files[folders.name] = sub_data

eeg_concat_list = {}
for sub, sub_list in eeg_files.items():
    eeg_concat = mne.concatenate_raws(sub_list)
    eeg_concat_list[sub] = eeg_concat


for eeg in eeg_concat_list.values():
    eeg.plot()
for sub, eeg in eeg_concat_list.items():
    eeg.interpolate_bads()
    eeg.resample(sfreq=125)
    res_path = eeg_results_path / 'concatenated_data' / 'ica' / sub / condition
    res_path.mkdir(parents=True, exist_ok=True)
    eeg.save(res_path / f'ica_concat_125Hz_{sub}_{condition}-raw.fif', overwrite=True)

def extract_bad_segments():
    all_bad_segments = {}
    for sub, eeg_files in eeg_concat_list.items():
        bad_segments = []
        for description, onset, duration in zip(eeg_files.annotations.description,
                                                eeg_files.annotations.onset,
                                                eeg_files.annotations.duration):
            if description.lower().startswith('bad'):
                offset = onset + duration
                bad_segments.append((onset, offset))
        all_bad_segments[sub] = bad_segments
    return all_bad_segments

eeg_lens = [eeg_file.n_times for eeg_file in eeg_concat_list.values()]
sfreq = 125
def set_bad_series(all_bad_segments):
    bad_series_all = {}
    for eeg_len, (sub, bads) in zip(eeg_lens, all_bad_segments.items()):
        bad_series = np.zeros(eeg_len)
        for block in bads:
            print(block)
            onset_samples = int(np.round(block[0] * sfreq))
            offset_samples = int(np.round(block[1] * sfreq))
            bad_series[onset_samples:offset_samples] = -999
        bad_series_all[sub] = bad_series
    return bad_series_all

stim_dur = 0.745
base_dir = r"C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/concatenated_data/ica"
def load_condition_eeg_files(base_dir, condition):
    eeg_concat_list = {}
    for subfolder in os.listdir(base_dir):
        sub_path = os.path.join(base_dir, subfolder)
        if os.path.isdir(sub_path) and subfolder.startswith("sub"):
            condition_path = os.path.join(sub_path, condition)
            if os.path.isdir(condition_path):
                for file in os.listdir(condition_path):
                    if file.endswith('.fif'):
                        file_path = os.path.join(condition_path, file)
                        raw = mne.io.read_raw_fif(file_path, preload=True)
                        eeg_concat_list[subfolder] = raw
                        break  # Only load the first .fif file found
    return eeg_concat_list

def save_bad_series(bad_series_all):
    for sub, series in bad_series_all.items():
        save_path = predictors_path / 'bad_segments' / sub / condition / 'concatenated'
        save_path.mkdir(parents=True, exist_ok=True)
        np.savez(save_path/f'{sub}_{condition}_bad_series_concat.npz',
                 bad_series=series,
                 sfreq=sfreq,
                 stim_duration_samples=int(stim_dur * sfreq),
                 stream_label=f'bad_series'
                 )

def apply_bad_mask(eeg_concat_list, bad_series_all):
    masked_eeg_data = {}

    for sub, raw in eeg_concat_list.items():
        eeg_data = raw.get_data()
        if sub in bad_series_all:
            bad_series = bad_series_all[sub]
            good_samples = bad_series == 0  # Boolean mask: True for good
            eeg_data_clean = eeg_data[:, good_samples]
            masked_eeg_data[sub] = eeg_data_clean
            print(f"Masked bad samples for {sub}. Clean shape: {eeg_data_clean.shape}")
        else:
            print(f"No bad_series found for {sub}, using full data.")
            masked_eeg_data[sub] = eeg_data

    return masked_eeg_data

envelope_predictor = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/predictors/envelopes')
stim1 = 'stream1'
stim2 = 'stream2'

for files in envelope_predictor.iterdir():
    if 'sub' in files.name:
        for file in files.iterdir():
            if condition in file.name:
                for stim_types in file.iterdir():
                    if stim1 in stim_types.name:
                        for array in stim_types.iterdir():
                            if 'concat' in array.name:
                                stream1 = np.load(array)
                    if stim2 in stim_types.name:
                        for array2 in stim_types.iterdir():
                            if 'concat' in array2.name:
                                stream2 = np.load(array2)


# bin edges
with open(f'{ANALYSIS_DIR}/variables/bin_edges.pkl', 'rb') as f:
    bin_edges = pickle.load(f)

# config file
with open(f'{ANALYSIS_DIR}/variables/config.json', 'r') as f:
    config = json.load(f)

subjects = [f for f in os.listdir(DATA_DIR) if 'sub' in f]

all_trials = []

for subject in subjects:
    subject_trials = [f for f in os.listdir(f'{DATA_DIR}/{subject}') if 'sub' in f]
    for trial in subject_trials:
        all_trials.append(f'{subject}/{trial}')

random.seed(42)
selected_trials = random.sample(all_trials, 72)

# Store all trials separately
stimulus = []
response = []

for idx, trial in enumerate(selected_trials):

    print('|||||||||||||||||||')
    print(f'Trial: {idx + 1} / {len(selected_trials)}')
    print(trial)
    print('|||||||||||||||||||')

    subject, stim = trial.split('/')

    data = pandas.read_csv(f'{DATA_DIR}/{subject}/{stim}', index_col=0)
    start_time = 2
    end_time = data.index[-1] - 1
    data = data.loc[start_time:end_time]

    # Extract EEG channels
    eeg_channels = [col for col in data.columns if col.startswith(('Fp', 'F', 'T', 'C', 'P', 'O', 'AF', 'PO')) and col not in ['Prob', 'FCz']]
    print(len(eeg_channels))

    # Extract relevant columns
    envelope = data['envelope'].values
    loc_change = data['loc_change'].values
    H = data['H'].values

    binned_H = numpy.digitize(H, bin_edges, right=True)

    H_df = pandas.DataFrame({'H': H, 'bin': binned_H})
    H_df['normalized_H'] = H_df.groupby('bin')['H'].transform(normalize_within_bin)
    H_df.loc[H_df['bin'] != 0, 'normalized_H'] = H_df['normalized_H'].replace(0, 0.001)

    # Prepare stimulus features
    temp = H_df.pivot(columns='bin', values='normalized_H').fillna(0)
    stim_H = temp.iloc[:, 1:].to_numpy()

    resp = data[eeg_channels].values

    # Ensure stim_H and resp are the same length
    min_length = min(stim_H.shape[0], resp.shape[0])
    stim_H = stim_H[:min_length, :]
    resp = resp[:min_length, :]
    print(stim_H.shape, resp.shape)

    # Pack stimulus and response for cross-validation
    stim_combined = numpy.column_stack([envelope[:min_length], loc_change[:min_length], stim_H])
    stimulus.append(stim_combined)
    response.append(resp)


# ----- TRF MODEL for optimization ---------

trf = TRF(direction=1)
trf.train(stimulus, response, config['fs'], config['tmin'], config['tmax'], config['lambda'])

pred, r = trf.predict(stimulus, response, average=False)

numpy.save(f'{ANALYSIS_DIR}/variables/r_scalp.npy', r)
numpy.save(f'{ANALYSIS_DIR}/variables/eeg_channels.npy', eeg_channels)

# --------- VISUALISE R values -----------

r = numpy.load(f'{ANALYSIS_DIR}/variables/r_scalp.npy')
eeg_channels = numpy.load(f'{ANALYSIS_DIR}/variables/eeg_channels.npy').tolist()

montage = mne.channels.make_standard_montage('brainproducts-RNP-BA-128')
info = mne.create_info(ch_names=eeg_channels, sfreq=config['fs'], ch_types='eeg')

evoked = mne.EvokedArray(r[:, numpy.newaxis], info)
evoked.set_montage(montage)


fig, ax = plt.subplots(figsize=(5, 5))
img, _ = mne.viz.plot_topomap(r, evoked.info, axes=ax, vlim=(0, 0.1), show=True)

# Create colorbar using the "mappable" object from plot_topomap
cbar = plt.colorbar(img, ax=ax, shrink=0.75, orientation='vertical')
cbar.set_label('Pearson r')

plt.show()

# Save the figure
fig.savefig(f'{DIR}/plots/r_scalp_topography.svg', dpi=800)

# --------------- SELECT ROI -----------------

channels_with_r = pandas.DataFrame({"eeg_channel": eeg_channels, "r_value": r})
channels_with_r_filtered = channels_with_r[
    (channels_with_r["r_value"] > 0.051) & ~channels_with_r["eeg_channel"].str.startswith("T")
]


# ---- Mark the ROI on viz -----

mask = numpy.zeros(len(evoked.ch_names), dtype=bool)

# Get the indices of the filtered channels in the original channel list
for channel in channels_with_r_filtered['eeg_channel']:
    mask[evoked.ch_names.index(channel)] = True

fig, ax = plt.subplots(figsize=(5, 5))
img, _ = mne.viz.plot_topomap(r, evoked.info, axes=ax, vlim=(0, 0.1),
                             show=True, mask=mask,
                             mask_params=dict(marker='o', markerfacecolor='black',
                                            markeredgecolor='black', linewidth=0,
                                            markersize=6))

# Create colorbar using the "mappable" object from plot_topomap
cbar = plt.colorbar(img, ax=ax, shrink=0.75, orientation='vertical')
cbar.set_label('Pearson r')

plt.show()

# Save the figure
fig.savefig(f'{DIR}/plots/r_scalp_topography_roi.svg', dpi=800)
fig.savefig(f'{DIR}/plots/r_scalp_topography_roi.png', dpi=800)


# Save the roi

channels_with_r_filtered.to_csv(f'{ANALYSIS_DIR}/variables/roi.csv')