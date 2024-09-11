'''
Define Detection Thresholds:

    Threshold Definition: Set thresholds to detect muscle activity.
    For example, define a detection threshold that is above the baseline noise level
    (mean + 2 standard deviations or RMS of baseline) but below the MVC level.
    This threshold helps to identify when muscle activity occurs during the task.

Calculate Partial Errors:

    Identify time windows during a task where muscle activation deviates from the baseline
    but does not reach the level of a full button press.
    Use the defined detection thresholds to quantify such deviations as "partial errors."
'''
# libraries:
import mne
from pathlib import Path
import os
import numpy as np
from scipy.signal import butter, filtfilt
from collections import Counter
import json
from meegkit import dss
from matplotlib import pyplot as plt, patches
from meegkit.dss import dss_line

from helper import grad_psd, snr

sub_input = input("Give sub number as subn (n for number): ")
sub = [sub.strip() for sub in sub_input.split(',')]
cm = 1 / 2.54
name = 'sub00'
# 0. LOAD THE DATA
sub_dirs = []
fig_paths = []
epochs_folders = []
evokeds_folders = []
results_paths = []
for subs in sub:
    default_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data')
    raw_dir = default_dir / 'eeg' / 'raw'
    sub_dir = raw_dir / subs
    sub_dirs.append(sub_dir)
    json_path = default_dir / 'misc'
    fig_path = default_dir / 'eeg' / 'preprocessed' / 'results' / subs / 'figures'
    fig_paths.append(fig_path)
    results_path = default_dir / 'eeg' / 'preprocessed' / 'results' / subs
    results_paths.append(results_path)
    epochs_folder = results_path / "epochs"
    epochs_folders.append(epochs_folder)
    evokeds_folder = results_path / 'evokeds'
    evokeds_folders.append(evokeds_folder)
    raw_fif = sub_dir / 'raw files'
    for folder in sub_dir, fig_path, results_path, epochs_folder, evokeds_folder, raw_fif:
        if not os.path.isdir(folder):
            os.makedirs(folder)

markers_dict = {
    's1_events': {'Stimulus/S 1': 1, 'Stimulus/S 2': 2, 'Stimulus/S 3': 3, 'Stimulus/S 4': 4, 'Stimulus/S 5': 5,
                  'Stimulus/S 6': 6, 'Stimulus/S 8': 8, 'Stimulus/S 9': 9},
    # stimulus 1 markers
    's2_events': {'Stimulus/S 72': 72, 'Stimulus/S 73': 73, 'Stimulus/S 65': 65, 'Stimulus/S 66': 66,
                  'Stimulus/S 69': 69, 'Stimulus/S 70': 70, 'Stimulus/S 68': 68,
                  'Stimulus/S 67': 67},  # stimulus 2 markers
    'response_events': {'Stimulus/S132': 132, 'Stimulus/S130': 130, 'Stimulus/S134': 134, 'Stimulus/S137': 137,
                        'Stimulus/S136': 136, 'Stimulus/S129': 129, 'Stimulus/S131': 131,
                        'Stimulus/S133': 133}}  # response markers
s1_events = markers_dict['s1_events']
s2_events = markers_dict['s2_events']  # stimulus 2 markers
response_events = markers_dict['response_events']  # response markers

# config files
with open(json_path / "preproc_config.json") as file:
    cfg = json.load(file)
with open(json_path / "electrode_names.json") as file:
    mapping = json.load(file)

# Run pre-processing steps:
''' 4 conditions:
    - a1: azimuth, s1 target
    - a2: azimuth, s2 target
    - e1: elevation, s1 target
    - e2: elevation, s2 target '''
condition = input('Please provide condition (exp. EEG): ')

### Concatenate block files to one raw file in raw_folder

target_header_files_list = []
for sub_dir in sub_dirs:
    header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
    filtered_files = [file for file in header_files if condition in file]
    if filtered_files:
        target_header_files_list.append(filtered_files)

# create custom montage:
standard_montage = mne.channels.make_standard_montage('standard_1020')
emg1 = 'EMG1'  # Channel 65
emg2 = 'EMG2'  # Channel 66
emg_ref = 'EMG_REF'  # Channel 67
emg_positions = {
    emg1: [0.1, -0.1, 0.0],  # Example approximate position
    emg2: [0.1, -0.12, 0.0],
    emg_ref: [0.1, -0.11, 0.0]}
# positions that are to be kept out of the 97 of the standard montage:
filtered_positions = {
    "Fp1": [-0.0294367, 0.0839171, -0.00699],
    "Fp2": [0.0298723, 0.0848959, -0.00708],
    "F7": [-0.0702629, 0.0424743, -0.01142],
    "F3": [-0.0502438, 0.0531112, 0.042192],
    "Fz": [0.0003122, 0.058512, 0.066462],
    "F4": [0.0518362, 0.0543048, 0.040814],
    "F8": [0.0730431, 0.0444217, -0.012],
    "FC5": [-0.0772149, 0.0186433, 0.02446],
    "FC1": [-0.0340619, 0.0260111, 0.079987],
    "FC2": [0.0347841, 0.0264379, 0.078808],
    "FC6": [0.0795341, 0.0199357, 0.024438],
    "T7": [-0.0841611, -0.0160187, -0.009346],
    "C3": [-0.0653581, -0.0116317, 0.064358],
    "Cz": [0.0004009, -0.009167, 0.100244],
    "C4": [0.0671179, -0.0109003, 0.06358],
    "T8": [0.0850799, -0.0150203, -0.00949],
    "TP9": [-0.0856192, -0.0465147, -0.045707],
    "CP5": [-0.0795922, -0.0465507, 0.030949],
    "CP1": [-0.0355131, -0.0472919, 0.091315],
    "CP2": [0.0383838, -0.0470731, 0.090695],
    "CP6": [0.0833218, -0.0461013, 0.031206],
    "TP10": [0.0861618, -0.0470353, -0.045869],
    "P7": [-0.0724343, -0.0734527, -0.002487],
    "P3": [-0.0530073, -0.0787878, 0.05594],
    "Pz": [0.0003247, -0.081115, 0.082615],
    "P4": [0.0556667, -0.0785602, 0.056561],
    "P8": [0.0730557, -0.0730683, -0.00254],
    "PO9": [-0.0549104, -0.0980448, -0.035465],
    "O1": [-0.0294134, -0.112449, 0.008839],
    "Oz": [0.0001076, -0.114892, 0.014657],
    "O2": [0.0298426, -0.112156, 0.0088],
    "PO10": [0.0549876, -0.0980911, -0.035541],
    "AF7": [-0.0548397, 0.0685722, -0.01059],
    "AF3": [-0.0337007, 0.0768371, 0.021227],
    "AF4": [0.0357123, 0.0777259, 0.021956],
    "AF8": [0.0557433, 0.0696568, -0.010755],
    "F5": [-0.0644658, 0.0480353, 0.016921],
    "F1": [-0.0274958, 0.0569311, 0.060342],
    "F2": [0.0295142, 0.0576019, 0.05954],
    "F6": [0.0679142, 0.0498297, 0.016367],
    "FT9": [-0.0840759, 0.0145673, -0.050429],
    "FT7": [-0.080775, 0.0141203, -0.011135],
    "FC3": [-0.0601819, 0.0227162, 0.055544],
    "FC4": [0.0622931, 0.0237228, 0.05563],
    "FT8": [0.0818151, 0.0154167, -0.01133],
    "FT10": [0.0841131, 0.0143647, -0.050538],
    "C5": [-0.0802801, -0.0137597, 0.02916],
    "C1": [-0.036158, -0.0099839, 0.089752],
    "C2": [0.037672, -0.0096241, 0.088412],
    "C6": [0.0834559, -0.0127763, 0.029208],
    "TP7": [-0.0848302, -0.0460217, -0.007056],
    "CP3": [-0.0635562, -0.0470088, 0.065624],
    "CPz": [0.0003858, -0.047318, 0.099432],
    "CP4": [0.0666118, -0.0466372, 0.06558],
    "TP8": [0.0855488, -0.0455453, -0.00713],
    "P5": [-0.0672723, -0.0762907, 0.028382],
    "P1": [-0.0286203, -0.0805249, 0.075436],
    "P2": [0.0319197, -0.0804871, 0.076716],
    "P6": [0.0678877, -0.0759043, 0.028091],
    "PO7": [-0.0548404, -0.0975279, 0.002792],
    "PO3": [-0.0365114, -0.1008529, 0.037167],
    "POz": [0.0002156, -0.102178, 0.050608],
    "PO4": [0.0367816, -0.1008491, 0.036397],
    "PO8": [0.0556666, -0.0976251, 0.00273],
    "EMG1": [0.1, -0.1, 0.0],
    "EMG2": [0.1, -0.12, 0.0],
    "EMG_REF": [0.1, -0.11, 0.0]
}
# combine standard montage with additional EMG montage:
ch_pos = {ch: pos for ch, pos in standard_montage.get_positions()['ch_pos'].items() if ch in filtered_positions}
# include additional EMG channels:
ch_pos.update(emg_positions)
custom_montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
info = mne.create_info(
    ch_names=list(ch_pos.keys()),
    sfreq=500,
    ch_types=['eeg'] * 64 + ['emg'] * 3)
info.set_channel_types({'EMG1': 'emg', 'EMG2': 'emg', 'EMG_REF': 'emg'})
info.set_montage(custom_montage)
# read EEG signals:
raw_files = []
for sub_dir, header_files in zip(sub_dirs, target_header_files_list):
    for header_file in header_files:
        full_path = os.path.join(sub_dir, header_file)
        print(full_path)
        raw_files.append(mne.io.read_raw_brainvision(full_path, preload=True))
raw = mne.concatenate_raws(raw_files)  # merge all files of one condition into one
# use montage for specifying electrode positions:
raw.rename_channels(mapping)
raw.set_montage(custom_montage)
# save
raw.save(raw_fif / f"{name}_{condition}_EMG_raw.fif", overwrite=True)  # here the data is saved as raw
print(f'{condition} EMG raw data saved. If raw is empty, make sure axis and condition are filled in correctly.')

# create epochs around button presses events:
events = mne.events_from_annotations(raw)[0]  # get events from annotations attribute of raw variable
events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
filtered_events = [event for event in events if event[2] in response_events.values()]
events = np.array(filtered_events)

# get epochs
tmin = -0.4
tmax = 0.1
epoch_parameters = [tmin, tmax, response_events]
tmin, tmax, event_ids = epoch_parameters
event_ids = {key: val for key, val in event_ids.items() if val in events[:, 2]}
target_epochs = mne.Epochs(raw,
                           events,
                           event_id=event_ids,
                           tmin=tmin,
                           tmax=tmax,
                           baseline=(None, 0),
                           detrend=0,
                           preload=True)
target_epochs.pick_channels(['EMG1', 'EMG2'])
target_epochs.plot()

# High-pass filter at 20 Hz to remove low-frequency artifacts
target_epochs.filter(l_freq=20, h_freq=None)

# Low-pass filter at 450 Hz to remove high-frequency noise
target_epochs.filter(l_freq=None, h_freq=150)

# Notch filter at 50 Hz to remove power line noise
from meegkit.dss import dss_line
print('Remove power line noise and apply minimum-phase highpass filter')  # Cheveigné, 2020
X = target_epochs.get_data().T
X, _ = dss_line(X, fline=50, sfreq=target_epochs.info["sfreq"], nremove=2)
target_epochs._data = X.T
del X

baseline_files = []
# load baseline file:
for file in sub_dir.iterdir():  # This iterates through each file in the directory
    # Check if the current item is a file and contains 'baseline' in its name
    if file.is_file() and 'baseline' in file.name: # file.is_file(): Ensures you're only working with files, not directories.
        # Add the file to the list if it meets the criteria
        baseline_files.append(file)
# maximum voluntary contraction (mvc):
mvc_files = []
for file in sub_dir.iterdir():
    if file.is_file() and 'mvc' in file.name:
        mvc_files.append(file)

# Filter files to get only '.vhdr' files, which are BrainVision header files
header_baseline = [file for file in baseline_files if file.suffix == '.vhdr']
# same for mvc file:
header_mvc = [file for file in mvc_files if file.suffix == '.vhdr']

# process both baseline and header files the same way: notch, bp filter
for header_file in header_baseline:
    # Read the BrainVision file using the .vhdr header
    baseline_raw = mne.io.read_raw_brainvision(header_file, preload=True)
    baseline_raw.rename_channels(mapping)
    baseline_raw.set_montage(custom_montage)
    baseline_raw.pick_channels(['EMG1', 'EMG2'])
    X = baseline_raw.get_data().T
    X, _ = dss_line(X, fline=50, sfreq=baseline_raw.info["sfreq"], nremove=2)
    baseline_raw._data = X.T
    del X
    baseline_raw.filter(l_freq=20, h_freq=150)
    baseline_events = mne.make_fixed_length_events(baseline_raw, duration=0.5)  # fixed-length epochs
    baseline_epochs = mne.Epochs(baseline_raw, baseline_events, tmin=-0.4, tmax=0.1, preload=True, baseline=None)
# get average signal of baseline epochs:
baseline_avg = baseline_epochs.average()
# extract baseline average data:
baseline_data_avg = baseline_avg.data  # Shape: (n_channels, n_times)
# apply baseline on EMG epochs:
signal_epochs = target_epochs.subtract_evoked(baseline_avg)

# same process for mvc: notch, bp filter
for header_file in header_mvc:
    mvc_raw = mne.io.read_raw_brainvision(header_file, preload=True)
    mvc_raw.rename_channels(mapping)
    mvc_raw.set_montage(custom_montage)
    mvc_raw.pick_channels(['EMG1', 'EMG2'])
    X = mvc_raw.get_data().T
    X, _ = dss_line(X, fline=50, sfreq=mvc_raw.info["sfreq"], nremove=2)
    mvc_raw._data = X.T
    del X
    mvc_raw.filter(l_freq=20, h_freq=150)
    mvc_events = mne.make_fixed_length_events(mvc_raw, duration=0.5)
    mvc_epochs = mne.Epochs(mvc_raw, mvc_events, tmin=-0.4, tmax=0.1, preload=True, baseline=None)

# function for smoothing data: convolving data
def smooth_data(data, window_size=10):
    smoothed_data = np.zeros_like(data)  # Initialize an array with the same shape as the input data
    for epoch_idx in range(data.shape[0]):  # Loop over each epoch
        for ch_idx in range(data.shape[1]):  # Loop over each channel
            smoothed_data[epoch_idx, ch_idx, :] = np.convolve(
                data[epoch_idx, ch_idx, :], np.ones(window_size) / window_size, mode='same'
            )
    return smoothed_data

# rectify EMG signal:
rectified_signal_epochs = signal_epochs.copy()
rectified_signal_epochs._data = np.abs(rectified_signal_epochs._data)

# apply smoothing on EMG data:
smoothed_signal_epochs = rectified_signal_epochs.copy()
smoothed_signal_epochs._data = smooth_data(rectified_signal_epochs._data)

# Calculate robust MVC max values
mvc_data = np.abs(mvc_epochs.get_data())  # Rectify MVC data
mvc_smoothed = smooth_data(mvc_data)  # Smoothing with a moving average

# Calculate robust MVC max values using the smoothed data
mvc_max_per_epoch = np.percentile(mvc_smoothed, 95, axis=2)  # 95th percentile across time points
mvc_max = np.mean(mvc_max_per_epoch, axis=0)  # Mean of robust maxima per channel
mvc_threshold = mvc_max * 1.1  # more dynamic threshold

# normalize EMG signal with MVC:
norm_signal_epochs = smoothed_signal_epochs.copy()
for ch_idx, ch_name in enumerate(['EMG1', 'EMG2']):
    # Normalize data for each channel by its MVC
    norm_signal_epochs._data[:, ch_idx, :] /= mvc_max[ch_idx]
    # Cap values exceeding the MVC threshold
    norm_signal_epochs._data[:, ch_idx, :] = np.clip(norm_signal_epochs._data[:, ch_idx, :], 0, mvc_threshold[ch_idx] / mvc_max[ch_idx])
    # Clip the data at 100% MVC to remove extreme outliers

norm_signal_epochs.plot()  # looks fine
norm_signal_epochs._data = np.clip(norm_signal_epochs._data, 0, 1)

# check normalized epochs data:
normalized_data = norm_signal_epochs.get_data()
for ch_idx, ch_name in enumerate(['EMG1', 'EMG2']):
    print(f"\nChannel: {ch_name}")
    print("Min:", normalized_data[:, ch_idx, :].min())
    print("Max:", normalized_data[:, ch_idx, :].max())
    print("Mean:", normalized_data[:, ch_idx, :].mean())
    print("Std Dev:", normalized_data[:, ch_idx, :].std())

# Extract normalized data for plotting
normalized_data = norm_signal_epochs._data  # Access the normalized data directly
times = norm_signal_epochs.times
# Create a plot for each channel
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Plot normalized data for each epoch and each channel
for ch_idx, ch_name in enumerate(['EMG1', 'EMG2']):
    for epoch in normalized_data:
        ax[ch_idx].plot(times, epoch[ch_idx, :], alpha=0.5)
        ax[ch_idx].set_title(f'Normalized Data: {ch_name}')
        ax[ch_idx].set_ylim(0, 1.2)  # Adjust y-limits to reflect the normalized range (0-1)
        ax[ch_idx].set_ylabel('Normalized Units (0-1)')
        ax[ch_idx].set_xlabel('Time (samples)')

plt.tight_layout()
plt.show()


# mean epochs normalized:
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
for ch_index, ch_name in enumerate(['EMG1', 'EMG2']):
    mean_epoch = np.mean(normalized_data[:, ch_index, :], axis=0)
    ax[ch_index].plot(times, mean_epoch, alpha=0.5)
    ax[ch_index].set_title(f'Average Normalized Data: {ch_name}')
    ax[ch_index].set_ylim(0, 1.2)
    ax[ch_index].set_ylabel('Normalized Units (0-1)')
    ax[ch_index].set_xlabel('Epoch Window (s)')
plt.tight_layout()
plt.show()


# mean absolute value:
mav_per_epoch = np.mean(normalized_data, axis=2)

# RMS:
rms_per_epoch = np.sqrt(np.mean(normalized_data ** 2, axis=2))
# Compute Integrated EMG (iEMG) for each epoch
iemg_per_epoch = np.sum(normalized_data, axis=2)

from mne.time_frequency import tfr_morlet

# Define frequencies of interest (e.g., 10 to 150 Hz)
frequencies = np.arange(10, 150, 10)
n_cycles = frequencies / 20  # Number of cycles in each frequency band

# Perform time-frequency decomposition using Morlet wavelets
power = tfr_morlet(norm_signal_epochs, freqs=frequencies, n_cycles=n_cycles, use_fft=True, return_itc=False, decim=3)

# Plot average power across epochs for selected channels
for ch_index, ch_name in enumerate(['EMG1', 'EMG2']):
    power.plot(picks=[f'{ch_name}'], mode='logratio', baseline=(None, 0), title=f"Time-Frequency Analysis of {ch_name}")

# extract features
features = np.column_stack((mav_per_epoch, rms_per_epoch, iemg_per_epoch))

# The feature matrix will be of shape (n_epochs, n_features), ready for machine learning
print(features.shape)


# change in power over time: TF analysis
# Define the frequency band of interest (e.g., 20-40 Hz)
freq_band = (20, 40)

# Get the indices for the frequency band
freq_idx = np.where((power.freqs >= freq_band[0]) & (power.freqs <= freq_band[1]))[0]

# Define the time window of interest (pre-button press, e.g., -0.2 to 0 seconds)
pre_button_press = (-0.4, 0)

# Get the indices for the time window
time_idx_pre = np.where((power.times >= pre_button_press[0]) & (power.times <= pre_button_press[1]))[0]
# Define channels of interest (e.g., 'EMG1' and 'EMG2')
channels_of_interest = ['EMG1', 'EMG2']

# Get the indices for the channels of interest
channel_indices = [power.ch_names.index(ch) for ch in channels_of_interest]

# Compute average power in the frequency band for the pre-button press window for the selected channels
avg_power_pre = np.mean(power.data[channel_indices, :, :][:, freq_idx, :][:, :, time_idx_pre], axis=(1, 2))

# Print the average power in the pre-button press period for each channel
print("Average Power in Pre-Button Press Period:")
print(avg_power_pre)

# stats
# t_stat, p_val = ttest_ind(rms_per_epoch[condition_1], rms_per_epoch[condition_2])
# print(f"T-statistic: {t_stat}, p-value: {p_val}")