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
from copy import deepcopy
import mne
from pathlib import Path
import os
import numpy as np
from scipy.stats import mode
from scipy.signal import butter, filtfilt
from collections import Counter
import json
from meegkit import dss
from matplotlib import pyplot as plt, patches
from meegkit.dss import dss_line
import pandas as pd

from helper import grad_psd, snr

sub_input = input("Give sub number as subn (n for number): ")
sub = [sub.strip() for sub in sub_input.split(',')]
cm = 1 / 2.54
# 0. LOAD THE DATA
sub_dirs = []
fig_paths = []
epochs_folders = []
evokeds_folders = []
results_paths = []
for subs in sub:
    default_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data')
    data_dir = default_dir / 'eeg' / 'raw'
    sub_dir = data_dir / subs
    emg_dir = default_dir / 'emg' / subs
    sub_dirs.append(sub_dir)
    json_path = default_dir / 'misc'
    fig_path = emg_dir / 'preprocessed' / 'figures'
    fig_paths.append(fig_path)
    results_path = emg_dir / 'preprocessed' / 'results'
    results_paths.append(results_path)
    epochs_folder = results_path / "epochs"
    epochs_folders.append(epochs_folder)
    evokeds_folder = results_path / 'evokeds'
    evokeds_folders.append(evokeds_folder)
    raw_fif = emg_dir / 'raw files'
    for folder in sub_dir, fig_path, results_path, epochs_folder, evokeds_folder, raw_fif:
        if not os.path.isdir(folder):
            os.makedirs(folder)

# load all necessary json files, that contain configuration we need for pre-processing: electrode names, filter bandwidth, EEG events, etc.
with open(json_path / "preproc_config.json") as file:
    cfg = json.load(file)
with open(json_path / "electrode_names.json") as file:
    mapping = json.load(file)
with open(json_path / 'eeg_events.json') as file:
    markers_dict = json.load(file)

# here we specify which markers correspond to which stimulus' events, from the EEG events dictionary 'markers_dict'
s1_events = markers_dict['s1_events']
s2_events = markers_dict['s2_events']  # stimulus 2 markers
response_events = markers_dict['response_events']  # response markers
# mapping necessary for later on:
s1_mapping ={
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    5: 5,
    6: 6,
    7: 7,
    8: 8,
    9: 9
}

s2_mapping = {
    1: 65,
    2: 66,
    3: 67,
    4: 68,
    5: 69,
    6: 70,
    7: 71,
    8: 72,
    9: 73
}
response_mapping = {
    1: 129,
    2: 130,
    3: 131,
    4: 132,
    5: 133,
    6: 134,
    7: 135,
    8: 136,
    9: 137
}


''' 4 conditions:
    - a1: azimuth, s1 target
    - a2: azimuth, s2 target
    - e1: elevation, s1 target
    - e2: elevation, s2 target '''

condition = input('Please provide condition (exp. EEG): ')  # choose a condition of interest for processing

### Concatenate block files to one raw file in raw_folder
target_header_files_list = []
for sub_dir in sub_dirs:
    header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
    filtered_files = [file for file in header_files if condition in file]
    if filtered_files:
        target_header_files_list.append(filtered_files)

# load and read EEG signals:
index = int(input('Choose file to load (0-4): '))
emg_file = target_header_files_list[0][index]
full_path = os.path.join(sub_dir, emg_file)
emg = mne.io.read_raw_brainvision(full_path, preload=True)

# set montage for file:
emg.rename_channels(mapping)
emg.set_montage('standard_1020')
emg = emg.pick_channels(['A2', 'M2', 'A1'])  # arbitrary channels names I used for EMG1, EMG2 and EMG Reference respectively
emg.set_eeg_reference(ref_channels=['A1'])   # reference electrode flattens
#This is normal and expected behavior when you re-reference to a single electrode.
# The reference electrode will no longer carry any meaningful signal because you’ve defined it as the point of reference,
# effectively "zeroing" its signal.
emg.drop_channels(['A1'])
# save
# emg.save(raw_fif / f"{sub_input}_{condition}_EMG_raw.fif", overwrite=True)  # here the data is saved as raw
# print(f'{condition} EMG raw data saved. If raw is empty, make sure axis and condition are filled in correctly.')

# create epochs around target stimuli events: how?
# 1. import csv file with information:
csv_path = default_dir / 'params' / 'block_sequences' / f'{sub_input}.csv'
# read csv path
csv = pd.read_csv(csv_path)  # delimiter=';', header=None


def get_target_blocks():
    # specify axis:
    if condition in ['a1', 'a2']:
        axis = 'azimuth'
    elif condition in ['e1', 'e2']:
        axis = 'elevation'
    else:
        print('No valid condition defined.')
    # define target stream: s1 or s2?
    # we know this from the condition
    if condition in ['a1', 'e1']:
        target_stream = 's1'
        target_mapping = s1_mapping
        distractor_mapping = s2_mapping
    elif condition in ['a2', 'e2']:
        target_stream = 's2'
        target_mapping = s2_mapping
        distractor_mapping = s1_mapping
    else:
        raise ValueError("Invalid condition provided.")  # Handle unexpected conditions
    target_blocks = []
    # iterate through the values of the csv path; first enumerate to get the indices of each row
    for index, items in enumerate(csv.values):
        block_seq = items[0]  # from first column, save info as variable block_seq
        block_condition = items[1]  # from second column, save info as var block_condition
        if block_seq == target_stream and block_condition == axis:
            block = csv.iloc[index]
            target_blocks.append(block)  # Append relevant rows
    target_blocks = pd.DataFrame(target_blocks).reset_index(drop=True)  # convert condition list to a dataframe
    return target_stream, target_blocks, axis, target_mapping, distractor_mapping

target_stream, target_blocks, axis, target_mapping, distractor_mapping = get_target_blocks()
# choose target block of interest:
target_block = [target_block for target_block in target_blocks.values[index]]
def define_events(target_stream):
    if target_stream in ['s1']:
        target_events = s1_events
    elif target_stream in ['s2']:
        target_events = s2_events
    else:
        print('No valid target stream defined.')
    if target_events == s1_events:
        distractor_events = s2_events
    elif target_events == s2_events:
        distractor_events = s1_events
    return target_events, distractor_events

target_events, distractor_events = define_events(target_stream)

def create_events(chosen_events, events_mapping):
    target_number = target_block[3]
    events = mne.events_from_annotations(emg)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in chosen_events.values()]
    event_value = [val for key, val in events_mapping.items() if key == target_number]
    event_value = event_value[0]
    final_events = [event for event in filtered_events if event[2] == event_value]
    return final_events

# get target, distractor and response events:
targets_emg_events = create_events(target_events, target_mapping)
distractors_emg_events = create_events(distractor_events, distractor_mapping)
responses_emg_events = create_events(response_events, response_mapping)


# band-pass filter 20-150 Hz
# Remove low-frequency noise: Eliminates motion artifacts or baseline drift that occur below 20 Hz.
# Remove high-frequency noise: Filters out high-frequency noise (e.g., electrical noise or other non-EMG signals) above 150-450 Hz,
# which isn't part of typical muscle activity.
def filter_emg(emg):
    emg_filt = emg.copy().filter(l_freq=20, h_freq=150)
    '''The typical frequency range for EMG signals from the Flexor Digitorum Profundus (FDP) muscle is around 20 to 450 Hz. 
    Most muscle activity is concentrated in the 50-150 Hz range, but high-frequency components can go up to 450 Hz.'''
    # Notch filter at 50 Hz to remove power line noise
    from meegkit.dss import dss_line
    print('Remove power line noise and apply minimum-phase highpass filter')  # Cheveigné, 2020
    X = emg_filt.get_data().T
    X, _ = dss_line(X, fline=50, sfreq=emg_filt.info["sfreq"], nremove=2)
    emg_filt._data = X.T
    del X
    emg_filt.append(emg_filt)
    return emg_filt


emg_filt = filter_emg(emg)
# rectify the EMG data:
def rectify(emg_filt):
    emg_rectified = emg_filt.copy()
    emg_rectified._data = np.abs(emg_rectified._data)
    return emg_rectified

emg_rectified = rectify(emg_filt)
'''Smoothing the rectified EMG signal helps reduce high-frequency noise and makes the muscle activation patterns more visible and interpretable. 
It helps to create an EMG envelope, showing overall trends in muscle activity rather than every small fluctuation.
Noise reduction: It minimizes random fluctuations in the signal, making it easier to identify meaningful muscle activity.
Highlight activation: Smoothing reveals broader trends in muscle activity, which is useful for detecting partial errors 
or pre-activation in EMG signals.'''

# apply smoothing on EMG data:
def smoothing(emg_rectified):
    emg_smoothed = emg_rectified.copy().filter(l_freq=None, h_freq=5)
    emg_smoothed._data *= 1e6  # Convert from μV to mV for better scale
    return emg_smoothed

target_emg_smoothed = smoothing(emg_rectified)
distractor_emg_smoothed = smoothing(emg_rectified)
response_emg_smoothed = smoothing(emg_rectified)

# get baseline and motor recording:
def get_helper_recoring(recording):
    for files in sub_dir.iterdir():
        if files.is_file and recording in files.name and files.suffix == '.vhdr':
            helper = files
    helper = mne.io.read_raw_brainvision(helper, preload=True)
    helper.rename_channels(mapping)
    helper.set_montage('standard_1020')
    helper = helper.pick_channels(['A2', 'M2', 'A1'])
    helper.set_eeg_reference(ref_channels=['A1'])
    helper.drop_channels(['A1'])
    helper_filt = helper.copy().filter(l_freq=20, h_freq=150)
    from meegkit.dss import dss_line
    print('Remove power line noise and apply minimum-phase highpass filter')  # Cheveigné, 2020
    X = helper_filt.get_data().T
    X, _ = dss_line(X, fline=50, sfreq=helper_filt.info["sfreq"], nremove=2)
    helper_filt._data = X.T
    del X

    helper_rectified = helper.copy()
    helper_rectified._data = np.abs(helper_rectified._data)
    helper_smoothed = helper_rectified.copy().filter(l_freq=None, h_freq=5)
    helper_smoothed._data *= 1e6
    # create helper epochs manually:
    helper_n_samples = len(helper_smoothed.times)
    return helper_smoothed, helper_n_samples

baseline_smoothed, baseline_n_samples = get_helper_recoring(recording='baseline')
motor_smoothed, motor_n_samples = get_helper_recoring(recording='motor')
    # Create dummy events every 0.8 seconds
def baseline_epochs(baseline_smoothed, baseline_n_samples):
    epoch_duration_samples = int(0.8 * baseline_smoothed.info['sfreq'])  # 0.8s * sampling rate (500 Hz)
    event_times = np.arange(0, baseline_n_samples, epoch_duration_samples)

    # Create dummy events with event ID 1
    dummy_events = np.array([[int(time), 0, 1] for time in event_times])

    # Now create epochs of 0.8 seconds
    helper_epochs = mne.Epochs(baseline_smoothed, dummy_events, event_id=1, tmin=0, tmax=1.2, baseline=None)
    return helper_epochs
baseline_epochs = baseline_epochs(baseline_smoothed, baseline_n_samples)

def motor_epochs(motor_smoothed, response_events):
    events = mne.events_from_annotations(motor_smoothed)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in response_events.values()]
    tmin = -0.3
    tmax = 0.9
    epoch_parameters = [tmin, tmax, filtered_events]
    tmin, tmax, event_ids = epoch_parameters
    event_ids = {key: val for key, val in response_events.items() if any(event[2] == val for event in filtered_events)}

    motor_epochs = mne.Epochs(motor_smoothed,
                               filtered_events,
                               event_id=event_ids,
                               tmin=tmin,
                               tmax=tmax,
                               baseline=(None, 0),
                               detrend=0,  # should we set it here?
                               preload=True)
    return motor_epochs

motor_epochs = motor_epochs(motor_smoothed, response_events)

# get emg epochs, around the target and distractor events:
def epochs(events_dict, emg_smoothed, events_emg):
    tmin = -0.3
    tmax = 0.9
    event_ids = {key: val for key, val in events_dict.items() if any(event[2] == val for event in events_emg)}
    epochs = mne.Epochs(emg_smoothed, events_emg, event_id=event_ids, tmin=tmin, tmax=tmax, baseline=(-0.3, 0.0))
    return epochs

target_epochs = epochs(target_events, target_emg_smoothed, targets_emg_events)
distractor_epochs = epochs(distractor_events, distractor_emg_smoothed, distractors_emg_events)
response_epochs = epochs(response_events, response_emg_smoothed, responses_emg_events)


# Deep copy the target_epochs to avoid modifying the original data
targets_epochs_copy = deepcopy(target_epochs)
distractors_epochs_copy = deepcopy(distractor_epochs)
motor_epochs_copy = deepcopy(motor_epochs)
baseline_epochs_copy = deepcopy(baseline_epochs)
# Step 1: Calculate baseline and motor Z-score
def get_helpers_z_scores(helper_epochs):
    helper_data = helper_epochs.get_data()

    # Calculate the mean and std for each epoch and each channel independently
    helper_mean = helper_data.mean(axis=2)  # Mean for each epoch and each channel (n_epochs, n_channels)
    helper_std = helper_data.std(axis=2)  # Std deviation for each epoch and each channel (n_epochs, n_channels)

    # Reshape mean and std for broadcasting: (n_epochs, n_channels, 1) to match the time axis
    helper_mean = helper_mean[:, :, np.newaxis]
    helper_std = helper_std[:, :, np.newaxis]

    # Compute z-scores for each epoch and channel
    helper_z_data = (helper_data - helper_mean) / helper_std
    return helper_data, helper_mean, helper_std, helper_z_data

baseline_data, baseline_mean, baseline_std, baseline_z_data = get_helpers_z_scores(baseline_epochs_copy)
motor_data, motor_mean, motor_std, motor_z_data = get_helpers_z_scores(motor_epochs_copy)


# Step 2: Calculate button press and baseline Z-scores
baseline_threshold = np.median(baseline_z_data.flatten())  # 95th percentile
# response_threshold = np.median(motor_z_data.flatten())  #
response_threshold = np.percentile(motor_z_data.flatten(), 95)
print("Baseline Threshold: ", baseline_threshold)
print("Response Threshold: ", response_threshold)

# Step 4: Z-score for target epochs
def z_scores(emg_epochs):
    emg_data = emg_epochs.get_data()
    emg_mean = emg_data.mean(axis=(0, 2))  # Mean across epochs and time points
    emg_std = emg_data.std(axis=(0, 2))    # Std deviation across epochs and time points
    emg_z_data = (emg_data - emg_mean[:, np.newaxis]) / emg_std[:, np.newaxis]  # Z-score of target epochs
    return emg_data, emg_mean, emg_std, emg_z_data
target_data, target_mean, target_std, target_z_data = z_scores(targets_epochs_copy)
distractor_data, distractor_mean, distractor_std, distractor_z_data = z_scores(distractors_epochs_copy)

# Step 1: Combine the z-scores of both channels
def combine_channels(z_data):
    combined_z_data = z_data.mean(axis=1)  # Average z-scores across the two channels
    return combined_z_data


# Step 2: Classify based on thresholds
def classify_epochs(z_data, baseline_threshold, response_threshold):
    classifications = np.empty(z_data.shape, dtype=object)  # Create an empty array for classifications

    for epoch_idx in range(z_data.shape[0]):
        if z_data[epoch_idx].max() > response_threshold:  # Check if the z-score is above response threshold
            classifications[epoch_idx] = 'True Response'
        elif z_data[
            epoch_idx].max() > baseline_threshold:  # Check if the z-score is above baseline but below response threshold
            classifications[epoch_idx] = 'Partial Error'
        else:  # Anything below the baseline threshold
            classifications[epoch_idx] = 'No Response'

    return classifications


# Step 3: Map classifications to numeric values for plotting
def map_classifications_to_numeric(classifications):
    y_values = np.zeros(classifications.shape[0])  # Initialize for epochs

    for epoch_idx in range(classifications.shape[0]):
        epoch_classifications = classifications[epoch_idx]

        # Get the most frequent classification (mode) within the epoch
        most_common_class = mode(epoch_classifications, axis=None)[0][0]

        if most_common_class == 'True Response':
            y_values[epoch_idx] = 2
        elif most_common_class == 'Partial Error':
            y_values[epoch_idx] = 1
        else:
            y_values[epoch_idx] = 0
    return y_values


# Step 4: Plot the classifications for each epoch
target = ['target', 'distractor']
def plot_epoch_classifications(target, classifications):
    y_values = map_classifications_to_numeric(classifications)

    # X-axis: epoch index (0 to n)
    x_values = np.arange(len(y_values))

    plt.figure(figsize=(12, 6))

    # Plot the classifications
    plt.scatter(x_values, y_values, label='Combined Channels', alpha=0.7, color='blue')

    # Customize the y-axis to represent the categories
    plt.yticks([0, 1, 2], ['No Response', 'Partial Error', 'True Response'])

    plt.xlabel('Epochs')
    plt.ylabel('Response Classification')
    plt.title('Epoch Classifications (Combined Channels)')

    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(fig_path/f'{sub_input}_response_{target}_claffifications_{condition}_{index}')


# Example usage with target z-data:
combined_target_z_data = combine_channels(target_z_data)
target_classifications = classify_epochs(combined_target_z_data, baseline_threshold, response_threshold)
plot_epoch_classifications(target[0], target_classifications)

combined_distractor_z_data = combine_channels(distractor_z_data)
distractor_classifications = classify_epochs(combined_distractor_z_data, baseline_threshold, response_threshold)
plot_epoch_classifications(target[1], distractor_classifications)