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
import matplotlib
from matplotlib import pyplot as plt, patches
from meegkit.dss import dss_line
import pandas as pd
from helper import grad_psd, snr

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


def create_events(chosen_events, events_mapping):
    target_number = target_block[3]
    events = mne.events_from_annotations(emg)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in chosen_events.values()]
    event_value = [int(key) for key, val in events_mapping.items() if val == target_number]  # specify key as integer,
    # otherwise event_value will be empty (as keys are strings normally)
    event_value = event_value[0]
    final_events = [event for event in filtered_events if event[2] == event_value]
    return final_events


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


# rectify the EMG data:
def rectify(emg_filt):
    emg_rectified = emg_filt.copy()
    emg_rectified._data = np.abs(emg_rectified._data)
    return emg_rectified


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


# Create dummy events every 1.2 seconds
def baseline_epochs(baseline_smoothed, baseline_n_samples):
    epoch_duration_samples = int(0.8 * baseline_smoothed.info['sfreq'])  # 0.8s * sampling rate (500 Hz)
    event_times = np.arange(0, baseline_n_samples, epoch_duration_samples)

    # Create dummy events with event ID 1
    dummy_events = np.array([[int(time), 0, 1] for time in event_times])

    # Now create epochs of 0.8 seconds
    helper_epochs = mne.Epochs(baseline_smoothed, dummy_events, event_id=1, tmin=0, tmax=1.2, baseline=None)
    return helper_epochs

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


# get emg epochs, around the target and distractor events:
def epochs(events_dict, emg_smoothed, events_emg):
    tmin = -0.3
    tmax = 0.9
    event_ids = {key: val for key, val in events_dict.items() if any(event[2] == val for event in events_emg)}
    epochs = mne.Epochs(emg_smoothed, events_emg, event_id=event_ids, tmin=tmin, tmax=tmax, baseline=(-0.3, 0.0))
    return epochs


# Calculate baseline and motor Z-score
def get_helpers_z_scores(helper_epochs):
    helper_data = helper_epochs.get_data(copy=True)

    # Calculate the mean and std for each epoch and each channel independently
    helper_mean = helper_data.mean(axis=2)  # Mean for each epoch and each channel (n_epochs, n_channels)
    helper_std = helper_data.std(axis=2)  # Std deviation for each epoch and each channel (n_epochs, n_channels)

    # Reshape mean and std for broadcasting: (n_epochs, n_channels, 1) to match the time axis
    helper_mean = helper_mean[:, :, np.newaxis]
    helper_std = helper_std[:, :, np.newaxis]

    # Compute z-scores for each epoch and channel
    helper_z_data = (helper_data - helper_mean) / helper_std
    return helper_data, helper_mean, helper_std, helper_z_data


# plot baseline z-data:
# matplotlib.use('TkAgg')
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
# ax1.boxplot(baseline_z_data.flatten())
# ax1.set_title('Baseline Z-Data Box-Plot')
# ax1.set_xlabel('Index')  # for one plot it's plt.xlabel, for a subplot specify which ax, then set_xlabel
# ax1.set_ylabel('Z-scores')
#
# ax2.boxplot(motor_z_data.flatten())
# ax2.set_title('Motor Z-Data Box-Plot')
# ax2.set_xlabel('Index')
# ax2.set_ylabel('Z-scores')
#
# plt.tight_layout()
# plt.show()
# plt.close()


# Z-score for target epochs
def z_scores(emg_epochs):
    emg_data = emg_epochs.get_data()
    emg_mean = emg_data.mean(axis=(0, 2))  # Mean across epochs and time points
    emg_std = emg_data.std(axis=(0, 2))    # Std deviation across epochs and time points
    emg_z_data = (emg_data - emg_mean[:, np.newaxis]) / emg_std[:, np.newaxis]  # Z-score of target epochs
    return emg_data, emg_mean, emg_std, emg_z_data


# Combine the z-scores of both channels
def combine_channels(z_data):
    combined_z_data = z_data.mean(axis=1)  # Average z-scores across the two channels
    return combined_z_data


# Classify based on thresholds
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


# Map classifications to numeric values for plotting
def map_classifications_to_numeric(classifications):
    y_values = np.zeros(classifications.shape[0])  # Initialize for epochs

    for epoch_idx in range(classifications.shape[0]):
        epoch_classifications = classifications[epoch_idx]

        # Get the most frequent classification (mode) within the epoch
        most_common_class = mode(epoch_classifications, axis=None, keepdims=False)[0][0]

        if most_common_class == 'True Response':
            y_values[epoch_idx] = 2
        elif most_common_class == 'Partial Error':
            y_values[epoch_idx] = 1
        else:
            y_values[epoch_idx] = 0
    return y_values


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
    try:
        plt.savefig(fig_path/f'{sub_input}_response_{target}_claffifications_{condition}_{index}.png')
    except Exception as e:
        print(f"Error saving plot: {e}")

def calculate_percentages(classifications):
    # Get unique classifications and their counts
    unique_classes, class_counts = np.unique(classifications, return_counts=True)

    # Map results for easier interpretation
    total_epochs = len(classifications)
    percentages = {class_name: (class_counts[idx] / total_epochs) * 100 for idx, class_name in
                   enumerate(unique_classes)}

    return percentages

def plot_pie_chart(percentages, target):
    labels = ['No Response', 'Partial Error', 'True Response']
    sizes = [percentages.get('No Response', 0),
             percentages.get('Partial Error', 0),
             percentages.get('True Response', 0)]

    plt.figure(figsize=(8, 8))
    plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title(f'{condition} {target} Event Classification Distribution')
    plt.axis('equal')  # Equal aspect ratio ensures that pie chart is drawn as a circle.
    plt.savefig(fig_path / f'{sub_input}_{target}_{condition}_event_distribution.png')



if __name__ == '__main__':
    sub_input = input("Give sub number as subn (n for number): ")
    sub = [sub.strip() for sub in sub_input.split(',')]
    cm = 1 / 2.54
    # 0. LOAD THE DATA
    sub_dirs = []
    fig_paths = []
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
        raw_fif = emg_dir / 'raw files'
        for folder in sub_dir, fig_path, results_path,  raw_fif:
            if not os.path.isdir(folder):
                os.makedirs(folder)

    # Load necessary JSON files
    with open(json_path / "preproc_config.json") as file:
        cfg = json.load(file)
    with open(json_path / "electrode_names.json") as file:
        mapping = json.load(file)
    with open(json_path / 'eeg_events.json') as file:
        markers_dict = json.load(file)

    s1_events = markers_dict['s1_events']
    s2_events = markers_dict['s2_events']  # stimulus 2 markers
    response_events = markers_dict['response_events']  # response markers

    mapping_path = json_path / 'events_mapping.json'
    with open(mapping_path, 'r') as file:
        events_mapping = json.load(file)
    s1_mapping = events_mapping[0]
    s2_mapping = events_mapping[1]
    response_mapping = events_mapping[2]

    condition = input('Please provide condition (exp. EEG): ')  # choose a condition of interest for processing

    ### Get target header files for all participants
    target_header_files_list = []
    for sub_dir in sub_dirs:
        header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
        filtered_files = [file for file in header_files if condition in file]
        if filtered_files:
            target_header_files_list.append(filtered_files)

    all_target_classifications = []
    all_distractor_classifications = []
    ### Process Each File under the Selected Condition ###
    for index in range(5):  # Loop through all the files matching the condition
        emg_file = target_header_files_list[0][index]
        full_path = os.path.join(sub_dir, emg_file)
        emg = mne.io.read_raw_brainvision(full_path, preload=True)

        # Set montage for file:
        emg.rename_channels(mapping)
        emg.set_montage('standard_1020')
        emg = emg.pick_channels(['A2', 'M2', 'A1'])
        emg.set_eeg_reference(ref_channels=['A1'])
        emg.drop_channels(['A1'])

        # Get Target Block information
        csv_path = default_dir / 'params' / 'block_sequences' / f'{sub_input}.csv'
        csv = pd.read_csv(csv_path)
        target_stream, target_blocks, axis, target_mapping, distractor_mapping = get_target_blocks()

        # Define target and distractor events
        target_block = [target_block for target_block in target_blocks.values[index]]
        target_events, distractor_events = define_events(target_stream)

        # Get events for EMG analysis
        targets_emg_events = create_events(target_events, target_mapping)
        distractors_emg_events = create_events(distractor_events, distractor_mapping)
        responses_emg_events = create_events(response_events, response_mapping)

        # Filter, rectify, and smooth the EMG data
        emg_filt = filter_emg(emg)
        emg_rectified = rectify(emg_filt)
        target_emg_smoothed = smoothing(emg_rectified)
        distractor_emg_smoothed = smoothing(emg_rectified)
        response_emg_smoothed = smoothing(emg_rectified)

        # Baseline and Motor Recordings
        baseline_smoothed, baseline_n_samples = get_helper_recoring(recording='baseline')
        motor_smoothed, motor_n_samples = get_helper_recoring(recording='motor')
        epochs_baseline = baseline_epochs(baseline_smoothed, baseline_n_samples)
        epochs_motor = motor_epochs(motor_smoothed, response_events)

        # Epoch the EMG data
        target_epochs = epochs(target_events, target_emg_smoothed, targets_emg_events)
        distractor_epochs = epochs(distractor_events, distractor_emg_smoothed, distractors_emg_events)
        response_epochs = epochs(response_events, response_emg_smoothed, responses_emg_events)

        # Copy epochs for processing
        targets_epochs_copy = deepcopy(target_epochs)
        distractors_epochs_copy = deepcopy(distractor_epochs)
        motor_epochs_copy = deepcopy(epochs_motor)
        baseline_epochs_copy = deepcopy(epochs_baseline)

        # Z-score calculation
        baseline_data, baseline_mean, baseline_std, baseline_z_data = get_helpers_z_scores(baseline_epochs_copy)
        motor_data, motor_mean, motor_std, motor_z_data = get_helpers_z_scores(motor_epochs_copy)
        baseline_threshold = np.median(baseline_z_data.flatten())
        response_threshold = np.percentile(motor_z_data.flatten(), 95)

        # Classify and plot results
        # Plot the classifications for each epoch
        target = ['target', 'distractor']
        target_data, target_mean, target_std, target_z_data = z_scores(targets_epochs_copy)
        distractor_data, distractor_mean, distractor_std, distractor_z_data = z_scores(distractors_epochs_copy)

        combined_target_z_data = combine_channels(target_z_data)
        target_classifications = classify_epochs(combined_target_z_data, baseline_threshold, response_threshold)
        plot_epoch_classifications(target[0], target_classifications)

        combined_distractor_z_data = combine_channels(distractor_z_data)
        distractor_classifications = classify_epochs(combined_distractor_z_data, baseline_threshold, response_threshold)
        plot_epoch_classifications(target[1], distractor_classifications)

        all_target_classifications.append(target_classifications)
        all_distractor_classifications.append(distractor_classifications)
        # Save all results for this file
        print(f"Processing completed for file {index + 1}/{len(target_header_files_list[0])}")

    concatenated_target_classifications = np.concatenate(all_target_classifications, axis=0)

    concatenated_distractor_classifications = np.concatenate(all_distractor_classifications, axis=0)

    z_score_path = results_path / 'z_score_data'
    if not os.path.exists(z_score_path):
        os.makedirs(z_score_path)

    # Calculate percentages for both targets and distractors
    target_percentages = calculate_percentages(concatenated_target_classifications)
    distractor_percentages = calculate_percentages(concatenated_distractor_classifications)

    # Plot for both targets and distractors
    plot_pie_chart(target_percentages, target='Target')
    plot_pie_chart(distractor_percentages, target='Distractor')