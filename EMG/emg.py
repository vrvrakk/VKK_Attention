'''
Pre-processing EMG data:
'''
# libraries:
import mne
from pathlib import Path
import numpy as np
import json
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from scipy.stats import shapiro, kruskal, levene
import scikit_posthocs as sp
import pickle
import os
import statsmodels.api as sm
import pingouin as pg

matplotlib.use('Agg')
samplerate = 500

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
        distractor_stream = 's2'
        target_mapping = s1_mapping
        distractor_mapping = s2_mapping
    elif condition in ['a2', 'e2']:
        target_stream = 's2'
        distractor_stream = 's1'
        target_mapping = s2_mapping
        distractor_mapping = s1_mapping
    else:
        raise ValueError("Invalid condition provided.")  # Handle unexpected conditions
    target_blocks = []
    # iterate through the values of the csv path; first enumerate to get the indices of each row
    for idx, items in enumerate(csv.values):
        block_seq = items[0]  # from first column, save info as variable block_seq
        block_condition = items[1]  # from second column, save info as var block_condition
        if block_seq == target_stream and block_condition == axis:
            block = csv.iloc[idx]
            target_blocks.append(block)  # Append relevant rows
    target_blocks = pd.DataFrame(target_blocks).reset_index(drop=True)  # convert condition list to a dataframe
    return target_stream, distractor_stream, target_blocks, axis, target_mapping, distractor_mapping

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


def create_response_events(chosen_events):
    events = mne.events_from_annotations(emg)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    events = np.array(events)
    filtered_events = [event for event in events if event[2] in chosen_events.values()]
    return filtered_events

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

def baseline_events(chosen_events, events_mapping):
    target_number = target_block[3]
    events = mne.events_from_annotations(emg)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in chosen_events.values()]
    event_value = [int(key) for key, val in events_mapping.items() if val == target_number]  # specify key as integer,
    # otherwise event_value will be empty (as keys are strings normally)
    event_value = event_value[0]
    final_events = [event for event in filtered_events if event[2] != event_value]
    return final_events

# band-pass filter
def filter_emg(emg):
    emg_filt = emg.copy().filter(l_freq=1, h_freq=150)
    '''The typical frequency range for EMG signals from the Flexor Digitorum Profundus (FDP) muscle is around 20 to 450 Hz. 
    Most muscle activity is concentrated in the 50-150 Hz range, but high-frequency components can go up to 450 Hz.'''
    # Notch filter at 50 Hz to remove power line noise
    print('Remove power line noise and apply minimum-phase highpass filter')  # Cheveigné, 2020
    emg_filt.copy().notch_filter(freqs=[50, 100, 150], method='fir')
    emg_filt.append(emg_filt)
    return emg_filt


# rectify the EMG data:
def rectify(emg_filt):
    emg_rectified = emg_filt.copy()
    emg_rectified._data = np.abs(emg_rectified._data)
    # emg_rectified._data *= 1e6
    return emg_rectified

# Create baseline events and epochs
def baseline_epochs(events_emg, emg_rectified, events_dict, target):
    tmin = -0.2
    tmax = 0.9
    event_ids = {key: val for key, val in events_dict.items() if any(event[2] == val for event in events_emg)}
    epochs_baseline = mne.Epochs(emg_rectified, events_emg, event_id=event_ids, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True)
    epochs_baseline.save(fif_path / f'{sub_input}_condition_{condition}_{index}_{target}-epo.fif', overwrite=True)
    return epochs_baseline


def baseline_normalization(emg_epochs, tmin=0.2, tmax=0.9):
    """Calculate the derivative and z-scores for the baseline period."""
    emg_epochs.load_data()  # Ensure data is available
    emg_window = emg_epochs.copy().crop(tmin, tmax)  # Crop to window of interest
    emg_data = emg_window.get_data(copy=True)  # Extract data as a NumPy array

    # Compute the derivative along the time axis
    emg_derivative = np.diff(emg_data, axis=-1)

    # Z-score normalization within each epoch (derivative normalized by its own mean/std)
    emg_derivative_z = (emg_derivative - np.mean(emg_derivative, axis=-1, keepdims=True)) / np.std(
        emg_derivative, axis=-1, keepdims=True)
    emg_var = np.var(emg_derivative_z, axis=-1)  # Variance of baseline
    emg_rms = np.sqrt(np.mean(np.square(emg_derivative_z), axis=-1))  # RMS of baseline

    return emg_data, emg_derivative, emg_derivative_z, emg_var, emg_rms


# Compute the signal power (variance of the response data)
def SNR(data, baseline_data):
    P_signal = np.mean(np.var(data, axis=-1))  # Variance of response data

    # Step 2: Compute the noise power (variance of the baseline data)
    P_noise = np.mean(np.var(baseline_data, axis=-1))  # Variance of baseline data

    # Step 3: Calculate the Signal-to-Noise Ratio (SNR)
    SNR = P_signal / P_noise

    print(f"SNR: {SNR}")


def categorize_events(target_events, distractor_events, non_target_events_target,
                          responses_emg_events, sampling_rate=samplerate):
        # Initialize the lists for categorization
        target_response_events = []
        target_no_response_events = []
        distractor_response_events = []
        distractor_no_response_events = []
        non_target_target_stimulus_events = []
        invalid_non_target_response_events = []
        invalid_target_response_events = []
        invalid_distractor_response_events = []
        response_only_epochs = []

        # Track which response events have been assigned
        unassigned_response_events = responses_emg_events

        def process_events(events, label, response_list, no_response_list, invalid_list):
            nonlocal unassigned_response_events
            for event in events:
                stim_timepoint = event[0] / sampling_rate
                time_start = stim_timepoint
                time_end = stim_timepoint + 0.9

                response_found = False
                for response_event in unassigned_response_events:
                    response_timepoint = response_event[0] / sampling_rate
                    if time_start < response_timepoint < time_end:
                        if response_timepoint - stim_timepoint < 0.2:
                            invalid_list.append(np.append(event, label))
                        else:
                            response_list.append(event)

                        idx_to_remove = np.where((unassigned_response_events == response_event).all(axis=1))[0][0]
                        unassigned_response_events = np.delete(unassigned_response_events, idx_to_remove, axis=0)
                        response_found = True
                        break

                if not response_found:
                    no_response_list.append(event)

        # Step 4: Process target events
        process_events(target_events, 'target', target_response_events, target_no_response_events,
                       invalid_target_response_events)

        # Step 5: Process distractor events
        process_events(distractor_events, 'distractor', distractor_response_events, distractor_no_response_events,
                       invalid_distractor_response_events)

        # Step 6: Process non-target events
        # non_target_combined_events = np.concatenate([non_target_events_target, non_target_events_distractor])
        # non_target_combined_events = non_target_combined_events[np.argsort(non_target_combined_events[:, 0])]
        process_events(non_target_events_target, 'non_target_target', [], non_target_target_stimulus_events,
                       invalid_non_target_response_events)

        # Step 7: Remaining responses are pure response events (response_only_epochs)
        response_only_epochs = unassigned_response_events

        return (
            target_response_events,
            target_no_response_events,
            invalid_target_response_events,
            distractor_response_events,
            distractor_no_response_events,
            invalid_distractor_response_events,
            non_target_target_stimulus_events,
            invalid_non_target_response_events,
            response_only_epochs)


# get emg epochs, around the target and distractor events:
def epochs(events_dict, emg_rectified, events_emg, target='', tmin=0, tmax=0, baseline=None):
    event_ids = {key: val for key, val in events_dict.items() if any(event[2] == val for event in events_emg)} # todo: check diff 1000 and 500 hz
    epochs = mne.Epochs(emg_rectified, events_emg, event_id=event_ids, tmin=tmin, tmax=tmax, baseline=baseline, preload=True, event_repeated='merge')
    epochs.save(fif_path / f'{sub_input}_condition_{condition}_{index}_{target}-epo.fif', overwrite=True)
    return epochs


def get_data(chosen_events, events_type, chosen_emg, chosen_emg_events, target='', tmin=-0.2, tmax=0.9, baseline=None):
    """
        Helper function to create epochs and extract data.

        Parameters:
            chosen_events (list): List of events to check.
            events_type (array): Event data type (e.g., 'target', 'distractor', etc.).
            chosen_emg (array): EMG data associated with the event type.
            chosen_emg_events (list): Specific events for which to create epochs.
            target (str): Label for the target.
            tmin (float): Start time before event.
            tmax (float): End time after event.
            baseline (tuple): Baseline correction window.

        Returns:
            tuple: Tuple containing the epochs object and its data.
        """
    if chosen_events and len(chosen_events) > 0:
        chosen_events_epochs = epochs(events_type, chosen_emg, chosen_emg_events, target=target, tmin=tmin, tmax=tmax, baseline=baseline)
        chosen_epochs_data = chosen_events_epochs.get_data(copy=True)
        return chosen_events_epochs, chosen_epochs_data
    return None, None


def is_valid_epoch_list(epochs):
    return [epoch for epoch in epochs if epoch is not None and isinstance(epoch, mne.Epochs) and len(epoch) > 0]

def combine_all_epochs(all_epochs_dict, condition_list, label=''):
    combined_epochs_dict = {cond: [] for cond in condition_list}
    combined_data_dict = {cond: [] for cond in condition_list}
    combined_events_dict = {cond: [] for cond in condition_list}
    for condition, epoch in all_epochs_dict.items():
        epochs = all_epochs_dict[condition]
        if epochs:
            valid_epochs = is_valid_epoch_list(epochs)
            if valid_epochs:
                combined_epoch = mne.concatenate_epochs(epochs)
                combined_data = combined_epoch.get_data(copy=True)
                # Save combined epochs
                combined_epoch.save(
                    combined_epochs / f'{sub_input}_condition_{condition}_combined_{condition}_epochs-epo.fif',
                    overwrite=True
                )


                combined_epochs_events = combined_epoch.events
                combined_epochs_dict[condition].append(combined_epoch)
                combined_data_dict[condition].append(combined_data)
                combined_events_dict[condition].append(combined_epochs_events)
            else:
                combined_epochs_dict[condition] = None
                combined_data_dict[condition] = None
                combined_events_dict[condition] = None
    return combined_epochs_dict, combined_data_dict, combined_events_dict

# get combined epochs with all epochs of 1 condition in one:

def tfa_heatmap(epochs, target):
    frequencies = np.logspace(np.log10(1), np.log10(150), num=150)  # Frequencies from 1 to 30 Hz
    n_cycles = np.minimum(frequencies / 2, 7)  # Number of cycles in Morlet wavelet (adapts to frequency)

    # Compute the Time-Frequency Representation (TFR) using Morlet wavelets
    epochs = epochs.copy().crop(tmin=-0.2, tmax=0.9)
    power = mne.time_frequency.tfr_morlet(epochs, freqs=frequencies, n_cycles=n_cycles, return_itc=False)
    power.apply_baseline(baseline=(-0.2, 0.0), mode='logratio')

    # Plot TFR as a heatmap (power across time and frequency)
    power_plot = power.plot([0], title=f'TFR (Heatmap) {condition} {target}',
                            fmin=frequencies[0], fmax=frequencies[-1],
                            vmin=-1, vmax=1, cmap='viridis', show=True)
    # Customize y-axis ticks
    fixed_ticks = [1, 10, 50, 100, 150]  # Example fixed y-tick values

    for i, fig in enumerate(power_plot):
        # Access the current axis
        ax = fig.gca()
        ax.set_yticks(fixed_ticks)  # Set fixed y-ticks
        fig.savefig(
            psd_path / f'{sub_input}_{condition}_{target}_plot.png')  # Save with a unique name for each figure
        plt.close(fig)
    return power


def get_avg_band_power(power, bands, fmin, fmax, tmin=0.0, tmax=0.9):
    """Compute the average power for specified frequency bands within a time window."""
    time_mask = (power.times >= tmin) & (power.times <= tmax)  # Mask for the time window
    low_band = bands['low_band']
    middle_band = bands['mid_band']
    high_band = bands['high_band']

    # investigate entire epoch:
    for band_name, (fmin, fmax) in bands.items():
        freq_mask = (power.freqs >= fmin) & (power.freqs <= fmax)
        band_power = power.data[:, freq_mask, :][:, :, time_mask]  # Data within frequency band and time window

        # Compute the average power within the frequency band and time window
        avg_band_power = band_power.mean()  # Mean across all epochs, frequencies, and time within the window

    # investigate the three bands separately:
    low_freq_mask = (power.freqs >= low_band[0]) & (power.freqs <= low_band[1])
    low_band_power = power.data[:, low_freq_mask, :][:, :, time_mask]  # Data within frequency band and time window

    # Compute the average power within the frequency band and time window
    low_avg_band_power = low_band_power.mean()  # Mean across all epochs, frequencies, and time within the window

    middle_freq_mask = (power.freqs >= middle_band[0]) & (power.freqs <= middle_band[1])
    middle_band_power = power.data[:, middle_freq_mask, :][:, :,
                        time_mask]  # Data within frequency band and time window
    # Compute the average power within the frequency band and time window
    middle_avg_band_power = middle_band_power.mean()  # Mean across all epochs, frequencies, and time within the window

    high_freq_mask = (power.freqs >= high_band[0]) & (power.freqs <= high_band[1])
    high_band_power = power.data[:, high_freq_mask, :][:, :, time_mask]  # Data within frequency band and time window
    # Compute the average power within the frequency band and time window
    high_avg_band_power = high_band_power.mean()  # Mean across all epochs, frequencies, and time within the window

    # find dominant band:
    bands_avg = {'low_band_avg': low_avg_band_power, 'mid_band_avg': middle_avg_band_power,
                 'high_band_avg': high_avg_band_power}
    dominant_band = max(bands_avg, key=bands_avg.get)
    # find dominant frequency in dominant band:
    if dominant_band == 'low_band_avg':
        mean_power_across_time = low_band_power.mean(axis=2)
        dominant_freq_index = mean_power_across_time.argmax(axis=1)
        dominant_freq = int(power.freqs[dominant_freq_index])
    elif dominant_band == 'mid_band_avg':
        mean_power_across_time = middle_band_power.mean(axis=2)
        dominant_freq_index = mean_power_across_time.argmax(axis=1)
        dominant_freq = int(power.freqs[dominant_freq_index])
    else:
        mean_power_across_time = high_band_power.mean(axis=2)
        dominant_freq_index = mean_power_across_time.argmax(axis=1)
        dominant_freq = int(power.freqs[dominant_freq_index])
    vals = {'dominant band': dominant_band,
            'dominant freq': dominant_freq,
            'avg power': avg_band_power,
            'low avg power': low_avg_band_power,
            'mid avg power': middle_avg_band_power,
            'high avg power': high_avg_band_power}
    return vals


def epochs_vals(epochs_dict):
    frequencies = np.logspace(np.log10(1), np.log10(150), num=150)
    n_cycles = frequencies / 2

    # Define frequency bands
    bands = {
        'band_1_10': (1, 10),
        'band_11_20': (11, 20),
        'band_21_30': (21, 30),
        'band_31_40': (31, 40),
        'band_41_50': (41, 50),
        'band_51_60': (51, 60),
        'band_61_70': (61, 70),
        'band_71_80': (71, 80),
        'band_81_90': (81, 90),
        'band_91_100': (91, 100),
        'band_101_110': (101, 110),
        'band_111_120': (111, 120),
        'band_121_130': (121, 130),
        'band_131_140': (131, 140),
        'band_141_150': (141, 150)
    }

    # Initialize dictionary to store results for each condition and epoch
    results_dict = {}

    for condition, epochs_array in epochs_dict.items():
        # If epochs_array is a list, extract the first element
        if isinstance(epochs_array, list) and len(epochs_array) > 0:
            epochs_array = epochs_array[0]

        # Initialize a list to store metrics for each epoch under the current condition
        condition_epochs_vals = []

        # Get the data for all epochs in the current EpochsArray
        all_epochs_data = epochs_array.get_data(copy=False)  # Shape: (n_epochs, n_channels, n_samples)

        # Loop over each epoch's data
        for epoch_index, epoch_data in enumerate(all_epochs_data):
            # Get single epoch as an Epochs object for TFR computation
            epoch_info = epochs_array.info
            single_epoch = mne.EpochsArray(epoch_data[np.newaxis, ...], epoch_info, tmin=epochs_array.tmin)

            # Compute TFR for the individual epoch
            power = mne.time_frequency.tfr_morlet(single_epoch, freqs=frequencies, n_cycles=n_cycles, return_itc=False)

            # Mask for early and late phases
            time_mask = (power.times >= 0.0) & (power.times <= 0.9)

            # Dictionary to store metrics for each band in the current epoch
            epoch_vals = {}
            bands_avg = {}

            # Iterate through each frequency band and calculate metrics
            for band_name, (fmin, fmax) in bands.items():
                freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
                band_power = power.data[:, freq_mask, :][:, :, time_mask]

                # Calculate average power across the time dimension
                avg_band_power = band_power.mean()  # Average power over time

                # Store the average power for later determination of dominant band
                bands_avg[band_name] = avg_band_power

                # Store values in dictionary for each band
                epoch_vals[band_name] = {
                    'avg power': avg_band_power,
                }

            # Determine the dominant band based on the highest avg power across all bands
            dominant_band = max(bands_avg, key=bands_avg.get)
            dominant_band_freq_mask = (frequencies >= bands[dominant_band][0]) & (
                    frequencies <= bands[dominant_band][1]
            )
            dominant_band_power = power.data[:, dominant_band_freq_mask, :][:, :, time_mask]

            # Find the dominant frequency within the dominant band
            mean_power_across_time = dominant_band_power.mean(axis=2)
            dominant_freq_index = mean_power_across_time.argmax(axis=1)
            dominant_freq = frequencies[dominant_band_freq_mask][dominant_freq_index[0]]


            # Append overall metrics and dominant band/freq to epoch values
            epoch_vals.update({
                'overall_avg_power': sum(bands_avg.values()) / len(bands_avg),
                'dominant_band': dominant_band,
                'dominant_freq': dominant_freq
            })

            # Append metrics for the current epoch to the list for this condition
            condition_epochs_vals.append(epoch_vals)

        # Store the metrics for all epochs under the current condition in results_dict
        results_dict[condition] = condition_epochs_vals

    return results_dict

def filter_outliers_epochs(results_dict):
    """
    Filter out outliers from the 'overall_avg_power' field in the results_dict.
    """
    filtered_results = {}

    for condition, epochs_list in results_dict.items():
        # Extract 'overall_avg_power' values for all epochs in the condition
        all_powers = [epoch['overall_avg_power'] for epoch in epochs_list]

        # Compute IQR
        Q1 = np.percentile(all_powers, 25)
        Q3 = np.percentile(all_powers, 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter out epochs that have 'overall_avg_power' outside the IQR bounds
        filtered_epochs_list = [
            epoch for epoch in epochs_list
            if lower_bound <= epoch['overall_avg_power'] <= upper_bound
        ]

        # Store filtered epochs for the current condition
        filtered_results[condition] = filtered_epochs_list

    return filtered_results

import seaborn as sns


def add_labels(data, label, event_times, type):
    squeezed_data = data.squeeze(axis=1)  # (16, 551)
    data_df = pd.DataFrame(squeezed_data)
    data_df['Label'] = label
    data_df['Timepoints'] = event_times
    data_df['Type'] = type
    return data_df

def process_event_data(events, event_data, label, sampling_rate=samplerate, event_type=None, event_name=''):
    """
    Converts event data to array, calculates response times, and applies labels.

    Parameters:
        event_data (list): List of events to process.
        label (str): Label to assign, e.g., 'Response' or 'No Response'.
        sampling_rate (int): Sampling rate for converting timepoints. Default is samplerate Hz.
        event_type (str): Type of event, e.g., 'target' or 'distractor'.
        event_name (str): Name of the event variable to check within locals().

    Returns:
        DataFrame or None: Processed DataFrame with labeled events if the condition is met, otherwise None.
    """
    # Check if event_name is defined in locals and if event_data is non-empty
    if event_name in globals() and events is not None:
        event_array = np.array(events)
        event_times = event_array[:, 0] / sampling_rate
        event_df = add_labels(event_data, label=label, event_times=event_times, type=event_type)
        return event_df
    else:
        return None


def significance_label(p_val):
    if p_val < 0.0001:
        return "****"
    elif p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return "ns"

def cliffs_delta(x, y):
    n_x, n_y = len(x), len(y)
    greater = sum(1 for i in x for j in y if i > j)
    less = sum(1 for i in x for j in y if i < j)
    delta = (greater - less) / (n_x * n_y)
    return delta

# def add_bootstrapped_ci(data, group_col, value_col, ax, palette):
#     """
#     Adds bootstrapped confidence intervals to violin plots.
#     """
#     groups = data[group_col].unique()
#     for group in groups:
#         group_data = data[data[group_col] == group][value_col]
#         bootstrapped_means = [np.mean(np.random.choice(group_data, size=len(group_data), replace=True)) for _ in range(1000)]
#         ci_lower, ci_upper = np.percentile(bootstrapped_means, [2.5, 97.5])
#         x_pos = list(groups).index(group)
#         ax.errorbar(x_pos, np.mean(group_data), yerr=[[np.mean(group_data) - ci_lower], [ci_upper - np.mean(group_data)]],
#                     fmt='o', color=(1.0, 0.8509803921568627, 0.1843137254901961), capsize=5)


def plot_dominant_frequency_counts(target_results_dict, distractor_results_dict, non_target_results_dict):
    for condition in target_results_dict.keys():
        if condition not in distractor_results_dict or condition not in non_target_results_dict:
            print(f"Condition '{condition}' is missing in one of the dictionaries.")
            continue

        # Extract dominant frequencies for each condition
        target_dominant_freqs = [epoch['dominant_freq'] for epoch in target_results_dict[condition]]
        distractor_dominant_freqs = [epoch['dominant_freq'] for epoch in distractor_results_dict[condition]]
        non_target_dominant_freqs = [epoch['dominant_freq'] for epoch in non_target_results_dict[condition]]

        # Combine data for counts
        df = pd.DataFrame({
            'Frequency': target_dominant_freqs + distractor_dominant_freqs + non_target_dominant_freqs,
            'Epoch Type': (['Target'] * len(target_dominant_freqs)) +
                          (['Distractor'] * len(distractor_dominant_freqs)) +
                          (['Non-Target'] * len(non_target_dominant_freqs))
        })

        # Count frequencies for each epoch type
        counts = df.groupby(['Epoch Type', 'Frequency']).size().reset_index(name='Count')

        # Pivot data for plotting
        counts_pivot = counts.pivot(index='Frequency', columns='Epoch Type', values='Count').fillna(0)

        # Plot the counts
        plt.figure(figsize=(12, 8))
        ax = counts_pivot.plot(kind='bar', stacked=False, figsize=(12, 8), color=['darkviolet', 'gold', 'royalblue'])
        plt.title(f"{condition} Dominant Frequency Counts by Epoch Type")
        # Add xticks without labels
        plt.xticks(ticks=np.arange(80), labels=range(1, 81, 1), rotation=0, fontsize=8)
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Count")
        plt.legend(title="Epoch Type")
        plt.grid(axis='y', alpha=0.3)

        # Save the plot
        plt.tight_layout()
        plt.savefig(class_figs / f"{condition}_dominant_frequency_counts_by_epoch_type.png")
        plt.close()


from scipy.stats import chi2_contingency

def plot_dominant_band_distributions(target_results_dict, distractor_results_dict, non_target_results_dict):
    def calculate_band_counts(epoch_vals, band_types):
        dominant_bands = [epoch['dominant_band'] for epoch in epoch_vals]
        return {band: dominant_bands.count(band) for band in band_types}

    # Define the band types
    band_types = ['band_1_10', 'band_11_20', 'band_21_30', 'band_31_40', 'band_41_50', 'band_51_60', 'band_61_70', 'band_71_80', 'band_81_90', 'band_91_100',
        'band_101_110', 'band_111_120', 'band_121_130', 'band_131_140', 'band_141_150']
    metrics = {}

    for condition in target_results_dict.keys():
        if condition not in distractor_results_dict or condition not in non_target_results_dict:
            print(f"Condition '{condition}' is missing in one of the dictionaries.")
            continue

        # Calculate counts for each epoch type
        target_counts_dict = calculate_band_counts(target_results_dict[condition], band_types)
        distractor_counts_dict = calculate_band_counts(distractor_results_dict[condition], band_types)
        non_target_counts_dict = calculate_band_counts(non_target_results_dict[condition], band_types)

        # Find bands with non-zero counts in any of the three dictionaries
        filtered_bands = [
            band for band in band_types
            if target_counts_dict[band] > 0 or distractor_counts_dict[band] > 0 or non_target_counts_dict[band] > 0
        ]

        # Convert counts to lists for contingency table using the unified filtered bands
        target_counts = [target_counts_dict[band] for band in filtered_bands]
        distractor_counts = [distractor_counts_dict[band] for band in filtered_bands]
        non_target_counts = [non_target_counts_dict[band] for band in filtered_bands]

        # Create column names for the filtered bands
        filtered_columns = [
            f"Band {band.split('_')[1]}"  # Adjust column names based on the band (e.g., '1-10')
            for band in filtered_bands
        ]

        total_target = sum(target_counts)
        total_distractor = sum(distractor_counts)
        total_non_target = sum(non_target_counts)

        total_counts = sum(target_counts + distractor_counts + non_target_counts)

        # Perform Chi-Square Test on raw counts (not proportions)
        raw_contingency_table = pd.DataFrame(
            [target_counts, distractor_counts, non_target_counts],
            index=['Target', 'Distractor', 'Non-Target'],
            columns=filtered_columns)
        chi2, p, dof, expected = chi2_contingency(raw_contingency_table)
        print(f"{condition} - Chi-Square Statistic: {chi2}")
        print(f"{condition} - p-value: {p}")
        print(f"{condition} - Degrees of Freedom: {dof}")
        print(f"{condition} - Expected Frequencies:\n", expected)
        # Calculate Cramér’s V
        min_dim = min(raw_contingency_table.shape) - 1
        cramer_v = np.sqrt(chi2 / (total_counts * min_dim))

        # Save metrics
        metrics[condition] = {
            'Chi-Square Statistic': chi2,
            'p-value': p,
            'Degrees of Freedom': dof,
            'Cramér’s V': cramer_v,
            'Observed Counts': raw_contingency_table.to_dict(),
            'Expected Frequencies': pd.DataFrame(expected, index=['Target', 'Distractor', 'Non-Target'],
                                                 columns=filtered_columns).to_dict()
        }

        # Determine significance label based on p-value
        significance_label = ""
        if p < 0.0001:
            significance_label = "****"
        elif p < 0.001:
            significance_label = "***"
        elif p < 0.01:
            significance_label = "**"
        elif p < 0.05:
            significance_label = "*"

        # Visualization with a Stacked Bar Chart
        # Calculate the normalized contingency table
        normalized_contingency_table = raw_contingency_table.div(raw_contingency_table.sum(axis=1), axis=0)

        fig, ax = plt.subplots(figsize=(16, 12))
        epoch_types = ['Target', 'Distractor', 'Non-Target']
        bottom = np.zeros(3)  # Starting point for the bottom of each stack

        # Stacked bar plot
        for i, band in enumerate(filtered_columns):
            ax.bar(epoch_types, normalized_contingency_table[band], label=band, bottom=bottom)
            bottom += normalized_contingency_table[band]  # Update the bottom position for the next band

        # Labels and title
        ax.set_ylim(0, max(bottom) * 1.2)
        ax.set_ylabel("Count of Dominant Bands")
        ax.set_title(f"{condition} Dominant Band Distribution by Epoch Type {significance_label}")
        ax.legend(title="Band Type")

        plt.savefig(class_figs/f"{condition}_dominant_band_distribution_comparison.png")
        plt.close()
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        metrics_df.to_csv(results_path / f"{sub_input}_dominant_band_metrics.csv")


def plot_dominant_frequency_distributions(target_results_dict, distractor_results_dict, non_target_results_dict):
    metrics = {}
    for condition in target_results_dict.keys():
        if condition not in distractor_results_dict or condition not in non_target_results_dict:
            print(f"Condition '{condition}' is missing in one of the dictionaries.")
            continue

        # Extract dominant frequencies for each condition
        target_dominant_freqs = [epoch['dominant_freq'] for epoch in target_results_dict[condition]]
        distractor_dominant_freqs = [epoch['dominant_freq'] for epoch in distractor_results_dict[condition]]
        non_target_dominant_freqs = [epoch['dominant_freq'] for epoch in non_target_results_dict[condition]]


        # calculate Cliff's delta:
        # quantifies the amount of difference between two groups of observations beyond p-values interpretation.
        target_distractor_delta = cliffs_delta(target_dominant_freqs, distractor_dominant_freqs)
        tagret_non_target_delta = cliffs_delta(target_dominant_freqs, non_target_dominant_freqs)
        distractor_non_target_delta = cliffs_delta(distractor_dominant_freqs, non_target_dominant_freqs)
        # Prepare DataFrame for plotting and statistical testing
        df = pd.DataFrame({
            'Frequency': target_dominant_freqs + distractor_dominant_freqs + non_target_dominant_freqs,
            'Epoch Type': (['Target'] * len(target_dominant_freqs)) +
                          (['Distractor'] * len(distractor_dominant_freqs)) +
                          (['Non-Target'] * len(non_target_dominant_freqs))
                          })

        # Test for normality with wilk test
        alpha = 0.05
        is_normal = all(shapiro(group)[1] > alpha for group in
                        [target_dominant_freqs, distractor_dominant_freqs, non_target_dominant_freqs])

        significance_labels = {}
        if is_normal:
            # Test for Homogeneity of Variances
            levene_stat, levene_p = levene(target_dominant_freqs, distractor_dominant_freqs,
                                           non_target_dominant_freqs)
            print(f"{condition} - Levene's test p-value: {levene_p}")

            if levene_p < alpha:
                # Variances are unequal: Use Welch's ANOVA
                welch_result = pg.welch_anova(data=df, dv='Frequency', between='Epoch Type')
                welch_p_val = welch_result['p-unc'][0]
                eta_squared = welch_result['eta-square'][0]
                print(f"{condition} - Welch's ANOVA p-value: {welch_p_val}")
                print(f"{condition} - Welch's ANOVA Effect Size (eta-squared): {eta_squared}")

                metrics[f'{condition}_welch'] = {
                    'Welch ANOVA p-value': welch_p_val,
                    'Effect Size (eta-squared)': eta_squared
                }

                if welch_p_val < alpha:
                    posthoc = sp.posthoc_ttest(df, val_col='Frequency', group_col='Epoch Type',
                                               p_adjust='bonferroni')
                    significance_labels = {
                        ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                        ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
                        ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target']),
                    }
            else:
                # Variances are equal: Use standard ANOVA
                anova_result = sm.stats.anova_lm(sm.OLS.from_formula("Frequency ~ C(Epoch Type)", data=df).fit(),
                                                 typ=2)
                anova_p_val = anova_result['PR(>F)']['C(Epoch Type)']
                total_ss = anova_result['sum_sq'].sum()
                eta_squared = anova_result['sum_sq']['C(Epoch Type)'] / total_ss
                print(f"{condition} - ANOVA p-value: {anova_p_val}")
                print(f"{condition} - ANOVA Effect Size (eta-squared): {eta_squared}")

                metrics[f'{condition}_anova'] = {
                    'ANOVA p-value': anova_p_val,
                    'Effect Size (eta-squared)': eta_squared
                }

                if anova_p_val < alpha:
                    posthoc = sp.posthoc_ttest(df, val_col='Frequency', group_col='Epoch Type', p_adjust='bonferroni')
                    significance_labels = {
                        ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                        ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
                        ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target']),
                    }
                    metrics[f'{sub_input}_{condition}']['T-test Target vs. Distractor'] = posthoc.loc[
                        'Target', 'Distractor']
                    metrics[f'{sub_input}_{condition}']['T-test Target vs. Non-Target'] = posthoc.loc[
                        'Target', 'Non-Target']
                    metrics[f'{sub_input}_{condition}']['T-test Distractor vs. Non-Target'] = posthoc.loc[
                        'Distractor', 'Non-Target']
        else:
            # Use Kruskal-Wallis for non-normal data
            kruskal_h_val, kruskal_p_val = kruskal(target_dominant_freqs, distractor_dominant_freqs, non_target_dominant_freqs)

            def kruskal_eta_squared(H, N):
                return (H - (3 - 1)) / (N - (3 - 1))  # Adjust for 3 groups

            N = len(target_dominant_freqs) + len(distractor_dominant_freqs) + len(non_target_dominant_freqs)
            eta_squared = kruskal_eta_squared(kruskal_h_val, N)
            print(f"Kruskal-Wallis Effect Size (eta-squared): {eta_squared:.4f}")

            # Store the results
            metrics[f'{sub_input}_{condition}'] = {
                'Kruskal-Wallis H': kruskal_h_val,
                'Kruskal-Wallis p-value': kruskal_p_val,
                'Effect Size (eta-squared)': eta_squared
            }
            if kruskal_p_val < alpha:
                # Effect size calculation
                posthoc = sp.posthoc_dunn(df, val_col='Frequency', group_col='Epoch Type', p_adjust='bonferroni')
                metrics[f'{sub_input}_{condition}']['Dunn Posthoc Target vs. Distractor'] = posthoc.loc[
                    'Target', 'Distractor']
                metrics[f'{sub_input}_{condition}']['Dunn Posthoc Target vs. Non-Target'] = posthoc.loc[
                    'Target', 'Non-Target']
                metrics[f'{sub_input}_{condition}']['Dunn Posthoc Distractor vs. Non-Target'] = posthoc.loc[
                    'Distractor', 'Non-Target']

                significance_labels = {
                    ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                    ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
                    ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target'])}
                # Store the Cliff's Delta values in the metrics DataFrame
                metrics[f'{sub_input}_{condition}']['Cliff Delta Target vs Distractor'] = target_distractor_delta
                metrics[f'{sub_input}_{condition}']['Cliff Delta Target vs Non-Target'] = tagret_non_target_delta
                metrics[f'{sub_input}_{condition}']['Cliff Delta Distractor vs Non-Target'] = distractor_non_target_delta


        # Convert metrics to a DataFrame for saving
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        metrics_df.to_csv(results_path / f'{sub_input}_frequency_metrics.csv')
        # Plot violin plot for frequency distribution comparison
        plt.figure(figsize=(12, 10))
        colors = ['darkviolet', 'royalblue', 'gold']
        ax = sns.violinplot(x='Epoch Type', y='Frequency', data=df, palette=colors, hue='Epoch Type', legend=False)
        plt.legend(title=f'Sample Size: {len(target_dominant_freqs)}')
        # ax = sns.stripplot(data=df, x="Epoch Type", y="Frequency", color="black", alpha=0.5, jitter=True)
        # add_bootstrapped_ci(df, 'Epoch Type', 'Frequency', ax, colors)
        plt.title(f"{condition} Dominant Frequency Distribution by Epoch Type")
        plt.ylabel("Dominant Frequency (Hz)")


        # Add significance labels for all comparisons, including non-significant ones
        y_max = df['Frequency'].max()
        y_offset = y_max * 0.01
        pairs = [('Target', 'Distractor'), ('Target', 'Non-Target'), ('Distractor', 'Non-Target')]
        for (group1, group2) in pairs:
            x1, x2 = df['Epoch Type'].unique().tolist().index(group1), df['Epoch Type'].unique().tolist().index(group2)
            y = y_max + y_offset
            label = significance_labels.get((group1, group2), "ns")
            if label != "ns":
                plt.plot([x1, x2], [y, y], color='black', linestyle='solid')
                plt.text((x1 + x2) / 2, y + y_offset, label, ha='center', va='bottom', fontsize=12)
                y_max += y_offset

        # Save the plot
        plt.savefig(class_figs/f"{condition}_dominant_frequency_distributions_by_epoch_type.png")
        plt.close()

def plot_overall_avg_power_bar(target_results_dict, distractor_results_dict, non_target_results_dict):

    metrics = {}
    # Iterate over each condition
    for condition in target_results_dict.keys():
        # Ensure the condition exists in all input dictionaries
        if condition not in distractor_results_dict or condition not in non_target_results_dict:
            print(f"Condition '{condition}' is missing in one of the dictionaries.")
            continue

        # Extract overall average power for each epoch type within the current condition
        target_vals = target_results_dict[condition]
        distractor_vals = distractor_results_dict[condition]
        non_target_vals = non_target_results_dict[condition]

        # normalize powers for plotting: (range 0-1)
        target_overall_avg_powers = [epoch['overall_avg_power'] for epoch in target_vals]
        distractor_overall_avg_powers = [epoch['overall_avg_power'] for epoch in distractor_vals]
        non_target_avg_powers = [epoch['overall_avg_power'] for epoch in non_target_vals]
        all_avg_powers = target_overall_avg_powers + distractor_overall_avg_powers + non_target_avg_powers
        # log transformed values:
        # Log transform each value within the lists (element-wise)
        target_log_avg_powers = [np.log1p(value) for value in target_overall_avg_powers]
        distractor_log_avg_powers = [np.log1p(value) for value in distractor_overall_avg_powers]
        non_target_log_avg_powers = [np.log1p(value) for value in non_target_avg_powers]
        log_all_avg_powers = target_log_avg_powers + distractor_log_avg_powers + non_target_log_avg_powers  # Concatenate lists
        max_power = np.max(log_all_avg_powers)
        min_power = np.min(log_all_avg_powers)

        def normalize_values(values, min_power, max_power):
            return [(val - min_power) / (max_power - min_power) for val in values]

        # Normalize average powers for each epoch type
        normalized_target_avg_powers = normalize_values([value for value in target_log_avg_powers], min_power, max_power)
        normalized_distractor_avg_powers = normalize_values([value for value in distractor_log_avg_powers], min_power, max_power)
        normalized_non_target_avg_powers = normalize_values([value for value in non_target_log_avg_powers], min_power, max_power)

        # calculate Cliff's delta:
        # quantifies the amount of difference between two groups of observations beyond p-values interpretation.
        target_distractor_delta = cliffs_delta(target_log_avg_powers, distractor_overall_avg_powers)
        tagret_non_target_delta = cliffs_delta(target_log_avg_powers, non_target_log_avg_powers)
        distractor_non_target_delta = cliffs_delta(distractor_log_avg_powers, non_target_log_avg_powers)

        # Data for bar plot
        epoch_types = ['Target', 'Distractor', 'Non-Target']
        # Test for normality
        alpha = 0.05
        is_normal = True
        for avg_power, label in zip(
                [target_log_avg_powers, distractor_log_avg_powers, non_target_log_avg_powers],
                ["Target", "Distractor", "Non-Target"]
        ):
            stat, p = shapiro(avg_power)
            print(f"{condition} - {label} Group: Shapiro-Wilk p-value = {p:.4f}")
            if p < alpha:
                print(f"{condition} - {label} Group: Data is not normal (reject H0)")
                is_normal = False
            else:
                print(f"{condition} - {label} Group: Data is likely normal (fail to reject H0)")

        # Select appropriate test based on normality
        significance_labels = {}
        if is_normal:
            # Test for Homogeneity of Variances
            levene_stat, levene_p = levene(target_log_avg_powers, distractor_log_avg_powers,
                                           non_target_log_avg_powers)
            print(f"{condition} - Levene's test p-value: {levene_p}")

            if levene_p < alpha:
                # Variances are unequal: Use Welch's ANOVA
                welch_result = pg.welch_anova(data=df, dv='overall_avg_power', between='epoch_type')
                welch_p_val = welch_result['p-unc'][0]
                eta_squared = welch_result['eta-square'][0]
                print(f"{condition} - Welch's ANOVA p-value: {welch_p_val}")
                print(f"{condition} - Welch's ANOVA Effect Size (eta-squared): {eta_squared}")

                metrics[f'{condition}_welch'] = {
                    'Welch ANOVA p-value': welch_p_val,
                    'Effect Size (eta-squared)': eta_squared,
                }

                if welch_p_val < alpha:
                    posthoc = sp.posthoc_ttest(df, val_col='overall_avg_power', group_col='epoch_type',
                                               p_adjust='bonferroni')
                    significance_labels = {
                        ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                        ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
                        ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target'])}
            else:
                # Variances are equal: Use standard ANOVA
                anova_result = sm.stats.anova_lm(sm.OLS.from_formula("overall_avg_power ~ C(epoch_type)", data=df).fit(),
                                                 typ=2)
                anova_p_val = anova_result['PR(>F)']['C(epoch_type)']
                total_ss = anova_result['sum_sq'].sum()
                eta_squared = anova_result['sum_sq']['C(epoch_type)'] / total_ss
                print(f"{condition} - ANOVA p-value: {anova_p_val}")
                print(f"{condition} - ANOVA Effect Size (eta-squared): {eta_squared}")

                metrics[f'{condition}_anova'] = {
                    'ANOVA p-value': anova_p_val,
                    'Effect Size (eta-squared)': eta_squared,
                }
                # If ANOVA is significant, perform pairwise t-tests with Bonferroni correction
                if anova_p_val < alpha:
                    df = pd.DataFrame({
                        'overall_log_power': target_log_avg_powers + distractor_log_avg_powers + non_target_log_avg_powers,
                        'epoch_type': ['Target'] * len(target_log_avg_powers) +
                                      ['Distractor'] * len(distractor_log_avg_powers) +
                                      ['Non-Target'] * len(non_target_log_avg_powers)
                    })
                    normalized_df = pd.DataFrame({
                        'normalized_avg_power': normalized_target_avg_powers + normalized_distractor_avg_powers + normalized_non_target_avg_powers,
                        'epoch_type': ['Target'] * len(normalized_target_avg_powers) +
                                      ['Distractor'] * len(normalized_distractor_avg_powers) +
                                      ['Non-Target'] * len(normalized_non_target_avg_powers)
                    })
                    posthoc = sp.posthoc_ttest(df, val_col='overall_log_power', group_col='epoch_type',
                                               p_adjust='bonferroni')
                    significance_labels = {
                        ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                        ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
                        ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target'])}

                    metrics[f'{sub_input}_{condition}']['T-test Target vs. Distractor'] = posthoc.loc[
                        'Target', 'Distractor']
                    metrics[f'{sub_input}_{condition}']['T-test Target vs. Non-Target'] = posthoc.loc[
                        'Target', 'Non-Target']
                    metrics[f'{sub_input}_{condition}']['T-test Distractor vs. Non-Target'] = posthoc.loc[
                        'Distractor', 'Non-Target']


        else:
            # Use Kruskal-Wallis for non-normal data
            stat, p = levene(target_log_avg_powers, distractor_log_avg_powers, non_target_log_avg_powers)
            kruskal_h_val, kruskal_p_val = kruskal(target_log_avg_powers, distractor_log_avg_powers, non_target_log_avg_powers)
            print(f"{condition} - Kruskal-Wallis p-value: {kruskal_p_val}")

            # Effect size calculation
            def kruskal_eta_squared(H, N):
                return (H - (3 - 1)) / (N - (3 - 1))  # Adjust for 3 groups

            N = len(target_log_avg_powers) + len(distractor_log_avg_powers) + len(non_target_log_avg_powers)
            eta_squared = kruskal_eta_squared(kruskal_h_val, N)
            print(f"Kruskal-Wallis Effect Size (eta-squared): {eta_squared:.4f}")

            # Store the results
            metrics[f'{sub_input}_{condition}'] = {
                'Kruskal-Wallis H': kruskal_h_val,
                'Kruskal-Wallis p-value': kruskal_p_val,
                'Effect Size (eta-squared)': eta_squared,
                'Levene statistic (variance)': [stat, p]
            }

            # If Kruskal-Wallis is significant, use Dunn's post-hoc test
            if kruskal_p_val < alpha:
                df = pd.DataFrame({
                    'overall_log_power': target_log_avg_powers + distractor_log_avg_powers + non_target_avg_powers,
                    'epoch_type': ['Target'] * len(target_log_avg_powers) +
                                  ['Distractor'] * len(distractor_log_avg_powers) +
                                  ['Non-Target'] * len(non_target_log_avg_powers)
                })
                normalized_df = pd.DataFrame({
                    'normalized_avg_power': normalized_target_avg_powers + normalized_distractor_avg_powers + normalized_non_target_avg_powers,
                    'epoch_type': ['Target'] * len(normalized_target_avg_powers) +
                                  ['Distractor'] * len(normalized_distractor_avg_powers) +
                                  ['Non-Target'] * len(normalized_non_target_avg_powers)
                })
                posthoc = sp.posthoc_dunn(df, val_col='overall_log_power', group_col='epoch_type',
                                          p_adjust='bonferroni')

                significance_labels = {
                    ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                    ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
                    ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target'])}

                metrics[f'{sub_input}_{condition}']['Dunn Posthoc Target vs. Distractor'] = posthoc.loc[
                    'Target', 'Distractor']
                metrics[f'{sub_input}_{condition}']['Dunn Posthoc Target vs. Non-Target'] = posthoc.loc[
                    'Target', 'Non-Target']
                metrics[f'{sub_input}_{condition}']['Dunn Posthoc Distractor vs. Non-Target'] = posthoc.loc[
                    'Distractor', 'Non-Target']

                # Store the Cliff's Delta values in the metrics DataFrame
                metrics[f'{sub_input}_{condition}']['Cliff Delta Target vs Distractor'] = target_distractor_delta
                metrics[f'{sub_input}_{condition}']['Cliff Delta Target vs Non-Target'] = tagret_non_target_delta
                metrics[f'{sub_input}_{condition}']['Cliff Delta Distractor vs Non-Target'] = distractor_non_target_delta


        # Convert metrics to a DataFrame for saving
        metrics_df = pd.DataFrame.from_dict(metrics, orient='index')
        metrics_df.to_csv(results_path/f'{sub_input}_avg_power_metrics.csv')
        # Plot
        plt.figure(figsize=(12, 10))
        colors = sns.color_palette('flare', n_colors=3)
        ax = sns.violinplot(data=normalized_df, x='epoch_type', y='normalized_avg_power', hue='epoch_type', palette=colors, legend=False)
        # Optionally add a strip plot to show individual data points
        # sns.stripplot(data=normalized_df, x="epoch_type", y="normalized_avg_power", color="black", alpha=0.5, jitter=True)
        # add_bootstrapped_ci(normalized_df, 'epoch_type', 'normalized_avg_power', ax, colors)

        plt.legend(title=f'Sample Size: {len(normalized_target_avg_powers)}')
        plt.title("Violin Plot of Overall Average Power by Epoch Type")
        plt.xlabel("Epoch Type")
        plt.ylabel("Overall Average Power (W)")

        # Add significance labels
        y_max = max(normalized_target_avg_powers + normalized_distractor_avg_powers + normalized_non_target_avg_powers) # initializes the starting height for the significance line.
        y_offset = y_max * 0.01  # calculates how far above the highest bar the significance lines should start
        pairs = [('Target', 'Distractor'), ('Target', 'Non-Target'), ('Distractor', 'Non-Target')] # list of group pairs for which significance is being tested.
        for (group1, group2) in pairs:
            x1, x2 = epoch_types.index(group1), epoch_types.index(group2)  # Retrieves the positions (indices) of the two groups on the x-axis.
            y = y_max + y_offset
            label = significance_labels.get((group1, group2), "ns")  # Default to "ns" if not in significance_labels
            if label != 'ns':
                plt.plot([x1, x2], [y, y], color='black', linestyle='solid')
                plt.text((x1 + x2) / 2, y + y_offset, label, ha='center', va='bottom', fontsize=12) # Draws a horizontal line between the two x-positions (x1 and x2) at the current height y
                y_max += y_offset   # Move up for the next annotation

        plt.savefig(class_figs/f'{condition}_avg_powers_per_epoch_type.png')
        plt.close()



def save_subject_results(sub_input, target_results_dict, distractor_results_dict, non_target_results_dict):
    # Combine all results into a dictionary for easy saving
    results_to_save = {
        'Target': target_results_dict,
        'Distractor': distractor_results_dict,
        'Non-Target': non_target_results_dict,
    }

    # Create directory for the subject's results if it doesn't exist
    subject_dir = os.path.join(subject_results_dir)
    os.makedirs(subject_results_dir, exist_ok=True)

    # Save the dictionary as a .pkl file
    with open(os.path.join(subject_dir, f'{sub_input}_results.pkl'), 'wb') as f:
        pickle.dump(results_to_save, f)
    print(f"Results saved for subject: {sub_input}")

if __name__ == '__main__':
    # matplotlib.use('TkAgg')
    sub_input = input("Give sub number as subn (n for number): ")
    # conditions: a1, a2, e1 or e2
    conditions = input('Please provide condition (exp. EEG): ')  # choose a condition of interest for processing
    condition_list = [condition.strip() for condition in conditions.split(',')]
    cm = 1 / 2.54
    # 0. LOAD THE DATA
    default_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data')
    data_dir = default_dir / 'eeg' / 'raw'  # to get the raw EEG files (which will we 'convert' to EMG
    sub_dir = data_dir / sub_input
    subject_results_dir = default_dir / 'emg' / 'subject_results'
    emg_dir = default_dir / 'emg' / sub_input  # creating a folder dedicated to EMG files we will create
    results_path = emg_dir / 'preprocessed' / 'results'  # where results will be saved
    fig_path = emg_dir / 'preprocessed' / 'figures' # where figures from pre-processing and classification will be stored
    erp_path = fig_path / 'ERPs'
    z_figs = fig_path / 'z_distributions'
    class_figs = fig_path / 'classifications'
    df_path = default_dir / 'performance' / sub_input / 'tables'  # csv tables containing important information from vmrk files
    json_path = default_dir / 'misc'
    psd_path = fig_path / 'spectral_power'
    fif_path = emg_dir / 'fif files'  # pre-processed .eeg files of the EMG data will be stored
    combined_epochs = fif_path / 'combined_epochs'
    for folder in sub_dir, fif_path, results_path, fig_path, erp_path, z_figs, class_figs, df_path, combined_epochs, psd_path:
        if not os.path.isdir(folder):
            os.makedirs(folder)
    # Load necessary JSON files
    with open(json_path / "preproc_config.json") as file:
        cfg = json.load(file)
    with open(json_path / "electrode_names.json") as file:
        mapping = json.load(file)
    with open(json_path / 'eeg_events.json') as file:
        markers_dict = json.load(file)

    s1_events = markers_dict['s1_events'] # stimulus 1 markers
    s2_events = markers_dict['s2_events']  # stimulus 2 markers
    response_events = markers_dict['response_events']  # response markers

    # mapping events needed for creating epochs with only target-number events, for target, distractor AND response epochs
    mapping_path = json_path / 'events_mapping.json'
    with open(mapping_path, 'r') as file:
        events_mapping = json.load(file)
    s1_mapping = events_mapping[0]
    s2_mapping = events_mapping[1]
    response_mapping = events_mapping[2]

    # Get Target Block information
    csv_path = default_dir / 'params' / 'block_sequences' / f'{sub_input}.csv'
    csv = pd.read_csv(csv_path)

    # get baseline eeg: only once
    for files in sub_dir.iterdir():
        if files.is_file and 'baseline.vhdr' in files.name:
            baseline = mne.io.read_raw_brainvision(files, preload=True)
            baseline.resample(sfreq=500)

    baseline.rename_channels(mapping)
    baseline.set_montage('standard_1020')
    baseline = baseline.pick_channels(['A2', 'M2', 'A1'])  # select EMG-relevant files
    baseline.set_eeg_reference(ref_channels=['A1'])  # set correct reference channel
    baseline = mne.set_bipolar_reference(baseline, anode='A2', cathode='M2', ch_name='EMG_bipolar')  # change ref to EMG bipolar
    baseline.drop_channels(['A1'])  # drop reference channel (don't need to see it)

    # pre-process baseline:
    baseline_filt = filter_emg(baseline)
    baseline_rect = baseline_filt.copy()
    baseline_rectified = rectify(baseline_rect)
    # get mean and std:
    baseline_mean = baseline_rectified.get_data().mean(axis=1)  # Mean across time
    # Calculate the number of samples in the recording
    '''
    - epoch_length_s: Total length of each epoch in seconds.
    - tmin: Start of the epoch in seconds relative to each synthetic event.
    - tmax: End of the epoch in seconds relative to each synthetic event.
    '''
    n_samples = baseline_rectified.n_times
    duration_s = n_samples / samplerate
    epoch_length_s = 1.1
    tmin = -0.2
    tmax = 0.9
    # Calculate the step size in samples for each epoch (based on the epoch length)
    step_size = int(epoch_length_s * samplerate)
    # Generate synthetic events spaced at regular intervals across the continuous data
    # The first column is the sample index, second is filler (e.g., 0), third is the event ID (e.g., 1)
    events = np.array([[i, 0, 1] for i in range(0, n_samples - step_size, step_size)])

    # Create epochs based on these synthetic events
    epochs_baseline = mne.Epochs(baseline_rectified, events, event_id={'arbitrary': 1}, tmin=tmin, tmax=tmax, baseline=None, preload=True)

    # baseline normalization
    baseline_data, baseline_derivative, baseline_z_scores, baseline_var, baseline_rms = baseline_normalization(
        epochs_baseline, tmin=0.2, tmax=0.9)

    target_header_files_list = []

    ### Get target header files for all participants
    for index, condition in enumerate(condition_list):
        header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
        filtered_files = [file for file in header_files if condition in file]
        if filtered_files:
            target_header_files_list.append(filtered_files)

    all_target_response_epochs_dict = {cond: [] for cond in condition_list}
    all_target_no_response_epochs_dict = {cond: [] for cond in condition_list}
    all_distractor_response_epochs_dict = {cond: [] for cond in condition_list}
    all_distractor_no_response_epochs_dict = {cond: [] for cond in condition_list}

    # separate non targets, keep target's non-targets
    all_non_target_stim_epochs_dict = {cond: [] for cond in condition_list}

    all_invalid_non_target_epochs_dict = {cond: [] for cond in condition_list}
    all_invalid_target_epochs_dict = {cond: [] for cond in condition_list}
    all_invalid_distractor_epochs_dict = {cond: [] for cond in condition_list}
    # so far, so good.
    for condition, sublist in zip(condition_list, target_header_files_list):
        target_stream, distractor_stream, target_blocks, axis, target_mapping, distractor_mapping = get_target_blocks()
        for index, emg_file in enumerate(sublist):
            # Initialize an empty list for each condition in all dictionaries
            full_path = os.path.join(sub_dir, emg_file)
            emg = mne.io.read_raw_brainvision(full_path, preload=True)
            emg.resample(sfreq=500)
            # Set montage for file:
            emg.rename_channels(mapping)
            emg.set_montage('standard_1020')


            emg = emg.pick_channels(['A2', 'M2', 'A1'])  # select EMG-relevant files
            emg.set_eeg_reference(ref_channels=['A1'])  # set correct reference channel
            emg = mne.set_bipolar_reference(emg, anode='A2', cathode='M2',
                                            ch_name='EMG_bipolar')  # change ref to EMG bipolar
            emg.drop_channels(['A1'])  # drop reference channel (don't need to see it)
            # Filter and rectify the EMG data
            emg_filt = filter_emg(emg)
            emg_rect = emg_filt.copy()
            emg_rectified = rectify(emg_rect)
            # Apply baseline correction to continuous EMG data
            target_emg_rectified = rectify(emg_filt)
            target_emg_rectified._data -= baseline_mean[:, None]
            distractor_emg_rectified = rectify(emg_filt)
            distractor_emg_rectified._data -= baseline_mean[:, None]
            non_target_emg_rectified = rectify(emg_filt)
            non_target_emg_rectified._data -= baseline_mean[:, None]

            target_block = [target_block for target_block in target_blocks.values[index]]
            target_events, distractor_events = define_events(target_stream)
            # define events for each target block:
            combined_events = {**distractor_events, **target_events}
            # Get events for EMG analysis, based on filtered target and distractor, and response events
            targets_emg_events = create_events(target_events, target_mapping)
            distractors_emg_events = create_events(distractor_events, distractor_mapping)
            responses_emg_events = create_response_events(response_events)

            # Create non-target Stimuli Epochs: from target stream
            non_target_events = baseline_events(target_events, target_mapping)

            # categorize events based on epoch type and response type (response, no response, invalid)
            target_response_events, target_no_response_events, invalid_target_response_events, \
            distractor_response_events, distractor_no_response_events, invalid_distractor_response_events, \
            non_target_stimulus_events, invalid_non_target_response_events, response_only_epochs = \
                categorize_events(targets_emg_events, distractors_emg_events, non_target_events, responses_emg_events)
            # properly define types of invalid events:
            invalid_non_target_events = [np.array(event[:3], dtype=int) for event in invalid_non_target_response_events]
            invalid_target_events = [np.array(event[:3], dtype=int) for event in invalid_target_response_events]
            invalid_distractor_events = [np.array(event[:3], dtype=int) for event in invalid_distractor_response_events]

            # get epochs of each response type, based on the events created above:
            # Target stimuli with responses:
            target_response_epochs, target_response_data = get_data(target_response_events, target_events,
                                                                    target_emg_rectified, target_response_events,
                                                                    target='target_with_responses', tmin=-0.2, tmax=0.9,
                                                                    baseline=(-0.2, 0.0))
            # Target stimuli without responses:
            target_no_response_epochs, target_no_responses_data = get_data(target_no_response_events, target_events,
                                                                            target_emg_rectified,
                                                                            target_no_response_events,
                                                                            target='target_without_responses',
                                                                            tmin=-0.2,
                                                                            tmax=0.9, baseline=(-0.2, 0.0))

            # Distractor stimuli with responses:
            distractor_response_epochs, distractor_responses_data = get_data(distractor_response_events,
                                                                              distractor_events,
                                                                              distractor_emg_rectified,
                                                                              distractor_response_events,
                                                                              target='distractor_with_responses',
                                                                              tmin=-0.2,
                                                                              tmax=0.9, baseline=(-0.2, 0.0))

            # Distractor stimuli without responses:
            distractor_no_response_epochs, distractor_no_responses_data = get_data(distractor_no_response_events,
                                                                                    distractor_events,
                                                                                    distractor_emg_rectified,
                                                                                    distractor_no_response_events,
                                                                                    target='distractor_without_responses',
                                                                                    tmin=-0.2, tmax=0.9,
                                                                                    baseline=(-0.2, 0.0))

            # Non-target stimuli: Target
            non_target_stim_epochs, non_target_target_stim_data = get_data(non_target_stimulus_events, combined_events,
                                                                    emg_rect,
                                                                    non_target_stimulus_events,
                                                                    target='non_target_stimuli',
                                                                    tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))


            # Invalid non-target responses:
            invalid_non_target_epochs, invalid_non_target_data = get_data(invalid_non_target_events, combined_events,
                                                                          non_target_emg_rectified,
                                                                          invalid_non_target_events,
                                                                          target='invalid_non_target_responses', tmin=-0.2,
                                                                          tmax=0.9,
                                                                          baseline=(-0.2, 0.0))

            # Invalid target responses:
            invalid_target_epochs, invalid_target_data = get_data(invalid_target_events, combined_events,
                                                                  non_target_emg_rectified, invalid_target_events,
                                                                  target='invalid_target_responses', tmin=-0.2,
                                                                  tmax=0.9,
                                                                  baseline=(-0.2, 0.0))

            # Invalid distractor responses:
            invalid_distractor_epochs, invalid_distractor_data = get_data(invalid_distractor_events, combined_events,
                                                                          non_target_emg_rectified,
                                                                          invalid_distractor_events,
                                                                          target='invalid_distractor_responses',
                                                                          tmin=-0.2,
                                                                          tmax=0.9, baseline=(-0.2, 0.0))

            all_target_response_epochs_dict[condition].append(target_response_epochs)
            all_target_no_response_epochs_dict[condition].append(target_no_response_epochs)
            all_distractor_response_epochs_dict[condition].append(distractor_response_epochs)
            all_distractor_no_response_epochs_dict[condition].append(distractor_no_response_epochs)
            all_non_target_stim_epochs_dict[condition].append(non_target_stim_epochs)
            all_invalid_non_target_epochs_dict[condition].append(invalid_non_target_epochs)
            all_invalid_target_epochs_dict[condition].append(invalid_target_epochs)
            all_invalid_distractor_epochs_dict[condition].append(invalid_distractor_epochs)


            def filter_epochs_dict(all_epochs_dict):
                """Filter out None values and invalid Epochs from each condition in all_epochs_dict."""
                for condition, epochs_list in all_epochs_dict.items():
                    # Filter out None values and keep only valid, non-empty Epochs
                    all_epochs_dict[condition] = [epoch for epoch in epochs_list if
                                                  isinstance(epoch, mne.Epochs) and len(epoch) > 0]
                return all_epochs_dict


            # Apply the filter function to each of the all_name_epochs dictionaries
            all_target_response_epochs_dict = filter_epochs_dict(all_target_response_epochs_dict)
            all_target_no_response_epochs_dict = filter_epochs_dict(all_target_no_response_epochs_dict)
            all_distractor_response_epochs_dict = filter_epochs_dict(all_distractor_response_epochs_dict)
            all_distractor_no_response_epochs_dict = filter_epochs_dict(all_distractor_no_response_epochs_dict)
            all_non_target_stim_epochs_dict = filter_epochs_dict(all_non_target_stim_epochs_dict)
            all_invalid_non_target_epochs_dict = filter_epochs_dict(all_invalid_non_target_epochs_dict)
            all_invalid_target_epochs_dict = filter_epochs_dict(all_invalid_target_epochs_dict)
            all_invalid_distractor_epochs_dict = filter_epochs_dict(all_invalid_distractor_epochs_dict)

    combined_invalid_events_dict = {cond: [] for cond in condition_list}
    # here getting all invalid responses events into combined_invalid_events dictionary
    for condition, epoch_list in all_invalid_non_target_epochs_dict.items():
        epochs = all_invalid_non_target_epochs_dict[condition]
        for epoch in epochs:
            if epoch is not None and isinstance(epoch, mne.Epochs):
                events = epoch.events
                combined_invalid_events_dict[condition].append(events)
    for condition, epoch_list in all_invalid_target_epochs_dict.items():
        epochs = all_invalid_target_epochs_dict[condition]
        for epoch in epochs:
            if epoch is not None and isinstance(epoch, mne.Epochs):
                events = epoch.events
                combined_invalid_events_dict[condition].append(events)
    for condition, epoch_list in all_invalid_distractor_epochs_dict.items():
        epochs = all_invalid_distractor_epochs_dict[condition]
        for epoch in epochs:
            if epoch is not None and isinstance(epoch, mne.Epochs):
                events = epoch.events
                combined_invalid_events_dict[condition].append(events)

    # Process each condition's events, convert to DataFrame, and sort by time
    for condition, events_list in combined_invalid_events_dict.items():
        if events_list:  # Check if there are events to concatenate
            concatenated_events = np.concatenate(events_list, axis=0)
            # Convert to DataFrame and process
            df = pd.DataFrame(concatenated_events, columns=['Timepoints', 'Type', 'Stimulus'])
            df['Type'] = 'Invalid Response'
            df['Timepoints'] = (df['Timepoints'].astype(int) / samplerate)  # Convert samples to time
            df['Stimulus'] = df['Stimulus'].astype(int)
            df = df.sort_values(by='Timepoints').reset_index(drop=True)  # Sort by time and reset index
            combined_invalid_events_dict[condition] = df  # Store the DataFrame back in the dictionary
        else:
            # If no events, store an empty DataFrame
            combined_invalid_events_dict[condition] = pd.DataFrame(columns=['Timepoints', 'Type', 'Stimulus'])

    # Process each type of epoch list and store results using combine_all_epochs
    combined_target_response_epochs_dict, target_response_data_dict, target_response_events_dict = combine_all_epochs(all_target_response_epochs_dict, condition_list, label='target_response')
    combined_target_no_response_epochs_dict, target_no_response_data_dict, target_no_response_events_dict = combine_all_epochs(all_target_no_response_epochs_dict, condition_list, label='target_no_response')

    combined_distractor_response_epochs_dict, distractor_response_data_dict, distractor_response_events_dict = combine_all_epochs(all_distractor_response_epochs_dict, condition_list, label='distractor_response')
    combined_distractor_no_response_epochs_dict, distractor_no_response_data_dict, distractor_no_response_events_dict = combine_all_epochs(all_distractor_no_response_epochs_dict, condition_list, label='distractor_no_response')

    # newly separated non-targets from target stream:
    combined_non_target_stim_epochs_dict, non_target_stim_data_dict, non_target_stim_events_dict = combine_all_epochs(all_non_target_stim_epochs_dict, condition_list, label='non_target')

    combined_invalid_non_target_epochs_dict, invalid_non_target_data_dict, invalid_non_target_events_dict = combine_all_epochs(all_invalid_non_target_epochs_dict, condition_list, label='invalid_non_target')
    combined_invalid_target_epochs_dict, invalid_target_data_dict, invalid_target_events_dict = combine_all_epochs(all_invalid_target_epochs_dict, condition_list, label='invalid_target')
    combined_invalid_distractor_epochs_dict, invalid_distractor_data_dict, invalid_distractor_events_dict = combine_all_epochs(all_invalid_distractor_epochs_dict, condition_list, label='invalid_distractor')

    # Placeholder for the sampled dictionary: reducing non-target dicts lengths to match that of distractor events
    combined_non_target_stim_sampled_epochs_dict = {}
    combined_target_response_stim_sampled_epochs_dict = {}

    # Iterate through each condition
    for condition in combined_distractor_no_response_epochs_dict.keys():
        # Get the number of events for the distractor no-response epochs in the current condition
        target_event_count = len(combined_distractor_no_response_epochs_dict[condition][0].events)

        # Retrieve the non-target stim epochs for the current condition
        non_target_epochs = combined_non_target_stim_epochs_dict[condition][0]
        target_response_epochs = combined_target_response_epochs_dict[condition][0]

        # Randomly sample epochs
        np.random.seed(42)
        non_target_sampled_epochs = non_target_epochs[np.random.choice(len(non_target_epochs), target_event_count, replace=False)]
        target_sampled_epochs = target_response_epochs[np.random.choice(len(target_response_epochs), target_event_count, replace=False)]
        # Store in the new dictionary
        combined_non_target_stim_sampled_epochs_dict[condition] = [non_target_sampled_epochs]
        combined_target_response_stim_sampled_epochs_dict[condition] = [target_sampled_epochs]

    # Define an empty dictionary to store TFA results
    tfa_results_dict = {
        'target_response': {},
        'target_no_response': {},
        'distractor_response': {},
        'distractor_no_response': {},
        'non_target_stim': {},
        'invalid_non_target': {},
        'invalid_target': {},
        'invalid_distractor': {}
    }

    # Run TFA for each combined epoch type within the dictionaries
    for condition, combined_target_response_epochs in combined_target_response_stim_sampled_epochs_dict.items():
        combined_epochs = combined_target_response_epochs_dict[condition][0]
        if combined_epochs is not None and len(combined_epochs) > 0:
            response_power = tfa_heatmap(combined_epochs, target='target_response')
            tfa_results_dict['target_response'][condition] = response_power

    for condition, combined_distractor_no_response_epochs in combined_distractor_no_response_epochs_dict.items():
        if combined_distractor_no_response_epochs and len(combined_distractor_no_response_epochs[0]) > 0:
            combined_epochs = combined_distractor_no_response_epochs_dict[condition][0]
            if combined_epochs is not None and len(combined_epochs) > 0:
                distractor_power = tfa_heatmap(combined_epochs, target='distractor_no_response')
                tfa_results_dict['distractor_no_response'][condition] = distractor_power

    for condition, combined_non_target_stim_epochs in combined_non_target_stim_sampled_epochs_dict.items():
        if combined_non_target_stim_epochs and len(combined_non_target_stim_epochs[0]) > 0:
            combined_epochs = combined_non_target_stim_sampled_epochs_dict[condition][0]
            if combined_epochs is not None and len(combined_epochs) > 0:
                non_target_target_power = tfa_heatmap(combined_epochs, target='non_target_stim')
                tfa_results_dict['non_target_stim'][condition] = non_target_target_power

        bands = {
            'band_1_10': (1, 10),
            'band_11_20': (11, 20),
            'band_21_30': (21, 30),
            'band_31_40': (31, 40),
            'band_41_50': (41, 50),
            'band_51_60': (51, 60),
            'band_61_70': (61, 70),
            'band_71_80': (71, 80),
            'band_81_90': (81, 90),
            'band_91_100': (91, 100),
            'band_101_110': (101, 110),
            'band_111_120': (111, 120),
            'band_121_130': (121, 130),
            'band_131_140': (131, 140),
            'band_141_150': (141, 150)
        }

        target_results_dict = epochs_vals(combined_target_response_stim_sampled_epochs_dict)
        distractor_results_dict = epochs_vals(combined_distractor_no_response_epochs_dict)
        non_target_results_dict = epochs_vals(combined_non_target_stim_sampled_epochs_dict)

        # Filter outliers
        filtered_results_dict = filter_outliers_epochs(target_results_dict)
        filtered_distractor_results_dict = filter_outliers_epochs(distractor_results_dict)
        filtered_non_target_results_dict = filter_outliers_epochs(non_target_results_dict)

        filtered_tfa_results_dict = {'Target': tfa_results_dict['target_response'],
                            'Distractor': tfa_results_dict['distractor_no_response'],
                            'Non-Target': tfa_results_dict['non_target_stim']}


        with open(os.path.join(subject_results_dir/ 'tfa' , f'tfa_dict_{sub_input}.pkl'), 'wb') as f:
            pickle.dump(filtered_tfa_results_dict, f)

        plot_overall_avg_power_bar(target_results_dict, distractor_results_dict, non_target_results_dict)

        # Example usage for each epoch type
        plot_dominant_frequency_distributions(target_results_dict, distractor_results_dict, non_target_results_dict)

        bin_edges = np.linspace(1, 150, 1)  # Adjust as needed
        plot_dominant_frequency_counts(filtered_results_dict, filtered_distractor_results_dict, filtered_non_target_results_dict)

        plot_dominant_band_distributions(filtered_results_dict, filtered_distractor_results_dict, filtered_non_target_results_dict)

        save_subject_results(sub_input, filtered_results_dict, filtered_distractor_results_dict, filtered_non_target_results_dict)


