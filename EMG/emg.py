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


def categorize_events(target_events, distractor_events, non_target_events_target, non_target_events_distractor,
                          responses_emg_events, sampling_rate=samplerate):
        # Initialize the lists for categorization
        target_response_events = []
        target_no_response_events = []
        distractor_response_events = []
        distractor_no_response_events = []
        non_target_target_stimulus_events = []
        non_target_distractor_stimulus_events = []
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
        process_events(non_target_events_distractor, 'non_target_distractor', [], non_target_distractor_stimulus_events,
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
            non_target_distractor_stimulus_events,
            invalid_non_target_response_events,
            response_only_epochs)


# get emg epochs, around the target and distractor events:
def epochs(events_dict, emg_rectified, events_emg, target='', tmin=0, tmax=0, baseline=None):
    event_ids = {key: val for key, val in events_dict.items() if any(event[2] == val for event in events_emg)}
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
                    combined_epochs_path / f'{sub_input}_condition_{condition}_combined_{label}_epochs-epo.fif',
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


def run_tfa(tfa_results_dict, combined_epochs_dict, epoch_label='', mean_label='', median_label='', std_label=''):
    # Run TFA for each combined epoch type within the dictionaries
    for condition, combined_epochs in combined_epochs_dict.items():
        combined_epochs = combined_epochs_dict[condition][0]
        if combined_epochs is not None and len(combined_epochs) > 0:
            power = tfa_heatmap(combined_epochs, target=f'{epoch_label}')
            power_mean = np.mean(power.data)
            power_std = np.std(power.data)
            power_median = np.median(power.data)
            if condition not in tfa_results_dict[f'{epoch_label}']:
                tfa_results_dict[f'{epoch_label}'][condition] = {}
                tfa_results_dict[f'{epoch_label}'][condition]['power'] = power
                tfa_results_dict[f'{epoch_label}'][condition]['descriptive_statistics'] = {
                    f'{mean_label}': power_mean,
                    f'{median_label}': power_median,
                    f'{std_label}': power_std}
    return tfa_results_dict

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
    fixed_ticks = np.arange(0, 160, 10)  # Example fixed y-tick values

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


def save_subject_results(sub_input, filtered_target_results_dict, filtered_distractor_results_dict, filtered_non_target_target_results_dict,
                         filtered_non_target_distractor_results_dict):
    # Combine all results into a dictionary for easy saving
    results_to_save = {
        'Target': filtered_target_results_dict,
        'Distractor': filtered_distractor_results_dict,
        'Non-Target Target': filtered_non_target_target_results_dict,
        'Non-Target Distractor': filtered_non_target_distractor_results_dict
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
    combined_epochs_path = fif_path / 'combined_epochs'
    for folder in sub_dir, fif_path, results_path, fig_path, erp_path, z_figs, class_figs, df_path, combined_epochs_path, psd_path:
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
    all_non_target_target_stim_epochs_dict = {cond: [] for cond in condition_list}
    all_non_target_distractor_stim_epochs_dict = {cond: [] for cond in condition_list}

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
            non_target_target_events = baseline_events(target_events, target_mapping)
            non_target_distractor_events = baseline_events(distractor_events, distractor_mapping)

            # categorize events based on epoch type and response type (response, no response, invalid)
            target_response_events, target_no_response_events, invalid_target_response_events, \
            distractor_response_events, distractor_no_response_events, invalid_distractor_response_events, \
            non_target_target_stimulus_events, non_target_distractor_stimulus_events, invalid_non_target_response_events, response_only_epochs = \
                categorize_events(targets_emg_events, distractors_emg_events, non_target_target_events, non_target_distractor_events,responses_emg_events)
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
            non_target_target_stim_epochs, non_target_target_stim_data = get_data(non_target_target_stimulus_events, combined_events,
                                                                    emg_rect,
                                                                    non_target_target_stimulus_events,
                                                                    target='non_target_target_stimuli',
                                                                    tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))

            # Non-target stimuli: Distractor
            non_target_distractor_stim_epochs, non_target_distractor_stim_data = get_data(non_target_distractor_stimulus_events,
                                                                                  combined_events,
                                                                                  emg_rect,
                                                                                  non_target_distractor_stimulus_events,
                                                                                  target='non_target_distractor_stimuli',
                                                                                  tmin=-0.2, tmax=0.9,
                                                                                  baseline=(-0.2, 0.0))

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
            all_non_target_target_stim_epochs_dict[condition].append(non_target_target_stim_epochs)
            all_non_target_distractor_stim_epochs_dict[condition].append(non_target_distractor_stim_epochs)
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
            all_non_target_target_stim_epochs_dict = filter_epochs_dict(all_non_target_target_stim_epochs_dict)
            all_non_target_distractor_stim_epochs_dict = filter_epochs_dict(all_non_target_distractor_stim_epochs_dict)
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
    combined_non_target_target_stim_epochs_dict, non_target_target_stim_data_dict, non_target_target_stim_events_dict = combine_all_epochs(all_non_target_target_stim_epochs_dict, condition_list, label='non_target_target')
    # for distractor stream:
    combined_non_target_distractor_stim_epochs_dict, non_target_distractor_stim_data_dict, non_target_distractor_stim_events_dict = combine_all_epochs(all_non_target_distractor_stim_epochs_dict, condition_list, label='non_target_distractor')

    combined_invalid_non_target_epochs_dict, invalid_non_target_data_dict, invalid_non_target_events_dict = combine_all_epochs(all_invalid_non_target_epochs_dict, condition_list, label='invalid_non_target')
    combined_invalid_target_epochs_dict, invalid_target_data_dict, invalid_target_events_dict = combine_all_epochs(all_invalid_target_epochs_dict, condition_list, label='invalid_target')
    combined_invalid_distractor_epochs_dict, invalid_distractor_data_dict, invalid_distractor_events_dict = combine_all_epochs(all_invalid_distractor_epochs_dict, condition_list, label='invalid_distractor')

    # Placeholder for the sampled dictionary: reducing non-target dicts lengths to match that of distractor events
    combined_non_target_target_stim_sampled_epochs_dict = {}
    combined_non_target_distractor_stim_sampled_epochs_dict = {}
    combined_target_response_stim_sampled_epochs_dict = {}

    # Iterate through each condition
    for condition in combined_distractor_no_response_epochs_dict.keys():
        # Get the number of events for the distractor no-response epochs in the current condition
        target_event_count = len(combined_distractor_no_response_epochs_dict[condition][0].events)

        # Retrieve the non-target stim epochs for the current condition
        non_target_target_epochs = combined_non_target_target_stim_epochs_dict[condition][0]
        non_target_distractor_epochs = combined_non_target_distractor_stim_epochs_dict[condition][0]
        target_response_epochs = combined_target_response_epochs_dict[condition][0]

        # Randomly sample epochs
        np.random.seed(42)
        non_target_target_sampled_epochs = non_target_target_epochs[np.random.choice(len(non_target_target_epochs), target_event_count, replace=False)]
        non_target_distractor_sampled_epochs = non_target_distractor_epochs[np.random.choice(len(non_target_distractor_epochs), target_event_count, replace=False)]
        target_sampled_epochs = target_response_epochs[np.random.choice(len(target_response_epochs), target_event_count, replace=False)]
        # Store in the new dictionary
        combined_non_target_target_stim_sampled_epochs_dict[condition] = [non_target_target_sampled_epochs]
        combined_non_target_distractor_stim_sampled_epochs_dict[condition] = [non_target_distractor_sampled_epochs]

        combined_target_response_stim_sampled_epochs_dict[condition] = [target_sampled_epochs]

    # Define an empty dictionary to store TFA results
    tfa_results_dict = {
        'target_response': {},
        'distractor_no_response': {},
        'non_target_target_stim': {},
        'non_target_distractor_stim': {}}

    # Run TFA for each combined epoch type within the dictionaries
    tfa_results_dict = run_tfa(tfa_results_dict, combined_target_response_stim_sampled_epochs_dict, epoch_label='target_response', mean_label='response_power_mean', median_label='response_power_median', std_label='response_power_std')
    tfa_results_dict = run_tfa(tfa_results_dict, combined_distractor_no_response_epochs_dict, epoch_label='distractor_no_response', mean_label='distractor_no_response_power_mean', median_label='distractor_no_response_power_median', std_label='distractor_no_response_power_std')
    tfa_results_dict = run_tfa(tfa_results_dict, combined_non_target_target_stim_sampled_epochs_dict, epoch_label='non_target_target_stim', mean_label='non_target_target_mean', median_label='non_target_target_median', std_label='non_target_target_std')
    tfa_results_dict = run_tfa(tfa_results_dict, combined_non_target_distractor_stim_sampled_epochs_dict, epoch_label='non_target_distractor_stim', mean_label='non_target_distractor_mean', median_label='non_target_distractor_median', std_label='non_target_distractor_std')

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
    non_target_target_results_dict = epochs_vals(combined_non_target_target_stim_sampled_epochs_dict)
    non_target_distractor_results_dict = epochs_vals(combined_non_target_distractor_stim_sampled_epochs_dict)

    # Filter outliers
    filtered_target_results_dict = filter_outliers_epochs(target_results_dict)
    filtered_distractor_results_dict = filter_outliers_epochs(distractor_results_dict)
    filtered_non_target_target_results_dict = filter_outliers_epochs(non_target_target_results_dict)
    filtered_non_target_distractor_results_dict = filter_outliers_epochs(non_target_distractor_results_dict)

    filtered_tfa_results_dict = {'Target': tfa_results_dict['target_response'],
                        'Distractor': tfa_results_dict['distractor_no_response'],
                        'Non-Target Target': tfa_results_dict['non_target_target_stim'],
                        'Non-Target Distractor': tfa_results_dict['non_target_distractor_stim']}

    with open(os.path.join(subject_results_dir/ 'tfa' , f'tfa_dict_{sub_input}.pkl'), 'wb') as f:
        pickle.dump(filtered_tfa_results_dict, f)


    save_subject_results(sub_input, filtered_target_results_dict, filtered_distractor_results_dict, filtered_non_target_target_results_dict, filtered_non_target_distractor_results_dict)


