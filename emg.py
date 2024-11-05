'''
Pre-processing EMG data: baseline recording,
motor-only responses and epoching our signal around target
and distractor events respectively.
'''
# libraries:
from copy import deepcopy
from collections import Counter
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
matplotlib.use('Agg')
from matplotlib import pyplot as plt, patches
from meegkit.dss import dss_line
import pandas as pd
from helper import grad_psd, snr
from scipy.stats import kruskal, shapiro
from scikit_posthocs import posthoc_dunn
from scipy.stats import kruskal, shapiro
import scikit_posthocs as sp

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

# band-pass filter 20-150 Hz
# Remove low-frequency noise: Eliminates motion artifacts or baseline drift that occur below 20 Hz.
# Remove high-frequency noise: Filters out high-frequency noise (e.g., electrical noise or other non-EMG signals) above 150-450 Hz,
# which isn't part of typical muscle activity.
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
    helper = mne.set_bipolar_reference(helper, anode='A2', cathode='M2', ch_name='EMG_bipolar')
    helper.drop_channels(['A1'])
    helper_filt = helper.copy().filter(l_freq=1, h_freq=150)
    print('Remove power line noise and apply minimum-phase highpass filter')  # Cheveigné, 2020
    helper_filt.copy().notch_filter(freqs=[50, 100, 150], method='fir')
    helper_rectified = helper.copy()
    helper_rectified._data = np.abs(helper_rectified._data)
    # helper_rectified._data *= 1e6
    # create helper epochs manually:
    helper_n_samples = len(helper_rectified.times)
    return helper_rectified, helper_n_samples


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


# get emg epochs, around the target and distractor events:
def epochs(events_dict, emg_rectified, events_emg, target='', tmin=0, tmax=0, baseline=None):
    event_ids = {key: val for key, val in events_dict.items() if any(event[2] == val for event in events_emg)}
    epochs = mne.Epochs(emg_rectified, events_emg, event_id=event_ids, tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
    epochs.save(fif_path / f'{sub_input}_condition_{condition}_{index}_{target}-epo.fif', overwrite=True)
    return epochs


def get_data(chosen_events, events_type, chosen_emg, chosen_emg_events, target='', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0)):
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


def append_epochs(all_epochs, epochs):
    if epochs is not None:
        all_epochs.append(epochs)
    return all_epochs

def is_valid_epoch_list(epoch_list):
    return [epochs for epochs in epoch_list if isinstance(epochs, mne.Epochs) and len(epochs) > 0]

# get combined epochs with all epochs of 1 condition in one:
def combine_all_epochs(epoch_list, epoch_type):
    """
       Combine, save, plot, and extract data for a list of epochs.

       Parameters:
           epoch_list (list): List of epochs to combine.
           sub_input (str): Subject identifier for naming files.
           condition (str): Condition label for naming files.
           epoch_type (str): Type of epoch (e.g., 'target_response', 'distractor_response').
           combined_epochs_path (Path): Directory path to save combined epochs.
           erp_path (Path): Directory path to save ERP plots.

       Returns:
           dict: Dictionary containing 'epochs', 'erp', 'data', and 'events' for the combined epochs,
                 or None if the epoch list is invalid.
       """
    if epoch_list:
        valid_epochs = is_valid_epoch_list(epoch_list)
        if valid_epochs:
            # Concatenate epochs
            all_epochs_combined = mne.concatenate_epochs(valid_epochs)

            # Save combined epochs
            all_epochs_combined.save(
                combined_epochs / f'{sub_input}_condition_{condition}_combined_{epoch_type}_epochs-epo.fif',
                overwrite=True
            )

            # Plot and save ERP
            erp_fig = all_epochs_combined.average().plot()
            erp_fig.savefig(erp_path / f'{sub_input}_condition_{condition}_{epoch_type}_erp.png')
            plt.close(erp_fig)

            # Extract data and events
            data = all_epochs_combined.get_data(copy=True)
            events = all_epochs_combined.events

            return all_epochs_combined, erp_fig, data, events
    return None, None, None, None

def tfa_heatmap(epochs, target):
    frequencies = np.logspace(np.log10(1), np.log10(30), num=30)  # Frequencies from 1 to 30 Hz
    n_cycles = np.minimum(frequencies / 2, 7)  # Number of cycles in Morlet wavelet (adapts to frequency)

    # Compute the Time-Frequency Representation (TFR) using Morlet wavelets
    epochs = epochs.copy().crop(tmin=-0.2, tmax=0.9)
    power = mne.time_frequency.tfr_morlet(epochs, freqs=frequencies, n_cycles=n_cycles, return_itc=False)
    power.apply_baseline(baseline=(-0.2, 0.0), mode='logratio')

    # Plot TFR as a heatmap (power across time and frequency)
    power_plot = power.plot([0], title=f'TFR (Heatmap) {condition} {target}', show=True)
    for i, fig in enumerate(power_plot):
        fig.savefig(
            psd_path / f'{sub_input}_{condition}_{target}_plot.png')  # Save with a unique name for each figure
        plt.close(fig)
    return power


def get_avg_band_power(power, bands, fmin, fmax, tmin=0.0, tmax=0.9):
    """Compute the average power for specified frequency bands within a time window."""
    time_mask = (power.times >= tmin) & (power.times <= tmax)  # Mask for the time window
    early_phase = (power.times >= 0.0) & (power.times <= 0.2)
    late_phase = (power.times >= 0.2) & (power.times <= 0.9)
    low_band = bands['low_band']
    middle_band = bands['mid_band']
    high_band = bands['high_band']

    # investigate entire epoch:
    for band_name, (fmin, fmax) in bands.items():
        freq_mask = (power.freqs >= fmin) & (power.freqs <= fmax)
        band_power = power.data[:, freq_mask, :][:, :, time_mask]  # Data within frequency band and time window
        early_band_power = power.data[:, freq_mask, :][:, :, early_phase]  # band powers in early phase
        late_band_power = power.data[:, freq_mask, :][:, :, late_phase]  # band powers in late phase

        # Compute the average power within the frequency band and time window
        avg_band_power = band_power.mean()  # Mean across all epochs, frequencies, and time within the window
        early_avg_band_power = early_band_power.mean()
        late_avg_band_power = late_band_power.mean()
        early_vs_late = late_avg_band_power / early_avg_band_power  # overall avg powe trend of each epoch

    # investigate the three bands separately, and get ratios of early vs late phase:
    low_freq_mask = (power.freqs >= low_band[0]) & (power.freqs <= low_band[1])
    low_band_power = power.data[:, low_freq_mask, :][:, :, time_mask]  # Data within frequency band and time window
    low_early_band_power = power.data[:, low_freq_mask, :][:, :, early_phase]  # band powers in early phase
    low_late_band_power = power.data[:, low_freq_mask, :][:, :, late_phase]  # band powers in late phase

    # Compute the average power within the frequency band and time window
    low_avg_band_power = low_band_power.mean()  # Mean across all epochs, frequencies, and time within the window
    low_early_avg_band_power = low_early_band_power.mean()
    low_late_avg_band_power = low_late_band_power.mean()
    low_early_vs_late = low_late_avg_band_power / low_early_avg_band_power

    middle_freq_mask = (power.freqs >= middle_band[0]) & (power.freqs <= middle_band[1])
    middle_band_power = power.data[:, middle_freq_mask, :][:, :,
                        time_mask]  # Data within frequency band and time window
    middle_early_band_power = power.data[:, middle_freq_mask, :][:, :, early_phase]  # band powers in early phase
    middle_late_band_power = power.data[:, middle_freq_mask, :][:, :, late_phase]  # band powers in late phase

    # Compute the average power within the frequency band and time window
    middle_avg_band_power = middle_band_power.mean()  # Mean across all epochs, frequencies, and time within the window
    middle_early_avg_band_power = middle_early_band_power.mean()
    middle_late_avg_band_power = middle_late_band_power.mean()
    middle_early_vs_late = middle_late_avg_band_power / middle_early_avg_band_power

    high_freq_mask = (power.freqs >= high_band[0]) & (power.freqs <= high_band[1])
    high_band_power = power.data[:, high_freq_mask, :][:, :, time_mask]  # Data within frequency band and time window
    high_early_band_power = power.data[:, high_freq_mask, :][:, :, early_phase]  # band powers in early phase
    high_late_band_power = power.data[:, high_freq_mask, :][:, :, late_phase]  # band powers in late phase

    # Compute the average power within the frequency band and time window
    high_avg_band_power = high_band_power.mean()  # Mean across all epochs, frequencies, and time within the window
    high_early_avg_band_power = high_early_band_power.mean()
    high_late_avg_band_power = high_late_band_power.mean()
    high_early_vs_late = high_late_avg_band_power / high_early_avg_band_power

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
            'high avg power': high_avg_band_power,
            'LTER': early_vs_late,
            'low LTER': low_early_vs_late,
            'mid LTER': middle_early_vs_late,
            'high LTER': high_early_vs_late}

    return vals


def epochs_vals(epochs):
        frequencies = np.logspace(np.log10(1), np.log10(30), num=30)
        n_cycles = frequencies / 2

        # Define frequency bands
        bands = {
            'low_band': (1, 10),
            'mid_band': (10, 20),
            'high_band': (20, 30)
        }

        # Initialize list to store power and metrics for each epoch
        epochs_vals = []

        # Loop over each epoch and compute TFA independently
        for epoch_index in range(len(epochs)):
            # Extract single epoch as an Epochs object for TFR computation
            epoch_data = epochs.get_data(copy=True)[epoch_index][np.newaxis, :, :]
            epoch_info = epochs.info
            single_epoch = mne.EpochsArray(epoch_data, epoch_info, tmin=epochs.tmin)

            # Compute TFR for the individual epoch
            power = mne.time_frequency.tfr_morlet(single_epoch, freqs=frequencies, n_cycles=n_cycles, return_itc=False)

            # Mask for early and late phases
            time_mask = (power.times >= 0.0) & (power.times <= 0.9)
            early_phase = (power.times >= 0.0) & (power.times <= 0.2)
            late_phase = (power.times >= 0.2) & (power.times <= 0.9)

            # Dictionary to store metrics for each band in the current epoch
            epoch_vals = {}
            bands_avg = {}

            # Iterate through each frequency band and calculate metrics
            for band_name, (fmin, fmax) in bands.items():
                freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
                band_power = power.data[:, freq_mask, :][:, :, time_mask]

                # Calculate average power across the time dimension
                avg_band_power = band_power.mean()  # Average power over time
                early_band_power = power.data[:, freq_mask, :][:, :, early_phase].mean()
                late_band_power = power.data[:, freq_mask, :][:, :, late_phase].mean()
                early_vs_late_ratio = late_band_power / early_band_power  # Ratio for phase power changes

                # Store the average power for later determination of dominant band
                bands_avg[band_name] = avg_band_power

                # Store values in dictionary for each band
                epoch_vals[band_name] = {
                    'avg power': avg_band_power,
                    'early power': early_band_power,
                    'late power': late_band_power,
                    'LTER': early_vs_late_ratio
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

            # Correct calculation for overall LTER across the entire epoch
            overall_early_power = power.data[:, :, early_phase].mean()
            overall_late_power = power.data[:, :, late_phase].mean()
            overall_lter = overall_late_power / overall_early_power

            # Append overall metrics and dominant band/freq to epoch values
            epoch_vals.update({
                'overall_avg_power': sum(bands_avg.values()) / len(bands_avg),
                'LTER': overall_lter,
                'dominant_band': dominant_band,
                'dominant_freq': dominant_freq
            })

            # Append metrics for the current epoch
            epochs_vals.append(epoch_vals)

        return epochs_vals


import seaborn as sns


def plot_dominant_frequency_distributions(target_vals, distractor_vals, non_target_vals):
    # Extract dominant frequencies
    target_dominant_freqs = [epoch['dominant_freq'] for epoch in target_vals]
    distractor_dominant_freqs = [epoch['dominant_freq'] for epoch in distractor_vals]
    non_target_dominant_freqs = [epoch['dominant_freq'] for epoch in non_target_vals]

    # Prepare DataFrame for plotting and statistical testing
    df = pd.DataFrame({
        'Frequency': target_dominant_freqs + distractor_dominant_freqs + non_target_dominant_freqs,
        'Epoch Type': (['Target'] * len(target_dominant_freqs)) +
                      (['Distractor'] * len(distractor_dominant_freqs)) +
                      (['Non-Target'] * len(non_target_dominant_freqs))
    })

    # Kruskal-Wallis Test for significance across groups
    kruskal_p_val = kruskal(target_dominant_freqs, distractor_dominant_freqs, non_target_dominant_freqs).pvalue
    print(f"Kruskal-Wallis p-value for dominant frequencies: {kruskal_p_val}")

    # Dunn's post-hoc test if Kruskal-Wallis is significant
    significance_labels = None
    if kruskal_p_val < 0.05:
        posthoc = sp.posthoc_dunn(df, val_col='Frequency', group_col='Epoch Type', p_adjust='bonferroni')

        # Function for labeling significance with asterisks
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

        # Generate significance labels for each pair
        significance_labels = {
            ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
            ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
            ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target'])
        }

    # Plot violin plot for frequency distribution comparison
    plt.figure(figsize=(10, 6))
    sns.violinplot(x='Epoch Type', y='Frequency', data=df, palette=['blue', 'orange', 'green'])
    plt.title(f"{condition} Dominant Frequency Distribution by Epoch Type")
    plt.ylabel("Dominant Frequency (Hz)")

    # Add significance labels
    y_max = df['Frequency'].max()
    y_offset = y_max * 0.05
    if significance_labels:
        pairs = [('Target', 'Distractor'), ('Target', 'Non-Target'), ('Distractor', 'Non-Target')]
        for (group1, group2) in pairs:
            x1, x2 = df['Epoch Type'].unique().tolist().index(group1), df['Epoch Type'].unique().tolist().index(group2)
            y = y_max + y_offset
            label = significance_labels[(group1, group2)]
            if label != "ns":  # Only display significant comparisons
                plt.plot([x1, x2], [y, y], color='black')
                plt.text((x1 + x2) / 2, y + y_offset, label, ha='center', va='bottom', fontsize=12)
                y_max += y_offset * 1.5  # Move up for the next annotation

    # Save the plot
    plt.savefig(class_figs / f'{condition} dominant_frequency_distributions_by_epoch_type.png')
    plt.close()


def plot_dominant_band_distributions(target_vals, distractor_vals, non_target_vals):
    # Count dominant bands for each epoch type
    def calculate_band_percentages(epoch_vals):
        dominant_bands = [epoch['dominant_band'] for epoch in epoch_vals]
        bands = ['low_band', 'mid_band', 'high_band']
        band_counts = {band: dominant_bands.count(band) for band in bands}
        total = sum(band_counts.values())
        return {band: (count / total) * 100 for band, count in band_counts.items()}

    target_percentages = calculate_band_percentages(target_vals)
    distractor_percentages = calculate_band_percentages(distractor_vals)
    non_target_percentages = calculate_band_percentages(non_target_vals)

    # Create a DataFrame for seaborn bar plot
    data = pd.DataFrame({
        'Band': ['Low (1-10Hz)', 'Mid (10-20Hz)', 'High (20-30Hz)'] * 3,
        'Epoch Type': ['Target'] * 3 + ['Distractor'] * 3 + ['Non-Target'] * 3,
        'Percentage': [
            target_percentages['low_band'], target_percentages['mid_band'], target_percentages['high_band'],
            distractor_percentages['low_band'], distractor_percentages['mid_band'],
            distractor_percentages['high_band'],
            non_target_percentages['low_band'], non_target_percentages['mid_band'],
            non_target_percentages['high_band']
        ]
    })

    # Plot with seaborn
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Epoch Type', y='Percentage', hue='Band', data=data, palette=['blue', 'orange', 'green'])
    plt.title(f"{condition} Dominant Band Distribution by Epoch Type")
    plt.ylabel("Percentage of Epochs (%)")
    plt.xlabel("Epoch Type")

    # Kruskal-Wallis and post-hoc if significant
    kruskal_p_val = kruskal(
        [target_percentages['low_band'], target_percentages['mid_band'], target_percentages['high_band']],
        [distractor_percentages['low_band'], distractor_percentages['mid_band'],
         distractor_percentages['high_band']],
        [non_target_percentages['low_band'], non_target_percentages['mid_band'],
         non_target_percentages['high_band']]
    ).pvalue
    print(f"Kruskal-Wallis p-value for dominant bands: {kruskal_p_val}")

    if kruskal_p_val < 0.05:
        # Prepare data for Dunn's post-hoc test and add asterisks based on significance
        flat_data = target_percentages.values() + distractor_percentages.values() + non_target_percentages.values()
        labels = ['Target'] * 3 + ['Distractor'] * 3 + ['Non-Target'] * 3
        posthoc = sp.posthoc_dunn(flat_data, group_col=labels, val_col='Percentage', p_adjust='bonferroni')

        # Add significance annotations to the plot
        pairs = [('Target', 'Distractor'), ('Target', 'Non-Target'), ('Distractor', 'Non-Target')]
        y_offset = max(data['Percentage']) * 0.05
        for pair in pairs:
            x1, x2 = data['Epoch Type'].unique().tolist().index(pair[0]), data[
                'Epoch Type'].unique().tolist().index(pair[1])
            y = max(data['Percentage']) + y_offset
            p_val = posthoc.loc[pair[0], pair[1]]
            if p_val < 0.05:
                label = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*"
                plt.plot([x1, x2], [y, y], color='black')
                plt.text((x1 + x2) / 2, y + y_offset, label, ha='center', va='bottom', fontsize=12)

    plt.legend(title="Dominant Band")
    plt.tight_layout()
    plt.savefig(class_figs / f'{condition} dominant_band_distributions_by_epoch_type.png')
    plt.close()


def plot_overall_avg_power_bar(target_vals, distractor_vals, non_target_vals):
        # Extract overall average power for each epoch type
        target_avg_powers = [epoch['overall_avg_power'] for epoch in target_vals]
        distractor_avg_powers = [epoch['overall_avg_power'] for epoch in distractor_vals]
        non_target_avg_powers = [epoch['overall_avg_power'] for epoch in non_target_vals]

        # Calculate mean power values
        target_avg_power = np.mean(target_avg_powers)
        distractor_avg_power = np.mean(distractor_avg_powers)
        non_target_avg_power = np.mean(non_target_avg_powers)

        # Data for bar plot
        epoch_types = ['Target', 'Distractor', 'Non-Target']
        avg_powers = [target_avg_power, distractor_avg_power, non_target_avg_power]
        unit_label = 'Overall Average Power (W)'

        # Kruskal-Wallis Test
        kruskal_p_val = kruskal(target_avg_powers, distractor_avg_powers, non_target_avg_powers).pvalue
        print(f"Kruskal-Wallis p-value: {kruskal_p_val}")

        # Dunn's posthoc test if Kruskal-Wallis is significant
        significance_labels = None
        if kruskal_p_val < 0.05:
            # Prepare data for posthoc Dunn's test
            df = pd.DataFrame({
                'overall_avg_power': target_avg_powers + distractor_avg_powers + non_target_avg_powers,
                'epoch_type': ['Target'] * len(target_avg_powers) +
                              ['Distractor'] * len(distractor_avg_powers) +
                              ['Non-Target'] * len(non_target_avg_powers)
            })
            posthoc = sp.posthoc_dunn(df, val_col='overall_avg_power', group_col='epoch_type',
                                      p_adjust='bonferroni')

            # Function for labeling significance with asterisks
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

            # Generate significance labels for each pair
            significance_labels = {
                ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
                ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target'])
            }

        # Plot
        plt.figure(figsize=(8, 6))
        bars = plt.bar(epoch_types, avg_powers, color=['blue', 'orange', 'green'])
        plt.ylabel(unit_label)
        plt.title(f'{condition} Comparison of Overall Average Power by Epoch Type')

        # Add significance labels
        y_offset = max(avg_powers) * 0.05  # Offset for significance text above bars
        if significance_labels:
            y_max = max(avg_powers)
            pairs = [('Target', 'Distractor'), ('Target', 'Non-Target'), ('Distractor', 'Non-Target')]
            for (group1, group2) in pairs:
                x1, x2 = epoch_types.index(group1), epoch_types.index(group2)
                y = y_max + y_offset
                label = significance_labels[(group1, group2)]
                if label != "ns":  # Only display significant comparisons
                    plt.plot([x1, x2], [y, y], color='black')
                    plt.text((x1 + x2) / 2, y + y_offset, label, ha='center', va='bottom', fontsize=12)
                    y_max += y_offset * 2  # Move up for the next annotation

        plt.savefig(class_figs / f'{condition} avg_powers_per_epoch_type.png')
        plt.close()



def plot_LTER_distribution_violin_with_significance(target_vals, distractor_vals, non_target_vals):
    # Extract LTER values and prepare a DataFrame for Seaborn
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
            return "ns"  # ns for non-significant
    target_LTER = [{'LTER': epoch['LTER'], 'Type': 'Target'} for epoch in target_vals]
    distractor_LTER = [{'LTER': epoch['LTER'], 'Type': 'Distractor'} for epoch in distractor_vals]
    non_target_LTER = [{'LTER': epoch['LTER'], 'Type': 'Non-Target'} for epoch in non_target_vals]

    # Combine into a single DataFrame
    all_data = pd.DataFrame(target_LTER + distractor_LTER + non_target_LTER)

    # Perform normality tests on each group
    target_normality = shapiro(all_data[all_data['Type'] == 'Target']['LTER']).pvalue
    distractor_normality = shapiro(all_data[all_data['Type'] == 'Distractor']['LTER']).pvalue
    non_target_normality = shapiro(all_data[all_data['Type'] == 'Non-Target']['LTER']).pvalue
    print(
        f"Normality p-values - Target: {target_normality}, Distractor: {distractor_normality}, Non-Target: {non_target_normality}")

    # Use non-parametric Kruskal-Wallis if normality assumption is violated
    if min(target_normality, distractor_normality, non_target_normality) < 0.05:
        print("Non-normal distribution detected; using Kruskal-Wallis test")
        kruskal_stat, kruskal_p = kruskal(
            all_data[all_data['Type'] == 'Target']['LTER'],
            all_data[all_data['Type'] == 'Distractor']['LTER'],
            all_data[all_data['Type'] == 'Non-Target']['LTER']
        )
        if kruskal_p < 0.05:
            posthoc = posthoc_dunn(all_data, val_col='LTER', group_col='Type', p_adjust='bonferroni')
        else:
            posthoc = None
    else:
        print("Data seems normally distributed; consider parametric tests instead")
        kruskal_stat, kruskal_p = None, None
        posthoc = None

    # Plot the distributions
    plt.figure(figsize=(8, 6))
    sns.violinplot(x='Type', y='LTER', hue='Type', data=all_data, palette=['blue', 'orange', 'green'], dodge=False,
                   legend=False)
    plt.ylabel('LTER')
    plt.title(f'{condition} Distribution of Overall LTER by Epoch Type ')

    # Add significance annotations
    # Add significance annotations
    if posthoc is not None:
        y_max = all_data['LTER'].max()
        offset = 0.05 * y_max  # Offset for placing significance markers
        significance_pairs = {('Target', 'Distractor'): (0, 1), ('Target', 'Non-Target'): (0, 2),
                              ('Distractor', 'Non-Target'): (1, 2)}

        for (group1, group2), (x1, x2) in significance_pairs.items():
            p_val = posthoc[group1][group2]
            label = significance_label(p_val)
            if label != "ns":  # Only mark significant comparisons
                y = y_max + offset
                plt.plot([x1, x2], [y, y], color='black')
                plt.text((x1 + x2) / 2, y + offset, label, ha='center', va='bottom', fontsize=12, color='black')
                y_max += offset * 2  # Increase max for next annotation

    plt.savefig(class_figs/ f'{condition} LTER distribution violin significance.png')
    plt.close()


def calculate_and_plot_dominant_frequency_distribution(target_vals, distractor_vals, non_target_vals, bin_edges):
    def calculate_percentages(epoch_vals):
        # Extract dominant frequencies from epoch values
        dominant_freqs = [epoch['dominant_freq'] for epoch in epoch_vals]

        # Digitize dominant frequencies into bins
        binned_freqs = np.digitize(dominant_freqs, bin_edges)

        # Count frequencies within each bin
        freq_counts = Counter(binned_freqs)

        # Calculate the total count and percentages for each bin
        total_count = sum(freq_counts.values())
        percentages = {bin_edges[bin_idx - 1]: (count / total_count) * 100 for bin_idx, count in freq_counts.items()}
        return percentages

    # Calculate percentages for each epoch type
    target_freq_percentages = calculate_percentages(target_vals)
    distractor_freq_percentages = calculate_percentages(distractor_vals)
    non_target_freq_percentages = calculate_percentages(non_target_vals)

    # Plot each type's distribution in a single figure
    plt.figure(figsize=(12, 6))
    bar_width = 0.3

    # Plot target frequencies
    plt.bar([x - bar_width for x in target_freq_percentages.keys()],
            target_freq_percentages.values(),
            width=bar_width, color='blue', label='Target')

    # Plot distractor frequencies
    plt.bar(distractor_freq_percentages.keys(),
            distractor_freq_percentages.values(),
            width=bar_width, color='orange', label='Distractor')

    # Plot non-target frequencies
    plt.bar([x + bar_width for x in non_target_freq_percentages.keys()],
            non_target_freq_percentages.values(),
            width=bar_width, color='green', label='Non-Target')

    # Add plot details
    plt.xlabel('Dominant Frequency (Hz)')
    plt.ylabel('Percentage of Epochs (%)')
    plt.title(f'{condition} Dominant Frequency Distribution by Epoch Type')
    plt.xticks(bin_edges, rotation=45)
    plt.legend()
    plt.tight_layout()

    # Save and close the plot
    plt.savefig(class_figs / f'{condition}_dominant_frequency_distribution_by_epoch_type.png')
    plt.close()




from collections import Counter

if __name__ == '__main__':
    # matplotlib.use('TkAgg')
    sub_input = input("Give sub number as subn (n for number): ")
    sub = [sub.strip() for sub in sub_input.split(',')]
    cm = 1 / 2.54
    # 0. LOAD THE DATA
    sub_dirs = []
    for subs in sub:
        default_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data')
        data_dir = default_dir / 'eeg' / 'raw'  # to get the raw EEG files (which will we 'convert' to EMG
        sub_dir = data_dir / subs
        emg_dir = default_dir / 'emg' / subs  # creating a folder dedicated to EMG files we will create
        results_path = emg_dir / 'preprocessed' / 'results'  # where results will be saved
        fig_path = emg_dir / 'preprocessed' / 'figures' # where figures from pre-processing and classification will be stored
        erp_path = fig_path / 'ERPs'
        z_figs = fig_path / 'z_distributions'
        class_figs = fig_path / 'classifications'
        df_path = default_dir / 'performance' / subs / 'tables'  # csv tables containing important information from vmrk files
        sub_dirs.append(sub_dir)
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
    # conditions: a1, a2, e1 or e2
    condition = input('Please provide condition (exp. EEG): ')  # choose a condition of interest for processing
    ### Get target header files for all participants
    target_header_files_list = []
    for sub_dir in sub_dirs:
        header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
        filtered_files = [file for file in header_files if condition in file]
        if filtered_files:
            target_header_files_list.append(filtered_files)
    # TO CALCULATE SNR:
    for files in sub_dir.iterdir():
        if files.is_file and 'baseline.vhdr' in files.name:
            baseline = mne.io.read_raw_brainvision(files, preload=True)

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
    # Calculate the number of samples in the recording
    '''
    - epoch_length_s: Total length of each epoch in seconds.
    - tmin: Start of the epoch in seconds relative to each synthetic event.
    - tmax: End of the epoch in seconds relative to each synthetic event.
    '''
    n_samples = baseline_rectified.n_times
    duration_s = n_samples / 500
    epoch_length_s = 1.1
    tmin = -0.2
    tmax = 0.9
    # Calculate the step size in samples for each epoch (based on the epoch length)
    step_size = int(epoch_length_s * 500)
    # Generate synthetic events spaced at regular intervals across the continuous data
    # The first column is the sample index, second is filler (e.g., 0), third is the event ID (e.g., 1)
    events = np.array([[i, 0, 1] for i in range(0, n_samples - step_size, step_size)])

    # Create epochs based on these synthetic events
    epochs_baseline = mne.Epochs(baseline_rectified, events, event_id={'arbitrary': 1}, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    epochs_baseline_erp = epochs_baseline.average().plot()
    epochs_baseline_erp.savefig(erp_path / f'{sub_input}_baseline_ERP.png')
    plt.close(epochs_baseline_erp)

    all_target_response_epochs = []
    all_target_no_response_epochs = []
    all_distractor_response_epochs = []
    all_distractor_no_response_epochs = []
    all_non_target_stim_epochs = []
    all_invalid_non_target_epochs = []
    all_invalid_distractor_epochs = []
    all_invalid_target_epochs = []

    target_epochs_list = []
    distractor_epochs_list = []

    ### Process Each File under the Selected Condition ###
    for index in range(len(target_header_files_list[0])):
        # Loop through all the files matching the condition
        emg_file = target_header_files_list[0][index]
        full_path = os.path.join(sub_dir, emg_file)
        emg = mne.io.read_raw_brainvision(full_path, preload=True)

        # Set montage for file:
        emg.rename_channels(mapping)
        emg.set_montage('standard_1020')
        emg = emg.pick_channels(['A2', 'M2', 'A1']) # select EMG-relevant files
        emg.set_eeg_reference(ref_channels=['A1']) # set correct reference channel
        emg = mne.set_bipolar_reference(emg, anode='A2', cathode='M2', ch_name='EMG_bipolar')  # change ref to EMG bipolar
        emg.drop_channels(['A1'])  # drop reference channel (don't need to see it)

        # Get Target Block information
        csv_path = default_dir / 'params' / 'block_sequences' / f'{sub_input}.csv'
        csv = pd.read_csv(csv_path)
        # get target stream, blocks for selected condition, axis relevant to condition (az or ele)
        target_stream, distractor_stream, target_blocks, axis, target_mapping, distractor_mapping = get_target_blocks()

        # Define target and distractor events based on target number and stream
        target_block = [target_block for target_block in target_blocks.values[index]]
        target_events, distractor_events = define_events(target_stream)
        combined_events = {**distractor_events, **target_events}
        # Get events for EMG analysis, based on filtered target and distractor, and response events
        targets_emg_events = create_events(target_events, target_mapping)
        distractors_emg_events = create_events(distractor_events, distractor_mapping)
        responses_emg_events = create_events(response_events, response_mapping)

        # Filter and rectify the EMG data
        emg_filt = filter_emg(emg)
        emg_rect = emg_filt.copy()
        emg_rectified = rectify(emg_rect)
        target_emg_rectified = rectify(emg_filt)
        distractor_emg_rectified = rectify(emg_filt)
        non_target_emg_rectified = rectify(emg_filt)

        # # epoch target data: with responses, without:
        # # get response epochs separately first:
        # response_epochs = epochs(response_events, non_target_emg_rectified, responses_emg_events, target='responses', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))

        # Create non-target Stimuli Epochs:
        # for distractor:
        non_target_events_distractor = baseline_events(distractor_events, distractor_mapping)
        # for target:
        non_target_events_target = baseline_events(target_events, target_mapping)

        # Initialize the lists for categorization
        target_response_events = []
        target_no_response_events = []
        distractor_response_events = []
        distractor_no_response_events = []
        non_target_stimulus_events = []
        invalid_non_target_response_events = []
        invalid_target_response_events = []
        invalid_distractor_response_events = []
        invalid_response_epochs = []
        response_only_epochs = []  # Pure response events not linked to other events

        # Track which response events have been assigned (initialize as NumPy array)
        unassigned_response_events = responses_emg_events  # This should already be a NumPy array

        # Step 4: Process each target event
        for event in targets_emg_events:
            stim_timepoint = event[0] / 500  # Replace 500 with your actual sampling frequency if different
            time_start = stim_timepoint  # Start of the time window for a target event
            time_end = stim_timepoint + 0.9  # End of the time window for a target event

            # Find responses that fall within this target's time window
            response_found = False
            for response_event in unassigned_response_events:
                response_timepoint = response_event[0] / 500
                if time_start < response_timepoint < time_end:
                    if response_timepoint - stim_timepoint < 0.2:
                        invalid_target_response_events.append(np.append(event, 'target'))  # Classify as invalid response
                    else:
                        # If a valid response is found within the time window, assign it to target_response_events
                        target_response_events.append(event)

                    # Remove the response_event row using np.delete
                    idx_to_remove = np.where((unassigned_response_events == response_event).all(axis=1))[0][0]
                    unassigned_response_events = np.delete(unassigned_response_events, idx_to_remove, axis=0)

                    response_found = True
                    break  # Stop after finding the first response (adjust if multiple responses per target are needed)

            if not response_found:
                # If no response is found, add the target event to target_no_response_events
                target_no_response_events.append(event)
        # Step 5: Process leftover responses for distractor events
        for event in distractors_emg_events:
            stim_timepoint = event[0] / 500
            time_start = stim_timepoint
            time_end = stim_timepoint + 0.9

            # Check for unassigned responses that fall within the distractor's time window
            response_found = False
            for response_event in unassigned_response_events:
                response_timepoint = response_event[0] / 500
                if time_start < response_timepoint < time_end:
                    if response_timepoint - stim_timepoint < 0.2:
                        invalid_distractor_response_events.append(np.append(event, 'distractor'))  # Classify as invalid response
                    else:
                        # If a valid response is found within the time window, assign it to target_response_events
                        distractor_response_events.append(event)

                    idx_to_remove = np.where((unassigned_response_events == response_event).all(axis=1))[0][0]
                    unassigned_response_events = np.delete(unassigned_response_events, idx_to_remove, axis=0)
                    response_found = True
                    break

            if not response_found:
                # If no response is found, add the distractor to no_response list
                distractor_no_response_events.append(event)

        # Step 6: Process non-target events for any remaining responses
        non_target_combined_events = np.concatenate([non_target_events_target, non_target_events_distractor])
        non_target_combined_events = non_target_combined_events[np.argsort(non_target_combined_events[:, 0])]
        for event in non_target_combined_events:
            stim_timepoint = event[0] / 500
            time_start = stim_timepoint
            time_end = stim_timepoint + 0.9

            response_found = False
            for response_event in unassigned_response_events:
                response_timepoint = response_event[0] / 500
                if time_start < response_timepoint < time_end:
                    # If response falls in a non-target window, add to invalid responses
                    invalid_non_target_response_events.append(np.append(event, 'non-target'))
                    idx_to_remove = np.where((unassigned_response_events == response_event).all(axis=1))[0][0]
                    unassigned_response_events = np.delete(unassigned_response_events, idx_to_remove, axis=0)
                    response_found = True
                    break

            if not response_found:
                non_target_stimulus_events.append(event)

        # Step 7: Remaining responses are pure response events (response_only_epochs)
        response_only_epochs = unassigned_response_events  # should be empty
        invalid_non_target_events = [np.array(event[:3], dtype=int) for event in invalid_non_target_response_events]
        invalid_target_events = [np.array(event[:3], dtype=int) for event in invalid_target_response_events]
        invalid_distractor_events = [np.array(event[:3], dtype=int) for event in invalid_distractor_response_events]


        # Target stimuli with responses:
        target_response_epochs, target_response_data = get_data(target_response_events, target_events, target_emg_rectified, target_response_events, target='target_with_responses', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))
        # Target stimuli without responses:
        target_no_responses_epochs, target_no_responses_data = get_data( target_no_response_events, target_events, target_emg_rectified, target_no_response_events, target='target_without_responses', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))

        # Distractor stimuli with responses:
        distractor_responses_epochs, distractor_responses_data = get_data(distractor_response_events, distractor_events, distractor_emg_rectified, distractor_response_events, target='distractor_with_responses', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))

        # Distractor stimuli without responses:
        distractor_no_responses_epochs, distractor_no_responses_data = get_data(distractor_no_response_events, distractor_events, distractor_emg_rectified,distractor_no_response_events, target='distractor_without_responses', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))

        # Non-target stimuli:
        non_target_stim_epochs, non_target_stim_data = get_data(non_target_stimulus_events, combined_events, emg_rect, non_target_stimulus_events, target='non_target_stimuli', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))

        # Invalid non-target responses:
        invalid_non_target_epochs, invalid_non_target_data = get_data(invalid_non_target_events, combined_events, non_target_emg_rectified, invalid_non_target_events, target='invalid_responses', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))

        # Invalid target responses:
        invalid_target_epochs, invalid_target_data = get_data(invalid_target_events, combined_events, non_target_emg_rectified, invalid_target_events, target='invalid_target_responses', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))

        # Invalid distractor responses:
        invalid_distractor_epochs, invalid_distractor_data = get_data(invalid_distractor_events, combined_events, non_target_emg_rectified, invalid_distractor_events, target='invalid_distractor_responses', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))

        # Append the epochs using the updated function
        all_target_response_epochs = append_epochs(all_target_response_epochs, target_response_epochs)
        all_target_no_response_epochs = append_epochs(all_target_no_response_epochs, target_no_responses_epochs)
        all_distractor_response_epochs = append_epochs(all_distractor_response_epochs, distractor_responses_epochs)
        all_distractor_no_response_epochs = append_epochs(all_distractor_no_response_epochs, distractor_no_responses_epochs)
        all_invalid_non_target_epochs = append_epochs(all_invalid_non_target_epochs, invalid_non_target_epochs)
        all_invalid_target_epochs = append_epochs(all_invalid_target_epochs, invalid_target_epochs)
        all_invalid_distractor_epochs = append_epochs(all_invalid_distractor_epochs, invalid_distractor_epochs)
        all_non_target_stim_epochs = append_epochs(all_non_target_stim_epochs, non_target_stim_epochs)


        # Process each type of epoch list and store results
        combined_target_response_epochs, target_response_erp_fig, target_response_data, target_response_events = combine_all_epochs(all_target_response_epochs, 'target_response')
        combined_target_no_response_epochs, target_no_response_erp_fig, target_no_response_data, target_no_response_events = combine_all_epochs(all_target_no_response_epochs, 'target_no_response')
        combined_distractor_response_epochs, distractor_response_erp_fig, distractor_response_data, distractor_response_events = combine_all_epochs(all_distractor_response_epochs, 'distractor_response')
        combined_distractor_no_response_epochs, distractor_no_response_erp_fig, distractor_no_response_data, distractor_no_response_events = combine_all_epochs(all_distractor_no_response_epochs, 'distractor_no_response')
        combined_non_target_stim_epochs, non_target_stim_erp_fig, non_target_stim_data, non_target_stim_events = combine_all_epochs(all_non_target_stim_epochs, 'non_target_stim')
        combined_invalid_non_target_epochs, invalid_non_target_erp_fig, invalid_non_target_data, invalid_non_target_events = combine_all_epochs(all_invalid_non_target_epochs, 'invalid_non_target')
        combined_invalid_target_epochs, invalid_target_erp_fig, invalid_target_data, invalid_target_events = combine_all_epochs(all_invalid_target_epochs, 'invalid_target')
        combined_invalid_distractor_epochs, invalid_distractor_erp_fig, invalid_distractor_data, invalid_distractor_events = combine_all_epochs(all_invalid_distractor_epochs, 'invalid_distractor')

        all_invalid_events = []
        if 'invalid_target_response_events' in locals() and len(invalid_target_response_events) > 0:
            all_invalid_events.append(invalid_target_response_events)
        if 'invalid_distractor_response_events' in locals() and len(invalid_distractor_response_events) > 0:
            all_invalid_events.append(invalid_distractor_response_events)
        if 'invalid_non_target_response_events' in locals() and len(invalid_non_target_response_events) > 0:
            all_invalid_events.append(invalid_non_target_response_events)
        flattened_list = []
        if 'all_invalid_events' in locals() and len(all_invalid_events) > 0:
            for sublist in all_invalid_events:
                for event in sublist:
                    flattened_list.append(event)
            all_invalid_events = np.concatenate(all_invalid_events)
            all_invalid_events = pd.DataFrame(all_invalid_events)
            all_invalid_events = all_invalid_events.drop(columns=1)
            all_invalid_events.columns = ['Timepoints', 'Stimulus', 'Type']
            all_invalid_events['Timepoints'] = all_invalid_events['Timepoints'].astype(int)
            all_invalid_events['Timepoints'] = all_invalid_events['Timepoints'] / 500
            all_invalid_events['Stimulus'] = all_invalid_events['Stimulus'].astype(int)
            all_invalid_events = all_invalid_events.sort_values(by='Timepoints')
            combined_invalid_events = all_invalid_events

        # Apply baseline normalization to non-target epochs
        baseline_data, baseline_derivative, baseline_z_scores, baseline_var, baseline_rms = baseline_normalization(
            epochs_baseline, tmin=0.2, tmax=0.9)

        # TFA:
        # Define frequency range of interest (from 1 Hz to 30 Hz)
        response_power = tfa_heatmap(combined_target_response_epochs, target='target_response')
        baseline_power = tfa_heatmap(epochs_baseline, target='baseline')
        non_target_power = tfa_heatmap(combined_non_target_stim_epochs, target='non_target_stim')
        distractor_power = tfa_heatmap(combined_distractor_no_response_epochs, target='distractor')


        bands = {
            "low_band": (1, 10),
            "mid_band": (10, 20),
            "high_band": (20, 30)
        }


        target_response_epochs_vals = epochs_vals(combined_target_response_epochs)
        distractor_no_response_epochs_vals = epochs_vals(combined_distractor_no_response_epochs)
        non_target_epochs_vals = epochs_vals(combined_non_target_stim_epochs)

        plot_overall_avg_power_bar(target_response_epochs_vals, distractor_no_response_epochs_vals, non_target_epochs_vals)

        # Example usage for each epoch type
        plot_dominant_frequency_distributions(target_response_epochs_vals, distractor_no_response_epochs_vals, non_target_epochs_vals)
        # Example usage for each epoch type
        plot_dominant_band_distributions(target_response_epochs_vals, distractor_no_response_epochs_vals, non_target_epochs_vals)

        plot_LTER_distribution_violin_with_significance(target_response_epochs_vals, distractor_no_response_epochs_vals, non_target_epochs_vals)

        bin_edges = np.linspace(1, 30, 30)  # Adjust as needed
        calculate_and_plot_dominant_frequency_distribution(target_response_epochs_vals, distractor_no_response_epochs_vals, non_target_epochs_vals, bin_edges)

        def add_labels(data, label, event_times, type):
            squeezed_data = data.squeeze(axis=1)  # (16, 551)
            data_df = pd.DataFrame(squeezed_data)
            data_df['Label'] = label
            data_df['Timepoints'] = event_times
            data_df['Type'] = type
            return data_df

        # get epochs dfs:
        # Assuming you have arrays of event times for each type of event
        if 'target_response_events' in locals():
            target_response_events = np.array(target_response_events)
            target_response_times = target_response_events[:, 0] / 500  # Adjust 500 to your actual sampling rate
            target_response_df = add_labels(target_response_data, label='Response', event_times=target_response_times,
                                            type='target')

        if 'distractor_no_response_events' in locals():
            distractor_no_response_events = np.array(distractor_no_response_events)
            distractor_no_response_times = distractor_no_response_events[:, 0] / 500
            distractor_no_responses_df = add_labels(distractor_no_response_data, label='No Response',
                                                    event_times=distractor_no_response_times, type='distractor')

        if 'combined_invalid_events' in locals():
            combined_invalid_events['Label'] = 'invalid response'
            invalid_responses_df = combined_invalid_events

        if 'target_no_response_events' in locals():
            target_no_response_events = np.array(target_no_response_events)
            target_no_response_times = target_no_response_events[:, 0] / 500
            target_no_responses_df = add_labels(target_no_response_data, label='No Response',
                                                event_times=target_no_response_times, type='target')

        if 'distractor_response_events' in locals() and distractor_response_events is not None and distractor_response_events.size > 0:
            # Ensure distractor_response_events is an array with valid data
            distractor_response_events = np.array(distractor_response_events)

            # Check if distractor_response_events is not None and is not empty
            if distractor_response_events.ndim > 0 and distractor_response_events[0] is not None:
                distractor_response_times = distractor_response_events[:, 0] / 500
                distractor_responses_df = add_labels(distractor_response_data, label='Response',
                                                     event_times=distractor_response_times, type='distractor')

        if 'non_target_stim_events' in locals():
            non_target_stimulus_events = np.array(non_target_stim_events)
            non_target_stim_times = non_target_stimulus_events[:, 0] / 500
            non_target_stim_df = add_labels(non_target_stim_data, label='No Response', event_times=non_target_stim_times,
                                            type='non-target')

            # Epoch the EMG data for classification based on features:
        target_epochs = epochs(target_events, target_emg_rectified, targets_emg_events, target='target', tmin=-0.2,
                               tmax=0.9, baseline=(-0.2, 0.0))
        target_epochs_list.append(target_epochs)
        target_epochs = mne.concatenate_epochs(target_epochs_list)
        distractor_epochs = epochs(distractor_events, distractor_emg_rectified, distractors_emg_events,
                                   target='distractor', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))
        distractor_epochs_list.append(distractor_epochs)
        distractor_epochs = mne.concatenate_epochs(distractor_epochs_list)
        # plot percentages of each response type:
        # calculate percentages for plotting:
        # total target and distractor stimuli
        total_target_count = len(target_epochs)
        total_distractor_count = len(distractor_epochs)
        # total true target responses:
        target_response_count = len(target_response_df)
        # total true distractor responses:
        if 'distractor_responses_df' in locals() and len(distractor_responses_df) > 0:
            distractor_response_count = len(distractor_responses_df)
        else:
            distractor_response_count = 0
        # total no-response target epochs:
        target_no_response_count = len(target_no_responses_df)
        # total target invalid response epochs:s
        target_invalid_response_count = len(invalid_target_response_events)

        # total no-response distractor epochs:
        distractor_no_response_count = len(distractor_no_responses_df)
        # invalid distractor response epochs:
        if 'invalid_distractor_events' in locals() and invalid_distractor_events is not None:
            distractor_invalid_response_count = len(invalid_distractor_events)
        else:
            distractor_invalid_response_count = 0

        # total no-response non-target epochs:
        non_target_stimulus_count = len(non_target_stim_df)
        # total invalid non-target response epochs:
        invalid_non_target_response_count = len(invalid_non_target_response_events)
        # total non-target stimulus epochs:
        total_non_target_stim_count = non_target_stimulus_count + invalid_non_target_response_count

        # get percentages for performance:
        correct_responses = target_response_count * 100 / total_target_count
        distractor_responses = distractor_response_count * 100 / total_distractor_count
        # invalid target and distractor responses:
        invalid_target_responses = target_invalid_response_count * 100 / total_target_count
        invalid_distractor_responses = distractor_invalid_response_count * 100 / total_distractor_count
        # missed targets:
        misses_count = total_target_count - (target_response_count + target_invalid_response_count)
        missed_targets = misses_count * 100 / total_target_count

        invalid_non_target_responses = invalid_non_target_response_count * 100 / total_non_target_stim_count

        total_invalid_response_count = invalid_non_target_response_count + distractor_invalid_response_count + target_invalid_response_count
        # total percentage of all invalid responses, based on total sum of stimuli
        invalid_responses = total_invalid_response_count * 100 / (
                    total_target_count + total_distractor_count + total_non_target_stim_count)
        # total errors:
        target_errors = (target_invalid_response_count + misses_count) * 100 / total_target_count
        total_errors = (total_invalid_response_count + misses_count + distractor_response_count) * 100 / (
                    total_non_target_stim_count + total_target_count + total_distractor_count)

        # plot performance: correct responses, distractor, target misses, invalid target resp, total target error
        categories = ['Correct Target Responses', 'Distractor Responses', 'Targets Missed', 'Invalid Target Responses',
                      'Total Target Error']
        colors = ['blue', 'red', 'yellow', 'orange', 'black']
        values = [correct_responses, distractor_responses, missed_targets, invalid_target_responses, target_errors]
        plt.figure(figsize=(12, 6))
        plt.bar(categories, values, color=colors)
        plt.xlabel('Response Type')
        plt.ylabel('Performance (in %)')
        plt.title(f'{condition} performance of {sub_input}')
        plt.savefig(fig_path / f'{condition}_performance_{sub_input}.png')
        plt.close()
