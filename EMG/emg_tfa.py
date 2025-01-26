from pathlib import Path
import os
import mne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')  # For interactive plotting on Windows
import pandas as pd
import seaborn as sns

sub_list = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub08',
            'sub20', 'sub21', 'sub10', 'sub11','sub13', 'sub14', 'sub15', 'sub16',
            'sub17', 'sub18', 'sub19', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26',
            'sub27', 'sub28', 'sub29']

epoch_types = ['combined_target_response_epochs', 'combined_distractor_no_response_epochs',
               'combined_non_target_target_epochs', 'combined_non_target_distractor_epochs']

epoch_categories = ['Target', 'Distractor', 'Non-Target Target', 'Non-Target Distractor']

conditions = ['a1', 'a2', 'e1', 'e2']

data_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/emg')
results_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/emg/subject_results/figures')


def get_epochs(condition=''):
    excluded_subs = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub08']
    target_epochs = {}
    distractor_epochs = {}
    non_target_target_epochs = {}
    non_target_distractor_epochs = {}
    if condition in ['e1', 'e2']:
        filtered_sub_list = [sub for sub in sub_list if sub not in excluded_subs]
    else:
        filtered_sub_list = sub_list
    for sub in filtered_sub_list:
        target_epochs[sub] = {}
        distractor_epochs[sub] = {}
        non_target_target_epochs[sub] = {}
        non_target_distractor_epochs[sub] = {}
        folderpath = data_path / sub / 'fif files' / 'combined_epochs'
        for file in folderpath.iterdir():
            if epoch_types[0] in file.name and condition in file.name:
                epoch = mne.read_epochs(file, preload=True)
                target_epochs[sub] = epoch
            elif epoch_types[1] in file.name and condition in file.name:
                epoch = mne.read_epochs(file, preload=True)
                distractor_epochs[sub] = epoch
            elif epoch_types[2] in file.name and condition in file.name:
                epoch = mne.read_epochs(file, preload=True)
                non_target_target_epochs[sub] = epoch
            elif epoch_types[3] in file.name and condition in file.name:
                non_target_distractor_epochs[sub] = epoch
    return target_epochs, distractor_epochs, non_target_target_epochs, non_target_distractor_epochs


a1_target_epochs, a1_distractor_epochs, a1_non_target_target_epochs, a1_non_target_distractor_epochs = get_epochs(condition=conditions[0])
a2_target_epochs, a2_distractor_epochs, a2_non_target_target_epochs, a2_non_target_distractor_epochs = get_epochs(condition=conditions[1])
e1_target_epochs, e1_distractor_epochs, e1_non_target_target_epochs, e1_non_target_distractor_epochs = get_epochs(condition=conditions[2])
e2_target_epochs, e2_distractor_epochs, e2_non_target_target_epochs, e2_non_target_distractor_epochs = get_epochs(condition=conditions[3])


# concatenate all epochs:
def concatenate_epochs(*epochs_dicts):
    all_epochs = []
    # Loop through each dictionary and extract the Epochs objects
    for epochs_dict in epochs_dicts:
        for epochs in epochs_dict.values():
            all_epochs.append(epochs)

    # Concatenate all epochs into a single Epochs object
    concatenated_epochs = mne.concatenate_epochs(all_epochs)
    return all_epochs, concatenated_epochs

# azimuth:
all_azimuth_target_epochs, concatenated_epochs_azimuth_target = concatenate_epochs(a1_target_epochs, a2_target_epochs)
all_azimuth_distractor_epochs, concatenated_epochs_azimuth_distractor = concatenate_epochs(a1_distractor_epochs, a2_distractor_epochs)
all_azimuth_non_target_target_epochs, concatenated_epochs_azimuth_non_target_target = concatenate_epochs(a1_non_target_target_epochs, a2_non_target_target_epochs)
all_azimuth_non_target_distractor_epochs, concatenated_epochs_azimuth_non_target_distractor = concatenate_epochs(a1_non_target_distractor_epochs, a2_non_target_distractor_epochs)

# elevation
all_elevation_target_epochs, concatenated_epochs_elevation_target = concatenate_epochs(e1_target_epochs, e2_target_epochs)
all_elevation_distractor_epochs, concatenated_epochs_elevation_distractor = concatenate_epochs(e1_distractor_epochs, e2_distractor_epochs)
all_elevation_non_target_target_epochs, concatenated_epochs_elevation_non_target_target = concatenate_epochs(e1_non_target_target_epochs, e2_non_target_target_epochs)
all_elevation_non_target_distractor_epochs, concatenated_epochs_elevation_non_target_distractor = concatenate_epochs(e1_non_target_distractor_epochs, e2_non_target_distractor_epochs)

# drop overlaps between non-targets:
def drop_overlaps(concat_epochs1, concat_epochs2):
    sfreq_non_target_target = concat_epochs1.info['sfreq']
    sfreq_non_target_distractor = concat_epochs2.info['sfreq']

    events_non_target_target = concatenated_epochs_azimuth_non_target_target.events[:, 0] / sfreq_non_target_target
    events_non_target_distractor = concatenated_epochs_azimuth_non_target_distractor.events[:,
                                   0] / sfreq_non_target_distractor

    # Find overlapping events (time difference ≤ 0.2 seconds)
    overlaps = []
    for time_target in events_non_target_target:
        for time_distractor in events_non_target_distractor:
            if abs(time_target - time_distractor) <= 0.2:
                overlaps.append(time_target)
                overlaps.append(time_distractor)

    # Get indices of overlapping events in each Epochs object
    overlap_indices_target = [i for i, t in enumerate(events_non_target_target) if t in overlaps]
    overlap_indices_distractor = [i for i, t in enumerate(events_non_target_distractor) if t in overlaps]

    # Drop overlapping events from both Epochs objects
    clean_epochs1 = concatenated_epochs_azimuth_non_target_target.drop(overlap_indices_target)
    clean_epochs2 = concatenated_epochs_azimuth_non_target_distractor.drop(overlap_indices_distractor)
    return clean_epochs1, clean_epochs2

# azimuth:
clean_concatenated_epochs_azimuth_non_target_target, clean_concatenated_epochs_azimuth_non_target_distractor = drop_overlaps(
    concatenated_epochs_azimuth_non_target_target, concatenated_epochs_azimuth_non_target_distractor)

# elevation:
clean_concatenated_epochs_elevation_non_target_target, clean_concatenated_epochs_elevation_non_target_distractor = drop_overlaps(
    concatenated_epochs_elevation_non_target_target, concatenated_epochs_elevation_non_target_distractor)


# get minimum length of events from ALL concatenated epochs combined:
concatenated_epochs = [concatenated_epochs_azimuth_target,
                      concatenated_epochs_azimuth_distractor,
                      clean_concatenated_epochs_azimuth_non_target_target,
                      clean_concatenated_epochs_azimuth_non_target_distractor,
                      concatenated_epochs_elevation_target,
                      concatenated_epochs_elevation_distractor,
                      clean_concatenated_epochs_elevation_non_target_target,
                      clean_concatenated_epochs_elevation_non_target_distractor]

np.random.seed(42)
all_lens = []

# Calculate the minimum number of events across all provided epochs
for epochs in concatenated_epochs:
    length = len(epochs.events)
    all_lens.append(length)
min_length = min(all_lens)  # Find the minimum length

# make all concatenated epochs of same length:
def crop_lengths(min_length, concatenated_epochs):
    # Crop each epochs object to the minimum length
    cropped_concat_epochs = []
    for concat_epochs in concatenated_epochs:
        cropped_epochs = concat_epochs[np.random.choice(len(concat_epochs), min_length, replace=False)]
        cropped_concat_epochs.append(cropped_epochs)
    return cropped_concat_epochs

cropped_concat_epochs = crop_lengths(min_length, concatenated_epochs)

def plot_psd(cropped_concat_epochs, plane='', fmin=1, fmax=150):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12), constrained_layout=True)
    axes = axes.flatten()
    # Compute global min and max
    global_min = float('inf') # initializing global_min with the largest possible value
    global_max = float('-inf')
    for i, (category, epoch) in enumerate(zip(epoch_categories, cropped_concat_epochs[:4])):
        ax = axes[i]
        psd = epoch.compute_psd(fmin=fmin, fmax=fmax)
        power, freqs = psd.get_data(return_freqs=True)
        power_mean = np.mean(power, axis=0).flatten() # convert to decibels
        # Find the smallest non-zero value
        min_non_zero = np.min(power_mean[power_mean > 0])
        # Compute the scale factor as the inverse of the order of magnitude
        scale_factor = 10 ** (-np.floor(np.log10(min_non_zero)))
        # Scale the power values dynamically
        power_scaled = power_mean * scale_factor
        global_min = min(global_min, np.min(power_scaled))
        global_max = max(global_max, np.max(power_scaled))
        ax.plot(freqs, power_scaled.T)
        ax.set_title(f'{plane} - {category}')
        ax.set_xlabel(f'Frequency {fmin}-{fmax} Hz', fontsize=10)
        # Convert scale factor to scientific notation with base 10
        order_of_magnitude = int(np.log10(scale_factor))
        label_scale = f'$10^{{{-order_of_magnitude}}}$'  # Format as LaTeX
        ax.set_ylabel(f'Power x {label_scale}')
        ax.grid(True)
    for ax in axes:
        ymin = int(np.round(global_min) - 1)
        ymax = int(np.round(global_max) + 1)
        ax.set_yticks(np.arange(ymin, ymax + 1, 10))
        ax.set_ylim(ymin, ymax)
    plt.savefig(results_path / f'{plane}_psd_{fmin}-{fmax}_hz.png')
    plt.close(fig)


plot_psd(cropped_concat_epochs[:4], plane='Azimuth', fmin=1, fmax=150)
plot_psd(cropped_concat_epochs[4:], plane='Elevation', fmin=1, fmax=150)

plot_psd(cropped_concat_epochs[:4], plane='Azimuth', fmin=1, fmax=40)
plot_psd(cropped_concat_epochs[4:], plane='Elevation', fmin=1, fmax=40)


# plot heatmaps:
def plot_heatmaps(epoch_categories, epoch_list=[], plane='', fmax=150):
    frequencies = np.linspace(1, fmax, 100)  # Frequencies from 1 to 30 Hz
    n_cycles = frequencies / 2  # Number of cycles in Morlet wavelet (adapts to frequency)

    n_cols = 2
    n_rows = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 12), constrained_layout=True, dpi=500)
    # Flatten axes for simpler indexing (if there's only one row or column)
    axes = axes.flatten()
    # Compute the TFR using Morlet wavelets
    # Loop over each epoch and compute its TFR
    tfa_dict = {}
    for i, (epochs, category) in enumerate(zip(epoch_list, epoch_categories)):
        tfa_dict[category] = {}
        # Compute the TFR using Morlet wavelets
        tfa = mne.time_frequency.tfr_morlet(
            epochs,
            freqs=frequencies,
            n_cycles=n_cycles,
            return_itc=False,
            average=True,
            decim=1,
            n_jobs=1
        )

        # Apply baseline correction
        tfa.apply_baseline(baseline=(-0.2, 0.0), mode='percent')
        power_data = tfa.data[0]
        # normalized_data = power_data / np.mean(power_data, axis=0)
        times = tfa.times
        freqs = tfa.freqs
        tfa_dict[category] = tfa

        # Plot the TFA heatmap on the current axis
        im = axes[i].imshow(power_data, extent=[times[0], times[-1], freqs[0], freqs[-1]], aspect='auto', cmap='viridis', origin='lower', vmin=-0.4, vmax=0.4
                               )
        axes[i].set_title(f'{category} - {plane}', fontsize=10)
        axes[i].set_xlabel('Time (s)', fontsize=8)
        axes[i].set_ylabel('Frequency (Hz)', fontsize=8)
        # Add a shared colorbar
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.8, label='Power %')
    # Save and display the plot
    plt.savefig(results_path / f'all_tfa_{plane}_{fmax}_Hz.png', dpi=300)
    plt.show()
    plt.close(fig)
    return tfa_dict


# TFA:
'''- by default, it computes absolute power at each time and frequency point
- If you provide logarithmically spaced bins (e.g., via np.logspace), 
 it calculates power for those log-spaced frequencies.
- This means there will be more bins at lower frequencies (denser sampling) compared to higher frequencies, 
which can cause visual dominance of lower frequencies in your plots.
- by applying baseline and selecting a mode, we transform the power data
'''
tfa_azimuth_dict = plot_heatmaps(epoch_categories, epoch_list=cropped_concat_epochs[:4], plane='azimuth', fmax=150)
tfa_elevation_dict = plot_heatmaps(epoch_categories, epoch_list=cropped_concat_epochs[4:], plane='elevation', fmax=150)

tfa_azimuth_dict_40 = plot_heatmaps(epoch_categories, epoch_list=cropped_concat_epochs[:4], plane='azimuth', fmax=40)
tfa_elevation_dict_40 = plot_heatmaps(epoch_categories, epoch_list=cropped_concat_epochs[4:], plane='elevation', fmax=40)

azimuth_tfa_list = []
for name, sub_dict in tfa_azimuth_dict.items():
    azimuth_tfa_list.append([sub_dict])

elevation_tfa_list = []
for name, sub_dict in tfa_elevation_dict.items():
    elevation_tfa_list.append([sub_dict])

azimuth_data_list = cropped_concat_epochs[:4]
elevation_data_list = cropped_concat_epochs[4:]

def plot_average_powers(epoch_categories, data_list, time_window=(0.0, 0.9), freq_range=(None, None), plane=''):
    # Plot Power vs. Time
    cols = 2
    rows = 2
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 12))
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    # set global min and max for y lim:
    global_min = float('inf')
    global_max = float('-inf')
    for i, (category, tfa_data) in enumerate(zip(epoch_categories, data_list)):
        time = tfa_data[0].times  # Extract time points
        freqs = tfa_data[0].freqs  # Extract frequency points
        # Find the indices for the desired frequency range
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        time_mask = (time >= time_window[0]) & (time <= time_window[1])

        # Average power over the selected frequency range
        power = tfa_data[0].data[:, freq_mask, :][:, :, time_mask].mean(axis=1).flatten()  # Shape: (n_channels, freqs, time)
        global_min = min(global_min, power.min())
        global_max = max(global_max, power.max())
        ax = axes[i]
        ax.plot(time[time_mask], power)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power')
        ax.set_title(f'{category} - Power x Time ({freq_range[0]}-{freq_range[1]} Hz)')
        ax.grid(True)
        ax.set_xticks(np.arange(time_window[0], time_window[1], 0.1))  # Set consistent time spacing
        ax.set_yticks(np.arange(int(global_min), int(global_max)+1))  # Set consistent power spacing
    for ax in axes:
        ax.set_ylim(global_min -0.5, global_max +1)
    plt.tight_layout()
    save_path = results_path / f'Power_over_time_{plane}_{freq_range[0]}_{freq_range[1]}_Hz_{time_window[1]}_seconds.png'
    plt.savefig(save_path)
    plt.close(fig)

# plot_average_powers(epoch_categories, data_list=azimuth_tfa_list, time_window=(0.0, 0.9), freq_range=(1, 150), plane='azimuth')
# plot_average_powers(epoch_categories, data_list=elevation_tfa_list, time_window=(0.0, 0.9), freq_range=(1, 150), plane='elevation')

plot_average_powers(epoch_categories, data_list=azimuth_tfa_list, time_window=(0.0, 0.9), freq_range=(1, 40), plane='azimuth')
plot_average_powers(epoch_categories, data_list=elevation_tfa_list, time_window=(0.0, 0.9), freq_range=(1, 40), plane='elevation')

plot_average_powers(epoch_categories, data_list=azimuth_tfa_list, time_window=(0.0, 0.9), freq_range=(1, 150), plane='azimuth')
plot_average_powers(epoch_categories, data_list=elevation_tfa_list, time_window=(0.0, 0.9), freq_range=(1, 150), plane='elevation')
def get_amplitudes(data_list=[], plane=''):
    amplitudes = {'Category': [], 'Absolute Amplitudes (muV)': []}
    max_vals = []  # use to set ymax
    min_vals = []  # use to set ymin
    for category, epochs in zip(epoch_categories, data_list):
        epoch_data = (epochs._data * 1e6)
        mean_across_epochs = np.mean(epoch_data, axis=0)
        abs_means = np.abs(mean_across_epochs).flatten()
        max_val = np.max(abs_means)
        max_vals.append(max_val)
        min_val = np.min(abs_means)
        min_vals.append(min_val)
        amplitudes['Category'].extend([category] * len(abs_means))
        amplitudes['Absolute Amplitudes (muV)'].extend(abs_means)
    amplitudes_df = pd.DataFrame(amplitudes)
    plt.figure()
    sns.boxplot(data=amplitudes_df, x='Category',
                            y='Absolute Amplitudes (muV)', hue='Category',
                            palette=['royalblue', 'darkviolet', 'yellow', 'orange'])
    plt.ylabel('Amplitudes (muV)')
    plt.xlabel('Epoch Category')
    plt.title(f'{plane} - Absolute Amplitude Distribution')
    plt.savefig(results_path / f'{plane}_amplitude_distributions_all.png')
    plt.close()

get_amplitudes(data_list=azimuth_data_list, plane='Azimuth')
get_amplitudes(data_list=elevation_data_list, plane='Elevation')





