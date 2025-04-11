import json
from pathlib import Path
import mne
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

# Step 1. get epochs
def get_epochs(condition=''):
    subs_epochs = []
    for folders in epochs_path.iterdir():
        if 'sub' in folders.name and folders.name != 'all_subs':
            subject_path = epochs_path / folders.name / 'attention' / epoch_type
            for sub_folder in subject_path.iterdir():
                if epoch_type in sub_folder.name and condition in sub_folder.name:
                    epoch = mne.read_epochs(sub_folder, preload=True)
                    epoch.set_eeg_reference('average')
                    subs_epochs.append(epoch)
    return subs_epochs

# 24 for azimuth, 18 for elevation


# Step 2: get TFR per sub
def get_subject_TFR(subs_epochs, roi=None):
    epochs_tfr = []
    for epochs in subs_epochs:
        epochs.pick(roi)
        sub_tfrs = []
        power = mne.time_frequency.tfr_multitaper(
            epochs,  # <-- all trials
            freqs=freqs,
            n_cycles=n_cycles,
            average=False,  # <-- keeps all trial TFRs
            return_itc=False,
            decim=4,
            n_jobs=1
        )
        power.apply_baseline(baseline=(-0.2, 0), mode='logratio')
        power.data = power.data.astype(np.float32)
        # Then average across all trials
        avg_tfr = power.average()  # Returns an AverageTFR object
        sub_tfrs.append(avg_tfr)
        epochs_tfr.append(sub_tfrs)
    return epochs_tfr

def get_subject_ITC(subs_epochs, roi=None):
    epochs_itc = []
    for epochs in subs_epochs:
        epochs.pick(roi)
        power, itc = mne.time_frequency.tfr_multitaper(
            epochs,  # all trials at once
            freqs=freqs,
            n_cycles=n_cycles,
            average=True,
            return_itc=True,
            decim=4,
            n_jobs=1
        )
        itc.data = itc.data.astype(np.float32)
        epochs_itc.append(itc)
    return epochs_itc


# Step 3. Make an AverageTFR object for each sub
from mne import grand_average

def get_induced_oscillations(epoch_type, epochs_tfr=None, epochs_itc=None, condition='', region=''):
    induced_tfr = []
    induced_itc = []

    save_path = Path(
        'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/concatenated_data/epochs/all_subs'
    )
    fig_dir = save_path / 'induced_tfa' / 'figures' / epoch_type / condition / region
    tfr_data_path = save_path / 'induced_tfa' / 'data' / 'averaged_subs' / epoch_type / condition / region
    itc_data_path = save_path / 'ITC' / 'data' / 'averaged_subs' / epoch_type / condition / region
    itc_fig_dir = save_path / 'ITC' / 'figures' / epoch_type / condition / region

    fig_dir.mkdir(parents=True, exist_ok=True)
    tfr_data_path.mkdir(parents=True, exist_ok=True)
    itc_data_path.mkdir(parents=True, exist_ok=True)
    itc_fig_dir.mkdir(parents=True, exist_ok=True)

    for index, (sub_tfr, sub_itc) in enumerate(zip(epochs_tfr, epochs_itc)):
        # sub_tfr is a list of trial-level AverageTFRs → needs grand_average
        avg_tfr = sub_tfr[0]

        # sub_itc is already a subject-level AverageTFR → no need for grand_average
        avg_itc = sub_itc

        # Plot and save TFR
        tfr_fig = avg_tfr.plot(fmin=1, fmax=30, cmap='viridis', combine='mean', show=False)
        tfr_fig_file = fig_dir / f'{condition}_{epoch_type}_{index}.png'
        tfr_fig[0].savefig(tfr_fig_file, dpi=300)
        plt.close(tfr_fig[0])

        # Plot and save ITC
        itc_fig = avg_itc.plot(fmin=1, fmax=30, combine='mean', show=False)
        itc_fig_file = itc_fig_dir / f'{condition}_{epoch_type}_{index}_itc.png'
        itc_fig[0].savefig(itc_fig_file, dpi=300)
        plt.close(itc_fig[0])

        # Save both objects
        avg_tfr.save(tfr_data_path / f'{condition}_{epoch_type}_{index}-tfr.h5', overwrite=True)
        avg_itc.save(itc_data_path / f'{condition}_{epoch_type}_{index}_itc-tfr.h5', overwrite=True)

        induced_tfr.append(avg_tfr)
        induced_itc.append(avg_itc)

    return induced_tfr, induced_itc

# Extract frequency and time indices
def get_indices(tfr_obj, fmin, fmax, tmin, tmax):
    freq_mask = (tfr_obj.freqs >= fmin) & (tfr_obj.freqs <= fmax)
    time_mask = (tfr_obj.times >= tmin) & (tfr_obj.times <= tmax)
    return freq_mask, time_mask


    # Extract metrics per subject
def extract_theta_metrics(tfr_list, itc_list):
    results = []

    for subj_id, (tfr, itc) in enumerate(zip(tfr_list, itc_list)):
        freq_mask, time_mask = get_indices(tfr, *theta_range, *time_window)

        # Average across channels, time, and frequencies
        mean_power = tfr.data[:, freq_mask, :][:, :, time_mask].mean()
        mean_itc = itc.data[:, freq_mask, :][:, :, time_mask].mean()

        # Get ITC peak frequency (collapsed over channels and time)
        itc_avg = itc.data[:, freq_mask, :][:, :, time_mask].mean(axis=(0, 2))  # mean over channels and time
        peak_freq = itc.freqs[freq_mask][np.argmax(itc_avg)]

        results.append({
            "subject": subj_id,
            "mean_theta_power": mean_power,
            "mean_theta_ITC": mean_itc,
            "peak_theta_freq": peak_freq
        })

    return pd.DataFrame(results)



if __name__ == '__main__':
    ''' main goal would be:
        - is there a difference in the way distractor and target are tracked?
        - will I see consistent patterns inspite of ISI variation?
        - guesses/predictions:
            - higher theta power for target stream
            - target stream tracked by a higher theta band than distractor
            - ITC for target stream will be smaller
            - ITC for distractor may be higher
            - HOWEVER, theta band tracking distractor will be lower '''
    default_path = Path.cwd()
    epochs_path = default_path / 'data/eeg/preprocessed/results/concatenated_data/epochs'
    concat_epochs = epochs_path / 'all_subs'

    # define ROIs
    # Frontal ROI (including prefrontal and frontal midline)
    frontal_roi = ['F3', 'Fz', 'F4', 'FC1', 'FC2']
    epochs_all_names = []
    for fif_files in concat_epochs.iterdir():
        if 'fif' in fif_files.name:
            file_name = fif_files.name[13:-8]
            epochs_all_names.append(file_name)
    # TFA each epoch:
    freqs = np.logspace(np.log10(1), np.log10(30), num=100)  # 30 log-spaced frequencies
    n_cycles = freqs / 2  # Define cycles per frequency (standard approach)

    condition = 'e2'
    index = 4  # 3, 4 (non-targets)
    roi = frontal_roi
    epochs_names = np.unique(epochs_all_names)
    epoch_type = epochs_names[index]

    subs_epochs = get_epochs(condition=condition)
    epochs_tfr = get_subject_TFR(subs_epochs, roi=roi)
    epochs_itc = get_subject_ITC(subs_epochs, roi=roi)
    induced_tfr, induced_itc = get_induced_oscillations(epoch_type, epochs_tfr, epochs_itc, condition=condition, region='frontal')

    # Define theta band range and time window
    theta_range = (4, 8)  # Hz
    time_window = (0.0, 0.4)  # seconds

    theta_df = extract_theta_metrics(induced_tfr, induced_itc)
    theta_df.head()
    csv_path = concat_epochs / 'stats'
    theta_df.to_csv(csv_path/f'{condition}_{epoch_type}_theta_frontal.csv', sep=';')


#     # Occipitoparietal ROI (posterior regions)
#     # Left hemisphere electrodes
#     occipitoparietal_left = [
#         'P7', 'P3',
#         'PO7', 'PO3',
#         'O1', 'PO9'
#     ]
#     # Right hemisphere electrodes
#     occipitoparietal_right = [
#         'P8', 'P4',
#         'PO8', 'PO4',
#         'O2', 'PO10'
#     ]