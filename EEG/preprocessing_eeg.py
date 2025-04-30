import os
from pathlib import Path
import copy
import mne
from meegkit import dss
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import json
import EEG.extract_events
from EEG.extract_events import default_path, eeg_path, blocks_path, sub_list
from autoreject import AutoReject, Ransac

json_path = default_path / 'data' / 'misc'
with open(json_path / "electrode_names.json") as file:  # electrode_names
    mapping = json.load(file)

concat_eeg_path = default_path / 'data' / 'eeg' / 'preprocessed' / 'results'/ 'concatenated_data'/ 'epochs'
single_eeg_path = default_path / 'data' / 'eeg' / 'preprocessed'/ 'results'
actual_mapping = {'New Segment/': 99999,
                  'Stimulus/S  1': 1,
                  'Stimulus/S  2': 2,
                  'Stimulus/S  3': 3,
                  'Stimulus/S  4': 4,
                  'Stimulus/S  5': 5,
                  'Stimulus/S  6': 6,
                  'Stimulus/S  8': 8,
                  'Stimulus/S  9': 9,
                  'Stimulus/S 64': 64,
                  'Stimulus/S 65': 65,
                  'Stimulus/S 66': 66,
                  'Stimulus/S 67': 67,
                  'Stimulus/S 68': 68,
                  'Stimulus/S 69': 69,
                  'Stimulus/S 70': 70,
                  'Stimulus/S 71': 71,
                  'Stimulus/S 72': 72,
                  'Stimulus/S 73': 73,
                  'Stimulus/S129': 129,
                  'Stimulus/S130': 130,
                  'Stimulus/S131': 131,
                  'Stimulus/S132': 132,
                  'Stimulus/S133': 133,
                  'Stimulus/S134': 134,
                  'Stimulus/S136': 136,
                  'Stimulus/S137': 137
                  }
conditions = ['a1', 'a2', 'e1', 'e2']

# 3. concatenate all eeg files of one sub into a list:
def create_sub_eeg_list(eeg):
    eeg_files_list = {}
    for eeg_files in eeg:
        sub_name = eeg_files.filenames[0].split('\\')[-2]
        if sub_name not in eeg_files_list:
            eeg_files_list[sub_name] = [] #initialize with empty list, not dictionary
        eeg_files_list[sub_name].append(eeg_files)
    return eeg_files_list


# 5. investigate EEG signal manually and select bad channels and segments:
# change sub name accordingly and investigate...


# 6. apply montage and add FCz channel:
def add_montage(target_eeg_files, condition=''):
    for index, eeg in enumerate(target_eeg_files):
        eeg.resample(sfreq=500)  # downsample from 1000Hz to 500Hz
        if eeg.info['ch_names'] == list(mapping.keys()):
            eeg.rename_channels(mapping)
            eeg.add_reference_channels('FCz')  # add reference channel
            if eeg.get_montage() is None:
                eeg.set_montage('standard_1020')  # apply standard montage
                eeg.drop_channels(['A1', 'A2', 'M2'])
            os.makedirs(single_eeg_path/sub/'investigated', exist_ok=True)
            eeg.save(single_eeg_path/sub/'investigated'/ f'{sub}_{condition}_{index}_investigated-raw.fif', overwrite=True)
            target_eeg_files[index] = eeg
    return target_eeg_files


# 7. interpolate bad channels:
def interpolate_eeg(target_eeg_files_to_interp, condition='', sub=''):
    for index, concat_eeg in enumerate(target_eeg_files_to_interp):
        interp_eeg = concat_eeg.interpolate_bads(reset_bads=True)
        os.makedirs(single_eeg_path / sub / 'interpolated', exist_ok=True)
        interp_eeg.save(single_eeg_path / sub / 'interpolated' / f"{sub}_{condition}_{index}_interpolated-raw.fif", overwrite=True)
        target_eeg_files_to_interp[index] = interp_eeg
    return target_eeg_files_to_interp


# 8. 50Hz notch & bandpass filter with range of choice:
def filter_eeg(target_eeg_files_filter, freq_range=(1, 30, 1), condition=''):
    for index, eeg_files in enumerate(target_eeg_files_filter):
        eeg_filter = eeg_files.copy()
        data = mne.io.RawArray(data=eeg_files.get_data(), info=eeg_files.info)
        eeg_notch, iterations = dss.dss_line(eeg_files.get_data().T, fline=50,
                                             sfreq=data.info["sfreq"],
                                             nfft=400)

        eeg_filter._data = eeg_notch.T
        hi_filter = freq_range[0]
        lo_filter = freq_range[1]

        eeg_filtered = eeg_filter.copy().filter(hi_filter, lo_filter)
        os.makedirs(single_eeg_path/ sub / 'filtered', exist_ok=True)
        eeg_filtered.save(single_eeg_path/ sub / 'filtered' / f'{sub}_{condition}_concatenated_filtered_{freq_range[1]}-raw.fif', overwrite=True)
        target_eeg_files_filter[index] = eeg_filtered
    return target_eeg_files_filter

# 9. apply ICA:
# see below

# run script:
if __name__ == "__main__":
    # 1:
    condition = conditions[0]
    eeg_header_files = EEG.extract_events.extract_eeg_files(condition=condition)
    eeg_files = EEG.extract_events.load_eeg_files(eeg_header_files)
    eeg_files_list = create_sub_eeg_list(eeg_files)
    ######################
    ######################
    sub = 'sub11'  # 11, 15, 19, 20, 28
    target_eeg_files = eeg_files_list[sub]
    for eeg_file in target_eeg_files:
        data = eeg_file.get_data()  # shape: (n_channels, n_times)

        signal_power = np.var(np.mean(data, axis=0))  # mean across channels → shape: (n_times,)
        total_power = np.mean(np.var(data, axis=1))  # mean variance across channels

        snr_ratio = signal_power / (total_power - signal_power)
        print("SNR ratio:", snr_ratio)
        eeg_file.plot()
        # eeg_file.plot_psd()
    # 2: mark bad segments and channels
    target_eeg_files_marked = add_montage(target_eeg_files, condition=condition)
    # 3:
    target_eeg_files_filter = copy.deepcopy(target_eeg_files_marked)
    eeg_files_filtered = filter_eeg(target_eeg_files_filter, freq_range=(1, 30, 1), condition=condition)
    for eeg_file in eeg_files_filtered:
        data = eeg_file.get_data()  # shape: (n_channels, n_times)

        signal_power = np.var(np.mean(data, axis=0))  # mean across channels → shape: (n_times,)
        total_power = np.mean(np.var(data, axis=1))  # mean variance across channels

        snr_ratio = signal_power / (total_power - signal_power)
        print("SNR ratio:", snr_ratio)
        # eeg_file.plot()
        eeg_file.info['bads'].append('FCz')  # Add FCz to the list of bad channels
        eeg_file.plot_psd()
    # 4: interpolate bad channels
    target_eeg_files_to_interp = copy.deepcopy(eeg_files_filtered)
    interp_eeg_files = interpolate_eeg(target_eeg_files_to_interp, condition=condition, sub=sub)
    # 5: ICA
    ica_eeg_files = copy.deepcopy(interp_eeg_files)
    ##################
    index = 0  # repeat ICA application for all indices from 0-4.
    ##################
    # a. fit ICA:
    eeg_file = ica_eeg_files[index]  # change variable according to condition
    data = eeg_file.get_data()  # shape: (n_channels, n_times)

    signal_power = np.var(np.mean(data, axis=0))  # mean across channels → shape: (n_times,)
    total_power = np.mean(np.var(data, axis=1))  # mean variance across channels
    snr_ratio = signal_power / (total_power - signal_power)
    print("SNR ratio:", snr_ratio)
    eeg_ica = eeg_file.copy()
    eeg_ica.info['bads'].append('FCz')  # Add FCz to the list of bad channels
    ica = mne.preprocessing.ICA(n_components=0.999, method='picard', random_state=99)
    ica.fit(eeg_ica)  # bad segments that were marked in the EEG signal will be excluded.
    # b. investigate...:
    ica.plot_components()
    ica.plot_sources(eeg_ica)
    # c. apply ICA to remove selected components: blinks, eye movements etc.
    ica.apply(eeg_ica)
    # d. re-reference with average:
    eeg_ica.info['bads'].remove('FCz')
    data_ica = eeg_ica.get_data()  # shape: (n_channels, n_times)

    signal_power_ica = np.var(np.mean(data_ica, axis=0))  # mean across channels → shape: (n_times,)
    total_power_ica = np.mean(np.var(data_ica, axis=1))  # mean variance across channels
    snr_ratio_ica = signal_power_ica / (total_power_ica - signal_power_ica)
    print("SNR ratio:", snr_ratio_ica)

    eeg_ica.set_eeg_reference(ref_channels='average')
    # e. save
    ica_eeg_files[index] = eeg_ica
    os.makedirs(single_eeg_path / sub / 'ica', exist_ok=True)
    eeg_ica.save(single_eeg_path / sub / 'ica' / f'{sub}_{condition}_{index}_ica-raw.fif', overwrite=True)
    if index < 4:
        index += 1
    else:
        index = 0


