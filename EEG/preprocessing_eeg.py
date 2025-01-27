import os
from pathlib import Path
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

json_path = default_path / 'data' / 'misc'
with open(json_path / "electrode_names.json") as file: #electrode_names
    mapping = json.load(file)

results_path = default_path / 'data' / 'eeg' / 'preprocessed'/ 'results'/ 'concatenated_data'/ 'continuous'
os.makedirs(results_path, exist_ok=True)

# 1. load header files:
a1_eeg_header_files = EEG.extract_events.extract_eeg_files(condition='a1')
# a2_eeg_header_files = EEG.extract_events.extract_eeg_files(condition='a2')
# e1_eeg_header_files = EEG.extract_events.extract_eeg_files(condition='e1')
# e2_eeg_header_files = EEG.extract_events.extract_eeg_files(condition='e2')


# 2. load eeg files from headers:
a1_eeg = EEG.extract_events.load_eeg_files(a1_eeg_header_files)
# a2_eeg = EEG.extract_events.load_eeg_files(a2_eeg_header_files)
# e1_eeg = EEG.extract_events.load_eeg_files(e1_eeg_header_files)
# e2_eeg = EEG.extract_events.load_eeg_files(e2_eeg_header_files)


# 3. concatenate all eeg files of one sub into a list:
def create_sub_eeg_list(eeg):
    concatenated_eeg_files_list = {}
    for eeg_files in eeg:
        sub_name = eeg_files.filenames[0].split('\\')[-2]
        if sub_name not in concatenated_eeg_files_list:
            concatenated_eeg_files_list[sub_name] = [] #initialize with empty list, not dictionary
        concatenated_eeg_files_list[sub_name].append(eeg_files)
    return concatenated_eeg_files_list


a1_concatenated_eeg_files_list = create_sub_eeg_list(a1_eeg)
# a2_concatenated_eeg_files_list = create_sub_eeg_list(a2_eeg)
# e1_concatenated_eeg_files_list = create_sub_eeg_list(e1_eeg)
# e2_concatenated_eeg_files_list = create_sub_eeg_list(e2_eeg)


# 4. concatenate eeg files using mne.concatenate
def concatenate_eeg_files(concatenated_eeg_files_list, condition=''):
    concatenated_eeg_files = {}
    for sub, eeg_list in concatenated_eeg_files_list.items():
        concatenated_eeg_files[sub] = []
        eeg_concat = mne.concatenate_raws(eeg_list)
        concatenated_eeg_files[sub] = eeg_concat
        eeg_concat.save(results_path / f"{sub}_{condition}_concatenated-raw.fif", overwrite=True)
    return concatenated_eeg_files


a1_concatenated_eeg_files = concatenate_eeg_files(a1_concatenated_eeg_files_list, condition='a1')
# a2_concatenated_eeg_files = concatenate_eeg_files(a2_concatenated_eeg_files_list)
# e1_concatenated_eeg_files = concatenate_eeg_files(e1_concatenated_eeg_files_list)
# e2_concatenated_eeg_files = concatenate_eeg_files(e2_concatenated_eeg_files_list)

# 5. investigate EEG signal manually and select bad channels and segments:
for sub, concat_eeg in a1_concatenated_eeg_files.items():
    concat_eeg.plot()
    plt.show()  # Waits until the current plot window is closed





# 6. apply montage and add FCz channel:
def add_montage(concatenated_eeg_files):
    for sub, concat_eeg in concatenated_eeg_files.items():
        concat_eeg.resample(sfreq=500)  # downsample from 1000Hz to 500Hz
        concat_eeg.rename_channels(mapping)
        concat_eeg.add_reference_channels('FCz')  # add reference channel
        concat_eeg.set_montage('standard_1020')  # apply standard montage
        concat_eeg.drop_channels(['A1', 'A2', 'M2'])
    return concatenated_eeg_files


# 7. drop bad segments
# get annotations info:
def drop_bad_segments(concatenated_eeg_files):
    cleaned_eeg_files = {}
    for sub, concat_eeg in concatenated_eeg_files.items():
        cleaned_eeg_files[sub] = []
        onsets = concat_eeg.annotations.onset
        durations = concat_eeg.annotations.duration
        descriptions = concat_eeg.annotations.description

        # Find good segments
        good_intervals = []
        last_good_end = 0
        for onset, duration, description in zip(onsets, durations, descriptions):
            if 'BAD' in description or 'bad' in description:  # description name may vary for each file (Bad boundary)
                good_intervals.append((last_good_end, onset))
                last_good_end = onset + duration
        # Add the final good segment
        good_intervals.append((last_good_end, concat_eeg.times[-1]))

        # Crop and concatenate good segments
        good_segments = [concat_eeg.copy().crop(tmin=start, tmax=end) for start, end in good_intervals]
        cleaned_concat_eeg = mne.concatenate_raws(good_segments)
        cleaned_eeg_files[sub] = cleaned_concat_eeg
    return cleaned_eeg_files


# 8. interpolate bad channels:
def interpolate_eeg(cleaned_eeg_files, condition=''):
    interp_eeg_files = {}
    for sub, eeg in cleaned_eeg_files.items():
        interp_eeg_files[sub] = []
        interp_eeg = concat_eeg.interpolate_bads(reset_bads=True)
        interp_eeg_files[sub] = interp_eeg
        interp_eeg.save(results_path / f"{sub}_{condition}_interpolated-raw.fif", overwrite=True)
    return interp_eeg_files


# 9. 50Hz notch & bandpass filter with range of choice:
def filter_eeg(interp_eeg_files, freq_range=(1, 30, 1)):
    filtered_eeg_files = {}
    for sub, eeg_files in interp_eeg_files.items():
        filtered_eeg_files[sub] = []
        eeg_filter = eeg_files.copy()
        data = mne.io.RawArray(data=eeg_files.get_data(), info=eeg_files.info)
        eeg_notch, iterations = dss.dss_line(eeg_files.get_data().T, fline=50,
                                             sfreq=data.info["sfreq"],
                                             nfft=400)

        eeg_filter._data = eeg_notch.T
        hi_filter = freq_range[0]
        lo_filter = freq_range[1]

        eeg_filtered = eeg_filter.copy().filter(hi_filter, lo_filter)
        filtered_eeg_files[sub] = eeg_filtered
    return filtered_eeg_files

# 10. apply ICA:
def apply_ica(filtered_eeg_files):
    ica_eeg_files = {}
    for sub, eeg_file in filtered_eeg_files:
        ica_eeg_files[sub] = []
        eeg_ica = eeg_file.copy()
        eeg_ica.info['bads'].append('FCz')  # Add FCz to the list of bad channels
        ica = mne.preprocessing.ICA(n_components=0.999, method='fastica', random_state=99)
        ica.fit(eeg_ica)
        ica.plot_components()
        plt.show()
        plt.close()

        ica.plot_sources(eeg_ica)
        plt.show()
        plt.close()

        ica.apply(eeg_ica)
        eeg_ica.info['bads'].remove('FCz')
        eeg_ica.set_eeg_reference(ref_channels='average')
        ica_eeg_files[sub] = eeg_ica
    return ica_eeg_files

# 11a. pick channels of interest:
def pick_channels(ica_eeg_files, focus=''):
    eeg_files_selected_chs = {}
    for sub, eeg_files in ica_eeg_files:
        eeg_files_selected_chs[sub] = []
        if focus == 'a':
            occipital_channels = ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8", "P5", "P6", "P7", "P8"]
            print('Further processing focused on attention. Selecting channels accordingly...')
            eeg_files.drop_channels(occipital_channels)
            eeg_files_selected_chs[sub] = eeg_files
        elif focus == 'm':
            motor_channels = ['C3', 'C4', 'CP3', 'CP4', 'Cz', 'FC3', 'FC4']
            print('Further processing on motor responses. Selecting channels accordingly...')
            eeg_files.picks(motor_channels)
            eeg_files_selected_chs[sub] = eeg_files
    return eeg_files_selected_chs
# 11b. IF attentional focus: subtract motor response from EEG signal:
# for this, create a mega motor-response ERP with smooth edges, and subtract.



# 12: use events to create epochs:


