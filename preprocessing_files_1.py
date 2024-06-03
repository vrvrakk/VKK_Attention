'''
### STEP 1: Concatenate block files to one raw file in raw_folder ###
### STEP 2: bandpass filtering of the data at 1=40 Hz ###
### STEP 3: Apply ICA ###
### STEP 4: Epoch the raw data and apply baseline ###
### STEP 5: Run RANSAC/ exclude bad channels ###
### STEP 6: Rereference the epochs ###
### STEP 7: Apply AutoReject ###
### STEP 8: Average epochs and write evokeds###
'''
# libraries:
import mne
from pathlib import Path
import os
import numpy as np
from autoreject import AutoReject, Ransac
from collections import Counter
import json
from meegkit import dss
from matplotlib import pyplot as plt, patches
from helper import grad_psd, snr

sub_input = input("Give sub number as subn (n for number): ")
sub = [sub.strip() for sub in sub_input.split(',')]
cm = 1 / 2.54
name = ''
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
# to read file:
# file_path
# mne.read_epochs(file_path)
# events:
markers_dict = {
    's1_events': {'Stimulus/S 1': 1, 'Stimulus/S 2': 2, 'Stimulus/S 3': 3, 'Stimulus/S 4': 4, 'Stimulus/S 5': 5,
                  'Stimulus/S 6': 6, 'Stimulus/S 8': 8, 'Stimulus/S 9': 9},
    # stimulus 1 markers
    's2_events': {'Stimulus/S 72': 18, 'Stimulus/S 73': 19, 'Stimulus/S 65': 11, 'Stimulus/S 66': 12,
                  'Stimulus/S 69': 15, 'Stimulus/S 70': 16, 'Stimulus/S 68': 14,
                  'Stimulus/S 67': 13},  # stimulus 2 markers
    'response_events': {'Stimulus/S132': 24, 'Stimulus/S130': 22, 'Stimulus/S134': 26, 'Stimulus/S137': 29,
                        'Stimulus/S136': 28, 'Stimulus/S129': 21, 'Stimulus/S131': 23,
                        'Stimulus/S133': 25}}  # response markers
s1_events = markers_dict['s1_events']
s2_events = markers_dict['s2_events']
both_stim = [s1_events, s2_events]
response_events = markers_dict['response_events']

# config files
with open(json_path / "preproc_config.json") as file:
    cfg = json.load(file)
with open(json_path / "electrode_names.json") as file:
    mapping = json.load(file)

# Run pre-processing steps:
condition = input('Please provide condition (exp. EEG): ')
axis = input('Please provide axis (exp. EEG): ')


### STEP 0: Concatenate block files to one raw file in raw_folder
def choose_header_files(condition=condition, axis=axis):
    target_header_files_list = []
    for sub_dir in sub_dirs:
        header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
        filtered_files = [file for file in header_files if condition in file]
        filt_files = [file for file in filtered_files if axis in file]
        if filt_files:
            target_header_files_list.append(filt_files)
    return target_header_files_list, condition, axis


def get_raw_files(target_header_files_list, condition, axis):
    raw_files = []
    for sub_dir, header_files in zip(sub_dirs, target_header_files_list):
        for header_file in header_files:
            full_path = os.path.join(sub_dir, header_file)
            print(full_path)
            raw_files.append(mne.io.read_raw_brainvision(full_path, preload=True))
    raw = mne.concatenate_raws(raw_files)  # read BrainVision files.
    # append all files from a participant
    raw.rename_channels(mapping)
    # Use BrainVision montage file to specify electrode positions.
    raw.set_montage("standard_1020")
    raw.save(raw_fif / f"{name}_{condition}_{axis}_raw.fif", overwrite=True)  # here the data is saved as raw
    print(f'{condition} raw data saved. If raw is empty, make sure axis and condition are filled in correctly.')
    return raw


def get_events(raw, target_events):
    events = mne.events_from_annotations(raw)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in target_events.values()]
    events = np.array(filtered_events)
    return events


# 2. Interpolate
def interpolate(raw, condition):
    raw_interp = raw.copy().interpolate_bads(reset_bads=True)
    raw_interp.plot()
    raw.save(raw_fif / f"{sub}_{condition}_{axis}_interpolated.fif", overwrite=True)
    return raw_interp


# 3. HIGH- AND LOW-PASS FILTER + POWER NOISE REMOVAL
def filtering(raw, data):
    cfg["filtering"]["notch"] = 50
    # remove the power noise
    raw_filter = raw.copy()
    raw_notch, iterations = dss.dss_line(raw_filter.get_data().T, fline=cfg["filtering"]["notch"],
                                         sfreq=data.info["sfreq"],
                                         nfft=cfg["filtering"]["nfft"])

    raw_filter._data = raw_notch.T
    cfg["filtering"]["highpass"] = 1
    hi_filter = cfg["filtering"]["highpass"]
    lo_filter = cfg["filtering"]["lowpass"]

    raw_filtered = raw_filter.copy().filter(hi_filter, lo_filter)
    raw_filtered.plot()

    # plot the filtering
    grad_psd(raw, raw_filter, raw_filtered, fig_path)
    return raw, raw_filter, raw_filtered



# Run pre-processing steps:

target_header_files_list, condition, axis = choose_header_files()

target_raw = get_raw_files(target_header_files_list, condition, axis)

events1 = get_events(target_raw, s1_events)
events2 = get_events(target_raw, s2_events)
events3 = get_events(target_raw, response_events)

# to select bad channels, and select bad segmenmts:
target_raw.plot()
target_raw.plot_psd()

# get annotations info:
onsets = target_raw.annotations.onset
durations = target_raw.annotations.duration
descriptions = target_raw.annotations.description

# Find good segments
good_intervals = []
last_good_end = 0
for onset, duration, description in zip(onsets, durations, descriptions):
    if description == 'BAD_':  # description name may vary for each file (Bad boundary)
        good_intervals.append((last_good_end, onset))
        last_good_end = onset + duration
# Add the final good segment
good_intervals.append((last_good_end, target_raw.times[-1]))

# Crop and concatenate good segments
good_segments = [target_raw.copy().crop(tmin=start, tmax=end) for start, end in good_intervals]
target_raw = mne.concatenate_raws(good_segments)

# interpolate bad selected channels, after removing significant noise affecting many electrodes
target_interp = interpolate(target_raw, condition)

# get raw array, and info
target_data = mne.io.RawArray(data=target_interp.get_data(), info=target_interp.info)

# Filter: bandpas 1-40Hz
target_raw, target_filter, target_filtered = filtering(target_interp, target_data)
target_filtered.save(results_path / f'1-40Hz for {name}, conditions: {condition}, {axis} -raw.fif', overwrite=True)

############ from here on concatenated data are further processed


