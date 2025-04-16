import os
from pathlib import Path
import pandas as pd
import numpy as np
import mne


sub = 'sub01'
condition = 'a1'
stream1_label = 'target_stream'
stream2_label = 'distractor_stream'
default_path = Path.cwd()
# load eeg files:
results_path = default_path / 'data/eeg/preprocessed/results'
sfreq = 125
def load_eeg_files(sub='', condition=''):
    eeg_path = results_path / f'{sub}/ica'
    eeg_files_list = []
    for sub_files in eeg_path.iterdir():
        if '.fif' in sub_files.name:
            if condition in sub_files.name:
                eeg_file = mne.io.read_raw_fif(sub_files, preload=True)
                eeg_file.set_eeg_reference('average')
                eeg_file.resample(sfreq=sfreq)
                eeg_files_list.append(eeg_file)
    return eeg_files_list

eeg_files_list = load_eeg_files(sub=sub, condition=condition)

# get overlap ratios:
# if complete overlap = 1
# none = 0
# partial = between 0-1
# if target was first: +
# if distractor was first: -
# use later to shape the attention predictor

# load event arrays:
predictors_path = default_path / 'data' / 'eeg' / 'predictors'
events_path = predictors_path / 'streams_events'
sub_path = events_path / sub / condition
stream1 = []
stream2 = []
for event_arrays in sub_path.iterdir():
    if 'stream1' in event_arrays.name:
        events = np.load(event_arrays)
        stream1.append(events)
    elif 'stream2' in event_arrays.name:
        events = np.load(event_arrays)
        stream2.append(events)

sfreq = 125
stim_dur = 0.745
for i, (event_array1, event_array2) in enumerate(zip(stream1, stream2)):
    for event1 in event_array1:
        onset1 = event1[0] / sfreq
        stim1_time_window = onset1 + stim_dur
    for event2 in event_array2:
        onset2 = event2[0] / sfreq
        stim2_time_window = onset2 + stim_dur
    overlap = stim1_time_window - stim2_time_window

