from pathlib import Path
import os
import numpy as np
import mne
from EEG.extract_events import sub_list, default_path


results_path = default_path / 'data' / 'eeg' / 'preprocessed' / 'results'
concat_path = results_path / 'concatenated_data' / 'epochs'
occipital_channels = ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8", "P5", "P6", "P7", "P8"]
motor_channels = ['C3', 'C4', 'CP3', 'CP4', 'Cz', 'FC3', 'FC4']
# time to subtract baseline from ERPs:

condition = 'a1'  # complete for each condition ['a1', 'a2', 'e1', 'e2']
stimuli = ['target', 'distractor', 'non_targets_target', 'non_targets_distractor']  # complete for each stim type...
# stim_type = stimuli[2]
channels = ['motor', 'attention']
selected_ch = channels[0]

all_concat_epochs = {}
for sub in sub_list:
    all_concat_epochs[sub] = {}
    if sub == 'sub16' or sub == 'sub22':
        continue
    else:
        for stim_type in stimuli:
            all_concat_epochs[sub][stim_type] = {}
            baseline_erp = mne.read_evokeds(results_path / sub / 'baseline' / f'{sub}_baseline_erp-ave.fif')[0]
            target_epochs = mne.read_epochs(concat_path / sub / selected_ch / f'{sub}_{condition}_{stim_type}_{selected_ch}_concatenated-epo.fif', preload=True)
            # target_epochs.apply_baseline(-0.2, 0.0)

            baseline_erp = baseline_erp.pick_channels(motor_channels)
            baseline_data = baseline_erp._data  # Shape: (n_channels, n_times)

            target_epochs_corrected = target_epochs.copy()
            target_epochs_corrected._data -= baseline_data[np.newaxis, :, :]  # Broadcast subtraction
            all_concat_epochs[sub][stim_type] = target_epochs_corrected
            # 0(epochs, channels, times)
            # target_epochs_corrected.save(concat_path / sub / selected_ch / selected_ch/f'{sub}_{condition}_{stim_type}_corrected-epo.fif', overwrite=True)
            # Adds an extra dimension (np.newaxis) at the beginning.
            # Shape becomes: (1, n_channels, n_times).
            # allows NumPy broadcasting to match the shape
# compute ERP:

all_erps = {condition: [] for condition in stimuli}

# Loop through subjects
for sub, conditions in all_concat_epochs.items():
    for condition, epochs in conditions.items():
        erp = epochs.average()  # Compute ERP for this subject and condition
        all_erps[condition].append(erp)  # Store ERP in the correct condition

grand_erps = {condition: mne.grand_average(erps) for condition, erps in all_erps.items()}

for stim_type, erp in grand_erps.items():
    erp.plot()

# todo: also animal only epochs, and invalid distractor responses.
