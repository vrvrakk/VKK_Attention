from pathlib import Path
import os
import numpy as np
import mne
from EEG.extract_events import sub_list, default_path


results_path = default_path / 'data' / 'eeg' / 'preprocessed' / 'results'
concat_path = results_path / 'concatenated_data' / 'epochs'

# time to subtract baseline from ERPs:

condition = 'a1'  # complete for each condition ['a1', 'a2', 'e1', 'e2']
stimuli = ['target', 'distractor', 'non_targets_target', 'non_targets_distractor']  # complete for each stim type...
stim_type = stimuli[0]
for sub in sub_list:
    if sub == 'sub16':
        continue
    else:
        baseline_erp = mne.read_evokeds(results_path / sub / 'baseline' / f'{sub}_baseline_erp-ave.fif')[0]
        a1_target_epochs = mne.read_epochs(concat_path / sub / f'{sub}_{condition}_{stim_type}_concatenated-epo.fif', preload=True)

        motor_channels = ['C3', 'C4', 'CP3', 'CP4', 'Cz', 'FC3', 'FC4']
        baseline_erp = baseline_erp.pick_channels(motor_channels)
        baseline_data = baseline_erp.data  # Shape: (n_channels, n_times)

        a1_target_epochs_corrected = a1_target_epochs.copy()
        a1_target_epochs_corrected._data -= baseline_data[np.newaxis, :, :]  # Broadcast subtraction
        a1_target_epochs_corrected.save(concat_path / sub / f'{sub}_{condition}_{stim_type}_corrected-epo.fif', overwrite=True)
        # Adds an extra dimension (np.newaxis) at the beginning.
        # Shape becomes: (1, n_channels, n_times).
        # allows NumPy broadcasting to match the shape

        # Compute new ERP (average of corrected epochs)
        erp_corrected = a1_target_epochs_corrected.average()

        # Plot the new ERP
        erp_corrected.plot()
