from pathlib import Path
import os

import matplotlib.pyplot as plt
import numpy as np
import mne
from EEG.extract_events import sub_list, default_path, exceptions
from EEG.epochs_categories import event_types, conditions, channels


results_path = default_path / 'data' / 'eeg' / 'preprocessed' / 'results'
epochs_path = results_path / 'concatenated_data' / 'epochs'
concat_path = results_path / 'concatenated_data' / 'epochs'
occipital_channels = ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8", "P5", "P6", "P7", "P8"]
motor_channels = ['C3', 'CP3', 'FC3', 'C4',  'CP4', 'Cz',  'FC4']
# time to subtract baseline from ERPs:


if __name__ == '__main__':
    condition = conditions[1]
    selected_ch = channels[1]
    event_type = event_types[1]  # 0: animal sounds, 1: targets_with_valid_responses, 5: distractors_with_valid_responses (+ 6 + 7)
    # 8: distractors_without_responses, 12: non_targets_targets_no_response, 17: non_targets_distractor_no_response
    all_concat_epochs = {}

    for sub in sub_list[:6]:
        all_concat_epochs[sub] = {}
        if sub in exceptions:
            continue
        else:

            all_concat_epochs[sub][event_type] = {}
            # baseline_erp = mne.read_evokeds(results_path / sub / 'baseline' / f'{sub}_baseline_erp-ave.fif')[0]
            sub_path = concat_path / sub / selected_ch / event_type / f'{sub}_{condition}_{event_type}_{selected_ch}_concatenated-epo.fif'
            if not sub_path.exists():
                print(f'{sub} path for {event_type} does not exist.')
                continue
            selected_epochs = mne.read_epochs(sub_path, preload=True)

            selected_epochs_corrected = selected_epochs.copy()
            # selected_epochs_corrected._data -= baseline_data[np.newaxis, :, :]  # Broadcast subtraction
            # Adds an extra dimension (np.newaxis) at the beginning.
            # Shape becomes: (n_epochs, n_channels, n_times).
            # allows NumPy broadcasting to match the shape
            all_concat_epochs[sub][event_type] = selected_epochs_corrected
            # 0(epochs, channels, times)
            # corrected_epochs_path = concat_path / sub / 'corrected' / selected_ch / event_type
            # os.makedirs(corrected_epochs_path, exist_ok=True)
            # selected_epochs_corrected.save(corrected_epochs_path / f'{sub}_{condition}_{event_type}_{selected_ch}_corrected-epo.fif', overwrite=True)

    # Initialize dictionary to store all epochs of each event type from all subjects
    all_concat_epoch_types = {event_type: []}

    # Loop through subjects and collect epochs per event type
    for sub, epochs_dict in all_concat_epochs.items():
        if epochs_dict:
            for event_type, epoch in epochs_dict.items():
                if len(epoch) > 0:  # Ensure epochs exist for this event type
                    n_epochs = len(epoch.events)
                    all_concat_epoch_types[event_type].append((n_epochs, epoch))  # Append without overwriting

    for event_type in all_concat_epoch_types.keys():
        epochs_info = all_concat_epoch_types[event_type]

        if epochs_info:  # Ensure there is data
            # Extract only the second element (epochs) from all tuples
            epochs_list = [ep_tuple[1] for ep_tuple in epochs_info]
            total_epochs = sum(ep_tuple[0] for ep_tuple in epochs_info)  # Sum total trials
            num_subjects = len(epochs_info)  # Count number of subjects

            # Concatenate epochs from all subjects
            merged_epochs = mne.concatenate_epochs(epochs_list)
            merged_epochs_path = Path(results_path / 'concatenated_data' / 'epochs' / 'all_subs')
            merged_epochs.save(merged_epochs_path/f'{condition}_{selected_ch}_{event_type}-epo.fif', overwrite=True)

            # Compute and plot ERP
            erp = merged_epochs.average()
            erp.apply_baseline((-0.2, 0.0))

            titles = f'ERP for {event_type.replace("_" , " ")} | {total_epochs} epochs | {num_subjects} subjects'
            mne.viz.plot_compare_evokeds(erp, combine='mean', title=titles)

            freqs = np.linspace(1, 30, num=150)  # 30 log-spaced frequencies
            n_cycles = freqs / 2  # Define cycles per frequency (standard approach)
            power = mne.time_frequency.tfr_multitaper(merged_epochs, freqs=freqs, n_cycles=n_cycles, average=True,
                                                      return_itc=False, decim=1, n_jobs=1)

            # === Plot Time-Frequency Heatmap ===
            fig, ax = plt.subplots(figsize=(8, 6))
            power.plot(picks=['Cz'], baseline=(-0.2, 0), mode='percent',
                       title=f'TFA Heatmap: {event_type.replace("_" , " ")} | {total_epochs} epochs | {num_subjects} subjects', axes=ax, cmap='viridis')
            plt.show()




