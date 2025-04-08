# 1. import libraries
import os
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import mne
from autoreject import AutoReject, Ransac
import copy
import json
from EEG.params import sub_list, response_mapping, actual_mapping, stimuli_dict, conditions, event_types, matching_events, channels, exceptions
from EEG.preprocessing_eeg import single_eeg_path, concat_eeg_path
from EEG.extract_events import response_mapping


# 2. define params
# Paths
events_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/params/isolated_events')

# 3. Load chosen events
def load_chosen_events(condition='', event_type='', sub=''):
    all_chosen_events = {}
    subject_path = events_path / sub
    if subject_path.exists() and subject_path.is_dir():
        all_chosen_events[sub] = {}

        for condition_folder in subject_path.iterdir():
            for file in condition_folder.iterdir():
                index = int(file.name[-6:-5])
                if event_type in file.name and condition in file.name:
                    # Ensure the nested dictionaries exist
                    all_chosen_events[sub].setdefault(event_type, {})
                    all_chosen_events[sub][event_type].setdefault(index, [])
                    with file.open("r", encoding="utf-8") as f:
                        events_dict = json.load(f)
                        all_chosen_events[sub][event_type][index].append(events_dict)
    return all_chosen_events


# 4. Select channels of interest
def pick_channels(eeg_files_list, focus=''):
    for index, eeg_files in enumerate(eeg_files_list):
        if focus == 'a':  # Attention
            occipital_channels = ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8", "P5", "P6", "P7", "P8"]
            print('Selecting attention-related channels...')
            eeg_files.drop_channels(occipital_channels)
        elif focus == 'm':  # Motor
            motor_channels = ['C3', 'C4', 'CP3', 'CP4', 'Cz', 'FC3', 'FC4']
            print('Selecting motor-related channels...')
            eeg_files.pick_channels(motor_channels)
        eeg_files_list[index] = eeg_files
    return eeg_files_list


if __name__ == '__main__':
    selected_ch = channels[1]
    condition = conditions[0]
    event_type = event_types[0]
    # 0: animal sounds, 1: targets_with_valid_responses,
    # 5: distractors_with_valid_responses (+ 6 + 7)
    # 8: distractors_without_responses, 12: non_targets_targets_no_response, 16: non_targets_distractor_no_response

    # 5. Main processing loop
    for sub in sub_list:
        if sub in exceptions:
            continue

        print(f"/nProcessing {sub} | Condition: {condition} | Event Type: {event_type} | Channel: {selected_ch}")
        ica_eeg_files = []
        sub_eeg = single_eeg_path / sub / 'ica'

        # Load EEG files
        for file in sub_eeg.iterdir():
            if condition in file.name and file.suffix == '.fif':
                eeg = mne.io.read_raw_fif(file, preload=True)
                eeg.set_eeg_reference(['FCz'])
                ica_eeg_files.append(eeg)

        if not ica_eeg_files:
            print(f"No EEG files found for {sub} in condition {condition}. Skipping...")

        # Select channels
        eeg_files = copy.deepcopy(ica_eeg_files)
        # subtract motor-only erp from eeg_files
        motor_erp_path = f'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/erp/motor_smooth_erp-ave.fif'
        motor_erp = mne.read_evokeds(motor_erp_path)[0]
        baseline_erp_path = f'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/erp/baseline_smooth_erp-ave.fif'
        baseline_erp = mne.read_evokeds(baseline_erp_path)[0]
        for eeg_file in eeg_files:
            sfreq = eeg_file.info['sfreq']
            erp_duration_motor = motor_erp.times[-1] - motor_erp.times[0]
            n_samples_erp_motor = len(motor_erp.times)
            erp_duration_baseline = baseline_erp.times[-1] - baseline_erp.times[0]
            n_samples_erp_baseline = len(baseline_erp.times)
            # Subtract the motor ERP at each event time
            events, event_ids = mne.events_from_annotations(eeg_file)
            correct_event_ids = {key: value for key, value in actual_mapping.items() if key in event_ids and key not in {'New Segment/'}}
            for event in events:
                for key, old_value in event_ids.items():
                    if event[2] == old_value and key in correct_event_ids:
                        event[2] = correct_event_ids[key]
            for event in events:
                if event[2] in response_mapping.values():
                    event_sample = event[0]  # sample number of the event
                    start_sample = event_sample - int(motor_erp.times[0] * sfreq)
                    end_sample = start_sample + n_samples_erp_motor
                    # Check if the event is within the bounds of the raw data
                    if start_sample >= 0 and end_sample <= len(eeg_file.times):
                        # Subtract the ERP data from the raw data
                        eeg_file._data[:, start_sample:end_sample] -= motor_erp.data
            # # subtract baseline ERP:
            # for event in events:
            #     if event[2] in stimuli_dict.values():
            #         event_sample = event[0]
            #         start_sample = event_sample - int(baseline_erp.times[0] * sfreq)
            #         end_sample = start_sample + n_samples_erp_baseline
            #         if start_sample >= 0 and end_sample <= len(eeg_file.times):
            #             eeg_file._data[:, start_sample:end_sample] -= baseline_erp.data
        for index, eeg_file in enumerate(eeg_files):
            save_path = Path(single_eeg_path / sub / 'subtracted')
            os.makedirs(save_path, exist_ok=True)
            eeg_file.save(save_path / f'{sub}_{condition}_{index}_ica_subtracted-raw.fif', overwrite=True)

        # eeg_files_selected_chs = pick_channels(eeg_files, focus='m' if selected_ch == 'motor' else 'a')
        # keep all channels for now
        # Load events
        chosen_events_dicts = load_chosen_events(condition, event_type=event_type, sub=sub)
        # Create epochs
        epochs_list = []
        for eeg_idx, eeg in enumerate(eeg_files):
            if sub in chosen_events_dicts and event_type in chosen_events_dicts[sub]:
                event_list = chosen_events_dicts[sub][event_type]
                if eeg_idx in event_list:
                    events = event_list[eeg_idx]
                    if events:
                        unique_stimuli = set(event[2] for event in events[0])  # Get unique stimulus numbers
                        event_id = {str(stim): stim for stim in unique_stimuli}  # Create dictionary mapping

                        epochs = mne.Epochs(
                            eeg,
                            events=events[0],
                            event_id=event_id,  # Adjust as needed
                            tmin=-0.2,  # Pre-stimulus time
                            tmax=0.9,  # Post-stimulus time
                            baseline=(-0.2, 0.0),
                            preload=True
                        )
                        epochs_list.append(epochs)
                    else:
                        print(f"Warning: No events found for {sub} | Condition: {condition} | Event Index: {eeg_idx}")
                else:
                    print(f"Skipping EEG file {eeg_idx} for {sub} as no corresponding event file was found.")
        if not epochs_list:
            print(f"No epochs created for {sub}, condition {condition}, event type {event_type}. Skipping...")

        concat_sub_path = concat_eeg_path / sub / selected_ch / event_type
        os.makedirs(concat_sub_path, exist_ok=True)
        save_path = concat_sub_path / f"{sub}_{condition}_{event_type}_{selected_ch}_concatenated-epo.fif"
        evokeds_path = Path(
            'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/concatenated_data/evokeds')
        folder_path = evokeds_path / condition / event_type
        os.makedirs(folder_path, exist_ok=True)
        # Ransac artifact detection
        epochs_clean_list = []
        for index, epochs in enumerate(epochs_list):
            epochs_clean = epochs.copy()
            if len(epochs_clean) == 0:
                print(f"Skipping RANSAC for {sub} | Condition: {condition} | No valid epochs.")
                continue  # Skip to the next subject
            ransac = Ransac(n_jobs=1, n_resample=50, min_channels=0.25, min_corr=0.85, unbroken_time=0.4)
            ransac.fit(epochs_clean)

            # Mark additional bad channels
            bads = epochs.info['bads']
            for bad in bads:
                if bad not in ransac.bad_chs_:
                    ransac.bad_chs_.append(bad)

            print(f"Bad channels detected: {ransac.bad_chs_}")
            epochs_clean = ransac.transform(epochs_clean)
            epochs_clean_list.append(epochs_clean)

        if not epochs_clean_list:
            print(f"No clean epochs for {sub} in condition {condition}. Skipping...")

        # Concatenate epochs
        if len(epochs_clean_list) > 1:
            eeg_concat = mne.concatenate_epochs(epochs_clean_list)

            # AutoReject for final artifact rejection
            min_epochs = len(eeg_concat)
            if min_epochs < 5:
                print(f"Skipping AutoReject for {sub} | Condition: {condition} | Too few epochs ({min_epochs}).")
                epochs_ar_complete = eeg_concat  # Save concatenated epochs without AutoReject
                # Save results
                epochs_ar_complete.save(save_path, overwrite=True)
                print(f"Saving epochs to {save_path}")
                epochs_erp = epochs_ar_complete.average()
                epochs_erp.save(folder_path / f'erp_{sub}_{selected_ch}_{condition}_{event_type}-ave.fif',
                                overwrite=True)
            else:
                n_splits = min(10, min_epochs)  # Ensure n_splits is valid
                epochs_ar = copy.deepcopy(eeg_concat)
                ar = AutoReject(n_interpolate=[1, 4, 8, 16], n_jobs=4, cv=n_splits)
                ar.fit(epochs_ar)
                epochs_ar_complete, reject_log = ar.transform(epochs_ar, return_log=True)
                # Save results
                epochs_ar_complete.save(save_path, overwrite=True)
                print(f"Saving epochs to {save_path}")
                epochs_erp = epochs_ar_complete.average()
                epochs_erp.save(folder_path / f'erp_{sub}_{selected_ch}_{condition}_{event_type}-ave.fif', overwrite=True)
                # epochs_erp.plot()


