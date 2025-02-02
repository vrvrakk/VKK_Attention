import os
from pathlib import Path
import numpy as np
import mne
from autoreject import AutoReject, Ransac
import copy
import json
from EEG.preprocessing_eeg import actual_mapping, \
    single_eeg_path, concat_eeg_path, stimuli
from EEG.extract_events import sub_list


###########

# 1.
def get_csv_events(condition, stim_type):
    events_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/params/isolated_events')
    for events in events_path.iterdir():
        if f'{condition}_{stim_type}_events_dict.txt' in events.name:
            with open(events, 'r') as f:
                events_dict = json.load(f)
    return events_dict


# 2a. pick channels of interest:
def pick_channels(eeg_files_list, focus=''):
    for index, eeg_files in enumerate(eeg_files_list):
        if focus == 'a':  # a for attention
            occipital_channels = ["O1", "O2", "Oz", "PO3", "PO4", "PO7", "PO8", "P5", "P6", "P7", "P8"]
            print('Further processing focused on attention. Selecting channels accordingly...')
            eeg_files.drop_channels(occipital_channels)
            eeg_files_list[index] = eeg_files
        elif focus == 'm': # m for motor
            motor_channels = ['C3', 'C4', 'CP3', 'CP4', 'Cz', 'FC3', 'FC4']
            print('Further processing on motor responses. Selecting channels accordingly...')
            eeg_files.pick_channels(motor_channels)
            eeg_files_list[index] = eeg_files
    return eeg_files_list


# 2b. IF attentional focus: subtract motor response from EEG signal:
# for this, create a mega motor-response ERP with smooth edges, and subtract.

# 3. extract target events that were pre-created:
def extract_csv_events(events_dict, sub):
    events_0_csv = events_dict[f'{sub}.csv']['0']
    events_1_csv = events_dict[f'{sub}.csv']['1']
    events_2_csv = events_dict[f'{sub}.csv']['2']
    events_3_csv = events_dict[f'{sub}.csv']['3']
    events_4_csv = events_dict[f'{sub}.csv']['4']
    events_csv_list = [events_0_csv, events_1_csv, events_2_csv, events_3_csv, events_4_csv]
    return events_csv_list


# 4. filter events:
def filter_events(motor_eeg_files_to_epochs, events_csv_list, actual_mapping):
    target_events_filtered_list = []
    event_ids_list = []
    for index, csv_events in enumerate(events_csv_list):
        target_eeg_file = motor_eeg_files_to_epochs[index]
        target_events, event_ids = mne.events_from_annotations(target_eeg_file)
        event_ids_copy = copy.deepcopy(event_ids)
        for key, values in event_ids.items():
            if key in actual_mapping.keys():
                actual_value = actual_mapping[key]
                event_ids_copy[key] = actual_value
        for eeg_events in target_events:
            if eeg_events[2] in event_ids.values():
                print(eeg_events[2])
                key = [key for key, value in event_ids.items() if eeg_events[2] == value][0]
                print(key)
                eeg_events[2] = event_ids_copy[key]
        csv_sfreq = 1000
        eeg_sfreq = target_eeg_file.info['sfreq']
        # Apply time shift correction
        # Convert sample indices -> time (seconds) -> apply time shift -> convert back to samples
        csv_timepoints_resampled = [round(event[0] * (eeg_sfreq/csv_sfreq)) for event in csv_events]
        target_events_filtered = [event for event in target_events if event[0] in csv_timepoints_resampled]
        # some eeg events won't be there, because of pre-processing
        target_events_filtered = np.array(target_events_filtered)
        target_events_filtered_list.append(target_events_filtered)
        event_ids_list.append(event_ids_copy)
    return target_events_filtered_list, event_ids_list


# 5: create epochs
def create_epochs(events_filtered_list, motor_eeg_files_to_epochs, event_ids_list, tmin=None, tmax=None):
    epochs_list = []
    for index, (events, event_ids) in enumerate(zip(events_filtered_list, event_ids_list)):
        print(event_ids)
        target_eeg_file = motor_eeg_files_to_epochs[index]
        annotations = np.unique(target_eeg_file.annotations.description)
        reject_annotation = [annotation for annotation in annotations if 'BAD' in annotation or 'Bad'
                              in annotation or 'bad' in annotation or annotation is None]
        unique_event = np.unique(events[:, 2])
        if stim_type == 'target' or stim_type == 'distractor':
            event_id = [value for key, value in event_ids.items() if value == unique_event][0]
            event_id_filt = event_id
        else:
            event_id_filt = []
            for event in events:
                if event[2] in event_ids.values():
                    event_id = event[2]
                    event_id_filt.append(event_id)
            event_id_filt = np.unique(event_id_filt).tolist()
            print(event_id_filt)
        epochs = mne.Epochs(raw=target_eeg_file, events=events, event_id=event_id_filt,
                            tmin=tmin, tmax=tmax, baseline=(-0.2, 0.0), reject_by_annotation=reject_annotation, preload=True)
        epochs.save(single_eeg_path/ sub/ 'epochs'/ f'{sub}_{condition}_{index}_epochs-epo.fif', overwrite=True)
        epochs_list.append(epochs)
    return epochs_list


# 6: apply Ransac
# 7: concatenate eeg files using mne.concatenate
# 8: AutoReject
# 9: save


########
condition = 'a1'  # a1, e1, e2
stim_type = stimuli[0]
channels = ['motor', 'attention']
selected_ch = channels[0]
# alternate step for getting epochs of other stim types:

for sub in sub_list:
    if sub == 'sub16':
        continue
    else:
        ica_eeg_files = []
        sub_eeg = single_eeg_path / sub / 'ica'
        for files in sub_eeg.iterdir():
            if condition in files.name and '.fif' in files.name:
                eeg = mne.io.read_raw_fif(files, preload=True)
                ica_eeg_files.append(eeg)
        events_dict = get_csv_events(condition, stim_type)  # choose different stim_type
        #############
        # 1:
        if selected_ch == 'motor':
            eeg_files = copy.deepcopy(ica_eeg_files)
            eeg_files_selected_chs = pick_channels(eeg_files, focus='m')
        elif selected_ch == 'attention':
            eeg_files_motor = copy.deepcopy(ica_eeg_files)
            eeg_files_selected_chs = pick_channels(eeg_files, focus='m')
        # 2:
        events_csv_list = extract_csv_events(events_dict, sub)
        # these events are already cleaned from overlaps
        # 3:
        eeg_files_to_epochs = copy.deepcopy(eeg_files_selected_chs)
        target_events_filtered_list, event_ids_list = filter_events(eeg_files_to_epochs, events_csv_list, actual_mapping)
        # 4:
        epochs_list = create_epochs(target_events_filtered_list, eeg_files_to_epochs, event_ids_list, tmin=-0.2, tmax=0.9)
        # 5:
        epochs_clean_list = []
        for index, epochs in enumerate(epochs_list):
            epochs_clean = epochs.copy()
            ransac = Ransac(n_jobs=1, n_resample=50, min_channels=0.25, min_corr=0.75, unbroken_time=0.4)
            ransac.fit(epochs_clean)
            bads = epochs.info['bads']  # Add channel names to exclude here
            if len(bads) != 0:
                for bad in bads:
                    if bad not in ransac.bad_chs_:
                        ransac.bad_chs_.extend(bads)
            print(ransac.bad_chs_)
            epochs_clean = ransac.transform(epochs_clean)
            epochs_clean.save(single_eeg_path / sub / 'epochs' / 'ransac' / f'{sub}_{condition}_{stim_type}_{index}_epochs_ransac-epo.fif',
                              overwrite=True)
            epochs_clean_list.append(epochs_clean)
        # 7:
        eeg_concat = mne.concatenate_epochs(epochs_clean_list)
        # 6:
        # Get the smallest number of epochs across all conditions

        # Ensure `n_splits` does not exceed `min_epochs`
        n_splits = min(10, len(eeg_concat))  # Max 10, but adjust if fewer epochs exist

        epochs_ar = copy.deepcopy(eeg_concat)
        ar = AutoReject(n_interpolate=[1, 4, 8, 16], n_jobs=4, cv=n_splits)
        ar = ar.fit(epochs_ar)
        epochs_ar_complete, reject_log = ar.transform(epochs_ar, return_log=True)

        concat_sub_path = concat_eeg_path / sub / 'motor'
        os.makedirs(concat_sub_path, exist_ok=True)
        eeg_concat.save(concat_sub_path / f"{sub}_{condition}_{stim_type}_{selected_ch}_concatenated-epo.fif",
                        overwrite=True)

