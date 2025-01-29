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
with open(json_path / "electrode_names.json") as file: #electrode_names
    mapping = json.load(file)

concat_eeg_path = default_path / 'data' / 'eeg' / 'preprocessed'/ 'results'/ 'concatenated_data'/ 'epochs'
single_eeg_path = default_path / 'data' / 'eeg' / 'preprocessed'/ 'results'


# a2_eeg_header_files = EEG.extract_events.extract_eeg_files(condition='a2')
# e1_eeg_header_files = EEG.extract_events.extract_eeg_files(condition='e1')
# e2_eeg_header_files = EEG.extract_events.extract_eeg_files(condition='e2')


# 2. load eeg files from headers:
# a2_eeg = EEG.extract_events.load_eeg_files(a2_eeg_header_files)
# e1_eeg = EEG.extract_events.load_eeg_files(e1_eeg_header_files)
# e2_eeg = EEG.extract_events.load_eeg_files(e2_eeg_header_files)


# 3. concatenate all eeg files of one sub into a list:
def create_sub_eeg_list(eeg):
    eeg_files_list = {}
    for eeg_files in eeg:
        sub_name = eeg_files.filenames[0].split('\\')[-2]
        if sub_name not in eeg_files_list:
            eeg_files_list[sub_name] = [] #initialize with empty list, not dictionary
        eeg_files_list[sub_name].append(eeg_files)
    return eeg_files_list


# 4. extract csvs with events:
events_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/params/isolated_events')
for events in events_path.iterdir():
    if 'a1_target_events_dict.txt' in events.name:
        with open(events, 'r') as f:
            a1_target_events_dict = json.load(f)
    elif 'a1_distractor_events_dict.txt' in events.name:
        with open(events, 'r') as f:
            a1_distractor_events_dict = json.load(f)

# 5. investigate EEG signal manually and select bad channels and segments:
# change sub name accordingly and investigate...


# 6. apply montage and add FCz channel:
def add_montage(target_eeg_files, condition=''):
    for index, eeg in enumerate(target_eeg_files):
        eeg.resample(sfreq=500)  # downsample from 1000Hz to 500Hz
        eeg.rename_channels(mapping)
        eeg.add_reference_channels('FCz')  # add reference channel
        eeg.set_montage('standard_1020')  # apply standard montage
        eeg.drop_channels(['A1', 'A2', 'M2'])
        eeg.save(single_eeg_path/sub/'investigated'/ f'{sub}_{condition}_{index}_investigated-raw.fif', overwrite=True)
        target_eeg_files[index] = eeg
    return target_eeg_files


# 7. interpolate bad channels:
def interpolate_eeg(target_eeg_files_to_interp, condition='', sub=''):
    for index, concat_eeg in enumerate(target_eeg_files_to_interp):
        interp_eeg = concat_eeg.interpolate_bads(reset_bads=True)
        interp_eeg.save(single_eeg_path/ sub / 'interpolated' / f"{sub}_{condition}_{index}_interpolated-raw.fif", overwrite=True)
        target_eeg_files_to_interp[index] = interp_eeg
    return target_eeg_files_to_interp


# 8. 50Hz notch & bandpass filter with range of choice:
def filter_eeg(target_eeg_files_filter, freq_range=(1, 30, 1), condition='a1'):
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
        eeg_filtered.save(single_eeg_path/ sub / 'filtered' / f'{sub}_{condition}_concatenated_filtered_{freq_range[1]}-raw.fif', overwrite=True)
        target_eeg_files_filter[index] = eeg_filtered
    return target_eeg_files_filter

# 9. apply ICA:
# see below


# 10a. pick channels of interest:
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


# 10b. IF attentional focus: subtract motor response from EEG signal:
# for this, create a mega motor-response ERP with smooth edges, and subtract.

# 11. extract target events that were pre-created:
def extract_csv_events(target_events_dict, sub):
    events_0_csv = target_events_dict[f'{sub}.csv']['0']
    events_1_csv = target_events_dict[f'{sub}.csv']['1']
    events_2_csv = target_events_dict[f'{sub}.csv']['2']
    events_3_csv = target_events_dict[f'{sub}.csv']['3']
    events_4_csv = target_events_dict[f'{sub}.csv']['4']
    events_csv_list = [events_0_csv, events_1_csv, events_2_csv, events_3_csv, events_4_csv]
    return events_csv_list


# 12. filter events:
def filter_events(motor_eeg_files_to_epochs, events_csv_list):
    target_events_filtered_list = []
    event_ids_list = []
    for index, events in enumerate(events_csv_list):
        target_eeg_file = motor_eeg_files_to_epochs[index]
        target_events, event_ids = mne.events_from_annotations(target_eeg_file)
        csv_sfreq = 1000
        eeg_sfreq = target_eeg_file.info['sfreq']
        # Apply time shift correction
        # Convert sample indices -> time (seconds) -> apply time shift -> convert back to samples
        csv_timepoints_resampled = [round(event[0] * (eeg_sfreq/csv_sfreq)) for event in events]
        target_events_filtered = [event for event in target_events if event[0] in csv_timepoints_resampled]
        # some eeg events won't be there, because of pre-processing
        target_events_filtered = np.array(target_events_filtered)
        target_events_filtered_list.append(target_events_filtered)
        event_ids_list.append(event_ids)
    return target_events_filtered_list, event_ids_list


# 13: create epochs
def create_epochs(target_events_filtered_list, motor_eeg_files_to_epochs, event_ids_list):
    epochs_list = []
    for index, (events, event_ids) in enumerate(zip(target_events_filtered_list, event_ids_list)):
        target_eeg_file = motor_eeg_files_to_epochs[index]
        annotations = np.unique(target_eeg_file.annotations.description)
        reject_annotation = [annotation for annotation in annotations if 'BAD' in annotation or 'Bad'
                              in annotation or 'bad' in annotation or annotation is None]
        unique_event = np.unique(events[:, 2])
        event_id = [value for key, value in event_ids.items() if value == unique_event][0]
        epochs = mne.Epochs(raw=target_eeg_file, events=events, event_id=event_id,
                            tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0), reject_by_annotation=reject_annotation, preload=True)
        epochs.save(single_eeg_path/ sub/ 'epochs'/ f'{sub}_{condition}_{index}_epochs-epo.fif', overwrite=True)
        epochs_list.append(epochs)
    return epochs_list


# 14: apply Ransac
# 15: AutoReject
# 16. concatenate eeg files using mne.concatenate

# run script:
if __name__ == "__main__":
    # 1:
    condition = 'a1'
    a1_eeg_header_files = EEG.extract_events.extract_eeg_files(condition='a1')
    a1_eeg = EEG.extract_events.load_eeg_files(a1_eeg_header_files)
    a1_eeg_files_list = create_sub_eeg_list(a1_eeg)
    sub = 'sub20'
    a1_target_eeg_files = a1_eeg_files_list[sub]
    for eeg_file in a1_target_eeg_files:
        eeg_file.plot()
    # sub16 looks like shite
    # sub25 a lot of eye/head(?) movement
    # 2: mark bad segments and channels
    a1_target_eeg_files_marked = add_montage(a1_target_eeg_files, condition='a1')
    # 3: interpolate bad channels
    target_eeg_files_to_interp = copy.deepcopy(a1_target_eeg_files_marked)
    a1_interp_eeg_files = interpolate_eeg(target_eeg_files_to_interp, condition='a1', sub=sub)
    # 4:
    target_eeg_files_filter = copy.deepcopy(a1_interp_eeg_files)
    a1_eeg_files_filtered = filter_eeg(target_eeg_files_filter, freq_range=(1, 30, 1), condition='a1')
    # 5: ICA
    a1_ica_eeg_files = copy.deepcopy(a1_eeg_files_filtered)
    ##################
    index = 4  # repeat ICA application for all indices from 0-4.
    # a. fit ICA:
    eeg_file = a1_ica_eeg_files[index]  # change variable according to condition
    eeg_ica = eeg_file.copy()
    eeg_ica.info['bads'].append('FCz')  # Add FCz to the list of bad channels
    ica = mne.preprocessing.ICA(n_components=0.999, method='fastica', random_state=99)
    ica.fit(eeg_ica)  # bad segments that were marked in the EEG signal will be excluded.
    # b. investigate...:
    ica.plot_components()
    # ica.plot_sources(eeg_ica)
    # c. apply ICA to remove selected components: blinks, eye movements etc.
    ica.apply(eeg_ica)
    # d. re-reference with average:
    eeg_ica.info['bads'].remove('FCz')
    eeg_ica.set_eeg_reference(ref_channels='average')
    # e. save
    a1_ica_eeg_files[index] = eeg_ica
    eeg_ica.save(single_eeg_path / sub / 'ica' / f'{sub}_{condition}_{index}_ica-raw.fif', overwrite=True)

    # once done:
    del index
    # 8:
    eeg_files_motor = copy.deepcopy(a1_ica_eeg_files)
    a1_eeg_files_selected_chs_motor = pick_channels(eeg_files_motor, focus='m')
    # 9:
    events_csv_list = extract_csv_events(a1_target_events_dict, sub)
    # these events are already cleaned from overlaps
    # 10:
    motor_eeg_files_to_epochs = copy.deepcopy(a1_eeg_files_selected_chs_motor)
    target_events_filtered_list, event_ids_list = filter_events(motor_eeg_files_to_epochs, events_csv_list)
    # 11:
    epochs_list = create_epochs(target_events_filtered_list, motor_eeg_files_to_epochs, event_ids_list)
    # 12:
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
        epochs_clean.save(single_eeg_path / sub /'epochs' / 'ransac'/ f'{sub}_{condition}_{index}_epochs_ransac-epo.fif', overwrite=True)
        epochs_clean_list.append(epochs_clean)
    # 13:
    epochs_ar_list = []
    for index, epochs_clean in enumerate(epochs_clean_list):
        epochs_ar = copy.deepcopy(epochs_clean)
        ar = AutoReject(n_interpolate=[1, 4, 8, 16], n_jobs=4)
        ar = ar.fit(epochs_ar)
        epochs_ar_complete, reject_log = ar.transform(epochs_ar, return_log=True)
        epochs_ar.save(single_eeg_path / sub /'epochs'/ 'autoreject' / f"{sub}_conditions_{condition}_autoreject-epo.fif",
                       overwrite=True)
        epochs_ar_list.append(epochs_ar)
    # 14:
    stimuli = ['target', 'distractor', 'non_targets_target', 'non_targets_distractor']
    a1_eeg_concat = mne.concatenate_epochs(epochs_ar_list)
    concat_sub_path = concat_eeg_path / sub
    os.makedirs(concat_sub_path, exist_ok=True)
    a1_eeg_concat.save(concat_sub_path / f"{sub}_{condition}_{stimuli[0]}_concatenated-epo.fif", overwrite=True)











# 16: get ERPs
# for epochs_ar in epochs_ar_list:
#     erp = epochs_ar.average()
#     erp.plot()

# todo: save all pre-processed eeg files
# todo: concatenate them all together of one sub, one condition
# todo: concatenate all of one condition, across subs





# a1_concatenated_eeg_files = concatenate_eeg_files(a1_concatenated_eeg_files_list, condition='a1')
# a2_concatenated_eeg_files = concatenate_eeg_files(a2_concatenated_eeg_files_list)
# e1_concatenated_eeg_files = concatenate_eeg_files(e1_concatenated_eeg_files_list)
# e2_concatenated_eeg_files = concatenate_eeg_files(e2_concatenated_eeg_files_list)
