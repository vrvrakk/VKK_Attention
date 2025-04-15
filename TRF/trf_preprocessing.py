from pathlib import Path
import os
import mne
import mtrf
import numpy as np
import pandas as pd

from EEG.params import actual_mapping

# Step 1:
# load EEG files of selected sub
# load events for each file
# align and downsample both
# extract onsets

# get eeg files:
default_path = Path.cwd()
results_path = default_path / 'data/eeg/preprocessed/results'
sfreq = 125
sub = 'sub01'
condition = 'a1'
# load files:
def load_eeg_files(sub='', condition=''):
    eeg_path = results_path / f'{sub}/ica'
    eeg_files_list = []
    eeg_events_list = []
    for sub_files in eeg_path.iterdir():
        if '.fif' in sub_files.name:
            if condition in sub_files.name:
                eeg_file = mne.io.read_raw_fif(sub_files, preload=True)
                eeg_file.set_eeg_reference('average')
                eeg_file.resample(sfreq=sfreq)
                # get events from each eeg file
                eeg_events, eeg_event_ids = mne.events_from_annotations(eeg_file)
                # save each eeg file and eeg event list
                eeg_events_list.append((eeg_events, eeg_event_ids))
                eeg_files_list.append(eeg_file)
    return eeg_files_list, eeg_events_list

eeg_files_list, eeg_events_list = load_eeg_files(sub=sub, condition=condition)
# correct event IDs and values:
def update_eeg_events(eeg_events_list):
    for i, (events, event_ids) in enumerate(eeg_events_list):
        for event in events:
            value = event[2]
            key_of_value = [key for key, val in event_ids.items() if val == value][0]
            if key_of_value:
                if key_of_value == 'Stimulus/S 64':
                    event[2] = 64
                elif key_of_value in actual_mapping:
                    event[2] = actual_mapping[key_of_value]
            else:
                print(f"Value {value} not found in event_ids")

        # Replace event_ids values using actual_mapping
        updated_event_ids = {key: actual_mapping[key] for key in event_ids if key in actual_mapping}

        # Save back to eeg_events_list
        eeg_events_list[i] = (events, updated_event_ids)
    return eeg_events_list


updated_eeg_events_list = update_eeg_events(eeg_events_list)

# assign binary predictors:
stream1_nums = {1, 2, 3, 4, 5, 6, 7, 8, 9}
stream2_nums = {65: 1, 66: 2, 67: 3, 68: 4, 69: 5, 70: 6, 71: 7, 72: 8, 73: 9}
response_nums = {129: 1, 130: 2, 131: 3, 132: 4, 133: 5, 134: 6, 136: 8, 137: 9}

# deepcopy the lists of events
eeg_events_list_copy = [(events.copy(), event_ids.copy()) for events, event_ids in updated_eeg_events_list]
def segregate_stream_events(eeg_events_list_copy):
    stream1_events_list = []
    stream2_events_list = []
    response_events_list = []
    # continuous for target stream and distractor respectively
    for i, (events, event_ids) in enumerate(eeg_events_list_copy):
        stream1_events = [event for event in events if event[2] in stream1_nums]
        stream2_events = [event for event in events if event[2] in stream2_nums]
        response_events = [event for event in events if event[2] in response_nums.keys()]
        for events in response_events:
            events[1] = 1
        if condition in ['a1', 'e1']:
            deviant = 71
            target_num = response_nums[response_events[0][2]]
            distractor_num = [key for key, val in stream2_nums.items() if val == target_num][0]
            for event in stream1_events:
                if event[2] == target_num:
                    event[1] = 3  # task-relevant stim
                else:
                    event[1] = 1  # non-targets
            for event in stream2_events:
                if event[2] == distractor_num:
                    event[1] = 2  # task-relevant but ignored
                elif event[2] == deviant:
                    event[1] = 1  # deviant
                else:
                    event[1] = 0  # normal distractor
        elif condition in ['a2', 'e2']:
            deviant = 7
            target_num = [key for key, val in stream2_nums.items() if val == response_events[0][2]][0]
            distractor_num = response_nums[response_events[0][2]]
            for event in stream2_events:
                if event[2] == target_num:
                    event[1] = 3
                else:
                    event[1] = 1
            for event in stream1_events:
                if event[2] == distractor_num:
                    event[1] = 2
                elif event[2] == deviant:
                    event[1] = 1
                else:
                    event[1] = 0
        # Append after updating all events
        stream1_events_list.append(stream1_events)
        stream2_events_list.append(stream2_events)
        response_events_list.append(response_events)
    return stream1_events_list, stream2_events_list, response_events_list

stream1_events_list, stream2_events_list, response_events_list = segregate_stream_events(eeg_events_list_copy)

def save_stream_events(stream_events_list, sub='', condition='', stream=''):
    save_path = default_path / f'data/eeg/predictors/streams_events/{sub}/{condition}'
    os.makedirs(save_path, exist_ok=True)
    for i, events_array in enumerate(stream_events_list):
        filename = save_path / f'{stream}_{i}_events_array.npz'
        np.save(filename, events_array)  # Save for future use.

save_stream_events(stream1_events_list, sub=sub, condition=condition, stream='stream1')
save_stream_events(stream2_events_list, sub=sub, condition=condition, stream='stream2')
save_stream_events(response_events_list, sub=sub, condition=condition, stream='response')

def create_continuous_onsets_predictor(events_array, total_samples, sfreq=125, stim_duration_sec=0.745):
    """
    Turns an array of [sample_index, weight, spoken_number] into a continuous predictor.

    Args:
        events_array: list of np.arrays of shape (3,) → [time sample, weight, number]
        total_samples: length of the EEG data in samples (get from actual data)
        sfreq: EEG sampling rate (Hz) (125Hz)
        stim_duration_sec: duration of each stimulus in seconds (default 745 ms) -> hope it's fine ;(

    Returns:
        A 1D NumPy array of length `total_samples` with weighted regions.
    """
    predictor = np.zeros(total_samples)
    stim_dur_samples = int(stim_duration_sec * sfreq)

    for event in events_array:
        onset = int(event[0])
        weight = event[1]
        if onset >= total_samples:
            continue  # skip this event entirely
        offset = min(onset + stim_dur_samples, total_samples)
        predictor[onset:offset] = weight  # Hold weight for full duration

    return predictor


stream1_predictors_all = []
stream2_predictors_all = []
response_predictors_all = []
for i, eeg_file in enumerate(eeg_files_list):
    N = eeg_file.n_times  # Or sum all n_times across blocks

    # Build predictors for one subject
    predictor_stream1 = create_continuous_onsets_predictor(stream1_events_list[0], total_samples=N)
    predictor_stream2 = create_continuous_onsets_predictor(stream2_events_list[0], total_samples=N)
    predictor_responses = create_continuous_onsets_predictor(response_events_list[0], total_samples=N)
    stream1_predictors_all.append(predictor_stream1)
    stream2_predictors_all.append(predictor_stream2)
    response_predictors_all.append(predictor_responses)
stream1_weights_concat = np.concatenate(stream1_predictors_all)
stream2_weights_concat = np.concatenate(stream2_predictors_all)
response_weights_concat = np.concatenate(response_predictors_all)

# now concatenate the EEG data of selected sub and condition:
eeg_concatenated = mne.concatenate_raws(eeg_files_list)


def save_onset_predictors(sub='', condition=''):
    stim_dur = 0.745
    binary_weights_path = default_path / 'data/eeg/predictors/binary_weights'
    save_path = binary_weights_path / sub
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f'{sub}_{condition}_weights_series.npz'
    np.savez(
        save_path / f'{sub}_{condition}_predictors.npz',
        stream1=stream1_weights_concat,
        stream2=stream2_weights_concat,
        responses=response_weights_concat,
        sfreq=sfreq,
        stim_duration_samples=int(stim_dur * sfreq),
        stream1_label='target_stream',
        stream2_label='distractor_stream',
        response_label='responses_stream'
    )


save_onset_predictors(sub=sub, condition=condition)