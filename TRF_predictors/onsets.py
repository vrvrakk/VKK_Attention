from copy import deepcopy
from pathlib import Path
import os
import mne
import numpy as np
import pandas as pd
from TRF_predictors.config import sub, condition, \
    sfreq, stim_dur, \
    default_path, results_path, predictors_path, \
    stream1_label, stream2_label, \
    stream1_nums, stream2_nums, response_nums

from EEG.params import actual_mapping


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


# correct event IDs and values:
def update_eeg_events(eeg_events_list):
    for i, (events, event_ids) in enumerate(eeg_events_list):
        for event in events:
            value = event[2]
            key_of_value = [key for key, val in event_ids.items() if val == value][0]
            print(event[0], key_of_value)
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
                    event[1] = 4  # task-relevant stim
                elif event[2] == 7:
                    event[1] = 0 # false stim
                else:
                    event[1] = 2  # non-targets
            for event in stream2_events:
                if event[2] == distractor_num:
                    event[1] = 3  # task-relevant but ignored
                elif event[2] == deviant:
                    event[1] = 2  # deviant
                else:
                    event[1] = 1  # normal distractor
        elif condition in ['a2', 'e2']:
            deviant = 7
            distractor_num = response_nums[response_events[0][2]]
            target_num = [key for key, val in stream2_nums.items() if val == distractor_num][0]
            print(distractor_num, target_num)
            for event in stream2_events:
                print(event)
                if event[2] == target_num:
                    event[1] = 4
                elif event[2] == 71:
                    event[1] = 0 # false deviant
                else:
                    event[1] = 2
            for event in stream1_events:
                if event[2] == distractor_num:
                    event[1] = 3
                elif event[2] == deviant:
                    event[1] = 2
                else:
                    event[1] = 1
        # Append after updating all events
        stream1_events_list.append(stream1_events)
        stream2_events_list.append(stream2_events)
        response_events_list.append(response_events)
    return stream1_events_list, stream2_events_list, response_events_list


def save_stream_events(stream_events_list, sub='', condition='', stream=''):
    save_path = default_path / f'data/eeg/predictors/streams_events/{sub}/{condition}'
    os.makedirs(save_path, exist_ok=True)
    for i, events_array in enumerate(stream_events_list):
        filename = save_path / f'{stream}_{i}_events_array.npz'
        np.save(filename, events_array)  # Save for future use.
        print(f'{stream}_{i}_events_array.npz saved to {filename}')

# all ok until here

def save_predictor_blocks(predictors, stim_dur, stream_type):
    save_path = predictors_path / 'binary_weights' / sub / condition / stream_type
    save_path.mkdir(parents=True, exist_ok=True)
    for i, series in enumerate(predictors):
        filename_block = f'{sub}_{condition}_{stream_type}_{i}_weights_series.npz'
        np.savez(save_path / filename_block,
                 onsets=series,
                 sfreq=sfreq,
                 stim_duration_samples=int(stim_dur * sfreq),
                 stream_label=stream_type)
        print(f"Saved: {filename_block} in {save_path}")


def create_continuous_onsets_predictor(events_array, total_samples, sfreq=sfreq, stim_duration_sec=0.745):
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


def filter_continuous_predictor(eeg_files_list, streams_list, sfreq=sfreq, stim_duration_sec=0.745, stream_type='', event_num=None):
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
    stim_dur_samples = int(stim_duration_sec * sfreq)
    filtered_predictors = []
    binary_weights_path = predictors_path / 'binary_weights'
    save_path = binary_weights_path / sub / condition / stream_type
    save_path.mkdir(parents=True, exist_ok=True)
    for i, eeg_file in enumerate(eeg_files_list):
        total_samples = eeg_file.n_times
        filename = f'{sub}_{condition}_{i}_{stream_type}_weights_series.npz'
        filtered_predictor = np.zeros(total_samples)
        events_array = streams_list[i]
        for event in events_array:
            weight = event[1]
            event_onset = event[0]
            if weight == event_num:
                if event_onset >= total_samples:
                    continue  # skip this event entirely
                event_offset = min(event_onset + stim_dur_samples, total_samples)
                filtered_predictor[event_onset:event_offset] = weight
        np.savez(save_path/filename,
                 onsets=filtered_predictor,
                 sfreq=sfreq,
                 stim_duration_samples=int(stim_dur * sfreq),
                 stream_label=stream_type,
                 )
        print(f"Saved: {filename} in {save_path}")
        filtered_predictors.append(filtered_predictor)
    return filtered_predictors



def save_concat_predictors(series_concat, sub='', condition='', stream_type=''):
    stim_dur = 0.745
    binary_weights_path = predictors_path / 'binary_weights'
    save_path = binary_weights_path / sub / condition / stream_type
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f'{sub}_{condition}_{stream_type}_weights_series_concat.npz'
    np.savez(
        save_path / filename,
        onsets=series_concat,
        sfreq=sfreq,
        stim_duration_samples=int(stim_dur * sfreq),
        stream_label=stream_type)
    print(f"Saved: {filename} in {save_path}")

if __name__ == '__main__':
    # Step 1:
    # load EEG files of selected sub
    # load events for each file
    # align and downsample both
    # extract onsets

    # get eeg files:
    eeg_files_list, eeg_events_list = load_eeg_files(sub=sub, condition=condition)

    from copy import deepcopy
    updated_eeg_events_list = update_eeg_events(deepcopy(eeg_events_list)) # seems correct - assigning target / distr events

    # deepcopy the lists of events
    eeg_events_list_copy = [(events.copy(), event_ids.copy()) for events, event_ids in updated_eeg_events_list]

    stream1_events_list, stream2_events_list, response_events_list = segregate_stream_events(eeg_events_list_copy)
    print(len(stream1_events_list[0]), len(stream2_events_list[0])) # target is 147

    save_stream_events(stream1_events_list, sub=sub, condition=condition, stream='stream1')
    save_stream_events(stream2_events_list, sub=sub, condition=condition, stream='stream2')
    save_stream_events(response_events_list, sub=sub, condition=condition, stream='response')

    # Assign role based on condition
    if condition in ['a1', 'e1']:
        target_stream_events = stream1_events_list
        distractor_stream_events = stream2_events_list
    else:  # 'a2', 'e2'
        target_stream_events = stream2_events_list
        distractor_stream_events = stream1_events_list

    # Save target predictors
    targets_onsets = filter_continuous_predictor(eeg_files_list, target_stream_events, sfreq=sfreq, stim_duration_sec=0.745, stream_type='targets', event_num=4)
    targets_onsets_concat = np.concatenate(targets_onsets)
    save_concat_predictors(targets_onsets_concat, sub=sub, condition=condition, stream_type='targets')

    nt_target_onsets = filter_continuous_predictor(eeg_files_list, target_stream_events, sfreq=sfreq, stim_duration_sec=0.745, stream_type='nt_target', event_num=2)

    nt_targets_onsets_concat = np.concatenate(nt_target_onsets)

    save_concat_predictors(nt_targets_onsets_concat, sub=sub, condition=condition, stream_type='nt_target')

    # Save distractor predictors
    distractor_onsets = filter_continuous_predictor(eeg_files_list, distractor_stream_events, sfreq=sfreq, stim_duration_sec=0.745, stream_type='distractors', event_num=3)
    distractor_onsets_concat = np.concatenate(distractor_onsets)
    save_concat_predictors(distractor_onsets_concat, sub=sub, condition=condition, stream_type='distractors')

    nt_distractor_onsets = filter_continuous_predictor(eeg_files_list, distractor_stream_events, sfreq=sfreq, stim_duration_sec=0.745, stream_type='nt_distractor', event_num=1)
    nt_distractor_onsets_concat = np.concatenate(nt_distractor_onsets)
    save_concat_predictors(nt_distractor_onsets_concat, sub=sub, condition=condition, stream_type='nt_distractor')

    deviants_onsets = filter_continuous_predictor(eeg_files_list, distractor_stream_events, sfreq=sfreq, stim_duration_sec=0.745, stream_type='deviants', event_num=2)
    deviants_onsets_concat = np.concatenate(deviants_onsets)
    save_concat_predictors(deviants_onsets_concat, sub=sub, condition=condition, stream_type='deviants')

    print(targets_onsets_concat.shape, nt_targets_onsets_concat.shape, distractor_onsets_concat.shape, nt_distractor_onsets_concat.shape, deviants_onsets_concat.shape)

    if condition in ['a1', 'e1']:
        target_stream = 'stream1'
        distractor_stream = 'stream2'
    elif condition in ['a2', 'e2']:
        target_stream = 'stream2'
        distractor_stream = 'stream1'

    # Optional: save neutral stream1/stream2 predictors (if needed later)
    target_predictors_all = []
    distractor_predictors_all = []
    response_predictors_all = []

    for i, eeg_file in enumerate(eeg_files_list):
        # create predictor with all events for each eeg file
        N = eeg_file.n_times
        target_predictor_stream = create_continuous_onsets_predictor(target_stream_events[i], total_samples=N)
        distractor_predictor_stream = create_continuous_onsets_predictor(distractor_stream_events[i], total_samples=N)
        predictor_responses = create_continuous_onsets_predictor(response_events_list[i], total_samples=N)
        print(target_predictor_stream.shape, distractor_predictor_stream.shape, predictor_responses.shape, eeg_file.n_times)

        target_predictors_all.append(target_predictor_stream)
        distractor_predictors_all.append(distractor_predictor_stream)
        response_predictors_all.append(predictor_responses)

    save_predictor_blocks(target_predictors_all, stim_dur, stream_type=target_stream)
    save_predictor_blocks(distractor_predictors_all, stim_dur, stream_type=distractor_stream)
    save_predictor_blocks(response_predictors_all, stim_dur, stream_type='responses')

    target_weights_concat = np.concatenate(target_predictors_all)
    distractor_weights_concat = np.concatenate(distractor_predictors_all)
    response_weights_concat = np.concatenate(response_predictors_all)

    save_concat_predictors(target_weights_concat, sub=sub, condition=condition, stream_type=target_stream)
    save_concat_predictors(distractor_weights_concat, sub=sub, condition=condition, stream_type=distractor_stream)
    save_concat_predictors(response_weights_concat, sub=sub, condition=condition, stream_type='responses')

    # now concatenate the EEG data of selected sub and condition:
    eeg_concatenated = mne.concatenate_raws(eeg_files_list)
    # save_onset_predictors(sub=sub, condition=condition, stream1_label=stream1_label, stream2_label=stream2_label)

    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt
    plt.ion()
    from scipy.signal import welch
    from scipy.fft import rfft, rfftfreq


    def compute_fft_and_peak(signal, sfreq, label=''):
        # Remove mean to center the signal
        signal = signal - np.mean(signal)

        # Compute FFT
        N = len(signal)
        freqs = rfftfreq(N, 1 / sfreq)
        fft_vals = np.abs(rfft(signal))

        # Normalize power (z-score optional)
        fft_z = (fft_vals - np.mean(fft_vals)) / np.std(fft_vals)

        # Find peak freq
        peak_idx = np.argmax(fft_z)
        peak_freq = freqs[peak_idx]
        peak_power = fft_z[peak_idx]

        # Plot
        plt.figure(figsize=(8, 4))
        plt.plot(freqs, fft_z, label=f'{label} (peak: {peak_freq:.2f} Hz)')
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Z-scored Power')
        plt.title(f'Envelope Spectrum: {label}')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        return peak_freq, peak_power


    sfreq = 125  # adjust if yours is different

    # Assuming you already have:
    # stream1_envelopes_concat = np.array([...])
    # stream2_envelopes_concat = np.array([...])

    peak_freq_1, power_1 = compute_fft_and_peak(target_weights_concat, sfreq, label='Stream 1 Envelope')
    peak_freq_2, power_2 = compute_fft_and_peak(distractor_weights_concat, sfreq, label='Stream 2 Envelope')

    print(f"Stream 1 peak: {peak_freq_1:.3f} Hz | Z-power: {power_1:.2f}")
    print(f"Stream 2 peak: {peak_freq_2:.3f} Hz | Z-power: {power_2:.2f}")
