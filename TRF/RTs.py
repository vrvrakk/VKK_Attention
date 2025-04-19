import os
from pathlib import Path
import pandas as pd
import numpy as np
import mne
from TRF.overlap_ratios import load_eeg_files


def get_RTs(times):
    rt_per_block = []
    rt_labels_per_block = []
    for block, block_responses in zip(times, response_times):
        block_rts = []
        block_labels = []
        used = set()

        for time in block:
            rt = None
            rt_type = 'missed'

            for i, response_time in enumerate(block_responses):
                if i in used:
                    continue
                diff = response_time - time

                if 0.0 <= diff < 0.2:
                    rt = diff
                    rt_type = 'early'
                    used.add(i)
                    break
                elif 0.2 <= diff <= 0.9:
                    rt = diff
                    rt_type = 'valid'
                    used.add(i)
                    break
                elif 0.9 < diff <= 1.2:
                    rt = diff
                    rt_type = 'delayed'
                    used.add(i)
                    break

            block_rts.append(rt)  # None if missed
            block_labels.append(rt_type)

        rt_per_block.append(block_rts)
        rt_labels_per_block.append(block_labels)
    return rt_per_block, rt_labels_per_block


def get_rt_predictors(rt_per_block, rt_labels_per_block, times):
    rt_value_predictors = []
    rt_type_predictors = []
    for eeg_data, rts, labels, targets in zip(eeg_files_list, rt_per_block, rt_labels_per_block, times):
        eeg_len = eeg_data.n_times
        rt_value_pred = np.zeros(eeg_len)
        rt_type_pred = np.zeros(eeg_len)

        for rt, label, time in zip(rts, labels, targets):
            if rt is None:
                continue  # missed, keep predictor at 0
            sample_idx = int((time + rt) * sfreq)
            if sample_idx >= eeg_len:
                continue  # skip if out of bounds

            # RT value predictor (continuous)
            rt_value_pred[sample_idx] = rt

            # RT type predictor (categorical weights)
            if label == 'valid':
                rt_type_pred[sample_idx] = 1.0
            elif label == 'delayed':
                rt_type_pred[sample_idx] = 0.5
            elif label == 'early':
                rt_type_pred[sample_idx] = 0.25
            # missed = 0 by default

        rt_value_predictors.append(rt_value_pred)
        rt_type_predictors.append(rt_type_pred)
    return rt_value_predictors, rt_type_predictors


def save_RT_predictors(rt_per_block, rt_labels_per_block, stream_type=''):
    predictor_concat_rt = np.concatenate(rt_per_block)
    predictor_concat_rt_labels = np.concatenate(rt_labels_per_block)
    rt_path = default_path / f'data/eeg/predictors/RTs'
    save_path = rt_path / sub
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f'{sub}_{condition}_{stream_type}_RT_series.npz'
    np.savez(
        save_path / filename,
        stream_pre=predictor_concat_rt,
        stream_post=predictor_concat_rt_labels,
        sfreq=sfreq,
        stim_duration_samples=int(stim_dur * sfreq),
        stream_label=stream_type)
    return predictor_concat_rt, predictor_concat_rt_labels


if __name__ == '__main__':
    sub = 'sub10'
    condition = 'e1'
    default_path = Path.cwd()
    # load eeg files:
    results_path = default_path / 'data/eeg/preprocessed/results'
    sfreq = 125
    stim_dur = 0.745
    stim_dur_s = stim_dur  # in seconds
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

    # define target and stream events
    if condition in ['a1', 'e1']:
        target_stream = stream1
        distractor_stream = stream2
    elif condition in ['a2', 'e2']:
        target_stream = stream2
        distractor_stream = stream1

    responses = []
    for event_arrays in sub_path.iterdir():
        if 'response' in event_arrays.name:
            events = np.load(event_arrays)
            responses.append(events)

    response_times = []
    for response in responses:
        block_responses = []
        for response_event in response:
            response_time = response_event[0] / sfreq
            block_responses.append(response_time)
        response_times.append(block_responses)

    target_times = []
    for events_array in target_stream:
        block_target_t = []
        for events in events_array:
            if events[1] == 3:
                target_onset = events[0] / sfreq
                block_target_t.append(target_onset)
        target_times.append(block_target_t)

    distractor_times = []
    for events_array in distractor_stream:
        block_distractor_t = []
        for events in events_array:
            if events[1] == 2:
                distractor_onset = events[0] / sfreq
                block_distractor_t.append(distractor_onset)
        distractor_times.append(block_distractor_t)

    eeg_files_list = load_eeg_files(sub=sub, condition=condition, sfreq=sfreq, results_path=results_path)

    # get RTs predictors for each eeg file

    rt_per_block_target, rt_labels_per_block_target = get_RTs(target_times)
    rt_per_block_distractor, rt_labels_per_block_distractor = get_RTs(distractor_times)

    rt_value_predictors_target, rt_type_predictors_target = get_rt_predictors(rt_per_block_target, rt_labels_per_block_target, target_times)
    rt_value_predictors_distractor, rt_type_predictors_distractor = get_rt_predictors(rt_per_block_distractor, rt_labels_per_block_distractor, distractor_times)

    predictor_concat_rt_target, predictor_concat_rt_labels_target = save_RT_predictors(rt_value_predictors_target, rt_type_predictors_target, stream_type='target')
    predictor_concat_rt_distractor, predictor_concat_rt_labels_distractor = save_RT_predictors(rt_value_predictors_distractor, rt_type_predictors_distractor, stream_type='distractor')