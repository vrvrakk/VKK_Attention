import os
from pathlib import Path
import pandas as pd
import numpy as np
import mne
from TRF_predictors.overlap_ratios import load_eeg_files
from TRF_predictors.config import sub, condition, \
    sfreq, stim_dur, \
    default_path, results_path, events_path, predictors_path


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

                if 0.0 <= diff <= 0.2:
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
        print("Num RTs:", len(rts), "Num Labels:", len(labels), "Num Times:", len(targets))
        # rts, labels, targets all same len
        eeg_len = eeg_data.n_times
        rt_value_pred = np.zeros(eeg_len)
        rt_type_pred = np.zeros(eeg_len)

        for rt, label, time in zip(rts, labels, targets):

            if rt is None:
                continue  # missed, keep predictor at 0
            sample_start = int(time * sfreq)
            sample_end = int((time + rt) * sfreq)
            if sample_start >= eeg_len or sample_end >= eeg_len:
                continue  # skip if out of bounds

            # RT value predictor (continuous)
            print(f"rt={rt}, time={time}, sample_start={sample_start}, sample_end={sample_end} eeg_len={eeg_len}")
            rt_value_pred[sample_start:sample_end] += rt

            # RT type predictor (categorical weights)
            if label == 'valid':
                rt_type_pred[sample_start:sample_end] += 1.0
            elif label == 'delayed':
                rt_type_pred[sample_start:sample_end] += 0.5
            elif label == 'early':
                rt_type_pred[sample_start:sample_end] += 0.25
            # missed = 0 by default

        rt_value_predictors.append(rt_value_pred)
        rt_type_predictors.append(rt_type_pred)
    return rt_value_predictors, rt_type_predictors


def save_predictor_blocks(rt_per_block, rt_labels_per_block, stim_dur, stream_type=''):
    predictors_path = default_path / 'data/eeg/predictors'
    save_path = predictors_path / 'RTs' / sub / condition / stream_type
    save_path.mkdir(parents=True, exist_ok=True)
    for i, (series, series_labels) in enumerate(zip(rt_per_block, rt_labels_per_block)):
        filename_block = f'{sub}_{condition}_{stream_type}_{i}_RTs.npz'
        np.savez(save_path / filename_block,
                 RTs=series,
                 RT_labels=series_labels,
                 sfreq=sfreq,
                 stim_duration_samples=int(stim_dur * sfreq),
                 stream_label=stream_type)


def save_RT_predictors(rt_per_block, rt_labels_per_block, stream_type=''):
    predictor_concat_rt = np.concatenate(rt_per_block)
    predictor_concat_rt_labels = np.concatenate(rt_labels_per_block)
    rt_path = predictors_path / 'RTs'
    save_path = rt_path / sub / condition / stream_type
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f'{sub}_{condition}_{stream_type}_RTs_series_concat.npz'
    np.savez(
        save_path / filename,
        RTs=predictor_concat_rt,
        RT_labels=predictor_concat_rt_labels,
        sfreq=sfreq,
        stim_duration_samples=int(stim_dur * sfreq),
        stream_label=stream_type)
    return predictor_concat_rt, predictor_concat_rt_labels


if __name__ == '__main__':


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

    response_times = []  # len around 31
    for response in responses:
        block_responses = []
        for response_event in response:
            response_time = response_event[0] / sfreq
            block_responses.append(response_time)
        response_times.append(block_responses)

    target_times = []  # len around 31
    for events_array in target_stream:
        block_target_t = []
        for events in events_array:
            if events[1] == 4:
                target_onset = events[0] / sfreq
                block_target_t.append(target_onset)
        target_times.append(block_target_t)

    distractor_times = []  # len around 31
    for events_array in distractor_stream:
        block_distractor_t = []
        for events in events_array:
            if events[1] == 3:
                distractor_onset = events[0] / sfreq
                block_distractor_t.append(distractor_onset)
        distractor_times.append(block_distractor_t)

    eeg_files_list, _ = load_eeg_files(sub=sub, condition=condition, sfreq=sfreq, results_path=results_path)
    # sfreq is 125Hz

    # get RTs predictors for each eeg file
    if condition in ['a1', 'e1']:
        target_stream_name = 'nt_target'
        distractor_stream_name = 'nt_distractor'
    elif condition in ['a2', 'e2']:
        target_stream_name = 'nt_target'
        distractor_stream_name = 'nt_distractor'

    rt_per_block_target, rt_labels_per_block_target = get_RTs(target_times)
    rt_per_block_distractor, rt_labels_per_block_distractor = get_RTs(distractor_times)

    rt_value_predictors_target, rt_type_predictors_target = get_rt_predictors(rt_per_block_target, rt_labels_per_block_target, target_times)
    save_predictor_blocks(rt_value_predictors_target, rt_type_predictors_target, stim_dur, stream_type=target_stream_name)
    rt_value_predictors_distractor, rt_type_predictors_distractor = get_rt_predictors(rt_per_block_distractor, rt_labels_per_block_distractor, distractor_times)
    save_predictor_blocks(rt_value_predictors_distractor, rt_type_predictors_distractor, stim_dur, stream_type=distractor_stream_name)

    predictor_concat_rt_target, predictor_concat_rt_labels_target = \
        save_RT_predictors(rt_value_predictors_target, rt_type_predictors_target, stream_type=target_stream_name)
    predictor_concat_rt_distractor, predictor_concat_rt_labels_distractor = \
        save_RT_predictors(rt_value_predictors_distractor, rt_type_predictors_distractor, stream_type=distractor_stream_name)