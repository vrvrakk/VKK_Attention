import os
from pathlib import Path
import pandas as pd
import numpy as np
import mne
from TRF_predictors.overlap_ratios import load_eeg_files
from TRF_predictors.config import sub, condition, \
    sfreq, stim_dur, \
    results_path, events_path, predictors_path


def get_ISIs(eeg_lens, stream):
    proximity_predictors_pre = []
    proximity_predictors_post = []

    for i, (eeg_len, event_array) in enumerate(zip(eeg_lens, stream)):
        block_ISIs = []
        pre_predictor = np.zeros(eeg_len)
        post_predictor = np.zeros(eeg_len)
        # First collect all gaps
        for idx, event in enumerate(event_array):
            onset = event[0] / sfreq
            offset = onset + stim_dur
            if idx > 0:
                prev_offset = (event_array[idx - 1][0] / sfreq) + stim_dur
                block_ISIs.append(onset - prev_offset)
            if idx < len(event_array) - 1:
                next_onset = event_array[idx + 1][0] / sfreq
                block_ISIs.append(next_onset - offset)

        # Now calculate ISI bounds
        isi_min = min(block_ISIs)
        isi_max = max(block_ISIs)

        for idx, event in enumerate(event_array):
            onset = event[0] / sfreq
            offset = onset + stim_dur
            start_idx = int(onset * sfreq)
            end_idx = min(int(offset * sfreq), eeg_len)

            # Default
            pre_score, post_score = 0, 0

            if idx > 0:
                prev_offset = (event_array[idx - 1][0] / sfreq) + stim_dur
                prev_gap = onset - prev_offset
                pre_score = 1 - ((prev_gap - isi_min) / (isi_max - isi_min))
                pre_score = np.clip(pre_score, 0, 1)

            if idx < len(event_array) - 1:
                next_onset = event_array[idx + 1][0] / sfreq
                next_gap = next_onset - offset
                post_score = 1 - ((next_gap - isi_min) / (isi_max - isi_min))
                post_score = np.clip(post_score, 0, 1)

            pre_predictor[start_idx:end_idx] = pre_score
            post_predictor[start_idx:end_idx] = post_score

        proximity_predictors_pre.append(pre_predictor)
        proximity_predictors_post.append(post_predictor)
    return proximity_predictors_pre, proximity_predictors_post



# filter proximity series for specific stim type:
def get_proximity_for_stim_type(stream, eeg_lens, stim_type='target', sfreq=125, stim_dur_s=0.745):
    stim_code_map = {
        'target': 3,
        'nt_target': 1,
        'distractor': 2,
        'nt_distractor': 0,
        'deviant': 1,
    }
    stim_code = stim_code_map[stim_type]

    pre_prox_predictors = []
    post_prox_predictors = []

    for block_idx, (event_array, eeg_len) in enumerate(zip(stream, eeg_lens)):
        pre_predictor = np.zeros(eeg_len)
        post_predictor = np.zeros(eeg_len)

        # Extract only relevant events for the selected stim type
        selected_events = [event for event in event_array if event[1] == stim_code]

        if len(selected_events) < 2:
            # Not enough events to compute gaps
            pre_prox_predictors.append(pre_predictor)
            post_prox_predictors.append(post_predictor)
            continue

        # Compute all inter-event gaps
        block_ISIs = []
        for idx, event in enumerate(selected_events):
            onset = event[0] / sfreq
            offset = onset + stim_dur_s
            if idx > 0:
                prev_offset = (selected_events[idx - 1][0] / sfreq) + stim_dur_s
                block_ISIs.append(onset - prev_offset)
            if idx < len(selected_events) - 1:
                next_onset = selected_events[idx + 1][0] / sfreq
                block_ISIs.append(next_onset - offset)

        isi_min = min(block_ISIs)
        isi_max = max(block_ISIs)

        for idx, event in enumerate(selected_events):
            onset = event[0] / sfreq
            offset = onset + stim_dur_s
            start_idx = int(onset * sfreq)
            end_idx = min(int(offset * sfreq), eeg_len)

            pre_score = 0
            post_score = 0

            if idx > 0:
                prev_offset = (selected_events[idx - 1][0] / sfreq) + stim_dur_s
                prev_gap = onset - prev_offset
                pre_score = 1 - ((prev_gap - isi_min) / (isi_max - isi_min + 1e-6))
                pre_score = np.clip(pre_score, 0, 1)

            if idx < len(selected_events) - 1:
                next_onset = selected_events[idx + 1][0] / sfreq
                next_gap = next_onset - offset
                post_score = 1 - ((next_gap - isi_min) / (isi_max - isi_min + 1e-6))
                post_score = np.clip(post_score, 0, 1)

            pre_predictor[start_idx:end_idx] = pre_score
            post_predictor[start_idx:end_idx] = post_score

        pre_prox_predictors.append(pre_predictor)
        post_prox_predictors.append(post_predictor)

    return pre_prox_predictors, post_prox_predictors


def save_overlap_predictors(overlap_predictor_pre, overlap_predictor_post, stream_type=''):
    predictor_concat_pre = np.concatenate(overlap_predictor_pre)
    predictor_concat_post = np.concatenate(overlap_predictor_post)
    overlap_ratios_path = predictors_path / 'events_proximity'
    save_path = overlap_ratios_path / sub / condition
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f'{sub}_{condition}_{stream_type}_proximity_series_concat.npz'
    np.savez(
        save_path / filename,
        events_proximity_pre=predictor_concat_pre,
        events_proximity_post=predictor_concat_post,
        sfreq=sfreq,
        stim_duration_samples=int(stim_dur * sfreq),
        stream_label=stream_type)
    # save separate block predictors:
    for i, (series_pre, series_post) in enumerate(zip(overlap_predictor_pre, overlap_predictor_post)):
        filename_block = f'{sub}_{condition}_{stream_type}_{i}_proximity_series.npz'
        np.savez(save_path/filename_block,
                 events_proximity_pre=series_pre,
                 events_proximity_post=series_post,
                 sfreq=sfreq,
                 stim_duration_samples=int(stim_dur * sfreq),
                 stream_label=stream_type)


if __name__ == '__main__':

    # focus on proximity to previous/next events in the same stream.
    # otherwise it will get really complicated, really fast.
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

    eeg_files_list, _ = load_eeg_files(sub=sub, condition=condition, results_path=results_path, sfreq=sfreq)
    eeg_lens = [eeg_file.n_times for eeg_file in eeg_files_list]
    stream1 = []
    stream2 = []
    for event_arrays in sub_path.iterdir():
        if 'stream1' in event_arrays.name:
            events = np.load(event_arrays)
            stream1.append(events)
        elif 'stream2' in event_arrays.name:
            events = np.load(event_arrays)
            stream2.append(events)

    proximity_predictors_pre1, proximity_predictors_post1 = get_ISIs(eeg_lens, stream1)
    proximity_predictors_pre2, proximity_predictors_post2 = get_ISIs(eeg_lens, stream2)

    if condition in ['a1', 'e1']:
        target_stream = stream1
        distractor_stream = stream2
    elif condition in ['a2', 'e2']:
        target_stream = stream2
        distractor_stream = stream1
    # For target stream (e.g. stream1) and targets
    pre_target_prox, post_target_prox = get_proximity_for_stim_type(target_stream, eeg_lens, stim_type='target',
                                                                    sfreq=sfreq)
    pre_nt_target_prox, post_nt_target_prox = get_proximity_for_stim_type(target_stream, eeg_lens,
                                                                          stim_type='nt_target', sfreq=sfreq)

    # For distractor stream and deviants
    pre_distractor_prox, post_distractor_prox = get_proximity_for_stim_type(distractor_stream, eeg_lens,
                                                                            stim_type='distractor', sfreq=sfreq)
    pre_nt_distractor_prox, post_nt_distractor_prox = get_proximity_for_stim_type(distractor_stream, eeg_lens,
                                                                                  stim_type='nt_distractor', sfreq=sfreq)
    pre_deviant_prox, post_deviant_prox = get_proximity_for_stim_type(distractor_stream, eeg_lens, stim_type='deviant',
                                                                      sfreq=sfreq)

    save_overlap_predictors(proximity_predictors_pre1, proximity_predictors_post1, stream_type='stream1')
    save_overlap_predictors(proximity_predictors_pre2, proximity_predictors_post2, stream_type='stream2')

    save_overlap_predictors(pre_target_prox, post_target_prox, stream_type='targets')
    save_overlap_predictors(pre_nt_target_prox, post_nt_target_prox, stream_type='nt_target')

    save_overlap_predictors(pre_distractor_prox, post_distractor_prox, stream_type='distractors')
    save_overlap_predictors(pre_nt_distractor_prox, post_nt_distractor_prox, stream_type='nt_distractor')
    save_overlap_predictors(pre_deviant_prox, post_deviant_prox, stream_type='deviants')
