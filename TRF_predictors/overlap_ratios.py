import os
from pathlib import Path
import pandas as pd
import numpy as np
import mne
from TRF_predictors.config import sub, condition, \
    sfreq, stim_dur, \
    results_path, predictors_path, events_path, \
    base_target, base_distractor

from scipy.signal import welch


def load_eeg_files(sub='', condition='', results_path=None, sfreq=None):
    eeg_path = results_path / f'{sub}/ica'
    eeg_files_list = []
    eeg_events_list = []
    for sub_files in eeg_path.iterdir():
        if '.fif' in sub_files.name:
            if condition in sub_files.name:
                eeg_file = mne.io.read_raw_fif(sub_files, preload=True)
                data = eeg_file.get_data()
                f, psd = welch(data[0], fs=eeg_file.info['sfreq'])  # Use any channel
                signal_band = (f > 8) & (f < 13)  # Alpha band
                noise_band = (f > 20) & (f < 40)  # Noise floor
                snr = psd[signal_band].mean() / psd[noise_band].mean()
                print(f'SNR ratio: {snr}')
                eeg_file.resample(sfreq=sfreq)
                # get events from each eeg file
                eeg_events, eeg_event_ids = mne.events_from_annotations(eeg_file)
                # save each eeg file and eeg event list
                eeg_events_list.append((eeg_events, eeg_event_ids))
                eeg_files_list.append(eeg_file)
    return eeg_files_list, eeg_events_list


def get_overlap_predictors(stream1, stream2, base='s1'):
    primary_stream = stream1 if base == 's1' else stream2
    other_stream = stream2 if base == 's1' else stream1

    all_overlap_predictors = []

    for block_idx, (primary_array, other_array, eeg_len) in enumerate(zip(primary_stream, other_stream, eeg_lens)):
        overlap_predictor = np.zeros(eeg_len)  # generate empty predictor series with 0s with eeg len
        # Pick base stream
        for idx1, event1 in enumerate(primary_array):
            primary_onset = event1[0] / sfreq  # convert stim onset to s
            primary_offset = primary_onset + stim_dur  # stim offset in s

            best_overlap = 0  # Default: no overlap
            closest_tdiff = float('inf')  # positive infinity, which is just a big placeholder number

            for idx2, event2 in enumerate(other_array):
                other_onset = event2[0] / sfreq # onset and offset in s for other stream too
                other_offset = other_onset + stim_dur

                time_diff = abs(primary_onset - other_onset)  # get time diff between onsets of the two stims

                # Update closest time diff always
                if time_diff < closest_tdiff:  # for current primary stream stim, find other stream stim that is closest
                    closest_tdiff = time_diff  # replace infinity value of closest tdiff until smallest has been found
                    best_tdiff = primary_onset - other_onset  # assign the best time diff between closest onsets

                # Check for overlap (within same stim window)
                if time_diff > 0.9:  # if this time diff in onsets is smaller than 0.9
                    continue  # why? one Stim event is 745ms + ISI 70ms or 90ms + 0-10ms jitter (max possible len 845ms)

                overlap_start = max(primary_onset, other_onset)  # which was last? primary or the secondary stim onset?
                overlap_end = min(primary_offset, other_offset)  # larger onset - other stim's offset
                overlap_s = overlap_end - overlap_start  # overlap in s

                if overlap_s > 0:
                    overlap_ratio = overlap_s / stim_dur  # overlap in %
                    direction = np.sign(primary_onset - other_onset)  # -1 = other first, +1 = primary first
                    signed_overlap = overlap_ratio * direction

                    if abs(signed_overlap) > abs(best_overlap):
                        best_overlap = signed_overlap
                        # Convert overlap time to samples
                        start_idx = int(overlap_start * sfreq)
                        end_idx = int(overlap_end * sfreq)

                        # Clip to EEG range (avoid out-of-bounds errors)
                        start_idx = max(0, start_idx)
                        end_idx = min(eeg_len, end_idx)

                        # Fill 1s for the overlap window
                        overlap_predictor[start_idx:end_idx] = best_overlap
        all_overlap_predictors.append(overlap_predictor)
    return all_overlap_predictors


def save_predictor_blocks(predictors, stim_dur, stream_type=''):
    save_path = predictors_path / 'overlap_ratios' / sub / condition / stream_type
    save_path.mkdir(parents=True, exist_ok=True)
    for i, series in enumerate(predictors):
        filename_block = f'{sub}_{condition}_{stream_type}_{i}_overlap_ratios.npz'
        np.savez(save_path / filename_block,
                 overlap_ratios=series,
                 sfreq=sfreq,
                 stim_duration_samples=int(stim_dur * sfreq),
                 stream_label=stream_type)


def filter_stream(stream, stream_type=''):
    targets = []
    nt_targets = []
    distractors = []
    nt_distractors = []
    deviants = []
    for event_array in stream:
        target_block = []
        nt_target_block = []
        distractor_block = []
        nt_distractor_block = []
        deviant_block = []
        for event in event_array:
            if stream_type == 'targets':
                if event[1] == 4:
                    target = event
                    target_block.append(target)
                elif event[1] == 2:
                    nt_target = event
                    nt_target_block.append(nt_target)
            elif stream_type == 'distractors':
                if event[1] == 3:
                    distractor = event
                    distractor_block.append(distractor)
                elif event[1] == 1:
                    nt_distractor = event
                    nt_distractor_block.append(nt_distractor)
                elif event[1] == 2:
                    deviant = event
                    deviant_block.append(deviant)
        targets.append(target_block)
        nt_targets.append(nt_target_block)
        distractors.append(distractor_block)
        nt_distractors.append(nt_distractor_block)
        deviants.append(deviant_block)
    return targets, nt_targets, distractors, nt_distractors, deviants


def save_overlap_predictors(overlap_predictor, stream_type=''):
    predictor_concat = np.concatenate(overlap_predictor)
    overlap_ratios_path = predictors_path / 'overlap_ratios'
    save_path = overlap_ratios_path / sub / condition / stream_type
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f'{sub}_{condition}_{stream_type}_overlap_ratios_concat.npz'
    np.savez(
        save_path / filename,
        overlap_ratios=predictor_concat,
        sfreq=sfreq,
        stim_duration_samples=int(stim_dur * sfreq),
        stream_label=stream_type)


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

    # can apply this also for other target streams:
    eeg_files_list, _ = load_eeg_files(sub=sub, condition=condition, results_path=results_path, sfreq=sfreq)

    eeg_lens = [eeg_file.n_times for eeg_file in eeg_files_list]  # get EEG lengths
    # get overlap ratios:
    # if complete overlap = 1
    # none = 0
    # partial = between 0-1
    # if target was first: +
    # if distractor was first: -
    # use later to shape the attention predictor

    # load event arrays:

    # get overlap time-series for each stream separately.
    if condition in ['a1', 'e1']:
        target_stream = stream1
        distractor_stream = stream2
    elif condition in ['a2', 'e2']:
        target_stream = stream2
        distractor_stream = stream1
    stream1_overlap_predictor = get_overlap_predictors(stream1, stream2, base='s1')
    save_predictor_blocks(stream1_overlap_predictor, stim_dur, stream_type='stream1')
    stream2_overlap_predictor = get_overlap_predictors(stream1, stream2, base='s2')
    save_predictor_blocks(stream2_overlap_predictor, stim_dur, stream_type='stream2')

    targets, nt_targets, _, _, _ = filter_stream(target_stream, stream_type='targets')
    _, _, distractors, nt_distractors, deviants = filter_stream(distractor_stream, stream_type='distractors')

    targets_overlap_predictor = get_overlap_predictors(targets, distractor_stream, base=base_target)
    save_predictor_blocks(targets_overlap_predictor, stim_dur, stream_type='targets')
    nt_targets_overlap_predictor = get_overlap_predictors(nt_targets, distractor_stream, base=base_target)
    save_predictor_blocks(nt_targets_overlap_predictor, stim_dur, stream_type='nt_target')
    distractors_overlap_predictor = get_overlap_predictors(distractors, target_stream, base=base_distractor)
    save_predictor_blocks(distractors_overlap_predictor, stim_dur, stream_type='distractors')
    nt_distractors_overlap_predictor = get_overlap_predictors(nt_distractors, target_stream, base=base_distractor)
    save_predictor_blocks(nt_distractors_overlap_predictor, stim_dur, stream_type='nt_distractor')
    deviants_overlap_predictors = get_overlap_predictors(deviants, target_stream, base=base_distractor)
    save_predictor_blocks(deviants_overlap_predictors, stim_dur, stream_type='deviants')

    save_overlap_predictors(stream1_overlap_predictor, stream_type='stream1')
    save_overlap_predictors(stream2_overlap_predictor, stream_type='stream2')

    save_overlap_predictors(targets_overlap_predictor, stream_type='targets')
    save_overlap_predictors(nt_targets_overlap_predictor, stream_type='nt_target')

    save_overlap_predictors(distractors_overlap_predictor, stream_type='distractors')
    save_overlap_predictors(nt_distractors_overlap_predictor, stream_type='nt_distractor')
    save_overlap_predictors(deviants_overlap_predictors, stream_type='deviants')