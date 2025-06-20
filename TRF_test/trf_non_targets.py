import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()
import matplotlib.pyplot as plt
import mtrf
from mtrf import TRF
from mtrf.stats import crossval
from pathlib import Path
import os
import mne
from TRF_test.TRF_test_config import frontal_roi

folder_types = ['non_targets', 'target_nums', 'deviants']
stream_type1 = 'nt_target'
stream_type2 = 'nt_distractor'
folder_type = folder_types[0]


default_path = Path.cwd()
predictors_path = default_path / 'data/eeg/predictors'
eeg_results_path = default_path / 'data/eeg/preprocessed/results'


plane = 'elevation'
if plane == 'azimuth':
    cond = 'a1'
elif plane == 'elevation':
    cond = 'e1'


subs = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15',
        'sub17', 'sub18', 'sub19', 'sub20', 'sub21',
        'sub22', 'sub23', 'sub24', 'sub25', 'sub26',
        'sub27', 'sub28', 'sub29']


sfreq = 125

predictors_list = ['binary_weights', 'envelopes', 'overlap_ratios', 'RTs']
pred_types = ['onsets', 'envelopes', 'overlap_ratios', 'RT_labels']

stim1 = 'target_stream'
stim2 = 'distractor_stream'


def get_eeg_files(condition=''):
    eeg_files = {}
    for folders in eeg_results_path.iterdir():
        if 'sub' in folders.name:
            sub_data = []
            for files in folders.iterdir():
                if 'ica' in files.name:
                    for data in files.iterdir():
                        if condition in data.name:
                            eeg = mne.io.read_raw_fif(data, preload=True)
                            eeg.set_eeg_reference('average')
                            eeg.resample(sfreq=sfreq)
                            sub_data.append(eeg)
            eeg_files[folders.name] = sub_data
    return eeg_files


def pick_channels(eeg_files):
    eeg_concat_list = {}

    for sub, sub_list in eeg_files.items():
        if len(sub_list) > 0:
            eeg_concat = mne.concatenate_raws(sub_list)
            eeg_concat.resample(sfreq)
            eeg_concat.pick(frontal_roi)
            eeg_concat.filter(l_freq=1, h_freq=30)
            eeg_concat_list[sub] = eeg_concat
    return eeg_concat_list


def mask_bad_segmets(eeg_concat_list, condition):
    eeg_clean_list = {}
    for sub in eeg_concat_list:
        eeg_concat = eeg_concat_list[sub]
        eeg_data = eeg_concat.get_data()

        sub_bad_segments_path = predictors_path / 'bad_segments' / sub / condition

        if sub_bad_segments_path.exists():
            for file in sub_bad_segments_path.iterdir():
                if 'concat.npy' in file.name:
                    bad_segments = np.load(file)
                    bad_series = bad_segments['bad_series']
                    good_samples = bad_series == 0  # Boolean mask
                    print(f"Loaded bad segments for {sub} {condition}.")
                    eeg_masked = eeg_data[:, good_samples]
                    # z-scoring..
                    eeg_clean = (eeg_masked - eeg_masked.mean(axis=1, keepdims=True)) / eeg_masked.std(axis=1, keepdims=True)
                    print(f"{sub}: Clean EEG shape = {eeg_clean.shape}")
                    eeg_clean_list[sub] = eeg_clean
                    break
        else:
            print(f"No bad segments found for {sub} {condition}, assuming all samples are good.")
            eeg_len = eeg_concat.n_times
            good_samples = np.ones(eeg_len, dtype=bool)
            eeg_masked = eeg_data[:, good_samples]
            # z-scoring..
            eeg_clean = (eeg_masked - eeg_masked.mean(axis=1, keepdims=True)) / eeg_masked.std(axis=1, keepdims=True)
            print(f"{sub}: Clean EEG shape = {eeg_clean.shape}")
            eeg_clean_list[sub] = eeg_clean
    return eeg_clean_list


def remap_onsets_nested(predictor_dict):
    remapped_dict = {}
    for subj, stream_dict in predictor_dict.items():
        remapped_streams = {}
        for stream_key, arr in stream_dict.items():
            remapped_arr = arr.copy()
            for orig_val, new_val in semantic_mapping.items():
                remapped_arr[arr == orig_val] = new_val
            remapped_streams[stream_key] = remapped_arr
        remapped_dict[subj] = remapped_streams
    return remapped_dict


def centering_predictor_array(predictor_array, min_std=1e-6, predictor_name=''):
    """
    Normalize predictor arrays for TRF modeling.

    Rules:
    - 'envelopes' → z-score if dense and std > min_std.
    - All other predictors → left unchanged.
    - Fully zero predictors → returned unchanged.
    """

    if predictor_name == 'envelopes':
        std = predictor_array.std()
        nonzero_ratio = np.count_nonzero(predictor_array) / len(predictor_array)

        if nonzero_ratio > 0.5:
            print(f'{predictor_name}: Dense predictor, applying z-score.')
            mean = predictor_array.mean()
            return (predictor_array - mean) / std if std > min_std else predictor_array - mean

        elif nonzero_ratio > 0:
            print(f'{predictor_name}: Sparse predictor, mean-centering non-zero entries.')
            mask = predictor_array != 0
            mean = predictor_array[mask].mean()
            predictor_array = predictor_array.copy()
            predictor_array[mask] -= mean
            return predictor_array

        else:
            print(f'{predictor_name}: All zeros, returning unchanged.')
            return predictor_array

    else:
        print(f'{predictor_name}: No normalization applied.')
        return predictor_array


def get_predictor_dict(condition='', pred_type=''):
    predictor_dict = {}
    for files in predictor.iterdir():
        if 'sub' in files.name:
            sub_name = files.name  # e.g., "sub01"
            stream1_data, stream2_data = None, None
            for file in files.iterdir():
                if condition in file.name:
                    for stim_type in file.iterdir():
                        if stim_type.name == stream_type1:
                            for array in stim_type.iterdir():
                                if 'concat' in array.name:
                                    print(array)
                                    stream1_data = np.load(array)
                                    stream1_data = stream1_data[f'{pred_type}']
                        elif stim_type.name == stream_type2:
                            for array in stim_type.iterdir():
                                if 'concat' in array.name:
                                    stream2_data = np.load(array)
                                    stream2_data = stream2_data[f'{pred_type}']

            if stream1_data is not None and stream2_data is not None:
                predictor_dict[sub_name] = {
                    'stream1': stream1_data,
                    'stream2': stream2_data
                }
                print(f"Loaded predictors for {sub_name}: {stream1_data.shape}, {stream2_data.shape}")
            else:
                print(f"Missing predictor(s) for {sub_name} {condition}")
    return predictor_dict

def define_streams_dict(predictors):
    for pred_type, pred_dict in predictors.items():
        for sub, sub_dict in pred_dict.items():
            sub_dict[f'{stim1}'] = sub_dict.pop('stream1')  # pop to replace OG array, not add extra array with new key
            sub_dict[f'{stim2}'] = sub_dict.pop('stream2')
    return predictors


def predictor_mask_bads(predictor_dict, condition, predictor_name=''):
    predictor_dict_masked = {}
    predictor_dict_masked_raw = {}

    for sub, sub_dict in predictor_dict.items():
        sub_bad_segments_path = predictors_path / 'bad_segments' / sub / condition
        good_samples = None  # default fallback

        # --- Load bad segment mask if available
        if sub_bad_segments_path.exists():
            for file in sub_bad_segments_path.iterdir():
                if 'concat.npy' in file.name:
                    bad_segments = np.load(file)
                    bad_series = bad_segments['bad_series']
                    good_samples = bad_series == 0  # Boolean mask
                    print(f"Loaded bad segments for {sub} - {condition}.")
                    break  # Stop after finding the correct file

        sub_masked = {}
        sub_masked_raw = {}

        for stream_name, stream_array in sub_dict.items():
            if good_samples is not None and len(good_samples) == len(stream_array):
                stream_array_masked = stream_array[good_samples]
                stream_array_clean = centering_predictor_array(stream_array_masked, min_std=1e-6, predictor_name=predictor_name)
            else:
                stream_array_masked = stream_array  #full array if no mask found or mismatched
                stream_array_clean = centering_predictor_array(stream_array_masked, min_std=1e-6, predictor_name=predictor_name)
            sub_masked[stream_name] = stream_array_clean
            sub_masked_raw[stream_name] = stream_array_masked

        predictor_dict_masked[sub] = sub_masked
        predictor_dict_masked_raw[sub] = sub_masked_raw
    return predictor_dict_masked, predictor_dict_masked_raw


def get_events_dicts(folder_name1, folder_name2):
    event_length = int(0.745 * 125)  # 745ms at 125Hz
    weights_dir = default_path / 'data/eeg/predictors/binary_weights'
    target_mne_events = {}
    distractor_mne_events = {}

    for folders in weights_dir.iterdir():
        if folders.name in subs:
            for sub_folders in folders.iterdir():
                if cond in sub_folders.name:
                    for stim_folders in sub_folders.iterdir():
                        if folder_name1 in stim_folders.name:
                            stream_type = 'target'
                        elif folder_name2 in stim_folders.name:
                            stream_type = 'distractor'
                        else:
                            continue

                        # === Only process files once, avoiding overwrite ===
                        concat_files = [f for f in stim_folders.iterdir() if 'concat.npz' in f.name]
                        if not concat_files:
                            continue  # skip if no relevant file

                        file = np.load(concat_files[0], allow_pickle=True)
                        stream_data = file['onsets']

                        stream = stream_data.copy()

                        # Keep only onset value for each event
                        i = 0
                        while i < len(stream):
                            if stream[i] in [1, 2, 3, 4]:
                                stream[i + 1:i + event_length] = 0
                                i += event_length
                            else:
                                i += 1

                        onset_indices = np.where(stream != 0)[0]
                        event_values = stream[onset_indices].astype(int)
                        mne_events = np.column_stack((onset_indices,
                                                      np.zeros_like(onset_indices),
                                                      event_values))

                        if stream_type == 'target':
                            target_mne_events[folders.name] = mne_events
                        elif stream_type == 'distractor':
                            distractor_mne_events[folders.name] = mne_events
    return target_mne_events, distractor_mne_events



def get_event_timepoints(events, stream_name=''):
    """
    Extract timepoints per event type for a given stream.

    Parameters:
    -----------
    events : dict
        Dictionary of events per subject. Each value is a list of [sample, 0, event_code].
    stream_name : str
        Either 'target' or 'distractor'.

    Returns:
    --------
    dicts of timepoints by event code per subject.
    """
    if stream_name not in ['target', 'distractor']:
        raise ValueError("stream_name must be 'target' or 'distractor'")

    target_tp, nt_tp = {}, {}

    if stream_name == 'target':
        for sub, ev in events.items():
            target_tp[sub] = [e[0] for e in ev if e[2] == 4]  # target numbers
            nt_tp[sub] = [e[0] for e in ev if e[2] == 2]  # non-targets
            # print(f"{sub} | Target: {len(target_tp[sub])}, NT: {len(nt_tp[sub])}")

        return target_tp, nt_tp

    elif stream_name == 'distractor':
        deviant_tp = {}
        for sub, ev in events.items():
            target_tp[sub] = [e[0] for e in ev if e[2] == 3]  # distractor numbers
            nt_tp[sub] = [e[0] for e in ev if e[2] == 1]  # non-targets
            deviant_tp[sub] = [e[0] for e in ev if e[2] == 2]  # deviants
            # print(f"{sub} | Distractor: {len(target_tp[sub])}, NT: {len(nt_tp[sub])}, Deviant: {len(deviant_tp[sub])}")
        return target_tp, nt_tp, deviant_tp


def zero_out_predictor_windows(predictor_array, timepoints, duration):
    for tp in timepoints:
        predictor_array[tp:tp + duration] = 0
    return predictor_array


def zero_out_predictor_windows_variable_duration(predictor_array, timepoints):
    valid_timepoints = [tp for tp in timepoints if tp < len(predictor_array)]
    for tp in valid_timepoints:
        val = predictor_array[tp]
        if val != 0:
            # Zero until the value changes (i.e., end of that event segment)
            end = tp
            while end < len(predictor_array) and predictor_array[end] == val:
                end += 1
            predictor_array[tp:end] = 0
    return predictor_array



def match_trial_counts_and_mask_predictors(sub, nt_tps, dev_tps, predictors, selected_stream='', force_n_keep=None):
    n_deviants = len(dev_tps)
    n_to_keep = min(n_deviants, len(nt_tps))
    kept_nt_tps = sorted(random.sample(nt_tps, n_to_keep))
    dropped_nt_tps = set(nt_tps) - set(kept_nt_tps)

    print(f"Sub {sub}: Keeping {len(kept_nt_tps)}, Dropping {len(dropped_nt_tps)} non-targets")

    for pred_type in ['onsets', 'envelopes']:
        array = predictors[pred_type][sub][selected_stream]
        predictors[pred_type][sub][selected_stream] = zero_out_predictor_windows(
            array.copy(), dropped_nt_tps, stim_dur_samples)

    for pred_type in ['overlap_ratios', 'RT_labels']:
        array = predictors[pred_type][sub][selected_stream]
        predictors[pred_type][sub][selected_stream] = zero_out_predictor_windows_variable_duration(
            array.copy(), dropped_nt_tps)

    return predictors

eeg_files = get_eeg_files(condition=cond)

from copy import deepcopy
eeg_frontal = deepcopy(eeg_files)

eeg_frontal = pick_channels(eeg_frontal)

eeg_masked = mask_bad_segmets(eeg_frontal, cond)


target_events, distractor_events = get_events_dicts(folder_name1='stream1', folder_name2='stream2')

s1_predictors = {}
# Mapping semantic weights

semantic_mapping = {
    5.0: 1.0,
    4.0: 1,
    3.0: 1,
    2.0: 1,
    1.0: 1
}

for predictor_name, pred_type in zip(predictors_list, pred_types):
    predictor = default_path / f'data/eeg/predictors/{predictor_name}'
    predictor_dict1 = get_predictor_dict(condition=cond, pred_type=pred_type)
    predictor_dict_masked1, predictor_dict_masked_raw1 = predictor_mask_bads(predictor_dict1, condition=cond, predictor_name=pred_type)

    # Remap onsets (semantic weights) before storing
    if pred_type == 'onsets':
        predictor_dict_masked1 = remap_onsets_nested(predictor_dict_masked1)

    s1_predictors[pred_type] = predictor_dict_masked1

s1_predictors = define_streams_dict(s1_predictors)

target_t, nt_t = get_event_timepoints(target_events, stream_name='target')
distractor_t, nt_d, deviant_t = get_event_timepoints(distractor_events, stream_name='distractor')

import random

sfreq = 125
stim_dur_samples = int(0.745 * sfreq)


if folder_type == 'target_nums':
    # Compare target_nums (event code 4) vs deviants (event code 2)
    for sub in subs:
        s1_predictors = match_trial_counts_and_mask_predictors(
            sub=sub,
            nt_tps=target_t[sub],       # target_nums in attended stream
            dev_tps=deviant_t[sub],     # deviants in unattended stream
            predictors=s1_predictors,
            selected_stream='target_stream',
            force_n_keep=None # predictors for targets only
        )
        s1_predictors = match_trial_counts_and_mask_predictors(
            sub=sub,
            nt_tps=distractor_t[sub],  # target_nums in unattended stream
            dev_tps=deviant_t[sub],  # deviants in unattended stream
            predictors=s1_predictors,
            selected_stream='distractor_stream',
            force_n_keep=None  # predictors for targets only
        )

elif folder_type == 'deviants':
    # Compare deviants (event code 2) vs target_nums (event code 4)
    for sub in subs:
        s1_predictors = match_trial_counts_and_mask_predictors(
            sub=sub,
            nt_tps=deviant_t[sub],      # deviants in unattended stream
            dev_tps=target_t[sub],      # target_nums in attended stream
            predictors=s1_predictors,
            selected_stream='distractor_stream',
            force_n_keep=None
            # predictors for deviants only
        )

elif folder_type == 'non_targets':
    for sub in subs:
        # Mask each stream based on the non-kept timepoints
        s1_predictors = match_trial_counts_and_mask_predictors(
            sub=sub,
            nt_tps=nt_t[sub],
            dev_tps=deviant_t[sub],  # unused
            predictors=s1_predictors,
            selected_stream='target_stream')

        s1_predictors = match_trial_counts_and_mask_predictors(
            sub=sub,
            nt_tps=nt_d[sub],
            dev_tps=deviant_t[sub],
            predictors=s1_predictors,
            selected_stream='distractor_stream')

# === Set output directory for averaged-subject analysis ===
predictor_short = "_".join([p[:2] for p in pred_types])
output_dir = default_path / f"data/eeg/trf/trf_testing/results/averaged/{plane}/{cond}/{folder_type}/{predictor_short}"
output_dir.mkdir(parents=True, exist_ok=True)

weights_dir = output_dir / "weights"
weights_dir.mkdir(parents=True, exist_ok=True)

selected_streams = ['target_stream', 'distractor_stream']

for selected_stream in selected_streams:
    if folder_type == 'deviants' and selected_stream == 'target_stream':
        continue

    all_weights = []
    all_r = []
    all_r_cv = []

    for sub in subs:
        onsets = s1_predictors['onsets'][sub][selected_stream]
        envelopes = s1_predictors['envelopes'][sub][selected_stream]
        overlap_ratios = s1_predictors['overlap_ratios'][sub][selected_stream]
        RT_labels = s1_predictors['RT_labels'][sub][selected_stream]
        eeg = eeg_masked[sub].T  # shape: [time, channels]

        X = np.vstack([onsets, envelopes, overlap_ratios, RT_labels]).T
        min_len = min(len(X), len(eeg))
        X = X[:min_len]
        eeg = eeg[:min_len]

        trf = TRF(direction=1)
        trf.train(X, eeg, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=1.0, seed=42)
        _, r = trf.predict(X, eeg)

        # Cross-validation
        X1, X2 = np.array_split(X, 2)
        eeg1, eeg2 = np.array_split(eeg, 2)
        r_cv = crossval(trf, [X1, X2], [eeg1, eeg2], fs=sfreq, tmin=-0.1, tmax=1.0, regularization=1.0, seed=42)

        all_weights.append(trf.weights)
        all_r.append(r)
        all_r_cv.append(r_cv)


    # === Average TRF weights across subjects ===
    env_weights = [weights[1, :, :] for weights in all_weights]
    avg_weights = np.mean(env_weights, axis=-1)
    avg_r = np.mean(all_r)
    avg_r_cv = np.mean(all_r_cv)

    # Save
    np.save(weights_dir / f"avg_trf_weights_{selected_stream}.npy", avg_weights)

    np.savez(
        weights_dir / f"avg_trf_prediction_{selected_stream}.npz",
        prediction=None,  # prediction not averaged
        time_lags=trf.times,
        stream=selected_stream,
        plane=plane,
        r_value=avg_r,
        r_crossval=avg_r_cv,
        num_predictors=avg_weights.shape[0],
        num_lags=avg_weights.shape[1]
    )

    print(f"Saved averaged TRF for {selected_stream} (mean r = {avg_r:.4f}, crossval = {avg_r_cv:.4f})")