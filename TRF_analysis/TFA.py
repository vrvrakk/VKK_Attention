# === TFA on predicted EEG === #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
import os
from pathlib import Path
import mne
import pandas
from copy import deepcopy
from TRF_test.TRF_test_config import frontal_roi
from scipy.stats import ttest_rel, wilcoxon, shapiro
import pandas as pd
import seaborn as sns
from scipy.signal import hilbert
from scipy.fft import fft



# === Load relevant events and mask the bad segments === #

def get_pred_dicts(cond):
    predictions_dir = fr'C:/Users/pppar/PycharmProjects/VKK_Attention/data/eeg/trf/trf_testing/composite_model/single_sub/{plane}/{cond}/{folder_type}/{predictor_short}/weights/predictions'
    target_preds_dict = {}
    distractor_preds_dict = {}
    for pred_files in os.listdir(predictions_dir):
        if 'target_stream' in pred_files:
            target_predictions = np.load(os.path.join(predictions_dir, pred_files))
            sub = str(target_predictions['subject'])
            target_preds_dict[sub] = target_predictions['prediction'].squeeze()
        elif 'distractor_stream' in pred_files:
            distractor_predictions = np.load(os.path.join(predictions_dir, pred_files))
            sub = str(distractor_predictions['subject'])
            distractor_preds_dict[sub] = distractor_predictions['prediction'].squeeze()
    return target_preds_dict, distractor_preds_dict

def get_eeg_files(condition=''):
    eeg_files = {}
    for folders in eeg_results_path.iterdir():
        if folders.name in subs:
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


def get_events_dicts(folder_name1, folder_name2, cond):
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
                                stream[i+1:i+event_length] = 0
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


def drop_bad_segments(sub, cond, raw_copy):
    bad_segments_path = default_path / f'data/eeg/predictors/bad_segments/{sub}/{cond}'
    for bad_series in bad_segments_path.iterdir():
        if 'concat.npy.npz' in bad_series.name:
            bad_array = np.load(bad_series, allow_pickle=True)
            bads = bad_array['bad_series']
            good_samples = bads != -999
            raw_data = raw_copy._data
            raw_masked = raw_data[:,good_samples]
    return raw_masked



def get_residual_eegs(preds_dict=None, eeg_files=None, mne_events=None, cond=''):
    eeg_files_copy = deepcopy(eeg_files)
    epochs_dict = {}

    for sub in subs:
        print(f"\n[CHECKPOINT] Processing {sub}...")

        eeg_predicted = preds_dict[sub]
        print(f"[CHECKPOINT] {sub} prediction shape: {eeg_predicted.shape}")

        raw = mne.concatenate_raws(eeg_files_copy[sub])
        raw_copy = deepcopy(raw)
        raw_copy.pick(frontal_roi)
        print(f"[CHECKPOINT] {sub} prediction x eeg copy shape: {eeg_predicted.shape} x {raw_copy._data.shape}")


        # Drop bad segments
        raw_clean = drop_bad_segments(sub, cond, raw_copy)
        raw_clean = raw_clean.mean(axis=0)
        print(f"[CHECKPOINT] {sub} prediction x eeg copy shape: {eeg_predicted.shape} x {raw_clean.shape}")


        info = mne.create_info(ch_names=['avg'], sfreq=raw_copy.info['sfreq'], ch_types='eeg')

        # Subtract prediction from EEG to get residual

        eeg_residual = mne.io.RawArray(eeg_predicted[np.newaxis, :], info)

        # --- Event Filtering ---
        events = mne_events[sub]
        print(f"[CHECKPOINT] {sub} events loaded: {len(events)}")

        sfreq = raw.info['sfreq']
        n_samples = raw.n_times
        bad_time_mask = np.zeros(n_samples, dtype=bool)

        for ann in raw.annotations:
            if 'bad' in ann['description'].lower():
                start = int(ann['onset'] * sfreq)
                end = int((ann['onset'] + ann['duration']) * sfreq)
                bad_time_mask[start:end] = True

        filtered_events = np.array([
            ev for ev in events if not bad_time_mask[ev[0]]
        ])
        print(f"[INFO] {sub} events after bad segment exclusion: {len(filtered_events)}")

        # Filter events that fit epoch window
        tmin = -0.5
        tmax = 0.0
        tmin_samples = int(abs(tmin) * sfreq)
        tmax_samples = int(tmax * sfreq)

        valid_events = filtered_events[
            (filtered_events[:, 0] - tmin_samples >= 0) &
            (filtered_events[:, 0] + tmax_samples < n_samples)
            ]
        print(f"[CHECKPOINT] {sub} valid events after edge trimming: {len(valid_events)}")

        # Create epochs
        event_id = {str(i): i for i in np.unique(valid_events[:, 2].astype(int))}
        print(event_id)
        epochs = mne.Epochs(eeg_residual, events=valid_events.astype(int), event_id=event_id,
                            tmin=tmin, tmax=tmax, baseline=(tmin, -0.3), preload=True)

        print(f"[CHECKPOINT] {sub} residual epochs shape: {epochs.get_data().shape}")

        epochs_dict[sub] = epochs

    print(f"\n[CHECKPOINT] All subjects processed for residual epochs.\n")
    return epochs_dict


def extract_pre_stim_phase(epochs, freq_band=(4, 7)):
    """
    Bandpass filter epochs, apply Hilbert, return phase at t=0 per trial.
    """
    epochs_band = epochs.copy().filter(freq_band[0], freq_band[1], fir_design='firwin')
    data = epochs_band.get_data()[:, 0, :]  # (n_epochs, n_times)
    analytic = hilbert(data, axis=1)
    phase = np.angle(analytic)
    zero_idx = np.argmin(np.abs(epochs.times))
    return phase[:, zero_idx]

def bin_phases_and_get_behavior(phases, behavior_array, n_bins=8):
    bins = np.linspace(-np.pi, np.pi, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    binned = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (phases >= bins[i]) & (phases < bins[i+1])
        binned[i] = behavior_array[mask].mean() if np.any(mask) else np.nan

    return bin_centers, binned

def modulation_index(y):
    y = y - np.nanmean(y)
    fft_vals = np.abs(fft(y))
    return fft_vals[1] / fft_vals[0]  # 1st harmonic / DC



if __name__ == '__main__':
    pred_types = ['onsets', 'envelopes']
    predictor_short = "_".join([p[:2] for p in pred_types])

    subs = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19', 'sub20',
            'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29']

    plane = 'elevation'
    if plane == 'azimuth':
        cond1 = 'a1'
        cond2 = 'a2'
    elif plane == 'elevation':
        cond1 = 'e1'
        cond2 = 'e2'

    folder_types = ['all_stims', 'non_targets', 'target_nums', 'deviants']
    folder_type = folder_types[0]
    sfreq = 125

    default_path = Path.cwd()
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'


    target_preds_dict1, distractor_preds_dict1 = get_pred_dicts(cond=cond1)
    target_preds_dict2, distractor_preds_dict2 = get_pred_dicts(cond=cond2)

    eeg_files1 = get_eeg_files(condition=cond1)
    eeg_files2 = get_eeg_files(condition=cond2)

    target_mne_events1, distractor_mne_events1 = get_events_dicts(folder_name1='stream1', folder_name2='stream2', cond=cond1)

    if folder_type == 'target_nums':
        # Keep only code 4 in target, and 3 in distractor
        for sub in subs:
            target_mne_events1[sub] = target_mne_events1[sub][target_mne_events1[sub][:, 2] == 4]
            distractor_mne_events1[sub] = distractor_mne_events1[sub][distractor_mne_events1[sub][:, 2] == 3]

    elif folder_type == 'non_targets':
        # Keep only code 2 in target, and 1 in distractor
        for sub in subs:
            target_mne_events1[sub] = target_mne_events1[sub][target_mne_events1[sub][:, 2] == 2]
            distractor_mne_events1[sub] = distractor_mne_events1[sub][distractor_mne_events1[sub][:, 2] == 1]

    elif folder_type == 'deviants':
        # Keep only code 4 in target, and 2 in distractor
        for sub in subs:
            target_mne_events1[sub] = target_mne_events1[sub][target_mne_events1[sub][:, 2] == 4]
            distractor_mne_events1[sub] = distractor_mne_events1[sub][distractor_mne_events1[sub][:, 2] == 2]
    # else (e.g., "all_stims"), don't filter

    target_mne_events2, distractor_mne_events2 = get_events_dicts(folder_name1='stream2', folder_name2='stream1',cond=cond2)

    if folder_type == 'target_nums':
        for sub in subs:
            target_mne_events2[sub] = target_mne_events2[sub][target_mne_events2[sub][:, 2] == 4]
            distractor_mne_events2[sub] = distractor_mne_events2[sub][distractor_mne_events2[sub][:, 2] == 3]

    elif folder_type == 'non_targets':
        for sub in subs:
            target_mne_events2[sub] = target_mne_events2[sub][target_mne_events2[sub][:, 2] == 2]
            distractor_mne_events2[sub] = distractor_mne_events2[sub][distractor_mne_events2[sub][:, 2] == 1]

    elif folder_type == 'deviants':
        for sub in subs:
            target_mne_events2[sub] = target_mne_events2[sub][target_mne_events2[sub][:, 2] == 4]
            distractor_mne_events2[sub] = distractor_mne_events2[sub][distractor_mne_events2[sub][:, 2] == 2]


    targets_epochs_dict1 = get_residual_eegs(target_preds_dict1, eeg_files1, target_mne_events1, cond=cond1)
    distractors_epochs_dict1 = get_residual_eegs(distractor_preds_dict1, eeg_files1, distractor_mne_events1, cond=cond1)

    targets_epochs_dict2 = get_residual_eegs(target_preds_dict2, eeg_files2, target_mne_events2, cond=cond2)
    distractors_epochs_dict2 = get_residual_eegs(distractor_preds_dict2, eeg_files2, distractor_mne_events2, cond=cond2)

    freqs = np.logspace(np.log10(1), np.log10(30), num=100)  # 30 log-spaced frequencies
    n_cycles = freqs / 2  # balance temporal/frequency resolution
    phase_dict = {}  # store phase angles per subject

    for sub, epochs in targets_epochs_dict1.items():
        print(f"Processing {sub}...")

        # Compute TFR (returns one channel per epoch)
        power = epochs.compute_tfr(
            method='multitaper',
            freqs=freqs,
            n_cycles=n_cycles,
            average=False,  # Keep single-trial data
            n_jobs=1,
        )

        ch_idx = 0
        f_idx = np.argmin(np.abs(freqs - 5))
        t_idx = np.argmin(np.abs(power.times - -0.2))

        phases = np.angle(power.data[:, ch_idx, f_idx, t_idx])  # shape: (n_trials,)

        # Save per subject
        phase_dict[sub] = phases
