from TRF_predictors.overlap_ratios import load_eeg_files
from TRF_test.TRF_test_config import frontal_roi
import mne
import numpy as np
from mne import time_frequency
import matplotlib
matplotlib.use('TkAgg')
import os
from pathlib import Path
import matplotlib.pyplot as plt


subs = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15',
        'sub17', 'sub18', 'sub19', 'sub20', 'sub21',
        'sub22', 'sub23', 'sub24', 'sub25', 'sub26',
        'sub27', 'sub28', 'sub29']

default_path = Path.cwd()
predictors_path = default_path / 'data/eeg/predictors'
eeg_results_path = default_path / 'data/eeg/preprocessed/results'

sfreq = 125

def get_eeg_files(condition=''):
    eeg_files = {}
    for folders in eeg_results_path.iterdir():
        if 'sub' in folders.name:
            if folders.name not in subs:
                continue
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

plane='elevation'
if plane == 'azimuth':
    condition1 = 'a1'
    condition2 = 'a2'
elif plane == 'elevation':
    condition1 = 'e1'
    condition2 = 'e2'

eeg_files1 = get_eeg_files(condition=condition1)
eeg_files2 = get_eeg_files(condition=condition2)

def pick_channels(eeg_files):
    eeg_concat_list = {}

    for sub, sub_list in eeg_files.items():
        if len(sub_list) > 0:
            eeg_concat = mne.concatenate_raws(sub_list)
            eeg_concat.resample(sfreq)
            eeg_concat.pick(frontal_roi)
            eeg_concat.filter(l_freq=None, h_freq=30)
            eeg_concat_list[sub] = eeg_concat
    return eeg_concat_list

eeg_concat_list1 = pick_channels(eeg_files1)
eeg_concat_list2 = pick_channels(eeg_files2)

from TRF_predictors.overlap_ratios import load_eeg_files
from TRF_test.TRF_test_config import frontal_roi
import mne
import numpy as np
from mne import time_frequency
import matplotlib
matplotlib.use('TkAgg')
import os
from pathlib import Path
import matplotlib.pyplot as plt


subs = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15',
        'sub17', 'sub18', 'sub19', 'sub20', 'sub21',
        'sub22', 'sub23', 'sub24', 'sub25', 'sub26',
        'sub27', 'sub28', 'sub29']

default_path = Path.cwd()
predictors_path = default_path / 'data/eeg/predictors'
eeg_results_path = default_path / 'data/eeg/preprocessed/results'

sfreq = 125

def mask_bad_segmets(eeg_concat_list, condition):
    eeg_masked_list = {}
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
                    eeg_masked_list[sub] = eeg_masked
                    break
        else:
            print(f"No bad segments found for {sub} {condition}, assuming all samples are good.")
            eeg_len = eeg_concat.n_times
            good_samples = np.ones(eeg_len, dtype=bool)
            eeg_masked = eeg_data[:, good_samples]
            eeg_masked_list[sub] = eeg_masked
    return eeg_masked_list

eeg_masked_list1 = mask_bad_segmets(eeg_concat_list1, condition1)
eeg_masked_list2 = mask_bad_segmets(eeg_concat_list2, condition2)

# === Configuration ===
base_dir = rf"C:\Users\pppar\PycharmProjects\VKK_Attention\data\eeg\trf\trf_testing\composite_model\single_sub\{plane}\all_stims\on_en_RT_ov\weights\predictions"
target_predictions = {}
distractor_predictions = {}

for files in os.listdir(base_dir):
    print(files)
    if 'target_stream' in files:
       target_array = np.load(os.path.join(base_dir, files))
       target_prediction = target_array['prediction']
       target_predictions[files[:5]] = target_prediction

    elif 'distractor_stream' in files:
       distractor_array = np.load(os.path.join(base_dir, files))
       distractor_prediction = distractor_array['prediction']
       distractor_predictions[files[:5]] = distractor_prediction


### Subtraction: Continuous EEG - Predicted EEG

residuals_target = {}
residuals_distractor = {}

for sub in subs:
    if sub not in target_predictions or sub not in eeg_masked_list1:
        print(f"Skipping {sub} (missing data).")
        continue

    # Target stream subtraction
    eeg_target = eeg_masked_list1[sub].mean(axis=0)  # shape: (samples,)
    pred_target = target_predictions[sub].squeeze() # shape: (samples,)
    min_len = min(len(eeg_target), len(pred_target))
    residual_target = eeg_target[:min_len] - pred_target[:min_len]
    residuals_target[sub] = residual_target

    # Distractor stream subtraction
    eeg_distractor = eeg_masked_list2[sub].mean(axis=0)  # shape: (samples,)
    pred_distractor = distractor_predictions[sub].squeeze()  # shape: (samples,)
    min_len = min(len(eeg_distractor), len(pred_distractor))
    residual_distractor = eeg_distractor[:min_len] - pred_distractor[:min_len]
    residuals_distractor[sub] = residual_distractor

    print(f"{sub}: Residuals computed (target & distractor).")

### Plotting Residuals
import scipy.signal

# === Plot residuals (smoothed with Hamming window) ===
plot_dir = default_path / "plots/residuals"
plot_dir.mkdir(parents=True, exist_ok=True)

window_len = 125  # ~1s smoothing for 125 Hz sampling
hamming_window = np.hamming(window_len)
hamming_window /= hamming_window.sum()

n_samples = 125 * 60  # 1 minute
for sub in residuals_target:
    res_tgt = residuals_target[sub]
    res_dst = residuals_distractor[sub]

    # Smooth using convolution
    smooth_tgt = np.convolve(res_tgt, hamming_window, mode='same')
    smooth_dst = np.convolve(res_dst, hamming_window, mode='same')

    # Limit to 1 minute
    smooth_tgt = smooth_tgt[:n_samples]
    smooth_dst = smooth_dst[:n_samples]

    # Plot
    plt.figure(figsize=(10, 4))
    plt.plot(smooth_tgt, label='Target Residual', alpha=0.8)
    plt.plot(smooth_dst, label='Distractor Residual', alpha=0.8)
    plt.title(f"Smoothed Residual EEG - {sub}")
    plt.xlabel("Samples (~125 Hz)")
    plt.ylabel("Amplitude (a.u.)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_dir / f"{sub}_residuals_plot.png", dpi=150)
    plt.close()
    print(f"{sub}: Residual plot saved.")

#### TFA on each sub's residuals:

from mne.time_frequency import tfr_array_morlet

# === TFA config ===
freqs = np.logspace(np.log10(1), np.log10(15), num=100)  # 30 log-spaced freqs from 2–30 Hz
n_cycles = freqs / 2.  # reasonable tradeoff between time/frequency resolution
sfreq = 125
tfa_results = {}

for sub in residuals_target:
    res_tgt = residuals_target[sub]
    res_dst = residuals_distractor[sub]

    # Limit to 1-minute if needed
    res_tgt = res_tgt
    res_dst = res_dst

    # Reshape for MNE: (n_epochs, n_channels, n_times)
    res_tgt = res_tgt[np.newaxis, np.newaxis, :]
    res_dst = res_dst[np.newaxis, np.newaxis, :]

    # Compute power
    power_tgt = tfr_array_morlet(res_tgt, sfreq=sfreq, freqs=freqs,
                                  n_cycles=n_cycles, output='power')[0, 0, :, :]
    power_dst = tfr_array_morlet(res_dst, sfreq=sfreq, freqs=freqs,
                                  n_cycles=n_cycles, output='power')[0, 0, :, :]

    tfa_results[sub] = {
        'target': power_tgt,
        'distractor': power_dst
    }


# Initialize lists for stacking
target_stack = []
distractor_stack = []

for sub in tfa_results:
    target_stack.append(tfa_results[sub]['target'])
    distractor_stack.append(tfa_results[sub]['distractor'])

# Convert to arrays and average
min_timepoints_target = min(arr.shape[1] for arr in target_stack)
min_timepoints_distractor = min(arr.shape[1] for arr in distractor_stack)

# Step 2: Trim each array to that length
target_stack_trimmed = [arr[:, :min_timepoints_target] for arr in target_stack]
distractor_stacked_trimmed = [arr[:, :min_timepoints_distractor] for arr in distractor_stack]

# Step 3: Stack and average
target_stack_array = np.stack(target_stack_trimmed, axis=0)  # shape: (subjects, freqs, time)
distractor_stacked_array = np.stack(distractor_stacked_trimmed, axis=0)

avg_power_target = np.mean(target_stack_array, axis=0)  # shape: (freqs, time)
avg_power_distractor = np.mean(distractor_stacked_array, axis=0) # shape: (freqs, time)

import mne

def residual_to_raw(residual_array, ch_names, sfreq):
    if residual_array.ndim == 1:
        residual_array = residual_array[np.newaxis, :]
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw_resid = mne.io.RawArray(residual_array, info)
    return raw_resid

target_eeg_resids = {}
for sub, residual_array in residuals_target.items():
    raw_resid = residual_to_raw(residual_array, frontal_roi, sfreq)
    target_eeg_resids[sub] = raw_resid


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