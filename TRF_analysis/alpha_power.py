
# === TFA on predicted EEG === #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
import os
from pathlib import Path
import mne
from copy import deepcopy
from TRF_test.TRF_test_config import frontal_roi
import json
from mne.time_frequency import psd_array_welch

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

def get_epochs(eeg_files=None, cond='', roi=None):

    eeg_files_copy = deepcopy(eeg_files)
    epochs_dict = {}

    for sub in subs:
        print(f"/n[CHECKPOINT] Processing {sub}...")
        raw = mne.concatenate_raws(eeg_files_copy[sub])
        raw_copy = deepcopy(raw)
        raw_copy.pick(roi)

        # Drop bad segments
        raw_clean = drop_bad_segments(sub, cond, raw_copy)

        info = mne.create_info(ch_names=raw_copy.info['ch_names'], sfreq=raw.info['sfreq'], ch_types='eeg')

        # Subtract prediction from EEG to get residual

        eeg = mne.io.RawArray(raw_clean, info)
        eeg.filter(l_freq=1, h_freq=30, method='fir', fir_design='firwin', phase='zero')
        if eeg.get_montage() is None:
            eeg.set_montage('standard_1020')  # apply standard montage

        eeg.set_eeg_reference('average')

        epoch_length_sec = 60  # duration of each epoch in seconds
        sfreq = eeg.info['sfreq']
        samples_per_epoch = int(epoch_length_sec * sfreq)
        n_epochs = eeg.n_times // samples_per_epoch  # floor division

        print(f"[INFO] Creating {n_epochs} epochs of {epoch_length_sec}s each for {sub}")

        # Create annotations for epochs
        onsets = np.arange(0, n_epochs * samples_per_epoch, samples_per_epoch) / sfreq
        durations = np.ones(n_epochs) * epoch_length_sec
        descriptions = ['1min_epoch'] * n_epochs
        annotations = mne.Annotations(onsets, durations, descriptions)
        eeg.set_annotations(annotations)

        # Convert to epochs
        events, event_id = mne.events_from_annotations(eeg)
        epochs = mne.Epochs(eeg, events, event_id=event_id,
                            tmin=0, tmax=epoch_length_sec, baseline=None,
                            detrend=1, preload=True)

        # Store the epochs
        epochs_dict[sub] = epochs


    print(f"/n[CHECKPOINT] All subjects processed for epochs./n")
    return epochs_dict

def compute_tfr(epochs_dict, freqs, n_cycles):
    powers = {}
    for sub, epochs in epochs_dict.items():
        tfr = epochs.compute_tfr(
            method="morlet",
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=False,
            average=False,
            decim=1,
            n_jobs=1)
        powers[sub] = tfr
    return powers

def extract_alpha_metrics(power_dict):
    alpha_metrics = {}
    for sub, power in power_dict.items():
        power = np.array(power)  # Ensure it's a NumPy array
        power_mean = np.mean(power)
        power_rms = np.sqrt(np.mean(power ** 2))

        alpha_metrics[sub] = {
            'trials_power': power,
            'mean': power_mean,
            'RMS': power_rms
        }
    return alpha_metrics

def compute_relative_alpha_power(epochs_dict, occipital_channels, alpha_band=(8, 12), total_band=(1, 30)):
    relative_alpha_metrics = {}

    for sub, epochs in epochs_dict.items():
        # Compute PSD using Welch
        psd = epochs.compute_psd(method='welch', fmin=total_band[0], fmax=total_band[1], n_fft=1024)
        psds = psd.get_data()  # shape: (n_epochs, n_channels, n_freqs)
        freqs = psd.freqs
        ch_names = psd.ch_names

        # Find indices for occipital and non-occipital channels
        occ_idx = [ch_names.index(ch) for ch in occipital_channels if ch in ch_names]
        non_occ_idx = [i for i in range(len(ch_names)) if i not in occ_idx]

        # Alpha band mask
        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])

        # Sum alpha power over alpha band
        alpha_power = np.sum(psds[:, :, alpha_mask], axis=-1)  # shape: (n_epochs, n_channels)

        # Average across epochs
        alpha_power_mean = np.mean(alpha_power, axis=0)  # shape: (n_channels,)

        # Compute mean alpha in occipital and non-occipital channels
        occ_alpha = np.mean(alpha_power_mean[occ_idx]) if occ_idx else np.nan
        non_occ_alpha = np.mean(alpha_power_mean[non_occ_idx]) if non_occ_idx else np.nan

        # Relative occipital alpha: occipital / (occipital + non-occipital)
        with np.errstate(divide='ignore', invalid='ignore'):
            relative_alpha = occ_alpha / (occ_alpha + non_occ_alpha)
            if np.isnan(relative_alpha):
                relative_alpha = 0

        relative_alpha_metrics[sub] = relative_alpha

    return relative_alpha_metrics

def get_trial_epochs(eeg_files=None, cond='', roi=None, target_events=None, event_id=None):
    from copy import deepcopy
    import numpy as np
    import mne

    eeg_files_copy = deepcopy(eeg_files)
    epochs_dict = {}

    for sub in subs:
        print(f"\n[CHECKPOINT] Processing {sub}...")
        raw = mne.concatenate_raws(eeg_files_copy[sub])
        raw_copy = deepcopy(raw)
        raw_copy.pick(roi)

        # Drop bad segments
        raw_clean = drop_bad_segments(sub, cond, raw_copy)

        # Prepare info and Raw object
        info = mne.create_info(ch_names=raw_copy.info['ch_names'], sfreq=raw.info['sfreq'], ch_types='eeg')
        eeg = mne.io.RawArray(raw_clean, info)

        # Filtering and referencing
        eeg.filter(l_freq=1, h_freq=30, method='fir', fir_design='firwin', phase='zero')
        if eeg.get_montage() is None:
            eeg.set_montage('standard_1020')
        eeg.set_eeg_reference('average')

        # Fetch events for this subject
        events = target_events[sub]
        event_id = event_id

        # Epoching: -0.2 to 0.9 s with baseline -0.2 to 0.0
        epochs = mne.Epochs(
            eeg, events=events, event_id=event_id,
            tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0), preload=True
        )

        epochs_dict[sub] = epochs

    print(f"\n[CHECKPOINT] All subjects processed for epochs.\n")
    return epochs_dict




if __name__ == '__main__':
    # --- Parameters ---
    subs = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19', 'sub20',
            'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29']

    plane = 'azimuth'
    if plane == 'azimuth':
        cond='a1'
    else:
        cond = 'e1'
    occipital_channels = ['O1', 'O2', 'Oz', 'PO3', 'PO4', 'PO7', 'PO8', 'POz','P1', 'P2', 'Pz', 'PO9', 'PO10']

    sfreq = 125
    epoch_length = sfreq * 60  # samples in 1 minute

    fmin, fmax = 1, 30

    # Define channel info for single-channel data
    default_path = Path.cwd()
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'
    json_path = default_path / 'data' / 'misc'
    with open(json_path / "electrode_names.json") as file:  # electrode_names
        mapping = json.load(file)

    # load and filter EEG data 1-3Hz
    l_freq = 7.0
    h_freq= 13.0
    eeg_files = get_eeg_files(condition=cond)

    # get events:
    target_events, distractor_events = get_events_dicts(folder_name1='stream1', folder_name2='stream2', cond=cond)

    target_trial_epochs = get_trial_epochs(eeg_files=eeg_files, cond=cond, roi='all', target_events=target_events, event_id={'4': np.int64(4)})
    distractor_trial_epochs = get_trial_epochs(eeg_files, cond, 'all', distractor_events, event_id={'3': np.int64(3)})
    # epoch EEG data from 1 min epochs, around stimuli onsets (target and distractor stream respectively)
    epochs_dict = get_epochs(eeg_files, cond=cond, roi='all')

    # compute ITC for each epoch type:
    alpha = np.logspace(np.log10(l_freq), np.log10(h_freq), num=60)


    # compute morlet TFR for each sub's epochs:
    alpha_power = compute_tfr(epochs_dict, alpha, n_cycles = alpha / 2)
    # Extract average power across time and frequency for each subject and channel
    # Store per-subject alpha power averaged over epochs, freqs, and time

    # Parameters
    freq_range = (8, 12)  # Alpha band
    time_range = (0.0, 60.0)  # Full duration
    subject_alphas = []
    subjects_sorted = sorted(alpha_power.keys())

    # Step 1: extract raw alpha power from occipital ROI
    for sub in subjects_sorted:
        tfr = alpha_power[sub]

        # Channel indices
        ch_idx = [tfr.ch_names.index(ch) for ch in occipital_channels if ch in tfr.ch_names]
        if not ch_idx:
            print(f"Warning: No occipital channels found for {sub}")
            subject_alphas.append(np.nan)
            continue

        # Freq and time indices
        fmin_idx = np.argmin(np.abs(tfr.freqs - freq_range[0]))
        fmax_idx = np.argmin(np.abs(tfr.freqs - freq_range[1])) + 1
        tmin_idx = np.argmin(np.abs(tfr.times - time_range[0]))
        tmax_idx = np.argmin(np.abs(tfr.times - time_range[1])) + 1

        # Extract and average
        roi_power = tfr.data[:, ch_idx, fmin_idx:fmax_idx, tmin_idx:tmax_idx]
        avg_alpha = roi_power.mean()
        subject_alphas.append(avg_alpha)
        print(f"{sub}: {avg_alpha:.4e}")

    # Convert to array and handle NaNs
    alpha_array = np.array(subject_alphas)
    valid_mask = ~np.isnan(alpha_array)
    valid_alpha = alpha_array[valid_mask]

    # === Choose ONE normalization method below ===

    ## Option 1: Normalize to max (0–1 scale)
    normalized_alpha = np.full_like(alpha_array, np.nan)
    normalized_alpha[valid_mask] = valid_alpha / np.max(valid_alpha)

    ## Option 2: Z-score normalization
    # normalized_alpha = np.full_like(alpha_array, np.nan)
    # normalized_alpha[valid_mask] = (valid_alpha - np.mean(valid_alpha)) / np.std(valid_alpha)

    # Subject labels
    subject_labels = [f"S{str(i + 1).zfill(2)}" for i in range(len(subjects_sorted))]

    # Plot
    plt.figure(figsize=(10, 5), dpi=100)
    bars = plt.bar(subject_labels, normalized_alpha, color='gray', edgecolor='black')

    mean_val = np.nanmean(normalized_alpha)
    plt.axhline(mean_val, linestyle='--', color='red', label=f'Mean = {mean_val:.2f}')

    plt.ylabel('Normalized Occipital Alpha Power (8–12 Hz)', fontsize=12, fontweight='bold')
    plt.xlabel('Subject', fontsize=12, fontweight='bold')
    plt.title('Normalized Alpha Power in Occipital ROI', fontsize=13, fontweight='bold')
    plt.xticks(rotation=45)

    plt.ylim(0, 1.1 if np.nanmax(normalized_alpha) <= 1 else np.nanmax(normalized_alpha) + 0.1)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

    # compute morlet tfr trial-wise:
    # trials_alpha_power_t = compute_tfr(target_trial_epochs, alpha, n_cycles = alpha / 2)
    # trials_alpha_power_d = compute_tfr(distractor_trial_epochs, alpha, n_cycles = alpha / 2)

    # === trial-wise metrics === #

    # Settings
    alpha_band = (8, 13)
    broad_band = (1, 30)
    itc_band = (4, 8)  # Theta
    itc_time_window = (0.0, 0.3)
    rms_time_window = (0.0, 0.896)
    theta_window = (0.0, 0.3)

    # def trial_wise_metrics(trial_epochs):
    #     # Storage
    #     trial_wise_metrics = {}
    #
    #     for sub, epochs in trial_epochs.items():
    #
    #         print(f"[INFO] Processing {sub}...")
    #
    #         # === Relative Alpha Power ===
    #         alpha_epochs = deepcopy(epochs)
    #         alpha_epochs = alpha_epochs.pick(occipital_channels)
    #         psd, freqs = alpha_epochs.compute_psd(method='welch', fmin=1, fmax=30).get_data(return_freqs=True)
    #         alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
    #         broad_mask = (freqs >= broad_band[0]) & (freqs <= broad_band[1])
    #
    #         # Mean across channels
    #         rel_alpha = psd[:, :, alpha_mask].mean(axis=2) / psd[:, :, broad_mask].mean(axis=2)
    #         mean_rel_alpha = rel_alpha.mean(axis=1)  # shape: (n_trials,)
    #
    #         # === RMS from 0.0s onward ===
    #         frontal_epochs = deepcopy(epochs)
    #         frontal_epochs = frontal_epochs.pick(frontal_roi)
    #         data = frontal_epochs.get_data()  # (n_trials, n_channels, n_times)
    #         times = frontal_epochs.times
    #         rms_mask = (times >= rms_time_window[0]) & (times <= rms_time_window[1])
    #         rms = np.sqrt((data[:, :, rms_mask] ** 2).mean(axis=(1, 2)))  # (n_trials,)
    #         theta_freqs = np.logspace(np.log10(1), np.log10(8), 100)
    #         # === Single-trial Theta Phase Consistency (0.0–0.3s) ===
    #         power = frontal_epochs.compute_tfr(method='multitaper', freqs=theta_freqs,
    #             n_cycles=0.5 * theta_freqs, use_fft=True, return_itc=False,
    #             average=False, decim=1, n_jobs=1, output='complex'
    #         )
    #         real = power.data[:, :, 0, :, :]  # shape: (n_epochs, n_channels, n_freqs, n_times)
    #         imag = power.data[:, :, 1, :, :]
    #         complex_data = real + 1j * imag
    #
    #         time_mask = (power.times >= theta_window[0]) & (power.times <= theta_window[1])
    #         # Make a proper frequency mask for theta range
    #         theta_mask = (power.freqs >= 1) & (power.freqs <= 8)
    #         phases = np.angle(complex_data[:, :, theta_mask, :][:, :, :, time_mask])
    #         # shape: (n_trials, n_channels, n_freqs, n_times)
    #
    #         # Compute vector length across time and freq, average over channels
    #         complex_vectors = np.exp(1j * phases)
    #         phase_consistency = np.abs(complex_vectors.mean(axis=(2, 3)))  # (n_epochs, n_channels)
    #         mean_phase_consistency = phase_consistency.mean(axis=1)  # (n_epochs,)
    #
    #         # Save
    #         trial_wise_metrics[sub] = {
    #             'alpha': mean_rel_alpha,
    #             'rms': rms,
    #             'itc_like': mean_phase_consistency
    #         }
    #
    #     print("[✓] Trial-wise metrics extracted and saved.")
    #     return trial_wise_metrics

    # def plot_phase(trial_wise_metrics):
    #     import pandas as pd
    #
    #     plt.figure(figsize=(12, 6))
    #
    #     window_size = 25  # increase smoothing
    #
    #     for sub, metrics in trial_wise_metrics.items():
    #         itc_vals = pd.Series(metrics['itc_like'])
    #         smoothed = itc_vals.rolling(window=window_size, min_periods=1, center=True).mean()
    #         plt.plot(smoothed, alpha=0.6, label=sub)
    #
    #     plt.xlabel('Trial')
    #     plt.ylabel('Phase Consistency (0–0.3s)')
    #     plt.title('Smoothed Trial-wise Phase Consistency per Subject')
    #     plt.grid(alpha=0.3)
    #     plt.legend(loc='upper right', fontsize='small', ncol=2)
    #     plt.tight_layout()
    #     plt.show()


    # target_trial_wise_metrics = trial_wise_metrics(target_trial_epochs)
    # distractor_trial_wise_metrics = trial_wise_metrics(distractor_trial_epochs)

    # # save trial-wise itcs:
    # itc_path = default_path / f'/data/eeg/behaviour/{plane}/{cond}/trial_wise_itc'
    # itc_path.mkdir(parents=True, exist_ok=True)
    # np.savez(itc_path/'target_trial_wise_metrics.npz', target_trial_wise_metrics)
    # np.savez(itc_path/'distractor_trial_wise_metrics.npz', distractor_trial_wise_metrics)
    #
    # plot_phase(target_trial_wise_metrics)
    # plot_phase(distractor_trial_wise_metrics)
    #
    # from scipy.stats import wilcoxon, ttest_1samp
    # import numpy as np
    #
    # sub_diffs = []
    # p_vals = []
    #
    # for sub in target_trial_wise_metrics:
    #     target = target_trial_wise_metrics[sub]['itc_like']
    #     distractor = distractor_trial_wise_metrics[sub]['itc_like']
    #
    #     # Truncate to same length
    #     min_len = min(len(target), len(distractor))
    #     t = target[:min_len]
    #     d = distractor[:min_len]
    #
    #     # Compute per-subject difference
    #     diff = t - d
    #     sub_diffs.append(np.mean(diff))
    #
    #     # Paired test per subject (choose either)
    #     try:
    #         stat, p = wilcoxon(t, d)
    #     except ValueError:
    #         stat, p = np.nan, np.nan
    #     p_vals.append(p)
    #
    # # Summary across subjects
    # sub_diffs = np.array(sub_diffs)
    # group_stat, group_p = ttest_1samp(sub_diffs, popmean=0)
    #
    # print(f"==== SUBJECT-WISE STATS ====")
    # for i, (diff, p) in enumerate(zip(sub_diffs, p_vals)):
    #     print(f"sub{i + 1:02d}: Δ = {diff:.4f}, p = {p:.4f}")
    #
    # print("\n==== GROUP LEVEL ====")
    # print(f"Mean Δ (Target - Distractor): {np.mean(sub_diffs):.4f}")
    # print(f"1-sample t-test against 0: t = {group_stat:.2f}, p = {group_p:.4f}")
    #
    # # Cohen's d for within-subject (paired samples)
    # cohen_d = np.mean(sub_diffs) / np.std(sub_diffs, ddof=1)
    # print(f"Cohen's d: {cohen_d:.3f}")


    # get power mean and absolute power (?)
    # alpha_metrics = extract_alpha_metrics(alpha_power)


    # Compute relative alpha power across subjects
    relative_alpha_metrics = compute_relative_alpha_power(epochs_dict, occipital_channels, alpha_band=(8, 12), total_band=(1, 30))


    # # Sort by subject ID (optional)
    # sorted_items = sorted(relative_alpha_metrics.items())
    # subjects = [k for k, _ in sorted_items]
    # values = [v for _, v in sorted_items]
    #
    # # Plot
    # plt.figure(figsize=(10, 6))
    # bars = plt.bar(subjects, values, color='skyblue')
    # plt.axhline(y=sum(values) / len(values), color='red', linestyle='--', label='Mean Alpha Power')
    #
    # # Annotate values
    # for bar in bars:
    #     yval = bar.get_height()
    #     plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    #
    # plt.title(f"{plane.capitalize()} \nRelative Alpha Power by Subject", fontsize=12, fontweight='bold')
    # plt.xlabel("Subject ID")
    # plt.ylabel('Relative Alpha Power (8–12 Hz / 1–30 Hz)', fontsize=10, fontweight='bold')
    # plt.ylim(0, max(values) + 0.1)
    # plt.grid(axis='y', alpha=0.3)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # # plt.savefig(fig_path/'relative_alpha_distribution.png', dpi=300)
    #
    # for sub in alpha_metrics:
    #     if sub in relative_alpha_metrics:
    #         alpha_metrics[sub]['relative_alpha'] = relative_alpha_metrics[sub]
    #     else:
    #         print(f"[WARNING] No relative alpha data found for {sub}")
    #
    #
    # # save data
    # default_path = Path.cwd()
    # save_dir = default_path / f'data/eeg/alpha/{plane}/{cond}'
    # save_dir.mkdir(parents=True, exist_ok=True)
    # np.savez(save_dir / f'alpha_metrics{cond}.npz', **alpha_metrics)