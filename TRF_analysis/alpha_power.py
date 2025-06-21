
# === TFA on predicted EEG === #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

from EEG.merge_conditions import fig_path

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

def extract_alpha_metrics():
    alpha_metrics = {}
    for sub, power in alpha_power.items():
        power = np.array(power)  # Ensure it's a NumPy array
        power_mean = np.mean(power)
        power_rms = np.sqrt(np.mean(power ** 2))

        alpha_metrics[sub] = {
            'mean': power_mean,
            'RMS': power_rms
        }
    return alpha_metrics

def compute_relative_alpha_power(epochs_dict, alpha_band=(8, 12), total_band=(1, 30)):
    relative_alpha_metrics = {}

    for sub, epochs in epochs_dict.items():
        # Compute PSD using Welch via MNE's API
        psd = epochs.compute_psd(method='welch', fmin=total_band[0], fmax=total_band[1], n_fft=1024)
        psds = psd.get_data()  # shape: (n_epochs, n_channels, n_freqs)
        freqs = psd.freqs

        # Get frequency band masks
        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
        total_power = np.sum(psds, axis=-1)  # shape: (n_epochs, n_channels)
        alpha_power = np.sum(psds[:, :, alpha_mask], axis=-1)  # shape: (n_epochs, n_channels)

        # Relative power: alpha / total, per epoch & channel
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_alpha = np.true_divide(alpha_power, total_power)
            rel_alpha[np.isnan(rel_alpha)] = 0

        # Mean across epochs and ROI channels
        rel_alpha_mean = np.mean(rel_alpha, axis=0)  # mean across epochs → (n_channels,)
        rel_alpha_roi_mean = np.mean(rel_alpha_mean)

        relative_alpha_metrics[sub] = rel_alpha_roi_mean

    return relative_alpha_metrics


def compute_occipital_alpha_lateralization(epochs_dict, alpha_band=(8, 12), roi_left=('O1', 'PO7', 'PO9'), roi_right=('O2', 'PO8', 'PO10')):
    """
    Compute alpha lateralization index for each subject using occipital ROIs.
    ALI = (Right - Left) / (Right + Left)
    """
    lateralization_metrics = {}

    for sub, epochs in epochs_dict.items():
        picks_left = mne.pick_channels(epochs.info['ch_names'], roi_left)
        picks_right = mne.pick_channels(epochs.info['ch_names'], roi_right)

        if len(picks_left) == 0 or len(picks_right) == 0:
            print(f"[WARNING] Missing ROI channels for {sub}, skipping.")
            continue

        psd = epochs.compute_psd(method='welch', fmin=alpha_band[0], fmax=alpha_band[1], n_fft=1024)
        psds = psd.get_data()  # shape: (n_epochs, n_channels, n_freqs)
        freqs = psd.freqs

        alpha_mask = (freqs >= alpha_band[0]) & (freqs <= alpha_band[1])
        alpha_power = np.sum(psds[:, :, alpha_mask], axis=-1)  # shape: (n_epochs, n_channels)

        left_power = np.mean(alpha_power[:, picks_left])
        right_power = np.mean(alpha_power[:, picks_right])

        # Compute ALI (right - left) / total
        if (left_power + right_power) != 0:
            ali = (right_power - left_power) / (right_power + left_power)
        else:
            ali = np.nan

        lateralization_metrics[sub] = ali

    return lateralization_metrics


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
    # These groupings are speculative and depend on cap layout
    dorsal_occipital = ['POz', 'Pz', 'P1', 'P2']
    ventral_occipital = ['Oz', 'O1', 'O2']

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


    # epoch EEG data from 1 min epochs, around stimuli onsets (target and distractor stream respectively)
    epochs_dict = get_epochs(eeg_files, cond=cond, roi=occipital_channels)

    # compute ITC for each epoch type:
    alpha = np.logspace(np.log10(l_freq), np.log10(h_freq), num=100)


    # compute morlet TFR for each sub's epochs:

    alpha_power = compute_tfr(epochs_dict, alpha, n_cycles = alpha / 2)

    # get power mean and absolute power (?)

    alpha_metrics = extract_alpha_metrics()


    # Compute relative alpha power across subjects
    relative_alpha_metrics = compute_relative_alpha_power(epochs_dict)

    # Sort by subject ID (optional)
    sorted_items = sorted(relative_alpha_metrics.items())
    subjects = [k for k, _ in sorted_items]
    values = [v for _, v in sorted_items]

    # Plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(subjects, values, color='skyblue')
    plt.axhline(y=sum(values) / len(values), color='red', linestyle='--', label='Mean Alpha Power')

    # Annotate values
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f'{yval:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    plt.title(f"{plane.capitalize()} \nRelative Alpha Power by Subject", fontsize=12, fontweight='bold')
    plt.xlabel("Subject ID")
    plt.ylabel('Relative Alpha Power (8–12 Hz / 1–30 Hz)', fontsize=10, fontweight='bold')
    plt.ylim(0, max(values) + 0.1)
    plt.grid(axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(fig_path/'relative_alpha_distribution.png', dpi=300)

    for sub in alpha_metrics:
        if sub in relative_alpha_metrics:
            alpha_metrics[sub]['relative_alpha'] = relative_alpha_metrics[sub]
        else:
            print(f"[WARNING] No relative alpha data found for {sub}")

    # Compute lateralization (occipital alpha asymmetry)
    # if plane == 'azimuth':
    lateralization_metrics = compute_occipital_alpha_lateralization(epochs_dict)
    # elif plane == 'elevation':
    #     lateralization_metrics = compute_occipital_alpha_lateralization(epochs_dict, roi_left = dorsal_occipital, roi_right=ventral_occipital)

    # Add to each subject's alpha_metrics
    for sub in alpha_metrics:
        if sub in lateralization_metrics:
            alpha_metrics[sub]['alpha_lateralization'] = lateralization_metrics[sub]
        else:
            alpha_metrics[sub]['alpha_lateralization'] = np.nan
            print(f"[INFO] No ALI computed for {sub}")

    # save data
    default_path = Path.cwd()
    save_dir = default_path / f'data/eeg/alpha/{plane}/{cond}'
    save_dir.mkdir(parents=True, exist_ok=True)
    np.savez(save_dir / f'alpha_metrics{cond}.npz', **alpha_metrics)

    # Step 1: Concatenate all epochs across all subjects
    all_epochs = mne.concatenate_epochs(list(epochs_dict.values()))

    # Step 2: Define frequencies for TFR and compute
    freqs = np.linspace(8, 12, 40)  # Alpha band, high resolution
    n_cycles = freqs / 2.
    power = all_epochs.compute_tfr(
        method='morlet', freqs=freqs, n_cycles=n_cycles,
        use_fft=True, return_itc=False, average=True
    )

    # Step 3: Select and average alpha power across frequency band
    alpha_band = (8, 12)
    freq_mask = (power.freqs >= alpha_band[0]) & (power.freqs <= alpha_band[1])
    alpha_power = power.data[:, freq_mask, :].mean(axis=1)  # shape: (n_channels, n_times)

    # Step 4: Define ROIs
    roi_left = ['O1', 'PO7', 'PO3', 'PO9']
    roi_right = ['O2', 'PO8', 'PO4', 'PO10']

    # Get channel indices for ROIs
    ch_names = power.info['ch_names']
    idx_left = mne.pick_channels(ch_names, roi_left)
    idx_right = mne.pick_channels(ch_names, roi_right)

    # Step 5: Compute mean alpha power over ROIs
    left_avg = alpha_power[idx_left, :].mean(axis=0)
    right_avg = alpha_power[idx_right, :].mean(axis=0)
    ali = (right_avg - left_avg) / (right_avg + left_avg + 1e-10)  # Avoid division by zero

    # Step 6: Plot time courses
    times = power.times


    from scipy.signal import savgol_filter


    # Smooth the time series
    window_length = 501 if len(times) > 501 else len(times) // 2 * 2 + 1
    polyorder = 3
    smooth = lambda x: savgol_filter(x, window_length, polyorder)
    left_smooth = smooth(left_avg)
    right_smooth = smooth(right_avg)
    ali_smooth = smooth(ali)

    # --- Plot labels depending on plane ---
    # if plane == 'azimuth':
    left_label = 'Left ROI'
    right_label = 'Right ROI'
    ali_label = 'ALI = (Right - Left) / Total'
    ali_title = 'Alpha Lateralization Over Time'
    # else:  # elevation
    #     left_label = 'Dorsal ROI'
    #     right_label = 'Ventral ROI'
    #     ali_label = 'Alpha Index = (Ventral - Dorsal) / Total'
    #     ali_title = 'Dorsal–Ventral Alpha Dynamics'

    # --- Plot Settings ---
    plt.figure(figsize=(7, 4), dpi=300)
    plt.rcParams.update({
        'font.size': 6,
        'axes.spines.top': False,
        'axes.spines.right': False
    })

    # Subplot 1: Alpha Power
    plt.subplot(1, 2, 1)
    plt.plot(times, left_smooth, label=left_label, color='royalblue', linewidth=1.5)
    plt.plot(times, right_smooth, label=right_label, color='firebrick', linewidth=1.5)
    plt.xlabel('Time (s)')
    plt.ylabel('Alpha Power (8–12 Hz, a.u.)')
    plt.title('Alpha Power Over Time', fontsize=6, weight='bold')
    plt.legend(frameon=False, loc='upper right')
    plt.grid(alpha=0.3)

    # Subplot 2: Alpha Lateralization / Index
    plt.subplot(1, 2, 2)
    plt.plot(times, ali_smooth, label=ali_label, color='purple', linewidth=1.5)
    plt.axhline(0, color='gray', linestyle='--', linewidth=1)
    plt.xlabel('Time (s)')
    plt.ylabel('Alpha Asymmetry Index')
    plt.title(ali_title, fontsize=6, weight='bold')
    plt.grid(alpha=0.3)

    plt.suptitle(f'Occipital Alpha Dynamics Across Epochs\n{plane.capitalize()} Plane', fontsize=10, weight='bold',
                 y=1.05)
    plt.tight_layout()
    plt.show()
    fig_path = Path(default_path/f'data/eeg/trf/trf_testing/results/single_sub/alpha/{plane}/{cond}')
    fig_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path / 'alpha_power.png', dpi=300)