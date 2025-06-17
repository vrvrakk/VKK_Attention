# === TFA on predicted EEG === #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
import os
from pathlib import Path
import mne
import seaborn as sns
from matplotlib import cm
import pandas as pd
import scipy.stats
from copy import deepcopy
from scipy.stats import sem,  zscore, ttest_rel,  wilcoxon, shapiro
from scipy.signal import windows
from TRF_test.TRF_test_config import frontal_roi
from statsmodels.stats.multitest import fdrcorrection
from mne.time_frequency import tfr_multitaper
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
from scipy.signal import savgol_filter
from scipy.fft import fft, fftfreq
from matplotlib import rcParams
from mne.time_frequency import AverageTFR
from mne.filter import filter_data


# === Load relevant events and mask the bad segments === #

def get_pred_dicts(cond):
    predictions_dir = default_path / f'data/eeg/trf/trf_testing/composite_model/{plane}/{cond}/{folder_type}/data'
    sub_pred_dict = {}
    for pred_files in predictions_dir.iterdir():
        if f'{plane}_{folder_type}_both_streams_TRF_results.npz' in pred_files.name:
            predictions = np.load(predictions_dir / pred_files.name)
            predictions = predictions['results']
            sub = pred_files.name[:5]
            sub_pred_dict[sub] = predictions
    return sub_pred_dict # 18 subjects, shape (n_samples, ) -> averaged across channels


# --- Helper: FFT Power Extraction ---
def compute_zscored_power(evoked, sfreq, fmin=1, fmax=30):
    data = evoked.data.squeeze(axis=0) # mean across channels (already ROI)
    hann = windows.hann(len(data))
    windowed = data * hann
    fft = np.fft.rfft(windowed)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(data), d=1 / sfreq)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], zscore(power[mask])


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


def get_residual_eegs(mne_events, sub_preds_dict=None, eeg_files=None, cond=''):

    eeg_files_copy = deepcopy(eeg_files)
    epochs_dict = {}
    eeg_residuals_dict = {}

    for sub in subs:
        print(f"\n[CHECKPOINT] Processing {sub}...")

        eeg_predicted = sub_preds_dict[sub]
        print(f"[CHECKPOINT] {sub} prediction shape: {eeg_predicted.shape}")

        raw = mne.concatenate_raws(eeg_files_copy[sub])
        raw_copy = deepcopy(raw)
        # raw_copy.pick(frontal_roi)
        print(f"[CHECKPOINT] {sub} prediction x eeg copy shape: {eeg_predicted.shape} x {raw_copy._data.shape}")

        # Drop bad segments
        eeg_predicted = eeg_predicted.T
        raw_clean = drop_bad_segments(sub, cond1, raw_copy)
        print(f"[CHECKPOINT] {sub} prediction x eeg copy shape: {eeg_predicted.shape} x {raw_clean.shape}")

        # --- Event Filtering ---
        events = target_events1[sub]
        print(f"[CHECKPOINT] {sub} events loaded: {len(events)}")

        sfreq = raw.info['sfreq']
        n_samples = raw.n_times
        bad_time_mask = np.zeros(n_samples, dtype=bool)

        info = mne.create_info(ch_names=raw_copy.info['ch_names'], sfreq=raw_copy.info['sfreq'], ch_types='eeg')

        # Subtract prediction from EEG to get residual
        raw_clean = zscore(raw_clean)
        eeg_residual = raw_clean - eeg_predicted
        # Assuming eeg_residual shape: (n_times,)
        eeg = mne.io.RawArray(eeg_residual, info)
        if eeg.get_montage() is None:
            eeg.set_montage('standard_1020')
        eeg.filter(l_freq=1, h_freq=30, method='fir', fir_design='firwin', phase='zero')
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
        tmax = 0.5
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
        epochs = mne.Epochs(eeg, events=valid_events.astype(int), event_id=event_id,
                            tmin=tmin, tmax=tmax, baseline=(tmin, -0.3), preload=True)

        print(f"[CHECKPOINT] {sub} epochs shape: {epochs.get_data().shape}")
        epochs_dict[sub] = epochs
        eeg_residuals_dict[sub] = eeg

    return epochs_dict, eeg_residuals_dict


def compute_itc(epochs_dict, freqs, n_cycles):
    itcs = {}
    powers = {}
    for sub, epochs in epochs_dict.items():
        tfr = epochs.compute_tfr(
            method="morlet",
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=True,
            average=True,  # must be True for ITC
            decim=1,
            n_jobs=1)
        itcs[sub] = tfr[1]  # index 1 is ITC (index 0 is power)
        powers[sub] = tfr[0]
    return itcs, powers


def cohens_d_paired(x, y):
    diff = x - y
    return diff.mean() / diff.std(ddof=1)


# --- Collect z-scored power for all subjects ---
def z_scored_power(target_epochs_dict, distractor_epochs_dict):
    target_power = []
    distractor_power = []

    for sub in subs:
        targ_evoked = target_epochs_dict[sub].average()
        dist_evoked = distractor_epochs_dict[sub].average()

        power_freqs_t, targ_pow = compute_zscored_power(targ_evoked, sfreq, fmin, fmax)
        power_freqs_d, dist_pow = compute_zscored_power(dist_evoked, sfreq, fmin, fmax)

        target_power.append(targ_pow)
        distractor_power.append(dist_pow)

    target_power = np.array(target_power)
    distractor_power = np.array(distractor_power)
    return target_power, distractor_power, power_freqs_t, power_freqs_d

# --- Paired Wilcoxon per frequency ---
def paired_wilcoxon(target_dict, distractor_dict, cond='', resp='', folder_type='all_stims', plane=''):
    subjects = list(target_dict.keys())
    assert subjects == list(distractor_dict.keys()), "Subject lists must match"

    # Extract freqs and power for each subject
    power_list_t = []
    power_list_d = []

    for sub in subjects:
        tfr_t: AverageTFR = target_dict[sub]
        tfr_d: AverageTFR = distractor_dict[sub]

        # Time range of interest: -0.3 to 0.0 (pre-stim)
        time_mask_t = (tfr_t.times >= -0.3) & (tfr_t.times <= 0.0)
        time_mask_d = (tfr_d.times >= -0.3) & (tfr_d.times <= 0.0)

        # Average over time and channel (single-channel assumed)
        power_t = tfr_t.data[:, :, time_mask_t].mean(axis=2).squeeze()
        power_d = tfr_d.data[:, :, time_mask_d].mean(axis=2).squeeze()

        power_list_t.append(power_t)
        power_list_d.append(power_d)

    # Convert to arrays: shape (n_subjects, n_freqs)
    target_power = np.stack(power_list_t)
    distractor_power = np.stack(power_list_d)

    power_freqs_t = tfr_t.freqs
    power_freqs_d = tfr_d.freqs

    # Frequency range mask (1–10 Hz)
    mask_t = (power_freqs_t >= 1) & (power_freqs_t <= 10)
    mask_d = (power_freqs_d >= 1) & (power_freqs_d <= 10)

    target_power = target_power[:, mask_t]
    distractor_power = distractor_power[:, mask_d]
    power_freqs_t = power_freqs_t[mask_t]
    power_freqs_d = power_freqs_d[mask_d]

    # Wilcoxon test across subjects at each frequency
    wilcoxon_p = []
    for i in range(target_power.shape[1]):
        try:
            _, p = wilcoxon(target_power[:, i], distractor_power[:, i])
        except ValueError:
            p = 1.0
        wilcoxon_p.append(p)

    wilcoxon_p = np.array(wilcoxon_p)
    _, p_fdr = fdrcorrection(wilcoxon_p)

    # Plotting
    plt.figure(figsize=(7, 4), dpi=300, constrained_layout=True)
    rcParams.update({
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
    })
    ax = plt.gca()

    # Mean & SEM
    target_mean = target_power.mean(axis=0)
    distractor_mean = distractor_power.mean(axis=0)
    target_sem = target_power.std(axis=0) / np.sqrt(target_power.shape[0])
    distractor_sem = distractor_power.std(axis=0) / np.sqrt(distractor_power.shape[0])

    # Plot curves
    ax.plot(power_freqs_t, target_mean, label='Target', color='blue', linewidth=0.5)
    ax.fill_between(power_freqs_t, target_mean - target_sem, target_mean + target_sem,
                    color='blue', alpha=0.2)

    ax.plot(power_freqs_d, distractor_mean, label='Distractor', color='red', linewidth=0.5)
    ax.fill_between(power_freqs_d, distractor_mean - distractor_sem, distractor_mean + distractor_sem,
                    color='red', alpha=0.2)

    # Significant freqs
    sig_freqs = power_freqs_t[p_fdr < 0.05]
    for freq in sig_freqs:
        ax.axvline(x=freq, color='gray', linestyle='--', linewidth=0.2, alpha=0.5)
    ax.plot([], [], color='gray', linestyle='--', linewidth=0.2, label='p < 0.05 (FDR)')

    # Inset plot (1.0–1.4 Hz zoom)
    inset_ax = inset_axes(ax, width="20%", height="20%", loc='upper center', borderpad=1)
    inset_ax.plot(power_freqs_t, target_mean, color='blue', linewidth=0.5)
    inset_ax.fill_between(power_freqs_t, target_mean - target_sem, target_mean + target_sem,
                          color='blue', alpha=0.2)
    inset_ax.plot(power_freqs_d, distractor_mean, color='red', linewidth=0.5)
    inset_ax.fill_between(power_freqs_d, distractor_mean - distractor_sem, distractor_mean + distractor_sem,
                          color='red', alpha=0.2)
    inset_ax.set_xlim(1.0, 1.4)
    ymin = min(target_mean[(power_freqs_t >= 1.0) & (power_freqs_t <= 1.4)].min(),
               distractor_mean[(power_freqs_d >= 1.0) & (power_freqs_d <= 1.4)].min()) - 0.1
    ymax = max(target_mean[(power_freqs_t >= 1.0) & (power_freqs_t <= 1.4)].max(),
               distractor_mean[(power_freqs_d >= 1.0) & (power_freqs_d <= 1.4)].max()) + 0.1
    inset_ax.set_ylim(ymin, ymax)
    inset_ax.set_xticklabels([])
    inset_ax.set_yticklabels([])
    inset_ax.tick_params(length=0)
    inset_ax.grid(True, linestyle='--', linewidth=0.2, alpha=0.5)
    mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="gray", lw=0.2, alpha=0.3)

    # Annotate peaks
    peak_idx_t = np.argmax(target_mean)
    peak_freq_t = power_freqs_t[peak_idx_t]
    peak_power_t = target_mean[peak_idx_t]
    ax.text(peak_freq_t, peak_power_t + 1.5,
            f'Target Peak:\n{peak_freq_t:.2f} Hz, {peak_power_t:.2f}',
            fontsize=4, color='blue', ha='left', va='center', fontweight='bold')

    peak_idx_d = np.argmax(distractor_mean)
    peak_freq_d = power_freqs_d[peak_idx_d]
    peak_power_d = distractor_mean[peak_idx_d]
    ax.text(peak_freq_d, peak_power_d + 1.5,
            f'Distractor Peak:\n{peak_freq_d:.2f} Hz, {peak_power_d:.2f}',
            fontsize=4, color='red', ha='left', va='center', fontweight='bold')

    # Final touches
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Z-scored Power')
    ax.set_title('Power Spectrum (Target vs Distractor) in ROI')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.legend(loc='upper right')
    plt.tight_layout()

    # Save plot
    save_dir = Path(
        default_path / f'data/eeg/trf/trf_testing/composite_model/single_sub/figures/{plane}/{cond}/{folder_type}')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'target_x_distractor_{resp}_fft.png', dpi=300)
    plt.show()


def peak_freq_paired_test(target_dict, distractor_dict):
    subjects = list(target_dict.keys())
    assert subjects == list(distractor_dict.keys()), "Subject keys do not match."

    target_peaks = []
    distractor_peaks = []
    target_power_list = []
    distractor_power_list = []

    for sub in subjects:
        tfr_t = target_dict[sub]
        tfr_d = distractor_dict[sub]

        # Pre-stimulus time mask (-0.3 to 0.0s)
        time_mask_t = (tfr_t.times >= -0.3) & (tfr_t.times <= 0.0)
        time_mask_d = (tfr_d.times >= -0.3) & (tfr_d.times <= 0.0)

        power_t = tfr_t.data[:, :, time_mask_t].mean(axis=2).squeeze()  # shape: (n_freqs,)
        power_d = tfr_d.data[:, :, time_mask_d].mean(axis=2).squeeze()

        target_power_list.append(power_t)
        distractor_power_list.append(power_d)

        peak_freq_t = tfr_t.freqs[np.argmax(power_t)]
        peak_freq_d = tfr_d.freqs[np.argmax(power_d)]

        target_peaks.append(peak_freq_t)
        distractor_peaks.append(peak_freq_d)


    target_power = np.stack(target_power_list)  # shape (n_subjects, n_freqs)
    distractor_power = np.stack(distractor_power_list)

    target_peak_freqs = np.array(target_peaks)
    distractor_peak_freqs = np.array(distractor_peaks)
    print("Target peak freqs:", target_peak_freqs)
    print("Distractor peak freqs:", distractor_peak_freqs)

    # Paired Wilcoxon test on peak frequencies
    stat, p_peak = wilcoxon(target_peak_freqs, distractor_peak_freqs)
    print(f"Wilcoxon test on peak frequencies: p = {p_peak:.4f}")

    # Rank-biserial correlation
    diffs = target_peak_freqs - distractor_peak_freqs
    n_positive = np.sum(diffs > 0)
    n_negative = np.sum(diffs < 0)
    rbc = (n_positive - n_negative) / len(diffs)
    print(f"Rank-biserial correlation: r = {rbc:.3f}")

    # Normality tests for power distributions per frequency
    target_normality_p = []
    distractor_normality_p = []
    for i in range(target_power.shape[1]):
        try:
            _, p_targ = shapiro(target_power[:, i])
        except Exception:
            p_targ = 1.0
        try:
            _, p_dist = shapiro(distractor_power[:, i])
        except Exception:
            p_dist = 1.0
        target_normality_p.append(p_targ)
        distractor_normality_p.append(p_dist)

    target_normality_p = np.array(target_normality_p)
    distractor_normality_p = np.array(distractor_normality_p)

    alpha = 0.05
    n_non_normal_target = np.sum(target_normality_p < alpha)
    n_non_normal_distractor = np.sum(distractor_normality_p < alpha)

    print(f"Target: {n_non_normal_target} / {len(target_normality_p)} freqs fail normality")
    print(f"Distractor: {n_non_normal_distractor} / {len(distractor_normality_p)} freqs fail normality")

    return (
        target_peak_freqs,
        distractor_peak_freqs,
        target_normality_p,
        distractor_normality_p,
        rbc
    )


def prepare_containers(target_powers, distractor_powers):
    target_power_vals = []
    distractor_power_vals = []

    subjects = list(target_powers.keys())
    assert subjects == list(distractor_powers.keys()), "Subject keys do not match"

    for sub in subjects:
        # Get power: mean over channel and time for each frequency
        targ_pow = target_powers[sub].data.mean(axis=(0, 2))  # shape: (n_freqs,)
        dist_pow = distractor_powers[sub].data.mean(axis=(0, 2))
        target_power_vals.append(targ_pow)
        distractor_power_vals.append(dist_pow)

    target_power_vals = np.array(target_power_vals)      # shape: (n_subjects, n_freqs)
    distractor_power_vals = np.array(distractor_power_vals)

    # Use freq axis from any subject (they are the same)
    freqs = target_powers[subjects[0]].freqs

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, target_power_vals.mean(axis=0), label='Target', color='blue')
    plt.fill_between(freqs,
                     target_power_vals.mean(0) - sem(target_power_vals, axis=0),
                     target_power_vals.mean(0) + sem(target_power_vals, axis=0),
                     alpha=0.3, color='blue')

    plt.plot(freqs, distractor_power_vals.mean(axis=0), label='Distractor', color='red')
    plt.fill_between(freqs,
                     distractor_power_vals.mean(0) - sem(distractor_power_vals, axis=0),
                     distractor_power_vals.mean(0) + sem(distractor_power_vals, axis=0),
                     alpha=0.3, color='red')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Z-scored Power')
    plt.title('Average Power Spectrum (Target vs Distractor)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()


def itc_vals(target_itc, distractor_itc, band=None, band_name='', cond='', resp=''):
    target_vals = []
    distractor_vals = []

    for sub in subs:
        targ_data = target_itc[sub].data.mean(axis=(0, 2))
        dist_data = distractor_itc[sub].data.mean(axis=(0, 2))

        target_vals.append(targ_data)
        distractor_vals.append(dist_data)

    target_vals = np.array(target_vals)
    distractor_vals = np.array(distractor_vals)

    # Normality checks (optional use)
    n_freqs = target_vals.shape[1]
    for f in range(n_freqs):
        _ = shapiro(target_vals[:, f])
        _ = shapiro(distractor_vals[:, f])

    # Paired t-test and FDR
    t_vals, p_vals = ttest_rel(target_vals, distractor_vals, axis=0)
    _, p_fdr = fdrcorrection(p_vals)

    # Effect size (Cohen's d)
    effect_sizes = np.array([
        cohens_d_paired(target_vals[:, i], distractor_vals[:, i])
        for i in range(n_freqs)
    ])

    # Mean diff and SEM
    mean_diff = target_vals.mean(axis=0) - distractor_vals.mean(axis=0)
    sem_diff = (target_vals - distractor_vals).std(axis=0, ddof=1) / np.sqrt(target_vals.shape[0])

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(7, 4), dpi=300)

    # ITC difference
    ax.plot(band, mean_diff, label='Target - Distractor', color='indigo', linewidth=1)
    ax.fill_between(band, mean_diff - sem_diff, mean_diff + sem_diff, color='indigo', alpha=0.2)

    # Mark significance
    sig_mask = p_fdr < 0.05
    important_freq = band[np.argmax(np.abs(effect_sizes))]
    for freq, y in zip(band[sig_mask], mean_diff[sig_mask]):
        ax.text(freq, y + 0.005, '*', fontsize=6, ha='center', va='bottom', color='black')

    # Optional: Cohen's d as dashed gray line
    ax.plot(band, effect_sizes / 5, linestyle='--', color='gray', linewidth=0.8, label="Cohen's d (scaled)")

    ax.axhline(0, color='black', linestyle='--', linewidth=0.5)
    ax.set_xlabel('Frequency (Hz)', fontsize=6)
    ax.set_ylabel('ITC Difference\n(Target − Distractor)', fontsize=6)
    ax.set_title('ITC Difference Across Frequencies', fontsize=8)
    ax.tick_params(labelsize=6)
    ax.legend(fontsize=6, loc='upper right')
    ax.grid(True, linestyle='--', linewidth=0.2, alpha=0.3)

    fig.tight_layout()
    plt.show()
    save_dir = Path(default_path/ f'data/eeg/trf/trf_testing/composite_model/single_sub/figures/{plane}/{cond}/{folder_type}')
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'itc_diff_{band_name}_{resp}.png', dpi=300)

    return target_vals, distractor_vals, effect_sizes, p_fdr, important_freq
# === control fft of envelopes === #

def fft_envelopes(data, sfreq, fmin=1, fmax=30):
    """
    Compute FFT power of a 1D signal (like an envelope).
    """
    if data.ndim > 1:
        data = data.squeeze()

    hann = windows.hann(len(data))
    windowed = data * hann
    fft_vals = np.fft.rfft(windowed)
    power = np.abs(fft_vals) ** 2
    freqs = np.fft.rfftfreq(len(data), d=1 / sfreq)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], zscore(power[mask])


def get_env_fft(cond, stream_type, sfreq=125, fmin=1, fmax=30):
    envelope_power = []
    freqs_used = None

    for sub in subs:
        env_path = Path(
            default_path / f'data/eeg/predictors/envelopes/{sub}/{cond}/{stream_type}/{sub}_{cond}_{stream_type}_envelopes_series_concat.npz')
        if not env_path.exists():
            print(f"Missing file for {sub}")
            continue

        env = np.load(env_path)
        env_array = env['envelopes']  # shape: (n_samples,)

        freqs, env_pow = fft_envelopes(env_array, sfreq, fmin, fmax)
        envelope_power.append(env_pow)
        freqs_used = freqs  # consistent since we truncate later

    # Truncate to common min length if needed (in case fft returned slightly different lengths)
    min_len = min(arr.shape[0] for arr in envelope_power)
    envelope_power = np.array([arr[:min_len] for arr in envelope_power])
    freqs_used = freqs_used[:min_len]

    return envelope_power, freqs_used

def plot_env_vs_predicted(env_power_list, eeg_power_matrix, freqs_env, freqs_eeg,
                          stream_label, cond, resp='', plane='default_plane', folder_type=''):
    """
    Plot the average FFT spectra of the predicted EEG and stimulus envelope.

    Parameters:
    - env_power_list: list or array, shape (n_subjects, n_env_freqs), envelope power per subject
    - eeg_power_matrix: dict of mne.EpochsTFR objects, one per subject
    - freqs_env: array-like, frequencies corresponding to envelope power
    - freqs_eeg: array-like, target EEG frequency bins (e.g., 100 values)
    - stream_label: str, e.g., 'stream1' or 'stream2'
    - cond: str, experimental condition label
    - resp: optional str, tag for filename
    - plane: optional str, for subfolder naming
    - folder_type: optional str, for subfolder naming
    """
    from scipy.stats import sem
    from scipy.interpolate import interp1d
    import numpy as np
    import matplotlib.pyplot as plt
    from pathlib import Path

    save_dir = Path(f'data/eeg/trf/trf_testing/composite_model/single_sub/figures/{plane}/{cond}/{folder_type}')
    save_dir.mkdir(parents=True, exist_ok=True)

    # ----- Process EEG data -----
    eeg_power_list = [p.data.mean(axis=2).squeeze() for p in eeg_power_matrix.values()]  # shape: (n_subjects, n_freqs)
    eeg_power_array = np.array(eeg_power_list)
    eeg_power_mean = eeg_power_array.mean(axis=0)

    # ----- Process envelope data -----
    env_power_array = np.array(env_power_list)  # shape: (n_subjects, n_env_freqs)
    env_power_mean = env_power_array.mean(axis=0)
    env_power_sem = env_power_array.std(axis=0, ddof=1) / np.sqrt(env_power_array.shape[0])

    # ----- Interpolate envelope spectrum to EEG frequency resolution -----
    interp_env = interp1d(freqs_env, env_power_mean, kind='linear', bounds_error=False, fill_value='extrapolate')
    interp_sem = interp1d(freqs_env, env_power_sem, kind='linear', bounds_error=False, fill_value='extrapolate')
    env_interp = interp_env(freqs_eeg)
    sem_interp = interp_sem(freqs_eeg)

    # ----- Peak detection -----
    env_peak_idx = np.argmax(env_interp)
    eeg_peak_idx = np.argmax(eeg_power_mean)

    # ----- Plotting -----
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_eeg, eeg_power_mean, label='Predicted EEG', color='blue')
    plt.plot(freqs_eeg, env_interp, label='Envelope', color='green')
    plt.fill_between(freqs_eeg, env_interp - sem_interp, env_interp + sem_interp, alpha=0.3, color='green')

    # Mark peaks
    plt.scatter(freqs_eeg[env_peak_idx], env_interp[env_peak_idx], color='darkgreen', zorder=5)
    plt.text(freqs_eeg[env_peak_idx] + 0.1, env_interp[env_peak_idx] + 0.05,
             f"Env Peak = {freqs_eeg[env_peak_idx]:.2f} Hz\nPower = {env_interp[env_peak_idx]:.2f}",
             fontsize=9, color='darkgreen')

    plt.scatter(freqs_eeg[eeg_peak_idx], eeg_power_mean[eeg_peak_idx], color='darkblue', zorder=5)
    plt.text(freqs_eeg[eeg_peak_idx] + 0.1, eeg_power_mean[eeg_peak_idx] + 0.05,
             f"EEG Peak = {freqs_eeg[eeg_peak_idx]:.2f} Hz\nPower = {eeg_power_mean[eeg_peak_idx]:.2f}",
             fontsize=9, color='darkblue')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Z-scored Power')
    plt.title(f'{stream_label.capitalize()} — Envelope vs Predicted EEG FFT ({cond.upper()})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_dir / f'env_vs_predicted_fft_{resp}_{stream_label}.png', dpi=300)
    plt.show()



def sub_level_itc(target_vals, distractor_vals, band=None, label_subjects=False, axline=None, band_name='', cond='', resp=''):
    save_dir = Path(default_path / f'data/eeg/trf/trf_testing/composite_model/single_sub/figures/{plane}/{cond}/{folder_type}')
    save_dir.mkdir(parents=True, exist_ok=True)
    sns.set(style='whitegrid', context=None)
    plt.rcParams.update({
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 8,
    })
    num_subs = target_vals.shape[0]
    colors = plt.colormaps['tab20'](np.linspace(0, 1, num_subs))


    # --- Target Stream Plot ---
    plt.figure(figsize=(8, 4), dpi=300)
    for i, sub_itc in enumerate(target_vals):
        smoothed = savgol_filter(sub_itc, window_length=5, polyorder=2)
        plt.plot(band, smoothed, color=colors[i], linewidth=1)
        if label_subjects:
            peak_idx = np.argmax(sub_itc)
            plt.text(band[peak_idx], sub_itc[peak_idx], f'{i + 1}', fontsize=6, color=colors[i])
    mean_itc = savgol_filter(target_vals.mean(axis=0), window_length=5, polyorder=2)
    plt.plot(band, mean_itc, color='black', linewidth=2.5, label='Mean ITC')
    plt.axvline(x=axline, color='red', linestyle='--', label=f'~{axline:.2f} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('ITC (Target Stream)')
    plt.title('Subject-Level ITC Curves (Target Stream)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_dir / f'subs_ITC_{resp}_{band_name}_target.png', dpi=300)

    # --- Distractor Stream Plot ---
    plt.figure(figsize=(8, 4), dpi=300)
    for i, sub_itc in enumerate(distractor_vals):
        smoothed = savgol_filter(sub_itc, window_length=5, polyorder=2)
        plt.plot(band, smoothed, color=colors[i], linewidth=1)
        if label_subjects:
            peak_idx = np.argmax(sub_itc)
            plt.text(band[peak_idx], sub_itc[peak_idx], f'{i + 1}', fontsize=6, color=colors[i])
    mean_itc = savgol_filter(distractor_vals.mean(axis=0), window_length=5, polyorder=2)
    plt.plot(band, mean_itc, color='black', linewidth=2.5, label='Mean ITC')
    plt.axvline(x=axline, color='red', linestyle='--', label=f'~{axline:.2f} Hz')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('ITC (Distractor Stream)')
    plt.title('Subject-Level ITC Curves (Distractor Stream)')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_dir / f'subs_ITC_{resp}_{band_name}_distractor.png', dpi=300)

    # --- Violin Plot at ~4 Hz ---
    freq_of_interest = axline
    f_idx = np.argmin(np.abs(band - freq_of_interest))

    target_hz = target_vals[:, f_idx]
    distractor_hz = distractor_vals[:, f_idx]

    # Paired t-test
    t_stat, p_val = ttest_rel(target_hz, distractor_hz)

    df = pd.DataFrame({
        'Target': target_hz,
        'Distractor': distractor_hz
    }).melt(var_name='Stream', value_name='ITC')

    plt.figure(figsize=(6, 5), dpi=300)
    sns.violinplot(data=df, x='Stream', y='ITC', inner='point', palette={'Target': 'blue', 'Distractor': 'red'})
    plt.title(f'ITC at ~{band[f_idx]:.2f} Hz (p = {p_val:.3e})', fontsize=14)
    plt.ylabel('ITC', fontsize=12)
    plt.xlabel('')
    plt.grid(True, linestyle='--', alpha=0.3)

    # Optional statistical annotation
    if p_val < 0.001:
        stat_text = '*** p < 0.001'
    elif p_val < 0.01:
        stat_text = '** p < 0.01'
    elif p_val < 0.05:
        stat_text = '* p < 0.05'
    else:
        stat_text = f'n.s. (p = {p_val:.2f})'

    plt.text(0.5, max(df['ITC']) + 0.02, stat_text, ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.show()
    plt.savefig(save_dir / f'ITC_{resp}_{band_name}_violinplot.png', dpi=300)


def compare_amplitude_at_1hz(target_powers, distractor_powers):
    subjects = list(target_powers.keys())
    assert subjects == list(distractor_powers.keys()), "Subject keys must match"
    power_at_1hz_target = []
    power_at_1hz_distractor = []
    for sub in subjects:
        tfr_t = target_powers[sub]
        tfr_d = distractor_powers[sub]
        # Pre-stimulus time avg (-0.3 to 0.0s)
        time_mask_t = (tfr_t.times >= -0.3) & (tfr_t.times <= 0.0)
        time_mask_d = (tfr_d.times >= -0.3) & (tfr_d.times <= 0.0)
        # Power averaged over channel and time, then select closest freq to 1 Hz
        mean_pow_t = tfr_t.data[:, :, time_mask_t].mean(axis=(0, 2))  # shape: (n_freqs,)
        mean_pow_d = tfr_d.data[:, :, time_mask_d].mean(axis=(0, 2))
        freqs = tfr_t.freqs
        idx_1hz = np.argmin(np.abs(freqs - 1.0))
        power_at_1hz_target.append(mean_pow_t[idx_1hz])
        power_at_1hz_distractor.append(mean_pow_d[idx_1hz])
    power_at_1hz_target = np.array(power_at_1hz_target)
    power_at_1hz_distractor = np.array(power_at_1hz_distractor)
    # Wilcoxon test
    stat, p = wilcoxon(power_at_1hz_target, power_at_1hz_distractor)
    print(f"\nPower @ 1 Hz — Wilcoxon test: p = {p:.4f}")
    # Rank-biserial correlation
    diffs = power_at_1hz_target - power_at_1hz_distractor
    n_pos = np.sum(diffs > 0)
    n_neg = np.sum(diffs < 0)
    rbc = (n_pos - n_neg) / len(diffs)
    print(f"Rank-biserial correlation: r = {rbc:.3f}")
    return power_at_1hz_target, power_at_1hz_distractor, p, rbc

if __name__ == '__main__':
    # --- Parameters ---
    pred_types = ['onsets', 'envelopes']
    predictor_short = "_".join([p[:2] for p in pred_types])

    subs = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19', 'sub20',
            'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29']

    plane='elevation'

    if plane == 'azimuth':
        cond1 = 'a2'
    if plane == 'elevation':
        cond1 = 'e2'

    folder_types = ['all_stims']
    folder_type = folder_types[0]
    sfreq = 125
    epoch_length = sfreq * 60  # samples in 1 minute

    fmin, fmax = 1, 30

    # Define channel info for single-channel data
    ch_name = 'avg'  # or 'target' / 'distractor'
    ch_type = 'eeg'  # use 'misc' for predicted, non-EEG data
    default_path = Path.cwd()
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'


    sub_preds_dict = get_pred_dicts(cond=cond1)

    subs = list(sub_preds_dict.keys())

    eeg_files1 = get_eeg_files(condition=cond1)

    target_events1, distractor_events1 = get_events_dicts(folder_name1='stream1', folder_name2='stream2', cond=cond1)

    target_epochs_induced1 = get_residual_eegs(target_events1, sub_preds_dict=sub_preds_dict, eeg_files=eeg_files1, cond=cond1)

    distractor_epochs_induced1 = get_residual_eegs(distractor_events1, sub_pred_dict=sub_preds_dict, eeg_files=eeg_files1, cond=cond1)

    # Parameters
    itc_freqs = {'delta/theta': np.logspace(np.log10(1), np.log10(8), num=100)}

    target_power1, distractor_power1, power_freqs_t1, power_freqs_d1 = z_scored_power(target_epochs_induced1,
                                                                                      distractor_epochs_induced1)

    target_itc1_low, target_powers1_low = compute_itc(target_epochs_induced1, itc_freqs['delta/theta'], n_cycles=0.5 * itc_freqs['delta/theta'])

    distractor_itc1_low, distractor_powers1_low = compute_itc(distractor_epochs_induced1, itc_freqs['delta/theta'], n_cycles=0.5 * itc_freqs['delta/theta'])

    target_vals1_low, distractor_vals1_low, effect_sizes1_low, p_fdr1_low, peak_freq_low1 = itc_vals(target_itc1_low,
                                                                                                     distractor_itc1_low,
                                                                                                     band=itc_freqs['delta/theta'],
                                                                                                     band_name = 'delta_theta', cond=cond1)



    sub_level_itc(target_vals1_low, distractor_vals1_low, band=itc_freqs['delta/theta'], label_subjects=True, axline=peak_freq_low1, band_name='delta_theta', cond=cond1)


    envelope_power_target1, env_freqs_target1 = get_env_fft(cond=cond1, stream_type='stream1', fmin=1, fmax=8)

    plot_env_vs_predicted(envelope_power_target1, target_powers1_low, env_freqs_target1,  target_itc1_low['sub10'].freqs,  # This gives the full (100,) freq vector from EEG,
                          stream_label='target', cond=cond1, resp='ind', plane=plane, folder_type=folder_type)


    envelope_power_distractor1, env_freqs_distractor1 = get_env_fft(cond=cond1, stream_type='stream2')

    plot_env_vs_predicted(envelope_power_distractor1, distractor_powers1_low, env_freqs_distractor1, distractor_itc1_low['sub10'].freqs,
                          # This gives the full (100,) freq vector from EEG,
                          stream_label='distractor', cond=cond1, resp='ind', plane=plane, folder_type=folder_type)

    ##

    paired_wilcoxon(target_powers1_low, distractor_powers1_low, cond=cond1, resp='ind', folder_type='all_stims', plane=plane)

    peak_freq_paired_test(target_powers1_low, distractor_powers1_low)

    prepare_containers(target_powers1_low, distractor_powers1_low)

    target_amps, distractor_amps, p, r = compare_amplitude_at_1hz(target_powers1_low, distractor_powers1_low)
