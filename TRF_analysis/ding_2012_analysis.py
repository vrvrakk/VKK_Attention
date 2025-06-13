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
    return target_preds_dict, distractor_preds_dict # 18 subjects, shape (n_samples, ) -> averaged across channels



# Function to create mne.EpochsArray for each subject
def make_epochs(preds_dict, sfreq, epoch_length, ch_name='predicted', ch_type='misc'):
    epochs_dict = {}
    info = mne.create_info(ch_names=[ch_name], sfreq=sfreq, ch_types=[ch_type])

    for sub, data in preds_dict.items():
        n_epochs = data.shape[0] // epoch_length
        trimmed = data[:n_epochs * epoch_length]
        reshaped = trimmed.reshape(n_epochs, 1, epoch_length)  # shape: (n_epochs, n_channels=1, n_times)
        epochs = mne.EpochsArray(reshaped, info)
        epochs_dict[sub] = epochs

    return epochs_dict

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


def get_residual_eegs(preds_dict=None, eeg_files=None, cond=''):

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
        eeg_residual = raw_clean - eeg_predicted

        n_epochs = eeg_residual.shape[0] // epoch_length
        trimmed = eeg_residual[:n_epochs * epoch_length]
        reshaped = trimmed.reshape(n_epochs, 1, epoch_length)  # shape: (n_epochs, n_channels=1, n_times)
        epochs = mne.EpochsArray(reshaped, info)
        epochs_dict[sub] = epochs

    print(f"\n[CHECKPOINT] All subjects processed for residual epochs.\n")
    return epochs_dict


def compute_itc(epochs_dict, freqs, n_cycles):
    itcs = {}
    powers = {}
    for sub, epochs in epochs_dict.items():
        tfr = epochs.compute_tfr(
            method="multitaper",
            freqs=freqs,
            n_cycles=n_cycles,
            use_fft=True,
            return_itc=True,
            average=True,  # must be True for ITC
            decim=1,
            n_jobs=1
        )
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
        targ_evoked = target_epochs_dict[sub].average(picks='all')
        dist_evoked = distractor_epochs_dict[sub].average(picks='all')

        power_freqs_t, targ_pow = compute_zscored_power(targ_evoked, sfreq, fmin, fmax)
        power_freqs_d, dist_pow = compute_zscored_power(dist_evoked, sfreq, fmin, fmax)

        target_power.append(targ_pow)
        distractor_power.append(dist_pow)

    target_power = np.array(target_power)
    distractor_power = np.array(distractor_power)
    return target_power, distractor_power, power_freqs_t, power_freqs_d

# --- Paired Wilcoxon per frequency ---
def paired_wilcoxon(target_power, distractor_power, power_freqs_t, power_freqs_d):
    wilcoxon_p = []
    mask_t = (power_freqs_t >= 1) & (power_freqs_t <= 10)
    power_freqs_t = power_freqs_t1[mask_t]
    target_power = target_power[:, mask_t]

    mask_d = (power_freqs_d >= 1) & (power_freqs_d <= 10)
    power_freqs_d = power_freqs_d1[mask_d]
    distractor_power = distractor_power[:, mask_d]

    for i in range(target_power.shape[1]):
        try:
            _, p = wilcoxon(target_power[:, i], distractor_power[:, i])
        except ValueError:
            p = 1.0
        wilcoxon_p.append(p)

    wilcoxon_p = np.array(wilcoxon_p)
    _, p_fdr = fdrcorrection(wilcoxon_p)

    # --- Plot Settings ---
    plt.figure(figsize=(7, 4), dpi=300, constrained_layout=True)
    plt.rcParams.update({
        'axes.titlesize': 8,
        'axes.labelsize': 7,
        'xtick.labelsize': 7,
        'ytick.labelsize': 7,
        'legend.fontsize': 7,
    })

    # Mean and SEM
    target_mean = target_power.mean(axis=0)
    distractor_mean = distractor_power.mean(axis=0)
    target_sem = target_power.std(axis=0) / np.sqrt(target_power.shape[0])
    distractor_sem = distractor_power.std(axis=0) / np.sqrt(distractor_power.shape[0])

    # --- Create main plot ---
    ax = plt.gca()

    # Plot main curves
    ax.plot(power_freqs_t, target_mean, label='Target', color='blue', linewidth=0.5)
    ax.fill_between(power_freqs_t, target_mean - target_sem, target_mean + target_sem,
                    color='blue', alpha=0.2)

    ax.plot(power_freqs_d, distractor_mean, label='Distractor', color='red', linewidth=0.5)
    ax.fill_between(power_freqs_d, distractor_mean - distractor_sem, distractor_mean + distractor_sem,
                    color='red', alpha=0.2)

    # --- Mark significant frequencies ---
    sig_mask = p_fdr < 0.05
    sig_freqs = power_freqs_t[sig_mask]
    for freq in sig_freqs:
        ax.axvline(x=freq, color='gray', linestyle='--', linewidth=0.2, alpha=0.5)

    # Legend marker for significance
    ax.plot([], [], color='gray', linestyle='--', linewidth=0.2, label='p < 0.05 (FDR)')

    # --- Inset: Zoom into 1–1.4 Hz ---
    inset_ax = inset_axes(ax, width="20%", height="20%", loc='upper center', borderpad=1)

    inset_ax.plot(power_freqs_t, target_mean, color='blue', linewidth=0.5)
    inset_ax.fill_between(power_freqs_t, target_mean - target_sem, target_mean + target_sem,
                          color='blue', alpha=0.2)
    inset_ax.plot(power_freqs_d, distractor_mean, color='red', linewidth=0.5)
    inset_ax.fill_between(power_freqs_d, distractor_mean - distractor_sem, distractor_mean + distractor_sem,
                          color='red', alpha=0.2)

    inset_ax.set_xlim(1.0, 1.4)
    inset_ax.set_ylim(
        min(target_mean[(power_freqs_t >= 1.0) & (power_freqs_t <= 1.4)].min(),
            distractor_mean[(power_freqs_d >= 1.0) & (power_freqs_d <= 1.4)].min()) - 0.1,
        max(target_mean[(power_freqs_t >= 1.0) & (power_freqs_t <= 1.4)].max(),
            distractor_mean[(power_freqs_d >= 1.0) & (power_freqs_d <= 1.4)].max()) + 0.1
    )
    # Remove axis labels/ticks
    inset_ax.set_xticklabels([])
    inset_ax.set_yticklabels([])
    inset_ax.tick_params(length=0)
    inset_ax.grid(True, linestyle='--', linewidth=0.2, alpha=0.5)
    mark_inset(ax, inset_ax, loc1=2, loc2=4, fc="none", ec="gray", lw=0.2, alpha=0.3)

    # --- Final labels and layout ---
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Z-scored Power')
    ax.set_title('Power Spectrum (Target vs Distractor) in ROI')
    ax.grid(True, linestyle='--', alpha=0.3, linewidth=0.5)
    ax.legend(loc='upper right')
    # --- Annotate Target Peak ---
    peak_idx_t = np.argmax(target_mean)
    peak_freq_t = power_freqs_t[peak_idx_t]
    peak_power_t = target_mean[peak_idx_t]
    ax.text(peak_freq_t, peak_power_t + 1.5,
            f'Target Peak:\n{peak_freq_t:.2f} Hz, {peak_power_t:.2f}',
            fontsize=4, color='blue', ha='left', va='center', fontweight='bold')

    # --- Annotate Distractor Peak ---
    peak_idx_d = np.argmax(distractor_mean)
    peak_freq_d = power_freqs_d[peak_idx_d]
    peak_power_d = distractor_mean[peak_idx_d]
    ax.text(peak_freq_d, peak_power_d + 1.5,
            f'Distractor Peak:\n{peak_freq_d:.2f} Hz, {peak_power_d:.2f}',
            fontsize=4, color='red', ha='left', va='center', fontweight='bold')
    plt.tight_layout()
    plt.show()

def peak_freq_paired_test(power_freqs_t, power_freqs_d, target_power, distractor_power):
    target_peak_freqs = power_freqs_t[np.argmax(target_power, axis=1)]
    distractor_peak_freqs = power_freqs_d[np.argmax(distractor_power, axis=1)]

    # Optional: paired test on peak frequencies
    stat, p_peak = wilcoxon(target_peak_freqs, distractor_peak_freqs)
    print(f"Wilcoxon test on peak frequencies: p = {p_peak:.4f}")

    # Compute rank-biserial correlation
    diffs = target_peak_freqs - distractor_peak_freqs
    n_positive = np.sum(diffs > 0)
    n_negative = np.sum(diffs < 0)

    rbc = (n_positive - n_negative) / len(diffs)
    print(f"Rank-biserial correlation: r = {rbc:.3f}")

    # Run normality test per frequency
    target_normality_p = []
    distractor_normality_p = []

    for i in range(target_power.shape[1]):  # loop over frequency bins
        _, p_targ = shapiro(target_power[:, i])
        _, p_dist = shapiro(distractor_power[:, i])
        target_normality_p.append(p_targ)
        distractor_normality_p.append(p_dist)

    target_normality_p = np.array(target_normality_p)
    distractor_normality_p = np.array(distractor_normality_p)

    # Threshold for normality
    alpha = 0.05
    # Count how many frequencies reject normality
    n_non_normal_target = np.sum(target_normality_p < alpha)
    n_non_normal_distractor = np.sum(distractor_normality_p < alpha)
    print(f"Target: {n_non_normal_target} / {len(target_normality_p)} bins fail normality")
    print(f"Distractor: {n_non_normal_distractor} / {len(distractor_normality_p)} bins fail normality")
    return target_peak_freqs, distractor_peak_freqs, target_normality_p, distractor_normality_p, rbc

    # Prepare containers

def prepare_containers(target_powers, distractor_powers):
    target_power_vals = []
    distractor_power_vals = []

    for sub in target_powers:
        # shape: (n_channels, n_freqs, n_times)
        targ_pow = target_powers[sub].data.mean(axis=(0, 2))  # mean over channels and time
        dist_pow = distractor_powers[sub].data.mean(axis=(0, 2))
        target_power_vals.append(targ_pow)
        distractor_power_vals.append(dist_pow)

    target_power_vals = np.array(target_power_vals)  # shape: (n_subjects, n_freqs)
    distractor_power_vals = np.array(distractor_power_vals)
    freqs_t = target_powers[sub].freqs  # get frequency axis from one subject
    freqs_d = distractor_powers[sub].freqs

    plt.figure(figsize=(10, 5))
    plt.plot(freqs_t, target_power_vals.mean(0), label='Target', color='blue')
    plt.fill_between(freqs_t,
                     target_power_vals.mean(0) - sem(target_power_vals, axis=0),
                     target_power_vals.mean(0) + sem(target_power_vals, axis=0),
                     alpha=0.3, color='blue')

    plt.plot(freqs_d, distractor_power_vals.mean(0), label='Distractor', color='red')
    plt.fill_between(freqs_d,
                     distractor_power_vals.mean(0) - sem(distractor_power_vals, axis=0),
                     distractor_power_vals.mean(0) + sem(distractor_power_vals, axis=0),
                     alpha=0.3, color='red')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power')
    plt.title('Average Power Spectrum (Target vs Distractor)')
    plt.legend()
    plt.tight_layout()
    plt.show()

def itc_vals(target_itc, distractor_itc, band=None):
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

    return target_vals, distractor_vals, effect_sizes, p_fdr, important_freq
# === control fft of envelopes === #

def fft_envelopes(evoked, sfreq, fmin=1, fmax=30):
    data = evoked.data  # mean across channels (already ROI)
    hann = windows.hann(len(data))
    windowed = data * hann
    fft = np.fft.rfft(windowed)
    power = np.abs(fft) ** 2
    freqs = np.fft.rfftfreq(len(data), d=1 / sfreq)
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], zscore(power[mask])

    # Loop through subjects

def get_env_fft(cond, stream_type):
    # Store power spectra
    envelope_power = []
    for sub in subs:
        env_path = Path(
            default_path / f'data/eeg/predictors/envelopes/{sub}/{cond}/{stream_type}/{sub}_{cond}_{stream_type}_envelopes_series_concat.npz')
        if not env_path.exists():
            print(f"Missing file for {sub}")
            continue

        env = np.load(env_path)
        env_array = env['envelopes']  # shape: (n_samples,)

        env_freqs, env_power = fft_envelopes(env_array, sfreq, fmin, fmax)

        envelope_power.append(env_power)

    # Convert to array
    # Step 1: Find minimum common length
    min_len = min(arr.shape[0] for arr in envelope_power)

    # Step 2: Truncate and stack
    envelope_power_trunc = np.array([arr[:min_len] for arr in envelope_power])
    return envelope_power_trunc, env_freqs

def plot_env_vs_predicted(env_power_list, eeg_power_matrix, freqs_env, freqs_eeg, stream_label, cond):
    # Compute means and SEM
    mean_env = np.mean(env_power_list, axis=0)
    sem_env = sem(env_power_list, axis=0)
    mean_eeg = np.mean(eeg_power_matrix, axis=0)

    # Match frequency length
    min_len = min(len(mean_env), len(freqs_eeg), len(freqs_env))
    freqs_env = freqs_env[:min_len]
    freqs_eeg = freqs_eeg[:min_len]
    mean_env = mean_env[:min_len]
    sem_env = sem_env[:min_len]
    mean_eeg = mean_eeg[:min_len]

    # Find envelope peak
    env_peak_idx = np.argmax(mean_env)
    env_peak_freq = freqs_env[env_peak_idx]
    env_peak_power = mean_env[env_peak_idx]

    # Find EEG predicted peak
    eeg_peak_idx = np.argmax(mean_eeg)
    eeg_peak_freq = freqs_eeg[eeg_peak_idx]
    eeg_peak_power = mean_eeg[eeg_peak_idx]

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(freqs_eeg, mean_eeg, label='Predicted EEG', color='blue')
    plt.plot(freqs_env, mean_env, label='Envelope', color='green')
    plt.fill_between(freqs_env,
                     mean_env - sem_env,
                     mean_env + sem_env,
                     alpha=0.3, color='green')

    # Annotate peaks
    plt.scatter(env_peak_freq, env_peak_power, color='darkgreen', marker='o', zorder=5)
    plt.text(env_peak_freq + 0.1, env_peak_power + 0.05,
             f"Env Peak = {env_peak_freq:.2f} Hz\nPower = {env_peak_power:.2f}",
             fontsize=9, color='darkgreen')

    plt.scatter(eeg_peak_freq, eeg_peak_power, color='darkblue', marker='o', zorder=5)
    plt.text(eeg_peak_freq + 0.1, eeg_peak_power + 0.05,
             f"EEG Peak = {eeg_peak_freq:.2f} Hz\nPower = {eeg_peak_power:.2f}",
             fontsize=9, color='darkblue')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Z-scored Power')
    plt.title(f'{stream_label.capitalize()} — Envelope vs Predicted EEG FFT ({cond.upper()})')
    plt.legend()
    plt.tight_layout()
    plt.grid(True)
    plt.show()


def sub_level_itc(target_vals, distractor_vals, band=None, label_subjects=False, axline=None):
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


if __name__ == '__main__':
    # --- Parameters ---
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
    epoch_length = sfreq * 60  # samples in 1 minute

    fmin, fmax = 1, 30

    # Define channel info for single-channel data
    ch_name = 'predicted'  # or 'target' / 'distractor'
    ch_type = 'avg'  # use 'misc' for predicted, non-EEG data
    default_path = Path.cwd()
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'


    target_preds_dict1, distractor_preds_dict1 = get_pred_dicts(cond=cond1)

    # Create target and distractor epoch objects
    target_epochs_dict1 = make_epochs(target_preds_dict1, sfreq, epoch_length, ch_name='target_pred')
    distractor_epochs_dict1 = make_epochs(distractor_preds_dict1, sfreq, epoch_length, ch_name='distractor_pred')


    subs = list(target_preds_dict1.keys())

    target_power1, distractor_power1, power_freqs_t1, power_freqs_d1 = z_scored_power(target_epochs_dict1, distractor_epochs_dict1)

    paired_wilcoxon(target_power1, distractor_power1, power_freqs_t1, power_freqs_d1)

    target_peak_freqs1, distractor_peak_freqs1, target_normality_p1, distractor_normality_p1, rbc1 = peak_freq_paired_test(power_freqs_t1, power_freqs_d1, target_power1, distractor_power1)

    eeg_files1 = get_eeg_files(condition=cond1)

    target_epochs_induced1 = get_residual_eegs(preds_dict=target_preds_dict1, eeg_files=eeg_files1, cond=cond1)

    distractor_epochs_induced1 = get_residual_eegs(preds_dict=distractor_preds_dict1, eeg_files=eeg_files1, cond=cond1)

    # Parameters
    itc_freqs = {'delta/theta': np.linspace(1, 7, num=60),
                 'alpha': np.linspace(7, 13, num=60),
                 'beta': np.linspace(13, 30, num=60)}


    target_itc1_low, target_powers1_low = compute_itc(target_epochs_induced1, itc_freqs['delta/theta'], n_cycles=2 * itc_freqs['delta/theta'])
    distractor_itc1_low, distractor_powers1_low = compute_itc(distractor_epochs_induced1, itc_freqs['delta/theta'], n_cycles=2 * itc_freqs['delta/theta'])
    target_vals1_low, distractor_vals1_low, effect_sizes1_low, p_fdr1_low, peak_freq_low1 = itc_vals(target_itc1_low, distractor_itc1_low, band=itc_freqs['delta/theta'])

    # ITC on residuals: alpha/beta bands
    target_itc1_alpha, target_powers1_alpha = compute_itc(target_epochs_induced1, itc_freqs['alpha'], n_cycles=2 * itc_freqs['alpha'])
    distractor_itc1_alpha, distractor_powers1_alpha = compute_itc(distractor_epochs_induced1, itc_freqs['alpha'], n_cycles=2 * itc_freqs['alpha'])
    target_vals1_alpha, distractor_vals1_alpha, effect_sizes1_alpha, p_fdr1_alpha, peak_freq_alpha1 = itc_vals(target_itc1_alpha, distractor_itc1_alpha, band=itc_freqs['alpha'])
    
    # ITC on residuals: beta/beta bands
    target_itc1_beta, target_powers1_beta = compute_itc(target_epochs_induced1, itc_freqs['beta'], n_cycles=2 * itc_freqs['beta'])
    distractor_itc1_beta, distractor_powers1_beta = compute_itc(distractor_epochs_induced1, itc_freqs['beta'], n_cycles=2 * itc_freqs['beta'])
    target_vals1_beta, distractor_vals1_beta, effect_sizes1_beta, p_fdr1_beta, peak_freq_beta1 = itc_vals(target_itc1_beta, distractor_itc1_beta, band=itc_freqs['beta'])


    sub_level_itc(target_vals1_low, distractor_vals1_low, band=itc_freqs['delta/theta'], label_subjects=True, axline=peak_freq_low1)

    sub_level_itc(target_vals1_alpha, distractor_vals1_alpha, band=itc_freqs['alpha'], label_subjects=True, axline=peak_freq_alpha1)

    sub_level_itc(target_vals1_beta, distractor_vals1_beta, band=itc_freqs['beta'], label_subjects=True, axline=peak_freq_beta1)
    

    envelope_power_target1, env_freqs_target1 = get_env_fft(cond=cond1, stream_type='stream1')

    plot_env_vs_predicted(envelope_power_target1, target_power1, env_freqs_target1, power_freqs_t1, stream_label='target', cond=cond1)


    envelope_power_distractor1, env_freqs_distractor1 = get_env_fft(cond=cond1, stream_type='stream2')

    plot_env_vs_predicted(envelope_power_distractor1, distractor_power1, env_freqs_distractor1, power_freqs_d1, stream_label='distractor', cond=cond1)


    # cond2

    plt.close('all')

    target_preds_dict2, distractor_preds_dict2 = get_pred_dicts(cond=cond2)

    # Create target and distractor epoch objects
    target_epochs_dict2 = make_epochs(target_preds_dict2, sfreq, epoch_length, ch_name='target_pred')

    distractor_epochs_dict2 = make_epochs(distractor_preds_dict2, sfreq, epoch_length, ch_name='distractor_pred')

    target_power2, distractor_power2, power_freqs_t2, power_freqs_d2 = z_scored_power(target_epochs_dict2, distractor_epochs_dict2)

    paired_wilcoxon(target_power2, distractor_power2, power_freqs_t2, power_freqs_d2)

    target_peak_freqs2, distractor_peak_freqs2, target_normality_p2, distractor_normality_p2, rbc2 = peak_freq_paired_test(power_freqs_t2, power_freqs_d2, target_power2, distractor_power2)

    eeg_files2 = get_eeg_files(condition=cond2)

    target_epochs_induced2 = get_residual_eegs(preds_dict=target_preds_dict2, eeg_files=eeg_files2, cond=cond2)

    distractor_epochs_induced2 = get_residual_eegs(preds_dict=distractor_preds_dict2, eeg_files=eeg_files2, cond=cond2)

    target_itc2_low, target_powers2_low = compute_itc(target_epochs_induced1, itc_freqs['delta/theta'], n_cycles=2 * itc_freqs['delta/theta'])
    distractor_itc2_low, distractor_powers2_low = compute_itc(distractor_epochs_induced1, itc_freqs['delta/theta'], n_cycles=2 * itc_freqs['delta/theta'])
    target_vals2_low, distractor_vals2_low, effect_sizes2_low, p_fdr2_low = itc_vals(target_itc2_low, distractor_itc2_low, band=itc_freqs['delta/theta'])
    

    # ITC on residuals: alpha/beta bands
    target_itc2_alpha, target_powers2_alpha = compute_itc(target_epochs_induced1, itc_freqs['alpha'], n_cycles=2 * itc_freqs['alpha'])
    distractor_itc2_alpha, distractor_powers2_alpha = compute_itc(distractor_epochs_induced1, itc_freqs['alpha'], n_cycles=2 * itc_freqs['alpha'])
    target_vals2_alpha, distractor_vals2_alpha, effect_sizes2_alpha, p_fdr2_alpha = itc_vals(target_itc2_alpha, distractor_itc2_alpha, band=itc_freqs['alpha'])

    # ITC on residuals: beta/beta bands
    target_itc2_beta, target_powers2_beta = compute_itc(target_epochs_induced1, itc_freqs['beta'], n_cycles=2 * itc_freqs['beta'])
    distractor_itc2_beta, distractor_powers2_beta = compute_itc(distractor_epochs_induced1, itc_freqs['beta'], n_cycles=2 * itc_freqs['beta'])
    target_vals2_beta, distractor_vals2_beta, effect_sizes2_beta, p_fdr2_beta = itc_vals(target_itc2_beta, distractor_itc2_beta, band=itc_freqs['beta'])

    sub_level_itc(target_vals2_low, distractor_vals2_low, band=itc_freqs['delta/theta'], label_subjects=True,
                  axline=6.0)

    sub_level_itc(target_vals2_alpha, distractor_vals2_alpha, band=itc_freqs['alpha'], label_subjects=True, axline=13.5)

    sub_level_itc(target_vals2_alpha, distractor_vals2_alpha, band=itc_freqs['beta'], label_subjects=True, axline=13.5)
    envelope_power_target2, env_freqs_target2 = get_env_fft(cond=cond2, stream_type='stream2')

    plot_env_vs_predicted(envelope_power_target2, target_power2, env_freqs_target2, power_freqs_t2, stream_label='target', cond=cond2)

    envelope_power_distractor2, env_freqs_distractor2 = get_env_fft(cond=cond2, stream_type='stream1')


    plot_env_vs_predicted(envelope_power_distractor2, distractor_power2, env_freqs_distractor2, power_freqs_d2, stream_label='distractor', cond=cond2)
