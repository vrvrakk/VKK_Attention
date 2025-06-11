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


# === Load relevant events and mask the bad segments === #

def get_pred_dicts(cond):
    predictions_dir = fr'C:/Users/pppar/PycharmProjects/VKK_Attention/data/eeg/trf/trf_testing/composite_model/single_sub/debug/{plane}/{cond}/{folder_type}/{predictor_short}/weights/predictions'
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
                        concat_files = [f for f in stim_folders.iterdir() if 'concat' in f.name]
                        if not concat_files:
                            continue  # skip if no relevant file

                        file = np.load(concat_files[0], allow_pickle=True)
                        stream_data = file['onsets']

                        stream = stream_data.copy()

                        # Keep only onset value for each event
                        i = 0
                        while i < len(stream):
                            if stream[i] in [1, 2, 3, 4]:
                                print(i)
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


def drop_bad_segments(raw):
    """
    Returns a copy of raw with all BAD segments removed.
    """
    bad_annots = [a for a in raw.annotations if 'bad' in a['description'].lower()]
    bad_intervals = [(a['onset'], a['onset'] + a['duration']) for a in bad_annots]

    # Sort and merge overlapping intervals (just in case)
    bad_intervals.sort()
    merged = []
    for start, end in bad_intervals:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)

    # Compute good segments (in seconds)
    good_intervals = []
    last_end = 0.0
    for start, end in merged:
        if start > last_end:
            good_intervals.append((last_end, start))
        last_end = end
    if last_end < raw.times[-1]:
        good_intervals.append((last_end, raw.times[-1]))

    # Crop and collect clean segments
    cleaned_raws = []
    min_duration = 1.0 / raw.info['sfreq']  # at least one sample long
    for start, end in good_intervals:
        if (end - start) >= min_duration:
            cleaned_raws.append(raw.copy().crop(tmin=start, tmax=end, include_tmax=False))
        else:
            print(f"Skipping too-short segment: {start:.3f}–{end:.3f} s")

    # Concatenate clean parts
    if cleaned_raws:
        return mne.concatenate_raws(cleaned_raws)
    else:
        raise RuntimeError("No clean data segments left after removing bad annotations.")


def get_residual_eegs(preds_dict=None, eeg_files=None, mne_events=None):
    eeg_files_copy = deepcopy(eeg_files)
    epochs_dict = {}

    for sub in subs:
        print(f"\n[CHECKPOINT] Processing {sub}...")

        eeg_predicted = preds_dict[sub]
        print(f"[CHECKPOINT] {sub} prediction shape: {eeg_predicted.shape}")

        raw = mne.concatenate_raws(eeg_files_copy[sub])
        raw_copy = deepcopy(raw)
        raw_copy.pick(frontal_roi)

        # Drop bad segments
        raw_clean = drop_bad_segments(raw_copy)
        print(f"[INFO] {sub} raw_clean duration: {raw_clean.times[-1]:.2f}s, samples: {raw_clean.n_times}")

        # Average over channels (shape: 1 x n_times)
        avg_data = raw_clean.get_data().mean(axis=0, keepdims=True)

        info = mne.create_info(ch_names=['avg'], sfreq=raw_clean.info['sfreq'], ch_types='eeg')
        raw_avg = mne.io.RawArray(avg_data, info)
        raw_avg = raw_avg.filter(l_freq=1, h_freq=30)

        min_len = min(raw_avg._data.shape[1], eeg_predicted.shape[0])
        print(
            f"[CHECKPOINT] {sub} | min_len: {min_len}, raw_avg: {raw_avg._data.shape[1]}, prediction: {eeg_predicted.shape[0]}")

        # Subtract prediction from EEG to get residual
        eeg_residual_data = raw_avg._data[:, :min_len] - eeg_predicted[:min_len][np.newaxis, :]
        eeg_residual = mne.io.RawArray(eeg_residual_data, info)

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
        tmin = -0.2
        tmax = 0.8
        tmin_samples = int(abs(tmin) * sfreq)
        tmax_samples = int(tmax * sfreq)

        valid_events = filtered_events[
            (filtered_events[:, 0] - tmin_samples >= 0) &
            (filtered_events[:, 0] + tmax_samples < n_samples)
            ]
        print(f"[CHECKPOINT] {sub} valid events after edge trimming: {len(valid_events)}")

        # Create epochs
        event_id = {str(i): i for i in np.unique(valid_events[:, 2].astype(int))}
        epochs = mne.Epochs(eeg_residual, events=valid_events.astype(int), event_id=event_id,
                            tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True)

        print(f"[CHECKPOINT] {sub} residual epochs shape: {epochs.get_data().shape}")

        epochs_dict[sub] = epochs

    print(f"\n[CHECKPOINT] All subjects processed for residual epochs.\n")
    return epochs_dict


def compute_avg_tfr_across_subjects(epochs_dict, baseline=(-0.2, 0.0),
                                    mode='logratio'):
    """
    Computes time-frequency representation (TFR) on all epochs of every subject,
    averages across subjects, and plots the result.

    Parameters:
        epochs_dict: dict of {sub_id: mne.Epochs}
        freqs: array-like, frequencies of interest
        n_cycles: float | array-like, number of cycles per frequency
        baseline: tuple, baseline period for correction
        mode: str, baseline correction mode

    Returns:
        tfr_avg: mne.AverageTFR, the grand average across subjects
    """
    freqs = np.logspace(np.log10(1), np.log10(30), 100)
    n_cycles = freqs / 2  # fewer cycles at low freqs, more at high freqs
    all_tfrs = []

    for sub, epochs in epochs_dict.items():
        print(f"Computing TFR for {sub} ...")
        tfr = epochs.compute_tfr(
            method='multitaper',
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=False,
            average=False,  # <-- important for induced activity
            decim=1,
            n_jobs=1,
        )
        tfr.apply_baseline(baseline=baseline, mode=mode)
        tfr_avg = tfr.average()  # average *power* across trials
        all_tfrs.append(tfr_avg)

    print("Averaging induced TFRs across subjects ...")
    tfr_avg = mne.grand_average(all_tfrs)

    return tfr_avg, all_tfrs


def compute_avg_itc_across_subjects(epochs_dict, stream='', cond='', tfa_type=''):
    """
    Computes average ITC (inter-trial coherence) for each subject.

    Returns:
        itc_avg: mne.EvokedTFR (grand average across subjects)
        all_itcs: list of subject-level ITCs
    """

    freqs = np.logspace(np.log10(1), np.log10(30), 100)

    n_cycles = freqs / 2

    all_itcs = []

    for sub, epochs in epochs_dict.items():
        print(f"Computing ITC for {sub} ...")
        power, itc = epochs.compute_tfr(
            method='multitaper',
            freqs=freqs,
            n_cycles=n_cycles,
            return_itc=True,
            average=True,
            decim=1,
            n_jobs=1,
        )

        all_itcs.append(itc)
        # Save ITC data (just the array, not the full object)
        save_dir = default_path / f'data/eeg/trf/trf_testing/single_sub/debug/{plane}/{cond}/{folder_type}/{tfa_type}'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{sub}_{cond}_{stream}_{tfa_type}.npy"
        np.save(save_path, itc.data)

    print("Averaging ITC across subjects ...")
    itc_avg = mne.grand_average(all_itcs)

    return all_itcs, itc_avg


def plot_custom_tfr(tfr, title='Induced TFR', vmin=-0.5, vmax=0.5, cmap='RdBu_r', cond='', stream=''):
    data = tfr.data[0]  # shape: (n_freqs, n_times)
    times = tfr.times
    freqs = tfr.freqs

    plt.figure(figsize=(8, 4))
    plt.imshow(data, aspect='auto', origin='lower',
               extent=[times[0], times[-1], freqs[0], freqs[-1]],
               cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(label='Power (log ratio)')
    plt.xlabel('Time (s)')
    plt.ylabel('Frequency (Hz)')
    plt.title(title)
    plt.tight_layout()
    plt.show()
    tfa_fig_dir = default_path / f'data/eeg/trf/trf_testing/single_sub/debug/{plane}/{cond}/{folder_type}/TFA/figures'
    tfa_fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(tfa_fig_dir/ f'induced_tfr_{cond}_{stream}_{folder_type}.png')


def extract_band_power(tfr, band, time_window=(0.0, 0.5)):
    """
    Extracts mean power in a frequency band and time window for a single TFR.
    band: tuple (fmin, fmax)
    time_window: tuple (tmin, tmax)
    """
    fmin, fmax = band
    tmin, tmax = time_window
    freq_mask = (tfr.freqs >= fmin) & (tfr.freqs <= fmax)
    time_mask = (tfr.times >= tmin) & (tfr.times <= tmax)
    return tfr.data[0][freq_mask][:, time_mask].mean()


def extract_band_itc(itc, band, time_window=(0.0, 0.5)):
    fmin, fmax = band
    tmin, tmax = time_window
    freq_mask = (itc.freqs >= fmin) & (itc.freqs <= fmax)
    time_mask = (itc.times >= tmin) & (itc.times <= tmax)
    return itc.data[0][freq_mask][:, time_mask].mean()


def tfa_metrics(tfa_all, stream, type):
    if type == 'TFR':
        for band_name, band_range in bands.items():
            powers = [extract_band_power(tfr, band=band_range) for tfr in tfa_all]
            rms = np.sqrt(np.mean(np.array(powers) ** 2))
            print(
                f"{stream.capitalize()} - {band_name} mean power across subjects: {np.mean(powers):.3f} ± {np.std(powers):.3f}")
            print(f"{band_name} RMS power across subjects: {rms:.4f}")

    elif type == 'ITC':
        for band_name, band_range in bands.items():
            band_vals = [extract_band_itc(itc, band=band_range) for itc in tfa_all]
            print(
                f"{stream.capitalize()} - {band_name} mean ITC across subjects: {np.mean(band_vals):.3f} ± {np.std(band_vals):.3f}")

def plot_tfa(epochs_dict, tfa_all, stream='', type=''):
    for sub, tfa in zip(epochs_dict.keys(), tfa_all):
        if type == 'TFR':
            plot_custom_tfr(tfa, title=f'{stream.capitalize()} Stream - Induced {type} – {sub}')
        elif type == 'ITC':
            plot_custom_tfr(tfa, title=f'{stream.capitalize()} Stream- {type} – {sub}', vmin=0, vmax=1, cmap='viridis')


def paired_stats(target_vals, distractor_vals, band, metric):
    target_vals = np.array(target_vals)
    distractor_vals = np.array(distractor_vals)
    diffs = target_vals - distractor_vals

    # Normality test
    stat, p_norm = shapiro(diffs)

    if p_norm > 0.05:
        # Normal → Paired t-test
        t, p_val = ttest_rel(target_vals, distractor_vals)
        test_used = "Paired t-test"
    else:
        # Not normal → Wilcoxon signed-rank test
        t, p_val = wilcoxon(target_vals, distractor_vals)
        test_used = "Wilcoxon signed-rank test"
        # Effect size: Cohen's d (for t-test) or r equivalent for Wilcoxon
    cohen_d = np.mean(diffs) / np.std(diffs, ddof=1)

    print(f"/n{band.upper()} {metric} comparison ({test_used}):")
    print(f"p(normality) = {p_norm:.4f}")
    print(f"stat = {t:.3f}, p = {p_val:.4f}, Cohen's d = {cohen_d:.3f}")


def plot_dual_itc_time_series(itcs1, itcs2, band_range, band_name='Theta', label1='Target', label2='Distractor', color1='blue', color2='red', cond = '', show_individuals=False, show_sem=True, show_sd=False):
    """
    Plots ITC over time for two streams (e.g., Target vs Distractor).

    Parameters:
    - itcs1, itcs2: lists of AverageTFR (per subject)
    - band_range: tuple of (fmin, fmax)
    - band_name: str label for title
    - label1, label2: str labels for legend
    - color1, color2: colors for each stream
    - show_individuals: if True, plot individual subject lines
    - show_sem: if True, plot SEM shading
    - show_sd: if True, plot SD shading
    """
    def extract_band_data(itcs):
        band_data = []
        for itc in itcs:
            fmask = (itc.freqs >= band_range[0]) & (itc.freqs <= band_range[1])
            mean_band = itc.data[:, fmask, :].mean(axis=1).squeeze()
            band_data.append(mean_band)
        return np.array(band_data)

    data1 = extract_band_data(itcs1)
    data2 = extract_band_data(itcs2)
    times = itcs1[0].times

    mean1 = data1.mean(axis=0)
    sem1 = data1.std(axis=0) / np.sqrt(data1.shape[0])
    sd1 = data1.std(axis=0)

    mean2 = data2.mean(axis=0)
    sem2 = data2.std(axis=0) / np.sqrt(data2.shape[0])
    sd2 = data2.std(axis=0)

    plt.figure(figsize=(14, 6))

    if show_individuals:
        for subj in data1:
            plt.plot(times, subj, color=color1, alpha=0.2, linewidth=1)
        for subj in data2:
            plt.plot(times, subj, color=color2, alpha=0.2, linewidth=1)

    plt.plot(times, mean1, label=label1, color=color1, linewidth=2)
    plt.plot(times, mean2, label=label2, color=color2, linewidth=2)

    if show_sem:
        plt.fill_between(times, mean1 - sem1, mean1 + sem1, alpha=0.3, color=color1, label=f'{label1} SEM')
        plt.fill_between(times, mean2 - sem2, mean2 + sem2, alpha=0.3, color=color2, label=f'{label2} SEM')

    if show_sd:
        plt.fill_between(times, mean1 - sd1, mean1 + sd1, alpha=0.15, color=color1, label=f'{label1} SD')
        plt.fill_between(times, mean2 - sd2, mean2 + sd2, alpha=0.15, color=color2, label=f'{label2} SD')

    plt.axvline(0, linestyle='--', color='gray')
    plt.title(f"{band_name} ITC Over Time")
    plt.xlabel("Time (s)")
    plt.ylabel("ITC")
    plt.legend()
    plt.tight_layout()
    plt.show()
    itc_fig_dir = default_path / f'data/eeg/trf/trf_testing/composite_model/single_sub/debug/{plane}/{cond}/{folder_type}/{predictor_short}/ITC/figures'
    itc_fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(itc_fig_dir/'itc_time_series.png')

if __name__ == '__main__':
    pred_types = ['onsets', 'envelopes', 'RT_labels', 'overlap_ratios']
    predictor_short = "_".join([p[:2] for p in pred_types])

    subs = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19', 'sub20',
            'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29']

    plane = 'azimuth'
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


    targets_epochs_dict1 = get_residual_eegs(target_preds_dict1, eeg_files1, target_mne_events1)
    distractors_epochs_dict1 = get_residual_eegs(distractor_preds_dict1, eeg_files1, distractor_mne_events1)

    targets_epochs_dict2 = get_residual_eegs(target_preds_dict2, eeg_files2, target_mne_events2)
    distractors_epochs_dict2 = get_residual_eegs(distractor_preds_dict2, eeg_files2, distractor_mne_events2)


    targets_all_itcs1, targets_itc_avg1 = compute_avg_itc_across_subjects(targets_epochs_dict1, stream='target', cond=cond1, tfa_type='ITC')
    distractors_all_itcs1, distractors_itc_avg1 = compute_avg_itc_across_subjects(distractors_epochs_dict1, 'distractor', cond=cond1, tfa_type='ITC')

    targets_all_itcs2, targets_itc_avg2 = compute_avg_itc_across_subjects(targets_epochs_dict2, stream='target', cond=cond2, tfa_type='ITC')
    distractors_all_itcs2, distractors_itc_avg2 = compute_avg_itc_across_subjects(distractors_epochs_dict2, stream='distractor', cond=cond2, tfa_type='ITC')


    tfa_targets1, target_all_tfrs1 = compute_avg_tfr_across_subjects(targets_epochs_dict1, baseline=(-0.2, 0.0), mode='logratio')
    tfa_distractor1, distractor_all_tfrs1 = compute_avg_tfr_across_subjects(distractors_epochs_dict1, baseline=(-0.2, 0.0), mode='logratio')

    tfa_targets2, target_all_tfrs2 = compute_avg_tfr_across_subjects(targets_epochs_dict2, baseline=(-0.2, 0.0), mode='logratio')
    tfa_distractor2, distractor_all_tfrs2 = compute_avg_tfr_across_subjects(distractors_epochs_dict2, baseline=(-0.2, 0.0), mode='logratio')

    # Get time and frequency axes from first subject
    def subs_itc_over_time(target_all_itcs, distractor_all_itcs, band=(4,7), cond='', band_range=''):
        itc_fig_dir = default_path / f'data/eeg/trf/trf_testing/composite_model/single_sub/{plane}/{cond}/{folder_type}/on_en_RT_ov/ITC/figures'
        itc_fig_dir.mkdir(parents=True, exist_ok=True)
        itc_times = target_all_itcs[0].times
        itc_freqs = target_all_itcs[0].freqs

        # Frequency mask for theta
        freq_mask = (itc_freqs >= band[0]) & (itc_freqs <= band[1])

        # Set up subplots
        fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

        # === Plot Target ITC ===
        for idx, itc in enumerate(target_all_itcs):
            subject_itc = itc.data[0][freq_mask].mean(axis=0)
            axes[0].plot(itc_times, subject_itc, label=f"sub{10 + idx:02d}", alpha=0.8)

        axes[0].set_title(f"{cond} - Target Stream – {band_range.capitalize()} ITC Over Time")
        axes[0].set_xlabel("Time (s)")
        axes[0].set_ylabel("ITC")
        axes[0].grid(True)
        axes[0].legend(fontsize='small', loc='upper right')

        # === Plot Distractor ITC ===
        for idx, itc in enumerate(distractor_all_itcs):
            subject_itc = itc.data[0][freq_mask].mean(axis=0)
            axes[1].plot(itc_times, subject_itc, label=f"sub{10 + idx:02d}", alpha=0.8)

        axes[1].set_title(f"{cond} - Distractor Stream – {band_range.capitalize()} ITC Over Time")
        axes[1].set_xlabel("Time (s)")
        axes[1].grid(True)
        axes[1].legend(fontsize='small', loc='upper right')

        plt.suptitle(f"{band_range.capitalize()} Band ITC Over Time per Subject", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()
        plt.savefig(itc_fig_dir / f'{cond}_{band_range}_sub_itcs_over_time.png')


    subs_itc_over_time(targets_all_itcs1, distractors_all_itcs1, band=(4,7), cond=cond1, band_range='theta')
    subs_itc_over_time(targets_all_itcs2, distractors_all_itcs2,band=(4,7), cond=cond2, band_range='theta')

    subs_itc_over_time(targets_all_itcs1, distractors_all_itcs1, band=(1,4), cond=cond1, band_range='delta')
    subs_itc_over_time(targets_all_itcs2, distractors_all_itcs2,band=(1,4), cond=cond2, band_range='delta')

    subs_itc_over_time(targets_all_itcs1, distractors_all_itcs1, band=(7,13), cond=cond1, band_range='alpha')
    subs_itc_over_time(targets_all_itcs2, distractors_all_itcs2,band=(7,13), cond=cond2, band_range='alpha')

    subs_itc_over_time(targets_all_itcs1, distractors_all_itcs1, band=(13, 30), cond=cond1, band_range='beta')
    subs_itc_over_time(targets_all_itcs2, distractors_all_itcs2, band=(13, 30), cond=cond2, band_range='beta')



    bands = {
        'delta': (1, 4),
        'theta': (4, 7),
        'alpha': (8, 12),
        'beta': (13, 25),
    }


    tfa_metrics(target_all_tfrs1, stream='target', type='TFR')
    tfa_metrics(distractor_all_tfrs1, stream='distractor', type='TFR')

    tfa_metrics(targets_all_itcs1, stream='target', type='ITC')
    tfa_metrics(distractors_all_itcs1, stream='distractor', type='ITC')

    # stat comparison:
    for band, band_range in bands.items():
        target_vals = [extract_band_itc(itc, band=band_range) for itc in targets_all_itcs1]
        distractor_vals = [extract_band_itc(itc, band=band_range) for itc in distractors_all_itcs1]
        paired_stats(target_vals, distractor_vals, band, 'ITC')

    for band, band_range in bands.items():
        target_vals = [extract_band_itc(itc, band=band_range) for itc in targets_all_itcs2]
        distractor_vals = [extract_band_itc(itc, band=band_range) for itc in distractors_all_itcs2]
        paired_stats(target_vals, distractor_vals, band, 'ITC')

    for band, band_range in bands.items():
        target_vals = [extract_band_power(tfr, band=band_range) for tfr in target_all_tfrs1]
        distractor_vals = [extract_band_power(tfr, band=band_range) for tfr in distractor_all_tfrs1]
        paired_stats(target_vals, distractor_vals, band, 'Power')

    plot_dual_itc_time_series(
        itcs1=targets_all_itcs1,
        itcs2=distractors_all_itcs1,
        band_range=(4, 7),
        band_name='Theta',
        label1='Target',
        label2='Distractor',
        color1='blue',
        color2='red',
        cond=cond1,
        show_individuals=False,
        show_sem=True,
        show_sd=True
    )

    plot_dual_itc_time_series(
        itcs1=targets_all_itcs2,
        itcs2=distractors_all_itcs2,
        band_range=(4, 7),
        band_name='Theta',
        label1='Target',
        label2='Distractor',
        color1='blue',
        color2='red',
        cond=cond2,
        show_individuals=False,
        show_sem=True,
        show_sd=True
    )

    # === Focus on ITCs for smaller theta bands, and compare across conditions === #

    # Define theta sub-bands (e.g., 0.05 Hz bins from 4 to 7 Hz)
    theta_bins = [(round(f, 2), round(f + 0.2, 1)) for f in np.arange(4, 7)]
    delta_bins = [(round(f, 1), round(f + 0.5, 1)) for f in np.arange(1, 4)]
    alpha_bins = [(round(f, 1), round(f + 0.5, 1)) for f in np.arange(7, 13)]

    def single_sub_itcs(band_bins, targets_all_itcs, distractors_all_itcs, band='', cond=''):
        # Collect ITC values per subject per bin
        save_dir = Path(f'C:/Users/pppar/PycharmProjects/VKK_Attention/data/eeg/trf/trf_testing/composite_model/single_sub/{plane}/{cond}/{folder_type}/on_en_RT_ov/figures')
        os.makedirs(save_dir, exist_ok=True)
        results = {
            'Subject': [],
            'Stream': [],
            'Freq_Low': [],
            'Freq_High': [],
            'ITC': []
        }

        for (fmin, fmax) in band_bins:
            for sub, target_itc, distractor_itc in zip(subs, targets_all_itcs, distractors_all_itcs):
                for stream_label, itc_obj in [('target', target_itc), ('distractor', distractor_itc)]:
                    band_val = extract_band_itc(itc_obj, band=(fmin, fmax))
                    results['Subject'].append(sub)
                    results['Stream'].append(stream_label)
                    results['Freq_Low'].append(fmin)
                    results['Freq_High'].append(fmax)
                    results['ITC'].append(band_val)

        # Convert to DataFrame
        import pandas as pd
        df = pd.DataFrame(results)

        # === Statistical comparisons per bin === #
        print("/n====== Paired comparisons per narrow theta bin ======")
        for (fmin, fmax) in band_bins:
            df_bin = df[(df['Freq_Low'] == fmin) & (df['Freq_High'] == fmax)]
            df_pivot = df_bin.pivot(index='Subject', columns='Stream', values='ITC')
            diffs = df_pivot['target'] - df_pivot['distractor']
            stat, p_norm = shapiro(diffs)

            if p_norm > 0.05:
                # Normal distribution → t-test
                t_stat, p_val = ttest_rel(df_pivot['target'], df_pivot['distractor'])
                test_used = "Paired t-test"
            else:
                # Non-normal → Wilcoxon
                t_stat, p_val = wilcoxon(df_pivot['target'], df_pivot['distractor'])
                test_used = "Wilcoxon signed-rank"

            cohen_d = diffs.mean() / diffs.std(ddof=1)

            print(f"{band} {fmin:.1f}–{fmax:.1f} Hz | {test_used}: stat={t_stat:.3f}, p={p_val:.4f}, d={cohen_d:.3f}")


        # single sub trends:
        # Set up the subplots
        import seaborn as sns

        fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)

        # Define the two streams
        streams = ['target', 'distractor']
        colors = ['blue', 'orange']

        for ax, stream, color in zip(axes, streams, colors):
            df_stream = df[df['Stream'] == stream]

            # Plot each subject's line with bold lines and subject ID in legend
            for subject_id in df_stream['Subject'].unique():
                df_sub = df_stream[df_stream['Subject'] == subject_id]
                ax.plot(
                    df_sub['Freq_Low'],
                    df_sub['ITC'],
                    label=subject_id,
                    linewidth=2.0,  # <-- make bolder
                    alpha=0.8
                )

            # Plot group mean
            sns.lineplot(
                data=df_stream,
                x='Freq_Low',
                y='ITC',
                color=color,
                label=f'{stream.capitalize()} Mean',
                ax=ax,
                errorbar='se',
                lw=3.0,
                linestyle='--'
            )

            ax.set_title(f"{stream.capitalize()} Stream")
            ax.set_xlabel("Frequency Bin Start (Hz)")
            ax.set_ylabel("ITC")
            ax.set_xticks(sorted(df_stream['Freq_Low'].unique()))
            ax.grid(True)
            ax.legend(loc='upper right', fontsize='small', title='Subjects', frameon=True)

        plt.suptitle(f"Individual ITC Trends Across {band} Bins by Subject", fontsize=16)
        plt.tight_layout()
        plt.show()
        plt.savefig(save_dir / f'{band}_bins_single_subs.png')
        return df

    delta_df1 = single_sub_itcs(delta_bins, targets_all_itcs1, distractors_all_itcs1,'Delta')
    theta_df1 = single_sub_itcs(theta_bins, targets_all_itcs1, distractors_all_itcs1,'Theta')
    alpha_df1 = single_sub_itcs(alpha_bins, targets_all_itcs1, distractors_all_itcs1,'Alpha')

    delta_df2 = single_sub_itcs(delta_bins, targets_all_itcs2, distractors_all_itcs2,'Delta')
    theta_df2 = single_sub_itcs(theta_bins, targets_all_itcs2, distractors_all_itcs2,'Theta')
    alpha_df2 = single_sub_itcs(alpha_bins, targets_all_itcs2, distractors_all_itcs2,'Alpha')

    # === Compare peak theta ITC frequency: target vs. distractor === #
    def extract_peak_theta_freqs_precise(df_theta):
        peak_freqs = {
            'Subject': [],
            'Target_PeakFreq_Hz': [],
            'Distractor_PeakFreq_Hz': [],
            'Target_PeakITC': [],
            'Distractor_PeakITC': []
        }

        for sub in df_theta['Subject'].unique():
            df_sub = df_theta[df_theta['Subject'] == sub]

            df_target = df_sub[df_sub['Stream'] == 'target']
            df_distractor = df_sub[df_sub['Stream'] == 'distractor']

            # Find peak ITC bin and calculate its center frequency
            target_idx = df_target['ITC'].idxmax()
            distractor_idx = df_distractor['ITC'].idxmax()

            target_row = df_target.loc[target_idx]
            distractor_row = df_distractor.loc[distractor_idx]

            target_freq_hz = (target_row['Freq_Low'] + target_row['Freq_High']) / 2
            distractor_freq_hz = (distractor_row['Freq_Low'] + distractor_row['Freq_High']) / 2

            peak_freqs['Subject'].append(sub)
            peak_freqs['Target_PeakFreq_Hz'].append(target_freq_hz)
            peak_freqs['Distractor_PeakFreq_Hz'].append(distractor_freq_hz)
            peak_freqs['Target_PeakITC'].append(target_row['ITC'])
            peak_freqs['Distractor_PeakITC'].append(distractor_row['ITC'])

        return pd.DataFrame(peak_freqs)

    peak_df1 = extract_peak_theta_freqs_precise(theta_df1)
    peak_df2 = extract_peak_theta_freqs_precise(theta_df2)

    peak_alpha_df1 = extract_peak_theta_freqs_precise(alpha_df1)
    peak_alpha_df2 = extract_peak_theta_freqs_precise(alpha_df2)

    def peak_diffs(peak_df):
        print("/n=== Subject-wise Peak Theta Frequencies (Hz) ===")
        print(peak_df[['Subject', 'Target_PeakFreq_Hz', 'Distractor_PeakFreq_Hz']])

        # Difference in Hz
        diffs = peak_df['Target_PeakFreq_Hz'] - peak_df['Distractor_PeakFreq_Hz']

        # Count how many subjects show expected direction (target > distractor)
        n_higher = (diffs > 0).sum()
        n_equal = (diffs == 0).sum()
        n_lower = (diffs < 0).sum()

        print(f"/nSubjects with Target Peak > Distractor: {n_higher}/{len(diffs)}")
        print(f"Subjects with Target = Distractor: {n_equal}/{len(diffs)}")
        print(f"Subjects with Target < Distractor: {n_lower}/{len(diffs)}")

        # Paired comparison
        diffs = peak_df['Target_PeakFreq_Hz'] - peak_df['Distractor_PeakFreq_Hz']
        stat, p_norm = shapiro(diffs)

        if p_norm > 0.05:
            stat, p = ttest_rel(peak_df['Target_PeakFreq_Hz'], peak_df['Distractor_PeakFreq_Hz'])
            test_used = "Paired t-test"
        else:
            stat, p = wilcoxon(peak_df['Target_PeakFreq_Hz'], peak_df['Distractor_PeakFreq_Hz'])
            test_used = "Wilcoxon signed-rank"

        cohen_d = diffs.mean() / diffs.std(ddof=1)

        print(f"/n=== Peak Theta Frequency (Hz) Comparison ({test_used}) ===")
        print(f"Mean Δfreq (target - distractor): {diffs.mean():.2f} Hz")
        print(f"Stat = {stat:.3f}, p = {p:.4f}, Cohen's d = {cohen_d:.3f}")


    peak_diffs(peak_df1)
    peak_diffs(peak_df2)

    peak_diffs(peak_alpha_df1)
    peak_diffs(peak_alpha_df2)