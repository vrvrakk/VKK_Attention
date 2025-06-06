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


# === Load relevant events and mask the bad segments === #

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

    # Loop through all subjects and extract streams
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

                        for files in stim_folders.iterdir():
                            if 'concat' in files.name:
                                stream_data = np.load(files, allow_pickle=True)['onsets']
                                stream = stream_data.copy()

                                # Keep only onset value for each event
                                i = 0
                                while i < len(stream):
                                    val = stream[i]
                                    if val in [1, 2, 3, 4, 5]:
                                        stream[i+1:i+event_length] = 0
                                        i += event_length
                                    else:
                                        i += 1

                                # Convert to MNE-style event array
                                onset_indices = np.where(stream != 0)[0]
                                event_values = stream[onset_indices].astype(int)
                                mne_events = np.column_stack((onset_indices,
                                                              np.zeros_like(onset_indices),
                                                              event_values))

                                # Store in dictionary
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

def get_residual_eegs(cond, stream=''):
    if cond == cond1:
        eeg_files = eeg_files1
        if stream == 'target':
            mne_events = target_mne_events1
        elif stream == 'distractor':
            mne_events = distractor_mne_events1
    elif cond == cond2:
        eeg_files = eeg_files2
        if stream == 'target':
            mne_events = target_mne_events2
        elif stream == 'distractor':
            mne_events = distractor_mne_events2


    epochs_dict = {}
    for sub in subs:
        print(sub)
        eeg_predicted = target_preds_dict[sub]


        raw = mne.concatenate_raws(eeg_files[sub])  # already picked ROI
        raw_copy = deepcopy(raw)
        raw_copy.pick(frontal_roi)

        raw_clean = drop_bad_segments(raw_copy)

        avg_data = raw_clean.get_data().mean(axis=0, keepdims=True)  # shape: (1, n_times)

        # Create a new RawArray with just the averaged channel
        info = mne.create_info(ch_names=['avg'], sfreq=raw_clean.info['sfreq'], ch_types='eeg')
        raw_avg = mne.io.RawArray(avg_data, info)

        min_len = min(raw_avg._data.shape[1], eeg_predicted.shape[0])
        eeg_residual_data = raw_avg._data[:, :min_len] - eeg_predicted[:min_len][np.newaxis, :]
        eeg_residual = mne.io.RawArray(eeg_residual_data, info)

        # Load your custom event array (e.g., target or distractor)
        events = mne_events[sub]  # shape (n_events, 3)

        # e.g. for your MNE-style target_mne_events[sub]
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
        tmin = -0.2
        tmax = 0.8
        tmin_samples = int(abs(tmin) * sfreq)
        tmax_samples = int(tmax * sfreq)

        valid_events = filtered_events[
            (filtered_events[:, 0] - tmin_samples >= 0) &
            (filtered_events[:, 0] + tmax_samples < n_samples)
            ]

        # Optional: Define event_id mapping (if needed for filtering or labeling)
        event_id = {str(i): i for i in np.unique(valid_events[:, 2].astype(int))}  # e.g., {'1': 1, '2': 2, ...}

        # Step 3: Create epochs
        epochs = mne.Epochs(eeg_residual, events=valid_events.astype(int), event_id=event_id,
                            tmin=-0.2, tmax=0.8, baseline=(-0.2, 0), preload=True)

        epochs_dict[sub] = epochs
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


def compute_avg_itc_across_subjects(epochs_dict, stream=''):
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
        save_dir = default_path / f'data/eeg/trf/trf_comparison/{plane}/{folder_type}/ITC'
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{sub}_{stream}_itc.npy"
        np.save(save_path, itc.data)

    print("Averaging ITC across subjects ...")
    itc_avg = mne.grand_average(all_itcs)

    return all_itcs, itc_avg


def plot_custom_tfr(tfr, title='Induced TFR', vmin=-0.5, vmax=0.5, cmap='RdBu_r'):
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

    print(f"\n{band.upper()} {metric} comparison ({test_used}):")
    print(f"p(normality) = {p_norm:.4f}")
    print(f"stat = {t:.3f}, p = {p_val:.4f}, Cohen's d = {cohen_d:.3f}")


def plot_dual_itc_time_series(
    itcs1,
    itcs2,
    band_range,
    band_name='Theta',
    label1='Target',
    label2='Distractor',
    color1='blue',
    color2='red',
    show_individuals=False,
    show_sem=True,
    show_sd=False
):
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

if __name__ == '__main__':

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

    predictions_dir = fr'C:\Users\pppar\PycharmProjects\VKK_Attention\data\eeg\trf\trf_testing\composite_model\single_sub\{plane}\{folder_type}\on_en_RT_ov\weights\predictions'

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

    target_mne_events2, distractor_mne_events2 = get_events_dicts(folder_name1='stream2', folder_name2='stream1', cond=cond2)
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


    targets_epochs_dict1 = get_residual_eegs(cond=cond1, stream='target')
    distractors_epochs_dict1 = get_residual_eegs(cond=cond1, stream='distractor')

    target_epochs_dict2 = get_residual_eegs(cond=cond2, stream='target')
    distractor_epochs_dict2 = get_residual_eegs(cond=cond2, stream='distractor')


    targets_all_itcs1, targets_itc_avg1 = compute_avg_itc_across_subjects(targets_epochs_dict1, stream='target')
    distractors_all_itcs1, distractors_itc_avg1 = compute_avg_itc_across_subjects(distractors_epochs_dict1, 'distractor')

    targets_all_itcs2, targets_itc_avg2 = compute_avg_itc_across_subjects(target_epochs_dict2, stream='target')
    distractors_all_itcs2, distractors_itc_avg2 = compute_avg_itc_across_subjects(distractor_epochs_dict2, stream='distractor')


    tfa_targets1, target_all_tfrs1 = compute_avg_tfr_across_subjects(targets_epochs_dict1, baseline=(-0.2, 0.0), mode='logratio')
    tfa_distractor1, distractor_all_tfrs1 = compute_avg_tfr_across_subjects(distractors_epochs_dict1, baseline=(-0.2, 0.0), mode='logratio')

    tfa_targets2, target_all_tfrs2 = compute_avg_tfr_across_subjects(targets_epochs_dict1, baseline=(-0.2, 0.0),
                                                                     mode='logratio')
    tfa_distractor2, distractor_all_tfrs2 = compute_avg_tfr_across_subjects(distractors_epochs_dict1,
                                                                            baseline=(-0.2, 0.0), mode='logratio')


    # # plot tfr
    # plot_tfa(targets_epochs_dict1, target_all_tfrs1, stream='target', type='TFR')
    # plot_tfa(distractors_epochs_dict1, distractor_all_tfrs1, stream='distractor', type='TFR')
    #
    # # plot ITC:
    # plot_tfa(targets_epochs_dict1, targets_all_itcs1, stream='target', type='ITC')
    # plot_tfa(distractors_epochs_dict1, distractors_all_itcs1, stream='distractor', type='ITC')

    bands = {
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
        show_individuals=False,
        show_sem=True,
        show_sd=True
    )