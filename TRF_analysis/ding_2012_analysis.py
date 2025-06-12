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
from scipy.stats import ttest_rel, wilcoxon, shapiro
import pandas as pd
import seaborn as sns
from copy import deepcopy
from scipy.signal import windows
from scipy.stats import zscore, ttest_rel
from statsmodels.stats.multitest import fdrcorrection


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



if __name__ == '__main__':
    pred_types = ['onsets', 'envelopes']
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
    epoch_length = sfreq * 60  # samples in 1 minute

    # Define channel info for single-channel data
    ch_name = 'predicted'  # or 'target' / 'distractor'
    ch_type = 'misc'  # use 'misc' for predicted, non-EEG data
    default_path = Path.cwd()
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'


    target_preds_dict1, distractor_preds_dict1 = get_pred_dicts(cond=cond1)
    target_preds_dict2, distractor_preds_dict2 = get_pred_dicts(cond=cond2)

    # Create target and distractor epoch objects
    target_epochs_dict1 = make_epochs(target_preds_dict1, sfreq, epoch_length, ch_name='target_pred')
    distractor_epochs_dict1 = make_epochs(distractor_preds_dict1, sfreq, epoch_length, ch_name='distractor_pred')

    # Create target and distractor epoch objects
    target_epochs_dict2 = make_epochs(target_preds_dict2, sfreq, epoch_length, ch_name='target_pred')
    distractor_epochs_dict2 = make_epochs(distractor_preds_dict2, sfreq, epoch_length, ch_name='distractor_pred')


    # --- Parameters ---
    fmin, fmax = 1, 30
    sfreq = 125  # or your actual sampling rate

    subs = list(target_preds_dict1.keys())


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


    # --- Collect z-scored power for all subjects ---
    target_power = []
    distractor_power = []

    for sub in subs:
        targ_evoked = target_epochs_dict2[sub].average(picks='all')
        dist_evoked = distractor_epochs_dict2[sub].average(picks='all')

        power_freqs, targ_pow = compute_zscored_power(targ_evoked, sfreq, fmin, fmax)
        _, dist_pow = compute_zscored_power(dist_evoked, sfreq, fmin, fmax)

        target_power.append(targ_pow)
        distractor_power.append(dist_pow)

    target_power = np.array(target_power)
    distractor_power = np.array(distractor_power)

    # --- Paired Wilcoxon per frequency ---

    wilcoxon_p = []

    for i in range(target_power.shape[1]):  # for each frequency bin
        try:
            _, p = wilcoxon(target_power[:, i], distractor_power[:, i])
        except ValueError:
            p = 1.0  # fallback in case of constant data or errors
        wilcoxon_p.append(p)

    wilcoxon_p = np.array(wilcoxon_p)
    _, p_fdr = fdrcorrection(wilcoxon_p)

    # --- Plot ---
    plt.figure(figsize=(12, 6))
    plt.plot(freqs, target_power.mean(axis=0), label='Target', color='blue')
    plt.plot(freqs, distractor_power.mean(axis=0), label='Distractor', color='red')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Z-scored Power')
    plt.title('Power Spectrum (Target vs Distractor) in ROI')
    plt.legend()
    plt.grid(True)

    # Mark significant frequencies
    sig_freqs = freqs[p_fdr < 0.05]
    sig_heights = np.maximum(target_power.mean(axis=0), distractor_power.mean(axis=0))[p_fdr < 0.05]
    plt.scatter(sig_freqs, sig_heights + 0.2, color='green', s=30, label='p < 0.05 (FDR)')

    plt.tight_layout()
    plt.show()

    target_peak_freqs = freqs[np.argmax(target_power, axis=1)]
    distractor_peak_freqs = freqs[np.argmax(distractor_power, axis=1)]

    # Optional: paired test on peak frequencies
    from scipy.stats import wilcoxon

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

    # ITC on induced:

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


    eeg_files1 = get_eeg_files(condition=cond1)



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

    target_events1, distractor_events1 = get_events_dicts(folder_name1='stream1', folder_name2='stream2', cond=cond1)

    def get_residual_eegs(preds_dict=None, eeg_files=None, cond=''):
        from TRF_test.TRF_test_config import frontal_roi
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


    target_epochs_induced1 = get_residual_eegs(preds_dict=target_preds_dict1, eeg_files=eeg_files1, cond=cond1)
    distractor_epochs_induced1 = get_residual_eegs(preds_dict=distractor_preds_dict1, eeg_files=eeg_files1, cond=cond1)

    from mne.time_frequency import tfr_multitaper

    # Parameters
    fmin, fmax = 1, 10
    itc_freqs = np.linspace(fmin, fmax, num=60)  # e.g., 60 points between 1–10 Hz
    n_cycles = 2*freqs  # 1 cycle per frequency (adjust as needed)


    def compute_itc(epochs_dict, freqs, n_cycles):
        itcs = {}
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
        return itcs


    target_itc = compute_itc(target_epochs_induced1, freqs, n_cycles)
    distractor_itc = compute_itc(distractor_epochs_induced1, freqs, n_cycles)

    target_vals = []
    distractor_vals = []

    for sub in subs:
        # shape: (n_epochs, n_channels=1, n_freqs, n_times)
        targ_data = target_itc[sub].data.mean(axis=(0, 2))  # mean over channels and time
        dist_data = distractor_itc[sub].data.mean(axis=(0, 2))

        target_vals.append(targ_data)
        distractor_vals.append(dist_data)

    target_vals = np.array(target_vals)
    distractor_vals = np.array(distractor_vals)

    from scipy.stats import shapiro

    # Assume target_vals and distractor_vals are shaped (n_subjects, n_freqs)
    n_freqs = target_vals.shape[1]
    normal_target = []
    normal_distractor = []

    for f in range(n_freqs):
        p_targ = shapiro(target_vals[:, f]).pvalue
        p_dist = shapiro(distractor_vals[:, f]).pvalue
        normal_target.append(p_targ > 0.05)  # True if normal
        normal_distractor.append(p_dist > 0.05)


    from scipy.stats import ttest_rel
    from statsmodels.stats.multitest import fdrcorrection

    t_vals, p_vals = ttest_rel(target_vals, distractor_vals, axis=0)
    _, p_fdr = fdrcorrection(p_vals)

    sig_freqs = freqs[p_fdr < 0.05]
    print("Significant frequencies (FDR-corrected):", sig_freqs)


    def cohens_d_paired(x, y):
        diff = x - y
        return diff.mean() / diff.std(ddof=1)


    effect_sizes = np.array([cohens_d_paired(target_vals[:, i], distractor_vals[:, i])
                             for i in range(target_vals.shape[1])])

    # ITC difference
    mean_diff = target_vals.mean(axis=0) - distractor_vals.mean(axis=0)
    sem_diff = (target_vals - distractor_vals).std(axis=0, ddof=1) / np.sqrt(target_vals.shape[0])

    plt.figure(figsize=(12, 5))
    plt.plot(freqs, mean_diff, label='Target - Distractor', color='purple')
    plt.fill_between(freqs, mean_diff - sem_diff, mean_diff + sem_diff, alpha=0.3, color='purple')

    # Add significance markers
    sig_freqs = freqs[p_fdr < 0.05]
    sig_vals = mean_diff[p_fdr < 0.05] + 0.01  # Offset markers slightly above line
    plt.scatter(sig_freqs, sig_vals, color='black', marker='*', label='p < 0.05 (FDR)')

    # Add effect size as dashed line (optional)
    plt.plot(freqs, effect_sizes / 5, linestyle='--', color='gray', label="Cohen's d (scaled)")

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('ITC Difference (Target − Distractor)')
    plt.title('ITC Difference Across Frequencies')
    plt.axhline(0, color='black', linestyle='--')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Compute means and SEM
    from scipy.stats import sem
    target_mean = target_vals.mean(axis=0)
    distractor_mean = distractor_vals.mean(axis=0)
    target_sem = sem(target_vals, axis=0)
    distractor_sem = sem(distractor_vals, axis=0)

    # Plot
    plt.figure(figsize=(12, 5))
    plt.plot(freqs, target_mean, label='Target', color='blue')
    plt.fill_between(freqs, target_mean - target_sem, target_mean + target_sem, alpha=0.3, color='blue')

    plt.plot(freqs, distractor_mean, label='Distractor', color='red')
    plt.fill_between(freqs, distractor_mean - distractor_sem, distractor_mean + distractor_sem, alpha=0.3, color='red')

    # Add significance markers
    sig_freqs = freqs[p_fdr < 0.05]
    sig_vals = target_mean[p_fdr < 0.05] + 0.01  # Slightly above target line
    plt.scatter(sig_freqs, sig_vals, color='black', marker='*', label='p < 0.05 (FDR)')

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('ITC')
    plt.title('Inter-Trial Coherence (Target vs Distractor)')
    plt.axhline(0, linestyle='--', color='gray', linewidth=0.8)
    plt.legend()
    plt.tight_layout()
    plt.show()

    for sub_itc in target_vals:
        plt.plot(freqs, sub_itc, alpha=0.4, color='blue')
    plt.plot(freqs, target_vals.mean(axis=0), color='black', linewidth=2, label='Mean ITC')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('ITC (Target Stream)')
    plt.title('Subject-Level ITC Curves (Target)')
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.figure(figsize=(12, 5))
    for sub_itc in distractor_vals:
        plt.plot(freqs, sub_itc, alpha=0.4, color='blue')
    plt.plot(freqs, distractor_vals.mean(axis=0), color='black', linewidth=2, label='Mean ITC')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('ITC (Target Stream)')
    plt.title('Subject-Level ITC Curves (Target)')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Find index closest to 4 Hz
    freq_of_interest = 4.0
    f_idx = np.argmin(np.abs(freqs - freq_of_interest))

    import seaborn as sns
    import pandas as pd

    df = pd.DataFrame({
        'Target': target_vals[:, f_idx],
        'Distractor': distractor_vals[:, f_idx]
    }).melt(var_name='Stream', value_name='ITC')

    plt.figure(figsize=(6, 5))
    sns.violinplot(data=df, x='Stream', y='ITC', inner='point')
    plt.title(f'ITC at ~{freqs[f_idx]:.2f} Hz')
    plt.tight_layout()
    plt.show()

    from scipy.stats import spearmanr, pearsonr

    # Use Spearman if data are not normally distributed

    # If you already have ITC values per subject (target_vals)
    # For power (1–1.5 Hz)
    # Now mask correctly
    power_mask = (power_freqs >= 1.1) & (power_freqs <= 1.3)
    itc_mask = (itc_freqs >= 3.5) & (itc_freqs <= 4.5)

    power_band = target_power[:, power_mask]

    # Corresponding freqs
    band_freqs = power_freqs[power_mask]

    # Get peak frequency for each subject
    target_peak_freqs = np.array([
        band_freqs[np.argmax(power_band[i])] for i in range(power_band.shape[0])
    ])

    target_itc_band = target_vals[:, itc_mask].mean(axis=1)

    from scipy.stats import spearmanr

    rho, pval = spearmanr(target_peak_freqs, target_itc_band)
    print(f"Spearman r = {rho:.2f}, p = {pval:.4f}")


