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
    folder_type = folder_types[1]
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

    subs = list(target_epochs_dict1.keys())


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

        freqs, targ_pow = compute_zscored_power(targ_evoked, sfreq, fmin, fmax)
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



