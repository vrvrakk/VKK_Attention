# === TFA on predicted EEG === #
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mne import create_info

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
from scipy.stats import pearsonr


# === Load relevant events and mask the bad segments === #

# --- Helper: FFT Power Extraction ---
def compute_zscored_power(evoked, sfreq, fmin=1, fmax=30):
    data = evoked
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


def get_epochs(mne_events, eeg_files=None):

    eeg_files_copy = deepcopy(eeg_files)
    epochs_dict = {}

    for sub in subs:
        print(f"\n[CHECKPOINT] Processing {sub}...")


        raw = mne.concatenate_raws(eeg_files_copy[sub])
        raw_copy = deepcopy(raw)
        raw_copy.pick(frontal_roi)

        # Drop bad segments
        raw_clean = drop_bad_segments(sub, cond1, raw_copy)

        # --- Event Filtering ---
        events = mne_events[sub]
        print(f"[CHECKPOINT] {sub} events loaded: {len(events)}")

        sfreq = raw.info['sfreq']
        n_samples = raw.n_times
        bad_time_mask = np.zeros(n_samples, dtype=bool)

        info = mne.create_info(ch_names=raw_copy.info['ch_names'], sfreq=raw_copy.info['sfreq'], ch_types='eeg')

        # Assuming eeg_residual shape: (n_times,)
        eeg = mne.io.RawArray(raw_clean, info)

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
        tmax = 0.3
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
                            tmin=tmin, tmax=tmax, baseline=(tmin, -0.0), preload=True)

        print(f"[CHECKPOINT] {sub} epochs shape: {epochs.get_data().shape}")
        epochs_dict[sub] = epochs

    return epochs_dict

def get_envs():
    target_envs_dict = {}
    distractor_envs_dict = {}
    for sub in subs:
        target_env_path = default_path / f'data/eeg/predictors/envelopes/{sub}/{cond1}/stream1/{sub}_{cond1}_stream1_envelopes_series_concat.npz'
        target_env = np.load(target_env_path)
        target_env = target_env['envelopes']
        target_envs_dict[sub] = target_env
        distractor_env_path = default_path / f'data/eeg/predictors/envelopes/{sub}/{cond1}/stream2/{sub}_{cond1}_stream2_envelopes_series_concat.npz'
        distractor_env = np.load(distractor_env_path)
        distractor_env = distractor_env['envelopes']
        distractor_envs_dict[sub] = distractor_env
    return target_envs_dict, distractor_envs_dict


from scipy.signal import hilbert

def epoch_envelopes_from_eeg_epochs(epochs_dict, envelope_dict, sfreq, tmin=-0.5, tmax=0.3):
    """
    Epoch Hilbert-transformed envelopes using the same timing as EEG epochs.

    Parameters:
    - epochs_dict: dict of MNE Epochs objects (already epoched EEG)
    - envelope_dict: dict of continuous speech envelopes (1D numpy arrays)
    - sfreq: sampling rate of the envelopes (same as EEG)
    - tmin, tmax: epoch time window (should match EEG epoch window)

    Returns:
    - epoched_envs: dict of subject → 2D array (n_epochs x n_samples)
    """
    epoched_envs = {}

    for sub, epochs in epochs_dict.items():
        print(f"[INFO] Processing {sub}...")

        env = envelope_dict[sub]  # 1D array
        analytic_env = np.abs(hilbert(env))  # Hilbert-transformed envelope

        sample_tmin = int(round(tmin * sfreq))
        sample_tmax = int(round(tmax * sfreq))
        n_samples = sample_tmax - sample_tmin

        # Get EEG epoch sample onsets
        eeg_onsets = epochs.events[:, 0]
        env_epochs = []

        for onset in eeg_onsets:
            start = onset + sample_tmin
            end = onset + sample_tmax
            if start < 0 or end > len(analytic_env):
                continue  # skip if out of bounds
            env_epoch = analytic_env[start:end]
            env_epochs.append(env_epoch)

        epoched_envs[sub] = np.stack(env_epochs)
        print(f"→ {sub}: {len(env_epochs)} epochs, shape = {epoched_envs[sub].shape}")

    return epoched_envs

def compute_nsi_per_subject(target_phase_dict, distractor_phase_dict,
                             target_env_phase_dict, distractor_env_phase_dict):
    """
    Computes EEG-envelope phase correlations and NSI per subject.

    Returns:
    - r_target_dict: subject → mean r (target)
    - r_distractor_dict: subject → mean r (distractor)
    - nsi_dict: subject → NSI (target - distractor)
    """
    r_target_dict = {}
    r_distractor_dict = {}
    nsi_dict = {}

    for sub in target_phase_dict:
        target_eeg_phases = target_phase_dict[sub]         # list of [n_freqs x (n_trials x n_channels x n_times)]
        distractor_eeg_phases = distractor_phase_dict[sub]
        target_env_phases = target_env_phase_dict[sub]     # list of [n_freqs x (n_trials x n_times)]
        distractor_env_phases = distractor_env_phase_dict[sub]

        n_freqs = len(target_eeg_phases)
        target_r_vals = []
        distractor_r_vals = []

        for f in range(n_freqs):
            eeg_target = target_eeg_phases[f].mean(axis=1)[..., :100]
            env_target = target_env_phases[f]
            eeg_distractor = distractor_eeg_phases[f].mean(axis=1)[..., :100]
            env_distractor = distractor_env_phases[f]

            # Ensure all pairs match in trial count
            n_trials = min(eeg_target.shape[0], env_target.shape[0],
                           eeg_distractor.shape[0], env_distractor.shape[0])

            eeg_target = eeg_target[:n_trials]
            env_target = env_target[:n_trials]
            eeg_distractor = eeg_distractor[:n_trials]
            env_distractor = env_distractor[:n_trials]

            for t in range(n_trials):
                try:
                    r_t, _ = pearsonr(eeg_target[t], env_target[t])
                    r_d, _ = pearsonr(eeg_distractor[t], env_distractor[t])
                    target_r_vals.append(r_t)
                    distractor_r_vals.append(r_d)
                except Exception as e:
                    print(f"[WARN] Correlation failed on {sub} trial {t}, freq {f}: {e}")

        # Average across trials and freqs
        r_target = np.mean(target_r_vals)
        r_distractor = np.mean(distractor_r_vals)
        r_target_dict[sub] = r_target
        r_distractor_dict[sub] = r_distractor
        nsi_dict[sub] = r_target - r_distractor

        print(f"{sub} → r_target: {r_target:.3f}, r_distractor: {r_distractor:.3f}, NSI: {nsi_dict[sub]:.3f}")

    return r_target_dict, r_distractor_dict, nsi_dict

if __name__ == '__main__':
    # --- Parameters ---
    subs = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19', 'sub20',
            'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29']

    plane='azimuth'

    if plane == 'azimuth':
        cond1 = 'a1'
    if plane == 'elevation':
        cond1 = 'e1'

    folder_types = ['all_stims']
    folder_type = folder_types[0]
    sfreq = 125
    epoch_length = sfreq * 60  # samples in 1 minute

    fmin, fmax = 1, 30

    # Define channel info for single-channel data

    default_path = Path.cwd()
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'
    fig_path = default_path / 'data/eeg/trf/trf_testing/results/single_sub/figures/ITC/'
    fig_path.mkdir(parents=True, exist_ok=True)


    eeg_files1 = get_eeg_files(condition=cond1)

    subs = list(eeg_files1.keys())

    target_events1, distractor_events1 = get_events_dicts(folder_name1='stream1', folder_name2='stream2', cond=cond1)

    target_envs_dict, distractor_envs_dict = get_envs()

    target_epochs = get_epochs(target_events1, eeg_files1)
    distractor_epochs = get_epochs(distractor_events1, eeg_files1)

    target_env_epochs = epoch_envelopes_from_eeg_epochs(target_epochs, target_envs_dict, sfreq, tmin=-0.5, tmax=0.3)
    distractor_env_epochs = epoch_envelopes_from_eeg_epochs(distractor_epochs, distractor_envs_dict, sfreq, tmin=-0.5, tmax=0.3)

    # Parameters
    from scipy.signal import hilbert, butter, filtfilt


    def bandpass_hilbert(data, sfreq, freq_band):
        """Bandpass filter and extract phase via Hilbert transform."""
        nyq = sfreq / 2.
        low, high = freq_band[0] / nyq, freq_band[1] / nyq
        b, a = butter(N=4, Wn=[low, high], btype='band')
        filt_data = filtfilt(b, a, data, axis=-1)
        phase = np.angle(hilbert(filt_data, axis=-1))
        return phase


    freq_bands = [(f, f + 1) for f in range(1, 9)]  # 1–8 Hz


    def compute_phase_epochs(epochs_dict, freq_bands):
        phase_dict = {}
        for sub, epochs in epochs_dict.items():
            sfreq = epochs.info['sfreq']
            data = epochs.get_data()  # shape: (n_trials, n_channels, n_times)

            sub_phases = []
            for band in freq_bands:
                phase = bandpass_hilbert(data, sfreq, band)  # shape: same
                sub_phases.append(phase)  # list of arrays per freq

            phase_dict[sub] = sub_phases  # list of shape (n_freqs, n_trials, n_channels, n_times)

        return phase_dict


    target_phase_epochs = compute_phase_epochs(target_epochs, freq_bands)
    distractor_phase_epochs = compute_phase_epochs(distractor_epochs, freq_bands)


    def compute_env_phase_epochs(env_epochs_dict, freq_bands):
        phase_dict = {}
        for sub, epochs in env_epochs_dict.items():
            sfreq = 125
            data = epochs # shape: (n_trials, n_times)

            sub_phases = []
            for band in freq_bands:
                phase = bandpass_hilbert(data, sfreq, band)  # shape: same
                sub_phases.append(phase)  # list of arrays per freq

            phase_dict[sub] = sub_phases  # list of shape (n_freqs, n_trials, n_channels, n_times)

        return phase_dict

    target_env_phase_epochs = compute_env_phase_epochs(target_env_epochs, freq_bands)
    distractor_env_phase_epochs = compute_env_phase_epochs(distractor_env_epochs, freq_bands)

    r_target_dict, r_distractor_dict, nsi_dict = compute_nsi_per_subject(
        target_phase_epochs, distractor_phase_epochs,
        target_env_phase_epochs, distractor_env_phase_epochs
    )
    res_path = default_path / f'data/eeg/behaviour/{plane}/{cond1}'
    np.savez(res_path/'phase_nsi.npz', nsi_dict)

    def compute_itpc(phase_data):
        # phase_data: shape (n_trials, n_channels, n_times)
        complex_phase = np.exp(1j * phase_data)
        itpc = np.abs(np.mean(complex_phase, axis=0))  # shape: (n_channels, n_times)
        return itpc


    def itpc(phase_dict):
        itpc_dict = {}
        for sub in phase_dict:
            itpc_sub = []
            for freq_phase in phase_dict[sub]:  # each: (n_trials, n_channels, n_times)
                itpc = compute_itpc(freq_phase)
                itpc_sub.append(itpc)  # (n_channels, n_times)
            itpc_dict[sub] = itpc_sub  # list of n_freqs entries
        return itpc_dict

    target_itpc_dict = itpc(target_phase_epochs)
    distractor_itpc_dict = itpc(distractor_phase_epochs)

    def extract_itpc_avg(itpc_dict):
        roi_vals = []
        for sub in itpc_dict:
            avg = [np.mean(itpc) for itpc in itpc_dict[sub]]  # mean over all dims
            roi_vals.append(avg)
        return np.array(roi_vals)  # shape: (n_subs, n_freqs)


    target_roi_avg = extract_itpc_avg(target_itpc_dict)
    distractor_roi_avg = extract_itpc_avg(distractor_itpc_dict)

    from scipy.stats import ttest_rel


    def compare_itpc_conditions(target_roi_avg, distractor_roi_avg, n_perm=1000, seed=None):
        """
        Performs a paired t-test and permutation test between target and distractor ITPC.

        Parameters:
            target_roi_avg: np.ndarray of shape (n_subjects, n_freqs)
            distractor_roi_avg: np.ndarray of same shape
            n_perm: number of permutations for validation
            seed: random seed for reproducibility

        Returns:
            t_vals: array of t-values per frequency
            p_vals: array of p-values from t-test
            perm_p_vals: array of p-values from permutation test
        """
        rng = np.random.default_rng(seed)
        assert target_roi_avg.shape == distractor_roi_avg.shape
        n_subs, n_freqs = target_roi_avg.shape

        # Paired t-test
        t_vals, p_vals = ttest_rel(target_roi_avg, distractor_roi_avg, axis=0)

        # Permutation test
        observed_diff = np.mean(target_roi_avg - distractor_roi_avg, axis=0)
        perm_diffs = np.zeros((n_perm, n_freqs))

        for i in range(n_perm):
            # Randomly flip sign per subject (like a sign-flip permutation test)
            flips = rng.choice([1, -1], size=n_subs)
            perm_diff = np.mean((target_roi_avg - distractor_roi_avg) * flips[:, np.newaxis], axis=0)
            perm_diffs[i] = perm_diff

        # Two-tailed p-values
        perm_p_vals = np.mean(np.abs(perm_diffs) >= np.abs(observed_diff), axis=0)

        return t_vals, p_vals, perm_p_vals


    t_vals, p_vals, perm_p_vals = compare_itpc_conditions(target_roi_avg, distractor_roi_avg, n_perm=10000, seed=42)

    # Print or plot results
    for i, (t, p, pp) in enumerate(zip(t_vals, p_vals, perm_p_vals), start=1):
        print(f"Freq {i} Hz: t = {t:.3f}, p = {p:.4f}, perm p = {pp:.4f}")


    # --- Helper: Cohen's d for paired samples ---
    def cohens_d(x, y):
        diff = x - y
        return np.mean(diff) / np.std(diff, ddof=1)


    # --- Compute stats ---
    freqs = np.arange(1, 9)
    target_mean = np.mean(target_roi_avg, axis=0)
    distractor_mean = np.mean(distractor_roi_avg, axis=0)
    target_sem = np.std(target_roi_avg, axis=0, ddof=1) / np.sqrt(target_roi_avg.shape[0])
    distractor_sem = np.std(distractor_roi_avg, axis=0, ddof=1) / np.sqrt(distractor_roi_avg.shape[0])

    t_vals, p_vals = ttest_rel(target_roi_avg, distractor_roi_avg)
    cohens_d_vals = np.array([cohens_d(target_roi_avg[:, i], distractor_roi_avg[:, i]) for i in range(8)])

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    colors = ['#1f77b4', '#ff7f0e']

    plt.errorbar(freqs, target_mean, yerr=target_sem, label='Target', fmt='-o', capsize=5, color=colors[0], lw=2)
    plt.errorbar(freqs, distractor_mean, yerr=distractor_sem, label='Distractor', fmt='-o', capsize=5, color=colors[1],
                 lw=2)

    # Annotate significance + Cohen's d
    for i, p in enumerate(p_vals):
        if p < 0.05:
            ymax = max(target_mean[i] + target_sem[i], distractor_mean[i] + distractor_sem[i])
            label = f"* (d = {cohens_d_vals[i]:.2f})"
            plt.text(freqs[i], ymax + 0.002, label, ha='center', va='bottom', fontsize=11, color='black')

    # --- Formatting ---
    plt.xticks(freqs)
    plt.xlabel('Frequency (Hz)', fontsize=12)
    plt.ylabel('Mean ITPC', fontsize=12)
    plt.title('ITPC: Target vs. Distractor with Effect Size', fontsize=14)
    plt.legend(frameon=False)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.ylim([min(distractor_mean - distractor_sem) - 0.002, max(target_mean + target_sem) + 0.008])
    plt.tight_layout()
    plt.savefig(f'C:/Users/pppar/ITC_{plane}.png', dpi=300)
    plt.show()

    import seaborn as sns
    import pandas as pd
    import matplotlib.pyplot as plt

    # Prepare DataFrame
    df = pd.DataFrame({
        "Frequency": np.tile(freqs, target_roi_avg.shape[0] * 2),
        "ITPC": np.concatenate([target_roi_avg.flatten(), distractor_roi_avg.flatten()]),
        "Condition": ["Target"] * target_roi_avg.size + ["Distractor"] * distractor_roi_avg.size,
        "Subject": np.repeat(np.arange(target_roi_avg.shape[0]), 8 * 2)
    })

    # Set plot style
    sns.set(style="whitegrid", context="talk", font_scale=1.1)

    # Create figure
    plt.figure(figsize=(10, 5))
    palette = {"Target": "#1f77b4", "Distractor": "#ff7f0e"}

    # Boxplot
    sns.boxplot(
        data=df, x="Frequency", y="ITPC", hue="Condition",
        palette=palette, linewidth=1.5, width=0.6, fliersize=0
    )

    # Stripplot (aka swarm-like subject dots)
    sns.stripplot(
        data=df, x="Frequency", y="ITPC", hue="Condition",
        palette=palette, dodge=True, jitter=0.15, alpha=0.4, linewidth=0.5, edgecolor='gray'
    )

    # Format
    plt.title("Subject-wise ITPC Distributions per Frequency", fontsize=14)
    plt.xlabel("Frequency (Hz)", fontsize=12)
    plt.ylabel("ITPC", fontsize=12)
    plt.xticks(freqs)
    plt.grid(axis='y', linestyle='--', alpha=0.5)

    # Fix duplicate legend
    handles, labels = plt.gca().get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    plt.legend(by_label.values(), by_label.keys(), frameon=False, bbox_to_anchor=(1.02, 1), loc='upper left')

    plt.tight_layout()
    plt.show()



