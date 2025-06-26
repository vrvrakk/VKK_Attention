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
        itc_crop = tfr[1].copy().crop(tmin=0.0, tmax=0.3)
        itcs[sub] = itc_crop  # index 1 is ITC (index 0 is power)
        powers[sub] = tfr[0]
    return itcs, powers


def cohens_d_paired(x, y):
    diff = x - y
    return diff.mean() / diff.std(ddof=1)


# --- Collect z-scored power for all subjects ---
def z_scored_power(target_epochs, distractor_epochs):
    target_power = []
    distractor_power = []

    for sub in subs:
        targ_evoked = target_epochs[sub].average()
        targ_evoked = targ_evoked.get_data().mean(axis=0)
        dist_evoked = distractor_epochs[sub].average()
        dist_evoked = dist_evoked.get_data().mean(axis=0)

        power_freqs_t, targ_pow = compute_zscored_power(targ_evoked, sfreq, fmin, fmax)
        power_freqs_d, dist_pow = compute_zscored_power(dist_evoked, sfreq, fmin, fmax)

        target_power.append(targ_pow)
        distractor_power.append(dist_pow)

    target_power = np.array(target_power)
    distractor_power = np.array(distractor_power)
    return target_power, distractor_power, power_freqs_t, power_freqs_d


from matplotlib import colormaps as cm

def itc_vals(target_itc, distractor_itc, band=None, band_name='', cond='', resp=''):
    target_vals, distractor_vals = [], []

    for sub in subs:
        targ_data = target_itc[sub].data.mean(axis=(0, 2))
        dist_data = distractor_itc[sub].data.mean(axis=(0, 2))
        target_vals.append(targ_data)
        distractor_vals.append(dist_data)

    target_vals = np.array(target_vals)
    distractor_vals = np.array(distractor_vals)

    # Paired t-test and FDR correction
    t_vals, p_vals = ttest_rel(target_vals, distractor_vals, axis=0)
    _, p_fdr = fdrcorrection(p_vals)

    # Effect size (Cohen's d)
    effect_sizes = np.array([
        cohens_d_paired(target_vals[:, i], distractor_vals[:, i])
        for i in range(target_vals.shape[1])
    ])

    # Mean and SEM of difference
    mean_diff = target_vals.mean(axis=0) - distractor_vals.mean(axis=0)
    sem_diff = (target_vals - distractor_vals).std(axis=0, ddof=1) / np.sqrt(target_vals.shape[0])

    # Use modern color palette
    cmap = cm.get_cmap("tab10")
    itc_color = cmap(3)      # purple-blue
    cohens_d_color = cmap(0) # muted blue

    # --- Plotting ---
    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=300)
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 8,
        "axes.labelweight": "bold",
        "axes.titlesize": 10,
        "axes.titleweight": "bold"
    })

    # ITC difference plot
    ax1.plot(band, mean_diff, label='Target - Distractor', color=itc_color, linewidth=1.5)
    ax1.fill_between(band, mean_diff - sem_diff, mean_diff + sem_diff, color=itc_color, alpha=0.25)

    # Format y-axis
    y_margin = 0.01
    y_min = mean_diff.min() - y_margin
    y_max = mean_diff.max() + y_margin
    ax1.set_ylim(y_min, y_max)
    ax1.axhline(0, linestyle='--', color='gray', linewidth=0.7)
    ax1.set_ylabel('ITC Difference\n(Target − Distractor)', fontsize=8)
    ax1.set_xlabel('Frequency (Hz)', fontsize=8)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.3f}'))
    ax1.tick_params(axis='both', labelsize=7)
    ax1.grid(True, axis='y', linestyle='--', linewidth=0.3, alpha=0.4)

    # Effect size plot
    ax2 = ax1.twinx()
    ax2.plot(band, effect_sizes, linestyle='--', color=cohens_d_color, label="Cohen's d", linewidth=1)
    ax2.set_ylabel("Cohen's d", fontsize=8)
    ax2.tick_params(axis='y', labelsize=7)
    ax2.set_ylim(0, max(effect_sizes) + 0.1)

    # Significant region
    sig_mask = p_fdr < 0.05
    if np.any(sig_mask):
        sig_freqs = band[sig_mask]
        freq_range_str = f"{sig_freqs[0]:.1f}–{sig_freqs[-1]:.1f} Hz"
        ax1.axvspan(sig_freqs[0], sig_freqs[-1], color='gray', alpha=0.1, zorder=0)
        sig_label = f"\nSignificant Range: {freq_range_str} (p < 0.05)"
    else:
        sig_label = ""

    # Peak Cohen's d
    peak_freq = band[np.argmax(np.abs(effect_sizes))]
    ax1.axvline(peak_freq, color='gray', linestyle=':', linewidth=1, alpha=0.4)
    ax1.text(peak_freq, y_max, f'{peak_freq:.1f} Hz\n(peak d)', fontsize=6,
             color='gray', ha='center', va='top', alpha=0.6)

    # Title
    fig.suptitle(f'{plane.capitalize()} ITC Difference Across Frequencies{sig_label}', fontsize=10, weight='bold')

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc='upper right')

    plt.tight_layout()
    plt.show()

    # Save figure
    save_dir = fig_path / f'{plane}/{cond}/{folder_type}'
    save_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_dir / f'itc_diff_{band_name}_{resp}.png', dpi=300)

    return target_vals, distractor_vals, effect_sizes, p_fdr, peak_freq

def sub_level_itc(target_vals, distractor_vals, peak_freq, band=None, label_subjects=False, band_name='', cond='', resp=''):
    from matplotlib import colormaps as cm

    save_dir = fig_path / f'{plane}/{cond}/{folder_type}'
    save_dir.mkdir(parents=True, exist_ok=True)

    sns.set(style='whitegrid')
    plt.rcParams.update({
        'axes.titlesize': 10,
        'axes.labelsize': 8,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'legend.fontsize': 7,
        'font.family': 'DejaVu Sans'
    })

    num_subs = target_vals.shape[0]
    colors = cm.get_cmap('tab20')(np.linspace(0, 1, num_subs))

    # --- Identify significance range and peak freq ---
    from scipy.stats import ttest_rel
    from statsmodels.stats.multitest import fdrcorrection

    t_vals, p_vals = ttest_rel(target_vals, distractor_vals, axis=0)
    _, p_fdr = fdrcorrection(p_vals)
    sig_mask = p_fdr < 0.05
    peak_freq = peak_freq

    if np.any(sig_mask):
        sig_freqs = band[sig_mask]
        sig_range_str = f"{sig_freqs[0]:.1f}–{sig_freqs[-1]:.1f} Hz"
        sig_text = f"(p < 0.05: {sig_range_str})"
    else:
        sig_text = ""

    # --- Plot Target Stream ITC ---
    plt.figure(figsize=(8, 4), dpi=300)
    for i, sub_itc in enumerate(target_vals):
        smoothed = savgol_filter(sub_itc, window_length=7, polyorder=3)
        plt.plot(band, smoothed, color=colors[i], linewidth=1)
        if label_subjects:
            peak_idx = np.argmax(sub_itc)
            plt.text(band[peak_idx], sub_itc[peak_idx], f'{i+1}', fontsize=6, color=colors[i])
    mean_itc = savgol_filter(target_vals.mean(axis=0), window_length=7, polyorder=3)
    plt.plot(band, mean_itc, color='black', linewidth=2.5, label='Mean ITC')
    plt.axvline(x=peak_freq, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    if np.any(sig_mask):
        plt.axvspan(sig_freqs[0], sig_freqs[-1], color='gray', alpha=0.1)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('ITC (Target Stream)')
    plt.title(f'Target ITC Curves {sig_text}')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / f'subs_ITC_{resp}_{band_name}_target.png', dpi=300)
    plt.show()

    # --- Plot Distractor Stream ITC ---
    plt.figure(figsize=(8, 4), dpi=300)
    for i, sub_itc in enumerate(distractor_vals):
        smoothed = savgol_filter(sub_itc, window_length=7, polyorder=3)
        plt.plot(band, smoothed, color=colors[i], linewidth=1)
        if label_subjects:
            peak_idx = np.argmax(sub_itc)
            plt.text(band[peak_idx], sub_itc[peak_idx], f'{i+1}', fontsize=6, color=colors[i])
    mean_itc = savgol_filter(distractor_vals.mean(axis=0), window_length=7, polyorder=3)
    plt.plot(band, mean_itc, color='black', linewidth=2.5, label='Mean ITC')
    plt.axvline(x=peak_freq, color='gray', linestyle=':', linewidth=1, alpha=0.6)
    if np.any(sig_mask):
        plt.axvspan(sig_freqs[0], sig_freqs[-1], color='gray', alpha=0.1)

    plt.xlabel('Frequency (Hz)')
    plt.ylabel('ITC (Distractor Stream)')
    plt.title(f'Distractor ITC Curves {sig_text}')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / f'subs_ITC_{resp}_{band_name}_distractor.png', dpi=300)
    plt.show()

    # --- Violin Plot at Peak Frequency ---
    f_idx = np.argmin(np.abs(band - peak_freq))
    target_hz = target_vals[:, f_idx]
    distractor_hz = distractor_vals[:, f_idx]
    t_stat, p_val = ttest_rel(target_hz, distractor_hz)

    df = pd.DataFrame({
        'Target': target_hz,
        'Distractor': distractor_hz
    }).melt(var_name='Stream', value_name='ITC')

    # Compute stats
    mean_t = target_hz.mean()
    std_t = target_hz.std()
    mean_d = distractor_hz.mean()
    std_d = distractor_hz.std()

    # Plot
    plt.figure(figsize=(6, 5), dpi=300)
    ax = sns.violinplot(data=df, x='Stream', y='ITC', inner='point',
                        palette={'Target': 'steelblue', 'Distractor': 'lightcoral'})

    # Annotate statistical result
    y_max = max(df['ITC']) + 0.02
    if p_val < 0.001:
        stat_text = '*** p < 0.001'
    elif p_val < 0.01:
        stat_text = '** p < 0.01'
    elif p_val < 0.05:
        stat_text = '* p < 0.05'
    else:
        stat_text = f'n.s. (p = {p_val:.2f})'

    plt.text(0.5, y_max + 0.01, stat_text, ha='center', va='bottom', fontsize=9)

    # Format legend text
    legend_labels = [
        f"Target: {mean_t:.3f} ± {std_t:.3f}",
        f"Distractor: {mean_d:.3f} ± {std_d:.3f}"
    ]

    # Dummy handles for legend
    handles = [
        plt.Line2D([0], [0], color='steelblue', lw=4),
        plt.Line2D([0], [0], color='lightcoral', lw=4)
    ]
    plt.legend(handles, legend_labels, title='Mean ± SD', loc='upper right', frameon=False, fontsize=4, title_fontsize=6)

    plt.title(f'ITC at ~{band[f_idx]:.2f} Hz', fontsize=10)
    plt.ylabel('ITC')
    plt.xlabel('')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    plt.savefig(fig_path / f'ITC_{resp}_{band_name}_violinplot.png', dpi=300)
    plt.show()



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

    target_epochs = get_epochs(target_events1, eeg_files1)
    distractor_epochs = get_epochs(distractor_events1, eeg_files1)

    # Parameters
    itc_freqs = {'delta/theta': np.logspace(np.log10(4), np.log10(8), num=100)}

    target_power1, distractor_power1, power_freqs_t1, power_freqs_d1 = z_scored_power(target_epochs,
                                                                                      distractor_epochs)


    target_itc1_low, target_powers1_low = compute_itc(target_epochs, itc_freqs['delta/theta'], n_cycles=0.5 * itc_freqs['delta/theta'])

    distractor_itc1_low, distractor_powers1_low = compute_itc(distractor_epochs, itc_freqs['delta/theta'], n_cycles=0.5 * itc_freqs['delta/theta'])



    from matplotlib.ticker import FuncFormatter

    target_vals1_low, distractor_vals1_low, effect_sizes1_low, p_fdr1_low, peak_freq= itc_vals(target_itc1_low, distractor_itc1_low,
                                                                                                     band=itc_freqs['delta/theta'],
                                                                                                     band_name = 'delta_theta', cond=cond1)

    # Base save path
    itc_path = Path(f"C:/Users/pppar/PycharmProjects/VKK_Attention/data/eeg/trf/trf_testing/results/single_sub/ITC/{plane}/{cond1}")
    # Create directory if it doesn't exist
    os.makedirs(itc_path, exist_ok=True)

    # Save target  distractor stream ITC
    np.savez(os.path.join(itc_path, "itc_vals_all.npz"), itc_target=target_vals1_low,
             itc_distractor=distractor_vals1_low,
             effect_sizes=effect_sizes1_low,
             p_fdr=p_fdr1_low)


    print(f"Saved ITC files to {itc_path}")

    sub_level_itc(target_vals1_low, distractor_vals1_low, peak_freq, band=itc_freqs['delta/theta'], label_subjects=True, band_name='delta_theta', cond=cond1, resp='raw')
    import numpy as np
    import matplotlib.pyplot as plt
    from scipy.stats import ttest_rel
    from statsmodels.stats.multitest import fdrcorrection
    from matplotlib.ticker import FuncFormatter
    from matplotlib import colormaps as cm

    # === Inputs ===
    band = np.logspace(np.log10(4), np.log10(8), 100)  # Adjust if your actual band differs
    target_vals = target_vals1_low
    distractor_vals = distractor_vals1_low
    n_subs = target_vals.shape[0]

    # === Compute stats ===
    mean_target = target_vals.mean(axis=0)
    mean_distractor = distractor_vals.mean(axis=0)
    sem_target = target_vals.std(axis=0, ddof=1) / np.sqrt(n_subs)
    sem_distractor = distractor_vals.std(axis=0, ddof=1) / np.sqrt(n_subs)

    # Paired t-test
    t_vals, p_vals = ttest_rel(target_vals, distractor_vals, axis=0)
    _, p_fdr = fdrcorrection(p_vals)


    # Effect size
    def cohens_d(x, y):
        diff = x - y
        return diff.mean() / diff.std(ddof=1)


    effect_sizes = np.array([cohens_d(target_vals[:, i], distractor_vals[:, i]) for i in range(target_vals.shape[1])])

    # === Plotting ===
    fig, ax1 = plt.subplots(figsize=(7, 4), dpi=300)

    # Colors
    cmap = cm["tab10"]
    target_color = cmap(0)
    distractor_color = cmap(1)
    diff_color = cmap(2)
    effect_color = cmap(4)

    # Plot means ± SEM
    ax1.plot(band, mean_target, label='Target', color=target_color, linewidth=1.5)
    ax1.fill_between(band, mean_target - sem_target, mean_target + sem_target, alpha=0.2, color=target_color)

    ax1.plot(band, mean_distractor, label='Distractor', color=distractor_color, linewidth=1.5)
    ax1.fill_between(band, mean_distractor - sem_distractor, mean_distractor + sem_distractor, alpha=0.2,
                     color=distractor_color)

    # Axes formatting
    ax1.set_xlabel('Frequency (Hz)', fontsize=9)
    ax1.set_ylabel('ITC (Mean ± SEM)', fontsize=9)
    ax1.tick_params(labelsize=8)
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.3f}'))
    ax1.grid(True, linestyle='--', linewidth=0.3, alpha=0.4)

    # === Plot effect size ===
    ax2 = ax1.twinx()
    ax2.plot(band, effect_sizes, linestyle='--', linewidth=1.2, color=effect_color, label="Cohen's d")
    ax2.set_ylabel("Cohen's d", fontsize=9)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.set_ylim(0, np.nanmax(effect_sizes) + 0.1)

    # === Significant range annotation ===
    sig_mask = p_fdr < 0.05
    # Get the current axis limits
    y_min, y_max = ax1.get_ylim()
    x_min, x_max = ax1.get_xlim()

    # Annotate significant range ABOVE the plot (just outside the top edge)
    if np.any(sig_mask):
        sig_freqs = band[sig_mask]
        sig_range = f"{sig_freqs[0]:.1f}–{sig_freqs[-1]:.1f} Hz"
        ax1.axvspan(sig_freqs[0], sig_freqs[-1], color='gray', alpha=0.1, zorder=0)
        ax1.annotate(f'Significant: {sig_range} Hz',
                     xy=(sig_freqs[0], y_max),
                     xytext=(sig_freqs[0], y_max),
                     fontsize=7, ha='left', va='bottom',
                     color='gray', weight='bold',
                     annotation_clip=False)

    # Annotate peak frequency just below top edge, slightly staggered
    peak_idx = np.argmax(effect_sizes)
    peak_freq = band[peak_idx]
    ax1.axvline(peak_freq, color='gray', linestyle=':', alpha=0.5)
    ax2.grid(False)

    # === Legend and Title ===
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='upper right')
    fig.suptitle(f'{plane.capitalize()}\nITC Comparison Across Frequencies\n(Target vs. Distractor)', fontsize=10, weight='bold')

    plt.tight_layout()
    plt.savefig(fig_path/f'{plane}_itc_across_frequencies_t_vd_d.png', dpi=300)
    plt.show()

    # Extract the actual frequency array from the dictionary
    freqs = itc_freqs['delta/theta']


    # Get index of peak ITC per subject
    target_peak_indices = np.argmax(target_vals1_low, axis=1)  # shape: (18,)
    distractor_peak_indices = np.argmax(distractor_vals1_low, axis=1)  # shape: (18,)

    # Use indices to extract actual frequency values
    target_peak_freqs = freqs[target_peak_indices]
    distractor_peak_freqs = freqs[distractor_peak_indices]

    # Print results
    for i, (t_freq, d_freq) in enumerate(zip(target_peak_freqs, distractor_peak_freqs)):
        print(f"Subject {i + 1:02d} | Target peak: {t_freq:.2f} Hz | Distractor peak: {d_freq:.2f} Hz")


    def compute_tfr_complex(epochs_dict, freqs, n_cycles):
        tfrs = {}
        for sub, epochs in epochs_dict.items():
            tfr = epochs.compute_tfr(
                method="morlet",
                freqs=freqs,
                n_cycles=n_cycles,
                use_fft=True,
                return_itc=False,
                average=False,
                decim=1,
                n_jobs=1,
                output='complex'
            )
            tfrs[sub] = tfr
        return tfrs


    def phase_binning_analysis(tfrs, subject_freqs, phase_channel='FCz',
                               bin_count=6, stim_time=0.0, power_window=(0.0, 0.3)):
        all_bins = {}  # results per subject
        for sub, tfr in tfrs.items():
            freq_of_interest = subject_freqs[sub]
            freq_idx = np.argmin(np.abs(tfr.freqs - freq_of_interest))
            chan_idx = tfr.ch_names.index(phase_channel)
            t0_idx = np.argmin(np.abs(tfr.times - stim_time))
            start_idx = np.argmin(np.abs(tfr.times - power_window[0]))
            end_idx = np.argmin(np.abs(tfr.times - power_window[1]))

            # Extract phase and power
            complex_data = tfr.data[:, chan_idx, freq_idx, :]
            phase_at_0 = np.angle(complex_data[:, t0_idx])  # shape (n_epochs,)
            power = np.abs(complex_data[:, start_idx:end_idx]) ** 2
            mean_power = power.mean(axis=1)

            # Bin phases
            bin_edges = np.linspace(-np.pi, np.pi, bin_count + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_indices = np.digitize(phase_at_0, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, bin_count - 1)

            mean_per_bin = np.zeros(bin_count)
            counts_per_bin = np.zeros(bin_count)

            for i in range(bin_count):
                in_bin = bin_indices == i
                if np.any(in_bin):
                    mean_per_bin[i] = mean_power[in_bin].mean()
                    counts_per_bin[i] = in_bin.sum()

            all_bins[sub] = {
                'bin_centers': bin_centers,
                'mean_power_per_bin': mean_per_bin,
                'counts': counts_per_bin,
                'freq_used': tfr.freqs[freq_idx]
            }

        return all_bins

    # Step 1: Compute TFRs
    tfrs_target = compute_tfr_complex(target_epochs, freqs, n_cycles=0.5 * freqs)
    tfrs_distractor = compute_tfr_complex(distractor_epochs, freqs, n_cycles=0.5 * freqs)

    sub_ids = list(tfrs_target.keys())

    target_peak_freqs_dict = dict(zip(sub_ids, target_peak_freqs))
    distractor_peak_freqs_dict = dict(zip(sub_ids, distractor_peak_freqs))

    # === Step 3: Run phase binning with individual freq ===
    results_t = phase_binning_analysis(tfrs_target, target_peak_freqs_dict)
    results_d = phase_binning_analysis(tfrs_distractor, distractor_peak_freqs_dict)

    res_path = default_path / f'data/eeg/behaviour/figures/{plane}'

    # === Step 4: Plotting Attended ===
    plt.figure(figsize=(10, 5))
    for sub, res in results_t.items():
        plt.plot(res['bin_centers'], res['mean_power_per_bin'], marker='o', label=sub)
    plt.xlabel("Phase at 0 s (radians)")
    plt.ylabel("Mean Power (0–0.3 s)")
    plt.title("Attended Stream Phase-Binned Power per Subject")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(res_path/'target_phase_binned_power_subs.png', dpi=300)
    plt.show()

    # === Step 5: Plotting Unattended ===
    plt.figure(figsize=(10, 5))
    for sub, res in results_d.items():
        plt.plot(res['bin_centers'], res['mean_power_per_bin'], marker='o', label=sub)
    plt.xlabel("Phase at 0 s (radians)")
    plt.ylabel("Mean Power (0–0.3 s)")
    plt.title("Unattended Stream Phase-Binned Power per Subject")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(res_path/'distractor_phase_binned_power_subs.png', dpi=300)
    plt.show()
    plt.close()

    from scipy.optimize import curve_fit
    from scipy.stats import circmean, circstd, ttest_rel, zscore
    import matplotlib.pyplot as plt


    # Cosine function for fitting
    def cos_func(x, amp, phase, offset):
        return amp * np.cos(x - phase) + offset


    # Phase binning using per-subject frequency
    def phase_binning_analysis(tfrs, subj_freqs, phase_channel='FCz',
                               bin_count=6, stim_time=0.0, power_window=(0.0, 0.3)):
        all_bins = {}  # store results per subject
        for sub, tfr in tfrs.items():
            freq_of_interest = subj_freqs[sub]
            freq_idx = np.argmin(np.abs(tfr.freqs - freq_of_interest))
            chan_idx = tfr.ch_names.index(phase_channel)
            t0_idx = np.argmin(np.abs(tfr.times - stim_time))
            start_idx = np.argmin(np.abs(tfr.times - power_window[0]))
            end_idx = np.argmin(np.abs(tfr.times - power_window[1]))

            complex_data = tfr.data[:, chan_idx, freq_idx, :]
            phase_at_0 = np.angle(complex_data[:, t0_idx])  # (n_epochs,)
            power = np.abs(complex_data[:, start_idx:end_idx]) ** 2
            mean_power = power.mean(axis=1)

            bin_edges = np.linspace(-np.pi, np.pi, bin_count + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            bin_indices = np.digitize(phase_at_0, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, bin_count - 1)

            mean_per_bin = np.zeros(bin_count)
            counts_per_bin = np.zeros(bin_count)

            for i in range(bin_count):
                in_bin = bin_indices == i
                if np.any(in_bin):
                    mean_per_bin[i] = mean_power[in_bin].mean()
                    counts_per_bin[i] = in_bin.sum()

            all_bins[sub] = {
                'bin_centers': bin_centers,
                'mean_power_per_bin': mean_per_bin,
                'counts': counts_per_bin
            }

        return all_bins


    # Cosine fit per subject
    def analyze_phase_modulation(results_dict):
        preferred_phases = {}
        modulation_depths = {}
        zscored_power_bins = []

        for sub, res in results_dict.items():
            x = res['bin_centers']
            y = res['mean_power_per_bin']
            zscored_power_bins.append(zscore(y))

            try:
                popt, _ = curve_fit(cos_func, x, y, p0=[(max(y) - min(y)) / 2, 0, np.mean(y)])
                amp, phase, offset = popt
            except RuntimeError:
                amp, phase, offset = np.nan, np.nan, np.nan

            preferred_phases[sub] = phase
            modulation_depths[sub] = amp

        zscored_power_bins = np.array(zscored_power_bins)
        group_mean = zscored_power_bins.mean(axis=0)
        group_sem = zscored_power_bins.std(axis=0, ddof=1) / np.sqrt(zscored_power_bins.shape[0])

        return {
            "preferred_phases": preferred_phases,
            "modulation_depths": modulation_depths,
            "zscored_bins": zscored_power_bins,
            "group_mean": group_mean,
            "group_sem": group_sem,
            "bin_centers": x
        }


    # Group plot
    def plot_group_phase_modulation(group_data, condition_label):
        x = group_data["bin_centers"]
        y = group_data["group_mean"]
        sem = group_data["group_sem"]

        plt.figure(figsize=(8, 5))
        plt.errorbar(x, y, yerr=sem, fmt='o-', capsize=4)
        plt.xlabel("Phase at 0 s (radians)")
        plt.ylabel("Z-scored Mean Power (0–0.3s)")
        plt.title(f"Group Phase Modulation – {condition_label}")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(res_path / f'{condition_label}_group_phase_modulation.png', dpi=300)
        plt.show()


    # Paired comparison of cosine fits
    def compare_conditions(analysis_target, analysis_distractor):
        from scipy.stats import wilcoxon

        subs = set(analysis_target["preferred_phases"].keys()) & set(analysis_distractor["preferred_phases"].keys())
        target_phases = np.array([analysis_target["preferred_phases"][s] for s in subs])
        distractor_phases = np.array([analysis_distractor["preferred_phases"][s] for s in subs])

        phase_diff = np.angle(np.exp(1j * target_phases) / np.exp(1j * distractor_phases))

        target_mod = np.array([analysis_target["modulation_depths"][s] for s in subs])
        distractor_mod = np.array([analysis_distractor["modulation_depths"][s] for s in subs])
        _, p_mod = ttest_rel(target_mod, distractor_mod)

        print("Mean preferred phase (target):", circmean(target_phases, high=np.pi, low=-np.pi))
        print("Mean preferred phase (distractor):", circmean(distractor_phases, high=np.pi, low=-np.pi))
        print("Circular phase difference (target - distractor):")
        print("  Mean:", circmean(phase_diff, high=np.pi, low=-np.pi))
        print("  Std:", circstd(phase_diff, high=np.pi, low=-np.pi))
        print("Modulation depth: paired t-test p =", p_mod)

        return {
            "subjects": subs,
            "phase_diff": phase_diff,
            "target_mod": target_mod,
            "distractor_mod": distractor_mod,
            "mod_pval": p_mod
        }


    # Run binning using per-subject peak
    attended_results = phase_binning_analysis(tfrs_target, subj_freqs=target_peak_freqs_dict)
    unattended_results = phase_binning_analysis(tfrs_distractor, subj_freqs=distractor_peak_freqs_dict)

    # Cosine fits
    attended_analysis = analyze_phase_modulation(attended_results)
    unattended_analysis = analyze_phase_modulation(unattended_results)

    # Plots
    plot_group_phase_modulation(attended_analysis, "Attended")
    plot_group_phase_modulation(unattended_analysis, "Unattended")

    # Comparison stats
    stats = compare_conditions(attended_analysis, unattended_analysis)
    stats_path = default_path / f'data/eeg/trf/trf_testing/results/single_sub/ITC/{plane}/{cond1}'
    np.savez(stats_path/'phase_comparison.npz', stats)