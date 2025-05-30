import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
from pathlib import Path
from mtrf import TRF

def main(stream1, stream2, stims, plane, pred):
    default_path = Path.cwd()

    data_path1 = default_path / f'data/eeg/trf/trf_testing/composite_model/{plane}/{stims}/data/{plane}_{type}_{stream1}_TRF_results.npz'
    data_path2 = default_path / f'data/eeg/trf/trf_testing/composite_model/{plane}/{stims}/data/{plane}_{type}_{stream2}_TRF_results.npz'

    # Example: Load TRF results for azimuth plane
    target_data = np.load(data_path1, allow_pickle=True)
    distractor_data = np.load(data_path2, allow_pickle=True)
    list(target_data.keys())

    # --- Extract ---
    preds = target_data['preds'].tolist()
    time_lags = target_data['time_lags']  # shape: (n_lags,)
    tmin_plot = 0.1
    tmax_plot = 0.5
    lag_mask = (time_lags >= tmin_plot) & (time_lags <= tmax_plot)
    time_lags_trimmed = time_lags[lag_mask]

    # --- Envelope TRFs ---
    pred_idx = preds.index(f'{pred}_target_stream')
    target_weights = target_data['weights'][pred_idx].T[:, lag_mask]  # (channels, lags)
    distractor_weights = distractor_data['weights'][pred_idx].T[:, lag_mask]

    def get_weight_idx(pred):
        if pred == 'onsets':
            weight_idx = 0
        elif pred == 'envelopes':
            weight_idx = 1
        elif pred == 'overlap_ratios':
            weight_idx = 2
        elif pred == 'events_proximity_pre':
            weight_idx = 3
        elif pred == 'events_proximity_post':
            weight_idx = 4
        elif pred == 'RT_labels':
            weight_idx = 5
        return weight_idx


    weight_idx = get_weight_idx(pred)


    # --- Smooth ---
    def smooth_channels(weights, window_len=11):
        hamming_win = np.hamming(window_len)
        hamming_win /= hamming_win.sum()
        return np.array([
            np.convolve(weights[ch], hamming_win, mode='same')
            for ch in range(weights.shape[0])
        ])

    target_smoothed = smooth_channels(target_weights)
    distractor_smoothed = smooth_channels(distractor_weights)

    # --- Average across channels ---
    target_avg = target_smoothed.mean(axis=0)
    distractor_avg = distractor_smoothed.mean(axis=0)
    diff_wave = target_avg - distractor_avg

    #### comparisons stats #####

    import pandas as pd
    from scipy.stats import ttest_rel

    folder_type = 'all_stims'
    selected_stream1 = 'target_stream'
    selected_stream2 = 'distractor_stream'
    # Load metadata for subject matching
    metadata_path1 = default_path / f'data/eeg/trf/trf_testing/composite_model/single_sub/{plane}/{folder_type}/weights/metadata_{selected_stream1}.csv'
    df_meta1 = pd.read_csv(metadata_path1)

    metadata_path2 = default_path / f'data/eeg/trf/trf_testing/composite_model/single_sub/{plane}/{folder_type}/weights/metadata_{selected_stream2}.csv'
    df_meta2 = pd.read_csv(metadata_path2)

    # Drop exact duplicate rows (same values in all columns)
    df_meta1 = df_meta1.drop_duplicates()
    df_meta2 = df_meta2.drop_duplicates()
    # Ensure subjects with both streams exist
    subjects = sorted(df_meta1['subject'].unique()) # same in both df

    target_vals, distractor_vals = [], []
    target_rms_vals, distractor_rms_vals = [], []

    for subject in subjects:
        weights_dir1 = default_path / f'data/eeg/trf/trf_testing/composite_model/single_sub/{plane}/{folder_type}/weights'
        weights_path1 = weights_dir1 / f'{subject}_weights_{selected_stream1}.npy'

        weights_dir2 = default_path / f'data/eeg/trf/trf_testing/composite_model/single_sub/{plane}/{folder_type}/weights'
        weights_path2 = weights_dir1 / f'{subject}_weights_{selected_stream2}.npy'

        if not weights_path1.exists():
            print(f"Missing weights for {subject}")
            continue
        if not weights_path2.exists():
            print(f"Missing weights for {subject}")
            continue

        weights1 = np.load(weights_path1)  # (n_preds, n_lags, 1)
        weights2 = np.load(weights_path2)

        # Extract the predictor of interest (e.g., RT_labels)
        trf1 = weights1[weight_idx, :, 0]  # shape: (139,)
        trf2 = weights2[weight_idx, :, 0]
        # Apply lag mask
        trf_window1 = trf1[lag_mask]
        trf_window2 = trf2[lag_mask]

        # Optional smoothing
        window_len = 11
        hamming_win = np.hamming(window_len)
        hamming_win /= hamming_win.sum()
        trf_smoothed1 = np.convolve(trf_window1, hamming_win, mode='same')
        trf_smoothed2 = np.convolve(trf_window2, hamming_win, mode='same')

        # Compute AUC for the full TRF window
        rms_val1 = np.sqrt(np.mean(trf_smoothed1 ** 2))  # target stream
        rms_val2 = np.sqrt(np.mean(trf_smoothed2 ** 2))  # distractor stream

        # Metric: mean or peak amplitude
        # Metric: peak (absolute) amplitude in the window
        # Find peak index in target TRF
        # Use fixed latency window for attention (e.g., 150–300 ms)
        window_mask = (time_lags_trimmed >= 0.2) & (time_lags_trimmed <= 0.30)

        # Mean TRF amplitude in that window
        metric_val1 = trf_smoothed1[window_mask].max()  # target stream
        metric_val2 = trf_smoothed2[window_mask].max()  # distractor stream

        # Add to correct group
        subj_meta1 = df_meta1[df_meta1['subject'] == subject]
        if subj_meta1.empty:
            continue

        subj_meta2 = df_meta2[df_meta2['subject'] == subject]
        if subj_meta2.empty:
            continue

        stream1 = subj_meta1['stream'].values[0]
        stream2 = subj_meta2['stream'].values[0]
        if stream1 == 'target_stream':
            target_vals.append(metric_val1)
            target_rms_vals.append(rms_val1)
        if stream2 == 'distractor_stream':
            distractor_vals.append(metric_val2)
            distractor_rms_vals.append(rms_val2)

    # Convert to arrays
    target_vals = np.array(target_vals)
    distractor_vals = np.array(distractor_vals)

    # Ensure same size
    min_len = min(len(target_vals), len(distractor_vals))
    target_vals = target_vals[:min_len]
    distractor_vals = distractor_vals[:min_len]

    target_rms_vals = np.array(target_rms_vals)
    distractor_rms_vals = np.array(distractor_rms_vals)

    min_len_rms = min(len(target_rms_vals), len(distractor_rms_vals))
    target_rms_vals = target_rms_vals[:min_len_rms]
    distractor_rms_vals = distractor_rms_vals[:min_len_rms]
    t_stat_rms, p_val_rms = ttest_rel(target_rms_vals, distractor_rms_vals)
    print(f"t (RMS) = {t_stat_rms:.3f}, p = {p_val_rms:.4f}")


    # Paired t-test
    t_stat, p_val = ttest_rel(target_vals, distractor_vals)
    # p < 0.05
    print(f"\nPaired t-test (Target vs. Distractor for {preds[pred_idx]}):")
    print(f"t = {t_stat:.3f}, p = {p_val:.4f}")

    # Cohen's dz for amplitude
    diff_amp = target_vals - distractor_vals
    cohens_dz_amp = np.mean(diff_amp) / np.std(diff_amp, ddof=1)

    # Cohen's dz for RMS
    diff_rms = target_rms_vals - distractor_rms_vals
    cohens_dz_rms = np.mean(diff_rms) / np.std(diff_rms, ddof=1)

    print(f"Cohen's dz (Amplitude): {cohens_dz_amp:.3f}")
    print(f"Cohen's dz (RMS): {cohens_dz_rms:.3f}")


    ######
    # --- Plot ---
    if plane == 'azimuth':
        r = 0.1
    else:
        r = 0.07
    plt.figure(figsize=(10, 6))
    plt.plot(time_lags_trimmed, target_avg, label='Target', linewidth=2)
    plt.plot(time_lags_trimmed, distractor_avg, label='Distractor', linewidth=2)
    # plt.plot(time_lags_trimmed, diff_wave, label='Target - Distractor', linestyle='--', color='black')
    plt.axhline(0, color='gray', linestyle=':')
    plt.xlabel('Time Lag (s)')
    plt.ylabel('TRF Amplitude')
    plt.title(f'TRF Comparison for {pred.capitalize()} Predictor ({plane.capitalize()})')
    textstr = (
        f"Target Peak: {target_avg.max():.3f} uV @ {time_lags_trimmed[np.argmax(target_avg)]:.3f} s\n"
        f"Distractor Peak: {distractor_avg.max():.3f} uV @ {time_lags_trimmed[np.argmax(distractor_avg)]:.3f} s\n"
        f"p (Amplitude): {p_val:.3f}\n"
        f"Cohen's dz (Amp): {cohens_dz_amp:.2f}\n"
        f"RMS Target: {np.mean(target_rms_vals):.3f} uV\n"
        f"RMS Distractor: {np.mean(distractor_rms_vals):.3f} uV\n"
         f"Cohen's dz (RMS): {cohens_dz_rms:.2f}\n"
        f"p (RMS): {p_val_rms:.3f}\n"
        f"r (Pearson): {r}\n")

    # Add text box to plot
    plt.gca().text(
        0.65, 0.95, textstr,
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8)
    )
    plt.legend()
    plt.tight_layout()
    plt.show()
    save_path = default_path / f'data/eeg/trf/trf_comparison/{plane}/{stims}'
    save_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path/f'{pred}.png')

if __name__ == '__main__':
    plane1 = 'azimuth'
    plane2 = 'elevation'
    stims = 'all_stims'
    type = 'all'
    stream1 = 'target'
    stream2 = 'distractor'
    pred = 'envelopes'
    main(stream1=stream1, stream2=stream2, stims=stims, plane=plane1, pred=pred)
    main(stream1=stream1, stream2=stream2, stims=stims, plane=plane2, pred=pred)