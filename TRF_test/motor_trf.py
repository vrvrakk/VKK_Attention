from pathlib import Path
import matplotlib.pyplot as plt
import mne
import os
import numpy as np

default_path = Path.cwd()
motor_erp = default_path / 'data/eeg/preprocessed/results/erp'

for files in motor_erp.iterdir():
    if 'motor_smooth_erp-ave.fif' in files.name:
        motor_erp = mne.read_evokeds(files)
        motor_erp[0].resample(125)


rt_path = default_path / 'data/eeg/predictors/RTs'

condition = 'e2'
rt_targets_subs = {}
rt_distractors_subs = {}

def get_rt_pred(rt_subs_dict, stream_type):
    for sub in rt_path.iterdir():
        if 'sub' in sub.name:
            for sub_folder in sub.iterdir():
                if condition in sub_folder.name:
                    for files in sub_folder.iterdir():
                        if stream_type in files.name:
                            for np_files in files.iterdir():
                                if 'concat' in np_files.name:
                                    rt_targets = np.load(np_files)
                                    rt_targets = rt_targets['RTs']
                                    rt_subs_dict[sub.name] = rt_targets
    return rt_subs_dict


def count_nonzero_segments(array):
    count = 0
    in_segment = False
    for i in range(len(array)):
        if array[i] != 0 and not in_segment:
            in_segment = True
            count += 1
        elif array[i] == 0 and in_segment:
            in_segment = False
    return count


def mask_rt_arrays(rt_dict, bad_segments_dict):
    rt_masked = {}
    for sub_rt, rt_array in rt_dict.items():
        bad_array = bad_segments_dict[sub_rt]
        good_samples = bad_array == 0  # Boolean mask, keep zeros (-999 are bad segments)
        rt_masked[sub_rt] = rt_array[good_samples]
    return rt_masked


def concat_rt_arrays(rt_masked):
    rt_masked_list = [array for array in rt_masked.values()]
    rt_list = np.concatenate(rt_masked_list)
    print(len(rt_list))
    return rt_list


def assign_motor_erp(rt_targets_tagged, motor_pred, peak_sample=None):
    in_segment = False
    for i in range(len(rt_targets_tagged) - 1):
        current = rt_targets_tagged[i]
        next = rt_targets_tagged[i + 1]
        if current and not in_segment:
            in_segment = True
        if in_segment and current and not next:
            response_idx = i
            insert_idx = response_idx - peak_sample
            segment_dur = insert_idx + len(motor_data_avg)
            if insert_idx >= 0 and segment_dur <= len(motor_pred):
                motor_pred[insert_idx:segment_dur] += motor_data_avg
                # Using += ensures values get added instead of overwritten.
            else:
                samples_left = len(motor_pred) - insert_idx
                motor_pred[insert_idx:] += motor_data_avg[:samples_left]
        in_segment = False
    return motor_pred


rt_targets_subs = get_rt_pred(rt_targets_subs, stream_type='stream1')
rt_distractors_subs = get_rt_pred(rt_distractors_subs, stream_type='stream2')


# RTs are also based on sfreq of 125Hz
# remove bad segments:
bad_segments_path = default_path / 'data/eeg/predictors/bad_segments'
bad_segments_dict = {}
for sub in bad_segments_path.iterdir():
    for folders in sub.iterdir():
        if condition in folders.name:
            for files in folders.iterdir():
                if 'concat.npy' in files.name:
                    bad_segments = np.load(files)
                    bad_segments = bad_segments['bad_series']
                    bad_segments_dict[sub.name] = bad_segments


rt_targets_masked = mask_rt_arrays(rt_targets_subs, bad_segments_dict)
rt_distractors_masked = mask_rt_arrays(rt_distractors_subs, bad_segments_dict)


rt_targets_concat = concat_rt_arrays(rt_targets_masked)
rt_distractors_concat = concat_rt_arrays(rt_distractors_masked)

# get motor_erp data:
motor_data = motor_erp[0].get_data().T
motor_data_avg = np.mean(motor_data, axis=1)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 4), dpi=300)
# time_len = np.arange(len(motor_data_avg))
# plt.plot(time_len, motor_data_avg)
# plt.xlabel('Time samples')
# plt.ylabel('Amplitude (uV)')
# plt.title('Average motor ERP')
# plt.tight_layout()

# convolve rt_masked array with motor_erp
rt_targets_concat_tagged = rt_targets_concat != 0
rt_distractors_concat_tagged = rt_distractors_concat != 0

# assign motor_erps for True values:
motor_pred_target = np.zeros(len(rt_targets_concat_tagged))
motor_pred_distractor = np.zeros(len(rt_distractors_concat_tagged))


motor_pred_target = assign_motor_erp(rt_targets_concat_tagged, motor_pred_target, peak_sample=0)
motor_pred_distractor = assign_motor_erp(rt_distractors_concat_tagged, motor_pred_distractor, peak_sample=0)

n_insertions = np.count_nonzero(motor_pred_distractor) // 125
print(f"Motor ERP inserted {n_insertions} times in target predictor.")


# plt.figure(figsize=(12, 4), dpi=150)
# plt.plot(motor_pred_target)
# plt.xlabel("Time (samples)")
# plt.ylabel("Motor ERP Predictor")
# plt.title("Motor ERP Predictor for Distractor Responses")
# plt.tight_layout()
# plt.show()

# save files:
if condition in ['a1', 'e1']:
    target_stream = 'stream1'
    distractor_stream = 'stream2'
elif condition in ['a2', 'e2']:
    target_stream = 'stream2'
    distractor_stream = 'stream1'

save_path_target = rt_path / 'concatenated' / condition / target_stream
save_path_target.mkdir(parents=True, exist_ok=True)

save_path_distractor = rt_path / 'concatenated' / condition / distractor_stream
save_path_distractor.mkdir(parents=True, exist_ok=True)

target_filename = f'{condition}_{target_stream}_motor_pred_concat.npz'
np.savez(save_path_target / target_filename,
         motor=motor_pred_target)

distractor_filename = f'{condition}_{distractor_stream}_motor_pred_concat.npz'
np.savez(save_path_distractor/ distractor_filename,
         motor=motor_pred_distractor)

#############################################################
# now let's test directly here:

if __name__ == '__main__':
    def get_eeg_files(condition=''):
        eeg_files = {}
        for folders in eeg_results_path.iterdir():
            if 'sub' in folders.name:
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

    from TRF_test.TRF_test_config import frontal_roi
    def pick_channels(eeg_files):
        eeg_concat_list = {}

        for sub, sub_list in eeg_files.items():
            if len(sub_list) > 0:
                eeg_concat = mne.concatenate_raws(sub_list)
                eeg_concat.resample(sfreq)
                # eeg_concat.pick(frontal_roi)
                eeg_concat.filter(l_freq=None, h_freq=30)
                eeg_concat_list[sub] = eeg_concat
        return eeg_concat_list


    def mask_bad_segmets(eeg_concat_list, condition):
        eeg_clean_list = {}
        for sub in eeg_concat_list:
            eeg_concat = eeg_concat_list[sub]
            eeg_data = eeg_concat.get_data()

            sub_bad_segments_path = predictors_path / 'bad_segments' / sub / condition

            if sub_bad_segments_path.exists():
                for file in sub_bad_segments_path.iterdir():
                    if 'concat.npy' in file.name:
                        bad_segments = np.load(file)
                        bad_series = bad_segments['bad_series']
                        good_samples = bad_series == 0  # Boolean mask
                        print(f"Loaded bad segments for {sub} {condition}.")
                        eeg_clean = eeg_data[:, good_samples]
                        # z-scoring..
                        eeg_clean = (eeg_clean - eeg_clean.mean(axis=1, keepdims=True)) / eeg_clean.std(axis=1, keepdims=True)
                        print(f"{sub}: Clean EEG shape = {eeg_clean.shape}")
                        eeg_clean_list[sub] = eeg_clean
                        break
            else:
                print(f"No bad segments found for {sub} {condition}, assuming all samples are good.")
                eeg_len = eeg_concat.n_times
                good_samples = np.ones(eeg_len, dtype=bool)
                eeg_clean = eeg_data[:, good_samples]
                # z-scoring..
                eeg_clean = (eeg_clean - eeg_clean.mean(axis=1, keepdims=True)) / eeg_clean.std(axis=1, keepdims=True)
                print(f"{sub}: Clean EEG shape = {eeg_clean.shape}")
                eeg_clean_list[sub] = eeg_clean
        return eeg_clean_list


    def centering_predictor_array(predictor_array, min_std=1e-6, pred_type=''):
        """
        Normalize predictor arrays for TRF modeling.

        Rules:
        - 'envelopes', 'overlap_ratios', 'events_proximity', 'RTs' → z-score if std > min_std.
        - 'binary_weights' → leave unchanged (categorical codes: 0,1,2,3,4).
        - Sparse arrays (<50% non-zero values) → mean-center non-zeros only.
        """

        if pred_type == 'onsets' or pred_type == 'RT_labels': # do not normalize semantic weights arrays
            print("Predictor type is categorical (semantic_weights/RTs): skipping transformation.")
            return predictor_array

        std = predictor_array.std() # otherwise estimate STD and the non-zero vals ratio
        nonzero_ratio = np.count_nonzero(predictor_array) / len(predictor_array)

        if nonzero_ratio > 0.5:  # if non-zeros exceed 50% -> z-score
            # however, if std is close to 0: center the mean only
            # Dense predictor → full z-score
            print(f'{pred_type}: Dense predictor, applying z-score.')
            mean = predictor_array.mean()
            return (predictor_array - mean) / std if std > min_std else predictor_array - mean

        elif nonzero_ratio > 0:
            # Sparse → mean-center only non-zero values
            print(f'{pred_type}: Sparse predictor, mean-centering non-zero entries.')
            mask = predictor_array != 0
            mean = predictor_array[mask].mean()
            predictor_array[mask] -= mean
            return predictor_array

        else:
            # if all values are just zero: first of all, something is wrong with the array.
            print(f'{pred_type}: All zeros, returning unchanged.')
            return predictor_array


    predictors_path = default_path / 'data/eeg/predictors'
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'
    sfreq = 125

    eeg_files1 = get_eeg_files(condition='a1')
    eeg_files2 = get_eeg_files(condition='a2')
    plane = 'azimuth'

    eeg_concat_list1 = pick_channels(eeg_files1)
    eeg_concat_list2 = pick_channels(eeg_files2)

    eeg_clean_list_masked1 = mask_bad_segmets(eeg_concat_list1, condition='a1')
    eeg_clean_list_masked2 = mask_bad_segmets(eeg_concat_list2, condition='a2')

    all_eeg_clean1 = []
    all_eeg_clean2 = []

    for sub in eeg_clean_list_masked1.keys():
        eeg = eeg_clean_list_masked1[sub]
        all_eeg_clean1.append(eeg)

    for sub in eeg_clean_list_masked2.keys():
        eeg = eeg_clean_list_masked2[sub]
        all_eeg_clean2.append(eeg)

    all_eeg_clean = all_eeg_clean1 + all_eeg_clean2
    eeg_all = np.concatenate(all_eeg_clean, axis=1)
    eeg_all = eeg_all.T  # shape: (total_samples, channels)

    # get motor_pred:
    pred_path = rt_path / 'concatenated'
    for cond_fold in pred_path.iterdir():
        if plane == 'azimuth':
            if 'a1' in cond_fold.name:
                for stream_fold in cond_fold.iterdir():
                    if 'stream1' in stream_fold.name:
                        for files in stream_fold.iterdir():
                            stream1_array = np.load(files)
                            stream1_array = stream1_array['motor']
            elif 'a2' in cond_fold.name:
                for stream_fold in cond_fold.iterdir():
                    if 'stream2' in stream_fold.name:
                        for files in stream_fold.iterdir():
                            stream2_array = np.load(files)
                            stream2_array = stream2_array['motor']

    rt_array_all = np.concatenate((stream1_array, stream2_array))

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    # z-scoring each column independently:
    X_z = scaler.fit_transform(rt_array_all.reshape(-1, 1))
    # Each row is one sample, and the only column is your RT predictor
    Y_z = scaler.fit_transform(eeg_all)

    n_samples = sfreq * 60
    total_samples = len(rt_array_all)
    n_folds = total_samples // n_samples
    # Split predictors and EEG into subject chunks
    X_folds = np.array_split(X_z, n_folds)
    Y_folds = np.array_split(Y_z, n_folds)

    from mtrf import TRF
    from mtrf.stats import crossval

    lambdas = np.logspace(-2, 2, 20)  # based on prev literature
    scores = {}
    fwd_trf = TRF(direction=1)
    for l in lambdas:
        r = crossval(fwd_trf, X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=l)
        scores[l] = r

    best_r = np.max(list(scores.values()))
    best_lambda = [l for l, r in scores.items() if r == best_r][0]

    def plot_lambda_scores(scores):
        lambdas = list(scores.keys())
        performance = list(scores.values())
        plt.plot(lambdas, performance, marker='o')
        plt.xscale('log')
        plt.xlabel('Lambda (log scale)')
        plt.ylabel('Mean r')
        plt.title('TRF Performance vs. Lambda')
        plt.grid(True)
        plt.show()
        plt.savefig(save_path / f'{plane}_lambda_scores_RTs_motor.png')
        plt.close()


    trf = TRF(direction=1)

    trf.train(X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=best_lambda, seed=42)
    prediction, r = trf.predict(rt_array_all, eeg_all)
    print(f"Full model correlation: {r.round(3)}")

    weights = trf.weights  # shape: (n_features, n_lags, n_channels)
    time_lags = np.linspace(-0.1, 1.0, weights.shape[1])  # time axis

    # Loop and plot
    # Define your lag window of interest
    tmin_plot = 0.0
    tmax_plot = 1.0

    # Create a mask for valid time lags
    lag_mask = (time_lags >= tmin_plot) & (time_lags <= tmax_plot)
    time_lags_trimmed = time_lags[lag_mask]

    # Loop and plot
    predictor_name = 'RT_motor'
    save_path = default_path / f'data/eeg/trf/trf_testing/{predictor_name}/{plane}'
    save_path.mkdir(parents=True, exist_ok=True)

    plot_lambda_scores(scores)

    data_path = save_path / 'data'
    data_path.mkdir(parents=True, exist_ok=True)
    stim1 = 'stream1'
    stim2 = 'stream2'
    # Save TRF results for this condition
    # save both trf results and optimization results in one zip file
    np.savez(
        data_path / f'{plane}_{predictor_name}_{stim1}_{stim2}_TRF_results.npz',
        scores=scores,
        weights=weights,  # raw TRF weights (n_predictors, n_lags, n_channels)
        r=r,
        r_crossval=best_r,
        best_lambda=best_lambda,
        time_lags=time_lags,
        time_lags_trimmed=time_lags_trimmed,
        condition=plane
    )

    filename = f'{plane}_{predictor_name}_{stim1}_{stim2}_TRF_results.png'
    plt.figure(figsize=(8, 4))

    # Extract and reshape
    weights_2d = weights[0]  # shape: (n_lags, n_channels)
    trf_weights = weights_2d[lag_mask, :].T  # shape: (n_channels, selected_lags)

    # Smooth with Hamming window
    window_len = 11
    hamming_win = np.hamming(window_len)
    hamming_win /= hamming_win.sum()
    smoothed_weights = np.array([
        np.convolve(trf_weights[ch], hamming_win, mode='same')
        for ch in range(trf_weights.shape[0])
    ])

    # Plot all channels
    for ch in range(smoothed_weights.shape[0]):
        plt.plot(time_lags_trimmed, smoothed_weights[ch], alpha=0.4)

    # Plot formatting
    plt.title('TRF for Motor Responses')
    plt.xlabel('Time lag (s)')
    plt.ylabel('Amplitude')
    plt.plot([], [], ' ', label=f'λ = {best_lambda:.2f}, r = {best_r:.2f}')
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    # Save and show
    plt.savefig(save_path / filename, dpi=300)
    plt.show()
    plt.close()

    # Compute peak amplitude per channel (after smoothing)
    peak_amplitudes = np.max(np.abs(smoothed_weights), axis=1)

    # Get indices of top channels (e.g., top 5)
    top_channels_idx = np.argsort(peak_amplitudes)[-5:][::-1]

    # Step 1: Define EEG montage
    # Create mask to highlight top channels
    eeg_list1 = [values for values in eeg_concat_list1.values()]
    eeg_concat1 = mne.concatenate_raws(eeg_list1)
    eeg_info = eeg_concat1.info
    topo_data = np.max(np.abs(smoothed_weights), axis=1)
    ch_names = eeg_info['ch_names']
    top_ch_names = [ch_names[i] for i in top_channels_idx]
    print("Top TRF channels:", top_ch_names)
    mask = np.zeros_like(topo_data, dtype=bool)
    mask[top_channels_idx] = True

    # Plot topomap with highlighted top channels
    # Create the figure and axis first
    filename = f'{predictor_name}_{plane}_{stim1}_{stim2}.png'
    fig, ax = plt.subplots(figsize=(8, 6))
    # Pass the axis to MNE
    mne.viz.plot_topomap(
        topo_data, eeg_info, cmap='magma',
        mask=mask,
        mask_params=dict(marker='o', markersize=10, markerfacecolor='blue'),
        names=top_ch_names,
        axes=ax,  # <- use your axis
        show=False
    )
    # Create text string with channel names
    top_ch_text = '\n'.join([f'{i + 1}. {ch}' for i, ch in enumerate(top_ch_names)])

    # Add it as a text box in the figure (adjust x/y for placement)
    fig.text(0.85, 0.5, f'Top TRF channels:\n{top_ch_text}',
             fontsize=10, ha='left', va='center',
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))
    ax.set_title('Motor-ERP TRF Response Topomap')
    # Save the figure
    fig.savefig(save_path / filename, dpi=300)
    plt.close(fig)
