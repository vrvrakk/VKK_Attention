from pathlib import Path
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import mtrf
from mtrf import TRF
from mtrf.stats import crossval
from TRF_predictors.overlap_ratios import load_eeg_files
from TRF_test.TRF_test_config import frontal_roi
import psutil


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


def pick_channels(eeg_files):
    eeg_concat_list = {}

    for sub, sub_list in eeg_files.items():
        if len(sub_list) > 0:
            eeg_concat = mne.concatenate_raws(sub_list)
            eeg_concat.resample(sfreq)
            eeg_concat.pick(frontal_roi)
            eeg_concat.filter(l_freq=None, h_freq=30)
            eeg_concat_list[sub] = eeg_concat
    return eeg_concat_list


def mask_bad_segmets(eeg_concat_list, condition):
    eeg_clean_list = {}
    eeg_masked_list = {}
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
                    eeg_masked = eeg_data[:, good_samples]
                    # z-scoring..
                    eeg_clean = (eeg_masked - eeg_masked.mean(axis=1, keepdims=True)) / eeg_masked.std(axis=1, keepdims=True)
                    print(f"{sub}: Clean EEG shape = {eeg_clean.shape}")
                    eeg_clean_list[sub] = eeg_clean
                    eeg_masked_list[sub] = eeg_masked
                    break
        else:
            print(f"No bad segments found for {sub} {condition}, assuming all samples are good.")
            eeg_len = eeg_concat.n_times
            good_samples = np.ones(eeg_len, dtype=bool)
            eeg_masked = eeg_data[:, good_samples]
            # z-scoring..
            eeg_clean = (eeg_masked - eeg_masked.mean(axis=1, keepdims=True)) / eeg_masked.std(axis=1, keepdims=True)
            print(f"{sub}: Clean EEG shape = {eeg_clean.shape}")
            eeg_masked_list[sub] = eeg_masked
            eeg_clean_list[sub] = eeg_clean
    return eeg_clean_list, eeg_masked_list


def remap_onsets_nested(predictor_dict):
    remapped_dict = {}
    for subj, stream_dict in predictor_dict.items():
        remapped_streams = {}
        for stream_key, arr in stream_dict.items():
            remapped_arr = arr.copy()
            for orig_val, new_val in semantic_mapping.items():
                remapped_arr[arr == orig_val] = new_val
            remapped_streams[stream_key] = remapped_arr
        remapped_dict[subj] = remapped_streams
    return remapped_dict


def centering_predictor_array(predictor_array, min_std=1e-6, predictor_name=''):
    """
    Normalize predictor arrays for TRF modeling.

    Rules:
    - 'envelopes' → z-score if dense and std > min_std.
    - All other predictors → left unchanged.
    - Fully zero predictors → returned unchanged.
    """

    if predictor_name == 'envelopes':
        std = predictor_array.std()
        nonzero_ratio = np.count_nonzero(predictor_array) / len(predictor_array)

        if nonzero_ratio > 0.5:
            print(f'{predictor_name}: Dense predictor, applying z-score.')
            mean = predictor_array.mean()
            return (predictor_array - mean) / std if std > min_std else predictor_array - mean

        elif nonzero_ratio > 0:
            print(f'{predictor_name}: Sparse predictor, mean-centering non-zero entries.')
            mask = predictor_array != 0
            mean = predictor_array[mask].mean()
            predictor_array = predictor_array.copy()
            predictor_array[mask] -= mean
            return predictor_array

        else:
            print(f'{predictor_name}: All zeros, returning unchanged.')
            return predictor_array

    else:
        print(f'{predictor_name}: No normalization applied.')
        return predictor_array


def get_predictor_dict(condition='', pred_type=''):
    predictor_dict = {}
    for files in predictor.iterdir():
        if 'sub' in files.name:
            sub_name = files.name  # e.g., "sub01"
            stream1_data, stream2_data = None, None
            for file in files.iterdir():
                if condition in file.name:
                    for stim_type in file.iterdir():
                        if stim_type.name == stream_type1:
                            for array in stim_type.iterdir():
                                if 'concat' in array.name:
                                    print(array)
                                    stream1_data = np.load(array)
                                    stream1_data = stream1_data[f'{pred_type}']
                        elif stim_type.name == stream_type2:
                            for array in stim_type.iterdir():
                                if 'concat' in array.name:
                                    stream2_data = np.load(array)
                                    stream2_data = stream2_data[f'{pred_type}']

            if stream1_data is not None and stream2_data is not None:
                predictor_dict[sub_name] = {
                    'stream1': stream1_data,
                    'stream2': stream2_data
                }
                print(f"Loaded predictors for {sub_name}: {stream1_data.shape}, {stream2_data.shape}")
            else:
                print(f"Missing predictor(s) for {sub_name} {condition}")
    return predictor_dict

def define_streams_dict(predictors1, predictors2):
    for pred_type1, pred_dict1 in predictors1.items():
        for sub, sub_dict in pred_dict1.items():
            sub_dict[f'{stim1}'] = sub_dict.pop('stream1')  # pop to replace OG array, not add extra array with new key
            sub_dict[f'{stim2}'] = sub_dict.pop('stream2')

    for pred_type2, pred_dict2 in predictors2.items():
        for sub, sub_dict in pred_dict2.items():
            sub_dict[f'{stim1}'] = sub_dict.pop('stream2')
            sub_dict[f'{stim2}'] = sub_dict.pop('stream1')
    return predictors1, predictors2


def predictor_mask_bads(predictor_dict, condition, predictor_name=''):
    predictor_dict_masked = {}
    predictor_dict_masked_raw = {}

    for sub, sub_dict in predictor_dict.items():
        sub_bad_segments_path = predictors_path / 'bad_segments' / sub / condition
        good_samples = None  # default fallback

        # --- Load bad segment mask if available
        if sub_bad_segments_path.exists():
            for file in sub_bad_segments_path.iterdir():
                if 'concat.npy' in file.name:
                    bad_segments = np.load(file)
                    bad_series = bad_segments['bad_series']
                    good_samples = bad_series == 0  # Boolean mask
                    print(f"Loaded bad segments for {sub} - {condition}.")
                    break  # Stop after finding the correct file

        sub_masked = {}
        sub_masked_raw = {}

        for stream_name, stream_array in sub_dict.items():
            if good_samples is not None and len(good_samples) == len(stream_array):
                stream_array_masked = stream_array[good_samples]
                stream_array_clean = centering_predictor_array(stream_array_masked, min_std=1e-6, predictor_name=predictor_name)
            else:
                stream_array_masked = stream_array  #full array if no mask found or mismatched
                stream_array_clean = centering_predictor_array(stream_array_masked, min_std=1e-6, predictor_name=predictor_name)
            sub_masked[stream_name] = stream_array_clean
            sub_masked_raw[stream_name] = stream_array_masked

        predictor_dict_masked[sub] = sub_masked
        predictor_dict_masked_raw[sub] = sub_masked_raw
    return predictor_dict_masked, predictor_dict_masked_raw



if __name__ == '__main__':

    # best lambda based on investigation of data and model testing:
    best_lambda = 1.0

    print("Available CPUs:", os.cpu_count())
    print(f"Free RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")

    default_path = Path.cwd()
    predictors_path = default_path / 'data/eeg/predictors'
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'
    sfreq = 125

    stream_type1 = 'stream1'
    stream_type2 = 'stream2'
    folder_type = 'all_stims'

    # stream_type1 = 'nt_target'
    # stream_type2 = 'nt_distractor'
    # folder_type = 'non_targets'

    # stream_type1 = 'targets'
    # stream_type2 = 'distractors'
    # folder_type = 'target_nums'

    # stream_type1 = 'targets'
    # stream_type2 = 'deviants'
    # folder_type = 'deviants'

    plane = 'elevation'

    if plane == 'elevation':
        condition1 = 'e1'
        condition2 = 'e2'
    elif plane == 'azimuth':
        condition1 = 'a1'
        condition2 = 'a2'


    eeg_files1 = get_eeg_files(condition=condition1)
    eeg_files2 = get_eeg_files(condition=condition2)


    eeg_concat_list1 = pick_channels(eeg_files1)
    eeg_concat_list2 = pick_channels(eeg_files2)

    eeg_clean_list_masked1, eeg_masked_list1 = mask_bad_segmets(eeg_concat_list1, condition=condition1)
    eeg_clean_list_masked2, eeg_masked_list2 = mask_bad_segmets(eeg_concat_list2, condition=condition2)

    predictors_list = ['binary_weights', 'envelopes', 'overlap_ratios', 'RTs']
    pred_types = ['onsets', 'envelopes', 'overlap_ratios', 'RT_labels']
    predictor_names = "_".join(pred_types)


    stim1 = 'target_stream'
    stim2 = 'distractor_stream'

    selected_streams = ['target_stream', 'distractor_stream']



    for selected_stream in selected_streams:
        if selected_stream == 'target_stream' and folder_type == 'deviants':
            continue
        print(f'Running TRF for {selected_stream} - {folder_type}...')
        s1_predictors = {}
        s2_predictors = {}

        s1_predictors_raw = {}
        s2_predictors_raw = {}

        # Mapping semantic weights

        semantic_mapping = {
            4.0: 1.0,
            3.0: 1.0,
            2.0: 1.0,
            1.0: 1.0,
        }


        for predictor_name, pred_type in zip(predictors_list, pred_types):
            predictor = default_path/ f'data/eeg/predictors/{predictor_name}'
            predictor_dict1 = get_predictor_dict(condition=condition1, pred_type=pred_type)
            predictor_dict2 = get_predictor_dict(condition=condition2, pred_type=pred_type)

            predictor_dict_masked1, predictor_dict_masked_raw1 = predictor_mask_bads(predictor_dict1, condition=condition1, predictor_name=pred_type)
            predictor_dict_masked2, predictor_dict_masked_raw2 = predictor_mask_bads(predictor_dict2, condition=condition2, predictor_name=pred_type)

            # Remap onsets (semantic weights) before storing
            if pred_type == 'onsets':
                predictor_dict_masked1 = remap_onsets_nested(predictor_dict_masked1)
                predictor_dict_masked2 = remap_onsets_nested(predictor_dict_masked2)
                predictor_dict_masked_raw1 =  remap_onsets_nested(predictor_dict_masked_raw1)
                predictor_dict_masked_raw2  = remap_onsets_nested(predictor_dict_masked_raw2)

            s1_predictors[pred_type] = predictor_dict_masked1
            s1_predictors_raw[pred_type] = predictor_dict_masked_raw1
            s2_predictors[pred_type] = predictor_dict_masked2
            s2_predictors_raw[pred_type] = predictor_dict_masked_raw2


        s1_predictors, s2_predictors = define_streams_dict(s1_predictors, s2_predictors)
        s1_predictors_raw, s2_predictors_raw = define_streams_dict(s1_predictors_raw, s2_predictors_raw)


        # Configuration
        fs = 125
        model_type = 1  # forward model
        lag_start = -0.1
        lag_end = 1.0
        lambda_val = 1.0

        # Select which s_predictors to use
        def run_trf(cond):
            assert cond in [condition1, condition2]
            if cond == condition1:
                s_predictors = s1_predictors  # change to s2_predictors for e2 and a2
                eeg_clean_list_masked = eeg_clean_list_masked1
            elif cond == condition2:
                s_predictors = s2_predictors
                eeg_clean_list_masked = eeg_clean_list_masked2

            # Save R-values
            predictor_short = "_".join([p[:2] for p in pred_types])  # e.g., 'on_env_rt_ovr'
            output_dir = default_path / f'data/eeg/trf/trf_testing/results/single_sub/{plane}/{cond}/{folder_type}/{predictor_short}'
            output_dir.mkdir(parents=True, exist_ok=True)


            weights_dir = output_dir / "weights"
            weights_dir.mkdir(parents=True, exist_ok=True)

            # Initialize storage for R-values
            subject_rvals = {}
            subject_crossval_rvals = {}

            # Extract subject list from one of the predictors
            subjects = ['sub10', 'sub11', 'sub13', 'sub14',
                        'sub15', 'sub17', 'sub18', 'sub19',
                        'sub20', 'sub21', 'sub22', 'sub23',
                        'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29']

            # Run model per subject
            for subject in subjects:
                print(f"Running composite TRF for {subject}, {selected_stream} - {cond}, {plane}...")

                predictor_arrays = []
                for pred_type in pred_types:
                    arr = s_predictors[pred_type][subject][selected_stream]
                    predictor_arrays.append(arr)

                # Stack predictors
                X = pd.DataFrame(np.column_stack(predictor_arrays), columns=pred_types)

                # Get EEG data for this subject
                eeg = eeg_clean_list_masked[subject].mean(axis=0)  # average over ROI channels
                print(eeg.shape, arr.shape)

                # Ensure EEG and predictor array lengths match
                if eeg.shape != arr.shape:
                    print(f'Shape mismatch: eeg x predictor array / {subject}')
                    min_len = min(len(eeg), len(X))
                    X = X[:min_len]
                    eeg = eeg[:min_len]
                else:
                    X = X
                    eeg = eeg

                # Run TRF model
                trf = TRF(direction=model_type)
                trf.train(X.values, eeg, fs=fs, tmin=lag_start, tmax=lag_end, regularization=best_lambda, seed=42)
                prediction, r = trf.predict(X.values, eeg)
                print(f'Corr value for {subject} TRF: {r}')

                # Split into two halves (or more)
                X1, X2 = np.array_split(X.values, 2)
                eeg1, eeg2 = np.array_split(eeg, 2)

                # Now pass as lists of arrays
                X_list = [X1, X2]
                eeg_list = [eeg1, eeg2]

                # Then run crossval
                r_crossval = crossval(trf, X_list, eeg_list, fs=fs, tmin=lag_start, tmax=lag_end, regularization=best_lambda, seed=42)
                print(f'Crossval R value for {subject} TRF: {r_crossval}')

                subject_rvals[subject] = r
                subject_crossval_rvals[subject] = r_crossval
                # Save TRF weights (time × predictors)
                np.save(weights_dir / f"{subject}_weights_{selected_stream}.npy", trf.weights)

                # Save time lags once (same for all)
                if subject == subjects[0]:
                    np.save(weights_dir / "trf_time_lags.npy", trf.times)

                # Optionally: Save metadata as a CSV row
                pred_save_path = weights_dir / 'predictions'
                filename = f"{subject}_prediction_{selected_stream}.npz"
                pred_save_path.mkdir(parents=True, exist_ok=True)
                np.savez(pred_save_path/filename,
                         prediction=prediction,
                         subject=subject,
                         stream=selected_stream,
                         plane=plane,
                         r_value=r,
                         r_crossval=r_crossval,
                         num_predictors=trf.weights.shape[0],
                         num_lags=trf.weights.shape[1])
                print(f'Saved {subject} predictions {filename} in : {pred_save_path} ')


            np.save(output_dir / f"subjectwise_rvals_{plane}_{selected_stream}_{folder_type}_{predictor_short}.npy", subject_rvals)
            np.save(output_dir / f"subjectwise_crossval_rvals_{plane}_{selected_stream}_{folder_type}_{predictor_short}.npy", subject_crossval_rvals)

            # Plot
            plt.ion()  # interactive mode on
            plt.figure(figsize=(10, 5))
            plt.bar(subject_rvals.keys(), subject_crossval_rvals.values(), color='slateblue')
            plt.xticks(rotation=45)
            plt.ylabel('Correlation (r)')
            plt.title(f'TRF Composite Model: {plane.capitalize()} - {cond} - {selected_stream.replace('_', ' ').capitalize()} - {folder_type.replace('_', ' ').capitalize()}')
            plt.grid(True)
            plt.tight_layout()
            fig_path = default_path / f'data/eeg/trf/trf_testing/results/single_sub/figures/{plane}/{cond}/{folder_type}/crossval_rvals'
            fig_path.mkdir(parents=True, exist_ok=True)
            plt.savefig(fig_path / f"subjectwise_trf_{plane}_{cond}_{selected_stream}_{predictor_short}.png", dpi=300)
            plt.show()

            print(f'Finished TRF analysis for all subs: {plane} - {cond} - {selected_stream} - {folder_type} - {predictor_short}')


        run_trf(cond=condition1)
        run_trf(cond=condition2)


    def plot_all_subject_weights(condition, selected_stream, plane=plane, folder_type='all_stims',
                                 pred_types=['onsets', 'envelopes']):
        import numpy as np
        import matplotlib.pyplot as plt

        predictor_short = "_".join([p[:2] for p in pred_types])
        weights_path = default_path / f'data/eeg/trf/trf_testing/results/single_sub/{plane}/{condition}/{folder_type}/{predictor_short}/weights'
        rvals_path = default_path / f"data/eeg/trf/trf_testing/results/single_sub/{plane}/{condition}/{folder_type}/{predictor_short}"
        rvals_file = rvals_path / f"subjectwise_crossval_rvals_{plane}_{selected_stream}_{folder_type}_{predictor_short}.npy"

        subjects = ['sub10', 'sub11', 'sub13', 'sub14',
                    'sub15', 'sub17', 'sub18', 'sub19',
                    'sub20', 'sub21', 'sub22', 'sub23',
                    'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29']

        # Load time lags (shared across subjects)
        time_lags = np.load(weights_path / "trf_time_lags.npy")

        # Collect weights
        weights_all = []
        for subject in subjects:
            weights_file = weights_path / f"{subject}_weights_{selected_stream}.npy"
            if weights_file.exists():
                weights = np.load(weights_file).squeeze(axis=-1)
                weights_all.append(weights)
            else:
                print(f"Missing weights for {subject}")
                continue

        weights_all = np.array(weights_all)  # shape: (n_subjects, n_predictors, n_time_lags)
        weights_all = weights_all[:, 1, :]  # select envelopes only (index 1)

        # Smooth each subject's envelope weights
        window_len = 11
        hamming_win = np.hamming(window_len)
        hamming_win /= hamming_win.sum()
        smoothed_weights = np.array([
            np.convolve(w, hamming_win, mode='same') for w in weights_all
        ])

        # Compute mean, SD, and SEM
        mean_trf = smoothed_weights.mean(axis=0)
        std_trf = smoothed_weights.std(axis=0)
        sem_trf = std_trf / np.sqrt(smoothed_weights.shape[0])

        # Load and compute mean r value
        if rvals_file.exists():
            subject_rvals = np.load(rvals_file, allow_pickle=True).item()
            mean_r = np.mean(list(subject_rvals.values()))
            r_label = f"Mean r = {mean_r:.3f}"
        else:
            r_label = "Mean r = N/A"
            print(f"Missing r-values file: {rvals_file}")

        # Plot
        plt.figure(figsize=(10, 5))
        plt.plot(time_lags, mean_trf, color='black', linewidth=2, label=f'Mean ({r_label})')
        plt.fill_between(time_lags, mean_trf - sem_trf, mean_trf + sem_trf,
                         color='gray', alpha=0.3, label='±1 SEM')
        plt.title(f'TRF Weights – {pred_types[1].capitalize()} – {plane}, {condition}, {selected_stream}')
        plt.xlabel('Time lag (s)')
        plt.ylabel('Weight')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()

        # Save figure
        fig_path = default_path / f'data/eeg/trf/trf_testing/results/single_sub/figures/{plane}/{condition}/{folder_type}'
        fig_path.mkdir(parents=True, exist_ok=True)
        plt.savefig(fig_path / f'{selected_stream}.png', dpi=300)
        plt.show()


    # Example usage:
    # for a1, e1
    def plot_trfs(cond):
        if cond in ['a1', 'e1']:
            if  folder_type != 'deviants':
                plot_all_subject_weights(condition=condition1, selected_stream='target_stream', plane=plane,
                                         folder_type=folder_type, pred_types=pred_types)

            plot_all_subject_weights(condition=condition1, selected_stream='distractor_stream', plane=plane,
                                     folder_type=folder_type, pred_types=pred_types)
        elif cond in ['a2', 'e2']:
            # for a2, e2
            if folder_type != 'deviants':
                plot_all_subject_weights(condition=condition2, selected_stream='target_stream', plane=plane,
                                         folder_type=folder_type, pred_types=pred_types)
            plot_all_subject_weights(condition=condition2, selected_stream='distractor_stream', plane=plane,
                                     folder_type=folder_type, pred_types=pred_types)


    plot_trfs(cond=condition1)
    plot_trfs(cond=condition2)
