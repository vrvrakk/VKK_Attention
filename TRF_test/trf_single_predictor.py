from pathlib import Path
import os
import mne
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
from mtrf.model import TRF
from mtrf.stats import crossval
from TRF_predictors.overlap_ratios import load_eeg_files
from TRF_predictors.config import sfreq, condition


def get_predictor_series(type1=None, type2=None):
    weight_series1, weight_series2 = None, None
    if type1 is not None:
        chosen_predictor_path1 = predictors_path / selected_predictor / sub / condition / type1
        for item in chosen_predictor_path1.iterdir():
            if 'concat' in item.name:
                weight_series1 = np.load(item)
                break
        if weight_series1 is None:
            print(f'No concatenated series found in: {chosen_predictor_path1}')

    if type2 is not None:
        chosen_predictor_path2 = predictors_path / selected_predictor / sub / condition / type2
        for item in chosen_predictor_path2.iterdir():
            if 'concat' in item.name:
                weight_series2 = np.load(item)
                break
        if weight_series2 is None:
            print(f'No concatenated series found in: {chosen_predictor_path2}')

    return weight_series1, weight_series2


def filter_bad_segments(weight_series, predictor_key=None):
    weight_series_data = weight_series[predictor_key]
    eeg_clean = eeg_data[:, good_samples]       # still 2D: (n_channels, good_samples)
    eeg_clean = (eeg_clean - eeg_clean.mean(axis=1, keepdims=True)) / eeg_clean.std(axis=1, keepdims=True)
    predictor_clean = weight_series_data[good_samples]   # now 1D: (good_samples,)
    predictor_clean = (predictor_clean - predictor_clean.mean()) / predictor_clean.std()
    # z-scoring data...
    print(eeg_clean.shape)
    print(predictor_clean.shape)
    return eeg_clean, predictor_clean


def get_series(selected_predictor, type1='', type2=''):
    weight_series1 = get_predictor_series(type1=type1, type2=None)
    weight_keys = list(weight_series1[0].keys())
    weight_series2 = get_predictor_series(type1=None, type2=type2)
    return weight_series1, weight_series2, weight_keys


def orthogonalize(target, reference):
    """Orthogonalize 'target' with respect to 'reference'."""
    proj = np.dot(target, reference) / np.dot(reference, reference) * reference
    residual = target - proj
    return residual


def optimize_lambda(predictor, eeg, fs, tmin, tmax, lambdas):
    scores = []
    fwd_trf = TRF(direction=1)
    for l in lambdas:
        r = crossval(fwd_trf, predictor, eeg, fs, tmin, tmax, l)
        scores.append(r.mean())
    best_idx = np.argmax(scores)
    best_lambda = lambdas[best_idx]
    print(f"Best lambda: {best_lambda:.2e} (mean r = {scores[best_idx]:.3f})")
    return best_lambda


def plot_trf(fwd_trf, stream_type='', regularization=None):
    # plot:
    filename = f'{sub}_{condition}_{stream_type}'
    fig_path = trf_path / 'trf_testing' / selected_predictor / sub / condition / 'fig'
    fig_path.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(2)
    fwd_trf.plot(feature=0, axes=ax[0], show=False)
    fwd_trf.plot(channel='gfp', axes=ax[1], show=False)
    # Add a dummy line to create a legend
    ax[0].plot([], [], ' ', label=f'λ = {int(np.round(regularization))}')
    # Show the legend
    ax[0].legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_path / f'{filename}.png', dpi=300,  bbox_inches='tight')
    plt.close()


def save_trf_results(sub, condition, results, type1, type2):
    """
    Save TRF results for one subject to CSV and optionally npz.

    Parameters
    ----------
    sub : str
        Subject ID (e.g., 'sub01').
    condition : str
        Experimental condition (e.g., 'a1').
    results_list : list of dicts
        List of results, one per stream.
    save_dir : Path
        Directory where results will be saved.
    """
    save_dir = default_path / f'data/eeg/trf/trf_testing/{selected_predictor}/{sub}/{condition}/data'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Save as CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv(save_dir / f'{sub}_{condition}_{type1}_{type2}_{key_name}_trf_results.csv', index=False)

if __name__ == '__main__':
    frontal_roi = ['F3', 'Fz', 'F4', 'FC1', 'FC2']
    # predictors_list = ['binary_weights', 'envelopes', 'events_proximity', 'overlap_ratios', 'RTs']
    predictors_list = ['envelopes']

    default_path = Path.cwd()
    eeg_path = default_path / 'data/eeg/preprocessed/results'
    predictors_path = default_path / 'data/eeg/predictors'
    trf_path = default_path / 'data/eeg/trf'

    if condition in ['a1', 'a2']:
        from TRF_test.TRF_test_config import azimuth_subs
        subs = azimuth_subs
    elif condition in ['e1', 'e2']:
        from TRF_test.TRF_test_config import elevation_subs
        subs = elevation_subs

    if condition in ['a1', 'e1']:
        stream1_types = ['stream1', 'targets', 'nt_target', 'targets']
        stream2_types = ['stream2', 'distractors', 'nt_distractor', 'deviants']
    elif condition in ['a2', 'e2']:
        stream2_types = ['stream2', 'targets', 'nt_target', 'targets']
        stream1_types = ['stream1', 'distractors', 'nt_distractor', 'deviants']

    # --- Load EEG ---
    for type1, type2 in zip(stream1_types, stream2_types):
        for sub in subs:
            eeg_files_list, eeg_events_list = load_eeg_files(sub=sub, condition=condition, sfreq=sfreq, results_path=eeg_path)
            eeg_concat = mne.concatenate_raws(eeg_files_list)
            eeg_concat.pick(frontal_roi)
            eeg_data = eeg_concat.get_data()

            # --- Load bad segments ---
            sub_bad_segments_path = predictors_path / 'bad_segments' / sub / condition
            bad_segments_found = False
            if sub_bad_segments_path.exists():
                for file in sub_bad_segments_path.iterdir():
                    if 'concat' in file.name:
                        bad_segments = np.load(file)
                        bad_segments_found = True
                        bad_series = bad_segments['bad_series']
                        good_samples = bad_series == 0  # good samples only
                        print(f"Loaded bad segments for {sub} {condition}.")
                        break  # stop after finding the file
            else:
                print(f"No bad segments found for {sub} {condition}, assuming all samples are good.")
                # Create "fake good samples" (all good)
                eeg_len = eeg_concat.n_times
                good_samples = np.ones(eeg_len, dtype=bool)

            # --- Load predictors ---
            tmin, tmax = 0, 0.8  # range of time lag
            fs = 125
            lambdas = np.logspace(1, 8, 20)  # Test 20 lambdas between 10¹ and 10⁸
            n_trials = 5

            for i, predictors in enumerate(predictors_list):
                selected_predictor = predictors_list[0]
                if selected_predictor == 'RTs' and type2 != 'distractors':
                    continue
                weight_series1, weight_series2, weight_keys = get_series(selected_predictor, type1=type1, type2=type2)
                # Special handling for 'events_proximity'
                if selected_predictor == 'events_proximity':
                    key_indices = [0, 1]  # both pre and post
                else:
                    key_indices = [0]  # only pre (normal case)

                for key_idx in key_indices:
                    eeg_clean1, predictor_clean_onsets1 = filter_bad_segments(weight_series1[0], predictor_key=weight_keys[key_idx])

                    eeg_clean1 = eeg_clean1.T  # transpose to (samples, n_ch)
                    eeg_clean2, predictor_clean_onsets2 = filter_bad_segments(weight_series2[1], predictor_key=weight_keys[key_idx])
                    key_name = weight_keys[key_idx]

                    eeg_clean2 = eeg_clean2.T  # transpose to (samples, n_ch)
                    # --- Orthogonalize stream predictor ---
                    predictor_ortho_onsets1 = orthogonalize(predictor_clean_onsets1, predictor_clean_onsets2)
                    predictor_ortho_onsets2 = orthogonalize(predictor_clean_onsets2, predictor_clean_onsets1)

                    # Then split
                    predictor_trials1 = np.array_split(predictor_ortho_onsets1, n_trials)
                    eeg_trials1 = np.array_split(eeg_clean1, n_trials)

                    regularization1 = optimize_lambda(predictor_trials1, eeg_trials1, fs, tmin, tmax, lambdas)
                    fwd_trf1 = TRF(direction=1)
                    fwd_trf1.train(predictor_ortho_onsets1, eeg_clean1, fs=fs, tmin=tmin, tmax=tmax, regularization=regularization1)
                    r_crossval1 = crossval(fwd_trf1, predictor_trials1, eeg_trials1, fs, tmin, tmax, regularization1)
                    print(f"mean correlation between actual and predicted response: {r_crossval1.mean().round(3)}")

                    prediction1, r_fwd1 = fwd_trf1.predict(predictor_ortho_onsets1, eeg_clean1)
                    print(f"correlation between actual and predicted response: {r_fwd1.round(3)}")

                    fwd_trf2 = TRF(direction=1)
                    # Then split
                    predictor_trials2 = np.array_split(predictor_ortho_onsets2, n_trials)
                    eeg_trials2 = np.array_split(eeg_clean2, n_trials)

                    regularization2 = optimize_lambda(predictor_trials2, eeg_trials2, fs, tmin, tmax, lambdas)
                    fwd_trf2.train(predictor_ortho_onsets2, eeg_clean2, fs=fs, tmin=tmin, tmax=tmax, regularization=regularization2)
                    # overfitting
                    r_crossval2 = crossval(fwd_trf2, predictor_trials2, eeg_trials2, fs, tmin, tmax, regularization2)
                    print(f"mean correlation between actual and predicted response: {r_crossval2.mean().round(3)}")

                    prediction2, r_fwd2 = fwd_trf2.predict(predictor_clean_onsets2, eeg_clean2)
                    print(f"correlation between actual and predicted response: {r_fwd2.round(3)}")

                    stream_types = [f'{type1}', f'{type2}']
                    stream_name1 = stream_types[0]
                    stream_name2 = stream_types[1]
                    stream_type1 = f'{stream_name1}_{key_name}'
                    stream_type2 = f'{stream_name2}_{key_name}'
                    plot_trf(fwd_trf1, stream_type=stream_type1, regularization=regularization1)
                    plot_trf(fwd_trf2, stream_type=stream_type2, regularization=regularization2)
                    results_list = [
                        {
                            'subject': sub,
                            'condition': condition,
                            'stream': 'stream1',
                            'predictor': selected_predictor,
                            'key': key_name,
                            'best_lambda': regularization1,
                            'mean_crossval_r': r_crossval1.mean(),
                            'full_prediction_r': r_fwd1
                        },
                        {
                            'subject': sub,
                            'condition': condition,
                            'stream': 'stream2',
                            'predictor': selected_predictor,
                            'key': key_name,
                            'best_lambda': regularization2,
                            'mean_crossval_r': r_crossval2.mean(),
                            'full_prediction_r': r_fwd2
                        }
                    ]
                    save_trf_results(sub, condition, results_list, type1, type2)

