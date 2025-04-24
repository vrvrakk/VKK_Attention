from pathlib import Path
import os
import mne
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from TRF_predictors.overlap_ratios import load_eeg_files
from TRF_predictors.config import sub, sfreq, condition


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
    predictor_clean = weight_series_data[good_samples]   # now 1D: (good_samples,)
    std = predictor_clean.std()
    if std != 0:
        predictor_clean = (predictor_clean - predictor_clean.mean()) / predictor_clean.std()
    else:
        print(f"Predictor '{predictor_key}' has zero std (probably all zeros) — skipping z-score.")
        # Optionally leave it as is, or raise an error depending on your logic

    # z-scoring data...
    print(eeg_clean.shape)
    print(predictor_clean.shape)
    return eeg_clean, predictor_clean


def set_lags(tmin, tmax, predictor_clean, eeg_clean):
    lags = np.arange(round(tmin * sfreq), round(tmax * sfreq) + 1)
    time_lags_ms = lags * 1000 / sfreq  # convert to milliseconds
    n_lags = len(lags)
    n_samples = len(predictor_clean)
    X = np.zeros((n_samples, n_lags))

    for i, lag in enumerate(lags):
        if lag < 0:
            X[-lag:, i] = predictor_clean[:n_samples + lag]
        elif lag > 0:
            X[:n_samples - lag, i] = predictor_clean[lag:]
        else:
            X[:, i] = predictor_clean

    Y = eeg_clean.T  # shape = (samples, channels)
    return X, Y, time_lags_ms


def ridge_reg(alpha, X, Y):
    model = Ridge(alpha=alpha)
    trf_weights = []
    r2_scores = []

    for ch in range(Y.shape[1]):
        model.fit(X, Y[:, ch])
        trf_weights.append(model.coef_)
        r2_scores.append(r2_score(Y[:, ch], model.predict(X)))



    print("X shape:", X.shape)
    print("X mean:", X.mean())
    print("X std:", X.std())
    print("Unique values in X:", np.unique(X))
    return trf_weights, r2_scores


def plot_trf_results(trf_predictor1=None, trf_predictor2=None,
                     time_lags_ms1=None, time_lags_ms2=None, selected_predictor='',
                     r2_scores1=None, r2_scores2=None, stream_labels=('Stream 1', 'Stream 2'), stream_type=None):
    filename = f'{sub}_{condition}_{selected_predictor}_{stream_type}'

    # ====== Save TRF data ======
    trf_data_path = trf_path / 'trf_testing' / selected_predictor / sub / condition / 'data'
    trf_data_path.mkdir(parents=True, exist_ok=True)
    np.savez(
        trf_data_path / f'{filename}.npz',
        stream1_series=np.array(trf_predictor1) if trf_predictor1 else None,
        stream2_series=np.array(trf_predictor2) if trf_predictor2 else None,
        r2_scores1=r2_scores1,
        r2_scores2=r2_scores2,
        predictor=selected_predictor
    )

    # ====== Save Plot ======
    fig_path = trf_path / 'trf_testing' / selected_predictor / sub / condition / 'fig'
    fig_path.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 5))

    if trf_predictor1:
        trf_mean1 = np.array(trf_predictor1).mean(axis=0)
        plt.plot(time_lags_ms1, trf_mean1, label=stream_labels[0], linewidth=2)

    if trf_predictor2:
        trf_mean2 = np.array(trf_predictor2).mean(axis=0)
        plt.plot(time_lags_ms2, trf_mean2, label=stream_labels[1], linewidth=2)

    plt.axhline(0, color='gray', linestyle='--', linewidth=0.7)
    plt.axvline(0, color='black', linestyle='--', linewidth=0.7)
    plt.xlabel('Time Lag (ms)')
    plt.ylabel('TRF Weight')
    plt.title(f'TRF: {sub} - {condition} - {selected_predictor}')
    if trf_weights1 and trf_weights2:
        plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_path / f'{filename}.png', dpi=300)
    plt.show()
    plt.close()


def alpha_tuning(X, Y):
    alphas = [0.01, 0.1, 1, 10, 100]
    r2_scores_all = []
    save_path = trf_path / 'alpha_tuning' / sub / condition
    save_path.mkdir(parents=True, exist_ok=True)

    for a in alphas:
        trf_w, r2_s = ridge_reg(a, X, Y)  # stream 1 for example
        r2_scores_all.append(np.mean(r2_s))

    # Plot
    plt.plot(alphas, r2_scores_all, marker='o')
    plt.xscale('log')
    plt.xlabel('Alpha (log scale)')
    plt.ylabel('Mean R²')
    plt.title(f'Ridge Alpha Tuning: {sub} - {condition}')
    plt.grid(True)
    plt.savefig(save_path / f'{sub}_{condition}_{selected_predictor}_tuning.png', dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':
    frontal_roi = ['F3', 'Fz', 'F4', 'FC1', 'FC2']
    default_path = Path.cwd()
    eeg_path = default_path / 'data/eeg/preprocessed/results'
    predictors_path = default_path / 'data/eeg/predictors'
    trf_path = default_path / 'data/eeg/trf'

    predictors_list = ['bad_segments',
                       'binary_weights',
                       'envelopes',
                       'events_proximity',
                       'overlap_ratios',
                       'RTs']  # ALL PREDICTOR NAMES
    # choose a predictor
    # BAD SEGMENTS:
    predictor_name = predictors_list[0]

    eeg_files_list, eeg_events_list = load_eeg_files(sub=sub, condition=condition, sfreq=sfreq, results_path=eeg_path)
    eeg_concat = mne.concatenate_raws(eeg_files_list)
    eeg_concat.pick(frontal_roi)
    eeg_data = eeg_concat.get_data()
    sub_bad_segments_path = predictors_path / predictor_name / sub / condition

    for files in sub_bad_segments_path.iterdir():
        if 'concat' in files.name:
            bad_segments = np.load(files)

    bad_series = bad_segments['bad_series']
    good_samples = bad_series == 0  # getting good samples only

    selected_predictor = predictors_list[2]

    type1 = 'targets'
    type2 = 'deviants'

    weight_series1 = get_predictor_series(type1=type1, type2=None)
    weight_keys = list(weight_series1[0].keys())
    weight_series2 = get_predictor_series(type1=None, type2=type2)

    eeg_clean1, predictor_clean_onsets1 = filter_bad_segments(weight_series1[0], predictor_key=weight_keys[0])  # series1
    eeg_clean2, predictor_clean_onsets2 = filter_bad_segments(weight_series2[1], predictor_key=weight_keys[0])  # series2

    tmin = -0.1
    tmax = 0.8

    X1, Y1, time_lags_ms1 = set_lags(tmin, tmax, predictor_clean_onsets1, eeg_clean1)
    X2, Y2, time_lags_ms2 = set_lags(tmin, tmax, predictor_clean_onsets2, eeg_clean2)

    alpha = 0.1

    trf_weights1, r2_scores1 = ridge_reg(alpha, X1, Y1)
    trf_weights2, r2_scores2 = ridge_reg(alpha, X2, Y2)
    stim_types = ['all_stim', 'target_nums', 'non_targets', 'deviants']
    stim_type = stim_types[3]
    stream_type = f'{stim_type}_{selected_predictor}'
    plot_trf_results(trf_predictor1=trf_weights1, trf_predictor2=trf_weights2,
                     time_lags_ms1=time_lags_ms1, time_lags_ms2=time_lags_ms2,
                     selected_predictor=selected_predictor,
                     r2_scores1=r2_scores1, r2_scores2=r2_scores2, stream_type=stream_type)
    alpha_tuning(X1, Y1)
    alpha_tuning(X2, Y2)


# todo: adjust weights for nt distractors to 1, targets 4, distractors 3, deviants 2 and nt targets 2
