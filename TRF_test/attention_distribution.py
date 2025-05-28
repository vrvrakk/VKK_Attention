from pathlib import Path
import os
import mne
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import mtrf
from mtrf import TRF
from mtrf.stats import crossval
from TRF_predictors.overlap_ratios import load_eeg_files
from scipy.signal import welch
import copy
from TRF_test.TRF_test_config import frontal_roi
from joblib import Parallel, delayed
from tqdm import tqdm
import psutil
import scipy
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

''' A script to get dynamic attentional predictor per condition (azimuth vs elevation) - with stacked predictors - 5 per stream
semantic_weights, overlap_ratios, events_proximity_pre, events_proximity_post'''


def load_eeg(plane):
    model_input_path = default_path / 'data' / 'eeg' / 'trf' / 'model_inputs'
    for folders in model_input_path.iterdir():
        if plane in folders.name:
            for files in folders.iterdir():
                if 'eeg_all' in files.name:
                    eeg_all = np.load(files)
    return eeg_all

def load_model_inputs(plane_raw, array_type1='', array_type2=''):
  model_input_path = default_path / 'data' / 'eeg' / 'trf' / 'model_inputs'
  target_pred_array = None
  distractor_pred_array = None
  for folders in model_input_path.iterdir():
    if plane_raw in folders.name:
        for files in folders.iterdir():
            if f'{plane_raw}__pred_{array_type1}' in files.name and 'npz' in files.name:
                target_pred_array = np.load(files)
            if f'{plane_raw}__pred_{array_type2}' in files.name and 'npz' in files.name:
                distractor_pred_array = np.load(files)
  if target_pred_array is None or distractor_pred_array is None:
    raise FileNotFoundError("One or both predictor arrays could not be found. Check filenames or paths.")
  return target_pred_array, distractor_pred_array

if __name__ == '__main__':

    print("Available CPUs:", os.cpu_count())
    print(f"Free RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")

    default_path = Path.cwd()
    predictors_path = default_path / 'data/eeg/predictors'
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'
    sfreq = 125

    plane = 'elevation'
    stream_type1 = 'target_stream'
    stream_type2 = 'distractor_stream'

    eeg_all = load_eeg(plane=plane)
    target_pred_array, distractor_pred_array = load_model_inputs(plane_raw='elevation_raw', array_type1=f'{stream_type1}', array_type2=f'{stream_type2}')
    list(target_pred_array.keys())
    # Define order to ensure consistency
    ordered_keys = ['onsets', 'envelopes', 'overlap_ratios',
                    'events_proximity_pre', 'events_proximity_post']

    # Stack predictors for the target stream
    X_target = np.column_stack([target_pred_array[k] for k in ordered_keys])

    # Stack predictors for the distractor stream
    X_distractor = np.column_stack([distractor_pred_array[k] for k in ordered_keys])

    print("X_target shape:", X_target.shape)
    print("X_distractor shape:", X_distractor.shape)

    eeg_all = eeg_all.T

    # checking collinearity:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    stream = input('Select stream (target/distractor): ')

    if stream == 'target':
        X = pd.DataFrame(
            X_target, columns=[f'{k}' for k in ordered_keys]
    )
    elif stream == 'distractor':
        # Build combined DataFrame
        X = pd.DataFrame(X_distractor, columns=[f'{k}' for k in ordered_keys])


    # Add constant for VIF calculation
    X = sm.add_constant(X)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    print(vif)

    # if all good, create gamma distribution array: same len as eeg
    # Parameters
    a = 2.0
    b = 1.0
    stim_samples = 93
    sfreq = 125
    duration_sec = stim_samples / sfreq
    onsets = X['onsets']
    for index, values in enumerate(onsets):
        if onsets[index] == 5:
            onsets[index] = 1
        elif onsets[index] == 4:
            onsets[index] = 0.85
        elif onsets[index] == 3:
            onsets[index] = 0.65
        elif onsets[index] == 2:
            onsets[index] = 0.45
        elif onsets[index] == 1:
            onsets[index] = 0.25

    gamma = scipy.stats.gamma(a=a, scale=b)  # Shape α=2, scale β=1 -> what would make sense potentially
    # (α); controls the skewness of the curve.
    # (β); spreads the curve along the x-axis.
    # a bell-shaped, right-skewed curve resembling a quick rise and slower decay
    stim_duration_samples = 93  # 745ms at 125Hz
    total_len = onsets.shape[0]
    # Extend x-range to see full gamma shape
    x = np.linspace(0, 6, stim_samples)

    # Later, each gamma is scaled by the stimulus weight (1–4), so normalization is crucial.

    def build_attention_predictor(onsets, overlap_ratios, proximity_pre, proximity_post, base_shape=2.5, spread_weight=0.5, amplitude_weight=0.3, shift_scale=10, global_delay=0, stim_samples=93):
        attention_predictor = np.zeros(len(onsets))
        x = np.linspace(0, 6, stim_samples) # will have the len of 93 samples
        # attentional blink duration = 500ms


        gamma_cache = {}
        global_delay = int(round(global_delay))
        for i in range(len(onsets) - stim_samples - global_delay):
            weight = onsets[i]  # semantic weight (rescaled)
            if weight == 0:
                continue

            overlap = overlap_ratios[i]
            pre = proximity_pre[i]
            post = proximity_post[i]

            # Shape modulation
            spread_factor = (pre + post) / 2  # range 0–1
            shape = base_shape * (1 - spread_weight * spread_factor)  # reduce shape by up to 50%
            shape = np.clip(shape, 1.2, 6.5)

            # Optional: reduce amplitude slightly under crowding
            amplitude = weight * (1 - abs(overlap)) * (1 - amplitude_weight * spread_factor)
            # Shift gamma peak based on overlap direction
            shift = int(np.round(np.sign(overlap) * abs(overlap) * shift_scale))  # shift ±samples (e.g., up to ±10)

            # Shift the gamma curve
            shape_key = round(shape, 2)
            if shape_key not in gamma_cache:
                gamma_curve = scipy.stats.gamma(a=shape, scale=1.0).pdf(x)
                gamma_curve /= gamma_curve.max()
                gamma_cache[shape_key] = gamma_curve
            else:
                gamma_curve = gamma_cache[shape_key]

            # Shift the curve in time
            shifted_curve = np.roll(gamma_curve, shift)
            shifted_curve[:max(0, shift)] = 0
            shifted_curve[-max(0, -shift):] = 0

            modulated = shifted_curve * amplitude
            attention_predictor[i:i + stim_samples] += modulated

        return attention_predictor

    # modify attention gamma dist array:
    overlap_ratios = X['overlap_ratios']
    proximity_pre = X['events_proximity_pre']
    # events proximity post
    proximity_post = X['events_proximity_post']

    from mtrf import TRF
    from mtrf.stats import crossval

    def run_trf_model(X, Y):
        regularization = 1.0
        trf = TRF(direction=1)
        trf.train(X, Y, regularization=regularization, fs=sfreq, tmin=-0.1, tmax=1.0, seed=42)
        prediction, r = trf.predict(X, Y)
        return r, prediction


    import time


    def trf_score_loss(params, onsets, overlap_ratios, proximity_pre, proximity_post, eeg):
        start = time.time()

        base_shape, spread_weight, amplitude_weight, shift_scale,  global_delay = params
        predictor = build_attention_predictor(
            onsets, overlap_ratios, proximity_pre, proximity_post,
            base_shape=base_shape,
            spread_weight=spread_weight,
            amplitude_weight=amplitude_weight,
            shift_scale=shift_scale,
            global_delay=global_delay,
            stim_samples=93
        )
        r, prediction = run_trf_model(predictor, eeg)

        end = time.time()
        print(f"Params: {params}, -r: {-r:.4f}, Time: {end - start:.2f}s")
        return -r
        # By returning -r, you trick minimize() into maximizing r instead


    from scipy.optimize import differential_evolution

    bounds = [(1.2, 4.0), (0.0, 1.0), (0.0, 1.0), (5, 15), (0.0, 50)]
    # global delay of 50=400 ms
    Y = eeg_all

    res = differential_evolution(
        func=trf_score_loss,
        bounds=bounds,
        args=(onsets, overlap_ratios, proximity_pre, proximity_post, Y),
        strategy='best1bin',
        maxiter=20,
        disp=True
    )

    # it will test within these ranges, until it finds the best combination that yields the highest r

    print("Best parameters:", res.x)
    print("Best score (r):", -res.fun)

    # Define a time range to visualize (e.g., first 3000 samples = 24 seconds at 125Hz)
    def plot_gammas(predictor):
        start = 100000
        end = 102000
        time = np.arange(start, end) / 125.0  # seconds

        plt.figure(figsize=(12, 6))
        plt.plot(time, predictor[start:end], label=f'Attention Predictor ({stream})', color='red')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.title('Attention Predictor (Gamma-based)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # Show non-blocking plot
        plt.show(block=False)
        plt.pause(0.1)  # Let it render
        input("Press Enter to close plot...")
        plt.close()


    plot_gammas(attention_predictor)
    plt.close('all')

    # Build combined DataFrame
    X_gamma = pd.DataFrame(
        np.column_stack([
            attention_predictor,
        ]),
        columns=[f'{stream}_gamma_distributions']
    )

    # Z-score predictors
    scaler = StandardScaler()
    # z = (x - u) / s
    X_gamma_scaled_np = scaler.fit_transform(X_gamma)
    X_gamma_scaled = pd.DataFrame(X_gamma_scaled_np, columns=X_gamma.columns)


    # VIF checker
    def check_collinearity(df):
        X = sm.add_constant(df)
        vif = pd.Series(
            [variance_inflation_factor(X.values, i) for i in range(X.shape[1])],
            index=X.columns
        )
        print(vif)
        return vif

    # orthogonalize:
    model = LinearRegression()
    model.fit(X_gamma_scaled[['target_gamma_distributions']], X_gamma_scaled['distractor_gamma_distributions'])
    distractor_ortho = X_gamma_scaled['distractor_gamma_distributions'] - model.predict(X_gamma_scaled[['target_gamma_distributions']])

    # Replace distractor column with orthogonalized version
    X_gamma_scaled['distractor_gamma_distributions'] = distractor_ortho

    # Check VIF
    vif = check_collinearity(X_gamma_scaled)
    print(f"EEG shape: {eeg_all.shape}, Predictors shape: {X_gamma_scaled.shape}")

    def save_attention_predictors(df, plane, stream_type1, stream_type2):
        save_path = default_path / 'data' / 'eeg' / 'trf' / 'trf_testing' / 'attentional_predictor' / plane
        save_path.mkdir(parents=True, exist_ok=True)

        filename = f"{stream_type1}_{stream_type2}.npz"

        np.savez(
            save_path / filename,
            target_attention=df['target_gamma_distributions'].values,
            distractor_attention=df['distractor_gamma_distributions'].values,
            sfreq=sfreq,
            stim_duration_samples=93,
            stream1=stream_type1,
            stream2=stream_type2,
            plane=plane
        )
        print(f"Saved attention predictors to: {save_path / filename}")


    # Call save
    save_attention_predictors(
        X_gamma_scaled,
        plane=plane,
        stream_type1=stream_type1,
        stream_type2=stream_type2
    )