from pathlib import Path
import os
import mne
import numpy as np
import pandas as pd
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
semantic_weights, envelopes, overlap_ratios, events_proximity_pre, events_proximity_post, (RTs)'''


def load_eeg(plane):
    model_input_path = default_path / 'data' / 'eeg' / 'trf' / 'model_inputs'
    for folders in model_input_path.iterdir():
        if plane in folders.name:
            for files in folders.iterdir():
                if 'eeg_all' in files.name:
                    eeg_all = np.load(files)
    return eeg_all

def load_model_inputs(plane_raw, array_type=''):
  model_input_path = default_path / 'data' / 'eeg' / 'trf' / 'model_inputs'
  target_pred_array = None
  distractor_pred_array = None
  for folders in model_input_path.iterdir():
    if plane_raw in folders.name:
        for files in folders.iterdir():
            if f'{plane_raw}_{array_type}_pred_target' in files.name and 'npz' in files.name:
                target_pred_array = np.load(files)
            if f'{plane_raw}_{array_type}_pred_distractor' in files.name and 'npz' in files.name:
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
    stream_type1 = 'stream1'
    stream_type2 = 'stream2'

    eeg_all = load_eeg(plane=plane)
    target_pred_array, distractor_pred_array = load_model_inputs(plane_raw='elevation_raw', array_type=f'{stream_type1}_{stream_type2}')

    # Define order to ensure consistency
    ordered_keys = ['onsets', 'overlap_ratios',
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

    # Build combined DataFrame
    X = pd.DataFrame(
        np.column_stack([
            X_target,  # target predictors
            X_distractor  # distractor predictors
        ]),
        columns=[f'{k}_target' for k in ordered_keys] + [f'{k}_distractor' for k in ordered_keys]
    )

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
    target_onsets = X_target[:, 0]
    distractor_onsets = X_distractor[:, 0]
    gamma = scipy.stats.gamma(a=a, scale=b)  # Shape α=2, scale β=1 -> what would make sense potentially
    # (α); controls the skewness of the curve.
    # (β); spreads the curve along the x-axis.
    # a bell-shaped, right-skewed curve resembling a quick rise and slower decay
    stim_duration_samples = 93  # 745ms at 125Hz
    total_len = target_onsets.shape[0]
    # Extend x-range to see full gamma shape
    x = np.linspace(0, 6, stim_samples)

    # Later, each gamma is scaled by the stimulus weight (1–4), so normalization is crucial.

    def add_gamma_to_predictor(onsets, overlap_ratios, proximity_pre, proximity_post, a_base=2.0, b=1.0, stim_samples=93):
        attention_predictor = np.zeros(len(onsets))
        x = np.linspace(0, 6, stim_samples)
        gamma_cache = {}

        for i in range(len(onsets) - stim_samples):
            weight = onsets[i]
            overlap = overlap_ratios[i]
            pre_score = proximity_pre[i]
            post_score = proximity_post[i]

            if weight > 0:
                abs_overlap = abs(overlap)
                amplitude = weight * (1 - abs_overlap)

                # Gamma shape: modulate by overlap direction and crowding
                shape = a_base
                if overlap < 0:
                    shape += abs_overlap * 2.0
                shape += (pre_score + post_score) * 1.5
                shape = round(shape, 2)
                shape = np.clip(shape, 1.2, 6.0)

                # Retrieve or compute gamma
                if shape not in gamma_cache:
                    gamma_curve_mod = scipy.stats.gamma(a=shape, scale=b).pdf(x)
                    gamma_curve_mod /= gamma_curve_mod.max()
                    gamma_cache[shape] = gamma_curve_mod
                else:
                    gamma_curve_mod = gamma_cache[shape]

                # # Get corresponding envelope segment
                # envelope_segment = envelopes[i:i + stim_samples]
                #
                # # Normalize envelope segment (optional)
                # if envelope_segment.max() > 0:
                #     envelope_segment = envelope_segment / envelope_segment.max()

                # Combine gamma × envelope × amplitude (removed env for now)
                modulated_gamma = gamma_curve_mod * amplitude

                # Insert into predictor
                attention_predictor[i:i + stim_samples] += modulated_gamma

        return attention_predictor
   # get model inputs non z-scored:

    model_inputs_path = default_path /'data/eeg/trf/model_inputs' / f'{plane}_raw'
    def get_predictor_array(pred_type=''):
        for files in model_inputs_path.iterdir():
            if stream_type1 in files.name and stream_type2 in files.name:
                if 'target' in files.name:
                    target_stream_arrays = np.load(files)
                    target_stream_arrays = target_stream_arrays[pred_type]
                    target_stream_arrays = target_stream_arrays.astype(np.float32)
                elif 'distractor' in files.name:
                    distractor_stream_arrays = np.load(files)
                    distractor_stream_arrays = distractor_stream_arrays[pred_type]
                    distractor_stream_arrays = distractor_stream_arrays.astype(np.float32)
        return target_stream_arrays, distractor_stream_arrays

    # modify attention gamma dist array:
    target_overlap_ratios, distractor_overlap_ratios = get_predictor_array(pred_type='overlap_ratios')    # events proximity pre
    target_events_proximity_pre, distractor_events_proximity_pre = target_overlap_ratios, distractor_overlap_ratios = get_predictor_array(pred_type='events_proximity_pre')
    # events proximity post
    target_events_proximity_post, distractor_events_proximity_post = target_overlap_ratios, distractor_overlap_ratios = get_predictor_array(
        pred_type='events_proximity_post')

    # Apply for both streams
    attention_predictor_target = add_gamma_to_predictor(target_onsets, target_overlap_ratios, target_events_proximity_pre, target_events_proximity_post)
    attention_predictor_distractor = add_gamma_to_predictor(distractor_onsets, distractor_overlap_ratios, distractor_events_proximity_pre, distractor_events_proximity_post)

    # Define a time range to visualize (e.g., first 3000 samples = 24 seconds at 125Hz)
    def plot_gammas(predictor_target, predictor_distractor):
        start = 100000
        end = 102000
        time = np.arange(start, end) / 125.0  # seconds

        plt.figure(figsize=(12, 6))

        # Plot attention predictor
        plt.plot(time, predictor_target[start:end], label='Attention Predictor (Gamma-based) Target', color='red')
        plt.plot(time, predictor_distractor[start:end], label='Attention Predictor (Gamma-based) Distractor', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.title('Attention Predictor')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


    plot_gammas(attention_predictor_target, attention_predictor_distractor)
    plt.close('all')

    # Build combined DataFrame
    X_gamma = pd.DataFrame(
        np.column_stack([
            attention_predictor_target,
            attention_predictor_distractor
        ]),
        columns=['target_gamma_distributions', 'distractor_gamma_distributions']
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