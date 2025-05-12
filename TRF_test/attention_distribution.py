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

''' A script to get dynamic attentional predictor per condition (azimuth vs elevation) - with stacked predictors - 5 per stream
semantic_weights, envelopes, overlap_ratios, events_proximity_pre, events_proximity_post, (RTs)'''

def load_model_inputs(plane, array_type=''):
    model_input_path = default_path / 'data' / 'eeg' / 'trf' / 'model_inputs'
    for folders in model_input_path.iterdir():
        if plane in folders.name:
            for files in folders.iterdir():
                if 'eeg_all' in files.name:
                    eeg_all = np.load(files)
                elif f'{plane}_{array_type}_pred_target' in files.name and 'npz' in files.name:
                    target_pred_array = np.load(files)
                elif f'{plane}_{array_type}_pred_distractor' in files.name and 'npz' in files.name:
                    distractor_pred_array = np.load(files)
    return eeg_all, target_pred_array, distractor_pred_array




if __name__ == '__main__':

    print("Available CPUs:", os.cpu_count())
    print(f"Free RAM: {psutil.virtual_memory().available / 1e9:.2f} GB")

    default_path = Path.cwd()
    predictors_path = default_path / 'data/eeg/predictors'
    eeg_results_path = default_path / 'data/eeg/preprocessed/results'
    sfreq = 125

    plane = 'azimuth'
    stream_type1 = 'stream1'
    stream_type2 = 'stream2'

    eeg_all, target_pred_array, distractor_pred_array = load_model_inputs(plane, array_type=f'{stream_type1}_{stream_type2}')

    # Define order to ensure consistency
    if stream_type1 != 'targets':
        ordered_keys = ['onsets', 'envelopes', 'overlap_ratios',
                        'events_proximity_pre', 'events_proximity_post']
    else:
        ordered_keys = ['onsets', 'envelopes', 'overlap_ratios',
                        'events_proximity_pre', 'events_proximity_post', 'RTs']

    # Stack predictors for the target stream
    X_target = np.column_stack([target_pred_array[k] for k in ordered_keys])

    # Stack predictors for the distractor stream
    X_distractor = np.column_stack([distractor_pred_array[k] for k in ordered_keys])

    print("X_target shape:", X_target.shape)
    print("X_distractor shape:", X_distractor.shape)

    eeg_all = eeg_all.T

    # Convert to float32
    X_target = X_target.astype(np.float32)
    X_distractor = X_distractor.astype(np.float32)
    eeg_all = eeg_all.astype(np.float32)
    # A 7503×11 predictor matrix in float64 = ~0.6 MB → float32 = ~0.3 MB.
    # Multiply that by 482 folds × 2 streams × 2 jobs, and the savings are huge.

    print("Converted all arrays to float32.")

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
    gamma_curve = gamma.pdf(x)
    gamma_curve /= gamma_curve.max()
    # Computes the probability density values of the gamma distribution over the 93 x-values.
    # Normalizes the curve so that its maximum value = 1, ensuring consistency across events.

    # Later, each gamma is scaled by the stimulus weight (1–4), so normalization is crucial.

    # Helper: insert scaled gamma at index
    def add_gamma_to_predictor(onsets):
        attention_predictor = np.zeros(len(eeg_all))
        for i in range(len(onsets) - stim_duration_samples):
            weight = onsets[i]
            if weight > 0:
                attention_predictor[i:i + stim_duration_samples] += gamma_curve * weight
        return attention_predictor

    # Apply for both streams
    attention_predictor_target = add_gamma_to_predictor(target_onsets)
    attention_predictor_distractor = add_gamma_to_predictor(distractor_onsets)

    # Define a time range to visualize (e.g., first 3000 samples = 24 seconds at 125Hz)
    def plot_gammas(predictor_target, predictor_distractor):
        start = 100000
        end = 102000
        time = np.arange(start, end) / 125.0  # seconds

        plt.figure(figsize=(12, 6))

        # Plot attention predictor
        plt.plot(time, predictor_target[start:end], label='Attention Predictor (Gamma-based)', color='red')
        plt.plot(time, predictor_distractor[start:end], label='Attention Predictor (Gamma-based)', color='blue')
        plt.xlabel('Time (s)')
        plt.ylabel('Value')
        plt.title('Attention Predictor')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    # envelopes:
    target_overlaps = X_target[2]
    distractor_overlaps = X_distractor[2]

