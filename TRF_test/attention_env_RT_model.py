import os
from pathlib import Path
import numpy as np
from mtrf import TRF
from mtrf.stats import  crossval
import matplotlib.pyplot as plt
plane = 'azimuth'
# Loop and plot
default_path = Path.cwd()
save_path = default_path / 'data' / 'eeg' / 'trf' / 'trf_testing' / 'attentional_predictor' / plane
# Save TRF results for this condition
stream_type1 = 'stream1'
stream_type2 = 'stream2'
import pandas as pd

sfreq = 125

attentional_pred = np.load(save_path / f"{stream_type1}_{stream_type2}.npz")
target_attention = attentional_pred['target_attention']
distractor_attention = attentional_pred['distractor_attention']  # already ortho and z-scored

target_preds = default_path / f'data/eeg/trf/model_inputs/{plane}/{plane}_{stream_type1}_{stream_type2}_pred_target_stream_arrays.npz'
target_preds = np.load(target_preds)
rt_targets = target_preds['RTs']

distractor_preds = default_path / f'data/eeg/trf/model_inputs/{plane}/{plane}_{stream_type1}_{stream_type2}_pred_distractor_stream_arrays.npz'
distractor_preds = np.load(distractor_preds)
rt_distractors = distractor_preds['RTs']

# onsets
onsets_targets = target_preds['onsets']
onsets_distractors = distractor_preds['onsets']
# overlaps
overlaps_targets = target_preds['overlap_ratios']
overlaps_distractors = distractor_preds['overlap_ratios']
# prox pre
prox_pre_targets = target_preds['events_proximity_pre']
prox_pre_distractors = distractor_preds['events_proximity_pre']

# prox post
prox_post_targets = target_preds['events_proximity_post']
prox_post_distractors = distractor_preds['events_proximity_post']


eeg_all = default_path / f'data/eeg/trf/model_inputs/{plane}/{plane}_eeg_all.npy'
eeg_all = np.load(eeg_all)
eeg_all = eeg_all.T

envelopes_targets = target_preds['envelopes']
envelopes_distractors = distractor_preds['envelopes']

best_lambda = 1.0

# Build dicts
target_dict = {
    'attentional_predictor_target': target_attention,
    'envelopes_target': envelopes_targets,
    'RTs_target': rt_targets
}

distractor_dict = {
    'attentional_predictor_distractor': distractor_attention,
    'envelopes_distractor': envelopes_distractors,
    'RTs_distractor': rt_distractors
}

ordered_keys_target = ['attentional_predictor_target', 'envelopes_target', 'RTs_target']
ordered_keys_distractor = ['attentional_predictor_distractor', 'envelopes_distractor', 'RTs_distractor']
# Stack in order
X_target = np.column_stack([target_dict[k] for k in ordered_keys_target])
X_distractor = np.column_stack([distractor_dict[k] for k in ordered_keys_distractor])

print("X_target shape:", X_target.shape)
print("X_distractor shape:", X_distractor.shape)

# 3. Combine predictors into final matrix
X = pd.DataFrame(
    np.column_stack([X_target, X_distractor]),
    columns=[f'{k}' for k in ordered_keys_target] + [f'{k}' for k in ordered_keys_distractor]
)

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
# 4. Add constant for VIF calculation
X = sm.add_constant(X)
X = sm.add_constant(X)
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
print(vif)

# split into trials:
predictors_stacked = X.values  # ← ready for modeling
n_samples = sfreq * 60
total_samples = len(predictors_stacked)
n_folds = total_samples // n_samples
# Split predictors and EEG into subject chunks
X_folds = np.array_split(predictors_stacked, n_folds)
Y_folds = np.array_split(eeg_all, n_folds)

import random

random.seed(42)

trf = TRF(direction=1)
trf.train(X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=best_lambda, seed=42)

predictions = []
r_vals = []

from tqdm import tqdm

for i in tqdm(range(n_folds)):
    start = i * n_samples
    end = start + n_samples
    X_chunk = predictors_stacked[start:end]
    Y_chunk = eeg_all[start:end]

    pred_chunk, r_chunk = trf.predict(X_chunk, Y_chunk)
    predictions.append(pred_chunk)
    r_vals.append(r_chunk)

predicted_full = np.vstack(predictions)
r_mean = np.mean(r_vals, axis=0)
print("Avg r across all chunks:", np.round(r_mean, 3))

r_crossval = crossval(
    trf,
    X_folds,
    Y_folds,
    fs=sfreq,
    tmin=-0.1,
    tmax=1.0,
    regularization=best_lambda
)

stim1 = 'target_stream'
stim2 = 'distractor_stream'

predictor_names = [f'{stim1}', f'{stim2}']  # or however many you have
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
save_path = default_path / f'data/eeg/trf/trf_testing/composite_model/{plane}'
save_path.mkdir(parents=True, exist_ok=True)
data_path = save_path / 'data'
data_path.mkdir(parents=True, exist_ok=True)

trf_preds = list(X.columns)

model = 'attentional_model_env_RTs'
# Save TRF results for this condition
np.savez(
    data_path / f'{plane}_{model}_{stream_type1}_{stream_type2}_TRF_results.npz',
    results=predicted_full,
    preds=list(X.columns),
    weights=weights,  # raw TRF weights (n_predictors, n_lags, n_channels)
    r=r_mean,
    r_crossval=r_crossval,
    best_lambda=best_lambda,
    time_lags=time_lags,
    time_lags_trimmed=time_lags_trimmed,
    predictor_names=np.array(predictor_names),
    condition=plane
)

# Plot each predictor
for i, name in enumerate(trf_preds):
    filename = name + '_' + stream_type1 + '_' + stream_type2
    plt.figure(figsize=(8, 4))

    trf_weights = weights[i].T[:, lag_mask]  # shape: (n_channels, n_lags_selected)

    # Smooth for aesthetics
    window_len = 11
    hamming_win = np.hamming(window_len)
    hamming_win /= hamming_win.sum()
    smoothed_weights = np.array([
        np.convolve(trf_weights[ch], hamming_win, mode='same')
        for ch in range(trf_weights.shape[0])
    ])

    # Plot per channel
    for ch in range(smoothed_weights.shape[0]):
        plt.plot(time_lags_trimmed, smoothed_weights[ch], alpha=0.4)

    plt.title(f'TRF for {name}')
    plt.xlabel('Time lag (s)')
    plt.ylabel('Amplitude (a.u.)')
    plt.plot([], [], ' ', label=f'λ = {best_lambda:.2f}, r = {r_crossval:.2f}')
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    # Save and show
    plt.savefig(save_path / f'{filename}.png', dpi=300)
    plt.show()

plt.close('all')