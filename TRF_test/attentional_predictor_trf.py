from pathlib import Path
import os
import numpy as np
from mtrf import TRF
from mtrf.stats import crossval
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor


# import attentional predictor:
default_path = Path.cwd()
plane = 'elevation'
stim_type = 'stream1_stream2'
attentional_predictor = default_path / 'data' / 'eeg' / 'trf' / 'trf_testing' / "attentional_predictor" / plane
for files in attentional_predictor.iterdir():
    if stim_type in files.name:
        attention_array = np.load(files)
        target_attention_array = attention_array['target_attention']
        distractor_attention_array = attention_array['distractor_attention']


# Step 3: Rebuild stacked predictor matrix
predictors_stacked = np.vstack((target_attention_array, distractor_attention_array)).T  # shape (samples, 2)
save_path = default_path / f'data/eeg/trf/model_inputs/{plane}/attentional_predictors_stacked.npy'
np.savez(save_path, predictors=predictors_stacked, plane=plane)


# making sure again: collinearity check
X = pd.DataFrame(predictors_stacked, columns=['target', 'distractor'])
X = sm.add_constant(X)
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
print(vif)

sfreq = 125
n_samples = sfreq * 60
total_samples = len(predictors_stacked)
n_folds = total_samples // n_samples
# Split predictors and EEG into subject chunks

# import eeg data:
eeg_concat_path = default_path / f'data/eeg/trf/model_inputs/{plane}/{plane}_eeg_all.npy'
eeg_all = np.load(eeg_concat_path)
eeg_all = eeg_all.T

print(predictors_stacked.shape)
print(eeg_all.shape)


# n_folds = 5
X_folds = np.array_split(predictors_stacked, n_folds)
Y_folds = np.array_split(eeg_all, n_folds)

best_lambda = 1.0

trf = TRF(direction=1)
trf.train(X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=best_lambda, seed=42)
prediction, r = trf.predict(predictors_stacked, eeg_all)
print(f"Full model correlation: {r.round(3)}")

r_crossval = crossval(trf, X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=best_lambda, seed=42)

predictor_names = ['target_attention_model', 'distractor_attention_model']  # or however many you have
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
save_path = default_path / f'data/eeg/trf/trf_testing/attentional_predictor/{plane}/data'
save_path.mkdir(parents=True, exist_ok=True)
data_path = save_path / 'data'
data_path.mkdir(parents=True, exist_ok=True)
# Save TRF results for this condition
np.savez(
data_path / f'{plane}_TRF_results.npz',
    results=prediction,
    weights=weights,  # raw TRF weights (n_predictors, n_lags, n_channels)
    r=r,
    r_crossval=r_crossval,
    best_lambda=best_lambda,
    time_lags=time_lags,
    time_lags_trimmed=time_lags_trimmed,
    predictor_names=np.array(predictor_names),
    condition=plane
    )

for i, name in enumerate(predictor_names):
    filename = name
    plt.figure(figsize=(8, 4))
    trf_weights = weights[i].T[:, lag_mask]  # shape: (n_channels, selected_lags)
    # Smoothing with Hamming window for aesthetic purposes..
    window_len = 11
    hamming_win = np.hamming(window_len)
    hamming_win /= hamming_win.sum()
    smoothed_weights = np.array([
        np.convolve(trf_weights[ch], hamming_win, mode='same')
        for ch in range(trf_weights.shape[0])
    ])

    for ch in range(trf_weights.shape[0]):
        plt.plot(time_lags_trimmed, smoothed_weights[ch], alpha=0.4)

    plt.title(f'TRF for {name}')
    plt.xlabel('Time lag (s)')
    plt.ylabel('Amplitude')
    plt.plot([], [], ' ', label=f'λ = {best_lambda:.2f}, r = {r_crossval:.2f}')
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path / filename, dpi=300)
    plt.close('all')