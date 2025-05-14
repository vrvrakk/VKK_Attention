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
plane = 'azimuth'
stim_type = 'stream1_stream2'
attentional_predictor = default_path /'data'/ 'eeg' / 'trf' / 'trf_testing' / "attentional_predictor" / plane
for files in attentional_predictor.iterdir():
    if stim_type in files.name:
        attention_array = np.load(files)
        target_attention_array = attention_array['target_attention']
        distractor_attention_array = attention_array['distractor_attention']

from sklearn.linear_model import LinearRegression

# Step 1: Regress distractor on target
reg = LinearRegression()
reg.fit(target_attention_array.reshape(-1, 1), distractor_attention_array)

# Step 2: Compute residuals = orthogonalized distractor
distractor_ortho = distractor_attention_array - reg.predict(target_attention_array.reshape(-1, 1))

# Step 3: Rebuild stacked predictor matrix
predictors_stacked = np.vstack((target_attention_array, distractor_ortho)).T  # shape (samples, 2)

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
predictors_stacked_z = scaler.fit_transform(predictors_stacked)

X = pd.DataFrame(predictors_stacked, columns=['target', 'distractor'])
X = sm.add_constant(X)
vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
print(vif)

sfreq = 125
n_samples = sfreq * 60
total_samples = len(predictors_stacked)
n_folds = total_samples // n_samples
# Split predictors and EEG into subject chunks

#import eeg data:
eeg_concat_path = default_path / 'data'/ 'eeg' / 'preprocessed' / 'results' / 'concatenated' / 'continuous'/ plane
for eeg_array in eeg_concat_path.iterdir():
    eeg_all = np.load(eeg_array)
    eeg_all = eeg_all['eeg_data']

# n_folds = 5
X_folds = np.array_split(predictors_stacked, n_folds)
Y_folds = np.array_split(eeg_all, n_folds)

lambdas = np.logspace(-2, 2, 20)  # based on prev literature

def optimize_lambda(X_folds, Y_folds, fs, tmin, tmax, lambdas):
    def test_lambda(lmbda):
        fwd_trf = TRF(direction=1)
        r = crossval(fwd_trf, X_folds, Y_folds, fs, tmin, tmax, lmbda)
        return lmbda, r.mean()

    print(f"Running lambda optimization across {len(lambdas)} values...")
    results = []
    for lmbda in lambdas:
        lmbda_val, mean_r = test_lambda(lmbda)
        results.append((lmbda_val, mean_r))

    # Find best
    best_lambda, best_score = max(results, key=lambda x: x[1])
    print(f'Best lambda: {best_lambda:.2e} (mean r = {best_score:.3f})')
    return best_lambda

best_lambda = optimize_lambda(X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, lambdas=lambdas)

trf = TRF(direction=1)
trf.train(X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=best_lambda, seed=42)
prediction, r = trf.predict(predictors_stacked, eeg_all)
print(f"Full model correlation: {r.round(3)}")

r_crossval = crossval(trf, X_folds, Y_folds, fs=sfreq, tmin=-0.1, tmax=1.0, regularization=best_lambda, seed=42)
print(f"mean correlation between actual and predicted response: {r_crossval.mean().round(3)}")

predictor_names = ['target_attention_model', 'distractor_attention_model']  # or however many you have
weights = trf.weights  # shape: (n_features, n_lags, n_channels)
time_lags = np.linspace(-0.1, 1.0, weights.shape[1])  # time axis

# Loop and plot
# Define your lag window of interest
tmin_plot = 0.0
tmax_plot = 0.4

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
    plt.plot([], [], ' ', label=f'λ = {best_lambda:.2f}, r = {r:.3f}')
    plt.legend(loc='upper right', fontsize=8, ncol=2)
    plt.tight_layout()
    plt.show()
    plt.savefig(save_path / filename, dpi=300)
    plt.close('all')