# Import necessary libraries
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from mtrf.model import TRF
from scipy.stats import zscore
import librosa
import scipy.signal

# Sample function to load data (replace this with real data)
def load_data():
    # Load EEG data and audio file (simulated here)
    eeg_data = np.random.rand(3000, 1)  # Simulating EEG data
    audio_data, sr = librosa.load(librosa.ex('trumpet'))  # Load example audio
    return eeg_data, audio_data, sr

eeg_data, audio_data, sr = load_data()

# Function to extract audio features (envelope or mel-spectrogram)
def extract_audio_features(audio_data, sr, feature_type="envelope"):
    if feature_type == "envelope":
        # Compute the amplitude envelope using the hilbert transform
        envelope = np.abs(scipy.signal.hilbert(audio_data))
        return envelope
    elif feature_type == "mel":
        # Compute the mel-spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=40, fmax=8000)
        return mel_spectrogram.T
    else:
        raise ValueError("Invalid feature type specified. Choose 'envelope' or 'mel'.")

# Function to truncate or pad the data to ensure the same length
def match_data_lengths(data1, data2):
    min_len = min(len(data1), len(data2))
    data1 = data1[:min_len]
    data2 = data2[:min_len]
    return data1, data2

# Step 1: Extract envelope or mel-spectrogram from the audio
audio_feature_type = "envelope"  # Choose between 'envelope' and 'mel'
audio_features = extract_audio_features(audio_data, sr, feature_type=audio_feature_type)

# Step 2: Normalize the EEG and audio features
eeg_data = zscore(eeg_data) # scipy function
audio_features = zscore(audio_features)

# Step 3: Ensure EEG and audio features have the same number of samples
audio_features, eeg_data = match_data_lengths(audio_features, eeg_data)

# Step 4: Visualizing the EEG and audio features
plt.figure(figsize=(10, 4))
plt.plot(eeg_data[:300], label='EEG Data (first 300 samples)', color='blue')
plt.title('EEG Data (first 300 samples)')
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(audio_features[:300], label=f'Audio Features ({audio_feature_type}) (first 300 samples)', color='green')
plt.title(f'Audio Features ({audio_feature_type}) (first 300 samples)')
plt.show()


# Step 5: Setting up the mTRF model parameters
direction_forward = 1  # Stimulus -> EEG
direction_backward = -1  # EEG -> Stimulus
sampling_rate = sr
regularization = 1e3  # Regularization parameter
n_samples = min(eeg_data.shape[0], audio_features.shape[0])  # Use minimum length between EEG and audio features

# Calculate the maximum lag based on tmin and the adjusted tmax
tmin = 0
max_lag_samples = n_samples // 5  # We allow for a third of the signal length as max lag

# Adjust tmax dynamically based on the signal length
tmax = max_lag_samples / sampling_rate


# Convert tmin and tmax to samples
min_lag_samples = int(tmin * sampling_rate)
max_lag_samples = int(tmax * sampling_rate)

# Check if the maximum lag exceeds the number of samples in the data
if max_lag_samples > n_samples:
    raise ValueError(f"The maximum lag ({max_lag_samples} samples) is longer than the signal length ({n_samples} samples). Please reduce tmin/tmax or use more data.")

# Initialize TRF models
fwd_trf = TRF(direction=direction_forward)
back_trf = TRF(direction=direction_backward)
print(f"TRF models initialized with tmax={tmax:.3f} seconds (max lag: {max_lag_samples} samples)")

# Step 6: Forward Model (Audio Features -> EEG)

fwd_trf.train(audio_features, eeg_data, sampling_rate, tmin, tmax, regularization)
predicted_eeg, r_forward = fwd_trf.predict(audio_features, eeg_data)

# Convert predicted_eeg to a NumPy array before squeezing
predicted_eeg = np.array(predicted_eeg).squeeze()
eeg_data = np.array(eeg_data).squeeze()

# Visualize forward model results
plt.figure(figsize=(10, 4))
plt.plot(eeg_data[:300], label='Actual EEG (Channel 1)', color='blue')
plt.plot(predicted_eeg[:300], label='Predicted EEG (Channel 1)', color='orange', linestyle='--')
plt.title('Forward Model: Actual vs Predicted EEG (First 300 samples)')
plt.legend()
plt.show()

# Step 7: Backward Model (EEG -> Audio Features)
back_trf.train(eeg_data, audio_features, sampling_rate, tmin, tmax, regularization)
reconstructed_audio, r_backward = back_trf.predict(eeg_data, audio_features)

# Reshape the reconstructed and actual audio if necessary
reconstructed_audio = np.array(reconstructed_audio).squeeze()
audio_features = np.array(audio_features).squeeze()

# Visualize backward model results
plt.figure(figsize=(10, 4))
plt.plot(audio_features[:300], label='Actual Audio Features', color='green')
plt.plot(reconstructed_audio[:300], label='Reconstructed Audio Features', color='red', linestyle='--')
plt.title(f'Backward Model: Actual vs Reconstructed Audio Features (First 300 samples)')
plt.legend()
plt.show()


from sklearn.model_selection import KFold
import numpy as np
import matplotlib.pyplot as plt

# Define k-fold cross-validation with k=5
kf = KFold(n_splits=5)
correlations = []

# Set a maximum number of samples for lag calculations
min_samples_for_test = 50  # Example threshold to ensure a reasonable test size

# Define time lags
max_lag_samples = int((tmax - tmin) * sampling_rate)

# Store correlation results for each fold
correlation_per_fold = []

for fold, (train_idx, test_idx) in enumerate(kf.split(eeg_data)):
    # Get training and test data
    eeg_train, eeg_test = eeg_data[train_idx], eeg_data[test_idx]
    audio_train, audio_test = audio_features[train_idx], audio_features[test_idx]

    # Check number of samples in the training and test sets
    n_samples_train = eeg_train.shape[0]
    n_samples_test = eeg_test.shape[0]

    # Skip iterations where the test set size is too small
    if n_samples_test < min_samples_for_test:
        print(f"Skipping iteration: Test set contains only {n_samples_test} samples, which is too few for the maximum lag.")
        continue

    # Adjust tmax if the lag exceeds the number of samples
    if max_lag_samples > n_samples_train:
        print(f"Warning: Max lag ({max_lag_samples} samples) exceeds training signal length ({n_samples_train} samples).")
        max_lag_samples = n_samples_train
        tmax_adjusted = max_lag_samples / sampling_rate
    else:
        tmax_adjusted = tmax

    # Train the model on the training samples
    fwd_trf.train(audio_train, eeg_train, sampling_rate, tmin, tmax_adjusted, regularization)

    # Test the model on the left-out sample
    if max_lag_samples > n_samples_test:
        print(f"Warning: Max lag ({max_lag_samples} samples) exceeds test signal length ({n_samples_test} samples). Reducing max lag.")
        max_lag_samples = n_samples_test
        tmax_adjusted = max_lag_samples / sampling_rate

    # Predict EEG response
    predicted, _ = fwd_trf.predict(audio_test, eeg_test)

    # Convert predicted to numpy array if it's a list
    predicted = np.array(predicted).squeeze()

    # Calculate correlation for the test sample
    corr = np.corrcoef(eeg_test.squeeze(), predicted)[0, 1]
    correlations.append(corr)

    # Store correlation per fold
    correlation_per_fold.append(corr)
    print(f"Fold {fold + 1} correlation: {corr}")

# Print the average correlation coefficient
mean_correlation = np.mean(correlations)
print(f"Mean Correlation across k-Fold CV: {mean_correlation}")

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(correlation_per_fold) + 1), correlation_per_fold, marker='o', linestyle='-', color='b', label='Correlation per Fold')
plt.axhline(y=mean_correlation, color='r', linestyle='--', label=f'Mean Correlation: {mean_correlation:.2f}')
plt.title('Correlation per Fold in k-Fold Cross-Validation')
plt.xlabel('Fold Number')
plt.ylabel('Correlation Coefficient')
plt.legend()
plt.grid(True)
plt.show()


# Step 9: Auto-correlation and Cross-correlation
print("Visualizing Auto-correlation and Cross-correlation")
eeg_auto_corr = np.correlate(eeg_data, eeg_data, mode='full')
audio_auto_corr = np.correlate(audio_features, audio_features, mode='full')
cross_corr = np.correlate(audio_features, eeg_data, mode='full')


# Plot the EEG auto-correlation
plt.figure(figsize=(10, 4))
plt.plot(eeg_auto_corr, label='EEG Auto-correlation', color='purple')
plt.title('EEG Auto-correlation')
plt.legend()
plt.show()

# Plot the Audio Features auto-correlation
plt.figure(figsize=(10, 4))
plt.plot(audio_auto_corr, label=f'{audio_feature_type.capitalize()} Auto-correlation', color='orange')
plt.title(f'Audio Features ({audio_feature_type}) Auto-correlation')
plt.legend()
plt.show()


# Plot the cross-correlation
plt.figure(figsize=(10, 4))
plt.plot(cross_corr, label='Cross-correlation (Audio Features vs EEG)', color='red')
plt.title('Cross-correlation between Audio Features and EEG')
plt.legend()
plt.show()


# Step 10: Explore the Effect of Regularization (Lambda) on TRF
lambdas = [1e0, 1e3, 1e6]  # Example lambda values
for lam in lambdas:
    # Train the forward model with different lambda values
    fwd_trf.train(audio_features, eeg_data, sampling_rate, tmin, tmax, lam)
    predicted_eeg, _ = fwd_trf.predict(audio_features, eeg_data)

    # Convert predicted_eeg to a NumPy array if it's not already
    predicted_eeg = np.array(predicted_eeg)

    # Reshape or squeeze the predicted EEG to remove any extra dimensions
    predicted_eeg = predicted_eeg.squeeze()

    # Visualize the effect of different lambdas
    plt.figure(figsize=(10, 4))
    plt.plot(predicted_eeg[:300], label=f'Predicted EEG (Lambda={lam})', linestyle='--')
    plt.title(f'Forward Model with Lambda={lam} (First 300 samples)')
    plt.legend()
    plt.show()

# Step 11: Visualizing Time-Shift Effects on TRF

# Convert predicted_eeg to a NumPy array if it's not already
predicted_eeg = np.array(predicted_eeg)

# Reshape or squeeze the predicted EEG to remove any extra dimensions
predicted_eeg = predicted_eeg.squeeze()

# Define time shifts in samples
time_shifts = np.arange(int(tmin * sampling_rate), int(tmax * sampling_rate))

# Ensure we truncate the predicted EEG to match the number of time shifts (lags)
predicted_eeg_truncated = predicted_eeg[:len(time_shifts)]

print(f"Time shifts applied: {time_shifts} (samples)")

# Plot the mean response over time shifts
plt.figure(figsize=(10, 4))
plt.plot(time_shifts, predicted_eeg_truncated, label='Predicted EEG Mean Response over Time')
plt.title('Effect of Time Shift on Predicted EEG Response')
plt.xlabel('Time Lag (samples)')
plt.ylabel('Mean EEG Response')
plt.legend()
plt.show()



# Step 12: Summary of the analysis
print("Summary:")
print("- We loaded EEG and audio data, extracted features, trained forward and backward models.")
print("- We visualized actual vs predicted EEG and reconstructed audio features.")
print("- We applied Leave-One-Out Cross-Validation and explored the effect of different regularization parameters (lambdas).")