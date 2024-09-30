
'''
Goal of this script, is to load the pre-processed EMG signals (epochs), baseline, motor-only and experiment's as well.
Then we would extract features from baseline and response epochs, to train two machine learning models (KNN and SVM).
Aim would be, to be able to classify epochs from the experiment's EMG, to 'no response', 'true response' and 'partial error'.
The classification will be done based on features like RMS, variance of the signal, power of the signal, frequency-based, etc.
 '''

# FIRST: IMPORT PACKAGES AND MODULES:
from pathlib import Path
import os
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
import joblib

# SET PATHS:
sub = input('Provide subject number: ')
condition = input('Select a condition (a1, a2, e1 or e2): ')
default_dir = Path.cwd()
sub_dir = default_dir / 'data' / 'emg' / f'{sub}'
fif_path = sub_dir / 'fif files'
features_path = sub_dir/ 'preprocessed' / 'results' / 'features'
if not features_path.exists:
    os.makedirs(features_path)
# IMPORT .FIF FILES OF INTEREST:
motor_only_file = []
baseline_file = []
epochs_files = []
for files in fif_path.iterdir():
    if files.is_file:
        if condition in files.name:
            epochs_files.append(files)
        elif 'baseline' in files.name:
            baseline_file.append(files)
        elif 'motor' in files.name:
            motor_only_file.append(files)

# READ MNE FILES:
baseline = mne.read_epochs(baseline_file[0], preload=True)
motor = mne.read_epochs(motor_only_file[0], preload=True)
# categorize experiment's epochs to target, distractor and responses' epochs:
responses = []
targets = []
distractors = []
for files in epochs_files:
    if files.is_file and 'target' in files.name:
        targets.append(files)
    elif files.is_file and 'distractor' in files.name:
        distractors.append(files)
    elif files.is_file and 'responses' in files.name:
        responses.append(files)

# read files:
def read_epochs(epochs):
    chosen_epochs = []
    for index in range(len(epochs)):
        file = epochs[index]
        chosen_epoch = mne.read_epochs(file, preload=True)
        chosen_epochs.append(chosen_epoch)
    return chosen_epochs

target_epochs = read_epochs(targets)
distractor_epochs = read_epochs(distractors)
responses_epochs = read_epochs(responses)

# EXTRACTING FEATURES:
# from baseline and motor-only epochs:

def amplitude_rms_var(helper_epochs):
    features = []
    for epochs in helper_epochs:
        # Compute the feature for each epoch and each channel separately
        peak_amplitude = np.ptp(epochs, axis=1)  # Peak-to-peak for each channel (keep per-channel info)
        rms = np.sqrt(np.mean(np.square(epochs), axis=1))  # RMS for each channel (per-channel)
        var = np.var(epochs, axis=1)  # Variance for each channel (per-channel)
        features.append(np.concatenate([peak_amplitude, rms, var]))  # Collect features per epoch (for each channel)
        # # Concatenate into a single array for each epoch
    features = np.array(features)  # Shape will epochs, 6 features (2 channels for each feature)
    return features


baseline_features = amplitude_rms_var(baseline)
motor_features = amplitude_rms_var(motor)

# Z-scores from baseline, and motor-only recordings:
def get_helpers_z_scores(helper_epochs):
    helper_data = helper_epochs.get_data(copy=True)  # Shape (n_epochs, n_channels, n_samples)

    # Calculate the mean and std for each epoch and each channel independently
    helper_mean = helper_data.mean(axis=2)  # Mean for each epoch and each channel (n_epochs, n_channels)
    helper_std = helper_data.std(axis=2)  # Std deviation for each epoch and each channel (n_epochs, n_channels)

    # Reshape mean and std for broadcasting
    helper_mean = helper_mean[:, :, np.newaxis]
    helper_std = helper_std[:, :, np.newaxis]

    # Compute z-scores for each epoch and channel
    helper_z_data = (helper_data - helper_mean) / helper_std  # Z-scores per epoch and channel
    helper_avg_z_data = np.mean(helper_z_data, axis=-1)  # Z-scores averaged across time
    return helper_data, helper_mean, helper_std, helper_z_data, helper_avg_z_data  # Return per-channel z-scores

baseline_data, baseline_mean, baseline_std, baseline_z_data, baseline_avg_z_data = get_helpers_z_scores(baseline)
motor_data, motor_mean, motor_std, motor_z_data, motor_avg_z_data = get_helpers_z_scores(motor)

# extract power in the 20–45 Hz band, which is relevant for muscle data (FDP muscle)
'''
The Welch method is a way of estimating the power spectral density (PSD) of a signal.
In simple terms, it breaks the signal into smaller overlapping segments, calculates the PSD for each segment,
and then averages the results. 
This gives a more accurate and stable estimate of how the signal’s energy is distributed across different frequencies.
What does the Welch method do?
    It estimates the frequency content of a time series (such as an EMG or EEG signal).
    It’s based on Fourier Transform, which converts the signal from the time domain to the frequency domain.
    Welch improves on a basic Fourier transform by dividing the signal into segments
    to reduce noise and provide a smoother and more accurate PSD.
    
fs: The sampling frequency (in Hz) of your data. 
This tells the function how fast the data was sampled, helping it to map time samples into frequency.
nperseg: This is the length of each segment (how many data points per segment). If the segment length is too long, 
you risk capturing noise; if it’s too short, you miss important details

If you have an EMG signal, say from 0 to 500 Hz, 
the Welch method helps you understand how much power (or energy) is concentrated in different frequency ranges, 
like between 20 and 50 Hz (where most of the muscle activity is). 
The Welch method gives you a smooth and reliable estimate of how much power is in those frequencies
by using smaller segments and averaging them out.
'''

def welch(helper_epochs, fmin=20, fmax=45):
    kwargs = dict(fmin=fmin, fmax=fmax, n_jobs=None)
    psds_welch_mean, freqs_mean = helper_epochs.compute_psd("welch", average="mean", **kwargs).get_data(return_freqs=True)
    # compute_psd calculates the Power Spectral Density (PSD) using Welch’s method.
    # The average="mean" computes the mean of the segments' PSD, and average="median" computes the median.
    # psds_welch_mean and psds_welch_median store the power spectra,
    # and freqs_mean and freqs_median store the corresponding frequency values.
    # Convert power to dB scale.
    psds_mean_db = 10 * np.log10(psds_welch_mean)
    # Sum power in the specified frequency band (20-40 Hz)
    # No averaging across channels or epochs
    band_idx = (freqs_mean >= fmin) & (freqs_mean <= fmax)
    band_power = psds_mean_db[:, :, band_idx]  # Keep per-epoch, per-channel data
    # Shape (n epochs, 2 channels, n_freq_bins)
    return band_power

baseline_band_power = welch(baseline)
motor_band_power = welch(motor)

# plot time-frequency analysis using the function below:
def morlet_func(helper_epochs):
    # Morlet wavelet transform: This is a technique to extract time-frequency information from a signal. It uses wavelets (localized waves) to transform the data and analyze how power changes over both time and frequency.
    # helper_epochs: The epoch data you are analyzing.
    # freqs=freqs: These are the frequencies (20 Hz to 45 Hz) that the transform will look at.
    # n_cycles=n_cycles: The number of cycles to use for the wavelet transform, defined for each frequency in freqs.
    # average=True: This averages the data across trials (epochs), meaning you'll get an overall picture of how power changes over time and frequency across all epochs.
    # return_itc=True: This returns Inter-Trial Coherence (ITC), which measures the phase consistency across epochs. High ITC means the phase of the signal is consistent across trials, while low ITC indicates more phase variability.
    # decim=3: This reduces the temporal resolution by a factor of 3, which helps to speed up the computation.
    freqs = np.logspace(*np.log10([20, 45]), num=20)
    # num=8 specifies that 8 values should be generated between these two limits.
    # The result is 8 frequency values spaced in a logarithmic fashion between 6 and 35 Hz
    n_cycles = freqs / 2.0  # different number of cycle per frequency
    # return_itc=True: This returns Inter-Trial Coherence (ITC), which measures the phase consistency across epochs.
    # High ITC -> phase of the signal is consistent across trials, while low ITC indicates more phase variability.
    power, itc = mne.time_frequency.tfr_morlet(helper_epochs,
        freqs=freqs,
        n_cycles=n_cycles,
        average=True,
        return_itc=True,
        decim=3,
    )
    power.plot([0], baseline=(-0.3, 0), mode='logratio', title='Time-Frequency Power', cmap='inferno')
    # power: This contains the time-frequency representation of power for your data.
    # It shows how much power exists at each frequency at each time point.
    # itc: This contains the Inter-Trial Coherence,
    # which shows the consistency of the phase across trials at each frequency and time point.


# Now, all features are stored per epoch and per channel
# You can now concatenate all features into one large array (per epoch, per channel)

# Concatenate all features into one feature array for model training
name = ['baseline', 'motor', 'responses']
def concatenate_features(features, helper_avg_z_data, band_power, name):
    ## Step 1: Define column names for each feature type
    feature_names = ['ptp_channel_1', 'ptp_channel_2', 'rms_channel_1', 'rms_channel_2', 'var_channel_1', 'var_channel_2']
    z_score_names = ['z_score_channel_1', 'z_score_channel_2']
    band_power_names = [f'band_power_channel_1_freq_bin_{i}' for i in range(band_power.shape[2])] + \
                       [f'band_power_channel_2_freq_bin_{i}' for i in range(band_power.shape[2])]
    # Step 2: Concatenate all features
    # Concatenate features across axis 1 (for each epoch)
    all_features = np.hstack([
        features,  # (n epochs, 6)
        helper_avg_z_data,  # (n epochs, 2)
        band_power.reshape(len(band_power), -1)  # Flatten power bands to shape (76, 2*n_freq_bins)
    ])

    # Step 3: Create a DataFrame with the feature data
    # Combine the feature names
    column_names = feature_names + z_score_names + band_power_names
    features_df = pd.DataFrame(all_features, columns=column_names)

    # Step 4: Optionally save to CSV
    features_df.to_csv(features_path / f'{sub}_{name}_features.csv', index=False)

    return features_df

# For baseline:
baseline_feature_matrix = concatenate_features(baseline_features, baseline_avg_z_data, baseline_band_power, name=name[0])
# For motor:
motor_feature_matrix = concatenate_features(motor_features, motor_avg_z_data, motor_band_power, name=name[1])
# baseline features: (76, 3) -> 76 epochs: 1st column-> peak-to-peak amplitude; 2nd column-> RMS; 3rd column-> variance -> 3 values per epoch
# baseline_z_data: (76, 2, 601) -> 76 separate ndarrays (for each epoch); each table -> 2 rows (channels); 601 columns (time)
# baseline_band_power: (76, 2) -> 2 rows (for each channel), 76 columns(for each epoch)

#################
# EXTRACT FEATURES FROM EXPERIMENTAL DATA:
def extract_features_from_epochs(epochs_list):
    """
    Extract features from a list of MNE Epochs objects.

    Parameters:
    epochs_list : list of MNE Epochs objects

    Returns:
    concatenated_features : np.array, combined features for all epochs in all objects.
    """
    all_features = []

    # Iterate over all Epochs objects in the list
    for epochs in epochs_list:
        # Extract amplitude, RMS, variance for each epoch in this object
        features = amplitude_rms_var(epochs.get_data(copy=True))

        # Store the features for this particular set of epochs
        all_features.append(features)

    # Concatenate features across all Epoch objects into one array
    concatenated_features = np.vstack(all_features)

    return concatenated_features


def extract_power_band_features(epochs_list, fmin=20, fmax=45):
    """
    Extract power band features from a list of MNE Epochs objects.

    Parameters:
    epochs_list : list of MNE Epochs objects
    fmin : float, minimum frequency of the band
    fmax : float, maximum frequency of the band

    Returns:
    concatenated_band_power : np.array, combined band power features for all epochs
    """
    all_band_power = []

    # Iterate over all MNE Epochs objects
    for epochs in epochs_list:
        # Extract power band for each epoch
        band_power = welch(epochs, fmin=fmin, fmax=fmax)

        # Store the power band for this set of epochs
        all_band_power.append(band_power)

    # Concatenate all band power features across Epochs objects
    concatenated_band_power = np.vstack(all_band_power)

    return concatenated_band_power


def extract_z_scores_from_epochs(epochs_list):
    all_z_scores = []

    for epochs in epochs_list:
        _, _, _, _, avg_z_data = get_helpers_z_scores(epochs)
        all_z_scores.append(avg_z_data)

    return np.vstack(all_z_scores)


def classify_epochs(z_data_avg, baseline_threshold, true_response_threshold):
    classifications = []
    for z_score in z_data_avg:
        # Aggregate across channels, e.g., using the mean or max to get a single value
        z_score_aggregated = np.mean(z_score)  # Or np.max(z_score) if you prefer the maximum

        if z_score_aggregated > true_response_threshold:
            classifications.append('True Response')
        elif z_score_aggregated > baseline_threshold:
            classifications.append('Partial Error')
        else:
            classifications.append('No Response')
    return classifications

# Extract features from the experiment's epochs
target_features = extract_features_from_epochs(target_epochs)
distractor_features = extract_features_from_epochs(distractor_epochs)
responses_features = extract_features_from_epochs(responses_epochs)
true_responses_with_labels = np.hstack(responses_features)
# Extract power band features
target_band_power = extract_power_band_features(target_epochs)
distractor_band_power = extract_power_band_features(distractor_epochs)
responses_band_power = extract_power_band_features(responses_epochs)

# Extract z-scores
target_z_scores = extract_z_scores_from_epochs(target_epochs)
distractor_z_scores = extract_z_scores_from_epochs(distractor_epochs)
responses_z_scores = extract_z_scores_from_epochs(responses_epochs)

# Concatenate all features for target, distractor, and responses
target_feature_matrix = concatenate_features(target_features, target_z_scores, target_band_power, name=name[0])
distractor_feature_matrix = concatenate_features(distractor_features, distractor_z_scores, distractor_band_power, name=name[1])
responses_feature_matrix = concatenate_features(responses_features, responses_z_scores, responses_band_power, name=name[2])

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report

# Assuming baseline_feature_matrix is for 'no responses' (label = 0)
no_response_labels = np.zeros(baseline_feature_matrix.shape[0])  # Label all as '0'
true_response_labels = np.ones(responses_feature_matrix.shape[0])  # Label all as '1'

# Now, combine the features and labels for training
X_train = np.vstack([baseline_feature_matrix, responses_feature_matrix])  # Combine baseline and true responses
y_train = np.hstack([no_response_labels, true_response_labels])  # Combine labels for no response and true response

# Split training data (Optional step for testing)
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)


# Initialize the scaler
scaler = StandardScaler()

# Apply scaling to both training and testing datasets
X_train_scaled = scaler.fit_transform(X_train_split)
X_val_scaled = scaler.transform(X_val_split)
X_test_scaled = scaler.transform(target_feature_matrix.values)

# Initialize the SVM classifier with a non-linear kernel (e.g., 'rbf' for complex data)
svm_rbf = SVC(kernel='rbf', C=1, gamma='auto')  # You can adjust 'C' and 'gamma' for tuning

# Train the SVM model
svm_rbf.fit(X_train_split, y_train_split)

# Predict on validation data to check performance
y_pred = svm_rbf.predict(X_val_split)

# Evaluate the model's performance
accuracy = accuracy_score(y_val_split, y_pred)
report = classification_report(y_val_split, y_pred)

print(f"Validation Accuracy: {accuracy}")
print(f"Classification Report:\n{report}")

