# Import necessary libraries
from pathlib import Path
import os
import json
import mne
import librosa
import librosa.display
import scipy.signal
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import LeaveOneOut
from mtrf.model import TRF
from scipy.stats import zscore


sub_input = input("Give sub number as subn (n for number): ")
sub = [sub.strip() for sub in sub_input.split(',')]
condition = input('Please provide condition (exp. EEG): ')

cm = 1 / 2.54
name = sub_input

# 0. LOAD THE DATA
default_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data')
preprocessed_eeg = Path(default_dir/f'eeg/preprocessed/results/{name}')
results_path = default_dir / 'eeg' / 'preprocessed' /'results' / name
trf_path = results_path / 'TRF'
psd_path = results_path / 'psd'
erp_path = results_path / 'ERP'
json_path = default_dir / 'misc'
envelopes_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/voices_english/downsampled/envelopes')
tables_path = results_path / 'tables'

# Marker dictionaries and mappings:
markers_dict = {
    's1_events': {  # Stimulus 1 markers
        'Stimulus/S  1': 1,
        'Stimulus/S  2': 2,
        'Stimulus/S  3': 3,
        'Stimulus/S  4': 4,
        'Stimulus/S  5': 5,
        'Stimulus/S  6': 6,
        'Stimulus/S  8': 8,
        'Stimulus/S  9': 9
    },
    's2_events': {  # Stimulus 2 markers
        'Stimulus/S 65': 65,
        'Stimulus/S 66': 66,
        'Stimulus/S 67': 67,
        'Stimulus/S 68': 68,
        'Stimulus/S 69': 69,
        'Stimulus/S 70': 70,
        'Stimulus/S 72': 72,
        'Stimulus/S 73': 73
    },
    'response_events': {  # Response markers
        'Stimulus/S129': 129,
        'Stimulus/S130': 130,
        'Stimulus/S131': 131,
        'Stimulus/S132': 132,
        'Stimulus/S133': 133,
        'Stimulus/S134': 134,
        'Stimulus/S136': 136,
        'Stimulus/S137': 137
    }
}


s1_events = markers_dict['s1_events']
s2_events = markers_dict['s2_events']  # stimulus 2 markers
response_events = markers_dict['response_events']  # response markers

mapping_path = json_path / 'events_mapping.json'
with open(mapping_path, 'r') as file:
    events_mapping = json.load(file)
s1_mapping = events_mapping[0]
s2_mapping = events_mapping[1]
response_mapping = events_mapping[2]

animal_stimuli_events = {'Stimulus/S  7': 7, 'Stimulus/S 71': 71}
if condition == 'a1' or condition == 'e1':
    # Select the second pair ('Stimulus/S 71': 7)
    selected_animal_event = {'Stimulus/S 71': animal_stimuli_events['Stimulus/S 71']}
    s2_events = {**s2_events, **selected_animal_event}
elif condition == 'a2' or condition == 'e2':
    # Select the first pair ('Stimulus/S  7': 7)
    selected_animal_event = {'Stimulus/S  7': animal_stimuli_events['Stimulus/S  7']}
    s1_events = {**s1_events, **selected_animal_event}

# Combine with other stimuli events
all_events = {**s1_events, **s2_events, **response_events, **selected_animal_event}

# load data
raw_clean = mne.io.read_raw_fif(preprocessed_eeg / f'{name}_{condition}_preproccessed-raw.fif', preload=True)

# Read all dataframes:
target_nums_df = pd.read_csv(tables_path / f'{sub_input}_{condition}_target_nums_events_raw.csv')
distractor_nums_df = pd.read_csv(tables_path / f'{sub_input}_{condition}_distractor_nums_events_raw_raw.csv')
non_targets_target_df = pd.read_csv(tables_path / f'{sub_input}_{condition}_target_non_targets_events_raw.csv')
non_targets_distractor_df = pd.read_csv(tables_path / f'{sub_input}_{condition}_distractor_non_targets_events_raw.csv')
deviant_sounds_df = pd.read_csv(tables_path / f'{sub_input}_{condition}_deviant_sounds_events_raw.csv')

def transform_events(df, mapping):
    events = df.iloc[:, 1:3]
    events = np.array(events)
    events = np.hstack((
        events[:, [0]],  # First column (timepoints)
        np.zeros((events.shape[0], 1), dtype=int),  # New column of zeros
        events[:, [1]]  # Second column (event IDs)
    ))
    # Get unique event IDs from target_nums_events
    valid_event_ids = np.unique(events[:, 2])  # Extract the third column (event IDs)

    # Filter s1_events to include only the valid event IDs
    filtered_events = {k: v for k, v in mapping.items() if v in valid_event_ids}

    # Debugging: Print filtered event IDs
    print("Filtered Event IDs:", filtered_events)
    return events, filtered_events

target_nums_events, target_nums_mapping = transform_events(target_nums_df, mapping=s1_events)
distractor_nums_events, distractor_nums_mapping = transform_events(distractor_nums_df, mapping=s2_events)

non_targets_target_events, non_targets_target_mapping = transform_events(non_targets_target_df, mapping=s1_events)
non_targets_distractor_events, non_targets_distractor_mapping = transform_events(non_targets_distractor_df, mapping=s2_events)

deviant_sounds_events, animals_mapping = transform_events(deviant_sounds_df, mapping=selected_animal_event)


def get_events(eeg_data, all_events):  # get all events of selected data
    # Extract events and their corresponding IDs from annotations
    stream_events, stream_event_ids = mne.events_from_annotations(eeg_data)

    # Dynamically find the "New Segment" event ID
    new_segment_id = None
    for description, event_id in stream_event_ids.items():
        if description == 'New Segment/':  # Check for the "New Segment" label
            new_segment_id = event_id
            break

    # If "New Segment" is found, ensure it's added to the event mappings
    if new_segment_id is not None:
        if new_segment_id == 1:
            new_segment_id = 99999
            stream_event_ids['New Segment/'] = 99999
        elif new_segment_id == 99999:
            all_events['New Segment/'] = new_segment_id

    # Update the event IDs in the stream_events array based on all_events
    # todo: transform events of raw_clean based on all_events and dictionaries
    for i in range(len(stream_events)):
        current_value = stream_events[i, 2]
        if current_value == 1:
            stream_events[i, 2] = 99999 # convert New Segment/ event into 99999
    for i in range(len(stream_events)):
        current_value = stream_events[i, 2]
        stream_keys = list(stream_event_ids.keys())
        values_list = list(stream_event_ids.values())  # [101, 102, 103]
        values_index = values_list.index(current_value)  # the index of the value from the stream_event_ids
        matching_key = stream_keys[values_index]  # the key from stream_event_ids
        # find corresponding value from all_events:
        if matching_key in all_events.keys():
            actual_value = all_events[matching_key]
            stream_events[i, 2] = actual_value
    return stream_events, stream_event_ids


stream_events, stream_event_ids = get_events(raw_clean, all_events)

#todo: properly update dict of events
# Update event IDs
# Update event dictionary to match the predefined dictionary
updated_event_ids = {k: v for k, v in all_events.items() if k in stream_event_ids}
stream_events_preprocessed_df = pd.DataFrame(stream_events)
columns = stream_events_preprocessed_df.columns.tolist()
columns[0] = 'Timepoints'
columns[2] = 'Numbers'
stream_events_preprocessed_df.columns = columns

target_df_processed = {}
distractor_df_processed = {}

if condition == 'a1' or condition == 'e1':
    # Filter events for stimuli 1 to 9
    target_df_processed = stream_events_preprocessed_df[stream_events_preprocessed_df['Numbers'].isin(range(1, 10))]
    # Filter events for stimuli 65 to 73
    distractor_df_processed = stream_events_preprocessed_df[stream_events_preprocessed_df['Numbers'].isin(range(65, 74))]
elif condition == 'a2' or condition == 'e2':
    distractor_df_processed = stream_events_preprocessed_df[stream_events_preprocessed_df['Numbers'].isin(range(1, 10))]
    target_df_processed = stream_events_preprocessed_df[stream_events_preprocessed_df['Numbers'].isin(range(65, 74))]

# load original events dataframe:
events_df_raw_path = results_path / 'tables'
target_df_raw = events_df_raw_path / f'{name}_{condition}_target_stream_events_raw.csv'
distractor_df_raw = events_df_raw_path / f'{name}_{condition}_distractor_stream_events_raw.csv'

target_events_raw_df = pd.read_csv(target_df_raw, delimiter=',')
distractor_events_raw_df = pd.read_csv(distractor_df_raw, delimiter=',')

# filter events raw dataframes to match clean EEG data events:
target_events_timepoints_raw = np.unique(target_events_raw_df['Timepoints'])
distractor_events_timepoints_raw = np.unique(distractor_events_raw_df['Timepoints'])
# todo: get events from raw_clean
filtered_target_events_raw = target_events_raw_df[target_events_raw_df['Timepoints'].isin(target_df_processed['Timepoints'].values)]
filtered_distractor_events_raw = distractor_events_raw_df[distractor_events_raw_df['Timepoints'].isin(distractor_df_processed['Timepoints'].values)]


# alter continuous envelopes based on Envelope column of target and distractor dfs:
def get_selected_envelope(eeg, df):
    total_samples = int(eeg.n_times)  # Total samples in the EEG recording
    continuous_envelope = np.zeros(total_samples)

    timepoints = df['Timepoints']
    envelopes = df['Envelope']

    envelope_list = []
    for timepoint, envelope in zip(timepoints, envelopes):
        audio_data = np.load(envelope)
        envelope_list.append((timepoint, audio_data))
        # Step 1: Insert Target Envelopes into Continuous Array
    for timepoint, envelope in envelope_list:
        start_sample = int(timepoint)  # Starting position in samples
        end_sample = start_sample + len(envelope)  # Determine the ending position

        # Avoid overflow beyond the array boundary
        if end_sample > total_samples:
            envelope = envelope[:total_samples - start_sample]  # Truncate if necessary

        # Insert the envelope into the continuous array
        continuous_envelope[start_sample:end_sample] = envelope
    return continuous_envelope

target_nums_envelope = get_selected_envelope(raw_clean, target_nums_df)
distractor_nums_envelope = get_selected_envelope(raw_clean, distractor_nums_df)
non_targets_target_envelope = get_selected_envelope(raw_clean, non_targets_target_df)
non_targets_distractor_envelope = get_selected_envelope(raw_clean, non_targets_distractor_df)
deviant_sounds_envelope = get_selected_envelope(raw_clean, deviant_sounds_df)


# Verify the shapes
print("Target Continuous Envelope Shape:", target_continuous_envelope.shape)
print("Distractor Continuous Envelope Shape:", distractor_continuous_envelope.shape)

envelopes_streams_path = results_path / 'envelope_streams'
os.makedirs(envelopes_streams_path, exist_ok=True)

# save envelope streams:
with open(envelopes_streams_path/ f'{name}_{condition}_target_continuous_envelope.txt', 'w') as f:
    for value in target_continuous_envelope:
        f.write(f"{value}\n")

with open(envelopes_streams_path/ f'{name}_{condition}_distractor_continuous_envelope.txt', 'w') as d_f:
    for d_value in distractor_continuous_envelope:
        d_f.write(f'{d_value}\n')



sampling_rate = 500  # Hz
eeg_data = zscore(raw_clean.get_data(), axis=0)  # scipy function

# Step 2: Normalize the EEG and audio features
time_axis = np.arange(1000) / sampling_rate
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(time_axis, eeg_data[0][1000:2000], label="EEG (First Channel)")
plt.title("EEG Data (First Channel)")
plt.subplot(2, 1, 2)
plt.plot(time_axis, target_continuous_envelope[1000:2000], label="Audio Envelope", color='green')
plt.title("Audio Envelope")
plt.show()

# Step 3: downsample data to 200Hz
from scipy.signal import resample
new_sfreq = 100  # Hz
downsample_factor = sampling_rate // new_sfreq
# Downsample EEG data
eeg_data_downsampled = resample(eeg_data, eeg_data.shape[1] // downsample_factor, axis=1)
eeg_data_transposed = eeg_data_downsampled.T  # Shape: (n_samples, 65)

target_continuous_envelope_downsampled = resample(target_continuous_envelope, target_continuous_envelope.shape[0] // downsample_factor)
distractor_continuous_envelope_downsampled = resample(distractor_continuous_envelope, distractor_continuous_envelope.shape[0] // downsample_factor)


# Step 5: Segment both EEG and audio data into 30-second trials
def segment_data_2d(data, segment_duration_sec, new_sfreq):
    segment_size = segment_duration_sec * new_sfreq  # Convert duration to samples
    num_segments = data.shape[1] // segment_size  # Determine the number of segments
    if num_segments == 0:
        raise ValueError(f"Not enough data to segment into {segment_duration_sec}-second chunks")
    segments = [data[:, i:i + segment_size].T for i in range(0, data.shape[1], segment_size)]
    return segments


def segment_data_1d(data, segment_duration_sec, new_sfreq):
    segment_size = segment_duration_sec * new_sfreq  # Convert duration to samples
    num_segments = data.shape[0] // segment_size  # Determine the number of segments
    if num_segments == 0:
        raise ValueError(f"Not enough data to segment into {segment_duration_sec}-second chunks")
    segments = [data[i:i + segment_size].reshape(-1, 1) for i in range(0, len(data), segment_size)]
    return segments

# Segment the EEG and audio data (2D for EEG, 1D for audio)
segment_duration_sec = 30  # 30 seconds per segment (trial)
eeg_data_segments = segment_data_2d(eeg_data_downsampled, segment_duration_sec, new_sfreq)
target_envelope_segments = segment_data_1d(target_continuous_envelope_downsampled, segment_duration_sec, new_sfreq)
distractor_envelope_segments = segment_data_1d(distractor_continuous_envelope_downsampled, segment_duration_sec, new_sfreq)


# Step 6: setting the mTRF parameters:
# Step 5: Setting up the mTRF model parameters
direction_forward = 1  # Stimulus -> EEG
direction_backward = -1  # EEG -> Stimulus
sampling_rate = new_sfreq
regularization = 1e3  # Regularization parameter
target_n_samples = min(eeg_data_segments[0].shape[0], target_envelope_segments[0].shape[0])
distractor_n_samples = min(eeg_data_segments[0].shape[0], distractor_envelope_segments[0].shape[0])

# Calculate the maximum lag based on tmin and the adjusted tmax
def get_params(n_samples):
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
    return tmin, tmax, min_lag_samples, max_lag_samples

target_tmin, target_tmax, target_min_lag_samples, target_max_lag_samples = get_params(target_n_samples)
distractor_tmin, distractor_tmax, distractor_min_lag_samples, distractor_max_lag_samples = get_params(distractor_n_samples)


# Initialize TRF models
fwd_trf = TRF(direction=direction_forward)
back_trf = TRF(direction=direction_backward)
print(f"TRF models initialized with target tmax={target_tmax:.3f} seconds (target max lag: {target_max_lag_samples} samples)")
print(f"TRF models initialized with distractor tmax={distractor_tmax:.3f} seconds (distractor max lag: {distractor_max_lag_samples} samples)")

eeg_data_trials = eeg_data_segments
target_envelope_trials = target_envelope_segments
distractor_envelope_trials = distractor_envelope_segments

# Check length of each trial and drop the last segment if needed
eeg_data_trials = [eeg for eeg in eeg_data_trials if eeg.shape[0] == segment_duration_sec * new_sfreq]
target_envelope_trials = [env for env in target_envelope_trials if env.shape[0] == segment_duration_sec * new_sfreq]
distractor_envelope_trials = [env for env in distractor_envelope_trials if env.shape[0] == segment_duration_sec * new_sfreq]

print(f"Number of EEG trials after dropping short segments: {len(eeg_data_trials)}")

# Step 7: Train and test the model on Stream 1 and Stream 2 trials
target_stream_results = []
distractor_stream_results = []

for eeg_trial, target_trial, distractor_trial in zip(eeg_data_trials, target_envelope_trials,
                                               distractor_envelope_trials):
    # Debugging: Print shapes before training
    print(f"Training on Stream 1 trial shape: {target_trial.shape}, EEG trial shape: {eeg_trial.shape}")

    # Train on Stream 1
    back_trf.train(eeg_trial, target_trial, sampling_rate, target_tmin, target_tmax, regularization)
    target_reconstructed_audio, r_backward_target = back_trf.predict(eeg_trial, target_trial)
    target_stream_results.append(r_backward_target)

    # Train on Stream 2
    back_trf.train(eeg_trial, distractor_trial, sampling_rate, distractor_tmin, distractor_tmax, regularization)
    distractor_reconstructed_audio, r_backward_distractor = back_trf.predict(eeg_trial, distractor_trial)
    distractor_stream_results.append(r_backward_distractor)

# Visualization - Compare the correlation for Stream 1 vs Stream 2
plt.figure(figsize=(10, 6))
plt.plot(target_stream_results, label="Target Stream")
plt.plot(distractor_stream_results, label="Distractor Stream")
plt.title(f'Correlation of EEG with Target Stream vs Distractor Stream')
plt.xlabel('Trial')
plt.ylabel('Correlation')
plt.legend()
plt.show()

# Interpretation
attended_stream = 'Target Stream' if np.mean(target_stream_results) > np.mean(distractor_stream_results) else 'Distractor'
print(f"The participant most likely attended to: {attended_stream}")


# from sklearn.model_selection import KFold
#
# # Define k-fold cross-validation with k=5
# kf = KFold(n_splits=5)
# correlations = []
#
# # Set a maximum number of samples for lag calculations
# min_samples_for_test = 50  # Example threshold to ensure a reasonable test size
#
# # Define time lags
# max_lag_samples = int((target_tmax - target_tmin) * sampling_rate)
#
# # Store correlation results for each fold
# correlation_per_fold = []
#
# for fold, (train_idx, test_idx) in enumerate(kf.split(eeg_data_trials)):
#     print(train_idx, test_idx
#     # Get training and test data
#     eeg_train, eeg_test = eeg_data_trials[train_idx], eeg_data_trials[test_idx]
#     audio_train, audio_test = target_envelope_trials[train_idx], target_envelope_trials[test_idx]
#
#     # Check number of samples in the training and test sets
#     n_samples_train = eeg_train.shape[0]
#     n_samples_test = eeg_test.shape[0]
#
#     # Skip iterations where the test set size is too small
#     if n_samples_test < min_samples_for_test:
#         print(f"Skipping iteration: Test set contains only {n_samples_test} samples, which is too few for the maximum lag.")
#         continue
#
#     # Adjust tmax if the lag exceeds the number of samples
#     if max_lag_samples > n_samples_train:
#         print(f"Warning: Max lag ({max_lag_samples} samples) exceeds training signal length ({n_samples_train} samples).")
#         max_lag_samples = n_samples_train
#         target_tmax_adjusted = max_lag_samples / sampling_rate
#     else:
#         target_tmax_adjusted = target_tmax
#
#     # Train the model on the training samples
#     back_trf.train(audio_train, eeg_train, sampling_rate, target_tmin, target_tmax_adjusted, regularization)
#
#     # Test the model on the left-out sample
#     if max_lag_samples > n_samples_test:
#         print(f"Warning: Max lag ({max_lag_samples} samples) exceeds test signal length ({n_samples_test} samples). Reducing max lag.")
#         max_lag_samples = n_samples_test
#         tmax_adjusted = max_lag_samples / sampling_rate
#
#     # Predict EEG response
#     predicted, _ = back_trf.predict(audio_test, eeg_test)
#
#     # Convert predicted to numpy array if it's a list
#     predicted = np.array(predicted).squeeze()
#
#     # Calculate correlation for the test sample
#     corr = np.corrcoef(eeg_test.squeeze(), predicted)[0, 1]
#     correlations.append(corr)
#
#     # Store correlation per fold
#     correlation_per_fold.append(corr)
#     print(f"Fold {fold + 1} correlation: {corr}")
#
# # Print the average correlation coefficient
# mean_correlation = np.mean(correlations)
# print(f"Mean Correlation across k-Fold CV: {mean_correlation}")
#
# # Visualization
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, len(correlation_per_fold) + 1), correlation_per_fold, marker='o', linestyle='-', color='b', label='Correlation per Fold')
# plt.axhline(y=mean_correlation, color='r', linestyle='--', label=f'Mean Correlation: {mean_correlation:.2f}')
# plt.title('Correlation per Fold in k-Fold Cross-Validation')
# plt.xlabel('Fold Number')
# plt.ylabel('Correlation Coefficient')
# plt.legend()
# plt.grid(True)
# plt.show()
#
#
# # Step 9: Auto-correlation and Cross-correlation
# print("Visualizing Auto-correlation and Cross-correlation")
# eeg_auto_corr = np.correlate(eeg_data, eeg_data, mode='full')
# audio_auto_corr = np.correlate(audio_features, audio_features, mode='full')
# cross_corr = np.correlate(audio_features, eeg_data, mode='full')
#
#
# # Plot the EEG auto-correlation
# plt.figure(figsize=(10, 4))
# plt.plot(eeg_auto_corr, label='EEG Auto-correlation', color='purple')
# plt.title('EEG Auto-correlation')
# plt.legend()
# plt.show()
#
# # Plot the Audio Features auto-correlation
# plt.figure(figsize=(10, 4))
# plt.plot(audio_auto_corr, label=f'{audio_feature_type.capitalize()} Auto-correlation', color='orange')
# plt.title(f'Audio Features ({audio_feature_type}) Auto-correlation')
# plt.legend()
# plt.show()
#
#
# # Plot the cross-correlation
# plt.figure(figsize=(10, 4))
# plt.plot(cross_corr, label='Cross-correlation (Audio Features vs EEG)', color='red')
# plt.title('Cross-correlation between Audio Features and EEG')
# plt.legend()
# plt.show()
#
#
# # Step 10: Explore the Effect of Regularization (Lambda) on TRF
# lambdas = [1e0, 1e3, 1e6]  # Example lambda values
# for lam in lambdas:
#     # Train the forward model with different lambda values
#     fwd_trf.train(audio_features, eeg_data, sampling_rate, tmin, tmax, lam)
#     predicted_eeg, _ = fwd_trf.predict(audio_features, eeg_data)
#
#     # Convert predicted_eeg to a NumPy array if it's not already
#     predicted_eeg = np.array(predicted_eeg)
#
#     # Reshape or squeeze the predicted EEG to remove any extra dimensions
#     predicted_eeg = predicted_eeg.squeeze()
#
#     # Visualize the effect of different lambdas
#     plt.figure(figsize=(10, 4))
#     plt.plot(predicted_eeg[:300], label=f'Predicted EEG (Lambda={lam})', linestyle='--')
#     plt.title(f'Forward Model with Lambda={lam} (First 300 samples)')
#     plt.legend()
#     plt.show()
#
# # Step 11: Visualizing Time-Shift Effects on TRF
#
# # Convert predicted_eeg to a NumPy array if it's not already
# predicted_eeg = np.array(predicted_eeg)
#
# # Reshape or squeeze the predicted EEG to remove any extra dimensions
# predicted_eeg = predicted_eeg.squeeze()
#
# # Define time shifts in samples
# time_shifts = np.arange(int(tmin * sampling_rate), int(tmax * sampling_rate))
#
# # Ensure we truncate the predicted EEG to match the number of time shifts (lags)
# predicted_eeg_truncated = predicted_eeg[:len(time_shifts)]
#
# print(f"Time shifts applied: {time_shifts} (samples)")
#
# # Plot the mean response over time shifts
# plt.figure(figsize=(10, 4))
# plt.plot(time_shifts, predicted_eeg_truncated, label='Predicted EEG Mean Response over Time')
# plt.title('Effect of Time Shift on Predicted EEG Response')
# plt.xlabel('Time Lag (samples)')
# plt.ylabel('Mean EEG Response')
# plt.legend()
# plt.show()
#
#
#
# # Step 12: Summary of the analysis
# print("Summary:")
# print("- We loaded EEG and audio data, extracted features, trained forward and backward models.")
# print("- We visualized actual vs predicted EEG and reconstructed audio features.")
# print("- We applied Leave-One-Out Cross-Validation and explored the effect of different regularization parameters (lambdas).")
#
#


