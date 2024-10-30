# Required Libraries
import os
import mne
import librosa
import numpy as np
import matplotlib.pyplot as plt
from mtrf.model import TRF
from scipy.signal import hilbert
from sklearn.model_selection import KFold
from scipy.stats import zscore
import pandas as pd
from pathlib import Path

folder = r'C:/Users/vrvra/Desktop/trf_data_example'
file_path = r'C:/Users/vrvra/Desktop/trf_data_example/jr_a1_1.vhdr'

# todo: align files with condition: sub path, audio files based on block: voice
#  todo: add metadata to eeg
#  todo: epoch eeg around target and distractor events
#  todo: add animal wav files accordingly as well
#  todo: Helper Functions


def find_files(folder):
    """
    Search the folder for audio and EEG files.
    """
    eeg_file = None
    audio_file = None
    audio_files = []

    for file in os.listdir(folder):
        if file.endswith(('.fif', '.set', '.vhdr', '.dat')):
            eeg_file = os.path.join(folder, file)
        elif file.endswith(('.wav', '.ogg', '.mp3')):
            audio_file = os.path.join(folder, file)
            audio_files.append(audio_file)

    if not eeg_file or not audio_file:
        raise ValueError("No appropriate EEG or audio files found!")

    return eeg_file, audio_files


eeg_file, audio_files = find_files(folder)

def load_eeg_data(file_path):
    """
    Load EEG data using MNE depending on file type.
    """
    if file_path.endswith('.vhdr'):
        raw = mne.io.read_raw_brainvision(file_path, preload=True)
    else:
        raise ValueError("Unsupported EEG file format.")

    print(f"EEG Data Loaded: {file_path}")
    print(f"Sampling Rate: {raw.info['sfreq']} Hz")
    return raw

raw = load_eeg_data(file_path)

def get_marker_file(folder):
    for file in os.listdir(folder):
        print(file)
        if file.endswith(('vmrk')):
            marker_path = os.path.join(folder, file)
            vmrk = pd.read_csv(marker_path, delimiter='\t', header=None)
    return vmrk
vmrk = get_marker_file(folder)

def dataframe(vmrk):
    df = vmrk.iloc[10:]  # delete first 10 rows  (because they contain nothing useful for us)
    df = df.reset_index(drop=True, inplace=False)  # once rows are dropped, we need to reset index so it starts from 0 again
    df = df[0].str.split(',', expand=True).applymap(lambda x: None if x == '' else x) # separates info from every row into separate columns
    # information is split whenever a ',' comma is present; otherwise all info in each row is all under one column -> which sucks
    df = df.iloc[:, 1:3] # we keep only the columns 1:3, with all their data
    df.insert(0, 'Stimulus Type', None)  # we insert an additional column, which we will fill in later
    df.insert(2, 'Numbers', None)  # same here
    columns = ['Stimulus Stream', 'Position', 'Time Difference']  # we pre-define some columns of our dataframe;
    # position is time in data samples
    df.columns = ['Stimulus Type'] + [columns[0]] + ['Numbers'] + [columns[1]]  # we re-order our columns
    df['Timepoints'] = None
    df['Voice'] = 'Voice1'
    return df

df = dataframe(vmrk)
default_dir = Path.cwd()
import json
performance_events = default_dir / 'data' / 'misc' / 'performance_events.json'
# load performance_events dictionary:
with open(performance_events, 'r') as file:
    markers_dict = json.load(file)

# define events by creating separate dictionaries with keys and their corresponding values:
s1_events = markers_dict['s1_events']
s2_events = markers_dict['s2_events']  # stimulus 2 markers
response_events = markers_dict['response_events']  # response markers

def define_stimuli(df): # if you want to manipulate a variable that was created within a function, enter here in parenthesis
    # define our Stimulus Type:
    for index, stim_mrk in enumerate(df['Stimulus Stream']): # we go through the cells of Stimulus Stream
        if stim_mrk in s1_events.keys(): # if the key within one cell, matches an element from the s1_events dictionary
            df.at[index, 'Stimulus Type'] = 's1'  # enter value 's1' in the corresponding cell under Stimulus Type
            df.at[index, 'Numbers'] = next(value for key, value in s1_events.items() if key == stim_mrk) # if the key == stimulus marker,
            # iterate through the dictionary s1_events
            # and find the corresponding value to that key -> that value is the actual number said during that stimulus
            # for the numbers column, find the corresponding key/number from the s1_events dictionary, that matches the
            # s1 event at that specific row
        elif stim_mrk in s2_events.keys():  # same process for s2
            # print('stimulus marker is type 2')
            df.at[index, 'Stimulus Type'] = 's2'
            df.at[index, 'Numbers'] = next(value for key, value in s2_events.items() if key == stim_mrk)
        elif stim_mrk in response_events.keys(): # same for responses
            # print('stimulus marker is type response')
            df.at[index, 'Stimulus Type'] = 'response'
            df.at[index, 'Numbers'] = next(value for key, value in response_events.items() if key == stim_mrk)
    return df  # always return manipulated variable, if you are to further manipulate it in other functions


def clean_rows(df):
    # drop any remaining rows with None Stimulus Types (like: 'S 64')
    rows_to_drop = []
    for index, stim_mrk in enumerate(df['Stimulus Stream']):
        if stim_mrk not in s1_events.keys() and stim_mrk not in s2_events.keys() and stim_mrk not in response_events.keys():
            rows_to_drop.append(index)
    # Drop the marked rows from the DataFrame
    df.drop(rows_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

df['Timepoints'] = df['Position'].astype(float) / 500  # divide the datapoints by the sampling rate of 500
df['audio_features'] = None

sr = 24414
# todo: add animal sounds

def extract_audio_features(audio_files, sr, feature_type="envelope"):
    """
      Extract features from the audio file (envelope or mel-spectrogram).
      """
    envelopes = []
    mel_spectrograms = []
    audios = []

    # Load all audio files into the audios list
    for audio_file in audio_files:
        audio, _ = librosa.load(audio_file, sr=sr)
        audios.append(audio)

    # Extract the desired features for each audio file
    for audio in audios:
        if feature_type == "envelope":
            envelope = np.abs(hilbert(audio))  # Compute the amplitude envelope
            envelopes.append(envelope)
        elif feature_type == "mel":
            mel_spectrogram = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=16)
            mel_spectrograms.append(mel_spectrogram)

    # Return the appropriate feature list based on the feature type
    if feature_type == "envelope":
        return envelopes
    elif feature_type == "mel":
        return mel_spectrograms


envelopes = extract_audio_features(audio_files, sr, feature_type="envelope")


for index, row in df.iterrows():
    number = row['Numbers']  # Access the 'Numbers' value for the current row
    if number == 1:
        df.at[index, 'audio_features'] = envelopes[0]
    elif number == 2:
        df.at[index, 'audio_features'] = envelopes[1]
    elif number == 3:
        df.at[index, 'audio_features'] = envelopes[2]
    elif number == 4:
        df.at[index, 'audio_features'] = envelopes[3]
    elif number == 5:
        df.at[index, 'audio_features'] = envelopes[4]
    elif number == 6:
        df.at[index, 'audio_features'] = envelopes[5]
    elif number == 8:
        df.at[index, 'audio_features'] = envelopes[6]
    elif number == 9:
        df.at[index, 'audio_features'] = envelopes[7]

df = df.dropna(subset=['audio_features'])

def get_events(raw, target_events):  # if a1 or e1: choose s1_events; if a2 or e2: s2_events
    events = mne.events_from_annotations(raw)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in target_events.values()]
    events = np.array(filtered_events)
    return events
events1 = get_events(raw, s1_events)
events2 = get_events(raw, s2_events)


# epoch raw
def epochs(raw, event_dict, events):
    # Counter(events1[:, 2])
    tmin = -0.2
    tmax = 0.9
    epoch_parameters = [tmin, tmax, event_dict]
    tmin, tmax, event_ids = epoch_parameters
    event_ids = {key: val for key, val in event_ids.items() if val in events[:, 2]}

    epochs = mne.Epochs(raw,
                               events,
                               event_id=event_ids,
                               tmin=tmin,
                               tmax=tmax,
                               baseline=(None, 0),
                               detrend=0,  # should we set it here?
                               preload=True)
    return epochs

s1_epochs = epochs(raw, s1_events, events1)
s2_epochs = epochs(raw, s2_events, events2)

s1_df = df[df['Stimulus Type'] == 's1']
s1_epochs.metadata = s1_df

s2_df = df[df['Stimulus Type'] == 's2']
s2_epochs.metadata = s2_df

def prepare_eeg_data(eeg_raw, duration_sec=None, downsample_rate=100):
    """
    Prepare EEG data by resampling and extracting a specific duration if needed.
    """
    # Resample EEG data
    eeg_raw.resample(downsample_rate)

    # Optionally, extract a fixed duration
    if duration_sec:
        eeg_raw.crop(tmax=duration_sec)

    eeg_data = eeg_raw.get_data().T  # Transpose to time-by-channel format
    eeg_data = zscore(eeg_data, axis=0)  # Z-score normalization
    return eeg_data, downsample_rate

eeg_data, downsample_rate  = prepare_eeg_data(eeg_raw, duration_sec=None, downsample_rate=100)


# Main Execution

# Step 1: Specify folder to search for files
folder = input("Enter the folder path containing the EEG and audio files: ")

# Step 2: Find files
eeg_file, audio_file = find_files(folder)
print(f"Found EEG file: {eeg_file}")
print(f"Found Audio file: {audio_file}")

# Step 3: Load EEG data
eeg_raw = load_eeg_data(eeg_file)

# Step 4: Extract audio features (Envelope)
sampling_rate = int(eeg_raw.info['sfreq'])
audio_features = extract_audio_features(audio_file, sr=sampling_rate, feature_type="envelope")

# Step 5: Prepare EEG data (downsampled to match audio if necessary)
eeg_data, downsample_rate = prepare_eeg_data(eeg_raw)

# Step 6: Visualize Raw EEG and Audio Data
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(eeg_data[:1000, 0], label="EEG (First Channel)")
plt.title("EEG Data (First Channel)")
plt.subplot(2, 1, 2)
plt.plot(audio_features[:1000], label="Audio Envelope", color='green')
plt.title("Audio Envelope")
plt.show()

# Step 7: Train Forward and Backward Models

# Step 1: Ensure audio_features and eeg_data have the same number of samples
n_samples = min(audio_features.shape[0], eeg_data.shape[0])

# Trim both datasets to match the number of samples
audio_features = audio_features[:n_samples]
eeg_data = eeg_data[:n_samples]

# Step 2: Train Forward and Backward Models
tmin, tmax = 0, 0.4  # Time range (in seconds)
regularization = 1e3  # Regularization parameter

fwd_trf = TRF(direction=1)
back_trf = TRF(direction=-1)

# Train forward model (audio -> EEG)
fwd_trf.train(audio_features, eeg_data, downsample_rate, tmin, tmax, regularization)
predicted_eeg, r_forward = fwd_trf.predict(audio_features, eeg_data)

# Train backward model (EEG -> audio)
back_trf.train(eeg_data, audio_features, downsample_rate, tmin, tmax, regularization)
reconstructed_audio, r_backward = back_trf.predict(eeg_data, audio_features)

# Visualize the results
print("Forward Model (Audio -> EEG) correlation:", r_forward)
print("Backward Model (EEG -> Audio) correlation:", r_backward)


# Step 8: Visualizations for Forward and Backward Model

import numpy as np
import matplotlib.pyplot as plt

# Check shapes of EEG, predicted EEG, and audio data
print(f"Shape of EEG data: {eeg_data.shape}")
print(f"Shape of Predicted EEG: {predicted_eeg.shape}")
print(f"Shape of Audio Features: {audio_features.shape}")

# Convert reconstructed_audio to NumPy array if it is a list and squeeze it
if isinstance(reconstructed_audio, list):
    reconstructed_audio = np.array(reconstructed_audio)

print(f"Shape of Reconstructed Audio: {reconstructed_audio.shape}")

# Squeeze reconstructed_audio to remove extra dimensions
reconstructed_audio = np.squeeze(reconstructed_audio)

# Ensure both EEG and predicted EEG have the same number of samples
n_samples = min(eeg_data.shape[0], predicted_eeg.shape[0])

# Plot the actual vs predicted EEG for the first channel
plt.figure(figsize=(10, 8))  # Increase figure size

plt.subplot(2, 1, 1)
plt.plot(eeg_data[:n_samples, 0], label='Actual EEG (First Channel)', color='blue')
plt.plot(predicted_eeg[:n_samples, 0], label='Predicted EEG (First Channel)', linestyle='--', color='orange')
plt.title("Forward Model: Actual vs Predicted EEG")
plt.legend()

# For the audio, make sure we have matching dimensions
n_audio_samples = min(audio_features.shape[0], reconstructed_audio.shape[0])

plt.subplot(2, 1, 2)
plt.plot(audio_features[:n_audio_samples, 0], label='Actual Audio', color='green')
plt.plot(reconstructed_audio[:n_audio_samples, 0], label='Reconstructed Audio', linestyle='--', color='red')
plt.title("Backward Model: Actual vs Reconstructed Audio")
plt.legend()

# Adjust layout to prevent overlap
plt.tight_layout()

plt.show()

# Step 9: Correlation Coefficients
print(f"Forward Model Correlation: {r_forward}")
print(f"Backward Model Correlation: {r_backward}")