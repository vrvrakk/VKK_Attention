import mne
import numpy as np
from autoreject import AutoReject

# Step 1: Load the EEG data
baseline_file = 'path_to_baseline_file.vhdr'
stimuli_file = 'path_to_stimuli_file.vhdr'

baseline_raw = mne.io.read_raw_brainvision(baseline_file, preload=True)
stimuli_raw = mne.io.read_raw_brainvision(stimuli_file, preload=True)

# Step 2: Preprocess the data (e.g., filter the data, set montage)
baseline_raw.filter(l_freq=1.0, h_freq=40.0)
stimuli_raw.filter(l_freq=1.0, h_freq=40.0)

montage = mne.channels.make_standard_montage('standard_1020')
baseline_raw.set_montage(montage)
stimuli_raw.set_montage(montage)

# Step 3: Identify and Interpolate Bad Channels
stimuli_raw.info['bads'] = []  # Reset bad channels list if any
stimuli_raw.plot()  # Manually identify bad channels
stimuli_raw.interpolate_bads()

# Step 4: Manually create epochs for the baseline recording
epoch_duration = 1.0  # Duration of each epoch in seconds
overlap = 0.5  # Overlap between epochs in seconds

n_samples = len(baseline_raw.times)
epoch_length = int(epoch_duration * baseline_raw.info['sfreq'])
step_size = int((epoch_duration - overlap) * baseline_raw.info['sfreq'])
events_baseline = np.array([[i, 0, 1] for i in range(0, n_samples - epoch_length, step_size)])

baseline_epochs = mne.Epochs(baseline_raw, events_baseline, event_id=1, tmin=0, tmax=epoch_duration, baseline=None, preload=True)

# Step 5: Create epochs for the auditory stimuli
events_stimuli = mne.find_events(stimuli_raw)
tmin, tmax = -0.2, 0.5  # Start and end times for each epoch
stimuli_epochs = mne.Epochs(stimuli_raw, events_stimuli, event_id=None, tmin=tmin, tmax=tmax, baseline=(None, 0), preload=True)

# Step 6: Compute the noise profile from baseline epochs
baseline_data = baseline_epochs.get_data()
noise_profile = np.mean(baseline_data, axis=0)  # Compute the average noise profile

# Step 7: Apply the noise profile to stimuli epochs to remove noise
stimuli_data = stimuli_epochs.get_data()
cleaned_stimuli_data = stimuli_data - noise_profile  # Remove noise by subtracting the noise profile

# Create new Epochs object with cleaned data
cleaned_stimuli_epochs = mne.EpochsArray(cleaned_stimuli_data, stimuli_epochs.info, events_stimuli, tmin, baseline=(None, 0))

# Step 8: Perform ICA for Artifact Removal
ica = mne.preprocessing.ICA(n_components=15, random_state=97)
ica.fit(cleaned_stimuli_epochs)
ica.exclude = []  # Add ICA components to exclude here (based on manual inspection)
cleaned_stimuli_epochs = ica.apply(cleaned_stimuli_epochs)

# Step 9: Apply RANSAC and AutoReject
ar = AutoReject()
cleaned_stimuli_epochs, reject_log = ar.fit_transform(cleaned_stimuli_epochs, return_log=True)

# Step 10: Plot Evoked Responses
evoked = cleaned_stimuli_epochs.average()
evoked.plot()
