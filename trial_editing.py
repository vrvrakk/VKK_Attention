import slab
import numpy
import os
import random
import freefield
from pathlib import Path
# import matplotlib.pyplot as plt

# Dai & Shinn-Cunningham (2018):
n_blocks = 10
n_trials1 = 56
isi = (664, 758)
# choose speakers:
speakers_coordinates = (-17.5, 17.5, 0)  # directions for each streams
s2_delay = 2000
sample_freq = 24414
numbers = [1, 2, 3, 4, 5, 6, 8, 9]
data_path = Path.cwd() / 'data' / 'voices'

proc_list = [['RX81', 'RX8', Path.cwd() / 'experiment.rcx'],
             ['RX82', 'RX8', Path.cwd() / 'experiment.rcx']]

voice_idx = list(range(1, 5))
folder_paths = []
wav_folders = [folder for folder in os.listdir(data_path)]
for i, folder in zip(voice_idx, wav_folders):
    folder_path = data_path / folder
    folder_paths.append(folder_path)  # absolute path of each voice folder
# Initialize the corresponding wav_files list
wav_files_lists = []
for i, folder_path in zip(voice_idx, folder_paths):
    wav_files_in_folder = list(folder_path.glob("*.wav"))
    wav_files_lists.append(wav_files_in_folder)
    chosen_voice = random.choice(wav_files_lists)

n_samples = []
# First, calculate the number of samples for each file to get n_samples
for file_path in chosen_voice:
    if os.path.exists(file_path):
        s = slab.Sound(data=file_path)
        s = s.resample(24414)
        n_samples.append(s.n_samples)

# Calculate the mean duration in samples
mean_samples = int(numpy.mean(n_samples))

# Process each file to equalize its duration
equalized_sounds = []
equalized_durations_ms = []

for number, file_path in zip(numbers, chosen_voice):
    if os.path.exists(file_path):
        s = slab.Sound(data=file_path)
        s = s.resample(24414)
        if s.n_samples > mean_samples:
            # Trim the sound if it's longer than the mean
            trimmed_data = s.data[:, :mean_samples] if s.data.ndim == 2 else s.data[:mean_samples]
            s = slab.Sound(data=trimmed_data, samplerate=24414)
        elif s.n_samples < mean_samples:
            # Calculate the required length of the padding
            padding_length = mean_samples - s.n_samples
            if s.data.ndim == 2:
                # Number of channels in your sound data (e.g., 2 for stereo)
                num_channels = s.data.shape[0]
                # Create a 2D padding array with zeros, with one row for each channel
                padding = numpy.zeros((num_channels, padding_length))
            else:
                # Create a 1D padding array for mono sound
                padding = numpy.zeros(padding_length)
            # Concatenate the padding to the end of your sound data
            padded_data = numpy.concatenate((s.data, padding), axis=s.data.ndim - 1)
            # Update your sound object with the padded data
            s = slab.Sound(data=padded_data, samplerate=24414)
        equalized_sounds.append(s)
        equalized_durations_ms.append(int(s.n_samples / 24414))
        freefield.write(f'{number}', s.data, ['RX81', 'RX82'])

# Update n_samples_ms with the durations of the equalized sounds
n_samples_ms = equalized_durations_ms

