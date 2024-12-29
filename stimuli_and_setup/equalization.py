import slab
import numpy as np
import os
import random
import freefield
from pathlib import Path

sample_freq = 24414
data_path = Path.cwd() / 'data' / 'voices_padded'
numbers = [1, 2, 3, 4, 5, 6, 8, 9]


def get_file_names(data_path):  # create wav_list paths, and select a voice folder randomly
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
    return wav_files_lists


# run function
wav_files_lists = get_file_names(data_path)

# get s duration in samples
n_samples = []
max_len = 0  # start with 0 until we find max len
for voice in wav_files_lists:
    for number, file_path in zip(numbers, voice):
        print(number, file_path)
        s = slab.Sound(file_path).resample(sample_freq)
        n_samples.append((number, s.n_samples))
        max_len = s.n_samples if s.n_samples > max_len else max_len

# convert s duration from samples to ms
n_samples_s = []
for number, samples in n_samples:
    samples_ms = (samples/sample_freq)  # in s, / 1000 if we want in ms
    n_samples_s.append((number, samples_ms))



# calculate ISI variations:
# s1:
# ISI_1 = 741
# s1_ISI = []
# for number, silence in added_silence:
#     s1_total_ISI = silence + ISI_1
#     s1_ISI.append((number, s1_total_ISI))
# s1_min_ISI = np.min(s1_ISI, axis=0)[1]
# s1_max_ISI = np.max(s1_ISI, axis=0)[1]
# # s2:
# ISI_2 = 543
# s2_ISI = []
# for number, silence in added_silence:
#     s2_total_ISI = silence + ISI_2
#     s2_ISI.append((number, s2_total_ISI))
# s2_min_ISI = np.min(s2_ISI, axis=0)[1]
# s2_max_ISI = np.max(s2_ISI, axis=0)[1]


# max_len = 18210
for voice in wav_files_lists:
    for file_path in voice:
        s = slab.Sound(file_path).resample(sample_freq)
        if s.n_samples <= max_len:
            pad = slab.Sound.silence(duration=max_len - s.n_samples, samplerate=sample_freq)
            s_padded = slab.Sound.sequence(s, pad)
            file_path_padded = file_path.parent.parent / str(file_path.stem + "_padded" + file_path.suffix)
            s_padded.write(file_path_padded)

# Loop through each sound
# for i, sound in enumerate(sounds):
#     if sound.n_samples > desired_samples:
#         # Trim the sound to the desired number of samples
#         new_sound = sound.resize(desired_samples)
#     elif sound.n_samples < desired_samples:
#         # Create a silence sound to pad the existing sound to the desired duration
#         pad_duration = desired_duration - sound.duration
#         pad = slab.Sound.silence(duration=pad_duration, samplerate=samplerate)
#         # Concatenate the original sound with the padding
#         new_sound = slab.Sound.sequence([sound, pad])
#     else:
#         # If the sound's duration is already the desired duration, keep it as is
#         new_sound = sound
#     sounds[i] = new_sound
