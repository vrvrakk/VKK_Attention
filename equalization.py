import slab
import numpy
import os
import random
import freefield
from pathlib import Path

sample_freq = 24414
data_path = Path.cwd() / 'data' / 'voices'


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

wav_files_lists = get_file_names(data_path)

max_len = 0
for voice in wav_files_lists:
    for file_path in voice:
        s = slab.Sound(file_path).resample(sample_freq)
        max_len = s.n_samples if s.n_samples > max_len else max_len

max_len = 18210
for voice in wav_files_lists:
    for file_path in voice:
        s = slab.Sound(file_path).resample(sample_freq)
        if s.n_samples <= max_len:
            pad = slab.Sound.silence(duration=max_len - s.n_samples, samplerate=sample_freq)
            s_padded = slab.Sound.sequence(s, pad)
            file_path_padded = file_path.parent.parent / str(file_path.stem + "_padded" + file_path.suffix)
            s_padded.write(file_path_padded)