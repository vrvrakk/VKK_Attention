import os
import numpy as np
import slab
from pathlib import Path

path = Path.cwd()
wav_path = path / 'data' / 'sounds'
processed_path = wav_path / 'processed'
wav_files = []

for files in os.listdir(wav_path):
    file_path = wav_path / files
    wav_files.append(file_path)

sounds = []

for files in wav_files:
    sound = slab.Sound.read(files)
    sounds.append(sound)

for sound in sounds:
    if sound.n_samples > 18188:
        sound.data = sound.data[:18188, :]
    elif sound.n_samples < 18188:
        padding_length = 18188 - sound.n_samples
        padding = np.zeros((padding_length, sound.data.shape[1]))
        sound.data = np.concatenate([sound.data, padding], axis=0)

for i, sound in enumerate(sounds):
    sound.write(f'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/sounds/processed/{i}.wav', fmt='WAV')


processed_files = [os.path.join(processed_path, file) for file in os.listdir(processed_path) if file.endswith('.wav')]

processed_sounds = []
for files in processed_files:
    processed_sound = slab.Sound.read(files)
    processed_sounds.append(processed_sound)

precomputed_sounds = slab.Precomputed(processed_sounds)

for i in range(0, 13, 1):
    precomputed_sounds[i].play()