import numpy
import librosa
from pathlib import Path
import os
from scipy.signal import hilbert

# load paths
sound_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/voices_english/downsampled')
animal_sounds_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/sounds/processed/downsampled')
envelope_path = Path(sound_path / 'envelopes')
animal_envelope_path = Path(animal_sounds_path / 'envelopes')
os.makedirs(envelope_path, exist_ok=True)
os.makedirs(animal_envelope_path, exist_ok=True)
voice1_list = []
voice2_list = []
voice3_list = []
voice4_list = []

for folders in sound_path.iterdir():
    if folders.is_dir():
        for wav_files in folders.iterdir():
            if 'voice1' in str(wav_files):
                voice1_list.append(wav_files)
            elif 'voice2' in str(wav_files):
                voice2_list.append(wav_files)
            elif 'voice3' in str(wav_files):
                voice3_list.append(wav_files)
            elif 'voice4' in str(wav_files):
                voice4_list.append(wav_files)

# wav files have been resampled to 500Hz with Audacity and saved in separate path
def get_sound_envelopes(voice_list, voice=''):
    voice_folder = f'{envelope_path}/{voice}'
    os.makedirs(voice_folder, exist_ok=True)
    for wav_file in voice_list:
        if wav_file.is_file():
            audio, sr = librosa.load(wav_file, sr=None)
            print(f'Audio loaded: {wav_file.name}')
            analytic_signal = hilbert(audio)
            envelope = numpy.abs(analytic_signal)
            # Create the filepath for the envelope file
            file_base_name = os.path.splitext(wav_file.name)[0]  # Strip .wav extension
            filepath = f'{voice_folder}/{file_base_name}.npy'
            numpy.save(filepath, envelope)
        else:
            print(f'Audio not found: {wav_file.name}')


get_sound_envelopes(voice1_list, voice='voice1')
get_sound_envelopes(voice2_list, voice='voice2')
get_sound_envelopes(voice3_list, voice='voice3')
get_sound_envelopes(voice4_list, voice='voice4')

# now do the same for animal sounds
# animal_names = ['cat', 'dog', 'frog', 'kitten', 'kookaburra', 'monkey', 'pig', 'sheep', 'turtle']

for animal_files in animal_sounds_path.iterdir():
    if animal_files.is_file():
        audio, sr = librosa.load(animal_files, sr=None)
        print(f'Audio loaded: {animal_files.name}')
        analytic_signal = hilbert(audio)
        envelope = numpy.abs(analytic_signal)
        # Create the filepath for the envelope file
        file_base_name = os.path.splitext(animal_files.name)[0]  # Strip .wav extension
        filepath = f'{animal_envelope_path}/{file_base_name}.npy'
        numpy.save(filepath, envelope)
    else:
        print(f'Audio not found: {animal_files.name}')



