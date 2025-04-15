import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne

sub = 'sub01'
condition = 'a1'
default_path = Path.cwd()

events_path = default_path / 'data/eeg/predictors/streams_events'
sub_events_path = events_path / f'{sub}/{condition}'

params_path = default_path / 'data' / 'params'
sub_block_path = params_path / f'block_sequences/{sub}.csv'

animal_blocks_path = params_path / 'animal_blocks'
sub_animal_path = None
for files in animal_blocks_path.iterdir():
    if sub in files.name and files.suffix == '.csv':
        sub_animal_path = files
        break
if sub_animal_path is None:
    print(f'No .csv file found for {sub}.')
else:
    print(f'Found file: {sub_animal_path}')

# load files:
# the stream of events:
event_type = 'stream1'
stream1_events_array = []
for files in sub_events_path.iterdir():
    if event_type in files.name:
        event_array = np.load(files, allow_pickle=True)
        stream1_events_array.append(event_array)
# the csv block:
block_data = pd.read_csv(sub_block_path)
# filter block and keep rows with matching condition:
def filter_block(condition):
    if condition == 'a1':
        condition_block = block_data[(block_data['block_seq'] == 's1') & (block_data['block_condition'] == 'azimuth')]
    elif condition == 'a2':
        condition_block = block_data[(block_data['block_seq'] == 's2') & (block_data['block_condition'] == 'azimuth')]
    elif condition == 'e1':
        condition_block = block_data[(block_data['block_seq'] == 's1') & (block_data['block_condition'] == 'elevation')]
    elif condition == 'e2':
        condition_block = block_data[(block_data['block_seq'] == 's2') & (block_data['block_condition'] == 'elevation')]
    return condition_block


condition_block = filter_block(condition)

# define voices_path:
voices_path = default_path / 'data' / 'voices_english'
downsampled_path = voices_path / 'downsampled'
# downsample wav files of each voice folder:
import librosa
import soundfile as sf

target_sfreq = 125  # EEG sample rate

def downsampled_wav_envelopes():
    for folders in voices_path.iterdir():
        if 'voice' in folders.name:
            # Create matching subfolder in downsampled directory
            new_folder = downsampled_path / folders.name
            new_folder.mkdir(parents=True, exist_ok=True)
            for wav_files in folders.iterdir():
                y, sr = librosa.load(wav_files, sr=None)  # Load with original sr
                samples_per_eeg_sample = int(sr / target_sfreq)  # ~195
                frame_length = samples_per_eeg_sample * 2  # for smoother envelope
                hop_length = samples_per_eeg_sample  # one envelope value per 125Hz timepoint
                rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
                rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
                # Create uniform time axis at 125 Hz (step size = 1/125 sec)
                target_times = np.arange(0, rms_times[-1], 1 / target_sfreq)
                # Interpolate RMS envelope to these timepoints
                rms_125Hz = np.interp(target_times, rms_times, rms)
                save_path = new_folder / f"{wav_files.stem}_rms_125Hz.npy"
                np.save(save_path, rms_125Hz)
                print(f"Saved envelope: {save_path.name} to {new_folder}")

downsampled_wav_envelopes()

voices_array = list(condition_block['Voices'])


for i, (voice, event_array) in enumerate(zip(voices_array, stream1_events_array)):
    wav_files_path = voices_path / voice