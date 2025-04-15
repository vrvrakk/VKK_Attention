import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne

sub = 'sub01'
condition = 'a1'
stream1_label = 'target_stream'
stream2_label = 'distractor_stream'
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
def segregate_streams(event_type=''):
    stream_events_array = []
    for files in sub_events_path.iterdir():
        if event_type in files.name:
            event_array = np.load(files, allow_pickle=True)
            stream_events_array.append(event_array)
    return stream_events_array

stream1_events_array = segregate_streams(event_type='stream1')
stream2_events_array = segregate_streams(event_type='stream2')


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
voices_path = default_path / 'data' / 'voices_english' / 'downsampled'
target_sfreq = 125  # EEG sample rate

voices_array = list(condition_block['Voices'])
wav_nums = [1, 2, 3, 4, 5, 6, 8, 9]
voices_dict = {}
for voices in voices_path.iterdir():
    voice_dict = {}
    for i, wav_files in zip(wav_nums, voices.iterdir()):
        voice_dict[wav_files.stem] = i
    voices_dict[voices.stem] = voice_dict


# load eeg files:
results_path = default_path / 'data/eeg/preprocessed/results'
sfreq = 125
def load_eeg_files(sub='', condition=''):
    eeg_path = results_path / f'{sub}/ica'
    eeg_files_list = []
    eeg_events_list = []
    for sub_files in eeg_path.iterdir():
        if '.fif' in sub_files.name:
            if condition in sub_files.name:
                eeg_file = mne.io.read_raw_fif(sub_files, preload=True)
                eeg_file.set_eeg_reference('average')
                eeg_file.resample(sfreq=sfreq)
                # get events from each eeg file
                eeg_events, eeg_event_ids = mne.events_from_annotations(eeg_file)
                # save each eeg file and eeg event list
                eeg_events_list.append((eeg_events, eeg_event_ids))
                eeg_files_list.append(eeg_file)
    return eeg_files_list, eeg_events_list


eeg_files_list, eeg_events_list = load_eeg_files(sub=sub, condition=condition)


def envelope_predictor(stream_events_array, stream='', condition='', sub=''):
    stream_predictors = []
    for i, (voice, event_array) in enumerate(zip(voices_array, stream_events_array)):
        current_voice = voices_dict[voice]
        eeg_len = eeg_files_list[i].n_times
        predictor = np.zeros(eeg_len)

        for event in event_array:
            number = event[2]
            onset = event[0]

            if number in current_voice.values():
                voice_key = [key for key, value in current_voice.items() if value == number][0]
                envelope_path = voices_path / voice / f'{voice_key}.npy'
                envelope = np.load(envelope_path)

                offset = onset + len(envelope)
                if offset > eeg_len:
                    envelope = envelope[:eeg_len - onset]  # crop to avoid overflow

                predictor[onset:onset + len(envelope)] = envelope

        stream_predictors.append(predictor)
    stream_envelopes_concat = np.concatenate(stream_predictors)
    return stream_envelopes_concat


stream1_envelopes_concat= envelope_predictor(stream1_events_array, condition=condition, sub=sub)
stream2_envelopes_concat= envelope_predictor(stream2_events_array, condition=condition, sub=sub)

# save
def save_envelope_predictors(stream1_envelopes_concat,stream2_envelopes_concat,  sub='', condition='', stream1_label='', stream2_label=''):
    stim_dur = 0.745
    envelope_save_path = default_path / f'data/eeg/predictors/envelopes'
    save_path = envelope_save_path / sub
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f'{sub}_{condition}_envelopes_series.npz'
    np.savez(
        save_path / filename,
        stream1=stream1_envelopes_concat,
        stream2=stream2_envelopes_concat,
        sfreq=sfreq,
        stim_duration_samples=int(stim_dur * sfreq),
        stream1_label=stream1_label,
        stream2_label=stream2_label
    )


save_envelope_predictors(stream1_envelopes_concat,stream2_envelopes_concat,  sub=sub, condition=condition)

# todo: add animal envelopes...
# todo: how to deal with bad segments in EEG data.
# todo: overlap ratios


########################################################################
# downsample wav files of each voice folder:
import librosa
import soundfile as sf
# downsampled_path = voices_path / 'downsampled'
#def downsampled_wav_envelopes():
#     for folders in voices_path.iterdir():
#         if 'voice' in folders.name:
#             # Create matching subfolder in downsampled directory
#             new_folder = downsampled_path / folders.name
#             new_folder.mkdir(parents=True, exist_ok=True)
#             for wav_files in folders.iterdir():
#                 y, sr = librosa.load(wav_files, sr=None)  # Load with original sr
#                 samples_per_eeg_sample = int(sr / target_sfreq)  # ~195
#                 frame_length = samples_per_eeg_sample * 2  # for smoother envelope
#                 hop_length = samples_per_eeg_sample  # one envelope value per 125Hz timepoint
#                 rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
#                 rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
#                 # Create uniform time axis at 125 Hz (step size = 1/125 sec)
#                 target_times = np.arange(0, rms_times[-1], 1 / target_sfreq)
#                 # Interpolate RMS envelope to these timepoints
#                 rms_125Hz = np.interp(target_times, rms_times, rms)
#                 save_path = new_folder / f"{wav_files.stem}_rms_125Hz.npy"
#                 np.save(save_path, rms_125Hz)
#                 print(f"Saved envelope: {save_path.name} to {new_folder}")
#
# downsampled_wav_envelopes()