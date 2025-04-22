import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from TRF.predictors_run import sub, condition, \
    default_path, results_path, voices_path, events_path, params_path, predictors_path, \
    sfreq, stim_dur


animal_lists = None


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


# load files:
# the stream of events:
def segregate_streams(event_type=''):
    stream_events_array = []
    for files in sub_events_path.iterdir():
        if event_type in files.name:
            event_array = np.load(files, allow_pickle=True)
            stream_events_array.append(event_array)
    return stream_events_array


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


def insert_envelope(predictor, number, onset, voice, eeg_len):
    if number not in voices_dict[voice].values():
        return
    voice_key = [key for key, value in voices_dict[voice].items() if value == number][0]
    envelope_path = voices_path / voice / f'{voice_key}.npy'
    envelope = np.load(envelope_path)
    offset = onset + len(envelope)
    if offset > eeg_len:
        envelope = envelope[:eeg_len - onset]
    predictor[onset:onset + len(envelope)] = envelope


# save separate block predictors:
def save_predictor_blocks(predictors, stim_dur, stream_type=''):
    save_path = predictors_path / 'envelopes' / sub / condition
    save_path.mkdir(parents=True, exist_ok=True)
    for i, series in enumerate(predictors):
        filename_block = f'{sub}_{condition}_{stream_type}_{i}_envelopes.npz'
        np.savez(save_path / filename_block,
                 envelopes=series,
                 sfreq=sfreq,
                 stim_duration_samples=int(stim_dur * sfreq),
                 stream_label=stream_type)
def envelope_predictor(stream_events_array, condition='', sub='', stream='', animal_lists = animal_lists):
    stream_predictors = []
    target_predictors = []
    distractor_predictors = []
    nt_target_predictors = []
    nt_distractor_predictors = []

    for i, (voice, event_array) in enumerate(zip(voices_array, stream2_events_array)):
        eeg_len = eeg_files_list[i].n_times
        predictor = np.zeros(eeg_len)
        target_predictor = np.zeros(eeg_len)
        distractor_predictor = np.zeros(eeg_len)
        nt_target_predictor = np.zeros(eeg_len)
        nt_distractor_predictor = np.zeros(eeg_len)

        for event in event_array:
            onset = event[0]
            stim_type = event[1]
            number = event[2]
            if number in [7, 71] and stream == 'distractor' and animal_lists is not None:
                for animal_file in animal_env_files:
                    if animal_lists[i]:  # make sure the block has animal names
                        animal_name = animal_lists[i].pop(0)  # get animal in order
                        # next time we hit an animal event (e.g., number 7 or 71), we get the next one ('cat'), and so on
                        if animal_name in animal_file.stem:
                            animal_env = np.load(animal_file)
                            offset = onset + len(animal_env)
                            if offset > eeg_len:
                                animal_env = animal_env[:eeg_len - onset]
                            distractor_predictor[onset:onset + len(animal_env)] = animal_env
                            break  # once match is found
            # Main full predictor
            insert_envelope(predictor, number, onset, voice, eeg_len)

            # Type-specific predictors
            if stim_type == 3:
                insert_envelope(target_predictor, number, onset, voice, eeg_len)
            elif stim_type == 2:
                insert_envelope(distractor_predictor, number, onset, voice, eeg_len)
            elif stim_type == 0:
                insert_envelope(nt_distractor_predictor, number, onset, voice, eeg_len)
            elif stim_type == 1 and stream == 'target':
                insert_envelope(nt_target_predictor, number, onset, voice, eeg_len)

        stream_predictors.append(predictor)
        save_predictor_blocks(stream_predictors, stim_dur, stream_type=stream)
        target_predictors.append(target_predictor)
        save_predictor_blocks(target_predictors, stim_dur, stream_type='targets')
        distractor_predictors.append(distractor_predictor)
        save_predictor_blocks(distractor_predictors, stim_dur, stream_type='distractors')
        nt_target_predictors.append(nt_target_predictor)
        save_predictor_blocks(nt_target_predictors, stim_dur, stream_type='nt_target')
        nt_distractor_predictors.append(nt_distractor_predictor)
        save_predictor_blocks(nt_distractor_predictors, stim_dur, stream_type='nt_distractor')

    stream_predictors_concat = np.concatenate(stream_predictors)
    target_predictors_concat = np.concatenate(target_predictors)
    distractor_predictors_concat = np.concatenate(distractor_predictors)
    nt_target_predictors_concat = np.concatenate(nt_target_predictors)
    nt_distractor_predictors_concat = np.concatenate(nt_distractor_predictors)
    return stream_predictors_concat, target_predictors_concat, distractor_predictors_concat,\
           nt_target_predictors_concat, nt_distractor_predictors_concat

# animal envelope predictor separately:
def animal_envelope_predictor(animal_lists, stream2_events_array, eeg_files_list):
    animal_stream_predictors = []
    animal_events_arrays = []
    eeg_lens = []
    predictors = []
    for i, (eeg_file, event_array) in enumerate(zip(eeg_files_list, stream2_events_array)):
        eeg_len = eeg_files_list[i].n_times
        eeg_lens.append(eeg_len)
        predictor = np.zeros(eeg_len)
        predictors.append(predictor)
        animal_events_array = []
        for event in event_array:
            if event[2] == 71 or event[2] == 7:
                animal_events_array.append(event)
        animal_events_arrays.append(animal_events_array)
    for i, (animal_array, animal_list, eeg_len, predictor) in enumerate(zip(animal_events_arrays, animal_lists, eeg_lens, predictors)):
        for animal, event in zip(animal_list, animal_array):
            for file in animal_env_files:
                if animal in file.stem:  # find envelope corresponding to animal in envs path
                    onset = event[0]
                    animal_env = np.load(file)
                    offset = onset + len(animal_env)
                    if offset > eeg_len:
                        animal_env = animal_env[:eeg_len - onset]  # crop to avoid overflow
                    predictor[onset:onset + len(animal_env)] = animal_env
        animal_stream_predictors.append(predictor)
        save_predictor_blocks(animal_stream_predictors, stim_dur, stream_type='deviants')
    animal_stream_envelopes_concat = np.concatenate(animal_stream_predictors)
    return animal_stream_envelopes_concat

# save
def save_envelope_predictors(stream1_envelopes_concat,stream2_envelopes_concat,  stim_dur, sub='', condition='', stream1_label='', stream2_label=''):
    envelope_save_path = predictors_path / 'envelopes'
    save_path = envelope_save_path / sub / condition
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f'{sub}_{condition}_envelopes_series_concat.npz'
    np.savez(
        save_path / filename,
        envelopes1=stream1_envelopes_concat,
        envelopes2=stream2_envelopes_concat,
        sfreq=sfreq,
        stim_duration_samples=int(stim_dur * sfreq),
        stream1_label=stream1_label,
        stream2_label=stream2_label
    )


def save_filtered_envelopes(stream_envelopes_concat,  stim_dur, sub='', condition='', stream_label=''):
    envelope_save_path = predictors_path / 'envelopes'
    save_path = envelope_save_path / sub / condition
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f'{sub}_{condition}_{stream_label}_envelopes_series_concat.npz'
    np.savez(
        save_path / filename,
        envelopes=stream_envelopes_concat,
        sfreq=sfreq,
        stim_duration_samples=int(stim_dur * sfreq),
        stream_label=stream_label
    )


if __name__ == '__main__':

    sub_events_path = events_path / f'{sub}/{condition}'
    sub_block_path = params_path / f'block_sequences/{sub}.csv'

    # the csv block:
    block_data = pd.read_csv(sub_block_path)
    # filter block and keep rows with matching condition:

    condition_block = filter_block(condition)

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
        # get animal sounds stream based on the csv file
        animal_sounds_path = default_path / 'data/sounds/processed'
        animal_sounds_envs = animal_sounds_path / 'downsampled'
        animals_csv = pd.read_csv(sub_animal_path)
        animals_csv = animals_csv.dropna(axis=1)  # drop columns (Axis 1) with NaN values
        animal_names = []
        for animal_wavs in animal_sounds_path.iterdir():
            animal_names.append(animal_wavs.stem)
        # find block rows in animal csv matching with condition block (based on index):
        block_indices = condition_block.index
        animal_blocks = animals_csv.iloc[block_indices]
        animal_lists = []
        for _, row in animal_blocks.iterrows():
            animal_lists.append(row.tolist())
        # match animals in each block's list with envelope:
        # prepare env files:
        animal_env_files = list(animal_sounds_envs.iterdir())

    stream1_events_array = segregate_streams(event_type='stream1')
    stream2_events_array = segregate_streams(event_type='stream2')

    # define voices_path:
    voices_array = list(condition_block['Voices'])
    wav_nums = [1, 2, 3, 4, 5, 6, 8, 9]
    voices_dict = {}
    for voices in voices_path.iterdir():
        voice_dict = {}
        for i, wav_files in zip(wav_nums, voices.iterdir()):
            voice_dict[wav_files.stem] = i
        voices_dict[voices.stem] = voice_dict

    eeg_files_list, eeg_events_list = load_eeg_files(sub=sub, condition=condition)

    stream1_envelopes_concat, target_predictors_concat, _, nt_target_predictors_concat, _ = envelope_predictor(
        stream1_events_array, condition=condition, sub=sub, stream='target', animal_lists=None)
    stream2_envelopes_concat, _, distractor_predictors_concat, _, nt_distractor_predictors_concat = envelope_predictor(
        stream2_events_array, condition=condition, sub=sub, stream='distractor',
        animal_lists=animal_lists.copy() if animal_lists else None)

    if animal_lists is not None:
        animal_stream_envelopes_concat = animal_envelope_predictor(animal_lists, stream2_events_array, eeg_files_list)
        save_filtered_envelopes(animal_stream_envelopes_concat, stim_dur, sub=sub, condition=condition,
                                stream_label='deviants')

    save_envelope_predictors(stream1_envelopes_concat, stream2_envelopes_concat, sub=sub, condition=condition)

    save_filtered_envelopes(target_predictors_concat, stim_dur, sub=sub, condition=condition, stream_label='targets')
    save_filtered_envelopes(nt_target_predictors_concat, stim_dur, sub=sub, condition=condition, stream_label='nt_target')
    save_filtered_envelopes(distractor_predictors_concat, stim_dur, sub=sub, condition=condition, stream_label='distractors')
    save_filtered_envelopes(nt_distractor_predictors_concat, stim_dur, sub=sub, condition=condition, stream_label='nt_distractor')

################################################## ANIMAL SOUNDS ENVELOPES #############################################
# import librosa
# import soundfile as sf

# def animal_sounds_envelopes():
    # animal_sounds_path = default_path / 'data/sounds/processed'
    # animal_sounds_downsampled = animal_sounds_path / 'downsampled'
    # animals_names = []
    # for wav_files in animal_sounds_path.glob('*.wav'):
    #     animals_names.append(wav_files.stem)
    #     y, sr = librosa.load(wav_files, sr=None)  # Load with original sr
    #     samples_per_eeg_sample = int(sr / target_sfreq)  # ~195
    #     frame_length = samples_per_eeg_sample * 2  # for smoother envelope
    #     hop_length = samples_per_eeg_sample  # one envelope value per 125Hz timepoint
    #     rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
    #     rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
    #     # Create uniform time axis at 125 Hz (step size = 1/125 sec)
    #     target_times = np.arange(0, rms_times[-1], 1 / target_sfreq)
    #     # Interpolate RMS envelope to these timepoints
    #     rms_125Hz = np.interp(target_times, rms_times, rms)
    #     save_path = animal_sounds_downsampled / f"{wav_files.stem}_rms_125Hz.npy"
    #     np.save(save_path, rms_125Hz)
    #     print(f"Saved envelope: {save_path.name} to {animal_sounds_downsampled}")
# animal_sounds_envelopes()

################################################## VOICES-NUMBERS ENVELOPES #############################################

# downsample wav files of each voice folder:


#def downsampled_wav_envelopes():
    # downsampled_path = voices_path / 'downsampled'
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