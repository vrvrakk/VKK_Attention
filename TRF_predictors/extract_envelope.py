import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne
from TRF_predictors.config import sub, condition, \
    default_path, results_path, voices_path, events_path, params_path, predictors_path, \
    sfreq, stim_dur
import copy

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
    eeg_path = Path(f'D:/VKK_Attention/data/eeg/preprocessed/results/{sub}/ica')
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


def get_voices_dict(wav_nums):
    voices_dict = {}
    for voices in voices_path.iterdir():
        voice_dict = {}
        for i, wav_files in zip(wav_nums, voices.iterdir()):
            voice_dict[wav_files.stem] = i
        voices_dict[voices.stem] = voice_dict
    return voices_dict


def insert_envelope(predictor, number, onset, voice, eeg_len, voices_dict=None):
    # Skip if number is 7 or 71
    if number in [7, 71]:
        return  # Do nothing, just exit the function
    keys_for_number = [key for key, value in voices_dict[voice].items() if value == number]
    if not keys_for_number:
        print(f"No key found for number {number} in voice {voice}")
        return
    voice_key = [key for key, value in voices_dict[voice].items() if value == number][0]
    envelope_path = voices_path / voice / f'{voice_key}.npy'
    envelope = np.load(envelope_path)
    offset = onset + len(envelope)
    if offset >= eeg_len:
        print(f"Warning: onset {onset} exceeds EEG length {eeg_len}")
        envelope = envelope[:eeg_len - onset]
    end = min(onset + len(envelope), eeg_len)
    predictor[onset:end] = envelope[: end - onset]


# save separate block predictors:
def save_predictor_blocks(predictors, stim_dur, stream_type=''):
    save_path = predictors_path / 'envelopes' / sub / condition / stream_type
    save_path.mkdir(parents=True, exist_ok=True)
    for i, series in enumerate(predictors):
        filename_block = f'{sub}_{condition}_{stream_type}_{i}_envelopes.npz'
        np.savez(save_path / filename_block,
                 envelopes=series,
                 sfreq=sfreq,
                 stim_duration_samples=int(stim_dur * sfreq),
                 stream_label=stream_type)
        print(f'Saved {filename_block} in {save_path}')


def target_envelope_predictor(stream_events_array, voices_dict, condition=''):
    stream_predictors = []
    target_predictors = []
    nt_target_predictors = []

    for i, (voice, event_array) in enumerate(zip(voices_array, stream_events_array)):
        eeg_len = eeg_files_list[i].n_times
        predictor = np.zeros(eeg_len)
        target_predictor = np.zeros(eeg_len)
        nt_target_predictor = np.zeros(eeg_len)

        for event in event_array:
            onset = event[0]
            stim_type = event[1]
            number = event[2]

            insert_envelope(predictor, number, onset, voice, eeg_len, voices_dict=voices_dict)

            if stim_type == 4:
                insert_envelope(target_predictor, number, onset, voice, eeg_len, voices_dict=voices_dict)
            elif stim_type == 2:
                insert_envelope(nt_target_predictor, number, onset, voice, eeg_len, voices_dict=voices_dict)

        stream_predictors.append(predictor)
        target_predictors.append(target_predictor)
        nt_target_predictors.append(nt_target_predictor)

    # Determine stream type name
    stream_name = 'stream2' if condition in ['a2', 'e2'] else 'stream1'
    save_predictor_blocks(stream_predictors, stim_dur, stream_type=stream_name)
    save_predictor_blocks(target_predictors, stim_dur, stream_type='targets')
    save_predictor_blocks(nt_target_predictors, stim_dur, stream_type='nt_target')

    stream_predictors_concat = np.concatenate(stream_predictors)
    target_predictors_concat = np.concatenate(target_predictors)
    nt_target_predictors_concat = np.concatenate(nt_target_predictors)

    return stream_predictors_concat, target_predictors_concat, nt_target_predictors_concat


def distractor_envelope_predictor(stream_events_array, voices_dict, condition='', animal_lists=None):
    stream_predictors = []
    distractor_predictors = []
    nt_distractor_predictors = []

    for i, (voice, event_array) in enumerate(zip(voices_array, stream_events_array)):
        eeg_len = eeg_files_list[i].n_times
        predictor = np.zeros(eeg_len)
        distractor_predictor = np.zeros(eeg_len)
        nt_distractor_predictor = np.zeros(eeg_len)

        for event in event_array:
            onset = event[0]
            stim_type = event[1]
            number = event[2]

            if number in [7, 71] and animal_lists is not None:
                if animal_lists[i]:  # Make sure animals are available
                    animal_name = animal_lists[i].pop(0)
                    for animal_file in animal_env_files:
                        if animal_name in animal_file.stem:
                            animal_env = np.load(animal_file)
                            end = min(onset + len(animal_env), eeg_len)
                            predictor[onset:end] = animal_env[:end - onset]
                            break
                continue  # Skip standard insertion for animal deviant

            insert_envelope(predictor, number, onset, voice, eeg_len, voices_dict=voices_dict)

            if stim_type == 3:
                insert_envelope(distractor_predictor, number, onset, voice, eeg_len, voices_dict=voices_dict)
            elif stim_type == 1:
                insert_envelope(nt_distractor_predictor, number, onset, voice, eeg_len, voices_dict=voices_dict)

        stream_predictors.append(predictor)
        distractor_predictors.append(distractor_predictor)
        nt_distractor_predictors.append(nt_distractor_predictor)

    stream_name = 'stream1' if condition in ['a2', 'e2'] else 'stream2'
    save_predictor_blocks(stream_predictors, stim_dur, stream_type=stream_name)
    save_predictor_blocks(distractor_predictors, stim_dur, stream_type='distractors')
    save_predictor_blocks(nt_distractor_predictors, stim_dur, stream_type='nt_distractor')

    stream_predictors_concat = np.concatenate(stream_predictors)
    distractor_predictors_concat = np.concatenate(distractor_predictors)
    nt_distractor_predictors_concat = np.concatenate(nt_distractor_predictors)

    return stream_predictors_concat, distractor_predictors_concat, nt_distractor_predictors_concat


# animal envelope predictor separately:
def animal_envelope_predictor(animal_lists_copy, stream2_events_array, eeg_files_list):
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
    for i, (animal_array, animal_list, eeg_len, predictor) in enumerate(zip(animal_events_arrays, animal_lists_copy, eeg_lens, predictors)):
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

    return np.concatenate(animal_stream_predictors)


def save_filtered_envelopes(stream_envelopes_concat,  stim_dur, sub='', condition='', stream_label=''):
    envelope_save_path = predictors_path / 'envelopes'
    save_path = envelope_save_path / sub / condition / stream_label
    save_path.mkdir(parents=True, exist_ok=True)
    filename = f'{sub}_{condition}_{stream_label}_envelopes_series_concat.npz'
    np.savez(
        save_path / filename,
        envelopes=stream_envelopes_concat,
        sfreq=sfreq,
        stim_duration_samples=int(stim_dur * sfreq),
        stream_label=stream_label
    )
    print(f'Saved {filename} envelopes to {save_path}')


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
    animal_lists_copy = copy.deepcopy(animal_lists)
    stream1_events_array = segregate_streams(event_type='stream1')
    stream2_events_array = segregate_streams(event_type='stream2')
    print(stream1_events_array[0].shape, stream2_events_array[0].shape)
    # if s1: n_trials1 + n_trials2 = 143 # a1, e1
    # if s2: n_trials1 = 140 & n_trials2 = 147 bc target # a2, e2
    # s2 always faster (70ms ISI)

    # define voices_path:
    voices_array = list(condition_block['Voices'])
    wav_nums1 = [1, 2, 3, 4, 5, 6, 8, 9]
    wav_nums2 = [65, 66, 67, 68, 69, 70, 72, 73]

    voices_dict1 = get_voices_dict(wav_nums1)
    voices_dict2 = get_voices_dict(wav_nums2)

    eeg_files_list, eeg_events_list = load_eeg_files(sub=sub, condition=condition)

    if condition in ['a2', 'e2']:
        print(f'Getting envelope predictors for {condition}...')
        stream2_envelopes_concat, target_predictors_concat, nt_target_predictors_concat = target_envelope_predictor(
            stream2_events_array, voices_dict2, condition=condition)
        stream1_envelopes_concat, distractor_predictors_concat, nt_distractor_predictors_concat = distractor_envelope_predictor(
            stream1_events_array, voices_dict1, condition=condition,
            animal_lists=animal_lists.copy() if animal_lists else None)
        if animal_lists_copy is not None:
            animal_stream_envelopes_concat = animal_envelope_predictor(animal_lists_copy, stream1_events_array, eeg_files_list)
            save_filtered_envelopes(animal_stream_envelopes_concat, stim_dur, sub=sub, condition=condition,
                                    stream_label='deviants')
    elif condition in ['e1', 'a1']:
        print(f'Getting envelope predictors for {condition}...')
        stream1_envelopes_concat, target_predictors_concat, nt_target_predictors_concat \
            = target_envelope_predictor(stream1_events_array, voices_dict1, condition=condition)
        stream2_envelopes_concat, distractor_predictors_concat, nt_distractor_predictors_concat = distractor_envelope_predictor(
            stream2_events_array, voices_dict2, condition=condition, animal_lists=animal_lists.copy() if animal_lists else None)
        if animal_lists_copy is not None:
            animal_stream_envelopes_concat = animal_envelope_predictor(animal_lists_copy, stream2_events_array, eeg_files_list)
            save_filtered_envelopes(animal_stream_envelopes_concat, stim_dur, sub=sub, condition=condition,
                                    stream_label='deviants')

    save_filtered_envelopes(target_predictors_concat, stim_dur, sub=sub, condition=condition, stream_label='targets')
    save_filtered_envelopes(nt_target_predictors_concat, stim_dur, sub=sub, condition=condition, stream_label='nt_target')
    save_filtered_envelopes(distractor_predictors_concat, stim_dur, sub=sub, condition=condition, stream_label='distractors')
    save_filtered_envelopes(nt_distractor_predictors_concat, stim_dur, sub=sub, condition=condition, stream_label='nt_distractor')

    save_filtered_envelopes(stream1_envelopes_concat, stim_dur, sub=sub, condition=condition,
                            stream_label='stream1')
    save_filtered_envelopes(stream2_envelopes_concat, stim_dur, sub=sub, condition=condition,
                            stream_label='stream2')

    # import numpy as np
    # import matplotlib.pyplot as plt
    # from scipy.signal import welch
    # from scipy.fft import rfft, rfftfreq
    #
    # sfreq = 125  # adjust if yours is different

    # Assuming you already have:
    # stream1_envelopes_concat = np.array([...])
    # stream2_envelopes_concat = np.array([...])
    #
    # peak_freq_1, power_1 = compute_fft_and_peak(stream1_envelopes_concat, sfreq, label='Stream 1 Envelope')
    # peak_freq_2, power_2 = compute_fft_and_peak(stream2_envelopes_concat, sfreq, label='Stream 2 Envelope')
    #
    # print(f"Stream 1 peak: {peak_freq_1:.3f} Hz | Z-power: {power_1:.2f}")
    # print(f"Stream 2 peak: {peak_freq_2:.3f} Hz | Z-power: {power_2:.2f}")

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


# def downsampled_wav_envelopes():
#     downsampled_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/voices_english/downsampled')
#     voices_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/voices_english')
#     for folders in voices_path.iterdir():
#         if 'voice' in folders.name:
#             # Create matching subfolder in downsampled directory
#             new_folder = downsampled_path / folders.name
#             new_folder.mkdir(parents=True, exist_ok=True)
#             for wav_files in folders.iterdir():
#                 y, sr = librosa.load(wav_files, sr=None)  # Load with original sr
#                 samples_per_eeg_sample = int(np.round(sr / sfreq))  # each EEG sample spans ~195 audio samples
#                 frame_length = samples_per_eeg_sample * 2  # Set the window size for calculating loudness (RMS)
#                 # twice as long as one EEG sample to smooth the envelope more.
#                 hop_length = samples_per_eeg_sample  # one RMS value per EEG sample (i.e., 125 values per second)
#                 # step size to match one EEG timepoint
#                 rms = librosa.feature.rms(y=y, frame_length=frame_length, hop_length=hop_length)[0]
#                 # loudness over time using the RMS (root mean square) method
#                 rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop_length)
#                 # timestamps for each RMS value (based on hop_length and audio sampling rate)
#                 # Create uniform time axis at 125 Hz (step size = 1/125 sec)
#                 target_times = np.arange(0, rms_times[-1], 1 / sfreq)
#                 # Create a uniform time axis matching EEG timing:
#                 # One point every 1/125 seconds = 125 Hz
#                 # Interpolate RMS envelope to these timepoints
#                 rms_125Hz = np.interp(target_times, rms_times, rms)
#                 save_path = new_folder / f"{wav_files.stem}_rms_125Hz.npy"
#                 np.save(save_path, rms_125Hz)
#                 print(f"Saved envelope: {save_path.name} to {new_folder}")

# downsampled_wav_envelopes()
#
# # hilbert
# target_sfreq = 125
# from scipy.signal import hilbert
# def hilbert_envelopes_animal_sounds():
#     animal_sounds_path = default_path / 'data/sounds/processed'
#     animal_sounds_hilbert = animal_sounds_path / 'hilbert'
#     animal_sounds_hilbert.mkdir(parents=True, exist_ok=True)
#     for wav_file in animal_sounds_path.glob('*.wav'):
#         y, sr = librosa.load(wav_file, sr=None)
#         analytic_signal = hilbert(y)
#         envelope = np.abs(analytic_signal)
#         # Interpolate to match EEG sampling rate (125 Hz)
#         duration = len(y) / sr
#         source_times = np.arange(len(envelope)) / sr
#         target_times = np.arange(0, duration, 1 / sfreq)
#         envelope_125Hz = np.interp(target_times, source_times, envelope)
#         save_path = animal_sounds_hilbert / f"{wav_file.stem}_hilbert_125Hz.npy"
#         np.save(save_path, envelope)
#         print(f"Saved Hilbert envelope: {save_path.name} to {animal_sounds_hilbert}")
#
# hilbert_envelopes_animal_sounds()
# def hilbert_envelopes_voices():
#     voices_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/voices_english')
#     hilbert_path = voices_path / 'hilbert'
#     hilbert_path.mkdir(parents=True, exist_ok=True)
#     for folder in voices_path.iterdir():
#         if 'voice' in folder.name:
#             new_folder = hilbert_path / folder.name
#             new_folder.mkdir(parents=True, exist_ok=True)
#             for wav_file in folder.glob('*.wav'):
#                 y, sr = librosa.load(wav_file, sr=None)
#                 analytic_signal = hilbert(y)
#                 envelope = np.abs(analytic_signal)
#                 # Interpolate to match EEG sampling rate (125 Hz)
#                 duration = len(y) / sr
#                 source_times = np.arange(len(envelope)) / sr
#                 target_times = np.arange(0, duration, 1 / sfreq)
#                 envelope_125Hz = np.interp(target_times, source_times, envelope)
#                 save_path = new_folder / f"{wav_file.stem}_hilbert_125Hz.npy"
#                 np.save(save_path, envelope_125Hz)
#                 print(f"Saved Hilbert envelope: {save_path.name} to {new_folder}")
#
# hilbert_envelopes_voices()
#
#
# # plot
# import librosa
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.signal import hilbert
# from pathlib import Path
#
# # === Paths & Settings ===
# import numpy as np
# import matplotlib.pyplot as plt
# from pathlib import Path
#
# # === Settings ===
# sfreq = 125  # EEG sampling rate
# default_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention')
# hilbert_path = default_path / 'data/sounds/processed/hilbert'
#
# # === Load one .npy envelope ===
# npy_file = list(hilbert_path.glob('*.npy'))[0]  # or manually select one
# envelope = np.load(npy_file)
#
# # === Time axis for 125 Hz ===
# time = target_times
#
# # === Plot ===
# plt.figure(figsize=(12, 4))
# plt.plot(time, envelope_125Hz, label='Hilbert Envelope', color='orange')
# plt.title(f"Hilbert Envelope (125 Hz): {npy_file.stem}")
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.legend()
# plt.tight_layout()
# plt.show()