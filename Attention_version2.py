import slab
import numpy
import os
import freefield
import pickle as pkl
from get_streams_and_stream_params import get_delays, get_timepoints, streams_dfs, assign_numbers, get_trial_sequence, \
    get_stream_params, numbers, duration_s, isi, tlo1, tlo2, block_seqs_df, target_number_seq, choose_target_number, \
    increase_prob_target_number
from block_index import increment_block_index, block_index
from block_sequence import get_target_number_seq
from generate_voice_list import voice_seq, data_path
import datetime
from pathlib import Path
import pandas as pd
import random
import csv

current_path = Path.cwd()
params_dir = Path(current_path / 'data' / 'params')
equalization_path = Path.cwd() / 'data' / 'misc' / 'calibration' / 'calibration_speakers.pkl'
with open(equalization_path, 'rb') as file:
    calibration_azimuth = pkl.load(file)

calibration_left = None
calibration_right = None

# Iterate over the dictionary and extract the first two sub-dictionaries
for index, (key, values) in enumerate(calibration_azimuth.items()):
    if index == 0:
        calibration_left = values  # Save the first sub-dictionary
    elif index == 1:
        calibration_right = values  # Save the second sub-dictionary
        break  # Stop after saving the first two sub-dictionaries
filter_left = calibration_left["filter"]
filter_right = calibration_right["filter"]


animal_index = 0  # for animal sounds dataframe
# path:
sequence_path = Path.cwd() / 'data' / 'generated_sequences'

# processors:
proc_list = [['RX81', 'RX8', Path.cwd() / 'experiment_jitter.rcx'],
             ['RX82', 'RX8', Path.cwd() / 'experiment_jitter.rcx']]

voice_index = 0
nums = [1, 2, 3, 4, 5, 6, 8, 9]


def get_participant_id(subject_id):
    current_date = datetime.datetime.now().date()
    formatted_date = current_date.strftime('%Y%m%d')
    participant_id = formatted_date + '_' + subject_id
    return participant_id


def select_voice(block_seqs_df, data_path):  # write voice data onto rcx buffer
    chosen_voice_name = block_seqs_df.loc[block_index - 1, 'Voices']
    wav_list = []
    for folder in data_path.iterdir():
        wav_list.append(list(folder.iterdir()))
    voice_mapping = {
        'voice1': 0,
        'voice2': 1,
        'voice3': 2,
        'voice4': 3
    }
    chosen_voice = wav_list[voice_mapping[chosen_voice_name]]
    return chosen_voice, chosen_voice_name


# get animal sounds sequence:
def animal_sounds(noise_trials_count, idx_to_replace):
    wav_path = current_path / 'data' / 'sounds'
    processed_path = wav_path / 'processed'
    processed_files = [os.path.join(processed_path, file) for file in os.listdir(processed_path) if
                       file.endswith('.wav')]

    processed_sounds = []
    for files in processed_files:
        processed_sound = slab.Sound.read(files)
        processed_sounds.append(processed_sound)

    animals = []
    while len(animals) < noise_trials_count:
        # make sure wav_list is not empty
        if not processed_sounds:
            processed_sounds = []
            for files in processed_files:
                processed_sound = slab.Sound.read(files)
                processed_sounds.append(processed_sound)
        animal = random.choice(processed_sounds)
        animals.append(animal)
        processed_sounds.remove(animal)
    animal_names = []
    precomputed_animal_sounds = slab.Precomputed(animals)
    for i in precomputed_animal_sounds:
        animal_name = i.name
        animal_names.append(animal_name)
    animal_names = [os.path.splitext(os.path.basename(path))[0] for path in animal_names]  # keep only animal name

    for i in range(len(precomputed_animal_sounds)):
        precomputed_animal_sounds[i].level = 80  # animal sounds a bit louder

    concatenated_animal_sounds = numpy.concatenate([sound.data.flatten() for sound in precomputed_animal_sounds])
    return precomputed_animal_sounds, concatenated_animal_sounds, animal_names

animal_df = pd.DataFrame(index=range(20), columns=range(10))

# Save the empty DataFrame to a CSV file.

# Initialize a mutable list for keeping track of the current index.
animal_index = [0]


def save_animal_names_to_file(animal_names, animal_index):

    file_path = animal_sounds_csv    # Load the DataFrame from the CSV file.
    animal_df = pd.read_csv(file_path)

    # Iterate over animal_names and add them to the DataFrame if the cell is NaN.
    for i, name in enumerate(animal_names):
        # Use .iloc to access by index position and only fill NaN cells in the specified row
        current_index = animal_index[0]
        if pd.isna(animal_df.iloc[current_index, i]):  # Check if the cell is NaN before updating
            animal_df.iloc[current_index, i] = name

    # Increment animal_index for the next call
    animal_index[0] += 1

    # Save the updated DataFrame back to the CSV file.
    animal_df.to_csv(file_path, index=False)


def speaker_filters(s1_params, s2_params, axis, filter_left, filter_right):
    if axis == 'elevation':
        [speaker1] = freefield.pick_speakers((s1_params.get('speakers_coordinates')))  # 17.5 az, 0.0 ele (target), or -12.5 ele
        [speaker2] = freefield.pick_speakers((s2_params.get('speakers_coordinates')))
        filter1 = speaker1.filter
        filter1.samplerate = 24414
        filter2 = speaker2.filter
        filter2.samplerate = 24414
    elif axis == 'azimuth':
        filter1 = filter_right
        filter1.samplerate = 24414
        filter2 = filter_left
        filter2.samplerate = 24414
    return filter1, filter2

def equalize_animal_sounds(precomputed_animal_sounds, filter1, filter2, target_stream):
    if target_stream == 's1':
        for sounds in precomputed_animal_sounds:
            filter2.apply(sounds)
            print('Animal sounds equalized in distractor stream 2 (left).')
    elif target_stream == 's2':
        for sounds in precomputed_animal_sounds:
            filter1.apply(sounds)
            print('Animal sounds equalized in distractor stream 1 (right).')
    return precomputed_animal_sounds


def write_buffer(chosen_voice, precomputed_animal_sounds, concatenated_animal_sounds, stream, filter, axis):
    # calibration works for elevation
    for number, file_path in zip(nums, chosen_voice):
        # combine lists into a single iterable
        # elements from corresponding positions are paired together
        if os.path.exists(file_path):
            s = slab.Sound(data=file_path)  # .resample(samplerate=24414) # not needed
            s.level = 80
            filt_s = filter.apply(s)
            freefield.write(f'{number}', filt_s.data, ['RX81', 'RX82'])  # loads array on buffer
            freefield.write(f'{number}_n_samples{stream}', filt_s.n_samples, ['RX81', 'RX82'])
            # sets total buffer size according to numeration
    freefield.write('noise', concatenated_animal_sounds, ['RX81', 'RX82'])
    freefield.write('noise_n_samples', int(concatenated_animal_sounds.size / len(precomputed_animal_sounds)),
                    ['RX81', 'RX82'])
    freefield.write('noise_size', concatenated_animal_sounds.size, ['RX81', 'RX82'])


def save_block_seq():  # works
    blocks_dir = params_dir / f'{subject_id}.csv'
    block_seqs_df.to_csv(blocks_dir, sep=',', index=False)


def run_block(trial_seq1, trial_seq2, tlo1, tlo2, s1_params, s2_params):
    # REMEMBER TO CHANGE N_PULSE FOR S2 WHEN RUNNING EXP AFTER TRAINING
    speakers_coordinates = (0, 0)
    [speaker1] = freefield.pick_speakers(
        (s1_params.get('speakers_coordinates')))  # 17.5 az, 0.0 ele (target), or -12.5 ele
    [speaker2] = freefield.pick_speakers((s2_params.get('speakers_coordinates')))  # 0.0 az, 0.0 ele, or 12.5 ele
    [animal] = freefield.pick_speakers((0, 0))
    # [speaker1] = freefield.pick_speakers((speakers_coordinates[1], 0))  # 17.5 az, 0.0 ele (target), or -12.5 ele
    # [speaker2] = freefield.pick_speakers((speakers_coordinates[0], 0))
    sequence1 = numpy.array(trial_seq1).astype('int32')
    sequence1 = numpy.append(0, sequence1)
    sequence2 = numpy.array(trial_seq2).astype('int32')
    sequence2 = numpy.append(0, sequence2)
    # here we set tlo to RX8
    freefield.write('tlo1', tlo1, ['RX81', 'RX82'])
    freefield.write('tlo2', tlo2, ['RX81', 'RX82'])
    # set n_trials to pulse trains sheet0/sheet1
    freefield.write('n_trials1', s1_params.get('n_trials') + 1,
                    ['RX81', 'RX82'])  # analog_proc attribute from speaker table dom txt file
    freefield.write('n_trials2', s2_params.get('n_trials') + 1, ['RX81', 'RX82'])
    freefield.write('trial_seq1', sequence1, ['RX81', 'RX82'])
    freefield.write('trial_seq2', sequence2, ['RX81', 'RX82'])

    freefield.write('s1_delay', s1_params.get('s_delay'), ['RX81', 'RX82'])
    freefield.write('s2_delay', s2_params.get('s_delay'), ['RX81', 'RX82'])
    # set output speakers for both streams
    proc_list = ['RX81', 'RX82']
    proc_list.remove(speaker1.analog_proc)
    freefield.write('channel1', speaker1.analog_channel, speaker1.analog_proc)
    freefield.write('channel1', 25, proc_list)
    # s1 target both to RX8I
    proc_list = ['RX81', 'RX82']
    proc_list.remove(speaker2.analog_proc)
    freefield.write('channel2', speaker2.analog_channel, speaker2.analog_proc)  # s2 distractor
    freefield.write('channel2', 25, proc_list)
    # freefield.write('noise_channel', animal.analog_channel, animal.analog_channel)
    statement = input('Start experiment? y/n: ')
    if statement.lower() in ['y', '']:  # works
        freefield.play()
    else:
        freefield.halt()


def run_experiment():  # works as desired
    global block_index
    s1_delay, s2_delay, target_stream, n_trials1, n_trials2 = get_delays(duration_s, isi)
    t1_total, t2_total = get_timepoints(tlo1, tlo2, n_trials1, n_trials2)
    streams_df = streams_dfs(tlo1, tlo2, t1_total, t2_total, s1_delay, s2_delay)
    target_number = choose_target_number(block_seqs_df)
    streams_df, target_stream_df = assign_numbers(streams_df, numbers, tlo1, target_stream, target_number)
    streams_df_updated = increase_prob_target_number(streams_df, target_number, target_stream, target_stream_df)
    trial_seq1, trial_seq2 = get_trial_sequence(streams_df_updated)
    s1_params, s2_params, axis, block_index, trial_seq1, trial_seq2, noise_trials_count, idx_to_replace = get_stream_params(s1_delay, s2_delay, n_trials1, n_trials2, trial_seq1, trial_seq2,target_number)  # block index incremented in this function
    precomputed_animal_sounds, concatenated_animal_sounds, animal_names = animal_sounds(noise_trials_count, idx_to_replace)
    save_animal_names_to_file(animal_names, animal_index)
    chosen_voice, chosen_voice_name = select_voice(block_seqs_df, data_path)
    filter1, filter2 = speaker_filters(s1_params, s2_params, axis, filter_left, filter_right)
    precomputed_animal_sounds_equalized = equalize_animal_sounds(precomputed_animal_sounds, filter1, filter2, target_stream)
    write_buffer(chosen_voice, precomputed_animal_sounds_equalized, concatenated_animal_sounds, stream=1, filter=filter1, axis=axis)
    write_buffer(chosen_voice, precomputed_animal_sounds_equalized, concatenated_animal_sounds, stream=2, filter=filter2, axis=axis)
    run_block(trial_seq1, trial_seq2, tlo1, tlo2, s1_params, s2_params)
    return s1_delay, s2_delay, target_stream, s1_params, s2_params, axis, block_index, chosen_voice, \
           chosen_voice_name, filter1, filter2, tlo1, tlo2, t1_total, t2_total, streams_df, trial_seq1, trial_seq2, noise_trials_count, \
           idx_to_replace, precomputed_animal_sounds, concatenated_animal_sounds, animal_names


if __name__ == "__main__":
    subject_id = input('subject_id: ') # subject number i.e. sub01
    participant_id = get_participant_id(subject_id)
    animal_sounds_csv = params_dir / f'animal_blocks/{participant_id}.csv'
    animal_df.to_csv(animal_sounds_csv, index=False)
    freefield.initialize('dome', device=proc_list)
    save_block_seq()  # works
    s1_delay, s2_delay, target_stream, s1_params, s2_params, axis, block_index, chosen_voice, \
    chosen_voice_name, filter1, filter2, tlo1, tlo2, t1_total, t2_total, streams_df, trial_seq1, trial_seq2, noise_trials_count, \
    idx_to_replace, precomputed_animal_sounds, concatenated_animal_sounds, animal_names = run_experiment()
#   # always check speaker/processors

