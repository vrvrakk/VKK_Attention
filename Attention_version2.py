import slab
import numpy
import os
import freefield
from get_streams_and_stream_params import get_delays, get_timepoints, streams_dfs, assign_numbers, get_trial_sequence, \
    get_stream_params, numbers, duration_s, isi, tlo1, tlo2, block_seqs_df, target_number_seq, choose_target_number,increase_prob_target_number
from block_index import increment_block_index, block_index
from block_sequence import get_target_number_seq
from generate_voice_list import voice_seq
import datetime
from pathlib import Path
import pandas as pd
import random


current_path = Path.cwd()
params_dir = Path(current_path / 'data'/'params')

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


def select_voice():  # write voice data onto rcx buffer
    global voice_index  # changes globally every time function is ran
    if voice_index is None:
        voice_index = 0
    if voice_index >= len(voice_seq):
        raise IndexError('last element of voice_index has been reached.')
    chosen_voice = voice_seq[voice_index]
    chosen_voice_name = chosen_voice[0].parent.name
    voice_index += 1

    return chosen_voice, chosen_voice_name


# get animal sounds sequence:
def animal_sounds(noise_trials_count, idx_to_replace):
    wav_path = current_path / 'data' / 'sounds'
    processed_path = wav_path / 'processed'
    processed_files = [os.path.join(processed_path, file) for file in os.listdir(processed_path) if file.endswith('.wav')]

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

    precomputed_animal_sounds = slab.Precomputed(animals)
    # selected_animal = precomputed_animal_sounds[block_index]
    for i in range(len(precomputed_animal_sounds)):
        precomputed_animal_sounds[i].level = 90  # make all same level
    concatenated_animal_sounds = numpy.concatenate([sound.data.flatten() for sound in precomputed_animal_sounds])
    return precomputed_animal_sounds, concatenated_animal_sounds

def write_buffer(chosen_voice, precomputed_animal_sounds, concatenated_animal_sounds):
    for number, file_path in zip(nums, chosen_voice):
        # combine lists into a single iterable
        # elements from corresponding positions are paired together
        if os.path.exists(file_path):
            s = slab.Sound(data=file_path) #.resample(samplerate=24414) # not needed
            s.level = 80
            freefield.write(f'{number}', s.data, ['RX81', 'RX82'])  # loads array on buffer
            freefield.write(f'{number}_n_samples', s.n_samples, ['RX81', 'RX82'])
            # sets total buffer size according to numeration

    freefield.write('noise', concatenated_animal_sounds, ['RX81', 'RX82'])
    freefield.write('noise_n_samples', int(concatenated_animal_sounds.size/len(precomputed_animal_sounds)), ['RX81', 'RX82'])
    # freefield.write('noise_n_samples', int(concatenated_animal_sounds.size/len(precomputed_animal_sounds)), ['RX81', 'RX82'])
    freefield.write('noise_size', concatenated_animal_sounds.size, ['RX81', 'RX82'])

def save_block_seq(): # works
    blocks_dir = params_dir / f'{subject_id}.csv'
    block_seqs_df.to_csv(blocks_dir, sep=';', index=False, columns=['block_seq', 'block_condition', 'Voices', 'Target Number'])



def run_block(trial_seq1, trial_seq2, tlo1, tlo2, s1_params, s2_params):
    # REMEMBER TO CHANGE N_PULSE FOR S2 WHEN RUNNING EXP AFTER TRAINING
    speakers_coordinates = (0, 0)
    [speaker1] = freefield.pick_speakers((s1_params.get('speakers_coordinates')))  # 17.5 az, 0.0 ele (target), or -12.5 ele
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
    statement = input('Start experiment? Y/n: ')
    if statement.lower() in ['y', '']:  # works
        freefield.play()
    else:
        freefield.halt()


def run_experiment():  # works as desired
    global block_index
    s1_delay, s2_delay, target_stream, n_trials1, n_trials2 = get_delays(duration_s, isi)
    t1_total, t2_total = get_timepoints(tlo1, tlo2, n_trials1, n_trials2)
    streams_df = streams_dfs(tlo1, tlo2, t1_total, t2_total, s1_delay, s2_delay)
    target_number = choose_target_number(target_number_seq)
    streams_df, target_stream_df = assign_numbers(streams_df, numbers, tlo1, target_stream, target_number)
    streams_df_updated = increase_prob_target_number(streams_df, target_number, target_stream, target_stream_df)
    trial_seq1, trial_seq2 = get_trial_sequence(streams_df_updated)
    s1_params, s2_params, axis, block_index, trial_seq1, trial_seq2, noise_trials_count, idx_to_replace = get_stream_params(s1_delay, s2_delay, n_trials1, n_trials2, trial_seq1, trial_seq2, target_number) # block index incremented in this function
    precomputed_animal_sounds, concatenated_animal_sounds = animal_sounds(noise_trials_count, idx_to_replace)
    chosen_voice, chosen_voice_name = select_voice()
    write_buffer(chosen_voice, precomputed_animal_sounds, concatenated_animal_sounds)
    run_block(trial_seq1, trial_seq2, tlo1, tlo2, s1_params, s2_params)
    return s1_delay, s2_delay, target, s1_params, s2_params, axis, block_index, chosen_voice, \
           chosen_voice_name, tlo1, tlo2, t1_total, t2_total, streams_df, trial_seq1, trial_seq2, noise_trials_count, idx_to_replace, precomputed_animal_sounds, concatenated_animal_sounds


if __name__ == "__main__":
    subject_id = input('subject_id: ')
    participant_id = get_participant_id(subject_id)
    freefield.initialize('dome', device=proc_list)
    save_block_seq()  # works
    s1_delay, s2_delay, target, s1_params, s2_params, axis, block_index, chosen_voice, \
    chosen_voice_name, tlo1, tlo2, t1_total, t2_total, streams_df, trial_seq1, trial_seq2, noise_trials_count, idx_to_replace, precomputed_animal_sounds, concatenated_animal_sounds = run_experiment()
#     # # always check speaker/processors


# todo check block seq data