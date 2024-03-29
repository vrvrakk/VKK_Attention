import slab
import numpy
import os
import freefield
from pathlib import Path
from generate_voice_list import voice_seq
from get_streams_and_stream_params import block_seqs_df
from get_streams_and_stream_params import get_delays, get_timepoints, streams_dfs, assign_numbers, get_trial_sequence, \
    get_stream_params, numbers
from block_index import increment_block_index, block_index
from generate_voice_list import voice_seq
import datetime


# subject ID:
subject_id = 'vkk'
# path:
sequence_path = Path.cwd() / 'data' / 'generated_sequences'

# processors:
proc_list = [['RX81', 'RX8', Path.cwd() / 'experiment.rcx'],
             ['RX82', 'RX8', Path.cwd() / 'experiment.rcx']]

voice_index = 0
nums = [1, 2, 3, 4, 5, 6, 8, 9]


def get_participant_id():
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
    statement = print(f'For voice_index: {voice_index}, {chosen_voice_name} was selected. Files: {chosen_voice}')
    voice_index += 1

    return chosen_voice, chosen_voice_name, statement


def write_buffer(chosen_voice):
        for number, file_path in zip(nums, chosen_voice):
            # combine lists into a single iterable
            # elements from corresponding positions are paired together
            if os.path.exists(file_path):
                print('file_path exists')
                s = slab.Sound(data=file_path)
                freefield.write(f'{number}', s.data, ['RX81', 'RX82'])  # loads array on buffer
                freefield.write(f'{number}_n_samples', s.n_samples, ['RX81', 'RX82'])
                print("write_buffer() execution completed successfully.")
                # sets total buffer size according to numeration


def save_sequence(target, axis, participant_id, streams_df, chosen_voice_name):
    if block_index > 0:
        index = block_index - 1
    else:
        index = block_index
    file_name = f'{sequence_path}/{target}_{index}_{participant_id}_{chosen_voice_name}_{axis}.csv'
    stem = f'{target}_{index}_{participant_id}_{chosen_voice_name}_{axis}'
    streams_df.to_csv(file_name, index=False, sep=';')
    return file_name, stem


def save_block_info_txt(stem):
    output_dir = Path.cwd() / 'data' / 'misc' / 'blocks_info.txt'
    with open(output_dir, 'a') as file:
        file.write(stem + '\n')


def run_block(trial_seq1, trial_seq2, tlo1, tlo2, s1_params, s2_params):
    [speaker1] = freefield.pick_speakers((s1_params.get('speakers_coordinates')))  # 17.5 az, 0.0 ele (target), or -12.5 ele
    [speaker2] = freefield.pick_speakers((s2_params.get('speakers_coordinates')))  # 0.0 az, 0.0 ele, or 12.5 ele

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
    freefield.write('channel1', speaker1.analog_channel, speaker1.analog_proc)  # s1 target both to RX8I
    freefield.write('channel2', speaker2.analog_channel, speaker2.analog_proc)  # s2 distractor
    statement = input('Start experiment? Y/n: ')
    if statement.lower() in ['y', '']:  # works
        freefield.play()
    else:
        freefield.halt()


def run_experiment():  # works as desired
    global block_index
    participant_id = get_participant_id()  # works
    s1_delay, s2_delay, target = get_delays()
    s1_params, s2_params, axis, block_index = get_stream_params(s1_delay, s2_delay) # block index incremented in this function
    chosen_voice, chosen_voice_name, statement = select_voice()
    write_buffer(chosen_voice)
    tlo1, tlo2, t1_total, t2_total = get_timepoints()
    streams_df = streams_dfs(tlo1, tlo2, t1_total, t2_total, s1_delay, s2_delay)
    streams_df = assign_numbers(streams_df, numbers, tlo1)
    trial_seq1, trial_seq2 = get_trial_sequence(streams_df)
    file_name, stem = save_sequence(target, axis, participant_id, streams_df, chosen_voice_name)
    save_block_info_txt(stem)
    run_block(trial_seq1, trial_seq2, tlo1, tlo2, s1_params, s2_params)
    return participant_id, s1_delay, s2_delay, target, s1_params, s2_params, axis, block_index, chosen_voice, \
           chosen_voice_name, tlo1, tlo2, t1_total, t2_total, streams_df, trial_seq1, trial_seq2, file_name, stem


if __name__ == "__main__":
    freefield.initialize('dome', device=proc_list)

    participant_id, s1_delay, s2_delay, target, s1_params, s2_params, axis, block_index, chosen_voice, \
    chosen_voice_name, tlo1, tlo2, t1_total, t2_total, streams_df, trial_seq1, trial_seq2, file_name, stem = run_experiment()
    # always check speaker/processors
''' 

# PLOTTING TRIAL SEQUENCES OVER TIME

trials_1 = [trial1[1] for trial1 in trials_dur1]  # Trial numbers for Stream 1
onsets_1 = [t1_onset[2] for t1_onset in trials_dur1]  # Onsets for Stream 1

trials_2 = [trial2[1] for trial2 in trials_dur2]  # Trial numbers for Stream 2
onsets_2 = [t2_onset[2] for t2_onset in trials_dur2]  # Onsets for Stream 2

# Create the plot
plt.figure(figsize=(15, 8))

# Plotting the trials with their onsets for both streams with lines
plt.plot(onsets_1, trials_1, label='Stream 1', marker='o', linestyle='-', alpha=0.7)
plt.plot(onsets_2, trials_2, label='Stream 2', marker='x', linestyle='-', alpha=0.7)

# Adding labels, title, and legend
plt.xlabel('Time Onset (ms)')
plt.ylabel('Trial Number')
plt.title('Trial Numbers Over Time for Stream 1 and Stream 2')
plt.legend()

# Optionally, set the limits for better visibility if needed
plt.xlim(min(onsets_1 + onsets_2), max(onsets_1 + onsets_2))
plt.ylim(min(trials_1 + trials_2) - 1, max(trials_1 + trials_2) + 1)  # Adjusted for better y-axis visibility

# Show grid
plt.grid(True)

# Show the plot
plt.show()
'''