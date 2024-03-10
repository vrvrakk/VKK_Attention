import slab
import numpy
import os
import freefield
from pathlib import Path
import pandas as pd
import random
import matplotlib.pyplot as plt

# s2_delay = 10
# Dai & Shinn-Cunningham (2018):
isi = numpy.array((741, 543))  # tlo1 = 985, tlo2 = 925
duration_s = 300  # 5 min total
stim_dur_ms = 745  # duration in ms
n_trials1 = int(numpy.floor((duration_s) / ((isi[0] + stim_dur_ms) / 1000)))
n_trials2 = int(numpy.floor((duration_s) / ((isi[1] + stim_dur_ms) / 1000)))
numbers = [1, 2, 3, 4, 5, 6, 8, 9]

# choose speakers:
speakers_coordinates = (17.5, 0)  # directions for each streams
# s2_delay = 2000 # delay not necessary if target always on the right
sample_freq = 24414
data_path = Path.cwd() / 'data' / 'voices_padded'
sequence_path = Path.cwd() / 'data' / 'generated_sequences'
chosen_voice_path = Path.cwd() / 'data' / 'chosen_voice'
participant_id = 'kolos'


proc_list = [['RX81', 'RX8', Path.cwd() / 'experiment.rcx'],
             ['RX82', 'RX8', Path.cwd() / 'experiment.rcx']]

freefield.set_logger('info')


def wav_list_select(data_path):  # create wav_list paths, and select a voice folder randomly
    voice_idx = list(range(1, 5))
    folder_paths = []
    wav_folders = [folder for folder in os.listdir(data_path)]
    for i, folder in zip(voice_idx, wav_folders):
        folder_path = data_path / folder
        folder_paths.append(folder_path)  # absolute path of each voice folder

    name_mapping = {
        0: 'Matilda',
        1: 'Johanna',
        2: 'Carsten',
        3: 'Marc'
    }
    # Initialize the corresponding wav_files list
    wav_files_lists = []
    for i, folder_path in zip(voice_idx, folder_paths):
        wav_files_in_folder = list(os.listdir(folder_path))
        wav_files_lists.append(wav_files_in_folder)

    chosen_voice = random.choice(wav_files_lists)
    chosen_voice_name = name_mapping[wav_files_lists.index(chosen_voice)]

    index = 1
    while True:
        chosen_voice_file = os.path.join(chosen_voice_path, f'{participant_id}_{chosen_voice_name}_block_{index}.txt')
        if not os.path.exists(chosen_voice_file):
            break
        index += 1

    with open(chosen_voice_file, 'w') as file:
        file.write(str(chosen_voice))

    return chosen_voice


def write_buffer(chosen_voice):  # write voice data onto rcx buffer
    for number, file_path in zip(numbers, chosen_voice):
        # combine lists into a single iterable
        # elements from corresponding positions are paired together
        if os.path.exists(file_path):
            s = slab.Sound(data=file_path)
            freefield.write(f'{number}', s.data, ['RX81', 'RX82'])  # loads array on buffer
            freefield.write(f'{number}_n_samples', s.n_samples, ['RX81', 'RX82'])
            # sets total buffer size according to numeration


def get_timepoints(n_trials1, n_trials2, stim_dur_ms, isi):

    tlo1 = stim_dur_ms + isi[0]
    tlo2 = stim_dur_ms + isi[1]

    t1_total = tlo1 * n_trials1
    t2_total = tlo2 * n_trials2
    return tlo1, tlo2, t1_total, t2_total


def streams_dfs(tlo1, tlo2, t1_total, t2_total):
    t1_timepoints = []
    for t1 in range(0, t1_total, tlo1):
        t1_timepoints.append((t1, 's1'))
    t1_df = pd.DataFrame(t1_timepoints, columns=['Timepoints', 'Stimulus Type'])

    t2_timepoints = []
    for t2 in range(0, t2_total, tlo2):
        t2_timepoints.append((t2, 's2'))
    t2_df = pd.DataFrame(t2_timepoints, columns=['Timepoints', 'Stimulus Type'])

    streams_df = pd.concat((t1_df, t2_df))
    streams_df = streams_df.sort_values(by='Timepoints', ascending=True).reset_index(drop=True)
    streams_df['Numbers'] = None

    return streams_df


def assign_numbers(streams_df, numbers, tlo1):
    # rolling window:
    random.shuffle(numbers)
    used_numbers = set()
    for index, row in streams_df.iterrows():
        window_start = row['Timepoints'] - tlo1
        window_end = row['Timepoints'] + tlo1

        window_data = streams_df[(streams_df['Timepoints'] >= window_start) & (streams_df['Timepoints'] <= window_end)]
        possible_numbers = [x for x in numbers if x not in window_data['Numbers'].tolist()]
        if possible_numbers:
            assigned_number = possible_numbers[0]
            streams_df.at[index, 'Numbers'] = assigned_number
            used_numbers.add(assigned_number)
            numbers.remove(assigned_number)
        else:
            if len(numbers) == 0:
                numbers = [1, 2, 3, 4, 5, 6, 8, 9]
                random.shuffle(numbers)
            possible_numbers = [x for x in numbers if x not in window_data['Numbers'].tolist()]
            assigned_number = possible_numbers[0]
            streams_df.at[index, 'Numbers'] = assigned_number
            used_numbers.add(assigned_number)
            numbers.remove(assigned_number)
    return streams_df

def get_trial_sequence(streams_df):
    # get trial sequences:
    trial_seq1 = streams_df.loc[streams_df['Stimulus Type'] == 's1', 'Numbers'].tolist()
    trial_seq2 = streams_df.loc[streams_df['Stimulus Type'] == 's2', 'Numbers'].tolist()
    return trial_seq1, trial_seq2


def save_sequence(participant_id, sequence_path, streams_df):
    file_name = f'{sequence_path}/{participant_id}_block_1.csv'
    index = 1
    while os.path.exists(file_name):
        file_name = f'{sequence_path}/{participant_id}_block_{index}.csv'
        index += 1
    streams_df.to_csv(file_name, index=False, sep=';')

    return file_name


def run_block(trial_seq1, trial_seq2, tlo1, tlo2):
    # [speaker1] = freefield.pick_speakers((speakers_coordinates[0], 0))  # speaker 31, 17.5 az, 0.0 ele (target)
    # [speaker2] = freefield.pick_speakers((speakers_coordinates[1], 0))  # speaker 23, 0.0 az, 0.0 ele

    # elevation coordinates: works
    [speaker1] = freefield.pick_speakers((speakers_coordinates[1], -12.5))  # s1 target
    [speaker2] = freefield.pick_speakers((speakers_coordinates[1], 12.5))  # s2 distractor

    sequence1 = numpy.array(trial_seq1).astype('int32')
    sequence1 = numpy.append(0, sequence1)
    sequence2 = numpy.array(trial_seq2).astype('int32')
    sequence2 = numpy.append(0, sequence2)
    # here we set tlo to RX8
    freefield.write('tlo1', tlo1, ['RX81', 'RX82'])
    freefield.write('tlo2', tlo2, ['RX81', 'RX82'])
    # set n_trials to pulse trains sheet0/sheet1
    freefield.write('n_trials1', n_trials1 + 1,
                    ['RX81', 'RX82'])  # analog_proc attribute from speaker table dom txt file
    freefield.write('n_trials2', n_trials2 + 1, ['RX81', 'RX82'])
    freefield.write('trial_seq1', sequence1, ['RX81', 'RX82'])
    freefield.write('trial_seq2', sequence2, ['RX81', 'RX82'])
    # set output speakers for both streams
    freefield.write('channel1', speaker1.analog_channel, speaker1.analog_proc)  # s1 target both to RX8I
    freefield.write('channel2', speaker2.analog_channel, speaker2.analog_proc)  # s2 distractor
    freefield.play()


def run_experiment():  # works as desired
    chosen_voice = wav_list_select(data_path)
    write_buffer(chosen_voice)
    tlo1, tlo2, t1_total, t2_total = get_timepoints(n_trials1, n_trials2, stim_dur_ms, isi)
    streams_df = streams_dfs(tlo1, tlo2, t1_total, t2_total)
    streams_df = assign_numbers(streams_df, numbers, tlo1)
    trial_seq1, trial_seq2 = get_trial_sequence(streams_df)
    file_name = save_sequence(participant_id, sequence_path, streams_df)

    run_block(trial_seq1, trial_seq2, tlo1, tlo2)


if __name__ == "__main__":
    freefield.initialize('dome', device=proc_list)

    run_experiment()
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