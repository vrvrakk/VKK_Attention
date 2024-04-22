import random
import numpy
import pandas as pd
from block_sequence import block_sequence
from block_index import increment_block_index, block_index


numbers = [1, 2, 3, 4, 5, 6, 8, 9]
isi = numpy.array((275, 180))
duration_s = 120  # 5 min total
stim_dur_ms = 745  # duration in ms
tlo1 = stim_dur_ms + isi[0]
tlo2 = stim_dur_ms + isi[1]
# isi_1 = (441, 243)
# isi_3 = (275, 180)


block_seqs_df = block_sequence()


def get_delays(duration_s, isi):
    global block_index
    # default delay values:
    if block_index >= len(block_seqs_df):
        return
    s1_delay = 1
    s2_delay = 1
    target = None
    current_value = block_seqs_df.values[block_index]
    if current_value[0] == 's1':
        print('Target is s1, therefore s2 is delayed by 2s.')
        target = current_value[0]
        s1_delay = 1
        s2_delay = tlo1 * 2
    elif current_value[0] == 's2':
        print('Target is s2, therefore s1 is delayed by 2s.')
        target = current_value[0]
        s1_delay = tlo1 * 2
        s2_delay = 1
    n_trials1 = int(numpy.floor((duration_s - (s1_delay / 1000)) / ((isi[0] + stim_dur_ms) / 1000))) # todo: do the math lol
    n_trials2 = int(numpy.floor((duration_s - (s2_delay / 1000)) / ((isi[1] + stim_dur_ms) / 1000)))
    return s1_delay, s2_delay, target, n_trials1, n_trials2


def get_timepoints(tlo1, tlo2, n_trials1, n_trials2):

    t1_total = (tlo1 * n_trials1)
    t2_total = (tlo2 * n_trials2)
    return t1_total, t2_total


def streams_dfs(tlo1, tlo2, t1_total, t2_total, s1_delay, s2_delay):
    t1_timepoints = []
    for t1 in range(0, t1_total, tlo1):
        t1_timepoints.append((t1 + s1_delay, 's1'))
    t1_df = pd.DataFrame(t1_timepoints, columns=['Timepoints', 'Stimulus Type'])

    t2_timepoints = []
    for t2 in range(0, t2_total, tlo2):
        t2_timepoints.append((t2 + s2_delay, 's2'))
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
    numbers = [1, 2, 3, 4, 5, 6, 8, 9]
    return streams_df


def get_trial_sequence(streams_df):
    # get trial sequences:
    trial_seq1 = streams_df.loc[streams_df['Stimulus Type'] == 's1', 'Numbers'].tolist()
    trial_seq2 = streams_df.loc[streams_df['Stimulus Type'] == 's2', 'Numbers'].tolist()
    return trial_seq1, trial_seq2


def get_stream_params(s1_delay, s2_delay, n_trials1, n_trials2):
    global block_seqs_df
    global block_index
    speakers_coordinates = (17.5, 0)  # directions for each streams
    azimuth = ((speakers_coordinates[0], 0), (speakers_coordinates[1], 0))
    elevation = ((speakers_coordinates[1], -37.5), (speakers_coordinates[1], -12.5))
    if block_index >= len(block_seqs_df):
        return
    s1_params = {}
    s2_params = {}
    axis = None
    current_values = block_seqs_df.values[block_index]
    if current_values[1] == 'azimuth':
        axis = current_values[1]
        s1_params = {'isi': isi[0], 's_delay': s1_delay, 'n_trials': n_trials1, 'speakers_coordinates': azimuth[0], 'block_index': block_index}
        s2_params = {'isi': isi[1], 's_delay': s2_delay, 'n_trials': n_trials2, 'speakers_coordinates': azimuth[1], 'block_index': block_index}
    elif current_values[1] == 'ele':
        axis = current_values[1]
        s1_params = {'isi': isi[0], 's_delay': s1_delay, 'n_trials': n_trials1, 'speakers_coordinates': elevation[0], 'block_index': block_index}
        s2_params = {'isi': isi[0], 's_delay': s2_delay, 'n_trials': n_trials2, 'speakers_coordinates': elevation[1], 'block_index': block_index}
    # parameters seem to be assigned as desired
    block_index = increment_block_index(block_index)
    return s1_params, s2_params, axis, block_index


