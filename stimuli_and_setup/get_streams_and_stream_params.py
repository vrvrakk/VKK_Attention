import random
import numpy as np
import pandas as pd
from stimuli_and_setup.block_sequence import block_sequence, get_target_number_seq
from stimuli_and_setup.block_index import increment_block_index, block_index
from stimuli_and_setup.generate_voice_list import voice_seq

'''FOR NOW ONLY AZIMUTH'''

numbers = [1, 2, 3, 4, 5, 6, 8, 9]
isi = np.array((90, 70))
duration_s = 120  # 2 min total
stim_dur_ms = 745  # duration in ms
tlo1 = stim_dur_ms + isi[0]
tlo2 = stim_dur_ms + isi[1]


target_number_seq = get_target_number_seq()
block_seqs_df = block_sequence(target_number_seq)

# so far works as expected

def get_delays(duration_s, isi):
    global block_index
    # default delay values:
    if block_index >= len(block_seqs_df):
        return
    s1_delay = 1
    s2_delay = 1
    target_stream = None
    current_value = block_seqs_df.values[block_index]
    if current_value[0] == 's1':
        print('Target stream is s1, therefore s2 is delayed by 2s.')
        target_stream = current_value[0]
        s1_delay = 1
        s2_delay = tlo1 * 3.5
    elif current_value[0] == 's2':
        print('Target stream is s2, therefore s1 is delayed by 2s.')
        target_stream = current_value[0]
        s1_delay = tlo1 * 3.5
        s2_delay = 1
    n_trials1 = int(np.floor((duration_s - (s1_delay / 1000)) / ((isi[0] + stim_dur_ms) / 1000)))
    n_trials2 = int(np.floor((duration_s - (s2_delay / 1000)) / ((isi[1] + stim_dur_ms) / 1000)))

    return s1_delay, s2_delay, target_stream, n_trials1, n_trials2


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


def choose_target_number(block_seqs_df):
    # fixed indexing of target number
    # block index is a global var
   target_number = block_seqs_df.loc[block_index, 'Target Number']
   return target_number

def assign_numbers(streams_df, numbers, tlo1, target_stream, target_number):
    random.shuffle(numbers)
    used_numbers = set()
    for index, row in streams_df.iterrows():
        window_start = row['Timepoints'] - (tlo1 + 0.2)
        window_end = row['Timepoints'] + (tlo1 + 0.2)  # changed window len

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
    # get dataframe of stream of numbers for each block:
    target_stream_df = streams_df[streams_df['Stimulus Type'] == target_stream]
    # to  get total number of target occurrences:
    target_number_df = target_stream_df[target_stream_df['Numbers'] == target_number]
    print(f'Total occurrences of {target_number}: {len(target_number_df)}')
    return streams_df, target_stream_df


def increase_prob_target_number(streams_df, target_number, target_stream, target_stream_df):
    timepoints = []
    indices = streams_df.index.tolist()
    for index in indices:
        if index < len(indices):
            timepoint = streams_df.at[index, 'Timepoints']
            timepoints.append(timepoint)
    time_differences = []
    for index in range(len(timepoints)-1):
        time_difference = timepoints[index+1] - timepoints[index]
        time_differences.append(time_difference)
    median_time_difference = int(np.median(time_differences))
    # set rolling window:
    time_window = median_time_difference
    current_values = block_seqs_df.values[block_index]
    target_stimulus = current_values[0]
    non_target_nums = target_stream_df[target_stream_df['Numbers'] != target_number]
    sum_options = len(non_target_nums)
    target_probability = 0.35
    sum_numbers_to_change = int(sum_options * target_probability)
    # get indices of rows of target stimulus, that are not the target number:
    non_target_indices = non_target_nums.index
    # Brute force selection and update
    np.random.seed(0)  # For reproducibility
    random_indices = np.random.choice(non_target_indices, size=sum_numbers_to_change, replace=False)

    # Update the 'Numbers' column for the selected indices with brute force
    for idx in random_indices:
        timepoint = streams_df.loc[idx, 'Timepoints']
        start_time = timepoint - median_time_difference - 600
        end_time = timepoint + median_time_difference + 600

        # Check for existence of target_number in the time window
        time_window = streams_df[(streams_df['Timepoints'] >= start_time) & (streams_df['Timepoints'] <= end_time)]
        if target_number not in time_window['Numbers'].values:
            streams_df.loc[idx, 'Numbers'] = target_number
    target_stream_df = streams_df[streams_df['Stimulus Type'] == target_stream]
    target_number_df = target_stream_df[target_stream_df['Numbers'] == target_number]
    print(f'Total occurrences of {target_number} in updated streams Df: {len(target_number_df)}')
    return streams_df


    # iterate over rows with numbers of target stimulus, and change some of them to target number, while ensuring that
    # other numbers in the streams_df within the time window, are not the same number

def get_trial_sequence(streams_df):
    # get trial sequences:
    trial_seq1 = streams_df.loc[streams_df['Stimulus Type'] == 's1', 'Numbers'].tolist()
    trial_seq2 = streams_df.loc[streams_df['Stimulus Type'] == 's2', 'Numbers'].tolist()
    return trial_seq1, trial_seq2


def get_stream_params(s1_delay, s2_delay, n_trials1, n_trials2, trial_seq1, trial_seq2, target_number):
    # speaker coordinates:
    speakers_coordinates = (17.5, -17.5)
    azimuth_s1_coordinates = (speakers_coordinates[0], 0)  # (azimuth, elevation)
    azimuth_s2_coordinates = (speakers_coordinates[1], 0)
    ele_s1_coordinates = (0, -37.5)
    ele_s2_coordinates = (0, 37.5)
    global block_seqs_df, idx_to_replace
    global block_index
    if block_index >= len(block_seqs_df):
        return
    s1_params = {}
    s2_params = {}
    axis = None
    current_values = block_seqs_df.values[block_index]
    if current_values[1] == 'azimuth':
        target = current_values[0]
        axis = current_values[1]
        s1_params = {'number': target_number, 'target': target, 'isi': isi[0], 's_delay': s1_delay, 'n_trials': n_trials1,
                     'speakers_coordinates': azimuth_s1_coordinates, 'block_index': block_index}
        s2_params = {'number': target_number, 'target': target, 'isi': isi[1], 's_delay': s2_delay, 'n_trials': n_trials2,
                     'speakers_coordinates': azimuth_s2_coordinates, 'block_index': block_index}
        print(
            f'Block {block_index}, Target Number: {target_number}, Target: {target}, Axis: {current_values[1]}, s1_coordinates: {azimuth_s1_coordinates}, s2_coordinates: {azimuth_s2_coordinates}')
        if target == 's1':
            noise_trials_count = int((len(trial_seq2) * 0.11))
            # Randomly select unique indices to be replaced
            while True:
                idx_to_replace = random.sample(range(len(trial_seq2)), noise_trials_count)
                idx_to_replace.sort()
                idx_diff = abs(np.diff(idx_to_replace))
                if np.all(idx_diff >= 4):
                    break
            # Replace the selected indices with 7
            for index in idx_to_replace:
                trial_seq2[index] = 7
        elif target == 's2':
            noise_trials_count = int((len(trial_seq1) * 0.11))
            # Randomly select unique indices to be replaced
            while True:
                idx_to_replace = random.sample(range(len(trial_seq1)), noise_trials_count)
                idx_to_replace.sort()
                idx_diff = abs(np.diff(idx_to_replace))
                if np.all(idx_diff >= 4):
                    break
            # Replace the selected indices with 7
            for index in idx_to_replace:
                trial_seq1[index] = 7
    elif current_values[1] == 'elevation':
        target = current_values[0]
        axis = current_values[1]
        s1_params = {'number': target_number, 'target': target, 'isi': isi[0], 's_delay': s1_delay, 'n_trials': n_trials1, 'speakers_coordinates': ele_s1_coordinates, 'block_index': block_index}
        s2_params = {'number': target_number, 'target': target, 'isi': isi[1], 's_delay': s2_delay, 'n_trials': n_trials2, 'speakers_coordinates': ele_s2_coordinates, 'block_index': block_index}
        print(
            f'Block {block_index}, Target Number: {target_number}, Target: {target}, Axis: {current_values[1]}, Target: {current_values[0]}, s1_coordinates: {ele_s1_coordinates}, s2_coordinates: {ele_s2_coordinates}')
        if target == 's1':
            noise_trials_count = int((len(trial_seq2) * 0.11))
            # Randomly select unique indices to be replaced
            while True:
                idx_to_replace = random.sample(range(len(trial_seq2)), noise_trials_count)
                idx_to_replace.sort()
                idx_diff = abs(np.diff(idx_to_replace))
                if np.all(idx_diff >= 4):
                    break
            # Replace the selected indices with 7
            for index in idx_to_replace:
                if trial_seq2[index] != target_number:
                    trial_seq2[index] = 7
        elif target == 's2':
            noise_trials_count = int((len(trial_seq1) * 0.11))
            # Randomly select unique indices to be replaced
            while True:
                idx_to_replace = random.sample(range(len(trial_seq1)), noise_trials_count)
                idx_to_replace.sort()
                idx_diff = abs(np.diff(idx_to_replace))
                if np.all(idx_diff >= 4):
                    break
            # Replace the selected indices with 7
            for index in idx_to_replace:
                if trial_seq1[index] != target_number:
                    trial_seq1[index] = 7
    # parameters seem to be assigned as desired
    block_index = increment_block_index(block_index)
    return s1_params, s2_params, axis, block_index, trial_seq1, trial_seq2, noise_trials_count, idx_to_replace
