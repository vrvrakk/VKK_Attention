import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

default_dir = Path.cwd()
sub = input('Subject number: ')

# specify condition to focus on:
condition = input('Choose a condition (a1, a2, e1 or e2): ')


sub_dir = default_dir / 'data' / 'eeg' / 'raw' / f'{sub}'
performance_events = default_dir / 'data' / 'misc' / 'performance_events.json'
fig_path = default_dir / 'data' / 'performance' / f'{sub}'
rt_path = fig_path / 'RTs'

if not os.path.exists(rt_path):
    os.makedirs(rt_path)

# load performance_events dictionary:
with open(performance_events, 'r') as file:
    markers_dict = json.load(file)


# define events by creating separate dictionaries with keys and their corresponding values:
s1_events = markers_dict['s1_events']
s2_events = markers_dict['s2_events']  # stimulus 2 markers
response_events = markers_dict['response_events']  # response markers

# first define csv path:
csv_path = default_dir / 'data' / 'params' / 'block_sequences' / f'{sub}.csv'
# read csv path
csv = pd.read_csv(csv_path)



def get_marker_files(): # if you define a variable outside a condition, you don't need to add it as a parameter in the parenthesis
    # define axis of condition:
    if condition == 'a1' or condition == 'a2':
        axis = 'azimuth'
    elif condition == 'e1' or condition == 'e2':
        axis = 'elevation'

    # extract marker files; they contain all the events that occurred during the exp.
    marker_files = []
    for files in sub_dir.iterdir():
        if files.is_file() and f'{condition}' in files.name:
            if files.is_file() and '.vmrk' in files.name:
                marker_files.append(files)
    return marker_files, axis


def dataframe():
    dfs = {}  # this dataFrame will contain sub-dataframes for every marker file

    for index, files in enumerate(marker_files):
        df = pd.read_csv(sub_dir/files.name, delimiter='\t', header=None)  # separate along rows
        df_name = f'df_{sub}_{condition}_{index}' # name according to files.name (contains sub initials + condition
        df = df.iloc[10:]  # delete first 10 rows  (because they contain nothing useful for us)
        df = df.reset_index(drop=True, inplace=False)  # once rows are dropped, we need to reset index so it starts from 0 again
        df = df[0].str.split(',', expand=True).applymap(lambda x: None if x == '' else x) # separates info from every row into separate columns
        # information is split whenever a ',' comma is present; otherwise all info in each row is all under one column -> which sucks
        df = df.iloc[:, 1:3] # we keep only the columns 1:3, with all their data
        df.insert(0, 'Stimulus Type', None)  # we insert an additional column, which we will fill in later
        df.insert(2, 'Numbers', None)  # same here
        columns = ['Stimulus Stream', 'Position', 'Time Difference']  # we pre-define some columns of our dataframe;
        # position is time in data samples
        df.columns = ['Stimulus Type'] + [columns[0]] + ['Numbers'] + [columns[1]]  # we re-order our columns
        df['Timepoints'] = None
        dfs[df_name] = df  # we save every single sub-dataframe into the empty dfs dictionary we created
    return dfs


def define_stimuli(dfs): # if you want to manipulate a variable that was created within a function, enter here in parenthesis
    # define our Stimulus Type:
    for df_name, df in dfs.items(): # we iterate through every sub-dataframe of chosen condition
        for index, stim_mrk in enumerate(df['Stimulus Stream']): # we go through the cells of Stimulus Stream
            if stim_mrk in s1_events.keys(): # if the key within one cell, matches an element from the s1_events dictionary
                df.at[index, 'Stimulus Type'] = 's1'  # enter value 's1' in the corresponding cell under Stimulus Type
                df.at[index, 'Numbers'] = next(value for key, value in s1_events.items() if key == stim_mrk) # if the key == stimulus marker,
                # iterate through the dictionary s1_events
                # and find the corresponding value to that key -> that value is the actual number said during that stimulus
                # for the numbers column, find the corresponding key/number from the s1_events dictionary, that matches the
                # s1 event at that specific row
            elif stim_mrk in s2_events.keys(): # same process for s2
                # print('stimulus marker is type 2')
                df.at[index, 'Stimulus Type'] = 's2'
                df.at[index, 'Numbers'] = next(value for key, value in s2_events.items() if key == stim_mrk)
            elif stim_mrk in response_events.keys(): # same for responses
                # print('stimulus marker is type response')
                df.at[index, 'Stimulus Type'] = 'response'
                df.at[index, 'Numbers'] = next(value for key, value in response_events.items() if key == stim_mrk)
    return dfs  # always return manipulated variable, if you are to further manipulate it in other functions


def clean_rows(dfs):
    # drop any remaining rows with None Stimulus Types (like: 'S 64')
    for df_name, df in dfs.items():
        rows_to_drop = []
        for index, stim_mrk in enumerate(df['Stimulus Stream']):
            if stim_mrk not in s1_events.keys() and stim_mrk not in s2_events.keys() and stim_mrk not in response_events.keys():
                rows_to_drop.append(index)
        # Drop the marked rows from the DataFrame
        df.drop(rows_to_drop, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return dfs


# define target number for each block:
def get_target_blocks():
    # define target stream: s1 or s2?
    # we know this from the condition
    if condition in ['a1', 'e1']:
        target_stream = 's1'
    elif condition in ['a2', 'e2']:
        target_stream = 's2'
    else:
        raise ValueError("Invalid condition provided.")  # Handle unexpected conditions
    target_blocks = []
    # iterate through the values of the csv path; first enumerate to get the indices of each row
    for index, items in enumerate(csv.values):
        block_seq = items[0]  # from first column, save info as variable block_seq
        block_condition = items[1]  # from second column, save info as var block_condition
        if block_seq == target_stream and block_condition == axis:
            block = csv.iloc[index]
            target_blocks.append(block)  # Append relevant rows
    target_blocks = pd.DataFrame(target_blocks).reset_index(drop=True)  # convert condition list to a dataframe
    return target_stream, target_blocks


def target_stimulis(target_blocks, dfs):
    target_stimuli_df = {}
    for target_block, df_name in zip(target_blocks.values, dfs): # iterate through target_block and the dataframes together (zipping-> gotta have same length)
        target_number = target_block[3]  # define target number as the value within the column[3] of the target_block dataframe
        df = dfs[df_name]  # which corresponds to a specific sub-df
        target_stimuli = df[df['Numbers'] == target_number]
        target_stimuli_df[df_name] = target_stimuli
        # (s1, s2 and response with target num)
    return target_stimuli_df


def responses(target_stimuli_df):
    response_df = {}
    for df_name, df in target_stimuli_df.items():
        responses = df[df['Stimulus Type'] == 'response'].copy()  # get a deepcopy to avoid SettingWithCopyWarning
        responses.sort_values(by=['Time Difference'], ascending=False, inplace=True)
        response_df[df_name] = responses
    reaction_time_stats_df = pd.DataFrame(columns=['Min', 'Max', 'Mean', 'Median'])
    # get mean, median and maximum reaction times from every block of selected condition
    for df_name, df in response_df.items():
        df = df.copy() # get a deepcopy to avoid SettingWithCopyWarning
        df = df.sort_values(by=['Time Difference'], ascending=True)
        min_val = np.min(df['Time Difference'].values)
        max_val = np.max(df['Time Difference'].values)
        mean_val = np.mean(df['Time Difference'].values)
        median_val = np.median(df['Time Difference'].values)
        reaction_time_stats_df.loc[df_name] = [min_val, max_val, mean_val, median_val]  # .loc allows us to assign values to a specific row in the DataFrame
        # by specifying the row label (df_name) and the column names (['Max', 'Mean', 'Median']
    return response_df, reaction_time_stats_df


def avg_rt_stats_combined(reaction_time_stats_df): # get average statistics of RTs for all blocks combined:
    min_val = reaction_time_stats_df['Min'].mean()
    max_val = reaction_time_stats_df['Max'].mean()
    mean_val = reaction_time_stats_df['Mean'].mean()
    median_val = reaction_time_stats_df['Median'].mean()

    std_val = reaction_time_stats_df['Mean'].std()

    avg_stats_df = pd.DataFrame({
        'Statistic': ['Min', 'Max', 'Mean', 'Median'],
        'Average Value': [min_val, max_val, mean_val, median_val]
    })
    print(f"Mean of Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Median: {median_val}")

    # Define time window around the mean reaction time ± 1.5 * standard deviation
    time_window_start = mean_val - 1.5 * std_val
    time_window_end = mean_val + 1.5 * std_val

    # Ensure the time window is within reasonable bounds (e.g., not negative)
    time_window_start = max(0, time_start)
    time_window_end = max(time_start, time_end)  # Ensure window end is not less than start

    print(f"Time window defined from {time_start:.3f} to {time_end:.3f} seconds")
    # Add time window information to the DataFrame
    time_window_df = pd.DataFrame({
        'Time Window Start': [time_start],
        'Time Window End': [time_end]
    })

    combined_df = pd.concat([avg_stats_df, time_window_df], axis=1)
    filename = rt_path / f'rt_stats_{sub}_{condition}.csv'
    combined_df.to_csv(filename, index=False)
    return time_start, time_end

# define the time window:
# I selected the following time window, based on the reaction time statistics: 0.3-0.9
# this means, after a target stimulus onset, if there was a response 0.3 to 0.9 after its onset, we add this response to the 'correct_responses'
def classify_responses(target_stimuli_df, time_start, time_end, target_stream):
    correct_responses = set()
    distractor_responses = set()

    # Iterate through each DataFrame in target_stimuli_df
    for df_name, df in target_stimuli_df.items():
        target_stimulus = df[df['Stimulus Type'] == target_stream]
        distractor_stimulus = df[(df['Stimulus Type'] != target_stream) & (df['Stimulus Type'] != 'response')]
        responses = df[df['Stimulus Type'] == 'response']
        responses = responses.iloc[1:]

        for response_index, response_row in responses.iterrows():
            response_time = response_row['Timepoints']  # Time when the response occurred

            # Calculate proximity to the nearest target and distractor stimuli
            closest_target_time_diff = np.inf  # Initialize with a large value
            closest_distractor_time_diff = np.inf  # Initialize with a large value

            # Find the closest target stimulus
            for stim_index, stim_row in target_stimulus.iterrows():
                target_time = stim_row['Timepoints']
                if time_start <= (response_time - target_time) <= time_end:  # Within time window
                    time_diff = abs(response_time - target_time)
                    if time_diff < closest_target_time_diff:
                        closest_target_time_diff = time_diff

            # Find the closest distractor stimulus
            for stim_index, stim_row in distractor_stimulus.iterrows():
                distractor_time = stim_row['Timepoints']
                if time_start <= (response_time - distractor_time) <= time_end:  # Within time window
                    time_diff = abs(response_time - distractor_time)
                    if time_diff < closest_distractor_time_diff:
                        closest_distractor_time_diff = time_diff

            # Classify response based on proximity
            if closest_target_time_diff < closest_distractor_time_diff:
                correct_responses.add(response_index)  # Closer to the target stimulus
            elif closest_distractor_time_diff < closest_target_time_diff:
                distractor_responses.add(response_index)  # Closer to the distractor stimulus

    return correct_responses, distractor_responses

# for when the button was pressed below 0.2 seconds, or above 0.9
def invalid_responses(target_stimuli_df, correct_responses, distractor_responses):
    invalid_resp = set()
    # either delayed or random presses
    for df_name, df in target_stimuli_df.items():
        responses = df[df['Stimulus Type'] == 'response']
        responses = responses.iloc[1:]

        # Find responses that are not in correct_responses or distractor_responses
        pending_indices = responses.index.difference(correct_responses | distractor_responses)

        # Filter the DataFrame using these indices
        pending_responses = responses.loc[pending_indices]
        # Add these responses to delayed_responses set
        invalid_resp.update(pending_responses.index)
    return invalid_resp


# calculate performance:
# get total number of target stimuli
# total number of correct responses
# total number of invalid or distractor responses
# calculate percentage

def performance(target_stimuli_df, correct_responses, distractor_responses, invalid_resp):
    total_targets_combined = 0
    distractor_targets_combined = 0
    for df_name, df in target_stimuli_df.items():
        total_targets = df[df['Stimulus Type'] == target_stream]
        distractor_targets = df[(df['Stimulus Type'] != target_stream) & (df['Stimulus Type'] != 'response')]
        total_targets_count = len(total_targets)
        # get total targets of all blocks combined
        distractor_targets_count = len(distractor_targets)

        total_targets_combined += total_targets_count
        distractor_targets_combined += distractor_targets_count


    total_hits = len(correct_responses)
    total_errors = total_targets_combined - total_hits

    distractions = len(distractor_responses) # additional errors
    invalid_responses = len(invalid_resp)
    # misses
    total_misses = total_errors - (distractions + invalid_responses)
    error_count = total_misses + invalid_responses + distractions

# Performance calculations
    hit_rate = (total_hits / total_targets_combined) * 100 if total_targets_combined > 0 else 0
    miss_rate = (total_misses / total_targets_combined) * 100 if total_targets_combined > 0 else 0
    invalid_response_rate = (invalid_responses / total_targets_combined) * 100 if total_targets_combined > 0 else 0
    distractor_response_rate = (distractions / distractor_targets_combined) * 100 if distractor_targets_combined > 0 else 0  # If you want to calculate per distractor

    # Combined error rate for targets (misses + invalid responses)
    error_rate_for_targets = ((total_misses + invalid_responses) / total_targets_combined) * 100 if total_targets_combined > 0 else 0

    # Combined total error rate (includes distractor responses)
    total_error_rate = ((total_misses + invalid_responses + distractions) / total_targets_combined) * 100 if total_targets_combined > 0 else 0
    return hit_rate, miss_rate, invalid_response_rate, distractor_response_rate, error_rate_for_targets, total_error_rate


def plot_performance():
    # plot performance:
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    # Data for plotting
    metrics = ['Hit Rate', 'Miss Rate', 'Invalid Response Rate (delayed/too fast)', 'Distractor Response Rate', 'Error Rate for Targets (misses and late responses)', 'Total Error Rate']
    values = [hit_rate, miss_rate, invalid_response_rate, distractor_response_rate, error_rate_for_targets, total_error_rate]

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['green', 'red', 'orange', 'blue', 'purple', 'gray'])
    plt.ylim(0, 100)
    plt.ylabel('Percentage (%)')
    plt.title(f'Performance Metrics for {sub}, condition {condition}')

    # Adding value labels on each bar
    for i, value in enumerate(values):
        plt.text(i, value + 1, f'{value:.2f}%', ha='center')

    # Show plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    plt.savefig(fig_path / f'{sub}_performance_{condition}')


def plot_rt(response_df):
    # plot combined RTs for plotting their distribution:
    all_rts = []
    for df_name, df in response_df.items():
        all_rts.extend(df['Time Difference'].values)

    all_rts = np.array(all_rts) # convert to numpy array for easier handling

    # choose figure size:
    plt.figure(figsize=(10, 6))

    # plot a histogram with all the values, with 30 bins:
    sns.histplot(all_rts, bins=30, kde=True, color='skyblue', edgecolor='black')

    # Adding labels and title
    plt.xlabel('Reaction Time (seconds)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {sub} Reaction Times Across All Blocks for {condition}')

    plt.savefig(fig_path/f'RTs_{sub}_{condition}')


# all steps:
# 0. get marker files of selected condition:
marker_files, axis = get_marker_files()
# for when the no responses are in the marker file:
marker_files = [file for file in marker_files if '_no_responses' not in file.name]
# 1. get dataframe of all the blocks from selected condition;
dfs = dataframe()
# 2. define the stimuli types in the Df
dfs = define_stimuli(dfs)
# 3. clean empty rows from Df
dfs = clean_rows(dfs)
# 4. get timepoints from samples in Positions:
for df_name, df in dfs.items():
    df['Timepoints'] = df['Position'].astype(float) / 500  # divide the datapoints by the sampling rate of 500
    df['Time Difference'] = df['Timepoints'].diff().fillna(0)
# 5. with this function, we get the blocks that have the condition we chose to focus on
target_stream, target_blocks = get_target_blocks()
# observe if there is a mismatch between target_blocks target number, and the responses of the subject in every block; if yes:
# target_blocks = target_blocks.iloc[:-1].reset_index(drop=True)
# target_blocks['Target Number'] = [5, 4, 2, 2] # adjust each block's target numbers based on responses
# discard empty block, where there are no responses:

# check reaction times of response rows:
target_stimuli_df = target_stimulis(target_blocks, dfs)
response_df, reaction_time_stats_df = responses(target_stimuli_df)
time_start, time_end = avg_rt_stats_combined(reaction_time_stats_df)
correct_responses, distractor_responses = classify_responses(target_stimuli_df, time_start, time_end, target_stream)
invalid_resp = invalid_responses(target_stimuli_df, correct_responses, distractor_responses)
hit_rate, miss_rate, invalid_response_rate, distractor_response_rate, error_rate_for_targets, total_error_rate = performance(target_stimuli_df, correct_responses, distractor_responses, invalid_resp)

plot_performance()

plot_rt(response_df)