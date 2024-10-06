import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json
import copy
from copy import deepcopy
matplotlib.use('Agg')  # This will ensure that the plots are rendered without requiring the main thread to be active.

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
            elif stim_mrk in s2_events.keys():  # same process for s2
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

def get_total_responses(dfs, target_blocks):
    total_resp_dfs = {}
    for df_name, df in dfs.items():
        df_keys_list = list(dfs.keys())  # Convert dfs.keys() to a list to use the index() method
        target_row = target_blocks.loc[target_blocks.index[df_keys_list.index(df_name)]]
        target_number = target_row['Target Number']
        responses = df[(df['Stimulus Type'] == 'response') & (df['Numbers'] == target_number)]
        total_resp_dfs[df_name] = responses
    return total_resp_dfs

# I selected the following time window, based on the reaction time statistics: 0.3-0.9
# this means, after a target stimulus onset, if there was a response 0.3 to 0.9 after its onset, we add this response to the 'correct_responses'
def classify_responses(target_blocks, target_stream, time_start, time_end, dfs):
    # Dictionaries to store DataFrames of classified responses
    target_stimuli_dfs = {}
    distractor_stimuli_dfs = {}
    correct_responses_dfs = {}
    distractor_responses_dfs = {}

    correct_rt_dfs = {}  # Dictionary to store the time differences for correct responses
    distractor_rt_dfs = {}
    for df_name, df in dfs.items():
        # Extract relevant target number for the current block
        # Assuming the order of target_blocks aligns with the order of dfs keys
        df_keys_list = list(dfs.keys())  # Convert dfs.keys() to a list to use the index() method
        target_row = target_blocks.loc[target_blocks.index[df_keys_list.index(df_name)]]
        target_number = target_row['Target Number']

        # filter stimuli and responses dataframes
        target_stimulus = df[(df['Stimulus Type'] == target_stream) & (df['Numbers'] == target_number)].copy()
        distractor_stimulus = df[(df['Stimulus Type'] != target_stream) & (df['Stimulus Type'] != 'response') & (df['Numbers'] == target_number)].copy()
        responses = df[(df['Stimulus Type'] == 'response') & (df['Numbers'] == target_number)]
        # responses = responses.iloc[1:]  # removing first response due to it being first-exposure
        # Add the 'Response' column, defaulting to 0 (missed)
        target_stimulus.loc[:, 'Response'] = 0  # Use .loc to avoid the SettingWithCopyWarning
        distractor_stimulus.loc[:, 'Response'] = 0  # Same here

        correct_indices = []
        distractor_indices = []
        correct_time_diffs = []  # List to store time differences of correct responses
        distractor_time_diffs = []
        for response_index, response_row in responses.iterrows():
            response_time = response_row['Timepoints']  # Time when the response occurred

            # Calculate proximity to the nearest target and distractor stimuli
            closest_target_time_diff = np.inf  # Initialize with a large value
            closest_distractor_time_diff = np.inf  # Initialize with a large value
            # It is used here to initialize variables closest_target_time_diff and closest_distractor_time_diff to a very large number
            # you'll calculate the actual time difference between the response and the stimulus.
            # If the calculated time difference is smaller than the current one, you update the variable with the new, smaller value.
            # Find the closest target stimulus
            for stim_index, stim_row in target_stimulus.iterrows():
                target_time = stim_row['Timepoints']
                time_diff = response_time - target_time
                if time_start <= time_diff <= time_end:  # Within time window
                    closest_target_time_diff = time_diff
                    # Update the 'Response' column to indicate a response was received
                    target_stimulus.loc[stim_index, 'Response'] = 1

            # Find the closest distractor stimulus
            for stim_index, stim_row in distractor_stimulus.iterrows():
                distractor_time = stim_row['Timepoints']
                time_diff = response_time - distractor_time
                if time_start <= time_diff <= time_end:  # Within time window
                    closest_distractor_time_diff = time_diff
                    distractor_stimulus.loc[stim_index, 'Response'] = 2

            # Classify response based on proximity
            if closest_target_time_diff < closest_distractor_time_diff:
                correct_indices.append(response_index)  # Add to the current DataFrame-specific list
                correct_time_diffs.append(closest_target_time_diff)  # Save time difference for correct response
            elif closest_distractor_time_diff < closest_target_time_diff:
                distractor_indices.append(response_index)
                distractor_time_diffs.append(closest_distractor_time_diff)
        target_stimuli_dfs[df_name] = target_stimulus
        distractor_stimuli_dfs[df_name] = distractor_stimulus
        correct_responses_dfs[df_name] = df.loc[correct_indices]
        correct_responses_dfs[df_name]['Response'] = 1
        distractor_responses_dfs[df_name] = df.loc[distractor_indices]
        distractor_responses_dfs[df_name]['Response'] = 2
        correct_rt_dfs[df_name] = pd.Series(correct_time_diffs, name='Time Difference', dtype='float64')  # Store time differences
        distractor_rt_dfs[df_name] = pd.Series(distractor_time_diffs, name='Time Difference', dtype='float64')
    for df_name, df in distractor_responses_dfs.items():
        df.to_csv(df_path / f'{df_name}_distractor_responses.csv')
    for df_name, df in correct_responses_dfs.items():
        df.to_csv(df_path / f'{df_name}_target_responses.csv')
    for df_name, df in target_stimuli_dfs.items():
        df.to_csv(df_path / f'{df_name}_target_stimuli.csv')
    for df_name, df in distractor_stimuli_dfs.items():
        df.to_csv(df_path / f'{df_name}_distractor_stimuli.csv')
    return target_stimuli_dfs, distractor_stimuli_dfs, df_keys_list, correct_responses_dfs, distractor_responses_dfs, correct_rt_dfs, distractor_rt_dfs

# for when the button was pressed below 0.2 seconds, or above 0.9
def invalid_responses(dfs, target_blocks, df_keys_list):
    correct_response_indices_df = {}
    distractor_response_indices_df = {}
    invalid_resp_dfs = {}

    # Step 1: Collect correct and distractor response indices
    for correct_df_name, correct_df in correct_responses_dfs.items():
        correct_response_indices_df[correct_df_name] = correct_df.index.tolist()

    for distractor_df_name, distractor_df in distractor_responses_dfs.items():
        distractor_response_indices_df[distractor_df_name] = distractor_df.index.tolist()

    # Step 2: Identify invalid responses for each DataFrame
    for df_name, df in dfs.items():
        # Find the target number for the given DataFrame

        target_row = target_blocks.loc[target_blocks.index[df_keys_list.index(df_name)]]
        target_number = target_row['Target Number']

        # Get responses specific to the target number in this DataFrame
        responses = df[(df['Stimulus Type'] == 'response') & (df['Numbers'] == target_number)]

        # Combine correct and distractor indices for the current df_name
        correct_indices = set(correct_response_indices_df.get(df_name, []))
        distractor_indices = set(distractor_response_indices_df.get(df_name, []))
        combined_indices = correct_indices | distractor_indices  # Union of correct and distractor indices

        # Collect unmatched indices (responses that are neither in correct nor distractor responses)
        unmatched_indices = [index for index in responses.index if index not in combined_indices]

        # Store these unmatched indices in a dictionary as "invalid responses"
        invalid_resp_dfs[df_name] = df.loc[unmatched_indices]  # Store the entire row corresponding to unmatched indices
    for df_name, df in invalid_resp_dfs.items():
        df['Response'] = 3
        df.to_csv(df_path / f'{df_name}_invalid_responses.csv')
    return correct_response_indices_df, invalid_resp_dfs
# here get the rows with target stimuli that got no response (misses or delays):

def avg_rt_stats_combined(correct_rt_dfs, target): # get average statistics of RTs for all blocks combined:
    avg_stats_df = {}
    # Create a list to store all 'Time Difference' values from each DataFrame
    all_time_differences = []
    for df_name, df in correct_rt_dfs.items():
        if not df.empty:  # Check if the DataFrame is not empty
            # Only process the DataFrame if it contains data
            min_val = df.min()
            max_val = df.max()
            mean_val = df.mean()
            median_val = df.median()

            # Append all 'Time Difference' values to the combined list
            all_time_differences.extend(df.values)

            avg_stats_df[df_name] = pd.DataFrame({
                'Statistic': ['Min', 'Max', 'Mean', 'Median'],
                'Average Value': [min_val, max_val, mean_val, median_val]
            })
            print(f"Min: {min_val}, Max: {max_val}, Mean: {mean_val}, Median: {median_val}")

    # Check if there are valid time differences collected from non-empty DataFrames
    if all_time_differences:
        # Convert all_time_differences to a NumPy array for easier manipulation
        all_time_differences = np.array(all_time_differences)
        # Calculate overall statistics for all blocks combined
        overall_min = np.min(all_time_differences)
        overall_max = np.max(all_time_differences)
        overall_mean = np.mean(all_time_differences)
        overall_median = np.median(all_time_differences)

        # Print overall statistics
        print(
            f"Overall Min for {sub}, condition {condition}: {overall_min}, Max: {overall_max}, Mean: {overall_mean}, Median: {overall_median}")

        # Save the overall statistics to a CSV file
        combined_stats_df = pd.DataFrame({
            'Statistic': ['Min', 'Max', 'Mean', 'Median'],
            'Overall Value': [overall_min, overall_max, overall_mean, overall_median]
        })

        combined_filename = rt_path / f'rt_stats_{sub}_{condition}_combined_{target}.csv'
        combined_stats_df.to_csv(combined_filename, index=False)
        return combined_stats_df
    else:
        print('Nothing to show. No responses detected.')
        return None

def performance(correct_responses_dfs, distractor_responses_dfs, invalid_resp_dfs, df_keys_list, target_blocks, total_resp_dfs):
    total_targets_combined = 0
    distractor_targets_combined = 0
    for df_name, df in dfs.items():
        target_row = target_blocks.loc[target_blocks.index[df_keys_list.index(df_name)]]
        target_number = target_row['Target Number']
        total_targets = df[(df['Stimulus Type'] == target_stream) & (df['Numbers'] == target_number)]
        distractor_targets = df[(df['Stimulus Type'] != target_stream) & (df['Stimulus Type'] != 'response') & (df['Numbers'] == target_number)]
        total_targets_count = len(total_targets)
        # get total targets of all blocks combined
        distractor_targets_count = len(distractor_targets)

        total_targets_combined += total_targets_count
        distractor_targets_combined += distractor_targets_count

        # Step 2: Calculate Total Hits from `correct_responses_dfs`
    total_hits = sum(len(sub_df) for sub_df in correct_responses_dfs.values())
    total_responses = sum(len(sub_df) for sub_df in total_resp_dfs.values())

    # Step 3: Calculate Distractions from `distractor_responses_dfs`
    distractions = sum(len(sub_df) for sub_df in distractor_responses_dfs.values())

    # Step 4: Calculate Invalid Responses
    invalid_responses = sum(len(sub_df) for sub_df in invalid_resp_dfs.values())
    # Step 5: Calculate Total Errors (misses and invalid responses)
    total_misses = total_targets_combined - total_hits  # Targets that were missed

    # Step 6: Calculate Overall Error Count
    error_count = total_misses + invalid_responses + distractions
# Performance calculations
    hit_rate = (total_hits / total_targets_combined) * 100 if total_targets_combined > 0 else 0
    miss_rate = (total_misses / total_targets_combined) * 100 if total_targets_combined > 0 else 0
    invalid_response_rate = (invalid_responses / total_responses) * 100 if total_responses > 0 else 0
    distractor_response_rate = (distractions / distractor_targets_combined) * 100 if distractor_targets_combined > 0 else 0  # If you want to calculate per distractor

    print(f'Hits: {hit_rate}; Misses: {miss_rate}; Invalid Responses: {invalid_response_rate}; Distractions: {distractor_response_rate}')
    return hit_rate, miss_rate, invalid_response_rate, distractor_response_rate


def plot_performance():
    # plot performance:
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    # Data for plotting
    metrics = ['Hit Rate', 'Miss Rate', 'Invalid Response Rate (unclassified; out of total response sum)', 'Distractor Response Rate']
    values = [hit_rate, miss_rate, invalid_response_rate, distractor_response_rate]

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['green', 'red', 'yellow', 'purple'])
    plt.ylim(0, 100)
    plt.ylabel('Percentage (%)')
    plt.title(f'Performance Metrics for {sub}, condition {condition}')

    # Adding value labels on each bar
    for i, value in enumerate(values):
        plt.text(i, value + 1, f'{value:.2f}%', ha='center')

    # Show plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(fig_path / f'{sub}_performance_{condition}.png')
    plt.close()


def plot_rt(correct_rt_dfs, combined_stats_df, target):
    # plot combined RTs for plotting their distribution:
    all_rts = []
    for df_name, df in correct_rt_dfs.items():
        all_rts.extend(df.values)
    # Convert all_rts to a NumPy array for easier handling
    all_rts = np.array(all_rts)
    # Ensure there is data to plot
    if len(all_rts) < 5:
        print(f"No reaction times to plot for {target}.")
        return  # Exit the function if there's no data
        # Ensure combined_stats_df is valid
    if combined_stats_df is None or combined_stats_df.empty:
        print(f"No statistics available to plot for {target}.")
        return

    # Check if the required statistics are present in the DataFrame
    try:
        min_val = combined_stats_df.loc[combined_stats_df['Statistic'] == 'Min', 'Overall Value'].values[0]
        max_val = combined_stats_df.loc[combined_stats_df['Statistic'] == 'Max', 'Overall Value'].values[0]
    except IndexError:
        print(f"Statistics missing for {target}.")
        return

    # choose figure size:
    plt.figure(figsize=(10, 6))

    # plot a histogram with all the values, with dynamically chosen bins:
    sns.histplot(all_rts, bins=int(len(all_rts) / 5), kde=True, color='skyblue', edgecolor='black')
    # Add mean and median lines
    mean_rt = np.mean(all_rts)
    median_rt = np.median(all_rts)
    plt.axvline(mean_rt, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_rt:.3f}s')
    plt.axvline(median_rt, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_rt:.3f}s')

    # Adding labels and title
    plt.xlabel('Reaction Time (seconds)')
    plt.ylabel('Frequency')
    plt.title(f'Distribution of {sub} Reaction Times Across All Blocks for {condition}, {target}')

    # Extract statistics from combined_stats_df to include in legend
    min_val = combined_stats_df.loc[combined_stats_df['Statistic'] == 'Min', 'Overall Value'].values[0]
    max_val = combined_stats_df.loc[combined_stats_df['Statistic'] == 'Max', 'Overall Value'].values[0]

    # Add overall statistics as text in the plot
    textstr = '\n'.join((
        f'Min: {min_val:.3f}s',
        f'Max: {max_val:.3f}s'
    ))

    plt.gca().text(0.98, 0.98, textstr, fontsize=10, verticalalignment='top', horizontalalignment='right',
                   transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.6))

    # Show the legend for mean and median in the center-top
    plt.legend(loc='upper left')

    plt.savefig(fig_path/f'RTs_{sub}_{condition}_{df_name[-1]}_{target}.png')
    plt.close()

def plot_stimuli_vs_responses(dfs, df_keys_list, target_blocks, target_stream, target):
    target_vs_response = {}
    for df_name, df in dfs.items():
        target_block = target_blocks.loc[target_blocks.index[df_keys_list.index(df_name)]]
        target_number = target_block['Target Number']
        responses = df.loc[(df['Stimulus Type'] == 'response') & (df['Numbers'] == target_number)]
        target_stim = df.loc[(df['Stimulus Type'] == target_stream) & (df['Numbers'] == target_number)]
        combined_events = pd.concat([target_stim, responses])
        combined_events = combined_events.sort_values(by='Timepoints').reset_index(drop=True)
        combined_events['Event Number'] = combined_events.index + 1
        target_vs_response[df_name] = combined_events

    for df_name, data in target_vs_response.items():
        target_stim_timepoints = data[data['Stimulus Type'] == target_stream]['Timepoints']
        response_timepoints = data[data['Stimulus Type'] == 'response']['Timepoints']

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plot target stimuli
        ax.plot(target_stim_timepoints.index, target_stim_timepoints, color='green', marker='x', linestyle='-',
                label=f'{target} Stimuli', markersize=8)

        # Plot responses
        ax.plot(response_timepoints.index, response_timepoints, color='red', marker='o', linestyle='-',
                label='Responses', markersize=8)
        # Customize the plot
        ax.set_xlabel('Event Number (Ordered by Time)')
        ax.set_ylabel('Time (seconds)')
        ax.set_title(f'{target} Stimuli vs Responses for condition {condition}_{df_name[-1]}')
        ax.legend(loc='upper right')

        # Save the plot
        plt.tight_layout()
        plt.savefig(events_vs_responses / f'{target}_vs_responses_{sub}_condition_{condition}_{df_name[-1]}.png')
        plt.close()
    return target_vs_response



if __name__ == "__main__":
    default_dir = Path.cwd()
    sub = input('Subject number: ')

    # specify condition to focus on:
    condition = input('Choose a condition (a1, a2, e1 or e2): ')


    sub_dir = default_dir / 'data' / 'eeg' / 'raw' / f'{sub}'
    performance_events = default_dir / 'data' / 'misc' / 'performance_events.json'
    fig_path = default_dir / 'data' / 'performance' / f'{sub}'
    df_path = fig_path / 'tables'
    os.makedirs(df_path, exist_ok=True)
    events_vs_responses = fig_path / 'events_vs_responses'
    rt_path = fig_path / 'RTs'

    if not os.path.exists(rt_path):
        os.makedirs(rt_path)
    if not os.path.exists(events_vs_responses):
        os.makedirs(events_vs_responses)

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
    csv = pd.read_csv(csv_path)  # delimiter=';', header=None

    # all steps:
    # 0. get marker files of selected condition:
    marker_files, axis = get_marker_files()
    # for when the no responses are in the marker file:
    to_drop = [index for index, file in enumerate(marker_files) if 'no_responses' in file.name]
    marker_files = [file for file in marker_files if 'no_responses' not in file.name]
    # 1. get dataframe of all the blocks from selected condition;
    dfs = dataframe()
    # 2. define the stimuli types in the Df
    dfs = define_stimuli(dfs)
    # 3. clean empty rows from Df
    dfs = clean_rows(dfs)
    # 4. get timepoints from samples in Positions:
    for df_name, df in dfs.items():
        df['Timepoints'] = df['Position'].astype(float) / 500  # divide the datapoints by the sampling rate of 500
    # 5. with this function, we get the blocks that have the condition we chose to focus on
    target_stream, target_blocks = get_target_blocks()
    # define distractor stream:
    if target_stream == 's1':
        distractor_stream = 's2'
    elif target_stream == 's2':
        distractor_stream = 's1'
    # for when a block's responses were not recorded
    target_blocks = target_blocks.drop(index=to_drop)
    # observe if there is a mismatch between target_blocks target number, and the responses of the subject in every block; if yes:
    # target_blocks['Target Number'] = [5, 5, 8, 9, 3] # adjust each block's target numbers based on responses
    # discard empty block, where there are no responses:

    # check reaction times of response rows:
    for df_name, df in dfs.items():
        df = df.copy()
        df['Time Difference'] = df['Timepoints'].diff().fillna(0)
        dfs[df_name] = df

    # define the time window:
    time_start = 0.2
    time_end = 0.9

    total_resp_dfs = get_total_responses(dfs, target_blocks)
    target_stimuli_dfs, distractor_stimuli_dfs, df_keys_list, correct_responses_dfs, distractor_responses_dfs, correct_rt_dfs, distractor_rt_dfs= classify_responses(target_blocks, target_stream, time_start, time_end, dfs)
    correct_response_indices_df, invalid_resp_dfs = invalid_responses(dfs, target_blocks, df_keys_list)
    hit_rate, miss_rate, invalid_response_rate, distractor_response_rate = performance(correct_responses_dfs, distractor_responses_dfs, invalid_resp_dfs, df_keys_list, target_blocks, total_resp_dfs)
    plot_performance()

    combined_stats_df = avg_rt_stats_combined(correct_rt_dfs, target='target')
    combined_stats_df = plot_rt(correct_rt_dfs, combined_stats_df, target='target')  # focusing on correct responses only.

    # for distractor, if not empty:
    if any(not df.empty for df in distractor_rt_dfs.values()):
        # Compute statistics for distractor RTs
        combined_stats_df_distractor = avg_rt_stats_combined(distractor_rt_dfs, target='distractor')

        if combined_stats_df_distractor is not None and not combined_stats_df_distractor.empty:
            # Plot RTs if stats are successfully computed
            combined_stats_df_distractor = plot_rt(distractor_rt_dfs, combined_stats_df_distractor, target='distractor')
    else:
        print("No valid distractor RT data available.")
    # to see the distribution of the stimuli and response events over time:
    target_vs_response = plot_stimuli_vs_responses(dfs, df_keys_list, target_blocks, target_stream, target='target')
    distractor_vs_response = plot_stimuli_vs_responses(dfs, df_keys_list, target_blocks, distractor_stream, target='distractor')
