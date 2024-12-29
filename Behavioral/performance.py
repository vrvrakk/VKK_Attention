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
    # Extended time window
    early_invalid_time = 0.0
    late_invalid_time = 1.2

    # Dictionaries to store DataFrames of classified responses
    target_stimuli_dfs = {}
    distractor_stimuli_dfs = {}
    valid_target_responses_dfs = {}
    invalid_target_responses_dfs = {}
    valid_distractor_responses_dfs = {}
    invalid_distractor_responses_dfs = {}
    missed_target_stimuli_dfs = {}
    non_target_invalid_responses_dfs = {}

    for df_name, df in dfs.items():
        df_keys_list = list(dfs.keys())
        target_row = target_blocks.loc[target_blocks.index[df_keys_list.index(df_name)]]
        target_number = target_row['Target Number']

        target_stimulus = df[(df['Stimulus Type'] == target_stream) & (df['Numbers'] == target_number)].copy()
        distractor_stimulus = df[(df['Stimulus Type'] != target_stream) & (df['Stimulus Type'] != 'response') & (df['Numbers'] == target_number)].copy()
        responses = df[(df['Stimulus Type'] == 'response') & (df['Numbers'] == target_number)]

        # Initialize Response columns in stimuli DataFrames
        target_stimulus['Response'] = 0
        distractor_stimulus['Response'] = 0

        # Track indices for each response category
        valid_target_indices = []
        invalid_target_indices = []
        valid_distractor_indices = []
        invalid_distractor_indices = []
        non_target_invalid_indices = []

        # Classify each response
        for response_index, response_row in responses.iterrows():
            response_time = response_row['Timepoints']
            assigned = False

            # Check for target stimuli
            for stim_index, stim_row in target_stimulus.iterrows():
                target_time = stim_row['Timepoints']
                time_diff = response_time - target_time


                if time_start <= time_diff <= time_end and target_stimulus.at[stim_index, 'Response'] == 0:
                    valid_target_indices.append(response_index)
                    target_stimulus.at[stim_index, 'Response'] = 1
                    assigned = True
                    break
                elif early_invalid_time <= time_diff < time_start or time_diff > time_end and time_diff <= late_invalid_time:
                    if target_stimulus.at[stim_index, 'Response'] == 0:
                        invalid_target_indices.append(response_index)
                        assigned = True
                        break

            # Check for distractor stimuli if not assigned
            if not assigned:
                for stim_index, stim_row in distractor_stimulus.iterrows():
                    distractor_time = stim_row['Timepoints']
                    time_diff = response_time - distractor_time

                    if time_start <= time_diff <= time_end and distractor_stimulus.at[stim_index, 'Response'] == 0:
                        valid_distractor_indices.append(response_index)
                        distractor_stimulus.at[stim_index, 'Response'] = 2
                        assigned = True
                        break
                    elif early_invalid_time <= time_diff < time_start or time_diff > time_end and time_diff <= late_invalid_time:
                        if distractor_stimulus.at[stim_index, 'Response'] == 0:
                            invalid_distractor_indices.append(response_index)
                            assigned = True
                            break

            # If no valid target or distractor stimulus is found, it's a non-target invalid response
            if not assigned:
                non_target_invalid_indices.append(response_index)

        # Mark missed target stimuli
        missed_target_indices = target_stimulus[target_stimulus['Response'] == 0].index.tolist()

        # Create DataFrames for each category
        valid_target_responses_dfs[df_name] = df.loc[valid_target_indices]
        invalid_target_responses_dfs[df_name] = df.loc[invalid_target_indices]
        valid_distractor_responses_dfs[df_name] = df.loc[valid_distractor_indices]
        invalid_distractor_responses_dfs[df_name] = df.loc[invalid_distractor_indices]
        missed_target_stimuli_dfs[df_name] = target_stimulus.loc[missed_target_indices]
        non_target_invalid_responses_dfs[df_name] = df.loc[non_target_invalid_indices]

        target_stimuli_dfs[df_name] = target_stimulus
        distractor_stimuli_dfs[df_name] = distractor_stimulus

    # Save CSVs for each category
    for df_name, df in valid_target_responses_dfs.items():
        df.to_csv(df_path / f'{df_name}_valid_target_responses.csv')
    for df_name, df in invalid_target_responses_dfs.items():
        df.to_csv(df_path / f'{df_name}_invalid_target_responses.csv')
    for df_name, df in valid_distractor_responses_dfs.items():
        df.to_csv(df_path / f'{df_name}_valid_distractor_responses.csv')
    for df_name, df in invalid_distractor_responses_dfs.items():
        df.to_csv(df_path / f'{df_name}_invalid_distractor_responses.csv')
    for df_name, df in missed_target_stimuli_dfs.items():
        df.to_csv(df_path / f'{df_name}_missed_target_stimuli.csv')
    for df_name, df in non_target_invalid_responses_dfs.items():
        df.to_csv(df_path / f'{df_name}_non_target_invalid_responses.csv')
    for df_name, df in target_stimuli_dfs.items():
        df.to_csv(df_path / f'{df_name}_all_target_stimuli.csv')
    for df_name, df in distractor_stimuli_dfs.items():
        df.to_csv(df_path / f'{df_name}_all_distractor_stimuli.csv')

    return (target_stimuli_dfs, distractor_stimuli_dfs, df_keys_list, valid_target_responses_dfs,
            invalid_target_responses_dfs, valid_distractor_responses_dfs, invalid_distractor_responses_dfs,
            missed_target_stimuli_dfs, non_target_invalid_responses_dfs)


def avg_rt_stats_combined(valid_target_responses_dfs, sub, condition, rt_path):
    # Create a list to store all 'Time Difference' values across all blocks
    all_time_differences = []

    for df_name, df in valid_target_responses_dfs.items():
        if not df.empty and 'Time Difference' in df.columns:  # Ensure non-empty and contains 'Time Difference'
            # Append all 'Time Difference' values to the combined list
            all_time_differences.extend(df['Time Difference'].dropna().values)

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
            f"Overall Min for {sub}, condition {condition}: {overall_min}, Max: {overall_max}, Mean: {overall_mean}, Median: {overall_median}"
        )

        # Save the overall statistics to a CSV file
        combined_stats_df = pd.DataFrame({
            'Statistic': ['Min', 'Max', 'Mean', 'Median'],
            'Overall Value': [overall_min, overall_max, overall_mean, overall_median]
        })

        combined_filename = rt_path / f'rt_stats_{sub}_{condition}_combined_target.csv'
        combined_stats_df.to_csv(combined_filename, index=False)
        return combined_stats_df
    else:
        print('Nothing to show. No valid responses detected.')
        return None


def performance(target_stimuli_dfs, distractor_stimuli_dfs, valid_target_responses_dfs, invalid_target_responses_dfs, valid_distractor_responses_dfs,
    invalid_distractor_responses_dfs, missed_target_stimuli_dfs, non_target_invalid_responses_dfs):
    # Initialize totals
    total_target_stimuli = sum(len(df) for df in target_stimuli_dfs.values())
    total_distractor_stimuli = sum(len(df) for df in distractor_stimuli_dfs.values())
    total_stimuli = total_target_stimuli + total_distractor_stimuli

    # Calculate valid responses
    total_valid_target_responses = sum(len(df) for df in valid_target_responses_dfs.values())
    total_valid_distractor_responses = sum(len(df) for df in valid_distractor_responses_dfs.values())

    # Calculate invalid responses
    total_invalid_target_responses = sum(len(df) for df in invalid_target_responses_dfs.values())
    total_invalid_distractor_responses = sum(len(df) for df in invalid_distractor_responses_dfs.values())
    total_non_target_invalid_responses = sum(len(df) for df in non_target_invalid_responses_dfs.values())
    total_invalid_responses = (
        total_invalid_target_responses + total_invalid_distractor_responses + total_non_target_invalid_responses
    )

    # Calculate missed responses
    total_missed_target_stimuli = sum(len(df) for df in missed_target_stimuli_dfs.values())

    # Performance calculations
    target_hit_rate = (total_valid_target_responses / total_target_stimuli) * 100 if total_target_stimuli > 0 else 0
    distractor_hit_rate = (total_valid_distractor_responses / total_distractor_stimuli) * 100 if total_distractor_stimuli > 0 else 0
    target_miss_rate = (total_missed_target_stimuli / total_target_stimuli) * 100 if total_target_stimuli > 0 else 0
    target_invalid_response_rate = (total_invalid_target_responses / total_target_stimuli) * 100 if total_target_stimuli > 0 else 0
    overall_invalid_response_rate = (total_invalid_responses / total_stimuli) * 100 if total_stimuli > 0 else 0

    # Display results
    print(f"Target Hit Rate: {target_hit_rate:.2f}%")
    print(f"Distractor Hit Rate: {distractor_hit_rate:.2f}%")
    print(f"Target Miss Rate: {target_miss_rate:.2f}%")
    print(f"Target Invalid Response Rate: {target_invalid_response_rate:.2f}%")
    print(f"Overall Invalid Response Rate: {overall_invalid_response_rate:.2f}%")

    return {
        "target_hit_rate": target_hit_rate,
        "distractor_hit_rate": distractor_hit_rate,
        "target_miss_rate": target_miss_rate,
        "target_invalid_response_rate": target_invalid_response_rate,
        "overall_invalid_response_rate": overall_invalid_response_rate
    }



def plot_performance(performance_results, sub, condition, fig_path):
    # Ensure the output directory exists
    if not os.path.isdir(fig_path):
        os.makedirs(fig_path)

    # Data for plotting
    metrics = [
        'Target Hit Rate',
        'Distractor Hit Rate',
        'Target Miss Rate',
        'Target Invalid Response Rate',
        'Overall Invalid Response Rate'
    ]
    values = [
        performance_results['target_hit_rate'],
        performance_results['distractor_hit_rate'],
        performance_results['target_miss_rate'],
        performance_results['target_invalid_response_rate'],
        performance_results['overall_invalid_response_rate']
    ]

    # Plotting the bar chart
    plt.figure(figsize=(10, 6))
    plt.bar(metrics, values, color=['green', 'blue', 'red', 'orange', 'purple'])
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


def plot_rt(valid_target_responses_dfs, combined_stats_df, sub, condition, fig_path, target=''):
    # Collect all RTs from valid target responses
    all_rts = []
    for df_name, df in valid_target_responses_dfs.items():
        if 'Time Difference' in df.columns:
            all_rts.extend(df['Time Difference'].dropna().values)

    # Convert all_rts to a NumPy array for easier handling
    all_rts = np.array(all_rts)

    # Ensure there is data to plot
    if len(all_rts) < 5:
        print(f"No reaction times to plot for target responses.")
        return

    # Ensure combined_stats_df is valid
    if combined_stats_df is None or combined_stats_df.empty:
        print(f"No statistics available to plot for target responses.")
        return

    # Extract overall min and max values for display
    try:
        min_val = combined_stats_df.loc[combined_stats_df['Statistic'] == 'Min', 'Overall Value'].values[0]
        max_val = combined_stats_df.loc[combined_stats_df['Statistic'] == 'Max', 'Overall Value'].values[0]
    except IndexError:
        print(f"Statistics missing for {target} responses.")
        return

    # Plotting the RT histogram
    plt.figure(figsize=(10, 6))
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

    # Add overall statistics as text in the plot
    textstr = '\n'.join((
        f'Min: {min_val:.3f}s',
        f'Max: {max_val:.3f}s'
    ))
    plt.gca().text(0.98, 0.98, textstr, fontsize=10, verticalalignment='top', horizontalalignment='right',
                   transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.6))

    # Show the legend for mean and median in the center-top
    plt.legend(loc='upper left')

    plt.savefig(fig_path / f'RTs_{sub}_{condition}_{target}.png')
    plt.close()


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
        df['Timepoints'] = df['Position'].astype(float) / 1000  # divide the datapoints by the sampling rate of 100
    # 5. with this function, we get the blocks that have the condition we chose to focus on
    target_stream, target_blocks = get_target_blocks()
    # define distractor stream:
    if target_stream == 's1':
        distractor_stream = 's2'
    elif target_stream == 's2':
        distractor_stream = 's1'
    # for when a block's responses were not recorded
    target_blocks = target_blocks.drop(index=to_drop)

    # check reaction times of response rows:
    for df_name, df in dfs.items():
        df = df.copy()
        df['Time Difference'] = df['Timepoints'].diff().fillna(0)
        dfs[df_name] = df

    # define the time window:
    time_start = 0.0
    time_end = 1.2

    (target_stimuli_dfs, distractor_stimuli_dfs, df_keys_list, valid_target_responses_dfs,
     invalid_target_responses_dfs, valid_distractor_responses_dfs, invalid_distractor_responses_dfs,
     missed_target_stimuli_dfs, non_target_invalid_responses_dfs) = classify_responses(target_blocks, target_stream, time_start, time_end, dfs)

    performance_results = performance(target_stimuli_dfs, distractor_stimuli_dfs, valid_target_responses_dfs,
        invalid_target_responses_dfs, valid_distractor_responses_dfs,
        invalid_distractor_responses_dfs, missed_target_stimuli_dfs,
        non_target_invalid_responses_dfs)

    # Calculate and plot RT statistics for target responses if valid data is present
    if any(not df.empty and 'Time Difference' in df.columns for df in valid_target_responses_dfs.values()):
        combined_stats_df_target = avg_rt_stats_combined(valid_target_responses_dfs, sub, condition, rt_path)

        if combined_stats_df_target is not None and not combined_stats_df_target.empty:
            # Plot RTs if stats are successfully computed
            plot_rt(valid_target_responses_dfs, combined_stats_df_target, sub, condition, fig_path, target='target')
    else:
        print("No valid target RT data available.")

    # Calculate and plot RT statistics for distractor responses if valid data is present
    if any(not df.empty and 'Time Difference' in df.columns for df in valid_distractor_responses_dfs.values()):
        combined_stats_df_distractor = avg_rt_stats_combined(valid_distractor_responses_dfs, sub,
                                                             f"{condition}_distractor", rt_path)

        if combined_stats_df_distractor is not None and not combined_stats_df_distractor.empty:
            # Plot RTs if stats are successfully computed
            plot_rt(valid_distractor_responses_dfs, combined_stats_df_distractor, sub, f"{condition}_distractor",
                    fig_path, target='distractor')
    else:
        print("No valid distractor RT data available.")

    # to see the distribution of the stimuli and response events over time:
    # Plot the performance metrics
    plot_performance(performance_results, sub, condition, fig_path)
