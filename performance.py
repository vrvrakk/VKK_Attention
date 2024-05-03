
import pandas as pd
from get_streams_and_stream_params import tlo1
import matplotlib.pyplot as plt
import matplotlib
import warnings
import numpy as np
from performance_dfs import select_marker_files, marker_files_dfs, stimulus_types, none_vals, convert_to_numeric, convert_stimuli
# Suppress all warnings
warnings.filterwarnings("ignore")

matplotlib.use('TkAgg')
tlo1 = tlo1 /1000

# copy updated dfs for processing of responses:
def copy_dfs(dfs):
    dfs_copy = {}
    for df_name, df in dfs.items():
        dfs_copy[df_name] = df.copy()
        dfs_copy[df_name] = df.assign(Reaction=0, Reaction_Time=0)
    return dfs_copy


def target_responses(dfs_copy):
    target_controlled_rows = set()
    target_responses_dict = {}
    target_responses_count = {}
    for df_name, df in dfs_copy.items():
        target_responses = []
        errors = []

        stim_indices = df['Stimulus Type'].index
        stim_types = df['Stimulus Type'].loc[stim_indices]

        # get target responses:
        for (stim_index, stimulus) in enumerate(stim_types):

            if stimulus == 'target':
                target_number = df.at[stim_index, 'Numbers']  # get corresponding number of target

                target_time = df.at[stim_index, 'Position']  # get corresponding time
                target_label = df.at[stim_index, 'Stimulus Type']

                # define time window:
                window_start = target_time
                window_end = target_time + tlo1 + 0.25
                window_data = df.loc[(df['Position'] >= window_start) & (df['Position'] <= window_end)]
                if 'response' in window_data['Stimulus Type'].values:
                    response_indices_within_window = window_data[window_data['Stimulus Type'] == 'response'].index
                    for response_index in response_indices_within_window:
                        response_number = window_data['Numbers'].loc[response_index]
                        response_time = window_data['Position'].loc[response_index]
                        response_label = window_data['Stimulus Type'].loc[response_index]
                        response_type = window_data['Reaction'].loc[response_index]
                        if response_number == target_number:
                            target_responses.append((stim_index, target_label, target_number, target_time,
                                                 response_index, response_label, response_number, response_time))
                            target_controlled_rows.add((df_name, stim_index, response_index, target_time))
            target_responses_df = pd.DataFrame(target_responses)
            target_responses_dict[df_name] = target_responses_df
            target_responses_count[df_name] = len(target_responses)
    return target_responses_dict, target_responses_count, target_controlled_rows


# change target responses columns names
def target_responses_columns(target_responses_dict):
    for df_name, df in target_responses_dict.items():
        if not df.empty:
            df.columns = ['Target Index', 'Target', 'Target Number', 'Target Position', 'Response Index', 'Response',
                          'Response Number', 'Response Position']
    return target_responses_dict


# apply Reaction vals
def target_reaction_values(dfs_copy, target_responses_dict, target_controlled_rows):
    for df_name, df in dfs_copy.items():
        if df_name in target_responses_dict:
            for name, stim_index, response_index, target_time in target_controlled_rows:
                if name == df_name:
                    df.at[stim_index, 'Reaction'] = 1
                    df.at[response_index, 'Reaction'] = 1
                    df.at[response_index, 'Reaction_Time'] = target_time
    return dfs_copy


def distractor_responses(dfs_copy):
    distractor_controlled_rows = set()
    distractor_responses_dict = {}
    distractor_responses_count = {}
    for df_name, df in dfs_copy.items():
        distractor_responses = []
        stim_indices = df['Stimulus Type'].index
        stim_types = df['Stimulus Type'].loc[stim_indices]
        for (stim_index, stimulus) in enumerate(stim_types):
            if stimulus == 'distractor':
                distractor_number = df.at[stim_index, 'Numbers']  # get corresponding number of s1
                distractor_time = df.at[stim_index, 'Position']  # get corresponding s1 time
                distractor_label = df.at[stim_index, 'Stimulus Type']

                window_start = distractor_time
                window_end = distractor_time + tlo1 + 0.250
                window_data = df.loc[(df['Position'] >= window_start) & (df['Position'] <= window_end)]
                if 'response' in window_data['Stimulus Type'].values:
                    response_indices_within_window = window_data[window_data['Stimulus Type'] == 'response'].index
                    for response_index in response_indices_within_window:
                        response_number = window_data['Numbers'].loc[response_index]
                        response_time = window_data['Position'].loc[response_index]
                        response_label = window_data['Stimulus Type'].loc[response_index]
                        response_type = window_data['Reaction'].loc[response_index]
                        if response_number == distractor_number and response_type == 0:
                            distractor_responses.append((stim_index, distractor_label, distractor_number, distractor_time,
                                                 response_index, response_label, response_number, response_time))
                            distractor_controlled_rows.add((df_name, stim_index, response_index, distractor_time))
        distractor_responses_df = pd.DataFrame(distractor_responses)
        distractor_responses_dict[df_name] = distractor_responses_df
        distractor_responses_count[df_name] = len(distractor_responses)
    return distractor_responses_dict, distractor_responses_count, distractor_controlled_rows


def distractor_responses_columns(distractor_responses_dict):
    for df_name, df in distractor_responses_dict.items():
        if not df.empty:
            df.columns = ['Target Index', 'Target', 'Target Number', 'Target Position', 'Response Index', 'Response',
                          'Response Number', 'Response Position']
    return distractor_responses_dict


def distractor_reaction_values(dfs_copy, distractor_responses_dict, distractor_controlled_rows):
    for df_name, df in dfs_copy.items():
        if df_name in distractor_responses_dict:
            for name, stim_index, response_index, distractor_time in distractor_controlled_rows:
                if name == df_name:
                    df.at[stim_index, 'Reaction'] = 2
                    df.at[response_index, 'Reaction'] = 2
                    df.at[response_index, 'Reaction_Time'] = distractor_time
    return dfs_copy


def total_responses(dfs_copy):
    total_responses_df = {}
    total_responses_count = {}
    for df_name, df in dfs_copy.items():
        responses = df[df['Stimulus Type'] == 'response']
        total_responses_df[df_name] = responses
        response_count = len(responses)
        total_responses_count[df_name] = response_count
    return total_responses_df, total_responses_count

# responses to target and distractor have been identified
# responses that do not have a Reaction that is '1', should be included in the false_responses dicts


def false_responses(total_responses_df):
    false_responses_dict = {}

    # Loop over each DataFrame to process the false responses
    for df_name, df in total_responses_df.items():
        # Filter responses that do not have Reaction equal to '1'
        false_responses = df[df['Reaction'] != 1]

        # Add the false responses to the dictionary
        false_responses_dict[df_name] = false_responses
    return false_responses_dict

# plot performance for this condition:
# title should be the df_name
# y axis responses in percentage
# x axis with 'correct hits' and 'errors'
# 2 bar plots: 1 for correct hits and 1 for errors
# all the target_responses summed from all the sub dfs, same for false_responses
def plot_performance(target_responses_dict, false_responses_dict):
    total_correct_hits = sum(len(df) for df in target_responses_dict.values())
    total_errors = sum(len(df) for df in false_responses_dict.values())

    total_responses = total_correct_hits + total_errors
    percentage_correct_hits = (total_correct_hits / total_responses) * 100
    percentage_errors = (total_errors / total_responses) * 100

    plt.figure(figsize=(8, 6))
    plt.bar(['Correct Hits', 'Errors'], [percentage_correct_hits, percentage_errors], color=['green', 'red'])
    plt.title(f'Performance')
    plt.xlabel('Response Type')
    plt.ylabel('Percentage of Responses')
    plt.ylim(0, 100)  # Set y-axis limits to ensure percentages are between 0 and 100
    plt.text(0, percentage_correct_hits + 2, f'{percentage_correct_hits:.2f}%', ha='center', color='black')
    plt.text(1, percentage_errors + 2, f'{percentage_errors:.2f}%', ha='center', color='black')
    return plt


def plot_figure(plt):
    plt.show()


def reaction_times(target_responses_dict):
    rts = []
    for df_name, df in target_responses_dict.items():
        target_time = df['Target Position']
        print(target_time)
        response_time = df['Response Position']
        rt = response_time - target_time
        rts.append(rt)
    mean_rt = np.mean(rts)
    max_rt = np.max(rts)
    min_rt = np.min(rts)
    return rts, mean_rt, max_rt, min_rt


def hist_rt(rts):
    total_rts = len(rts[0])
    percentage = np.linspace(0, 100, total_rts)
    hist = plt.figure(figsize=(8, 6))
    hist = plt.scatter(rts, percentage, color='blue', alpha=0.5, label='Scatter Plot')
    hist = plt.hist(rts, bins=22, color='red', alpha=0.5, label='Histogram')
    hist = plt.xlabel('Reaction Time (s)')
    hist = plt.ylabel('Frequency / Percentage')
    hist = plt.title('Reaction Time Distribution')

    # Add a legend
    hist = plt.legend()

    # Show the plot
    hist = plt.grid(True)
    hist = plt.show()
    return hist

'''# to calculate bins:
q1 = np.percentile(rts[0], 25)
q3 = np.percentile(rts[0], 75)
iqr = q3 - q1
# Calculate the bin width using the Freedman-Diaconis rule
n = len(rts[0])
bin_width = 2 * iqr / (n ** (1/3))
# Calculate the number of bins using the bin width
bin_count = int((max(rts[0]) - min(rts[0])) / bin_width)
print("Number of bins (Freedman-Diaconis rule):", bin_count)
Number of bins (Freedman-Diaconis rule): 22
'''


def run_performance():
    marker_files = select_marker_files()
    dfs = marker_files_dfs(marker_files)
    dfs = stimulus_types(dfs)
    dfs = none_vals(dfs)
    dfs = convert_to_numeric(dfs)
    dfs = convert_stimuli(marker_files, dfs)

    dfs_copy = copy_dfs(dfs)

    target_responses_dict, target_responses_count, target_controlled_rows = target_responses(dfs_copy)
    target_responses_dict = target_responses_columns(target_responses_dict)
    dfs_copy = target_reaction_values(dfs_copy, target_responses_dict, target_controlled_rows)

    distractor_responses_dict, distractor_responses_count, distractor_controlled_rows = distractor_responses(dfs_copy)
    distractor_responses_dict = distractor_responses_columns(distractor_responses_dict)
    dfs_copy = distractor_reaction_values(dfs_copy, distractor_responses_dict, distractor_controlled_rows)

    total_responses_df, total_responses_count = total_responses(dfs_copy)

    false_responses_dict = false_responses(total_responses_df)
    rts, mean_rt, min_rt, max_rt =reaction_times(target_responses_dict)
    hist = hist_rt(rts)

    plt = plot_performance(target_responses_dict, false_responses_dict)
    plot_figure(plt)

    return hist, plt, false_responses_dict, total_responses_df, total_responses_count, distractor_responses_count, distractor_responses_dict,\
           dfs_copy, target_responses_dict, target_responses_count



hist, plt, false_responses_dict, total_responses_df, total_responses_count, distractor_responses_count, distractor_responses_dict, \
dfs_copy, target_responses_dict, target_responses_count = run_performance()



