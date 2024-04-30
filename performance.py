import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import warnings

# Suppress all warnings
warnings.filterwarnings("ignore")

matplotlib.use('TkAgg')

data_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/raw')



s1 = {1: 'S  1', 2: 'S  2', 3: 'S  3', 4: 'S  4', 5: 'S  5', 6: 'S  6', 8: 'S  8', 9: 'S  9'}  # stimulus 1 markers
s2 = {1: 'S 65', 2: 'S 66', 3: 'S 67', 4: 'S 68', 5: 'S 69', 6: 'S 70', 8: 'S 72', 9: 'S 73'}  # stimulus 2 markers
response = {1: 'S129', 2: 'S130', 3: 'S131', 4: 'S132', 5: 'S133', 6: 'S134', 8: 'S136', 9: 'S137'}  # response markers

# select .vmrk files:
marker_files = []
for files in os.listdir(data_path):
    if files.endswith('azimuth.vmrk'):
        marker_files.append(data_path / files)



# save marker files as pandas dataframe:
columns = ['Stimulus Stream', 'Position', 'Time Difference']
dfs = {}
for index, file_info in enumerate(marker_files):
    file_name = file_info.stem
    df = pd.read_csv(file_info, delimiter='\t', header=None)  # t for tabs
    df_name = f'df_{file_name}'
    df = df.iloc[10:]  # delete first 10 rows
    df = df.reset_index(drop=True, inplace=False)
    df = df[0].str.split(',', expand=True).applymap(lambda x: None if x == '' else x)
    df = df.iloc[:, 1:3]
    df.insert(0, 'Stimulus Type', None)
    df.insert(2, 'Numbers', None)
    df.columns = ['Stimulus Type'] + [columns[0]] + ['Numbers'] + [columns[1]]
    dfs[df_name] = df

# define stimulus types:
for df_name, df in dfs.items():
    # print(df_name)
    for index, stim_mrk in enumerate(df['Stimulus Stream']):
        if stim_mrk in s1.values():
            # print('stimulus marker is type 1')
            df.at[index, 'Stimulus Type'] = 's1'
            df.at[index, 'Numbers'] = next(key for key, value in s1.items() if value == stim_mrk)
        elif stim_mrk in s2.values():
            # print('stimulus marker is type 2')
            df.at[index, 'Stimulus Type'] = 's2'
            df.at[index, 'Numbers'] = next(key for key, value in s2.items() if value == stim_mrk)
        elif stim_mrk in response.values():
            # print('stimulus marker is type response')
            df.at[index, 'Stimulus Type'] = 'response'
            df.at[index, 'Numbers'] = next(key for key, value in response.items() if value == stim_mrk)

# discard None values (pray no shit shifted around):
for df_name, df in dfs.items():
    rows_to_drop = []
    for index, stim_mrk in enumerate(df['Stimulus Stream']):
        if stim_mrk not in s1.values() and stim_mrk not in s2.values() and stim_mrk not in response.values():
            rows_to_drop.append(index)
    # Drop the marked rows from the DataFrame
    df.drop(rows_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

# remove Stimulus Stream, convert Numbers and Positions vals to int:
for df_name, df in dfs.items():
    df['Numbers'] = pd.to_numeric(df['Numbers'])
    df['Position'] = pd.to_numeric(df['Position']) / 500
    # df['Position'] = df['Position'] / 500
    # df['Time Differences'] = pd.to_numeric((df['Time Differences']))
    df.drop(columns=['Stimulus Stream'], inplace=True)

# for df_name, df in dfs.items():
#     df.loc[df['Stimulus Type'] == 's2', 'Stimulus Type'] = 'target'
#     df.loc[df['Stimulus Type'] == 's1', 'Stimulus Type'] = 'distractor'


# replace stimulus type with 'target' and 'distractor' accordingly
for df_name, df in dfs.items():
    if df.at[0, 'Stimulus Type'] == 's1':
        df['Stimulus Type'] = df['Stimulus Type'].replace('s1', 'target')
        df['Stimulus Type'] = df['Stimulus Type'].replace('s2', 'distractor')
    elif df.at[0, 'Stimulus Type'] == 's2':
        df['Stimulus Type'] = df['Stimulus Type'].replace('s2', 'target')
        df['Stimulus Type'] = df['Stimulus Type'].replace('s1', 'distractor')


# copy updated dfs for processing of responses:
dfs_copy = {}
for df_name, df in dfs.items():
    dfs_copy[df_name] = df.copy()
    dfs_copy[df_name] = df.assign(Reaction=0, Reaction_Time=0)



tlo1 = 1.020
tlo2 = 0.925
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

for df_name, df in target_responses_dict.items():
    if not df.empty:
        df.columns = ['Target Index', 'Target', 'Target Number', 'Target Position', 'Response Index', 'Response',
                      'Response Number', 'Response Position']

for df_name, df in dfs_copy.items():
    if df_name in target_responses_dict:
        for name, stim_index, response_index, target_time in target_controlled_rows:
            if name == df_name:
                df.at[stim_index, 'Reaction'] = 1
                df.at[response_index, 'Reaction'] = 1
                df.at[response_index, 'Reaction_Time'] = target_time


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
            window_end = distractor_time + tlo1 + 0.2
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

for df_name, df in distractor_responses_dict.items():
    if not df.empty:
        df.columns = ['Target Index', 'Target', 'Target Number', 'Target Position', 'Response Index', 'Response',
                      'Response Number', 'Response Position']


for df_name, df in dfs_copy.items():
    if df_name in distractor_responses_dict:
        for name, stim_index, response_index, distractor_time in distractor_controlled_rows:
            if name == df_name:
                df.at[stim_index, 'Reaction'] = 2
                df.at[response_index, 'Reaction'] = 2
                df.at[response_index, 'Reaction_Time'] = distractor_time

total_responses_df = {}
total_responses_count = {}
for df_name, df in dfs_copy.items():
    responses = df[df['Stimulus Type'] == 'response']
    total_responses_df[df_name] = responses
    response_count = len(responses)
    total_responses_count[df_name] = response_count

# responses to target and distractor have been identified
# responses that do not have a Reaction that is '1', should be included in the false_responses dicts
false_responses_dict = {}

# Loop over each DataFrame to process the false responses
for df_name, df in total_responses_df.items():
    # Filter responses that do not have Reaction equal to '1'
    false_responses = df[df['Reaction'] != 1]

    # Add the false responses to the dictionary
    false_responses_dict[df_name] = false_responses


# plot performance for this condition:
# title should be the df_name
# y axis responses in percentage
# x axis with 'correct hits' and 'errors'
# 2 bar plots: 1 for correct hits and 1 for errors
# all the target_responses summed from all the sub dfs, same for false_responses

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
plt.show()

