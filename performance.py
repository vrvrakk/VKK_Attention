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

data_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/raw/s1/new_azimuth/240424_ls')
s1_files = data_path / 's1'


s1 = {1: 'S  1', 2: 'S  2', 3: 'S  3', 4: 'S  4', 5: 'S  5', 6: 'S  6', 8: 'S  8', 9: 'S  9'}  # stimulus 1 markers
s2 = {1: 'S 65', 2: 'S 66', 3: 'S 67', 4: 'S 68', 5: 'S 69', 6: 'S 70', 8: 'S 72', 9: 'S 73'}  # stimulus 2 markers
response = {1: 'S129', 2: 'S130', 3: 'S131', 4: 'S132', 5: 'S133', 6: 'S134', 8: 'S136', 9: 'S137'}  # response markers

# select .vmrk files:
marker_files = []
for files in os.listdir(data_path):
    if files.endswith('s1_azimuth.vmrk'):
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
    df['Time'] = np.nan
    df['Time Differences'] = np.nan
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
    df['Position'] = pd.to_numeric(df['Position'])
    df['Time Differences'] = pd.to_numeric((df['Time Differences']))
    df['Time'] = df['Position'] / 500
    df.drop(columns=['Stimulus Stream'], inplace=True)
    # drop response rows that are invalid:
    # for index in range(len(df) - 1):
    #     if df.at[index, 'Stimulus Type'] == 'response':
    #         current_timestamp = df.at[index, 'Position'] / 500
    #         next_timestamp = df.at[index + 1, 'Position'] / 500
    #         time_difference = next_timestamp - current_timestamp
    #         df.at[index, 'Time Differences'] = time_difference


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

# target responses, errors
target_responses_dict = {}
distractor_responses_dict = {}
target_controlled_rows = set()
distractor_controlled_rows = set()
errors_dict = {}
tlo1 = 1.020  # 1.486  # 0.985
tlo2 = 0.925    # 1.288  # 0.925
for df_name, df in dfs_copy.items():
    target_responses = []
    errors = []

    stim_indices = df['Stimulus Type'].index
    stim_types = df['Stimulus Type'].loc[stim_indices]

    # get s1 responses:
    #todo: and correct responses vs errors (distractor responses, no response, wrong response)
    for (stim_index, stimulus) in enumerate(stim_types):
        if stimulus == 'target':
            target_number = df.at[stim_index, 'Numbers']  # get corresponding number of s1
            target_time = df.at[stim_index, 'Time']  # get corresponding s1 time
            target_label = df.at[stim_index, 'Stimulus Type']

            # define time window:
            window_start = target_time
            window_end = target_time + tlo1 + 0.25
            window_data = df.loc[(df['Time'] >= window_start) & (df['Time'] <= window_end)]
            if 'response' in window_data['Stimulus Type'].values:
                response_indices_within_window = window_data[window_data['Stimulus Type'] == 'response'].index
                for response_index in response_indices_within_window:
                    response_number = window_data['Numbers'].loc[response_index]
                    response_time = window_data['Time'].loc[response_index]
                    response_label = window_data['Stimulus Type'].loc[response_index]
                    response_type = window_data['Reaction'].loc[response_index]
                    if response_number == target_number:
                        target_responses.append((stim_index, target_label, target_number, target_time,
                                             response_index, response_label, response_number, response_time))
                        target_controlled_rows.add((df_name, stim_index, response_index, target_time))
        target_responses_df = pd.DataFrame(target_responses)
        target_responses_dict[df_name] = target_responses_df

for df_name, df in dfs_copy.items():
    if df_name in target_responses_dict:
        for name, stim_index, response_index, target_time in target_controlled_rows:
            if name == df_name:
                df.at[stim_index, 'Reaction'] = 1
                df.at[response_index, 'Reaction'] = 1
                df.at[response_index, 'Reaction_Time'] = target_time


for df_name, df in dfs_copy.items():
    distractor_responses = []
    stim_indices = df['Stimulus Type'].index
    stim_types = df['Stimulus Type'].loc[stim_indices]
    for (stim_index, stimulus) in enumerate(stim_types):
        if stimulus == 'distractor':
            distractor_number = df.at[stim_index, 'Numbers']  # get corresponding number of s1
            distractor_time = df.at[stim_index, 'Time']  # get corresponding s1 time
            distractor_label = df.at[stim_index, 'Stimulus Type']

            window_start = distractor_time
            window_end = distractor_time + tlo1 + 0.2
            window_data = df.loc[(df['Time'] >= window_start) & (df['Time'] <= window_end)]
            if 'response' in window_data['Stimulus Type'].values:
                response_indices_within_window = window_data[window_data['Stimulus Type'] == 'response'].index
                for response_index in response_indices_within_window:
                    response_number = window_data['Numbers'].loc[response_index]
                    response_time = window_data['Time'].loc[response_index]
                    response_label = window_data['Stimulus Type'].loc[response_index]
                    response_type = window_data['Reaction'].loc[response_index]
                    if response_number == distractor_number and response_type == 0:
                        distractor_responses.append((stim_index, distractor_label, distractor_number, distractor_time,
                                             response_index, response_label, response_number, response_time))
                        distractor_controlled_rows.add((df_name, stim_index, response_index, distractor_time))
    distractor_responses_df = pd.DataFrame(distractor_responses)
    distractor_responses_dict[df_name] = distractor_responses

for df_name, df in dfs_copy.items():
    if df_name in distractor_controlled_rows:
        for name, stim_index, response_index, distractor_time in distractor_controlled_rows:
            if name == df_name:
                df.at[stim_index, 'Reaction'] = 2
                df.at[response_index, 'Reaction'] = 2
                df.at[response_index, 'Reaction_Time'] = distractor_time

# convert to dfs for readability
target_responses_dfs = {}
for df_name, df in target_responses_dict.items():
    rows = [{'stim_index': row[0],
             'target_label': row[1],
             'target_number': row[2],
             'target_time': row[3],
             'response_index': row[4],
             'response_label': row[5],
             'response_number': row[6],
             'response_time': row[7]} for index, row in df.iterrows()]
    target_responses_df = pd.DataFrame(rows)
    target_responses_dfs[df_name] = target_responses_df


# Convert distractor_responses_dict
distractor_responses_dfs = {}
for df_name, sub_dict in distractor_responses_dict.items():
    rows = [{'stim_index': row[0],
             'distractor_label': row[1],
             'distractor_number': row[2],
             'distractor_time': row[3],
             'response_index': row[4],
             'response_label': row[5],
             'response_number': row[6],
             'response_time': row[7]} for row in sub_dict]
    distractor_responses_dfs[df_name] = rows
    distractor_responses_df = pd.DataFrame(rows)
    distractor_responses_dfs[df_name] = distractor_responses_df

data_labels = []
target_response_percentages = []
distractor_response_percentages = []
target_total_counts = []
distractor_total_counts = []

# Iterate through each DataFrame
for df_name, df in dfs_copy.items():
    # Count the instances of each stimulus type
    target_count = len(df[df['Stimulus Type'] == 'target'])
    distractor_count = len(df[df['Stimulus Type'] == 'distractor'])

    # Retrieve DataFrames for each response type
    combined_target_responses = target_responses_dfs[df_name]
    combined_distractor_responses = distractor_responses_dfs[df_name]

    # Group by response label and count occurrences
    target_counts = combined_target_responses.groupby(
        'response_label').size().to_dict() if not combined_target_responses.empty else {}
    distractor_counts = combined_distractor_responses.groupby(
        'response_label').size().to_dict() if not combined_distractor_responses.empty else {}

    # Calculate the percentage of 'response' for 'target' and 'distractor'
    if target_count > 0:
        target_response_percentage = (target_counts.get('response', 0) / target_count) * 100
    else:
        target_response_percentage = 0  # Avoid division by zero

    if distractor_count > 0:
        distractor_response_percentage = (distractor_counts.get('response', 0) / distractor_count) * 100
    else:
        distractor_response_percentage = 0  # Avoid division by zero

    # Store the calculated percentages
    target_response_percentages.append(target_response_percentage)
    distractor_response_percentages.append(distractor_response_percentage)

    # Store total counts of 'target' and 'distractor'
    target_total_counts.append(target_count)
    distractor_total_counts.append(distractor_count)

    # Label for this set of bars
    data_labels.append(df_name)

# Number of groups
n_groups = len(data_labels)

# Create figure and axes
fig, ax = plt.subplots(figsize=(12, 8))

# Index for the groups
index = np.arange(n_groups)
bar_width = 0.2

# Plotting each set of bars
ax.bar(index, target_response_percentages, bar_width, label='target Response %')
ax.bar(index + bar_width, distractor_response_percentages, bar_width, label='distractor Response %')
ax.bar(index + 2 * bar_width, np.array(target_total_counts) / np.array(target_total_counts).max() * 100, bar_width,
       label='Total target %')
ax.bar(index + 3 * bar_width, np.array(distractor_total_counts) / np.array(distractor_total_counts).max() * 100, bar_width,
       label='Total distractor %')

# Add titles and labels
ax.set_xlabel('DataFrame')
ax.set_ylabel('Performance (%)')
ax.set_title('Comparison of Response and Total Counts across DataFrames (as %)')
ax.set_xticks(index + 1.5 * bar_width)
ax.set_xticklabels(data_labels, rotation=45)
ax.legend()

# Show plot
plt.show()

# TODO: calculate RT:
# stimulus_time - response_time
# target:
RT_target = []
RT_target_dfs = {}
for i, (df_name, df) in enumerate(target_responses_dfs.items()):
    if not df.empty and 'target_time' in df.columns:
        target_time = df['target_time'].values
        response_target = df['response_time'].values
        RT = response_target - target_time
        RT_target.append(RT)
        RT_target_dfs[df_name] = pd.DataFrame({'RT': RT})

RT_vals_target = []
for df_name, RT_dfs in RT_target_dfs.items():
    if not df.empty and 'target_time' in df.columns:
        RT_vals_target.extend(RT_dfs['RT'].values)
plt.figure()
plt.hist(RT_vals_target, color='skyblue', edgecolor='black')
plt.xlabel('Reaction Time')
plt.ylabel('Frequency')
plt.title('Reaction Times Distribution')
plt.show()

# # median RT:
# RT_median = np.median(RT_vals_target)
# RT_mean = np.mean(RT_vals_target)
# RT_max = np.max(RT_vals_target)
# RT_min = np.min(RT_vals_target)
#
# RT_distractor = []
# RT_distractor_dfs = {}
# for i, (df_name, df) in enumerate(distractor_responses_dfs.items()):
#     if not df.empty and 'distractor_time' in df.columns:
#         distractor_time = df['distractor_time'].values
#         response_distractor = df['response_time'].values
#         RT = response_distractor - distractor_time
#         RT_distractor.append(RT)
#         RT_distractor_dfs[df_name] = pd.DataFrame({'RT': RT})
#
# RT_vals_distractor = []
# for df_name, RT_dfs in RT_distractor_dfs.items():
#     RT_vals_distractor.extend(RT_dfs['RT'].values)
# plt.figure()
# plt.hist(RT_vals_distractor, color='skyblue', edgecolor='black')
# plt.xlabel('Reaction Time')
# plt.ylabel('Frequency')
# plt.title('Reaction Times Distribution')
# plt.show()

# # calculate RTs
# RT_median = np.median(RT_vals_distractor)
# RT_mean = np.mean(RT_vals_distractor)
# RT_max = np.max(RT_vals_distractor)
# RT_min = np.min(RT_vals_distractor)
