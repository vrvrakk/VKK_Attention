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

data_path = Path('C:/Users/vrvra/PycharmProjects/EEG_Tools2')
s1_files = Path('C:/Users/vrvra/PycharmProjects/EEG_Tools2/eeg/s1')
s2_files = Path('C:/Users/vrvra/PycharmProjects/EEG_Tools2/eeg/s2')
new_data = Path('C:/Users/vrvra/PycharmProjects/EEG_Tools2/eeg/new_data')
test_data = Path('C:/Users/vrvra/PycharmProjects/EEG_Tools2/eeg/test')

s1 = {1: 'S  1', 2: 'S  2', 3: 'S  3', 4: 'S  4', 5: 'S  5', 6: 'S  6', 8: 'S  8', 9: 'S  9'}  # stimulus 1 markers
s2 = {1: 'S 65', 2: 'S 66', 3: 'S 67', 4: 'S 68', 5: 'S 69', 6: 'S 70', 8: 'S 72', 9: 'S 73'}  # stimulus 2 markers
response = {1: 'S129', 2: 'S130', 3: 'S131', 4: 'S132', 5: 'S133', 6: 'S134', 8: 'S136', 9: 'S137'}  # response markers

# select .vmrk files:
marker_files = []
for files in os.listdir(data_path):
    if files.endswith('test_both_voices_without_buttons.vmrk'):
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
    for index in range(len(df) - 1):
        if df.at[index, 'Stimulus Type'] == 'response':
            current_timestamp = df.at[index, 'Position'] / 500
            next_timestamp = df.at[index + 1, 'Position'] / 500
            time_difference = next_timestamp - current_timestamp
            df.at[index, 'Time Differences'] = time_difference

# clear invalid responses:
dfs_updated = {}
drop_responses_dict = {}
time_differences_dict = {}
for df_name, df in dfs.items():
    time_differences_list = []
    drop_responses = []

    for index, stimulus in enumerate(df['Stimulus Type']):
        if stimulus == 'response' and index < len(df) - 2:
            current_response = df.at[index, 'Numbers']
            next_stimulus = df.at[index + 1, 'Stimulus Type']
            next_response = df.at[index + 1, 'Numbers']
            over_next_stimulus = df.at[index + 2, 'Stimulus Type']
            over_next_response = df.at[index + 2, 'Numbers']

            if next_stimulus == 'response' and current_response == next_response:
                time_differences_list.append(df.at[index, 'Time Differences'])
                drop_responses.append(index)
            elif over_next_stimulus == 'response' and current_response == over_next_response:
                time_differences_list.extend(
                    [df.at[index + 1, 'Time Differences'], df.at[index + 2, 'Time Differences']])
                drop_responses.extend([index + 1, index + 2])

    drop_responses_dict[df_name] = drop_responses  # save invalid responses in dict
    time_differences_dict[df_name] = time_differences_list
    df = df.drop(drop_responses)
    df.reset_index(drop=True, inplace=True)
    dfs_updated[df_name] = df  # save updated dfs in dict


# copy updated dfs for processing of responses:
dfs_copy = {}
for df_name, df in dfs_updated.items():
    dfs_copy[df_name] = df.copy()
    dfs_copy[df_name] = df.assign(Reaction=0, Reaction_Time=0)

# S1 responses, S2 responses, no responses, errors.
s1_responses_dict = {}
s2_responses_dict = {}
errors_dict = {}
tlo1 = 0.985  # 1.486  # 0.985
tlo2 = 0.925    # 1.288  # 0.925
for df_name, df in dfs_copy.items():
    s1_responses = []
    errors = []
    s1_controlled_rows = set()
    stim_indices = df['Stimulus Type'].index
    stim_types = df['Stimulus Type'].loc[stim_indices]

    # get s1 responses:
    for (stim_index, stimulus) in enumerate(stim_types):
        if stimulus == 's1':
            s1_number = df.at[stim_index, 'Numbers']  # get corresponding number of s1
            s1_time = df.at[stim_index, 'Time']  # get corresponding s1 time
            s1_label = df.at[stim_index, 'Stimulus Type']

            # define time window:
            window_start = s1_time
            window_end = s1_time + tlo2 + 0.2
            window_data = df.loc[(df['Time'] >= window_start) & (df['Time'] <= window_end)]
            if 'response' in window_data['Stimulus Type'].values:
                response_indices_within_window = window_data[window_data['Stimulus Type'] == 'response'].index
                for response_index in response_indices_within_window:
                    response_number = window_data['Numbers'].loc[response_index]
                    response_time = window_data['Time'].loc[response_index]
                    response_label = window_data['Stimulus Type'].loc[response_index]
                    response_type = window_data['Reaction'].loc[response_index]
                    if response_number == s1_number:
                        s1_responses.append((stim_index, s1_label, s1_number, s1_time,
                                             response_index, response_label, response_number, response_time))
                        s1_controlled_rows.add((stim_index, response_index, s1_time))
    s1_responses_dict[df_name] = s1_responses
for df_name, df in dfs_copy.items():
    for stim_index, response_index, s1_time in s1_controlled_rows:
        df.at[stim_index, 'Reaction'] = 1
        df.at[response_index, 'Reaction'] = 1
        df.at[response_index, 'Reaction_Time'] = s1_time

for df_name, df in dfs_copy.items():
    s2_responses = []
    s2_controlled_rows = set()
    stim_indices = df['Stimulus Type'].index
    stim_types = df['Stimulus Type'].loc[stim_indices]
    for (stim_index, stimulus) in enumerate(stim_types):
        if stimulus == 's2':
            s2_number = df.at[stim_index, 'Numbers']  # get corresponding number of s1
            s2_time = df.at[stim_index, 'Time']  # get corresponding s1 time
            s2_label = df.at[stim_index, 'Stimulus Type']

            window_start = s2_time
            window_end = s2_time + tlo2 + 0.2
            window_data = df.loc[(df['Time'] >= window_start) & (df['Time'] <= window_end)]
            if 'response' in window_data['Stimulus Type'].values:
                response_indices_within_window = window_data[window_data['Stimulus Type'] == 'response'].index
                for response_index in response_indices_within_window:
                    response_number = window_data['Numbers'].loc[response_index]
                    response_time = window_data['Time'].loc[response_index]
                    response_label = window_data['Stimulus Type'].loc[response_index]
                    response_type = window_data['Reaction'].loc[response_index]
                    if response_number == s2_number and response_type == 0:
                        s2_responses.append((stim_index, s2_label, s2_number, s2_time,
                                             response_index, response_label, response_number, response_time))
                        s2_controlled_rows.add((stim_index, response_index, s2_time))
    s2_responses_dict[df_name] = s2_responses
for df_name, df in dfs_copy.items():
    for stim_index, response_index, s2_time in s2_controlled_rows:
        df.at[stim_index, 'Reaction'] = 2
        df.at[response_index, 'Reaction'] = 2
        df.at[response_index, 'Reaction_Time'] = s2_time

# convert to dfs for readability
s1_responses_dfs = {}
for df_name, sub_dict in s1_responses_dict.items():
    rows = [{'stim_index': row[0],
             's1_label': row[1],
             's1_number': row[2],
             's1_time': row[3],
             'response_index': row[4],
             'response_label': row[5],
             'response_number': row[6],
             'response_time': row[7]} for row in sub_dict]
    s1_responses_dfs[df_name] = rows
    s1_responses_df = pd.DataFrame(rows)
    s1_responses_dfs[df_name] = s1_responses_df


# Convert s2_responses_dict
s2_responses_dfs = {}
for df_name, sub_dict in s2_responses_dict.items():
    rows = [{'stim_index': row[0],
             's2_label': row[1],
             's2_number': row[2],
             's2_time': row[3],
             'response_index': row[4],
             'response_label': row[5],
             'response_number': row[6],
             'response_time': row[7]} for row in sub_dict]
    s2_responses_dfs[df_name] = rows
    s2_responses_df = pd.DataFrame(rows)
    s2_responses_dfs[df_name] = s2_responses_df

for df_name, df in dfs_copy.items():
    # s2_rows = df[df['Stimulus Type'] == 's2']
    # print("Sample Rows where 'Stimulus Type' is 's2':")
    # print(s2_rows.head())  # shit
    # for index in range(len(df) - 1):
    #     current_time = df.at[index, 'Time']
    #     # print(current_time)
    #     next_time = df.at[index + 1, 'Time']
    #     time_difference = next_time - current_time
    #     df.loc[index, 'Time Differences'] = time_difference
    # df.sort_values(by='Time Differences', ascending=True, inplace=True)
    s1_count = len(df[df['Stimulus Type'] == 's1'])
    s2_count = len(df[df['Stimulus Type'] == 's2'])

    combined_s1_responses = s1_responses_dfs[df_name]  # Retrieve DataFrame for df_name
    combined_s2_responses = s2_responses_dfs[df_name]  # Retrieve DataFrame for df_name

    s1_counts = combined_s1_responses.groupby('response_label').size().to_dict() if not combined_s1_responses.empty else {}
    s2_counts = combined_s2_responses.groupby('response_label').size().to_dict() if not combined_s2_responses.empty else {}

    combined_counts = {
        's1': s1_counts.get('response', 0),  # Get count of 'response' label for 's1'
        's2': s2_counts.get('response', 0),  # Get count of 'response' label for 's2'
    }

    # Create a bar plot with additional bars for total counts of 's1' and 's2'
    plt.figure(figsize=(12, 8))
    bars = plt.bar(range(len(combined_counts)), combined_counts.values(), color=['blue', 'orange'])

    # Add additional bars for total counts of 's1' and 's2'
    plt.bar(len(combined_counts), s1_count, color='lightblue', label='Total s1')
    plt.bar(len(combined_counts) + 1, s2_count, color='lightsalmon', label='Total s2')

    plt.title(f'Comparison of Response Counts between s1 and s2 for dataframe: {df_name}')
    plt.xlabel('Stimulus Type')
    plt.ylabel('Response Counts')
    plt.xticks(range(len(combined_counts) + 2), list(combined_counts.keys()) + [f'Total s1: {s1_count}', f'Total s2: {s2_count}'], rotation=45)
    plt.legend(title='Response Category')
    plt.show()

# TODO: calculate RT:
# stimulus_time - response_time
# s1:
RT_s1 = []
RT_s1_dfs = {}
for i, (df_name, df) in enumerate(s1_responses_dfs.items()):
    if not df.empty and 's1_time' in df.columns:
        s1_time = df['s1_time'].values
        response_s1 = df['response_time'].values
        RT = response_s1 - s1_time
        RT_s1.append(RT)
        RT_s1_dfs[df_name] = pd.DataFrame({'RT': RT})

RT_vals_s1 = []
for df_name, RT_dfs in RT_s1_dfs.items():
    if not df.empty and 's1_time' in df.columns:
        RT_vals_s1.extend(RT_dfs['RT'].values)
plt.figure()
plt.hist(RT_vals_s1, color='skyblue', edgecolor='black')
plt.xlabel('Reaction Time')
plt.ylabel('Frequency')
plt.title('Reaction Times Distribution')
plt.show()

# # median RT:
# RT_median = np.median(RT_vals_s1)
# RT_mean = np.mean(RT_vals_s1)
# RT_max = np.max(RT_vals_s1)
# RT_min = np.min(RT_vals_s1)
#
RT_s2 = []
RT_s2_dfs = {}
for i, (df_name, df) in enumerate(s2_responses_dfs.items()):
    if not df.empty and 's2_time' in df.columns:
        s2_time = df['s2_time'].values
        response_s2 = df['response_time'].values
        RT = response_s2 - s2_time
        RT_s2.append(RT)
        RT_s2_dfs[df_name] = pd.DataFrame({'RT': RT})

RT_vals_s2 = []
for df_name, RT_dfs in RT_s2_dfs.items():
    RT_vals_s2.extend(RT_dfs['RT'].values)
plt.figure()
plt.hist(RT_vals_s2, color='skyblue', edgecolor='black')
plt.xlabel('Reaction Time')
plt.ylabel('Frequency')
plt.title('Reaction Times Distribution')
plt.show()

# # calculate RTs
# RT_median = np.median(RT_vals_s2)
# RT_mean = np.mean(RT_vals_s2)
# RT_max = np.max(RT_vals_s2)
# RT_min = np.min(RT_vals_s2)
