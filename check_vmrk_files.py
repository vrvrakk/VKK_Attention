import os
from pathlib import Path
import pandas as pd
import numpy as np

data_path = Path.cwd()
generated_sequences = Path('C:/projects/VKK_Attention/data/generated_sequences')
# markers:
stim1 = {1: 'S  1', 2: 'S  2', 3: 'S  3', 4: 'S  4', 5: 'S  5', 6: 'S  6', 8: 'S  8', 9: 'S  9'}  # stimulus 1 markers
stim2 = {1: 'S 65', 2: 'S 66', 3: 'S 67', 4: 'S 68', 5: 'S 69', 6: 'S 70', 8: 'S 72', 9: 'S 73'}  # stimulus 2 markers
response = {1: 'S129', 2: 'S130', 3: 'S131', 4: 'S132', 5: 'S133', 6: 'S134', 8: 'S136', 9: 'S137'}  # response markers

# get files:
marker_file = []
sequence_file = []
for file in os.listdir(data_path):
    if file.endswith('koko1.vmrk'):
        marker_file.append(data_path/file)
for file in os.listdir(generated_sequences):
    if file.endswith('kolos_block_10.csv'):
        sequence_file.append(generated_sequences/file)

# convert to panda dfs:
columns = ['Stimulus Stream', 'Position', 'Time Difference']
dfs = {}
for index, file_info in enumerate(marker_file):
    file_name = file_info.stem
    df = pd.read_csv(file_info, delimiter='\t', header=None)
    df_name = f'{file_name}'
    df = df.iloc[10:] # ignore first 10 rows
    df.reset_index(drop=True, inplace=True) # reset index
    df = df[0].str.split(',', expand=True).applymap(lambda x: None if x == '' else x) # split by comma
    df = df.iloc[:, 1:3]  # ignore all but first 3 columns
    df.insert(0, 'Stimulus Type', None)  # insert column at idx 0
    df.insert(2, 'Numbers', None)
    df.columns = ['Stimulus Type'] + [columns[0]] + ['Numbers'] + [columns[1]]
    df['Time'] = np.nan
    df['Time Differences'] = np.nan
    dfs[df_name] = df

for df_name, df in dfs.items():
    # print(df_name)
    for index, stim_mrk in enumerate(df['Stimulus Stream']):
        if stim_mrk in stim1.values():
            # print('stimulus marker is type 1')
            df.at[index, 'Stimulus Type'] = 'stim1'
            df.at[index, 'Numbers'] = next(key for key, value in stim1.items() if value == stim_mrk)
        elif stim_mrk in stim2.values():
            # print('stimulus marker is type 2')
            df.at[index, 'Stimulus Type'] = 'stim2'
            df.at[index, 'Numbers'] = next(key for key, value in stim2.items() if value == stim_mrk)
        elif stim_mrk in response.values():
            # print('stimulus marker is type response')
            df.at[index, 'Stimulus Type'] = 'response'
            df.at[index, 'Numbers'] = next(key for key, value in response.items() if value == stim_mrk)

for df_name, df in dfs.items():
    rows_to_drop = []
    for index, stim_mrk in enumerate(df['Stimulus Stream']):
        if stim_mrk not in stim1.values() and stim_mrk not in stim2.values() and stim_mrk not in response.values():
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
    df.drop(df[df['Stimulus Type'] == 'response'].index, inplace=True)
    df.reset_index(inplace=True, drop=True)
    stim1_count = len(df[df['Stimulus Type'] == 'stim1'])
    stim2_count = len(df[df['Stimulus Type'] == 'stim2'])

for df_name, df in dfs.items():
    for index in range(len(df) - 1):
        current_time = df.at[index, 'Time']
        next_time = df.at[index + 1, 'Time']
        time_difference = next_time - current_time
        df.loc[index, 'Time Differences'] = time_difference


# read generated sequences file:
seq_dfs = {}
seq_columns = ['Event IDs',	'Stream of Trials',	'Time onsets',	'Time offsets',	'Stimulus',	'Conflicts']
seq_columns2 = ['Event IDs', 'Stimulus Type', 'Numbers']
new_order = ['Event IDs', 'Numbers', 'Numbers from df', 'Stimulus Type', 'Stimulus Type df', 'Time onsets',
             'Time offsets', 'Time Differences']
for index, file_info in enumerate(sequence_file):
    file_name = file_info.stem
    seq_df = pd.read_csv(file_info, delimiter=';', header=None)
    df_name = f'{file_name}'
    seq_df.columns = seq_columns
    seq_df = seq_df.iloc[1:]
    seq_df.drop(columns='Conflicts', inplace=True)
    seq_df['Time Differences'] = np.nan
    seq_df.rename(columns={'Stream of Trials': 'Numbers'}, inplace=True)
    seq_df['Numbers from df'] = df['Numbers']
    seq_df['Stimulus from df'] = df['Stimulus Type']
    seq_df = seq_df[new_order]
    seq_df['Numbers from df'] = seq_df['Numbers from df'].fillna(0).astype(int)
    seq_df.reset_index(inplace=True)
    seq_dfs[df_name] = seq_df




