import os
from pathlib import Path
import pandas as pd
import re
from get_streams_and_stream_params import tlo1
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

tlo1 = tlo1 /1000


# select .vmrk files:
def select_marker_files():
    marker_files = []
    pattern = r'\d{6}_\w{2}'
    regex = re.compile(pattern)
    for dir_name in os.listdir(data_path):
        dir_path = Path(data_path/dir_name)

        if os.path.isdir(dir_path): # if path exists
            if regex.match(dir_name):
                for file_name in os.listdir(dir_path):
                    if 'hz' in file_name:
                        if file_name.endswith('azimuth.vmrk'):
                            # file_path = os.path.join(dir_path, file_name)
                            marker_files.append(dir_path/file_name)
    return marker_files


# save marker files as pandas dataframe:
def marker_files_dfs(marker_files):
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
    return dfs


# define stimulus types:
def stimulus_types(dfs):
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
    return dfs


# discard None values (pray no shit shifted around):
def none_vals(dfs):
    for df_name, df in dfs.items():
        rows_to_drop = []
        for index, stim_mrk in enumerate(df['Stimulus Stream']):
            if stim_mrk not in s1.values() and stim_mrk not in s2.values() and stim_mrk not in response.values():
                rows_to_drop.append(index)
        # Drop the marked rows from the DataFrame
        df.drop(rows_to_drop, inplace=True)
        df.reset_index(drop=True, inplace=True)
    return dfs


# remove Stimulus Stream, convert Numbers and Positions vals to int:
def convert_to_numeric(dfs):
    for df_name, df in dfs.items():
        df['Numbers'] = pd.to_numeric(df['Numbers'])
        df['Position'] = pd.to_numeric(df['Position']) / 500
        # df['Position'] = df['Position'] / 500
        # df['Time Differences'] = pd.to_numeric((df['Time Differences']))
        df.drop(columns=['Stimulus Stream'], inplace=True)
    return dfs


def convert_stimuli(marker_files, dfs):
    for file_name in marker_files:
        name = file_name.name
        for df_name, df in dfs.items():
            if 's1' in name:
                if df_name[3:] in name:
                    df.loc[df['Stimulus Type'] == 's1', 'Stimulus Type'] = 'target'
                    df.loc[df['Stimulus Type'] == 's2', 'Stimulus Type'] = 'distractor'
            if 's2' in name:
                if df_name[3:] in name:
                    df.loc[df['Stimulus Type'] == 's2', 'Stimulus Type'] = 'target'
                    df.loc[df['Stimulus Type'] == 's1', 'Stimulus Type'] = 'distractor'
    return dfs
