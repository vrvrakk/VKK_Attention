import pandas as pd
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

default_dir = Path.cwd()
sub = input('Subject number: ')

sub_dir = default_dir / 'data' / 'eeg' / 'raw' / f'{sub}'
# specify condition to focus on:
condition = input('Choose a condition (a1, a2, e1 or e2): ')

# extract marker files; they contain all the events that occurred during the exp.
marker_files = []
for files in sub_dir.iterdir():
    if files.is_file() and f'{condition}' in files.name:
        if files.is_file() and '.vmrk' in files.name:
            marker_files.append(files)

# define events by creating dictionaries with keys and their corresponding values:
markers_dict = {
    's1_events': {  # Stimulus 1 markers
        'S  1': 1,
        'S  2': 2,
        'S  3': 3,
        'S  4': 4,
        'S  5': 5,
        'S  6': 6,
        'S  7': 7,
        'S  8': 8,
        'S  9': 9
    },
    's2_events': {  # Stimulus 2 markers
        'S 65': 1,
        'S 66': 2,
        'S 67': 3,
        'S 68': 4,
        'S 69': 5,
        'S 70': 6,
        'S 71': 7,
        'S 72': 8,
        'S 73': 9
    },
    'response_events': {  # Response markers
        'S129': 1,
        'S130': 2,
        'S131': 3,
        'S132': 4,
        'S133': 5,
        'S134': 6,
        'S136': 8,
        'S137': 9
    }
}
s1_events = markers_dict['s1_events']
s2_events = markers_dict['s2_events']  # stimulus 2 markers
response_events = markers_dict['response_events']  # response markers

# read marker files as csv files:
dfs = {}  # this dataFrame will contain sub-dataframes for every marker file

for index, files in enumerate(marker_files):
    df = pd.read_csv(sub_dir/files.name, delimiter='\t', header=None)  # separate along rows
    df_name = f'df_{files.name}' # name according to files.name (contains sub initials + condition
    df = df.iloc[10:]  # delete first 10 rows  (because they contain nothing useful for us)
    df = df.reset_index(drop=True, inplace=False)  # once rows are dropped, we need to reset index so it starts from 0 again
    df = df[0].str.split(',', expand=True).applymap(lambda x: None if x == '' else x) # separates info from every row into separate columns
    # information is split whenever a ',' comma is present; otherwise all info in each row is all under one column -> which sucks
    df = df.iloc[:, 1:3] # we keep only the columns 1:3, with all their data
    df.insert(0, 'Stimulus Type', None)  # we insert an additional column, which we will fill in later
    df.insert(2, 'Numbers', None)  # same here
    columns = ['Stimulus Stream', 'Position', 'Time Difference']  # we pre-define some columns of our dataframe;
    # position is time in data samples
    df.columns = ['Stimulus Type'] + [columns[0]] + ['Numbers'] + [columns[1]] # we re-order our columns
    dfs[df_name] = df # we save every single sub-dataframe into the empty dfs dictionary we created

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

# drop any remaining rows with None Stimulus Types (like: 'S 64')
for df_name, df in dfs.items():
    rows_to_drop = []
    for index, stim_mrk in enumerate(df['Stimulus Stream']):
        if stim_mrk not in s1_events.keys() and stim_mrk not in s2_events.keys() and stim_mrk not in response_events.keys():
            rows_to_drop.append(index)
    # Drop the marked rows from the DataFrame
    df.drop(rows_to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)


# define target stream: s1 or s2?
# we know this from the condition
# which number was the target? we get the info from the df; first response's number
if condition == 'a1' or 'e1':
    target_stream = 's1'
elif condition == 'a2' or 'e2':
    target_stream = 's2'

target_numbers = {}
# now target_number:
for df_name, df in dfs.items():
    first_response_row = df[df['Stimulus Type'] == 'response'].iloc[0]
    target_number = first_response_row['Numbers']
    target_numbers[df_name] = target_number

# extract response rows
all_responses = {}
for df_name, df in dfs.items():
     response = df[df['Stimulus Type'] == 'response']
     all_responses[df_name] = response

# count sum of responses
total_responses = {}
for df_name, df in all_responses.items():
    sum = len(df)
    total_responses[df_name] = sum

# get sum of target stimulus == target stream
all_target_stim = {}
for df_name, df in dfs.items():
   target_number = target_numbers[df_name]  # Get the target number for this DataFrame
   target_stim_rows = df[df['Stimulus Type'] == target_stream]  # Check if target stream is in 'Stimulus Type'
   filt_rows = target_stim_rows[target_stim_rows['Numbers'] == target_number] # keep only rows where stimulus number matches target num
   all_target_stim[df_name] = filt_rows # do this for every sub-df

total_stim = {}
for df_name, df in all_target_stim.items():
    sum = len(df)
    total_stim[df_name] = sum  # count total target numbers said by target stream

# calculate performance for every sub-dataframe: how?
# convert positions column values from samples to time: cell value/500
for df_name, df in dfs.items():
    df['Time (s)'] = df['Position'].astype(float) / 500  # Convert samples to time in seconds
# additional column with name 'score'
for df_name, df in dfs.items():
    df['Score'] = None  # Initialize the 'Score' column

# define a time window for the iteration; based on avg reaction time (button press) -> I would say within 0.5s after target number
time_window = 0.7
# find response rows for every sub-df; iterate through the df;
# if there was a target stimulus up to 0.5s before response, with the same number -> in the corresponding cell under 'score', enter 1
# if any response remains, check if within the time window a stimulus that is not the target_stream under Stimulus Type, has the same number as the target_number, under 'Numbers'
# if yes, that response receives a 0 under 'score'
# at the end, we get the sum of the score for score == 1 and for score == 0;

for df_name, df in dfs.items():
    target_number = target_numbers[df_name]  # Get the target number for the current DataFrame
    responses = df[df['Stimulus Type'] == 'response']  # Get all response rows

    for response_index, response_row in responses.iterrows():
        response_time = response_row['Time (s)']  # Time of the response

        # Find target stimuli within the time window before the response
        potential_target_stim = df[(df['Stimulus Type'] == target_stream) &
                                   (df['Numbers'] == target_number) &
                                   (df['Time (s)'] >= response_time - time_window) &
                                   (df['Time (s)'] < response_time)]

        if not potential_target_stim.empty:
            df.at[response_index, 'Score'] = 1  # Correct response

        else:
            # Check for distractor responses (same number but different stream)
            potential_distractor_stim = df[(df['Stimulus Type'] != target_stream) &
                                           (df['Numbers'] == target_number) &
                                           (df['Time (s)'] >= response_time - time_window) &
                                           (df['Time (s)'] < response_time)]

            if not potential_distractor_stim.empty:
                df.at[response_index, 'Score'] = 0  # Distractor response


# we get the performance percentage, based on the total amount of target numbers said by target stream, and the total correct responses
# 0 are classified as distractor responses
performance = {}

for df_name, df in dfs.items():
    correct_responses = len(df[(df['Stimulus Type'] == 'response') & (df['Score'] == 1)])  # Count correct responses
    total_target_numbers = total_stim[df_name]  # Total target numbers said by target stream

    if total_target_numbers > 0:  # Avoid division by zero
        performance[df_name] = (correct_responses / total_target_numbers) * 100  # Calculate percentage
    else:
        performance[df_name] = 0

print(performance)  # Display the performance percentages for each DataFrame


