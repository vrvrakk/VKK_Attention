import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

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

# get timepoints from samples in Positions:
for df_name, df in dfs.items():
    df['Timepoints'] = df['Position'].astype(float) / 500

# define target stream: s1 or s2?
# we know this from the condition
if condition == 'a1' or 'e1':
    target_stream = 's1'
elif condition == 'a2' or 'e2':
    target_stream = 's2'

# define target number for each block:
# first define csv path:
csv_path = default_dir / 'data' / 'params' / f'{sub}.csv'
# read csv path
csv = pd.read_csv(csv_path)

def get_target_blocks(condition):
    target_blocks = []
    # iterate through the values of the csv path; first enumerate to get the indices of each row
    for index, items in enumerate(csv.values):
        block_seq = items[0]  # from first column, save info as variable block_seq
        block_condition = items[1]  # from second column, save info as var block_condition
        target_number = items[3]  # for third, save info as target_number
        if condition == 'a1':  # if said condition is 'a1' -> azimuth, s1 stream
            if block_seq == 's1':
                if block_condition == 'azimuth':
                    block = csv.iloc[index]
                    target_blocks.append(block) # append all relevant rows into the empty condition list
    target_blocks = pd.DataFrame(target_blocks).reset_index(drop=True)  # convert condition list to a dataframe
    return target_blocks


target_blocks = get_target_blocks(condition)

# now target_number:
# define time window:
time_window = 0.7  # within this window we check for responses matching to the target stimulus
correct_responses_df = {}
total_target_nums_df = {}
total_responses = {}
reaction_times_df = {}  # Dictionary to store reaction times for each DataFrame
for target_block, df_name in zip(target_blocks.values, dfs): # iterate through target_block and the dataframes together (zipping-> gotta have same length)
    target_number = target_block[3]  # define target number as the value within the column[3] of the target_block dataframe
    df = dfs[df_name]  # which corresponds to a specific sub-df
    stimulus = df[df['Stimulus Type'] == target_stream]  # filter out rows with the target stimulus
    response = df[df['Stimulus Type'] == 'response']  # filter out rows with responses
    total_responses[df_name] = response
    # Iterate through each stimulus row in the `stimulus` DataFrame
    total_target_nums = []
    correct_responses = []
    reaction_times = []
    added_responses = set()  # Set to keep track of already added responses
    for stim_index, stim_row in stimulus.iterrows():
        stim_timepoint = stim_row['Timepoints']
        stim_num = stim_row['Numbers']
        # Check if the stimulus number matches the target number
        if stim_num == target_number:
            total_target_nums.append(stim_num)
        # is there a response 0.5 seconds after stim_num? and does the number match?
            matching_responses = response[
                (response['Numbers'] == target_number) &  # Match by target number
                (response['Timepoints'] >= stim_timepoint) &  # Response occurs after stimulus
                (response['Timepoints'] <= stim_timepoint + time_window)]  # Within time window
            for _, resp_row in matching_responses.iterrows():

                response_timepoint = resp_row['Timepoints']  # Define response_timepoint
                reaction_time = response_timepoint - stim_timepoint  # Calculate reaction time
                reaction_times.append(reaction_time)  # Store the reaction time

                # Use tuple of index values to avoid duplicate entries
                resp_index_tuple = tuple(resp_row)
                if resp_index_tuple not in added_responses:
                    correct_responses.append(resp_row)
                    added_responses.add(resp_index_tuple)
    if correct_responses:
        # Create a DataFrame from the list of unique matching responses
        correct_responses_df[df_name] = pd.DataFrame(correct_responses)
    if total_target_nums:
        total_target_nums_df[df_name] = pd.DataFrame(total_target_nums)
    if reaction_times:
        # Store reaction times in a DataFrame
        reaction_times_df[df_name] = pd.DataFrame(reaction_times, columns=['Reaction Time'])




# plot reaction times:

# Combine all reaction times from the reaction_times_df dictionary into a single list
all_reaction_times = []
for df_name, rt_df in reaction_times_df.items():
    all_reaction_times.extend(rt_df['Reaction Time'].tolist())  # use 'extend' to collect all RTs into a single list

# Convert the list to a DataFrame for easier manipulation
reaction_times_combined = pd.DataFrame(all_reaction_times, columns=['Reaction Time']) # re-create a dataframe with extended list of RTs

# Plot the distribution of reaction times
plt.figure(figsize=(10, 6))
sns.histplot(reaction_times_combined['Reaction Time'], bins=7, kde=False, stat="percent",
             binrange=(0.1, 0.7), color='blue')

# Set plot labels and title
plt.xlabel('Reaction Time (s)')
plt.ylabel('Percentage (%)')
plt.title('Distribution of Reaction Times (RTs)')

# Show the plot
plt.show()


########### performance

# Initialize performance metrics dictionary
performance_metrics = {}

for df_name in dfs.keys():  # Loop through each DataFrame name

    # Total target stimuli
    total_stimuli = len(total_target_nums_df[df_name]) if df_name in total_target_nums_df else 0

    # Total responses
    total_responses_count = len(total_responses[df_name]) if df_name in total_responses else 0

    # Total correct responses
    total_correct_responses = len(correct_responses_df[df_name]) if df_name in correct_responses_df else 0

    # Calculate Hit Rate
    hit_rate = total_correct_responses / total_stimuli if total_stimuli > 0 else 0

    # Calculate False Alarm Rate
    total_false_alarms = total_responses_count - total_correct_responses
    false_alarm_rate = total_false_alarms / total_responses_count if total_responses_count > 0 else 0

    # Calculate Mean and Std of Reaction Times
    if df_name in reaction_times_df:
        mean_reaction_time = reaction_times_df[df_name]['Reaction Time'].mean()
        std_reaction_time = reaction_times_df[df_name]['Reaction Time'].std()
    else:
        mean_reaction_time = None
        std_reaction_time = None

    # Store performance metrics
    performance_metrics[df_name] = {
        'Hit Rate': hit_rate,
        'False Alarm Rate': false_alarm_rate,
        'Mean Reaction Time': mean_reaction_time,
        'STD Reaction Time': std_reaction_time
    }

# Create a DataFrame to display performance metrics
performance_df = pd.DataFrame.from_dict(performance_metrics, orient='index')
print("Performance Metrics for Each DataFrame:")
print(performance_df)


# OVERALL PERFORMANCE METRICS
# Combine all total target numbers into a single DataFrame
all_target_nums_df = pd.concat(total_target_nums_df.values(), ignore_index=True) if total_target_nums_df else pd.DataFrame()

# Combine all responses into a single DataFrame
all_responses_df = pd.concat(total_responses.values(), ignore_index=True) if total_responses else pd.DataFrame()

# Combine all correct responses into a single DataFrame
all_correct_responses_df = pd.concat(correct_responses_df.values(), ignore_index=True) if correct_responses_df else pd.DataFrame()

# Combine all reaction times into a single DataFrame
all_reaction_times_df = pd.concat(reaction_times_df.values(), ignore_index=True) if reaction_times_df else pd.DataFrame()

# Calculate the total number of target stimuli
total_stimuli_combined = len(all_target_nums_df)

# Calculate the total number of responses
total_responses_combined = len(all_responses_df)

# Calculate the total number of correct responses
total_correct_responses_combined = len(all_correct_responses_df)

# Calculate Hit Rate
hit_rate_combined = total_correct_responses_combined / total_stimuli_combined if total_stimuli_combined > 0 else 0

# Calculate False Alarm Rate
total_false_alarms_combined = total_responses_combined - total_correct_responses_combined
false_alarm_rate_combined = total_false_alarms_combined / total_responses_combined if total_responses_combined > 0 else 0

# Calculate Mean and Std of Reaction Times
if not all_reaction_times_df.empty:
    mean_reaction_time_combined = all_reaction_times_df['Reaction Time'].mean()
    std_reaction_time_combined = all_reaction_times_df['Reaction Time'].std()
else:
    mean_reaction_time_combined = None
    std_reaction_time_combined = None

# Display the combined performance metrics
overall_performance = {
    'Total Stimuli': total_stimuli_combined,
    'Total Responses': total_responses_combined,
    'Total Correct Responses': total_correct_responses_combined,
    'Hit Rate': hit_rate_combined,
    'False Alarm Rate': false_alarm_rate_combined,
    'Mean Reaction Time': mean_reaction_time_combined,
    'STD Reaction Time': std_reaction_time_combined
}

overall_performance_df = pd.DataFrame([overall_performance])
print("Overall Performance Metrics:")
print(overall_performance_df)