# 1. Import libraries:
import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import copy
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
import json

# 2. define params and paths:
default_path = Path.cwd()
eeg_path = default_path / 'data' / 'eeg' / 'raw'
blocks_path = default_path / 'data' / 'params' / 'block_sequences'
response_mapping = {'1': 129, '65': 129,
                    '2': 130, '66': 130,
                    '3': 131, '67': 131,
                    '4': 132, '68': 132,
                    '5': 133, '69': 133,
                    '6': 134, '70': 134,
                    '8': 136, '72': 136,
                    '9': 137, '73': 137}
actual_mapping = {'New Segment/': 99999,
  'Stimulus/S  1': 1,
  'Stimulus/S  2': 2,
  'Stimulus/S  3': 3,
  'Stimulus/S  4': 4,
  'Stimulus/S  5': 5,
  'Stimulus/S  6': 6,
  'Stimulus/S  8': 8,
  'Stimulus/S  9': 9,
  'Stimulus/S 64': 64,
  'Stimulus/S 65': 65,
  'Stimulus/S 66': 66,
  'Stimulus/S 67': 67,
  'Stimulus/S 68': 68,
  'Stimulus/S 69': 69,
  'Stimulus/S 70': 70,
  'Stimulus/S 71': 71,
  'Stimulus/S 72': 72,
  'Stimulus/S 73': 73,
  'Stimulus/S129': 129,
  'Stimulus/S130': 130,
  'Stimulus/S131': 131,
  'Stimulus/S132': 132,
  'Stimulus/S133': 133,
  'Stimulus/S134': 134,
  'Stimulus/S136': 136,
  'Stimulus/S137': 137
                  }


# 3. create sub_list:
sub_list = []
for i in range(1, 30, 1):
    # .zfill(2):
    # Adds leading zeros to the string until its length is 2 characters.
    string = f'sub{str(i).zfill(2)}'
    if string in ['sub07', 'sub09', 'sub12']:
        continue
    else:
        sub_list.append(string)

# 4. extract eeg files:
def extract_eeg_files(condition=''):
    eeg_header_files = []
    for folders in eeg_path.iterdir():
        if condition in ['e1', 'e2'] and folders.name in ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub08']:
            continue
        else:
            for files in folders.iterdir():
                if files.is_file() and 'vhdr' in files.name:
                    if condition in files.name:
                        eeg_header_files.append(files)
    return eeg_header_files

# 5. load eeg files:
def load_eeg_files(eeg_header_files):
    eeg_list = []
    for files in eeg_header_files:
        eeg = mne.io.read_raw_brainvision(files, preload=True)
        eeg_list.append(eeg)
    return eeg_list


# 6. load block sequences: matching target stimuli in each eeg file
# csv files from params path:
blocks_dict = {}
for csv_files in blocks_path.iterdir():
    if csv_files.name.endswith('.csv'):
        blocks_dict[csv_files.name] = {}
        with open(csv_files, 'rb') as f:
            csv = pd.read_csv(f)
            blocks_dict[csv_files.name] = csv
            
# 7. filter csv based on condition:
a1_csv = {}
a2_csv = {}
e1_csv = {}
e2_csv = {}
for name, block in blocks_dict.items():
    a1_csv[name] = {}
    a2_csv[name] = {}
    e1_csv[name] = {}
    e2_csv[name] = {}
    a1_block = block[(block['block_seq'] == 's1') & (block['block_condition'] == 'azimuth')]
    a1_csv[name] = a1_block
    a2_block = block[(block['block_seq'] == 's2') & (block['block_condition'] == 'azimuth')]
    a2_csv[name] = a2_block
    e1_block = block[(block['block_seq'] == 's1') & (block['block_condition'] == 'elevation')]
    e1_csv[name] = e1_block
    e2_block = block[(block['block_seq'] == 's2') & (block['block_condition'] == 'elevation')]
    e2_csv[name] = e2_block

# 8. get EEG events:
def extract_events(csv, eeg_files):
    events_dict = {}
    eeg_index = 0
    events_isolated = {}
    for sub_name, block in csv.items():
        # Ensure the subject's sub-dictionary exists, and is not re-initialized repeatedly for each iteration
        if sub_name not in events_dict and sub_name not in events_isolated:
            events_dict[sub_name] = {}
            events_isolated[sub_name] = {}
        for condition_idx in range(len(block)):
            eeg_file = eeg_files[eeg_index]
            events, events_id = mne.events_from_annotations(eeg_file)
            eeg_index += 1
            events_dict[sub_name][f'{condition_idx}'] = {'events': events, 'events_id': events_id}
            events_isolated[sub_name][f'{condition_idx}'] = events
    return events_dict, events_isolated


# 9. clean events that are overlapping
def remove_overlaps(events_isolated):
    clean_events_dict = copy.deepcopy(events_isolated)  # Create a deep copy to avoid modifying the original
    for sub, items in clean_events_dict.items():
        for index, (idx, sub_dict) in enumerate(items.items()):
            indices_to_drop = []
            events = sub_dict
            if sub in ['sub01.csv', 'sub02.csv', 'sub03.csv', 'sub04.csv', 'sub05.csv', 'sub06.csv', 'sub08.csv']:
                times = events[:, 0] / 500
            else:
                times = events[:, 0] / 1000
            for time_idx, time in enumerate(times[:-1]):
                first_time = times[time_idx]
                next_time = times[time_idx + 1]
                time_diff = next_time - first_time
                if time_diff <= 0.2:
                    indices_to_drop.extend([time_idx, time_idx + 1])  # Add indices to drop
                    # drop overlapping events from each events object:
            events_cleaned = [event for index, event in enumerate(events) if index not in indices_to_drop]
            items[idx] = events_cleaned  # Update the dictionary with cleaned events
    return clean_events_dict


# 10. extract target numbers per block:
# 10a. create dictionary matching numbers from target and distractor stream:
matching_events = {'1': 65, '2': 66, '3': 67, '4': 68, '5': 69, '6': 70, '7': 71, '8': 72, '9': 73}
def extract_target_nums(csv):
    nums_dict = {}
    for csv_sub, csv_block in csv.items():
        nums_dict[csv_sub] = {}
        dict_index = 0 # This ensures indices start from 0 for each subject.
        for index, (idx, row) in enumerate(csv_block.iterrows()):
            target_number = row['Target Number']
            target_stream = row['block_seq']
            if target_stream == 's1':
                # get distractor number:
                distractor_num = [value for key, value in matching_events.items() if str(target_number) == key][0]
                nums_dict[csv_sub][dict_index] = {'Target': target_number, 'Distractor': distractor_num, 'Target Stream': target_stream}
                dict_index += 1
            elif target_stream == 's2':
                distractor_num = [value for key, value in matching_events.items() if target_number == int(key)][0]
                nums_dict[csv_sub][dict_index] = {'Target': target_number, 'Distractor': distractor_num, 'Target Stream': target_stream}
                dict_index += 1
    return nums_dict


# 11. separate events with and without response, as well ass invalid responses:
# a. Define function to categorize response timing
def categorize_events(nums_dict, events_dict):
    for sub in a1_nums_dict.keys():  # every sub (sub01-sub29 with exclusions)
        if sub in ['sub01.csv', 'sub02.csv', 'sub03.csv', 'sub04.csv', 'sub05.csv', 'sub06.csv', 'sub08.csv']:
            sfreq = 500
        else:
            sfreq = 1000
        blocks = a1_nums_dict[sub]
        for block in blocks.keys():  # each block processed (0-4)
            block_config = blocks[block] # each block has its own configurations
            target_num = block_config['Target']
            distractor_num = block_config['Distractor']
            target_stream = block_config['Target Stream']
            if target_stream == 's1':  # depending on the target stream, response number output changes
                response_num = response_mapping[str(target_num)]
            elif target_stream == 's2':
                response_num = response_mapping[str(target_num)]
            events = a1_events_dict[sub][str(block)]['events']  # extract events from specific sub, and sub-block
            event_onsets = [event[0] / sfreq for event in events]
            event_numbers = [event[2] for event in events]
            valid_window = (0.2, 0.9)

            valid_target_stim_indices_list = []
            valid_target_stim_indices = set()
            valid_resp_indices_list = []
            valid_resp_indices = set()
            early_target_stim_list = []
            early_target_stim_indices = set()
            early_target_resp_list = []
            early_target_resp_indices = set()
            delayed_target_stim_list = []
            delayed_target_stim_indices = set()
            delayed_target_resp_list = []
            delayed_target_resp_indices = set()
            target_no_response_stim_list = []
            target_no_response_indices = set()
            unclassified_responses_list = []
            unclassified_responses_indices = set()
            for stim_idx, (stim_num, stim_onset) in enumerate(zip(event_numbers, event_onsets)):
                if stim_num == target_num:
                    target_onset = stim_onset
                    response_found = False
                    for time_idx in range(stim_idx + 1, len(event_onsets)):
                        next_onset = event_onsets[time_idx]
                        next_num = event_numbers[time_idx]
                        if target_onset <= next_onset <= target_onset + 1.2 and next_num == response_num:
                            response_found = True
                            # response within time-window detected
                            if target_onset + valid_window[0] <= next_onset <= target_onset + valid_window[1]:
                                # valid response detected: add all indices into list
                                valid_target_stim_indices_list.append(stim_idx)
                                valid_resp_indices_list.append(time_idx)
                            elif target_onset <= next_onset <= target_onset + valid_window[0]:
                                # early response detected:
                                early_target_stim_list.append(stim_idx)
                                early_target_resp_list.append(time_idx)
                            elif target_onset + valid_window[1] <= next_onset <= target_onset + 1.2:
                                # delayed response detected:
                                delayed_target_stim_list.append(stim_idx)
                                delayed_target_resp_list.append(time_idx)
                            elif next_onset > target_onset + 1.2:
                                unclassified_responses_list.append(time_idx)
                elif response_found == False:
                    target_no_response_stim_list.append(stim_idx)
            valid_target_stim_indices.update(valid_target_stim_indices_list)
            valid_resp_indices.update(valid_resp_indices_list)
            early_target_stim_indices.update(early_target_stim_list)
            early_target_resp_indices.update(early_target_resp_list)
            delayed_target_stim_indices.update(delayed_target_stim_list)
            delayed_target_resp_indices.update(delayed_target_resp_list)
            target_no_response_indices.update(target_no_response_stim_list)
            unclassified_responses_indices.update(unclassified_responses_list)
            if len(unclassified_responses_indices) == 0:
                print(f'For block {block} no unclassified responses remain.')
            elif len(unclassified_responses_indices) > 0:
                print(f'For block {block} some responses are to be further analyzed. Checking')












if __name__ == "__main__":

    a1_eeg_header_files = extract_eeg_files(condition='a1')
    a2_eeg_header_files = extract_eeg_files(condition='a2')
    e1_eeg_header_files = extract_eeg_files(condition='e1')
    e2_eeg_header_files = extract_eeg_files(condition='e2')

    # for these I include sub01 to sub08
    a1_eeg = load_eeg_files(a1_eeg_header_files)
    a1_events_dict, a1_events_isolated = extract_events(a1_csv, a1_eeg)
    a1_nums_dict = extract_target_nums(a1_csv)
    # function for extracting diff event categories

    a1_clean_events = remove_overlaps(a1_events_isolated)
    # a1_target_events_dict, a1_distractor_events_dict, a1_non_targets_target_dict, \
    # a1_non_targets_distractor_dict, a1_animal_events_dict = isolate_events(a1_clean_events, a1_nums_dict, condition='a1')

    # for these I include sub01 to sub08
    a2_eeg = load_eeg_files(a2_eeg_header_files)
    a2_events_dict, a2_events_isolated = extract_events(a2_csv, a2_eeg)
    a2_clean_events = remove_overlaps(a2_events_isolated)
    a2_nums_dict = extract_target_nums(a2_csv)
    a2_target_events_dict, a2_distractor_events_dict, a2_non_targets_target_dict, \
    a2_non_targets_distractor_dict, a2_animal_events_dict = isolate_events(a2_clean_events, a2_nums_dict, condition='a2')

    e1_eeg = load_eeg_files(e1_eeg_header_files)
    e1_events_dict, e1_events_isolated = extract_events(e1_csv, e1_eeg)
    e1_clean_events = remove_overlaps(e1_events_isolated)
    e1_nums_dict = extract_target_nums(e1_csv)
    e1_target_events_dict, e1_distractor_events_dict, e1_non_targets_target_dict, \
    e1_non_targets_distractor_dict, e1_animal_events_dict = isolate_events(e1_clean_events, e1_nums_dict, condition='e1')


    e2_eeg = load_eeg_files(e2_eeg_header_files)
    e2_events_dict, e2_events_isolated = extract_events(e2_csv, e2_eeg)
    e2_clean_events = remove_overlaps(e2_events_isolated)
    e2_nums_dict = extract_target_nums(e2_csv)
    e2_target_events_dict, e2_distractor_events_dict, e2_non_targets_target_dict, \
    e2_non_targets_distractor_dict, e2_animal_events_dict = isolate_events(e2_clean_events, e2_nums_dict, condition='e2')












