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


# 3. create sub_list:
sub_list = []
for i in range(1, 30, 1):
    # .zfill(2):
    # Adds leading zeros to the string until its length is 2 characters.
    string = f'sub{str(i).zfill(2)}'
    if string in ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08', 'sub09', 'sub12']:
        continue
    else:
        sub_list.append(string)

# 4. extract eeg files:
def extract_eeg_files(condition=''):
    eeg_header_files = []
    for sub in sub_list:
        for folders in eeg_path.iterdir():
            for files in folders.iterdir():
                if files.is_file() and 'vhdr' in files.name and sub in files.name:
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



# 11. isolate specific events:
# a. target stimuli in target stream and in distractor stream respectively
# b. non-targets in target vs distractor stream


def isolate_events(clean_events, nums_dict, condition=''):
    target_events_dict = {}
    distractor_events_dict = {}
    non_targets_target_dict = {}
    non_targets_distractor_dict = {}
    animal_events_dict = {}
    for (nums_name, num_item), (name, item) in zip(nums_dict.items(), clean_events.items()):
        idx = 0
        target_events_dict[name] = {}
        distractor_events_dict[name] = {}
        non_targets_target_dict[name] = {}
        non_targets_distractor_dict[name] = {}
        animal_events_dict[name] = {}
        for (nums_index, nums_dict), (index, events_dict) in zip(num_item.items(), item.items()):
            block_events = events_dict
            target_num = nums_dict['Target']
            distractor_num = nums_dict['Distractor']
            target_stream = nums_dict['Target Stream']
            target_events = [events for events in block_events if target_num == events[2]]
            distractor_events = [events for events in block_events if distractor_num == events[2]]
            target_events_dict[name][idx] = np.array(target_events).tolist()
            distractor_events_dict[name][idx] = np.array(distractor_events).tolist()
            if target_stream == 's1':
                animal_sound = 71
                non_targets_target_events = [events for events in block_events if
                                             str(events[2]) in matching_events.keys() and events[2] != target_num]
                non_targets_target_dict[name][idx] = np.array(non_targets_target_events).tolist()
                non_targets_distractor_events = [events for events in block_events if events[2]
                                                 in matching_events.values() and events[2] != distractor_num and events[2] != animal_sound]
                non_targets_distractor_dict[name][idx] = np.array(non_targets_distractor_events).tolist()
                animal_events = [events for events in block_events if events[2] == animal_sound]
                animal_events_dict[name][idx] = np.array(animal_events).tolist()
                idx += 1
            elif target_stream == 's2':
                animal_sound = 7
                non_targets_target_events = [events for events in block_events if events[2] in matching_events.values()
                                             and events[2] != target_num and events[2] != animal_sound]
                non_targets_target_dict[name][idx] = np.array(non_targets_target_events).tolist()
                non_targets_distractor_events = [events for events in block_events if str(events[2]) in
                                                 matching_events.keys() and events[2] != distractor_num]
                non_targets_distractor_dict[name][idx] = np.array(non_targets_distractor_events).tolist()
                animal_events = [events for events in block_events if events[2] == animal_sound]
                animal_events_dict[name][idx] = np.array(animal_events).tolist()
                idx += 1
    # List of dictionaries and their corresponding filenames
    dictionaries = [
        (target_events_dict, "target_events_dict.txt"),
        (distractor_events_dict, "distractor_events_dict.txt"),
        (non_targets_target_dict, "non_targets_target_dict.txt"),
        (non_targets_distractor_dict, "non_targets_distractor_dict.txt"),
        (animal_events_dict, "animal_events_dict.txt"),
    ]

    # Save each dictionary as a JSON-formatted text file
    events_path = 'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/params/isolated_events/'
    for data, filename in dictionaries:
        condition_filename = f"{condition}_{filename}"
        with open(events_path + condition_filename, "w") as f:
            json.dump(data, f, indent=4)

    return target_events_dict, distractor_events_dict, non_targets_target_dict, non_targets_distractor_dict, animal_events_dict


if __name__ == "__main__":

    a1_eeg_header_files = extract_eeg_files(condition='a1')
    a2_eeg_header_files = extract_eeg_files(condition='a2')
    e1_eeg_header_files = extract_eeg_files(condition='e1')
    e2_eeg_header_files = extract_eeg_files(condition='e2')



    a1_eeg = load_eeg_files(a1_eeg_header_files)
    a1_events_dict, a1_events_isolated = extract_events(a1_csv, a1_eeg)
    a1_clean_events = remove_overlaps(a1_events_isolated)
    a1_nums_dict = extract_target_nums(a1_csv)
    a1_target_events_dict, a1_distractor_events_dict, a1_non_targets_target_dict, \
    a1_non_targets_distractor_dict, a1_animal_events_dict = isolate_events(a1_clean_events, a1_nums_dict, condition='a1')


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












