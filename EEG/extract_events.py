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
    if string in ['sub06', 'sub07', 'sub09', 'sub12']:
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
exceptions_csv = ['sub06.csv']
blocks_dict = {}
for csv_files in blocks_path.iterdir():
    if csv_files.name in exceptions_csv:
        continue
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
exceptions = ['sub06']
for name, block in blocks_dict.items():
    a1_csv[name] = {}
    a2_csv[name] = {}
    e1_csv[name] = {}
    e2_csv[name] = {}
    a1_block = block[(block['block_seq'] == 's1') & (block['block_condition'] == 'azimuth')]
    a1_csv[name] = a1_block
    a2_block = block[(block['block_seq'] == 's2') & (block['block_condition'] == 'azimuth')]
    a2_csv[name] = a2_block
    if name in exceptions:
        continue
    e1_block = block[(block['block_seq'] == 's1') & (block['block_condition'] == 'elevation')]
    e1_csv[name] = e1_block
    e2_block = block[(block['block_seq'] == 's2') & (block['block_condition'] == 'elevation')]
    e2_csv[name] = e2_block
    # Remove empty entries from e1_csv and e2_csv
    e1_csv = {name: data for name, data in e1_csv.items() if name not in exceptions}
    e2_csv = {name: data for name, data in e2_csv.items() if name not in exceptions}


# 8. get EEG events:
def extract_events(csv, eeg_files, condition=''):
    elevation_exceptions = ['sub01.csv', 'sub02.csv', 'sub03.csv', 'sub04.csv', 'sub05.csv', 'sub08.csv']
    events_dict = {}
    eeg_index = 0
    events_isolated = {}
    for sub_name, block in csv.items():
        if condition == 'e1' or condition == 'e2':
            if sub_name in elevation_exceptions:
                continue
        # Ensure the subject's sub-dictionary exists, and is not re-initialized repeatedly for each iteration
        if sub_name not in events_dict and sub_name not in events_isolated:
            events_dict[sub_name] = {}
            events_isolated[sub_name] = {}
        for condition_idx in range(len(block)):
            if eeg_index >= len(eeg_files):  # Check to prevent out-of-range error
                print(f"Warning: eeg_index {eeg_index} exceeds available EEG files ({len(eeg_files)}).")
                break
            eeg_file = eeg_files[eeg_index]
            events, events_id = mne.events_from_annotations(eeg_file)
            eeg_index += 1
            events_dict[sub_name][f'{condition_idx}'] = {'events': events, 'events_id': events_id}
            events_isolated[sub_name][f'{condition_idx}'] = events
    return events_dict, events_isolated


# 9. extract target numbers per block:
# 9a. create dictionary matching numbers from target and distractor stream:
matching_events = {'1': 65, '2': 66, '3': 67, '4': 68, '5': 69, '6': 70, '7': 71, '8': 72, '9': 73}
def extract_target_nums(csv):
    nums_dict = {}
    for csv_sub, csv_block in csv.items():
        nums_dict[csv_sub] = {}
        dict_index = 0  # This ensures indices start from 0 for each subject.
        for index, (idx, row) in enumerate(csv_block.iterrows()):
            target_number = row['Target Number']
            target_stream = row['block_seq']
            if target_stream == 's1':
                # get distractor number:
                distractor_num = [value for key, value in matching_events.items() if str(target_number) == key][0]
                nums_dict[csv_sub][dict_index] = {'Target': target_number, 'Distractor': distractor_num, 'Target Stream': target_stream}
                dict_index += 1
            elif target_stream == 's2':
                target_number = [value for key, value in matching_events.items() if int(key) == target_number][0]
                distractor_num = [int(key) for key, value in matching_events.items() if target_number == value][0]
                nums_dict[csv_sub][dict_index] = {'Target': target_number, 'Distractor': distractor_num, 'Target Stream': target_stream}
                dict_index += 1
    return nums_dict


# 10. separate events with and without response, as well ass invalid responses:
# a. Define function to categorize response timing
def categorize_events(nums_dict, events_dict, condition=''):
    results_dict = {}  # Dictionary to store results
    all_events_dict = {}
    for sub in nums_dict.keys():  # every sub (sub01-sub29 with exclusions)
        if condition == 'e1' or condition == 'e2':
            if sub in ['sub01.csv', 'sub02.csv', 'sub03.csv', 'sub04.csv', 'sub05.csv', 'sub08.csv']:
                continue
        results_dict[sub] = {}
        all_events_dict[sub] = {}
        if sub in ['sub01.csv', 'sub02.csv', 'sub03.csv', 'sub04.csv', 'sub05.csv', 'sub08.csv']:
            sfreq = 500
        else:
            sfreq = 1000
        blocks = nums_dict[sub]
        for block in blocks.keys():  # each block processed (0-4)
            block_config = blocks[block]  # each block has its own configurations
            target_num = block_config['Target']
            distractor_num = block_config['Distractor']
            target_stream = block_config['Target Stream']
            if target_stream == 's1':  # depending on the target stream, response number output changes
                response_num = response_mapping[str(target_num)]
            elif target_stream == 's2':
                response_num = response_mapping[str(target_num)]
            events = events_dict[sub][str(block)]['events']  # extract events from specific sub, and sub-block
            response_events = [event for event in events if event[2] == response_num]
            response_indices = [index for index, event in enumerate(events) if event[2] == response_num]
            target_events = [event for event in events if event[2] == target_num]
            target_indices = [index for index, event in enumerate(events) if event[2] == target_num]
            distractor_events = [event for event in events if event[2] == distractor_num]
            distractor_indices = [index for index, event in enumerate(events) if event[2] == distractor_num]
            if target_stream == 's1':
                animal_sounds = [event for event in events if event[2] == 71]
                animal_sounds_indices = [index for index, event in enumerate(events) if event[2] == 71]
                non_targets_target = [event for event in events if str(event[2]) in list(matching_events.keys()) and event[2] != target_num]
                non_targets_target_indices = [index for index, event in enumerate(events) if str(event[2]) in list(matching_events.keys())]
                non_targets_distractor = [event for event in events if event[2] in matching_events.values() and event[2] != distractor_num]
                non_targets_distractor_indices = [index for index, event in enumerate(events) if event[2] in matching_events.values()]
            elif target_stream == 's2':
                animal_sounds = [event for event in events if event[2] == 7]
                animal_sounds_indices = [index for index, event in enumerate(events) if event[2] == 7]
                non_targets_target = [event for event in events if event[2] in matching_events.values() and event[2] != target_num]
                non_targets_target_indices = [index for index, event in enumerate(events) if event[2] in matching_events.values()]
                non_targets_distractor = [event for event in events if str(event[2]) in list(matching_events.keys()) and event[2] != distractor_num]
                non_targets_distractor_indices = [index for index, event in enumerate(events) if str(event[2]) in list(matching_events.keys())]
            event_onsets = [event[0] / sfreq for event in events]
            event_numbers = [event[2] for event in events]
            valid_window = (0.2, 0.9)
            # Initialize result storage for this block
            all_events_results = {'target': target_events,
                                  'target_indices': target_indices,
                                  'distractor': distractor_events,
                                  'distractor_indices': distractor_indices,
                                  'non_targets_target': non_targets_target,
                                  'non_targets_target_indices': non_targets_target_indices,
                                  'non_targets_distractor': non_targets_distractor,
                                  'non_targets_distractor_indices': non_targets_distractor_indices,
                                  'response': response_events,
                                  'response_indices': response_indices,
                                  'animal_sounds': animal_sounds,
                                  'animal_sounds_indices': animal_sounds_indices}
            block_results = {
                # Target Responses
                "valid_target_stim": set(),
                "valid_resp": set(),
                "early_target_stim": set(),
                "early_target_resp": set(),
                "delayed_target_stim": set(),
                "delayed_target_resp": set(),
                "target_no_response": set(),

                # Distractor Responses
                "valid_distractor_stim": set(),
                "valid_distractor_resp": set(),
                "early_distractor_stim": set(),
                "early_distractor_resp": set(),
                "delayed_distractor_stim": set(),
                "delayed_distractor_resp": set(),
                "distractor_no_response": set(),

                # Non-Target Responses
                "valid_non_target_stim": set(),
                "valid_non_target_resp": set(),
                "early_non_target_stim": set(),
                "early_non_target_resp": set(),
                "delayed_non_target_stim": set(),
                "delayed_non_target_resp": set(),
                "non_target_no_response": set(),

                # Responses that remain unclassified
                "unclassified_responses": set(response_indices),
                'unclassified_target_stimuli': set(target_indices),
                'unclassified_distractor_stimuli': set(distractor_indices),
                'unclassified_non_targets_target_stimuli': set(non_targets_target_indices),
                'unclassified_non_targets_distractor_stimuli': set(non_targets_distractor_indices)
            }
            # Helper function to categorize responses
            def classify_responses(stimulus_type, goal=''):
                stim_indices_list, resp_indices_list = [], []
                early_stim_list, early_resp_list = [], []
                delayed_stim_list, delayed_resp_list = [] , []
                no_response_stim_list = []

                for stim_idx, (stim_num, stim_onset) in enumerate(zip(event_numbers, event_onsets)):
                    if ((stimulus_type == "target" and stim_num == target_num) or
                        (stimulus_type == "distractor" and stim_num == distractor_num) or
                        (stimulus_type == "non-target" and stim_num not in {target_num, distractor_num,
                                                                            response_num})):
                        stimulus_onset = stim_onset
                        response_found = False

                        for time_idx in range(stim_idx + 1, len(event_onsets)):
                            next_onset = event_onsets[time_idx]
                            next_num = event_numbers[time_idx]
                            # Check if response is already classified
                            if time_idx not in block_results["unclassified_responses"]:
                                continue  # Skip this response if it's already classified

                            if stimulus_onset <= next_onset <= stimulus_onset + 1.2 and next_num == response_num:
                                response_found = True

                                if stimulus_onset + valid_window[0] <= next_onset <= stimulus_onset + valid_window[1]:
                                    # Valid response
                                    stim_indices_list.append(stim_idx)
                                    resp_indices_list.append(time_idx)
                                elif stimulus_onset <= next_onset < stimulus_onset + valid_window[0]:
                                    # Early response
                                    early_stim_list.append(stim_idx)
                                    early_resp_list.append(time_idx)
                                elif stimulus_onset + valid_window[1] < next_onset <= stimulus_onset + 1.2:
                                    # Delayed response
                                    delayed_stim_list.append(stim_idx)
                                    delayed_resp_list.append(time_idx)

                                # Remove classified responses from unclassified set
                                # Remove classified responses and stimuli
                                if time_idx in block_results["unclassified_responses"]:
                                    block_results["unclassified_responses"].discard(time_idx)
                                if goal == 'target':
                                    block_results["unclassified_target_stimuli"].discard(stim_idx)
                                elif goal == 'distractor':
                                    block_results["unclassified_distractor_stimuli"].discard(stim_idx)
                                elif goal == 'non_target':
                                    block_results["unclassified_non_targets_target_stimuli"].discard(stim_idx)
                                    block_results["unclassified_non_targets_distractor_stimuli"].discard(stim_idx)

                            elif next_onset > stimulus_onset + 1.2:
                                break

                        if not response_found:
                            no_response_stim_list.append(stim_idx)

                return stim_indices_list, resp_indices_list, early_stim_list, early_resp_list, delayed_stim_list, delayed_resp_list, no_response_stim_list
            # **Step 1: Classify Target Stimuli Responses**
            (
                block_results["valid_target_stim"],
                block_results["valid_resp"],
                block_results["early_target_stim"],
                block_results["early_target_resp"],
                block_results["delayed_target_stim"],
                block_results["delayed_target_resp"],
                block_results["target_no_response"]
            ) = map(set, classify_responses("target", goal='target'))
            block_results["unclassified_responses"].difference_update(block_results['valid_resp'])
            block_results["unclassified_responses"].difference_update(block_results['early_target_resp'])
            block_results["unclassified_responses"].difference_update(block_results['delayed_target_resp'])

            # Check for unclassified responses
            if len(block_results["unclassified_responses"]) > 0:
                print(f'For {sub} block {block}, checking distractor stimuli...')

                # **Step 2: Classify Distractor Stimuli Responses**
                (
                    block_results['valid_distractor_stim'],
                    block_results['valid_distractor_resp'],
                    block_results['early_distractor_stim'],
                    block_results['early_distractor_resp'],
                    block_results['delayed_distractor_stim'],
                    block_results['delayed_distractor_resp'],
                    block_results['distractor_no_response']) = map(set, classify_responses("distractor", goal='distractor'))

                block_results["unclassified_responses"].difference_update(block_results['valid_distractor_resp'])
                block_results["unclassified_responses"].difference_update(block_results['early_distractor_resp'])
                block_results["unclassified_responses"].difference_update(block_results['delayed_distractor_resp'])
                block_results["unclassified_distractor_stimuli"].difference_update(block_results['valid_distractor_stim'])
                block_results["unclassified_distractor_stimuli"].difference_update(block_results['early_distractor_stim'])
                block_results["unclassified_distractor_stimuli"].difference_update(block_results['delayed_distractor_stim'])

            # Check for remaining unclassified responses
            if len(block_results["unclassified_responses"]) > 0:
                print(f'For {sub} block {block}, checking non-target stimuli...')

                # **Step 3: Classify Non-Target Stimuli Responses**
                (
                    block_results['valid_non_target_stim'],
                    block_results['valid_non_target_resp'],
                    block_results['early_non_target_stim'],
                    block_results['early_non_target_resp'],
                    block_results['delayed_non_target_stim'],
                    block_results['delayed_non_target_resp'],
                    block_results['non_target_no_response']
                ) = map(set, classify_responses("non-target", goal='non_target'))

                block_results["unclassified_responses"].difference_update(block_results['valid_non_target_resp'])
                block_results["unclassified_responses"].difference_update(block_results['early_non_target_resp'])
                block_results["unclassified_responses"].difference_update(block_results['delayed_non_target_resp'])
                block_results["unclassified_non_targets_target_stimuli"].difference_update(block_results['valid_non_target_stim'])
                block_results["unclassified_non_targets_distractor_stimuli"].difference_update(block_results['valid_non_target_stim'])
                block_results["unclassified_non_targets_target_stimuli"].difference_update(block_results['early_non_target_stim'])
                block_results["unclassified_non_targets_distractor_stimuli"].difference_update(block_results['early_non_target_stim'])
                block_results["unclassified_non_targets_target_stimuli"].difference_update(block_results['delayed_non_target_stim'])
                block_results["unclassified_non_targets_distractor_stimuli"].difference_update(block_results['delayed_non_target_stim'])

            # Final check
            if len(block_results["unclassified_distractor_stimuli"]) > 0:
                if "distractor_no_response" in block_results:
                    block_results['distractor_no_response'].update(block_results["unclassified_distractor_stimuli"])
                else:
                    block_results['distractor_no_response'] = block_results["unclassified_distractor_stimuli"]

            if len(block_results["unclassified_non_targets_target_stimuli"]) > 0:
                if "non_target_no_response" in block_results:
                    block_results['non_target_no_response'].update(
                        block_results["unclassified_non_targets_target_stimuli"])
                else:
                    block_results['non_target_no_response'] = block_results[
                        "unclassified_non_targets_target_stimuli"]
            if len(block_results["unclassified_non_targets_distractor_stimuli"]) > 0:
                if "non_target_no_response" in block_results:
                    block_results['non_target_no_response'].update(
                        block_results["unclassified_non_targets_distractor_stimuli"])
                else:
                    block_results['non_target_no_response'] = block_results[
                        "unclassified_non_targets_distractor_stimuli"]

            if len(block_results["unclassified_responses"]) == 0:
                print(f'For {sub} block {block}, all responses classified.')
            else:
                print(f'For {sub} block {block}, still unclassified responses exist!')

            # Store block results in the subject's dictionary
            results_dict[sub][block] = block_results
            all_events_dict[sub][block] = all_events_results
    return results_dict, all_events_dict


# 11. group events based on different conditions (no response, early or delayed response, or valid)
def group_categorized_events(results_dict, all_events_dict):
    grouped_events = {}  # Dictionary to store categorized events

    for sub in results_dict.keys():  # Loop through subjects
        grouped_events[sub] = {}

        for block in results_dict[sub].keys():  # Loop through blocks
            block_results = results_dict[sub][block]  # Get categorized results
            block_all_events = all_events_dict[sub][block]
            # Initialize event groups
            block_events = {
                "targets_with_valid_responses": [],
                "targets_with_early_responses": [],
                "targets_with_delayed_responses": [],
                "targets_without_responses": [],

                "distractors_with_valid_responses": [],
                "distractors_with_early_responses": [],
                "distractors_with_delayed_responses": [],
                "distractors_without_responses": [],

                "valid_non_targets_target_stim": [],
                'valid_non_targets_target_resp': [],
                'early_non_targets_target_stim': [],
                'early_non_targets_target_resp': [],
                'delayed_non_targets_target_stim': [],
                'delayed_non_targets_target_resp': [],
                'non_targets_target_no_response': [],

                "valid_non_targets_distractor_stim": [],
                'valid_non_targets_distractor_resp': [],
                'early_non_targets_distractor_stim': [],
                'early_non_targets_distractor_resp': [],
                'delayed_non_targets_distractor_stim': [],
                'delayed_non_targets_distractor_resp': [],
                'non_targets_distractor_no_response': [],

                'animal_sounds': [],
                'unclassified_responses': []
            }

            for event_idx, event in zip(block_all_events['animal_sounds_indices'], block_all_events['animal_sounds']):
                block_events['animal_sounds'].append(event)

            for event_idx, event in zip(block_all_events['target_indices'], block_all_events["target"]):  # Iterate through target events
                # **TARGETS**
                if event_idx in block_results["valid_target_stim"]:
                    block_events["targets_with_valid_responses"].append(event)
                elif event_idx in block_results["early_target_stim"]:
                    block_events["targets_with_early_responses"].append(event)
                elif event_idx in block_results["delayed_target_stim"]:
                    block_events["targets_with_delayed_responses"].append(event)
                elif event_idx in block_results["target_no_response"]:
                    block_events["targets_without_responses"].append(event)

            # **Handle Distractor Events**
            classified_distractor_indices = set()  # Track classified distractors
            for event_idx, event in zip(block_all_events['distractor_indices'], block_all_events["distractor"]):  # Iterate through target events
                # **DISTRACTORS**
                if event_idx in block_results["valid_distractor_stim"]:
                    block_events["distractors_with_valid_responses"].append(event)
                    classified_distractor_indices.add(event_idx)
                elif event_idx in block_results["early_distractor_stim"]:
                    block_events["distractors_with_early_responses"].append(event)
                    classified_distractor_indices.add(event_idx)
                elif event_idx in block_results["delayed_distractor_stim"]:
                    block_events["distractors_with_delayed_responses"].append(event)
                    classified_distractor_indices.add(event_idx)
                elif event_idx in block_results["distractor_no_response"]:
                    block_events["distractors_without_responses"].append(event)
                    classified_distractor_indices.add(event_idx)

            # **NON-TARGETS**
            # non-target target stream
            classified_non_targets_target_indices = set()  # Track classified distractors
            for event_idx, event in zip(block_all_events['non_targets_target_indices'], block_all_events["non_targets_target"]): 
                # Iterate through target events
                if event_idx in block_results["valid_non_target_stim"]:
                    block_events["valid_non_targets_target_stim"].append(event)
                    classified_non_targets_target_indices.add(event_idx)
                elif event_idx in block_results["early_non_target_stim"]:
                    block_events["early_non_targets_target_stim"].append(event)
                    classified_non_targets_target_indices.add(event_idx)
                elif event_idx in block_results["delayed_non_target_stim"]:
                    block_events["delayed_non_targets_target_stim"].append(event)
                    classified_non_targets_target_indices.add(event_idx)
                elif event_idx in block_results["non_target_no_response"]:
                    block_events["non_targets_target_no_response"].append(event)
                    classified_non_targets_target_indices.add(event_idx)

            # non-targets distractor stream:
            classified_non_targets_distractor_indices = set()  # Track classified distractors
            for event_idx, event in zip(block_all_events['non_targets_distractor_indices'], block_all_events["non_targets_distractor"]):
                if event[2] == 7 or event[2] == 71:
                    continue
                # Iterate through target events
                if event_idx in block_results["valid_non_target_stim"]:
                    block_events["valid_non_targets_distractor_stim"].append(event)
                    classified_non_targets_distractor_indices.add(event_idx)
                elif event_idx in block_results["early_non_target_stim"]:
                    block_events["early_non_targets_distractor_stim"].append(event)
                    classified_non_targets_distractor_indices.add(event_idx)
                elif event_idx in block_results["delayed_non_target_stim"]:
                    block_events["delayed_non_targets_distractor_stim"].append(event)
                    classified_non_targets_distractor_indices.add(event_idx)
                elif event_idx in block_results["non_target_no_response"]:
                    block_events["non_targets_distractor_no_response"].append(event)
                    classified_non_targets_distractor_indices.add(event_idx)

            grouped_events[sub][block] = block_events  # Store categorized events for this block
    return grouped_events


# 13. sort specific events: #todo


# 14. remove overlapping stimuli events (within time_threshold)
def remove_overlaps(grouped_events, time_threshold_backward=0.2, time_threshold_forward=0.4):
    grouped_events_filtered = copy.deepcopy(grouped_events)
    for sub in grouped_events_filtered.keys():  #
        if sub in exceptions:
            sfreq = 500
        else:
            sfreq = 1000
        sub_dict = grouped_events_filtered[sub]
        for block in sub_dict:
            block_dict = sub_dict[block]

            event_times = []
            for list_name, event_list in block_dict.items():
                if len(event_list) > 0:
                    for idx, event in enumerate(event_list):
                        event_time = event[0] / sfreq  # Convert to seconds
                        event_times.append((event_time, list_name, idx))
            # Sort by event time
            event_times.sort()
            # Find overlapping events
            to_remove = set()  # Store (list_name, index) of events to remove
            for i in range(1, len(event_times) - 1):
                time1, list1, idx1 = event_times[i]
                time2, list2, idx2 = event_times[i + 1]
                time3, list3, idx3 = event_times[i - 1]
                time_diff_forward = time2 - time1
                time_diff_backwards = time1 - time3
                if 'distractor' not in list1 and 'animal' not in list1:
                    before_match = ('distractor' in list3 or 'animal' in list3) and time_diff_backwards <= time_threshold_backward
                    after_match = ('distractor' in list2 or 'animal' in list2) and time_diff_forward <= time_threshold_forward
                    if before_match and after_match:
                        # Both before and after → Remove target and both distractors
                        to_remove.add((list1, idx1))
                        to_remove.add((list2, idx2))
                        to_remove.add((list3, idx3))
                    elif before_match:
                        # Only before → Remove target and previous distractor
                        to_remove.add((list1, idx1))
                        to_remove.add((list3, idx3))
                    elif after_match:
                        # Only after → Remove target and next distractor
                        to_remove.add((list1, idx1))
                        to_remove.add((list2, idx2))
            # Remove overlapping events correctly
            for list_name in block_dict.keys():
                block_dict[list_name] = [event for i, event in enumerate(block_dict[list_name]) if (list_name, i) not in to_remove]
    return grouped_events_filtered


def save_events(grouped_events_filtered, condition=''):
    for sub in grouped_events_filtered.keys():
        sub_name = sub[:-4]
        sub_blocks = grouped_events_filtered[sub]
        for block_index in sub_blocks:
            block = sub_blocks[block_index]
            for events_name, event_array in block.items():
                if events_name not in {'unclassified_responses'}:
                    if len(event_array) > 0:
                        # Convert numpy arrays to lists
                        event_array_serializable = [event.tolist() for event in event_array]
                        file_path = default_path / 'data' / 'params' / 'isolated_events' / sub_name / condition
                        os.makedirs(file_path, exist_ok=True)
                        with open(file_path / f'{sub_name}_{condition}_{events_name}_{block_index}.json', 'w') as f:
                            json.dump(event_array_serializable, f)


if __name__ == "__main__":

    a1_eeg_header_files = extract_eeg_files(condition='a1')
    a2_eeg_header_files = extract_eeg_files(condition='a2')
    e1_eeg_header_files = extract_eeg_files(condition='e1')
    e2_eeg_header_files = extract_eeg_files(condition='e2')

    # for these I include sub01 to sub08
    a1_eeg = load_eeg_files(a1_eeg_header_files)
    a1_events_dict, a1_events_isolated = extract_events(a1_csv, a1_eeg, condition='a1')
    a1_nums_dict = extract_target_nums(a1_csv)
    a1_results_dict, a1_all_events_dict = categorize_events(a1_nums_dict, a1_events_dict, condition='a1')
    a1_grouped_events = group_categorized_events(a1_results_dict, a1_all_events_dict)
    a1_events_filtered = remove_overlaps(a1_grouped_events, time_threshold_backward=0.2, time_threshold_forward=0.5)
    save_events(a1_events_filtered, condition='a1')

    # for these I include sub01 to sub08
    a2_eeg = load_eeg_files(a2_eeg_header_files)
    a2_events_dict, a2_events_isolated = extract_events(a2_csv, a2_eeg, condition='a2')
    a2_nums_dict = extract_target_nums(a2_csv)
    a2_results_dict, a2_all_events_dict = categorize_events(a2_nums_dict, a2_events_dict, condition='a2')
    a2_grouped_events = group_categorized_events(a2_results_dict, a2_all_events_dict)
    a2_events_filtered = remove_overlaps(a2_grouped_events, time_threshold_backward=0.2, time_threshold_forward=0.4)
    save_events(a2_events_filtered, condition='a2')

    e1_eeg = load_eeg_files(e1_eeg_header_files)
    e1_events_dict, e1_events_isolated = extract_events(e1_csv, e1_eeg, condition='e1')
    e1_nums_dict = extract_target_nums(e1_csv)
    e1_results_dict, e1_all_events_dict = categorize_events(e1_nums_dict, e1_events_dict, condition='e1')
    e1_grouped_events = group_categorized_events(e1_results_dict, e1_all_events_dict)
    e1_events_filtered = remove_overlaps(e1_grouped_events, time_threshold_backward=0.2, time_threshold_forward=0.4)

    e2_eeg = load_eeg_files(e2_eeg_header_files)
    e2_events_dict, e2_events_isolated = extract_events(e2_csv, e2_eeg, condition='e2')
    e2_nums_dict = extract_target_nums(e2_csv)
    e2_results_dict, e2_all_events_dict = categorize_events(e2_nums_dict, e2_events_dict, condition='e2')
    e2_grouped_events = group_categorized_events(e2_results_dict, e2_all_events_dict)
    e2_events_filtered = remove_overlaps(e2_grouped_events, time_threshold_backward=0.2, time_threshold_forward=0.4)

    save_events(e2_events_filtered, condition='e2')












