import mne
import pandas as pd
import numpy as np
from pathlib import Path
import os

sub_input = input("Give sub number as subn (n for number): ")
sub = [sub.strip() for sub in sub_input.split(',')]
cm = 1 / 2.54
name = sub_input
# 0. LOAD THE DATA
default_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data')
raw_dir = default_dir / 'eeg' / 'raw'
sub_dir = raw_dir / name
json_path = default_dir / 'misc'
fig_path = default_dir / 'eeg' / 'preprocessed' / 'results' / name / 'figures'
results_path = default_dir / 'eeg' / 'preprocessed' / 'results' / name
psd_path = results_path / 'psd'
epochs_folder = results_path / "epochs"
evokeds_folder = results_path / 'evokeds'
raw_fif = sub_dir / 'raw files'
tables_path = results_path / 'tables'
for folder in sub_dir, fig_path, results_path, epochs_folder, evokeds_folder, raw_fif, psd_path, tables_path:
    if not os.path.isdir(folder):
        os.makedirs(folder)

# events:
markers_dict = {
    's1_events': {  # Stimulus 1 markers
        'Stimulus/S  1': 1,
        'Stimulus/S  2': 2,
        'Stimulus/S  3': 3,
        'Stimulus/S  4': 4,
        'Stimulus/S  5': 5,
        'Stimulus/S  6': 6,
        'Stimulus/S  8': 8,
        'Stimulus/S  9': 9
    },
    's2_events': {  # Stimulus 2 markers
        'Stimulus/S 65': 65,
        'Stimulus/S 66': 66,
        'Stimulus/S 67': 67,
        'Stimulus/S 68': 68,
        'Stimulus/S 69': 69,
        'Stimulus/S 70': 70,
        'Stimulus/S 72': 72,
        'Stimulus/S 73': 73
    },
    'response_events': {  # Response markers
        'Stimulus/S129': 129,
        'Stimulus/S130': 130,
        'Stimulus/S131': 131,
        'Stimulus/S132': 132,
        'Stimulus/S133': 133,
        'Stimulus/S134': 134,
        'Stimulus/S136': 136,
        'Stimulus/S137': 137
    }
}
s1_events = markers_dict['s1_events']
s2_events = markers_dict['s2_events']  # stimulus 2 markers
response_events = markers_dict['response_events']  # response markers

# Run pre-processing steps:
''' 4 conditions:
    - a1: azimuth, s1 target
    - a2: azimuth, s2 target
    - e1: elevation, s1 target
    - e2: elevation, s2 target '''
condition = input('Please provide condition (exp. EEG): ')

animal_block_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/params/animal_blocks')
for files in animal_block_path.iterdir():
    if name in files.name:
        animal_block = pd.read_csv(files)

# determine which marker represents the animal sound in selected condition:
animal_stimuli_events = {'Stimulus/S  7': 7, 'Stimulus/S 71': 71}
if condition == 'a1' or condition == 'e1':
    # Select the second pair ('Stimulus/S 71': 7)
    selected_animal_event = {'Stimulus/S 71': animal_stimuli_events['Stimulus/S 71']}
    s2_events = {**s2_events, **selected_animal_event}
elif condition == 'a2' or condition == 'e2':
    # Select the first pair ('Stimulus/S  7': 7)
    selected_animal_event = {'Stimulus/S  7': animal_stimuli_events['Stimulus/S  7']}
    s1_events = {**s1_events, **selected_animal_event}

all_events = {**s1_events, **s2_events, **response_events, **selected_animal_event}

# Read all dataframes:
target_nums_df = pd.read_csv(tables_path / f'{sub_input}_{condition}_target_nums_events_raw.csv')
target_nums_df = target_nums_df.iloc[:, 1:3]

distractor_nums_df = pd.read_csv(tables_path / f'{sub_input}_{condition}_distractor_nums_events_raw_raw.csv')
distractor_nums_df = distractor_nums_df.iloc[:, 1:3]

non_targets_target_df = pd.read_csv(tables_path / f'{sub_input}_{condition}_target_non_targets_events_raw.csv')
non_targets_target_df = non_targets_target_df.iloc[:, 1:3]

non_targets_distractor_df = pd.read_csv(tables_path / f'{sub_input}_{condition}_distractor_non_targets_events_raw.csv')
non_targets_distractor_df = non_targets_distractor_df.iloc[:, 1:3]

deviant_sounds_df = pd.read_csv(tables_path / f'{sub_input}_{condition}_deviant_sounds_events_raw.csv')
deviant_sounds_df = deviant_sounds_df.iloc[:, 1:3]

# load EEG file:
raw_clean = mne.io.read_raw_fif(results_path / f'{name}_{condition}_preproccessed-raw.fif', preload=True)

# Extract events from annotations
raw_clean_events, raw_clean_event_ids = mne.events_from_annotations(raw_clean)

# Update event IDs
for i in range(len(raw_clean_events)):
    description = list(raw_clean_event_ids.keys())[list(raw_clean_event_ids.values()).index(raw_clean_events[i, 2])]
    if description in all_events:
        raw_clean_events[i, 2] = all_events[description]

# Update event dictionary to match the predefined dictionary
updated_event_ids = {k: v for k, v in all_events.items() if k in raw_clean_event_ids}

# Verify the updated events
print("Updated Events:\n", raw_clean_events)
print("Updated Event IDs:\n", updated_event_ids)
# Define the minimum time difference (in samples) between consecutive events
sfreq = raw_clean.info['sfreq']  # Sampling frequency
min_time_diff = 0.3  # 100ms
min_samples_diff = int(min_time_diff * sfreq)

# Initialize a list to store valid events
valid_events = []

# Iterate through the events and keep only those that are not within 100ms
for i in range(len(raw_clean_events)):
    if i == 0:  # Always keep the first event
        valid_events.append(raw_clean_events[i])
    else:
        # Check time difference between the current and the previous valid event
        if raw_clean_events[i, 0] - valid_events[-1][0] > min_samples_diff:
            valid_events.append(raw_clean_events[i])

# Convert valid events back to a NumPy array
valid_events = np.array(valid_events)
# Extract target and distractor events with isin
target_mask = np.isin(valid_events[:, 2], range(1, 10))
distractor_mask = np.isin(valid_events[:, 2], range(65, 76))

target_events = valid_events[target_mask]
distractor_events = valid_events[distractor_mask]
# Debugging: Print the number of events before and after filtering
print(f"Original number of events: {len(raw_clean_events)}")
print(f"Number of valid events after filtering: {len(valid_events)}")


target_epochs = mne.Epochs(raw_clean, target_events, s1_events, tmin=-0.2, tmax=0.9, baseline=(-0.2, 0), preload=True)
distractor_epochs = mne.Epochs(raw_clean, distractor_events, s2_events, tmin=-0.2, tmax=0.9, baseline=(-0.2, 0), preload=True)
# check ERP:
target_erp = target_epochs.average()
distractor_erp = distractor_epochs.average()
#
# # compute ERPs
target_erp.plot(picks=['F1'], titles='Target ERP - Key Channels')
distractor_erp.plot(picks=['F1'], titles='Distractor ERP - Key Channels')


# use the different dataframes to get ERP plots for:
# 1: target num in target vs distractor stream
# 2: distractor num vs deviant sounds
# 3: non target s1 vs non target s2
def transform_events(df, mapping):
    events = df.iloc[:, :2]
    events = np.array(events)
    events = np.hstack((
        events[:, [0]],  # First column (timepoints)
        np.zeros((events.shape[0], 1), dtype=int),  # New column of zeros
        events[:, [1]]  # Second column (event IDs)
    ))
    # Get unique event IDs from target_nums_events
    valid_event_ids = np.unique(events[:, 2])  # Extract the third column (event IDs)

    # Filter s1_events to include only the valid event IDs
    filtered_events = {k: v for k, v in mapping.items() if v in valid_event_ids}

    # Debugging: Print filtered event IDs
    print("Filtered Event IDs:", filtered_events)
    return events, filtered_events

target_nums_events, target_nums_mapping = transform_events(target_nums_df, mapping=s1_events)
distractor_nums_events, distractor_nums_mapping = transform_events(distractor_nums_df, mapping=s2_events)

non_targets_target_events, non_targets_target_mapping = transform_events(non_targets_target_df, mapping=s1_events)
non_targets_distractor_events, non_targets_distractor_mapping = transform_events(non_targets_distractor_df, mapping=s2_events)

deviant_sounds_events, animals_mapping = transform_events(deviant_sounds_df, mapping=selected_animal_event)

# create different epochs and ERPs:
target_nums_epochs = mne.Epochs(raw_clean, target_nums_events, target_nums_mapping, tmin=-0.2, tmax=0.9, baseline=(-0.2, 0), preload=True)
distractor_nums_epochs = mne.Epochs(raw_clean, distractor_nums_events, distractor_nums_mapping, tmin=-0.2, tmax=0.9, baseline=(-0.2, 0), preload=True)

target_nums_erp = target_nums_epochs.average().plot(picks=['F1'], titles='Target Stream - Target Number')
distractor_nums_erp = distractor_nums_epochs.average().plot(picks=['F1'], titles='Distractor Stream - Target number')

# Non-targets in the target stream
non_targets_target_epochs = mne.Epochs(raw_clean, non_targets_target_events, non_targets_target_mapping, tmin=-0.2, tmax=0.9, baseline=(-0.2, 0), preload=True)

# Non-targets in the distractor stream
non_targets_distractor_epochs = mne.Epochs(raw_clean, non_targets_distractor_events, non_targets_distractor_mapping, tmin=-0.2, tmax=0.9, baseline=(-0.2, 0), preload=True)

non_targets_distractor_erp = non_targets_distractor_epochs.average().plot(picks=['F1'], titles='Distractor Stream - Non-target Numbers')
non_targets_target_erp = non_targets_target_epochs.average().plot(picks=['F1'], titles='Target Stream - Non-target Numbers')


# Deviant animal sounds in the distractor stream
deviant_sounds_epochs = mne.Epochs(raw_clean, deviant_sounds_events, animals_mapping,tmin=-0.2, tmax=0.9, baseline=(-0.2, 0), preload=True)

deviant_sounds_erp = deviant_sounds_epochs.average().plot(picks=['F1'], titles='Distractor Stream - Deviant Sounds')
