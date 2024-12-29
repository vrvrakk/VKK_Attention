'''
The steps:
1. load EEG files for selected condition: a, a2, e1 or e2
2. load motor-only EEG file as well
## following steps apply for both motor-only and main EEG data ##
3. set up montage, using the standard electrode system (10-20)
4. drop A1, A2 and M2 channels (EMG channels)
5. downsample to 500Hz
6. add FCz electrode as reference
7. interpolate bad channels
8. mark bad segments
9. bp filter from 1-40Hz
10. apply ICA: remove eye blinks, eye movements
# motor-only:
11. create epochs around the button presses (0.5s long) -> average ERP
12. smooth motor's ERP edges (0.2 ramp on both ends, using a cosine ramp)
## now final steps:
13. clean the main EEG signal with the motor ERP, around the button-press events (use response-mapping)
14. create dummy 1s epochs for the main EEG data
15. apply AutoReject, to drop or correct any bad epochs
16. re-create the raw signal by concatenating the epochs back together
17. re-apply the annotations that remain
18. save clean EEG data
'''
# libraries:
from pathlib import Path
import os
import json
import pandas as pd
import numpy as np
import mne
from autoreject import AutoReject, Ransac
from meegkit import dss
from TRF.helper import grad_psd, snr
from matplotlib import pyplot as plt
import librosa

# todo: finalize pre-processing script
# todo: downsample both EEG and sound data into 100Hz
# todo: extract all predictors of choice: envelope, onset, target and distractor number tags, deviant tag
# todo: use baseline for boosting function and regularization?
# todo: choose time lags

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
for folder in sub_dir, fig_path, results_path, epochs_folder, evokeds_folder, raw_fif, psd_path:
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

mapping_path = json_path / 'events_mapping.json'
with open(mapping_path, 'r') as file:
    events_mapping = json.load(file)
s1_mapping = events_mapping[0]
s2_mapping = events_mapping[1]
response_mapping = events_mapping[2]


# config files
with open(json_path / "preproc_config.json") as file:
    cfg = json.load(file)
# load electrode names file:
with open(json_path / "electrode_names.json") as file: #electrode_names
    mapping = json.load(file)
# load EEF markers dictionary (contains all events of s1, s2 and button-presses)
with open(json_path/'eeg_events.json') as file:
    markers_dict = json.load(file)

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

# Combine with other stimuli events
all_events = {**s1_events, **s2_events, **response_events, **selected_animal_event}

# get motor EEG:
for files in sub_dir.iterdir():
    if files.is_file and 'motor.vhdr' in files.name:
        motor = mne.io.read_raw_brainvision(files, preload=True)

motor.rename_channels(mapping)  # apply custom map created for channel names
motor.add_reference_channels('FCz')  # add reference channel
motor.set_montage('standard_1020')  # apply standard montage
motor = motor.drop_channels(['A2', 'M2', 'A1'])  # select EMG-relevant files and drop them
motor.resample(sfreq=500)

### STEP 0: Concatenate block files to one raw file in raw_folder
# select header files with the condition in their name:
def choose_header_files(condition=condition):
    target_header_files_list = []
    header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
    filtered_files = [file for file in header_files if condition in file]
    if filtered_files:
        target_header_files_list.append(filtered_files)
    return target_header_files_list, condition

def get_raw_files(target_header_files_list):
    raw_files = []
    for header_file in target_header_files_list[0]:
        full_path = os.path.join(sub_dir, header_file)
        raw_file = mne.io.read_raw_brainvision(full_path, preload=True)
        raw_file.resample(sfreq=500) # downsample from 1000Hz to 500Hz
        raw_file.rename_channels(mapping)
        raw_file.add_reference_channels('FCz') # add reference channel
        raw_file.set_montage('standard_1020') # apply standard montage
        raw_files.append(raw_file)
    return raw_files


def get_events(eeg_data, all_events):  # get all events of selected data
    # Extract events and their corresponding IDs from annotations
    stream_events, stream_event_ids = mne.events_from_annotations(eeg_data)

    # Dynamically find the "New Segment" event ID
    new_segment_id = None
    for description, event_id in stream_event_ids.items():
        if description == 'New Segment/':  # Check for the "New Segment" label
            new_segment_id = event_id
            break

    # If "New Segment" is found, ensure it's added to the event mappings
    if new_segment_id is not None:
        all_events['New Segment/'] = new_segment_id

    # Update the event IDs in the stream_events array based on all_events
    for i in range(len(stream_events)):
        description = list(stream_event_ids.keys())[list(stream_event_ids.values()).index(stream_events[i, 2])]
        if description in all_events:
            stream_events[i, 2] = all_events[description]

    # Filter events to keep only those in all_events values (including "New Segment")
    filtered_events = [event for event in stream_events if event[2] in all_events.values()]
    stream_events = np.array(filtered_events)
    return stream_events, stream_event_ids

def get_response_events(raw, target_events):  # if a1 or e1: choose s1_events; if a2 or e2: s2_events
    events = mne.events_from_annotations(raw)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in target_events.values()]
    events = np.array(filtered_events)
    return events

def get_motor_events(motor_ica, response_mapping):  # if a1 or e1: choose s1_events; if a2 or e2: s2_events
    events = mne.events_from_annotations(motor_ica)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if str(event[2]) in response_mapping.keys()] # get motor-response events only
    events = np.array(filtered_events)
    return events

# 2. Interpolate
def interpolate(raw, condition):
    raw_interp = raw.copy().interpolate_bads(reset_bads=True)
    raw_interp.plot()
    raw.save(raw_fif / f"{name}_{condition}_interpolated-raw.fif", overwrite=True)
    return raw_interp


# 3. HIGH- AND LOW-PASS FILTER + POWER NOISE REMOVAL
def filtering(raw, data):
    cfg["filtering"]["notch"] = 50
    # remove the power noise
    raw_filter = raw.copy()
    raw_notch, iterations = dss.dss_line(raw_filter.get_data().T, fline=cfg["filtering"]["notch"],
                                         sfreq=data.info["sfreq"],
                                         nfft=cfg["filtering"]["nfft"])

    raw_filter._data = raw_notch.T
    cfg["filtering"]["highpass"] = 1
    hi_filter = cfg["filtering"]["highpass"]
    lo_filter = cfg["filtering"]["lowpass"]

    raw_filtered = raw_filter.copy().filter(hi_filter, lo_filter)
    raw_filtered.plot()

    # plot the filtering
    grad_psd(raw, raw_filter, raw_filtered, fig_path)
    return raw, raw_filter, raw_filtered


# Run pre-processing steps:
# load relevant EEG files:
target_header_files_list, condition = choose_header_files()
# set montage:
target_raw_files = get_raw_files(target_header_files_list)
raw_files_copy = target_raw_files.copy()
target_raw = mne.concatenate_raws(raw_files_copy)  # concatenate all into one raw, but copy first

stimuli_events, stimuli_events_ids = get_events(target_raw, all_events)
# response_events = {**response_events, **new_segment}
button_events = get_response_events(target_raw, response_events)

# to select bad channels, and select bad segmenmts:
target_raw.plot()
motor.plot()

# remove bad segments:
# get annotations info:
# def drop_bad_segments(eeg):
#     onsets = eeg.annotations.onset
#     durations = eeg.annotations.duration
#     descriptions = eeg.annotations.description
#
#     # Find good segments
#     good_intervals = []
#     last_good_end = 0
#     for onset, duration, description in zip(onsets, durations, descriptions):
#         if 'BAD' in description or 'bad' in description:  # description name may vary for each file (Bad boundary)
#             good_intervals.append((last_good_end, onset))
#             last_good_end = onset + duration
#     # Add the final good segment
#     good_intervals.append((last_good_end, eeg.times[-1]))
#
#     # Crop and concatenate good segments
#     good_segments = [eeg.copy().crop(tmin=start, tmax=end) for start, end in good_intervals]
#     eeg = mne.concatenate_raws(good_segments)
#
#     return eeg
# target_raw = drop_bad_segments(target_raw)
# motor = drop_bad_segments(motor)

# drop EMG channels
target_raw.drop_channels(['A1', 'A2', 'M2'])
target_raw.save(raw_fif / f"{name}_{condition}-raw.fif", overwrite=True)  # here the data is saved as raw
motor.save(raw_fif / f"{name}_motor-raw.fif", overwrite=True)  # here the data is saved as raw
print(f'{condition} raw data saved. If raw is empty, make sure axis and condition are filled in correctly.')

target_raw.plot_psd(exclude=['FCz'])
motor.plot_psd(exclude=['FCz'])

# interpolate bad selected channels, after removing significant noise affecting many electrodes
target_interp = interpolate(target_raw, condition)
motor_interp = interpolate(motor, condition='motor')

# get raw array, and info for filtering
raw_data = mne.io.RawArray(data=target_interp.get_data(), info=target_interp.info)
motor_data = mne.io.RawArray(data=motor_interp.get_data(), info=motor_interp.info)
# Filter: bandpas 1-40Hz
target_raw, target_filter, target_filtered = filtering(target_interp, raw_data)
motor, motor_filter, motor_filtered = filtering(motor_interp, motor_data)
target_filtered.save(results_path / f'1-40Hz_{name}_{condition}-raw.fif', overwrite=True)
motor_filtered.save(results_path / f'1-40Hz_{name}_conditions_motor-raw.fif', overwrite=True)


############ subtract motor noise:
# 4. ICA
target_ica = target_filtered.copy()
target_ica.info['bads'].append('FCz')  # Add FCz to the list of bad channels
ica = mne.preprocessing.ICA(n_components=cfg["ica"]["n_components"], method=cfg["ica"]["method"], random_state=99)
ica.fit(target_ica)
ica.plot_components()
ica.plot_sources(target_ica)
ica.apply(target_ica)
target_ica.info['bads'].remove('FCz')
target_ica.set_eeg_reference(ref_channels='average')
target_ica.save(results_path / f'{condition}_{name}_ICA-raw.fif', overwrite=True)

# apply ICA on motor EEG:
motor_ica = motor_filtered.copy()
motor_ica.info['bads'].append('FCz')
ica = mne.preprocessing.ICA(n_components=cfg["ica"]["n_components"], method=cfg["ica"]["method"], random_state=99)
ica.fit(motor_ica)
ica.plot_components()
ica.plot_sources(motor_ica)
ica.apply(motor_ica)
motor_ica.info['bads'].remove('FCz')
motor_ica.set_eeg_reference(ref_channels='average')
motor_ica.save(results_path / f'{name}_ICA_motor-raw.fif', overwrite=True)

# create motor EEG epochs:
motor_events = get_motor_events(motor_ica, response_mapping)
motor_markers = motor_events[:, 2]
response_markers = np.array(list(response_mapping.keys()), dtype='int32')
filtered_response_markers = motor_markers[np.isin(motor_markers, response_markers)]
filtered_response_events = [event for event in response_markers if event in filtered_response_markers]
motor_unique_annotations = np.unique(motor_ica.annotations.description)
motor_reject_annotation = None
for unique_annotation in motor_unique_annotations:
    if 'BAD' in unique_annotation:
        motor_reject_annotation = unique_annotation
    elif 'bad' in unique_annotation:
        motor_reject_annotation = unique_annotation

motor_epochs = mne.Epochs(motor_ica, motor_events, filtered_response_events, tmin=-0.1, tmax=0.4, baseline=(None, 0), reject_by_annotation=motor_reject_annotation, preload=True)
# now time for padding the ERP:
# # Step 1: Compute the average ERP from motor epochs
motor_erp = motor_epochs.average()
duration = 0.2  # Length of ramp in seconds
sfreq = motor_erp.info['sfreq']  # Sampling frequency (e.g., 500 Hz)
size = int(duration * sfreq)  # Ramp size in data samples

# Step 2: Define cosine envelope function
envelope = lambda t: 0.5 * (1 + np.cos(np.pi * (t - 1)))  # Cosine envelope
ramp_multiplier = envelope(np.linspace(0.0, 1.0, size))  # Generate ramp multiplier

# Step 3: Smooth edges of the ERP data
motor_erp_smooth = motor_erp.copy()  # Create a copy to preserve the original ERP
for ch in range(motor_erp_smooth.data.shape[0]):  # Loop over each channel
    motor_erp_smooth.data[ch, :size] *= ramp_multiplier  # Apply ramp at the beginning
    motor_erp_smooth.data[ch, -size:] *= ramp_multiplier[::-1]  # Apply ramp at the end
# Step 8: Plot the padded ERP
motor_erp_smooth.plot()
motor_path = results_path / 'motor'
os.makedirs(motor_path, exist_ok=True)
motor_erp_smooth.save(motor_path/f'{condition}_{name}_motor_erp_smooth-ave.fif', overwrite=True)

# load saved files:
motor_erp_smooth = mne.read_evokeds(motor_path / f'{condition}_{name}_motor_erp_smooth-ave.fif')

# subtract motor noise from response events of raw data
# clean raw EEG signal from motor noise
time_start = motor_erp_smooth[0].times[0]
time_end = motor_erp_smooth[0].times[-1]
time_window = time_end - time_start
n_samples_erp = int(time_window * sfreq) + 1
# # # Subtract the ERP at each event time
for event in button_events:
    event_sample = event[0]  # sample number of the event
    start_sample = event_sample - int(time_start * sfreq)
    end_sample = start_sample + n_samples_erp

    # Check if the event is within the bounds of the raw data
    if start_sample >= 0 and end_sample <= target_ica._data.shape[1]:
        # Subtract the ERP data from the raw data
        target_ica._data[:, start_sample:end_sample] -= motor_erp_smooth[0].data

target_ica.save(results_path / f'{condition}_cleaned_motor_raw_ica_{name}-raw.fif', overwrite=True)

# extract events from target_ica:
preprocessed_eeg = Path(default_dir/f'eeg/preprocessed/results/{name}')
# target_ica = mne.io.read_raw_fif(preprocessed_eeg / f'{condition}_cleaned_motor_raw_ica_{name}-raw.fif', preload=True)

edge_boundaries = np.where(stimuli_events[:, 2] == all_events['New Segment/'])
stream_events_df = pd.DataFrame(stimuli_events)
stream_events_df = stream_events_df.drop(columns=[1])
stream_events_df['Stimulus Type'] = None
stream_events_df['Voice'] = None
stream_events_df['Envelope'] = None


columns = stream_events_df.columns.tolist()
# Replace the column name at index 1 with 'Numbers' and 0 with 'Timepoints:
columns[0] = 'Timepoints'
columns[1] = 'Numbers'
# Assign the updated column list back to the DataFrame
stream_events_df.columns = columns

block_path = Path(f'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/params/block_sequences/{name}.csv')

with open(block_path, 'r') as f:
    target_block = pd.read_csv(f)


# filter rows for selected condition:
def filter_target_block(target_block, condition=''):
    if condition == 'a1':
        target_block_filtered = target_block[(target_block['block_condition'] == 'azimuth') & (target_block['block_seq'] == 's1')]
    elif condition == 'a2':
        target_block_filtered = target_block[(target_block['block_condition'] == 'azimuth') & (target_block['block_seq'] == 's2')]
    elif condition == 'e1':
        target_block_filtered = target_block[(target_block['block_condition'] == 'elevation') & (target_block['block_seq'] == 's1')]
    elif condition == 'e2':
        target_block_filtered = target_block[(target_block['block_condition'] == 'elevation') & (target_block['block_seq'] == 's2')]
    return target_block_filtered

target_block_filtered = filter_target_block(target_block, condition=condition)
target_block_voice = target_block_filtered['Voices']
target_block_numbers = target_block_filtered['Target Number']

def define_target_and_distractor(target_number):
    distractor_number = None
    if target_number == 1:
        distractor_number = 65
    elif target_number == 2:
        distractor_number = 66
    elif target_number == 3:
        distractor_number = 67
    elif target_number == 4:
        distractor_number = 68
    elif target_number == 5:
        distractor_number = 69
    elif target_number == 6:
        distractor_number = 70
    elif target_number == 8:
        distractor_number = 72
    elif target_number == 9:
        distractor_number = 73
    return distractor_number

# Assign voices to segments
for i, voice in enumerate(target_block_voice):
    start_index = edge_boundaries[0][i]
    target_number = target_block_numbers.iloc[i]
    distractor_number = define_target_and_distractor(target_number)
    if i < len(edge_boundaries) - 1:
        end_index = edge_boundaries[0][i + 1] - 1
    else:
        end_index = len(stream_events_df) - 1
    # assign voice per segment
    stream_events_df.loc[start_index:end_index, 'Voice'] = voice
    # Assign Stimulus Type to the segment
    for idx in range(start_index, end_index + 1):
        number = stream_events_df.at[idx, 'Numbers']
        if number == target_number:
           stream_events_df.at[idx,'Stimulus Type'] = 'Target Number'
        elif number == list(selected_animal_event.values())[0]:
           stream_events_df.at[idx, 'Stimulus Type'] = 'Deviant'
        elif number == distractor_number:
             stream_events_df.at[idx, 'Stimulus Type'] = 'Distractor Number'
        elif 1 <= number <= 9:
            if condition == 'a1' or condition == 'e1':
                stream_events_df.at[idx, 'Stimulus Type'] = 'Target non-targets'
            elif condition == 'a2' or condition == 'e2':
                stream_events_df.at[idx, 'Stimulus Type'] = 'Distractor non-targets'
        elif 65 <= number <= 73:
            if condition == 'a1' or condition == 'e1':
                stream_events_df.at[idx, 'Stimulus Type'] = 'Distractor non-targets'
            elif condition == 'a2' or condition == 'e2':
                stream_events_df.at[idx, 'Stimulus Type'] = 'Target non-targets'



# iterate over enevelopes_path:
envelopes_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/voices_english/downsampled/envelopes')
voice1 = []
voice2 = []
voice3 = []
voice4 = []

for folders in envelopes_path.iterdir():
    for files in folders.iterdir():
        if folders.name == 'voice1':
            voice1.append(files)
        elif folders.name == 'voice2':
            voice2.append(files)
        elif folders.name == 'voice3':
            voice3.append(files)
        elif folders.name == 'voice4':
            voice4.append(files)

stimulus_to_index = {
    1: 0, 65: 0,
    2: 1, 66: 1,
    3: 2, 67: 2,
    4: 3, 68: 3,
    5: 4, 69: 4,
    6: 5, 70: 5,
    8: 6, 72: 6,
    9: 7, 73: 7
}

for index, row in stream_events_df.iterrows():
    voice = row['Voice']  # Extract the voice
    stimulus = row['Numbers']  # Extract the stimulus number
    # Check if the voice and stimulus are valid
    if voice == 'voice1' and stimulus in stimulus_to_index:
        array = voice1[stimulus_to_index[stimulus]]
    elif voice == 'voice2' and stimulus in stimulus_to_index:
        array = voice2[stimulus_to_index[stimulus]]
    elif voice == 'voice3' and stimulus in stimulus_to_index:
        array = voice3[stimulus_to_index[stimulus]]
    elif voice == 'voice4' and stimulus in stimulus_to_index:
        array = voice4[stimulus_to_index[stimulus]]
    else:
        array = None  # Default to None if no match is found
    stream_events_df.at[index, 'Envelope'] = array

stream_events_df.loc[(stream_events_df['Envelope'].isnull()) & (stream_events_df['Numbers'] == 0), 'Envelope'] = 'New Segment'
selected_animal_value = list(selected_animal_event.values())[0]
stream_events_df.loc[(stream_events_df['Envelope'].isnull()) & (stream_events_df['Numbers'] == selected_animal_value), 'Envelope'] = 'Animal'
stream_events_df.loc[stream_events_df['Envelope'].isnull(), 'Envelope'] = 'Response'

# filter animal block to get the ones for selected condition:
target_indices = list(target_block_filtered.index)
animal_block_filtered = animal_block.loc[target_indices]
animal_block_filtered = animal_block_filtered.drop(columns=['15', '16', '17', '18', '19'])
animal_list = animal_block_filtered.values.flatten()  # Flatten the DataFrame into a 1D list
animal_index = 0  # Start with the first animal
for index, row in stream_events_df.iterrows():
    if row['Envelope'] == 'Animal':  # Check for placeholder "Animal"
        if animal_index < len(animal_list):  # Ensure we don't exceed the list
            stream_events_df.at[index, 'Envelope'] = animal_list[animal_index]
            animal_index += 1  # Move to the next animal in the list


animal_envelopes_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/sounds/processed/downsampled/envelopes')
animal_envelopes = []
for envelopes in animal_envelopes_path.iterdir():
    animal_envelopes.append(envelopes)

animal_sounds = ['cat', 'dog', 'frog', 'kitten', 'kookaburra', 'monkey', 'pig', 'sheep', 'turtle']
for index, row_value in stream_events_df['Envelope'].items():
    if row_value == 'cat':
        stream_events_df.at[index, 'Envelope'] = animal_envelopes[0]
    elif row_value == 'dog':
        stream_events_df.at[index, 'Envelope'] = animal_envelopes[1]
    elif row_value == 'frog':
        stream_events_df.at[index, 'Envelope'] = animal_envelopes[2]
    elif row_value == 'kitten':
        stream_events_df.at[index, 'Envelope'] = animal_envelopes[3]
    elif row_value == 'kookaburra':
        stream_events_df.at[index, 'Envelope'] = animal_envelopes[4]
    elif row_value == 'monkey':
        stream_events_df.at[index, 'Envelope'] = animal_envelopes[5]
    elif row_value == 'pig':
        stream_events_df.at[index, 'Envelope'] = animal_envelopes[6]
    elif row_value == 'sheep':
        stream_events_df.at[index, 'Envelope'] = animal_envelopes[7]
    elif row_value == 'turtle':
        stream_events_df.at[index, 'Envelope'] = animal_envelopes[8]

# Drop rows where 'Envelope' contains 'response' (case-insensitive)
stream_events_df = stream_events_df[stream_events_df['Envelope'] != 'Response']

# Optionally reset the index after dropping rows
stream_events_df.reset_index(drop=True, inplace=True)


target_df = {}
distractor_df = {}

if condition == 'a1' or condition == 'e1':
    # Filter events for stimuli 1 to 9
    target_df = stream_events_df[stream_events_df['Numbers'].isin(range(1, 10))]
    # Filter events for stimuli 65 to 73
    distractor_df = stream_events_df[stream_events_df['Numbers'].isin(range(65, 74))]
elif condition == 'a2' or condition == 'e2':
    distractor_df = stream_events_df[stream_events_df['Numbers'].isin(range(1, 10))]
    target_df = stream_events_df[stream_events_df['Numbers'].isin(range(65, 74))]

# save continuous envelope tracks and dataframes:
tables_path = results_path / 'tables'
os.makedirs(tables_path, exist_ok=True)
target_df.to_csv(tables_path / f'{sub_input}_{condition}_target_stream_events_raw.csv')
distractor_df.to_csv(tables_path / f'{sub_input}_{condition}_distractor_stream_events_raw.csv')

# save dataframes for different analyses:
# target num and distractor num only:
target_nums_df = target_df[target_df['Stimulus Type'] == 'Target Number']
distractor_nums_df = distractor_df[distractor_df['Stimulus Type'] == 'Distractor Number']
# animal sounds dataframe:
deviant_sounds_df = distractor_df[distractor_df['Stimulus Type'] == 'Deviant']
# filter out if necessary:
deviant_sounds_df = deviant_sounds_df[deviant_sounds_df['Envelope'] != 'Animal']
# now non-target dataframes:
non_targets_target_df = target_df[target_df['Stimulus Type'] == 'Target non-targets']
non_targets_distractor_df = distractor_df[distractor_df['Stimulus Type'] == 'Distractor non-targets']

# save all dataframes:
target_nums_df.to_csv(tables_path / f'{sub_input}_{condition}_target_nums_events_raw.csv')
distractor_nums_df.to_csv(tables_path / f'{sub_input}_{condition}_distractor_nums_events_raw_raw.csv')
non_targets_target_df.to_csv(tables_path / f'{sub_input}_{condition}_target_non_targets_events_raw.csv')
non_targets_distractor_df.to_csv(tables_path / f'{sub_input}_{condition}_distractor_non_targets_events_raw.csv')

deviant_sounds_df.to_csv(tables_path / f'{sub_input}_{condition}_deviant_sounds_events_raw.csv')


# create dummy epochs for AutoReject:
epoch_len = 1
events = mne.make_fixed_length_events(target_ica, duration=epoch_len)
unique_annotations = np.unique(target_ica.annotations.description)
reject_annotation = None
for unique_annotation in unique_annotations:
    if 'BAD' in unique_annotation:
        reject_annotation = unique_annotation
    elif 'bad' in unique_annotation:
        reject_annotation = unique_annotation

epochs = mne.Epochs(target_ica, events, tmin=0, tmax=epoch_len, baseline=None, reject_by_annotation=reject_annotation, preload=True)

# apply RANSAC for interpolating bad channels:
epochs_clean = epochs.copy()
ransac = Ransac(n_jobs=1, n_resample=50, min_channels=0.25, min_corr=0.75, unbroken_time=0.4)
ransac.fit(epochs_clean)
bads = epochs.info['bads']  # Add channel names to exclude here
if len(bads) != 0:
    for bad in bads:
        if bad not in ransac.bad_chs_:
            ransac.bad_chs_.extend(bads)
print(ransac.bad_chs_)
epochs_clean = ransac.transform(epochs_clean)


# apply AutoReject:
def ar(epochs, name):
    ar = AutoReject(n_interpolate=cfg["autoreject"]["n_interpolate"], n_jobs=cfg["autoreject"]["n_jobs"])
    ar = ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)

    # target_epochs1[reject_log1.bad_epochs].plot(scalings=dict(eeg=100e-6))
    # reject_log1.plot('horizontal', show=False)

    # plot and save the final results
    fig, ax = plt.subplots(2, constrained_layout=True)
    epochs_ar.average().plot_image(titles=f"SNR:{snr(epochs_ar):.2f}", show=False, axes=ax[0])
    epochs_ar.average().plot(show=False, axes=ax[1])
    plt.savefig(results_path / 'figures' / f"{name} clean_evoked {condition}.pdf", dpi=800)
    plt.close()
    epochs_ar.save(results_path / 'epochs' / f"{name}_conditions_{condition}-epo.fif",
                   overwrite=True)
    return epochs_ar

epochs_ar = ar(epochs_clean, name)
epochs_data = epochs_ar._data

data_concat = np.concatenate(epochs_data, axis=-1)

info = epochs_ar.info  # Reuse the info from the epochs

original_annotations = target_ica.annotations
original_events, original_event_id = mne.events_from_annotations(target_ica)

raw_clean = mne.io.RawArray(data_concat, info)
# Reapply original annotations
raw_clean.set_annotations(original_annotations)

raw_clean.save(results_path / f'{name}_{condition}_preproccessed-raw.fif', overwrite=True)

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
min_time_diff = 0.2  # 200ms
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
# save distractor and target nums events:



non_targets_target_events, non_targets_target_mapping = transform_events(non_targets_target_df, mapping=s1_events)
non_targets_distractor_events, non_targets_distractor_mapping = transform_events(non_targets_distractor_df, mapping=s2_events)

deviant_sounds_events, animals_mapping = transform_events(deviant_sounds_df, mapping=selected_animal_event)

# create different epochs and ERPs:
target_nums_epochs = mne.Epochs(raw_clean, target_nums_events, target_nums_mapping, tmin=-0.2, tmax=0.9, baseline=(-0.2, 0), preload=True)
distractor_nums_epochs = mne.Epochs(raw_clean, distractor_nums_events, distractor_nums_mapping, tmin=-0.2, tmax=0.9, baseline=(-0.2, 0), preload=True)

target_nums_erp = target_nums_epochs.average().plot(picks=['F1'], titles='Target Stream - Target Number')
distractor_nums_erp = distractor_nums_epochs.average().plot(picks=['F1'], titles='Distractor Stream - Target number')

# Non-targets in the target stream
non_targets_target_epochs = mne.Epochs(raw_clean, non_targets_target_events, non_targets_target_mapping,tmin=-0.2, tmax=0.9, baseline=(-0.2, 0), preload=True)
non_targets_target_erp = non_targets_target_epochs.average().plot(picks=['F1'], titles='Target Stream - Non-target Numbers')

# Non-targets in the distractor stream
non_targets_distractor_epochs = mne.Epochs(raw_clean, non_targets_distractor_events, non_targets_distractor_mapping, tmin=-0.2, tmax=0.9, baseline=(-0.2, 0), preload=True)
non_targets_distractor_erp = non_targets_distractor_epochs.average().plot(picks=['F1'], titles='Distractor Stream - Non-target Numbers')

# Deviant animal sounds in the distractor stream
deviant_sounds_epochs = mne.Epochs(raw_clean, deviant_sounds_events, animals_mapping, tmin=-0.2, tmax=0.9, baseline=(-0.2, 0), preload=True)
deviant_sounds_erp = deviant_sounds_epochs.average().plot(picks=['F1'], titles='Distractor Stream - Deviant Sounds')
