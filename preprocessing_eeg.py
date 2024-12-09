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
import pandas as pd
import mne
from pathlib import Path
import os
import numpy as np
from autoreject import AutoReject, Ransac
from collections import Counter
import json
from meegkit import dss
from matplotlib import pyplot as plt, patches
from helper import grad_psd, snr
import slab

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

# get motor EEG:
for files in sub_dir.iterdir():
    if files.is_file and 'motor.vhdr' in files.name:
        motor = mne.io.read_raw_brainvision(files, preload=True)

motor.rename_channels(mapping)
motor.add_reference_channels('FCz')
motor.set_montage('standard_1020')
motor = motor.drop_channels(['A2', 'M2', 'A1'])  # select EMG-relevant files


### STEP 0: Concatenate block files to one raw file in raw_folder

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
        raw_file.resample(sfreq=500)
        raw_file.rename_channels(mapping)
        raw_file.add_reference_channels('FCz')
        raw_file.set_montage('standard_1020')
        raw_files.append(raw_file)
    return raw_files




def get_events(raw, target_events):  # if a1 or e1: choose s1_events; if a2 or e2: s2_events
    events = mne.events_from_annotations(raw)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in target_events.values()]
    events = np.array(filtered_events)
    return events

def get_motor_events(motor_ica, response_mapping):  # if a1 or e1: choose s1_events; if a2 or e2: s2_events
    events = mne.events_from_annotations(motor_ica)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if str(event[2]) in response_mapping.keys()]
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
motor.resample(sfreq=500)

events1 = get_events(target_raw, s1_events)
events2 = get_events(target_raw, s2_events)
events3 = get_events(target_raw, response_events)

# to select bad channels, and select bad segmenmts:
target_raw.plot()
motor.plot()


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
target_ica.save(results_path / f'{condition}_{name}_ICA-raw.fif', overwrite=True)
target_ica.info['bads'].remove('FCz')
target_ica.set_eeg_reference(ref_channels='average')

# apply ICA on motor EEG:
motor_ica = motor_filtered.copy()
motor_ica.info['bads'].append('FCz')
ica = mne.preprocessing.ICA(n_components=cfg["ica"]["n_components"], method=cfg["ica"]["method"], random_state=99)
ica.fit(motor_ica)
ica.plot_components()
ica.plot_sources(motor_ica)
ica.apply(motor_ica)
motor_ica.save(results_path / f'{name}_ICA_motor-raw.fif', overwrite=True)
motor_ica.info['bads'].remove('FCz')
motor_ica.set_eeg_reference(ref_channels='average')

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
# target_ica = mne.io.read_raw_fif(results_path / f'{condition}_{name}_ICA-raw.fif', preload=True)

# subtract motor noise from response events of raw data
# clean raw EEG signal from motor noise
time_start = motor_erp_smooth[0].times[0]
time_end = motor_erp_smooth[0].times[-1]
time_window = time_end - time_start
n_samples_erp = int(time_window * sfreq) + 1
# # # Subtract the ERP at each event time
for event in events3:
    event_sample = event[0]  # sample number of the event
    start_sample = event_sample - int(time_start * sfreq)
    end_sample = start_sample + n_samples_erp

    # Check if the event is within the bounds of the raw data
    if start_sample >= 0 and end_sample <= target_ica._data.shape[1]:
        # Subtract the ERP data from the raw data
        target_ica._data[:, start_sample:end_sample] -= motor_erp_smooth[0].data

target_ica.save(results_path / f'{condition}_cleaned_motor_raw_ica_{name}-raw.fif', overwrite=True)

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
ransac = Ransac(n_jobs=1, n_resample=50,
                min_channels=0.25, min_corr=0.75,
                unbroken_time=0.4)
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

raw_clean.save(results_path / f'{name}_{condition}_preproccessed-raw.fif')


# check ERP:
target_epochs = mne.Epochs(raw_clean, events1, s1_events, tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0), reject_by_annotation=reject_annotation, preload=True)
target_epochs.average().plot()

# add wav files as meta_data
sound_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/voices_english/downsampled')
animal_sounds_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/sounds/processed/downsampled')
voice1_list = []
voice2_list = []
voice3_list = []
voice4_list = []
for folders in sound_path.iterdir():
    if folders.is_dir():
        for wav_files in folders.iterdir():
            if 'voice1' in str(wav_files):
                voice1_list.append(wav_files)
            elif 'voice2' in str(wav_files):
                voice2_list.append(wav_files)
            elif 'voice3' in str(wav_files):
                voice3_list.append(wav_files)
            elif 'voice4' in str(wav_files):
                voice4_list.append(wav_files)
# wav files have been resampled to 500Hz with Audacity and saved in separate path
voice1_arrays = []
voice2_arrays = []
voice3_arrays = []
voice4_arrays = []
def get_sound_arrays(voice_arrays, voice_list):
    for wav_file in voice_list:
        sound = slab.Sound(wav_file)
        voice_arrays.append(sound.data)
    return voice_arrays

voice1_arrays = get_sound_arrays(voice1_arrays, voice1_list)
voice2_arrays = get_sound_arrays(voice2_arrays, voice2_list)
voice3_arrays = get_sound_arrays(voice3_arrays, voice3_list)
voice4_arrays = get_sound_arrays(voice4_arrays, voice4_list)

# now do the same for animal sounds
animal_sounds = []
animal_names = ['cat', 'dog', 'frog', 'kitten', 'kookaburra', 'monkey', 'pig', 'sheep', 'turtle']

for files, animal_name in zip(animal_sounds_path.iterdir(), animal_names):
    animal_sound = slab.Sound(files)
    animal_sounds.append((animal_sound.data, animal_name))


# raw_clean = mne.io.read_raw_fif(results_path / f'{name}_{condition}_preproccessed-raw.fif', preload=True)
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
all_stimuli_events = {**s1_events, **s2_events, **selected_animal_event}
# raw_clean_events = mne.events_from_annotations(raw_clean, all_stimuli_events)

# create dictionaries to match events with wav_files:
stream1_events = {'1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9}
stream2_events = {'65': 1, '66': 2, '67': 3, '68': 4, '69': 5, '70': 6, '71': 7, '72': 8, '73': 9}
button_events = {'129': 1, '130': 2, '131': 3, '132': 4, '133': 5, '134': 6, '136': 8, '137': 9}

# todo: align raw_clean events with the correct wav file
block_path = Path(f'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/params/block_sequences/{name}.csv')
animal_block_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/params/animal_blocks')
for files in animal_block_path.iterdir():
    if name in files.name:
        animal_block = pd.read_csv(files)


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
target_voices_all = []
target_nums_all = []
target_block_events_all = []
#todo: remember to do this before concatenating all raws
for index, raw_data in enumerate(target_raw_files):
    target_block_voice = target_block_filtered['Voices'].iloc[index]
    target_voices_all.append(target_block_voice)
    target_block_number = target_block_filtered['Target Number'].iloc[index]
    target_nums_all.append(target_block_number)
    target_block_event = get_events(raw_data, all_stimuli_events)
    target_block_events_all.append(target_block_event)

# convert target block events into df, and add another column for corresponding file:
# potentially get arrays of wav files first
all_events_df = {}
for index, block in enumerate(target_block_events_all):
    all_events_df[index] = pd.DataFrame(block)
    all_events_df[index].columns = ['Timepoints', '0', 'Stimuli']
    all_events_df[index] = all_events_df[index].drop(columns='0')
    all_events_df[index]['Array'] = None

all_events_df_renamed = []
for voice, block in zip(target_voices_all, all_events_df.keys()):
    all_events_df_renamed.append((voice, all_events_df[block]))

# add sound arrays in Array column: both voices and animal sounds
stimuli_values = list(all_stimuli_events.values())
for name, events in all_events_df_renamed:
    for index, stimulus in enumerate(events['Stimuli'].values):
        if name == 'voice1' and stimulus in stimuli_values:
            if stimulus == 1 or stimulus == 65:
                array = voice1_arrays[0]
            elif stimulus == 2 or stimulus == 66:
                array = voice1_arrays[1]
            elif stimulus == 3 or stimulus == 67:
                array = voice1_arrays[2]
            elif stimulus == 4 or stimulus == 68:
                array = voice1_arrays[3]
            elif stimulus == 5 or stimulus == 69:
                array = voice1_arrays[4]
            elif stimulus == 6 or stimulus == 70:
                array = voice1_arrays[5]
            elif stimulus == 8 or stimulus == 72:
                array = voice1_arrays[6]
            elif stimulus == 9 or stimulus == 73:
                array = voice1_arrays[7]
            else:
                array = None
            events.at[index, 'Array'] = array

        if name == 'voice2' and stimulus in stimuli_values:
            if stimulus == 1 or stimulus == 65:
                array = voice2_arrays[0]
            elif stimulus == 2 or stimulus == 66:
                array = voice2_arrays[1]
            elif stimulus == 3 or stimulus == 67:
                array = voice2_arrays[2]
            elif stimulus == 4 or stimulus == 68:
                array = voice2_arrays[3]
            elif stimulus == 5 or stimulus == 69:
                array = voice2_arrays[4]
            elif stimulus == 6 or stimulus == 70:
                array = voice2_arrays[5]
            elif stimulus == 8 or stimulus == 72:
                array = voice2_arrays[6]
            elif stimulus == 9 or stimulus == 73:
                array = voice2_arrays[7]
            else:
                array = None
            events.at[index, 'Array'] = array
        if name == 'voice3' and stimulus in stimuli_values:
            if stimulus == 1 or stimulus == 65:
                array = voice3_arrays[0]
            elif stimulus == 2 or stimulus == 66:
                array = voice3_arrays[1]
            elif stimulus == 3 or stimulus == 67:
                array = voice3_arrays[2]
            elif stimulus == 4 or stimulus == 68:
                array = voice3_arrays[3]
            elif stimulus == 5 or stimulus == 69:
                array = voice3_arrays[4]
            elif stimulus == 6 or stimulus == 70:
                array = voice3_arrays[5]
            elif stimulus == 8 or stimulus == 72:
                array = voice3_arrays[6]
            elif stimulus == 9 or stimulus == 73:
                array = voice3_arrays[7]
            else:
                array = None
            events.at[index, 'Array'] = array
        if name == 'voice4' and stimulus in stimuli_values:
            if stimulus == 1 or stimulus == 65:
                array = voice4_arrays[0]
            elif stimulus == 2 or stimulus == 66:
                array = voice4_arrays[1]
            elif stimulus == 3 or stimulus == 67:
                array = voice4_arrays[2]
            elif stimulus == 4 or stimulus == 68:
                array = voice4_arrays[3]
            elif stimulus == 5 or stimulus == 69:
                array = voice4_arrays[4]
            elif stimulus == 6 or stimulus == 70:
                array = voice4_arrays[5]
            elif stimulus == 8 or stimulus == 72:
                array = voice4_arrays[6]
            elif stimulus == 9 or stimulus == 73:
                array = voice4_arrays[7]
            else:
                array = None
            events.at[index, 'Array'] = array
# filter animal block to get the ones for selected condition:
target_indices = list(target_block_filtered.index)
animal_block_filtered = animal_block.loc[target_indices]
animal_block_filtered = animal_block_filtered.drop(columns=['15', '16', '17', '18', '19'])
# now add the animal sound arrays:
for (name, events), animal_row in zip(all_events_df_renamed, animal_block_filtered.itertuples(index=False)):
    animal_index = 0  # Track the index of the current animal within the block
    for stim_index, stimulus in enumerate(events['Stimuli'].values):
        if stimulus == 7 or stimulus == 71:
            # Get the corresponding animal name for this stimulus
            animal_name = animal_row[animal_index]
            print(animal_name)

            # Find the array corresponding to this animal name
            for array, name in animal_sounds:
                if name == animal_name:
                    events.at[stim_index, 'Array'] = array  # Fill the Array column
                    break

            # Move to the next animal in the row
            animal_index += 1

stream2_dfs = []
stream1_dfs = []

# Iterate over all sub-DataFrames in `all_events_df_renamed`
for name, events in all_events_df_renamed:
    # Filter events for stimuli 65 to 73
    stream2 = events[events['Stimuli'].isin(range(65, 74))]
    stream2_dfs.append((name, stream2))

    # Filter events for stimuli 1 to 9
    stream1 = events[events['Stimuli'].isin(range(1, 10))]
    stream1_dfs.append((name, stream1))


s1_epochs_all = []
s2_epochs_all = []
for raw_data1, events1, (name1, block_events1) in zip(target_raw_files, target_block_events_all, stream1_dfs):
    s1_epochs = mne.Epochs(raw_data1, events1, s1_events,  tmin=-0.2, tmax=0.9, baseline=None, preload=True)
    s1_epochs.metadata = block_events1[['Array']]
    s1_epochs_all.append(s1_epochs)

for raw_data2, events2, (name2, block_events2) in zip(target_raw_files, target_block_events_all, stream2_dfs):
    s2_epochs = mne.Epochs(raw_data2, events2, s2_events, tmin=-0.2, tmax=0.9, baseline=None, preload=True)
    s2_epochs.metadata = block_events2[['Array']]
    s2_epochs_all.append(s2_epochs)










