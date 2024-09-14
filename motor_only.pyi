import mne  # mne is the EEG package
from pathlib import Path
import os
# os and path are used to define file and folder paths, and to open and load files from your pc
import numpy as np
from autoreject import AutoReject, Ransac # sophisticated interpolation methods used when pre-processing EEG data
from collections import Counter
import json
from meegkit import dss # used when filtering the signal
from matplotlib import pyplot as plt, patches
from helper import grad_psd, snr # helper files for the pre-processing script

''' this script will be used to process the motor-only recording, 
before applying it onto the actual EEG data we want to pre-process and analyze'''

sub_input = input("Give sub number as subn (n for number): ")

default_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data')
json_path = default_dir / 'misc' # where the configuration files are located
sub_dir = default_dir /'eeg'/ 'raw'/ sub_input
motor_dir = sub_dir / 'motor-only'

if not os.path.isdir(motor_dir):
    os.makedirs(motor_dir)

# iterate through files in the subject Path, and extract header file path
# with 'baseline' in its name:
motor_header = []
for files in sub_dir.iterdir():
    if files.is_file() and 'motor.vhdr' in files.name:
        motor_header.append(files)

# load and read the eeg file that corresponds to this header file:
raw = mne.io.read_raw_brainvision(motor_header[0], preload=True)

# open configuration file for mapping the electrodes with standard names:
with open(json_path / "electrode_names.json") as file:
    mapping = json.load(file)
# a custom map created based on the standard electrode positions and names;
# last three electrodes: 65-67 I gave them arbitrary names from the standard set-up
# A1, A2 and M2
raw.rename_channels(mapping)
raw.set_montage('standard_1020') # to use standard EEG channel names
# drop EMG channels:
raw.drop_channels(['A1', 'A2', 'M2']) # we do not need them for this script, will be analyzed separately!

raw.save(motor_dir / f"{sub_input}_motor_raw.fif", overwrite=True)  # here the data is saved as raw

# interpolate shitty channels by:
# 1. plotting raw eeg data and investigating:
raw.plot()
# 2. select bad channels by right clicking, then close window;
# 3. then:
raw_interp = raw.copy().interpolate_bads(reset_bads=True)

# now, time to filter:
# 1. load json file with configuration parameters for filtering:
with open(json_path / "preproc_config.json") as file:
    cfg = json.load(file)

# 2. extract signal's data:
data = mne.io.RawArray(data=raw_interp.get_data(), info=raw_interp.info)

# notch filter for power line noise:
cfg["filtering"]["notch"] = 50
# dss_line is some sophisticated method, haven't looked much into it yet lol
raw_filter = raw_interp.copy()
raw_notch, iterations = dss.dss_line(raw_filter.get_data().T, fline=cfg["filtering"]["notch"],
                                     sfreq=data.info["sfreq"],
                                     nfft=cfg["filtering"]["nfft"])

raw_filter._data = raw_notch.T
# bandpass filter: 1-25Hz
cfg["filtering"]["highpass"] = 1
lo_filter = cfg["filtering"]["lowpass"] = 25
hi_filter = cfg["filtering"]["highpass"]
lo_filter = cfg["filtering"]["lowpass"]

raw_filtered = raw_filter.copy().filter(hi_filter, lo_filter)
raw_filtered.plot()
# specify a figure path where figures related to the eeg file will be saved:
fig_path = motor_dir / 'figures'
if not os.path.isdir(fig_path):
    os.makedirs(fig_path)


# plot the filtering
grad_psd(raw, raw_filter, raw_filtered, fig_path)

# check the filtered eeg signal; if necessary, remove bad segments where there is clearly a lot of noise
# do so, if noise affects a lot of channels at once; because this prevents further analysis, like ICA
# ICA separates the signal into different components, that, when combined all together
# make the signal -> some of these components will represent noise

onsets = raw_filtered.annotations.onset # you get the times whenever a button press was marked in the eeg signal
durations = raw_filtered.annotations.duration # duration of each button press event
descriptions = raw_filtered.annotations.description # what names are used for the markers?

# Find good segments
good_intervals = []
last_good_end = 0
for onset, duration, description in zip(onsets, durations, descriptions):
    if description == 'BAD boundary' or 'BAD_' in description:  # description name may vary for each file (Bad boundary)
        good_intervals.append((last_good_end, onset))
        last_good_end = onset + duration
        # if the loop comes across a description called 'bad', it saves all the data that occurred before it
        # you append every good segment in a list called good_intervals
# Add the final good segment
good_intervals.append((last_good_end, raw_filtered.times[-1]))

# Crop and concatenate good segments
good_segments = [raw_filtered.copy().crop(tmin=start, tmax=end) for start, end in good_intervals]
raw_filtered = mne.concatenate_raws(good_segments)

# remove eye-blinks etc. with ICA:
motor_ica = raw_filtered.copy()
ica = mne.preprocessing.ICA(n_components=cfg["ica"]["n_components"], method=cfg["ica"]["method"], random_state=99)
ica.fit(motor_ica)
# check components that are related to eye blinks, heart-rate, eye movement, general body movement
ica.plot_components()
ica.plot_sources(motor_ica)
# use both sources and components to investigate and select accordingly
# by applying, these components are removed from the signal
ica.apply(motor_ica)
motor_ica.save(motor_dir / f'{sub_input}_motor_ICA-raw.fif', overwrite=True)


# EPOCHING:
# get button presses events:
markers_dict = {
    's1_events': {'Stimulus/S 1': 1, 'Stimulus/S 2': 2, 'Stimulus/S 3': 3, 'Stimulus/S 4': 4, 'Stimulus/S 5': 5,
                  'Stimulus/S 6': 6, 'Stimulus/S 8': 8, 'Stimulus/S 9': 9},
    # stimulus 1 markers
    's2_events': {'Stimulus/S 72': 72, 'Stimulus/S 73': 73, 'Stimulus/S 65': 65, 'Stimulus/S 66': 66,
                  'Stimulus/S 69': 69, 'Stimulus/S 70': 70, 'Stimulus/S 68': 68,
                  'Stimulus/S 67': 67},  # stimulus 2 markers
    'response_events': {'Stimulus/S132': 132, 'Stimulus/S130': 130, 'Stimulus/S134': 134, 'Stimulus/S137': 137,
                        'Stimulus/S136': 136, 'Stimulus/S129': 129, 'Stimulus/S131': 131,
                        'Stimulus/S133': 133}}  # all markers
response_events = markers_dict['response_events']  # response markers

# extract an array with all the response events, based on the markers with the matching names/descriptions
# which events and when these events took place, is found in the .vmrk file
events = mne.events_from_annotations(motor_ica)[0]  # get events from annotations attribute of raw variable
events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
filtered_events = [event for event in events if event[2] in response_events.values()]
events = np.array(filtered_events)


# epoch signal around button-press events:
# Counter(events1[:, 2])
tmin = -0.4
tmax = 0.8
epoch_parameters = [tmin, tmax, response_events]
tmin, tmax, event_ids = epoch_parameters
event_ids = {key: val for key, val in event_ids.items() if val in events[:, 2]}

epochs = mne.Epochs(motor_ica,
                           events,
                           event_id=event_ids,
                           tmin=tmin,
                           tmax=tmax,
                           baseline=(None, 0),
                           detrend=0,  # should we set it here?
                           preload=True)

epochs.save(motor_dir/f'{sub_input}_epochs_motor-epo.fif', overwrite=True)

# RANSAC:
epochs.plot_psd() # to check any weird channels, and then add them in 'bads'
bads = []
epochs_clean = epochs.copy()
cfg["reref"]["ransac"]["min_corr"] = 0.75
ransac = Ransac(n_jobs=cfg["reref"]["ransac"]["n_jobs"], n_resample=cfg["reref"]["ransac"]["n_resample"],
                min_channels=cfg["reref"]["ransac"]["min_channels"], min_corr=cfg["reref"]["ransac"]["min_corr"],
                unbroken_time=cfg["reref"]["ransac"]["unbroken_time"])
ransac.fit(epochs_clean)

epochs_clean.average().plot(exclude=[])
epochs.average().plot(exclude=[])

if len(bads) != 0 and bads not in ransac.bad_chs_:
    ransac.bad_chs_.extend(bads)
ransac.transform(epochs_clean)

# get evoked ERPs by averaging the epochs:
evoked = epochs.average()
evoked_clean = epochs_clean.average()

evoked.info['bads'] = ransac.bad_chs_
evoked_clean.info['bads'] = ransac.bad_chs_

fig, ax = plt.subplots(2, constrained_layout=True)
evoked.plot(exclude=[], axes=ax[0], show=False)
evoked_clean.plot(exclude=[], axes=ax[1], show=False)
ax[0].set_title(f"Before RANSAC (bad chs:{ransac.bad_chs_})")
ax[1].set_title("After RANSAC")

fig.savefig(fig_path / f"{sub_input}_RANSAC_motor.pdf", dpi=800)
plt.close()

epochs_reref = epochs_clean.copy()
epochs_reref.set_eeg_reference(ref_channels='average')

# AutoReject:
ar = AutoReject(n_interpolate=cfg["autoreject"]["n_interpolate"], n_jobs=cfg["autoreject"]["n_jobs"])
ar = ar.fit(epochs_reref)
epochs_ar, reject_log = ar.transform(epochs_reref, return_log=True)

# target_epochs1[reject_log1.bad_epochs].plot(scalings=dict(eeg=100e-6))
# reject_log1.plot('horizontal', show=False)

# plot and save the final results
fig, ax = plt.subplots(2, constrained_layout=True)
epochs_ar.average().plot_image(titles=f"SNR:{snr(epochs_ar):.2f}", show=False, axes=ax[0])
epochs_ar.average().plot(show=False, axes=ax[1])
plt.savefig(fig_path/ f"{sub_input} for clean_evoked motor-only.pdf", dpi=800)
plt.close()
epochs_ar.save(motor_dir / f"{sub_input}_AR_ motor_only-epo.fif", overwrite=True)


# get evoked responses:
epochs = epochs_ar.copy()
event_ids = list(event_ids.values())
evokeds = []
for event_id in event_ids:
    # Find the indices of epochs matching the event ID
    matching_indices = np.where(epochs.events[:, 2] == event_id)[0]

    # Debug: Check the number of matching epochs
    print(f"Event ID {event_id} has {len(matching_indices)} matching epochs")

    # Check if there are any matching epochs before averaging
    if len(matching_indices) > 0:
        evoked = epochs[matching_indices].average()
        evokeds.append(evoked)
    else:
        print(f"No epochs found for event ID {event_id}")

# save evokeds:
grand_average = mne.grand_average(evokeds)
grand_average.save(motor_dir/f'{sub_input}_evokeds_motor-ave.fif')

# pad the grand average with zeroes at the edges, to smooth out the signal
# here we use a cosine ramp
from scipy.ndimage import gaussian_filter1d
# Get the data from the grand average (2D array: channels x times)
grand_average_smooth = grand_average.copy() # copy grand average epoch
evoked_data = grand_average.copy().data # extract data from grand average
duration = 0.2 # specify length of ramp in seconds
sfreq = 500 # sample frequency -> in 1s, 500 samples of data are obtained
size = int(duration * sfreq) # the size of the ramp in data samples
# define cosine envelope used for smoothing the edges of the signal:
envelope = lambda t: 0.5 * (1 + np.cos(np.pi * (t - 1)))
# generate the ramp multiplier
ramp_multiplier = envelope(np.linspace(0.0, 1.0, size))

for ch in range(evoked_data.shape[0]):  # Loop over each channel
    evoked_data[ch, :size] *= ramp_multiplier  # Apply ramp at the beginning
    evoked_data[ch, -size:] *= ramp_multiplier[::-1]  # Apply ramp at the end (reverse ramp)
grand_average_smooth.data = evoked_data
grand_average_smooth.save(motor_dir/'Smoothed 1-25Hz Grand Average Motor-ave.fif', overwrite=True)

# plot smoothed grand average:
cm = 1 / 2.54
fig, ax = plt.subplots(figsize=(30 * cm, 15 * cm))  # Adjust the figure size as needed
# Plot the grand average evoked response
mne.viz.plot_compare_evokeds(
    {'Grand Average': grand_average_smooth},
    combine='mean',
    title=f'{sub_input} - Grand Average Evoked Motor Response',
    colors={'Grand Average': 'r'},
    linestyles={'Grand Average': 'solid'},
    axes=ax,
    show=True  # Set to True to display the plot immediately
)

plt.savefig(
    motor_dir / f"smoothed ERP_{sub_input}_motor-only.pdf",
    dpi=800)
