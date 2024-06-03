'''
### STEP 1: Concatenate block files to one raw file in raw_folder ###
### STEP 2: bandpass filtering of the data at 1=40 Hz ###
### STEP 3: Apply ICA ###
### STEP 4: Epoch the raw data and apply baseline ###
### STEP 5: Run RANSAC/ exclude bad channels ###
### STEP 6: Rereference the epochs ###
### STEP 7: Apply AutoReject ###
### STEP 8: Average epochs and write evokeds###
'''
# libraries:
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
sub_input = input("Give sub number as subn (n for number): ")
sub = [sub.strip() for sub in sub_input.split(',')]
cm = 1 / 2.54

# 0. LOAD THE DATA
sub_dirs = []
fig_paths = []
epochs_folders = []
evokeds_folders = []
results_paths = []
for subs in sub:
    default_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data')
    raw_dir = default_dir / 'eeg' / 'raw'
    sub_dir = raw_dir / subs
    sub_dirs.append(sub_dir)
    json_path = default_dir / 'misc'
    fig_path = default_dir / 'eeg' / 'preprocessed' / 'results' / subs / 'figures'
    fig_paths.append(fig_path)
    results_path = default_dir / 'eeg' / 'preprocessed' / 'results' / subs
    results_paths.append(results_path)
    epochs_folder = results_path / "epochs"
    epochs_folders.append(epochs_folder)
    evokeds_folder = results_path / 'evokeds'
    evokeds_folders.append(evokeds_folder)
    raw_figures = sub_dir / 'figures'
    for folder in sub_dir, fig_path, results_path, epochs_folder, evokeds_folder, raw_figures:
        if not os.path.isdir(folder):
            os.makedirs(folder)
# to read file:
# file_path
# mne.read_epochs(file_path)
# events:
markers_dict = {'s1_events': {'Stimulus/S 1': 1, 'Stimulus/S 2': 2, 'Stimulus/S 3': 3, 'Stimulus/S 4': 4, 'Stimulus/S 5': 5, 'Stimulus/S 6': 6, 'Stimulus/S 8': 8, 'Stimulus/S 9': 9},
                # stimulus 1 markers
                's2_events': {'Stimulus/S 72': 18, 'Stimulus/S 73': 19, 'Stimulus/S 65': 11, 'Stimulus/S 66': 12, 'Stimulus/S 69': 15, 'Stimulus/S 70': 16, 'Stimulus/S 68': 14,
                               'Stimulus/S 67': 13},  # stimulus 2 markers
                'response_events': {'Stimulus/S132': 24, 'Stimulus/S130': 22, 'Stimulus/S134': 26, 'Stimulus/S137': 29, 'Stimulus/S136': 28, 'Stimulus/S129': 21, 'Stimulus/S131': 23, 'Stimulus/S133': 25}}  # response markers
s1_events = markers_dict['s1_events']
s2_events = markers_dict['s2_events']
both_stim = [s1_events, s2_events]
response_events = markers_dict['response_events']

# config files
with open(json_path / "preproc_config.json") as file:
    cfg = json.load(file)
with open(json_path / "electrode_names.json") as file:
    mapping = json.load(file)


# Run pre-processing steps:
condition = input('Please provide condition (exp. EEG): ')
axis = input('Please provide axis (exp. EEG): ')


### STEP 0: Concatenate block files to one raw file in raw_folder
def choose_header_files(condition=condition, axis=axis):
    target_header_files_list = []
    for sub_dir in sub_dirs:
        header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
        filtered_files = [file for file in header_files if condition in file]
        filt_files = [file for file in filtered_files if axis in file]
        if filt_files:
            target_header_files_list.append(filt_files)
    return target_header_files_list, condition, axis


def get_raw_files(target_header_files_list, condition, axis):
    raw_files = []
    for sub_dir, header_files in zip(sub_dirs, target_header_files_list):
        for header_file in header_files:
            full_path = os.path.join(sub_dir, header_file)
            print(full_path)
            raw_files.append(mne.io.read_raw_brainvision(full_path, preload=True))
    raw = mne.concatenate_raws(raw_files)  # read BrainVision files.
    # append all files from a participant
    raw.rename_channels(mapping)
    # Use BrainVision montage file to specify electrode positions.
    raw.set_montage("standard_1020")
    raw.save(raw_figures / f"{sub_input}_{condition}_{axis}_raw.fif", overwrite=True)  # here the data is saved as raw
    print(f'{condition} raw data saved. If raw is empty, make sure axis and condition are filled in correctly.')
    return raw


def get_events(raw, target_events):
    events = mne.events_from_annotations(raw)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in target_events.values()]
    events = np.array(filtered_events)
    return events


# 2. Interpolate
def interpolate(raw, condition):
    raw_interp = raw.copy().interpolate_bads(reset_bads=True)
    raw_interp.plot()
    raw.save(raw_figures / f"{sub}_{condition}_{axis}_interpolated.fif", overwrite=True)
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
    # todo detrend notch dss_line instead of bp and notch (Ole block)
    cfg["filtering"]["highpass"] = 1
    hi_filter = cfg["filtering"]["highpass"]
    lo_filter = cfg["filtering"]["lowpass"]

    raw_filtered = raw_filter.copy().filter(hi_filter, lo_filter)
    raw_filtered.plot()

    # plot the filtering
    grad_psd(raw, raw_filter, raw_filtered, fig_path)
    return raw, raw_filter, raw_filtered


# 5. Epochs
def epochs(target_ica, event_dict, events):
    # Counter(events1[:, 2])
    tmin = -0.2
    tmax = 0.7
    epoch_parameters = [tmin, tmax, event_dict]
    tmin, tmax, event_ids = epoch_parameters

    target_epochs = mne.Epochs(target_ica,
                               events,
                               event_id=event_ids,
                               tmin=tmin,
                               tmax=tmax,
                               detrend=0,
                               baseline=(-0.2, 0),  # should we set it here?
                               preload=True)
    return target_epochs


# 6. SOPHISITICATED RANSAC GOES HERE
def ransac(target_epochs, target, bads):
    epochs_clean = target_epochs.copy()
    cfg["reref"]["ransac"]["min_corr"] = 0.75
    ransac = Ransac(n_jobs=cfg["reref"]["ransac"]["n_jobs"], n_resample=cfg["reref"]["ransac"]["n_resample"],
                    min_channels=cfg["reref"]["ransac"]["min_channels"], min_corr=cfg["reref"]["ransac"]["min_corr"],
                    unbroken_time=cfg["reref"]["ransac"]["unbroken_time"])
    ransac.fit(epochs_clean)

    epochs_clean.average().plot(exclude=[])
    target_epochs.average().plot(exclude=[])

    if len(bads) != 0 and bads not in ransac.bad_chs_:
        ransac.bad_chs_.extend(bads)
    ransac.transform(epochs_clean)

    evoked = target_epochs.average()
    evoked_clean = epochs_clean.average()

    evoked.info['bads'] = ransac.bad_chs_
    evoked_clean.info['bads'] = ransac.bad_chs_

    fig, ax = plt.subplots(2, constrained_layout=True)
    evoked.plot(exclude=[], axes=ax[0], show=False)
    evoked_clean.plot(exclude=[], axes=ax[1], show=False)
    ax[0].set_title(f"Before RANSAC (bad chs:{ransac.bad_chs_})")
    ax[1].set_title("After RANSAC")
    fig.savefig(fig_path / f"{target} for RANSAC,{axis} {condition}.pdf", dpi=800)
    plt.close()
    return epochs_clean, ransac


# 7. REFERENCE TO THE AVERAGE
def reref(epochs_clean):
    epochs_reref = epochs_clean.copy()
    epochs_reref.set_eeg_reference(ref_channels='average')
    return epochs_reref


# 8. AUTOREJECT EPOCHS
def ar(epochs_reref, target, name):
    ar = AutoReject(n_interpolate=cfg["autoreject"]["n_interpolate"], n_jobs=cfg["autoreject"]["n_jobs"])
    ar = ar.fit(epochs_reref)
    epochs_ar, reject_log = ar.transform(epochs_reref, return_log=True)

    # target_epochs1[reject_log1.bad_epochs].plot(scalings=dict(eeg=100e-6))
    # reject_log1.plot('horizontal', show=False)

    # plot and save the final results
    fig, ax = plt.subplots(2, constrained_layout=True)
    epochs_ar.average().plot_image(titles=f"SNR:{snr(epochs_ar):.2f}", show=False, axes=ax[0])
    epochs_ar.average().plot(show=False, axes=ax[1])
    plt.savefig(fig_path / f"{target} for clean_evoked {axis} {condition}.pdf", dpi=800)
    plt.close()
    epochs_ar.save(results_path / 'epochs' / f"{target} for {name}-{axis},{condition}-epo.fif", overwrite=True)
    return epochs_ar


# 9. EVOKEDS
def get_evokeds(epochs_ar, event_ids):
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

    return evokeds


# Grand average response:
def grand_avg(evokeds, target, name):
    grand_average = mne.grand_average(evokeds)
    fig, ax = plt.subplots(figsize=(30 * cm, 15 * cm))  # Adjust the figure size as needed
    # Plot the grand average evoked response
    mne.viz.plot_compare_evokeds(
        {'Grand Average': grand_average},
        picks=['Cz'],
        combine='mean',
        title=f'{sub} - Grand Average Evoked Response',
        colors={'Grand Average': 'r'},
        linestyles={'Grand Average': 'solid'},
        axes=ax,
        show=True  # Set to True to display the plot immediately
    )
    plt.savefig(evokeds_folder / f"{name} {target} ERP from {condition}, {axis}, motor-only subtracted.pdf", dpi=800)
    plt.close()
    return grand_average


def s1_vs_s2(grand_average1, grand_average2, name):
    evokeds_total = {
        'Stim1': grand_average1,
        'Stim2': grand_average2
    }

    # Plot the grand averages
    fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the figure size as needed

    mne.viz.plot_compare_evokeds(evokeds_total, picks='Cz', axes=ax, colors={'Stim1': 'r', 'Stim2': 'b'})
    plt.title(f'{name} Grand Average Evoked Response for S1 and S2')
    plt.show()
    plt.savefig(evokeds_folder / f"{name} S1 vs S2 GRAND AVERAGE for {condition}, {axis}, motor-only subtracted.pdf", dpi=800)
    return evokeds_total


# Run pre-processing steps:

target_header_files_list, condition, axis = choose_header_files()

target_raw = get_raw_files(target_header_files_list, condition, axis)

events1 = get_events(target_raw, s1_events)
events2 = get_events(target_raw, s2_events)


target_raw.plot()  # to select bad channels, and crop out pauses
# target_raw.drop_channels(target_raw.info['bads'])
target_raw.plot_psd()
onsets = target_raw.annotations.onset
durations = target_raw.annotations.duration
descriptions = target_raw.annotations.description

# Find good segments
good_intervals = []
last_good_end = 0
for onset, duration, description in zip(onsets, durations, descriptions):
    if description == 'BAD_':
        good_intervals.append((last_good_end, onset))
        last_good_end = onset + duration
# Add the final good segment
good_intervals.append((last_good_end, target_raw.times[-1]))

# Crop and concatenate good segments
good_segments = [target_raw.copy().crop(tmin=start, tmax=end) for start, end in good_intervals]
target_raw = mne.concatenate_raws(good_segments)


# interpolate bad selected channels
target_interp = interpolate(target_raw, condition)

# get raw array, and info
target_data = mne.io.RawArray(data=target_interp.get_data(), info=target_interp.info)

target_raw, target_filter, target_filtered = filtering(target_interp, target_data)
target_filtered.save(results_path / 's1-ele, Raw filtered.fif', overwrite=True)
# 4. ICA
target_ica = target_filtered.copy()
ica = mne.preprocessing.ICA(n_components=cfg["ica"]["n_components"], method=cfg["ica"]["method"], random_state=99)
ica.fit(target_ica)
# target_ica.save('motor-only ICA eeg.fif', overwrite=True)
ica.plot_components()
# ica.save('motor-only ICA', overwrite=True)
ica.plot_sources(target_ica)
ica.apply(target_ica)
target_ica.save(results_path/'s1-ele ICA concatenated, motor-only subtraction-raw.fif')
tmin, tmax = -0.2, 0.7
time_window = 0.2
sfreq = target_raw.info['sfreq']
min_distance_samples = int(0.2 * sfreq)
def filter_non_overlapping(events_primary, events_secondary, min_distance_samples):
    non_overlapping = []
    last_event_time = -np.inf

    for event in events_primary:
        event_time = event[0]
        if all(abs(event_time - e[0]) > min_distance_samples for e in events_secondary) and (
                event_time - last_event_time > min_distance_samples):
            non_overlapping.append(event)
            last_event_time = event_time
    return np.array(non_overlapping)


# Filter non-overlapping events
events1_clean = filter_non_overlapping(events1, events2, min_distance_samples)
events2_clean = filter_non_overlapping(events2, events1, min_distance_samples)

target_epochs1 = epochs(target_ica, s1_events, events1_clean)  # stim1 epochs
target_epochs2 = epochs(target_ica, s2_events, events2_clean)  # stim 2 epochs


# check psd plot for shite channels and add to bads:
bads = []
epochs_clean1, ransac1 = ransac(target_epochs1, target='s1', bads=bads)
epochs_clean2, ransac2 = ransac(target_epochs2, target='s2', bads=bads)


epochs_reref1 = reref(epochs_clean1)
epochs_reref2 = reref(epochs_clean2)

reref_evkd1 = epochs_reref1.average()
reref_evkd2 = epochs_reref2.average()

fig, ax = plt.subplots(2, constrained_layout=True)
reref_evkd1.plot(axes=ax[0], show=False)
reref_evkd2.plot(axes=ax[1], show=False)
ax[0].set_title(f"Re-referenced cleaned epochs s1")
ax[1].set_title("Re-referenced cleaned epochs s2")
fig.savefig(fig_path / f"Re-referenced cleaned epochs, {axis} {condition}.pdf", dpi=800)

name = 'concatenated EEG'
epochs_ar1 = ar(epochs_reref1, target='s1', name=name)
epochs_ar2 = ar(epochs_reref2, target='s2', name=name)


evokeds1 = get_evokeds(epochs_ar1, s1_events)
evokeds2 = get_evokeds(epochs_ar2, s2_events)

grand_average1 = grand_avg(evokeds1, target='s1', name=name)
grand_average2 = grand_avg(evokeds2, target='s2', name=name)
# filter low-pass 25Hz:
grand_average1.filter(l_freq=None, h_freq=25)
grand_average2.filter(l_freq=None, h_freq=25)
# save ERPs:
concatenated_data = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/concatenated_data')
grand_average1.save(concatenated_data / '1-25Hz s1 target, elevation, motor-only subtraction stim1-ave.fif', overwrite=True)
grand_average2.save(concatenated_data / '1-25Hz s1 target, elevation, motor-only subtraction stim2-ave.fif', overwrite=True)

evokeds_total = s1_vs_s2(grand_average1, grand_average2, name=name)

full_paths = []
for folder in results_paths:
    path = str(folder)
    for file in os.listdir(folder):
        if '.fif' in file:
            if 'ele' in file:
                if 's1' in file:
                    full_path = f'{path}/{file}'
                    full_paths.append(full_path)

raws = []
for fif_file in full_paths:
    raws.append(mne.io.read_raw_fif(fif_file, preload=True))
exp_raw = mne.concatenate_raws(raws)
exp_raw.save(concatenated_data/'EEG elevation with motor-only ERP subtracted-raw.fif')
exp_raw = mne.io.read_raw_fif(concatenated_data / 'EEG elevation with motor-only ERP subtracted-raw.fif')
