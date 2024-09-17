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
name = sub_input
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
    raw_fif = sub_dir / 'raw files'
    for folder in sub_dir, fig_path, results_path, epochs_folder, evokeds_folder, raw_fif:
        if not os.path.isdir(folder):
            os.makedirs(folder)

# events:
markers_dict = {
    's1_events': {  # Stimulus 1 markers
        'Stimulus/S 1': 1,
        'Stimulus/S 2': 2,
        'Stimulus/S 3': 3,
        'Stimulus/S 4': 4,
        'Stimulus/S 5': 5,
        'Stimulus/S 6': 6,
        'Stimulus/S 8': 8,
        'Stimulus/S 9': 9
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

# config files
with open(json_path / "preproc_config.json") as file:
    cfg = json.load(file)
# load electrode names file:
with open(json_path / "electrode_names.json") as file: #electrode_names - Copy
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

### STEP 0: Concatenate block files to one raw file in raw_folder
def choose_header_files(condition=condition):
    target_header_files_list = []
    for sub_dir in sub_dirs:
        header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
        filtered_files = [file for file in header_files if condition in file]
        if filtered_files:
            target_header_files_list.append(filtered_files)
    return target_header_files_list, condition

def get_raw_files(target_header_files_list, condition):
    raw_files = []
    for sub_dir, header_files in zip(sub_dirs, target_header_files_list):
        for header_file in header_files:
            full_path = os.path.join(sub_dir, header_file)
            print(full_path)
            raw_files.append(mne.io.read_raw_brainvision(full_path, preload=True))
    raw = mne.concatenate_raws(raw_files)  # read BrainVision files.
    # append all files from a participant
    raw.rename_channels(mapping)
    raw.set_montage('standard_1020')
    raw.save(raw_fif / f"{name}_{condition}_raw.fif", overwrite=True)  # here the data is saved as raw
    print(f'{condition} raw data saved. If raw is empty, make sure axis and condition are filled in correctly.')
    return raw


def get_events(raw, target_events):  # if a1 or e1: choose s1_events; if a2 or e2: s2_events
    events = mne.events_from_annotations(raw)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in target_events.values()]
    events = np.array(filtered_events)
    return events


# 2. Interpolate
def interpolate(raw, condition):
    raw_interp = raw.copy().interpolate_bads(reset_bads=True)
    raw_interp.plot()
    raw.save(raw_fif / f"{sub}_{condition}_interpolated-raw.fif", overwrite=True)
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
    lo_filter = cfg["filtering"]["lowpass"] = 25
    hi_filter = cfg["filtering"]["highpass"]
    lo_filter = cfg["filtering"]["lowpass"]

    raw_filtered = raw_filter.copy().filter(hi_filter, lo_filter)
    raw_filtered.plot()

    # plot the filtering
    grad_psd(raw, raw_filter, raw_filtered, fig_path)
    return raw, raw_filter, raw_filtered


# Run pre-processing steps:
target_header_files_list, condition = choose_header_files()
target_raw = get_raw_files(target_header_files_list, condition)

# drop EMG channels
target_raw.drop_channels(['A1', 'A2', 'M2'])


events1 = get_events(target_raw, s1_events)
events2 = get_events(target_raw, s2_events)
events3 = get_events(target_raw, response_events)

# to select bad channels, and select bad segmenmts:
target_raw.plot()

target_raw.plot_psd()

# get annotations info:
# make sure to check if annotation of bad segments is correct: BAD_, BAD boundary, BAD_boundary
def bad_segments(target_raw):
    onsets = target_raw.annotations.onset
    durations = target_raw.annotations.duration
    descriptions = target_raw.annotations.description

    # Find good segments
    good_intervals = []
    last_good_end = 0
    for onset, duration, description in zip(onsets, durations, descriptions):
        if description == 'BAD boundary' or 'BAD_' in descriptions:
            # description name may vary for each file (Bad boundary)
            good_intervals.append((last_good_end, onset))
            last_good_end = onset + duration
    # Add the final good segment
    good_intervals.append((last_good_end, target_raw.times[-1]))

    # Crop and concatenate good segments
    good_segments = [target_raw.copy().crop(tmin=start, tmax=end) for start, end in good_intervals]
    target_raw = mne.concatenate_raws(good_segments)
    return target_raw

target_raw = bad_segments(target_raw)

# interpolate bad selected channels, after removing significant noise affecting many electrodes
target_interp = interpolate(target_raw, condition)

# get raw array, and info for filtering
target_data = mne.io.RawArray(data=target_interp.get_data(), info=target_interp.info)


# Filter: bandpas 1-25Hz
target_raw, target_filter, target_filtered = filtering(target_interp, target_data)
# baseline_raw, baseline_filter, baseline_filtered = filtering(baseline_interp, baseline_data)
target_filtered.save(results_path / f'1-25Hz_{name}_conditions_{condition}-raw.fif', overwrite=True)

############ subtract motor noise:

padded_evoked = mne.read_evokeds(f'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/raw/{sub_input}/motor-only/Smoothed 1-25Hz Grand Average Motor-ave.fif')
sfreq = target_filtered.info['sfreq']
erp_duration = padded_evoked[0].times[-1] - padded_evoked[0].times[0]
n_samples_erp = len(padded_evoked[0].times)

# Subtract the ERP at each event time
for event in events3:
    event_sample = event[0]  # sample number of the event
    start_sample = event_sample - int(padded_evoked[0].times[0] * sfreq)
    end_sample = start_sample + n_samples_erp

    # Check if the event is within the bounds of the raw data
    if start_sample >= 0 and end_sample <= len(target_filtered.times):
        # Subtract the ERP data from the raw data
        target_filtered._data[:, start_sample:end_sample] -= padded_evoked[0].data

target_filtered.save(results_path / f'motor subtracted 1-25Hz for {name}_conditions_{condition}-raw.fif', overwrite=True)

# save cleaned eeg file
# load all relevant eeg files
# run rest

# 4. ICA
target_ica = target_filtered.copy()
ica = mne.preprocessing.ICA(n_components=cfg["ica"]["n_components"], method=cfg["ica"]["method"], random_state=99)
ica.fit(target_ica)
ica.plot_components()
# ica.save('motor-only ICA', overwrite=True)
ica.plot_sources(target_ica)
ica.apply(target_ica)
target_ica.save(results_path / f'{name}_{condition}_ICA-raw.fif', overwrite=True)


# 5. Epochs
def epochs(target_ica, event_dict, events):
    # Counter(events1[:, 2])
    tmin = -0.3
    tmax = 0.5
    epoch_parameters = [tmin, tmax, event_dict]
    tmin, tmax, event_ids = epoch_parameters
    event_ids = {key: val for key, val in event_ids.items() if val in events[:, 2]}

    target_epochs = mne.Epochs(target_ica,
                               events,
                               event_id=event_ids,
                               tmin=tmin,
                               tmax=tmax,
                               baseline=(None, 0),
                               detrend=0,  # should we set it here?
                               preload=True)
    return target_epochs

sfreq = 500.0
tmin, tmax = -0.3, 0.5
time_window = 0.2
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
target_epochs3 = epochs(target_ica, response_events, events3)

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
    fig.savefig(results_path / 'figures' / f"{target}_RANSAC_{condition}.pdf", dpi=800)
    plt.close()
    return epochs_clean, ransac
bads = []
epochs_clean1, ransac1 = ransac(target_epochs1, target='s1', bads=bads)
epochs_clean2, ransac2 = ransac(target_epochs2, target='s2', bads=bads)
epochs_clean3, ransac3 = ransac(target_epochs3, target='buttons', bads=bads)
# 7. REFERENCE TO THE AVERAGE
def reref(epochs_clean):
    epochs_reref = epochs_clean.copy()
    epochs_reref.set_eeg_reference(ref_channels='average')
    return epochs_reref

epochs_reref1 = reref(epochs_clean1)
epochs_reref2 = reref(epochs_clean2)
epochs_reref3 = reref(epochs_clean3)


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
    plt.savefig(results_path / 'figures' / f"{target} for clean_evoked {condition}.pdf", dpi=800)
    plt.close()
    epochs_ar.save(results_path / 'epochs' / f"{target}_{name}_conditions_{condition}-epo.fif",
                   overwrite=True)
    return epochs_ar

# after autoReject stops running, ignore the FutureWarning!
epochs_ar1 = ar(epochs_reref1, target='s1', name=name)
epochs_ar2 = ar(epochs_reref2, target='s2', name=name)
epochs_ar3 = ar(epochs_reref3, target='responses', name=name)


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


evokeds1 = get_evokeds(epochs_ar1, s1_events)
evokeds2 = get_evokeds(epochs_ar2, s2_events)
evokeds3 = get_evokeds(epochs_ar3, response_events)
# Grand average response:
def grand_avg(evokeds, target, name):
    grand_average = mne.grand_average(evokeds)
    fig, ax = plt.subplots(figsize=(30 * cm, 15 * cm))  # Adjust the figure size as needed
    # Plot the grand average evoked response
    mne.viz.plot_compare_evokeds(
        {'Grand Average': grand_average},
        combine='mean',
        title=f'{sub_input} - Grand Average Evoked Response {target}',
        colors={'Grand Average': 'r'},
        linestyles={'Grand Average': 'solid'},
        axes=ax,
        show=True  # Set to True to display the plot immediately
    )
    plt.savefig(
        results_path / 'evokeds' / f"ERP_{name}_{target}_{condition}.pdf",
        dpi=800)
    plt.close()
    return grand_average

grand_average1 = grand_avg(evokeds1, target='s1', name=name)
grand_average2 = grand_avg(evokeds2, target='s2', name=name)
grand_average3 = grand_avg(evokeds3, target='responses', name=name)

def s1_vs_s2(grand_average1, grand_average2, name):
    evokeds_total = {
        'Stim1': grand_average1,
        'Stim2': grand_average2
    }

    # Plot the grand averages
    fig, ax = plt.subplots(figsize=(10, 5))  # Adjust the figure size as needed

    mne.viz.plot_compare_evokeds(evokeds_total, picks=['Cz'],axes=ax, colors={'Stim1': 'r', 'Stim2': 'b'})
    plt.title(f'{name} Grand Average Evoked Response for S1 and S2')
    save_path = results_path / 'evokeds' / f"S1_vs_S2_GRAND_AVERAGE_from_{name}_conditions_{condition}.pdf"
    plt.savefig(save_path, dpi=800)
    plt.show()

    return evokeds_total

evokeds_total = s1_vs_s2(grand_average1, grand_average2, name=name)