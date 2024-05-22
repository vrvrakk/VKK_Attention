'''
### STEP 1: Concatenate block files to one raw file in raw_folder ###
### STEP 2: bandpass filtering of the data at 1=40 Hz ###
### STEP 3: Epoch the raw data and apply baseline ###
### STEP 4: Run RANSAC/ exclude bad channels ###
### STEP 5: Rereference the epochs ###
### STEP 6: Apply ICA ###
### STEP 7: Apply AutoReject ###
### STEP 8: Average epochs and write evokeds###
'''
# todo: TRF
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
cm = 1 / 2.54
# specify subject
sub_input = input("Give sub number as subn (n for number): ")
sub = [sub.strip() for sub in sub_input.split(',')]
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

# events:
markers_dict = {'s1_events':{'s1_1': 1, 's1_2': 2, 's1_3': 3, 's1_4': 4, 's1_5': 5, 's1_6': 6, 's1_8': 8, 's1_9': 9},  # stimulus 1 markers
's2_events':{'s2_1': 65, 's2_2': 66, 's2_3': 67, 's2_4': 68, 's2_5': 69, 's2_6': 70, 's2_8': 72, 's2_9': 73},  # stimulus 2 markers
'response_events':{'R_1': 129, 'R_2': 130, 'R_3': 131, 'R_4': 132, 'R_5': 133, 'R_6': 134, 'R_8': 136, 'R_9': 137}}  # response markers
s1_events = markers_dict['s1_events']
s2_events = markers_dict['s2_events']
response_events = markers_dict['response_events']

# config files
with open(json_path / "preproc_config.json") as file:
    cfg = json.load(file)
with open(json_path / "electrode_names.json") as file:
    mapping = json.load(file)

# find all .vhdr files in participant's folder


condition2 = input('Please provide condition2 (exp. EEG): ')
axis = input('Please provide axis (exp. EEG): ')


### STEP 0: Concatenate block files to one raw file in raw_folder
def choose_header_files(condition2=condition2, axis=axis):
    target_header_files_list = []
    for sub_dir in sub_dirs:
        header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
        filtered_files = [file for file in header_files if condition2 in file]
        filt_files = [file for file in filtered_files if axis in file]
        target_header_files_list.append(filt_files)
    return target_header_files_list, condition2, axis


target_header_files_list, condition2, axis = choose_header_files()


def get_raw_files(target_header_files_list, condition2, axis):
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
    raw.save(raw_figures / f"{sub_input}_{condition2}_{axis}_raw.fif", overwrite=True)  # here the data is saved as raw
    print(f'{condition2} raw data saved. If raw is empty, make sure axis and condition are filled in correctly.')
    return raw


target_raw = get_raw_files(target_header_files_list, condition2, axis)

# rename channels due to the trigger change
def get_events(raw, target_events):
    events = mne.events_from_annotations(raw)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in target_events.values()]
    events = np.array(filtered_events)
    return events

events = get_events(target_raw, s1_events)

# 2. Interpolate
def interpolate(raw, condition):
    raw_interp = raw.copy().interpolate_bads(reset_bads=True)
    raw_interp.plot()
    raw.save(raw_figures / f"{sub}_{condition}_{axis}_interpolated.fif", overwrite=True)
    return raw_interp


target_interp = interpolate(target_raw, condition2)
# get raw array, and info
target_data = mne.io.RawArray(data=target_interp.get_data(), info=target_interp.info)

# crop if necessary:
# start_time = int(input('Specify start_time: '))
# end_time = int(input('Specify start_time: '))
# raw.crop(tmin=start_time, tmax=end_time)


# 3. HIGH- AND LOW-PASS FILTER + POWER NOISE REMOVAL
def filtering(raw, data):
    cfg["filtering"]["notch"] = 50
    # remove the power noise
    raw_filter = raw.copy()
    raw_notch, iterations = dss.dss_line_iter(raw_filter.get_data().T, fline=cfg["filtering"]["notch"], sfreq=data.info["sfreq"],
                            nfft=cfg["filtering"]["nfft"])

    raw_filter._data = raw_notch.T
    #todo detrend notch dss_line instead of bp and notch (Ole block)
    cfg["filtering"]["highpass"] = 0.1
    hi_filter = cfg["filtering"]["highpass"]
    lo_filter = cfg["filtering"]["lowpass"]

    raw_filtered = raw_filter.copy().filter(hi_filter, lo_filter)
    raw_filtered.plot()

    # plot the filtering
    grad_psd(raw, raw_filter, raw_filtered, fig_path)
    return raw, raw_filter, raw_filtered


target_raw, target_filter, target_filtered = filtering(target_interp, target_data)

# 4. ICA
target_ica = target_filtered.copy()
ica = mne.preprocessing.ICA(n_components=cfg["ica"]["n_components"], method=cfg["ica"]["method"], random_state=99)
ica.fit(target_ica)
ica.plot_components()
ica.plot_sources(target_ica)
ica.apply(target_ica)


# 5. Epochs
# sanity check: number of different types of epochs
Counter(events[:, 2])
s1 = {'s1_1': 1, 's1_2': 2, 's1_3': 3, 's1_4': 4, 's1_5': 5, 's1_6': 6, 's1_8': 8, 's1_9': 9}
response = {'R_1': 129, 'R_2': 130, 'R_3': 131, 'R_4': 132, 'R_5': 133, 'R_6': 134, 'R_8': 136, 'R_9': 137}
tmin = -0.2
tmax = 0.7
epoch_parameters = [tmin, tmax, s1_events]
tmin, tmax, event_ids = epoch_parameters


target_epochs = mne.Epochs(target_ica,
                    events,
                    event_id=event_ids,
                    tmin=tmin,
                    tmax=tmax,
                    detrend=0,
                    baseline=(-0.2, 0),  # should we set it here?
                    preload=True)

# 6. SOPHISITICATED RANSAC GOES HERE
epochs_clean = target_epochs.copy()
cfg["reref"]["ransac"]["min_corr"] = 0.75
ransac = Ransac(n_jobs=cfg["reref"]["ransac"]["n_jobs"], n_resample=cfg["reref"]["ransac"]["n_resample"], min_channels=cfg["reref"]["ransac"]["min_channels"], min_corr=cfg["reref"]["ransac"]["min_corr"], unbroken_time=cfg["reref"]["ransac"]["unbroken_time"])
ransac.fit(epochs_clean)
epochs_clean.average().plot(exclude=[])
target_epochs.average().plot(exclude=[])
bads = []
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
fig.savefig(fig_path / f"RANSAC_results{condition2}.pdf", dpi=800)
plt.close()

# Re-reference:
# 7. REFERENCE TO THE AVERAGE
epochs_reref = epochs_clean.copy()
epochs_reref.set_eeg_reference(ref_channels='average')

# 8. AUTOREJECT EPOCHS
ar = AutoReject(n_interpolate=cfg["autoreject"]["n_interpolate"], n_jobs=cfg["autoreject"]["n_jobs"])
ar.fit(epochs_reref)
epochs_ar, reject_log = ar.transform(epochs_reref, return_log=True)
target_epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
reject_log.plot('horizontal', show=False)

# plot and save the final results
fig, ax = plt.subplots(2, constrained_layout=True)
epochs_ar.average().plot_image(titles=f"SNR:{snr(epochs_ar):.2f}", show=False, axes=ax[0])
epochs_ar.average().plot(show=False, axes=ax[1])
plt.savefig(fig_path / f"clean_evoked_{condition2}.pdf", dpi=800)
plt.close()
epochs_ar.save(results_path / 'epochs' / f"{sub}-{condition2}-epo.fif", overwrite=True)


# 9. EVOKEDS

epochs = epochs_ar.copy()
event_ids = list(event_ids.values())
evokeds = []
for event_id in event_ids:
    evoked = epochs[list(np.where(epochs.events[:, 2] == event_id)[0])].average()
    evokeds.append(evoked)

# Grand average response:
grand_average = mne.grand_average(evokeds)
picks = mne.pick_types(grand_average.info, eeg=True, meg=False)
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
plt.savefig(evokeds_folder / f"evoked across conditions mean Cz {condition2}.pdf", dpi=800)
plt.close()
