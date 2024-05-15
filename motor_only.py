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
sub = input("Give sub number as subn (n for number): ")
default_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data')
raw_dir = default_dir / 'eeg' / 'raw'
sub_dir = raw_dir / sub
json_path = default_dir / 'misc'
fig_path = default_dir / 'eeg' / 'preprocessed' / 'figures' / sub
results_path = default_dir / 'eeg' / 'preprocessed' / 'results' / sub
epochs_folder = results_path / "epochs"
evokeds_folder = results_path / 'evokeds'
raw_figures = sub_dir / 'figures'

# events:
markers_dict = {'s1_events':{'s1_1': 1, 's1_2': 2, 's1_3': 3, 's1_4': 4, 's1_5': 5, 's1_6': 6, 's1_8': 8, 's1_9': 9},  # stimulus 1 markers
's2_events':{'s2_1': 65, 's2_2': 66, 's2_3': 67, 's2_4': 68, 's2_5': 69, 's2_6': 70, 's2_8': 72, 's2_9': 73},  # stimulus 2 markers
'response_events':{'R_1': 129, 'R_2': 130, 'R_3': 131, 'R_4': 132, 'R_5': 133, 'R_6': 134, 'R_8': 136, 'R_9': 137}}  # response markers
s1_events = markers_dict['s1_events']
s2_events = markers_dict['s2_events']
response_events = markers_dict['response_events']
# 0. LOAD THE DATA
for folder in sub_dir, fig_path, results_path, epochs_folder, evokeds_folder, raw_figures:
    if not os.path.isdir(folder):
        os.makedirs(folder)

# config files
with open(json_path / "preproc_config.json") as file:
    cfg = json.load(file)
with open(json_path / "electrode_names.json") as file:
    mapping = json.load(file)

# find all .vhdr files in participant's folder

condition = input('Please provide condition: ')
axis = input('Please provide axis: ')
axis = None if axis.lower() == "none" or axis == "" else axis
def choose_header_files(condition=condition, axis=axis):
    header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
    header_files = [file for file in header_files if condition in file]
    if axis is not None:
        header_files = [file for file in header_files if axis in file]
    return header_files, condition, axis


header_files, condition, axis = choose_header_files()


def get_raw_files(header_files, condition, axis):
    raw_files = []
    for header_file in header_files:
        if len(header_files) > 1:
            raw_files.append(mne.io.read_raw_brainvision(os.path.join(sub_dir, header_file), preload=True))
            raw = mne.concatenate_raws(raw_files)# read BrainVision files.
        else:
            print(f'Only one header file found: {header_file}. Cannot append.')
            raw = mne.io.read_raw_brainvision(os.path.join(sub_dir, header_file), preload=True)
    # append all files from a participant
    raw.rename_channels(mapping)
    # Use BrainVision montage file to specify electrode positions.
    montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")
    raw.set_montage(montage)
    raw.save(raw_figures / f"{sub}_{condition}_{axis}_raw.fif", overwrite=True)  # here the data is saved as raw
    return raw

raw = get_raw_files(header_files, condition, axis)
# 1. CLEANING THE DATA

# rename channels due to the trigger change
def get_events(raw, s1_events, s2_events, response_events):
    events = mne.events_from_annotations(raw)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in s1_events.values()]
    events = np.array(filtered_events)
    return events

events = get_events(raw, s1_events, s2_events, response_events)

# get raw array, and info
data = mne.io.RawArray(data=raw.get_data(), info=raw.info)

# 2. REMOVE CRAZY CHANNELS
raw.plot()  # mark bad channels
# message = input('Do you need to crop something?: y/n: ')
# if message == 'y':
#     try:
#     start_time = int(input('Specify start_time: '))
#     end_time = int(input('Specify start_time: '))
#     raw.crop(tmin=start_time, tmax=end_time)
# else:
#     print('No cutoffs necessary.')
# start_time = 31.409
# raw.crop(tmin=start_time)
raw_interp = raw.copy().interpolate_bads(reset_bads=True)
raw_interp.plot()

# 3. HIGH- AND LOW-PASS FILTER + POWER NOISE REMOVAL
cfg["filtering"]["notch"] = 50
# remove the power noise
raw_filter = raw_interp.copy()
raw_notch, iterations = dss.dss_line_iter(raw_filter.get_data().T, fline=cfg["filtering"]["notch"], sfreq=data.info["sfreq"],
                        nfft=cfg["filtering"]["nfft"])

raw_filter._data = raw_notch.T

hi_filter = cfg["filtering"]["highpass"]
lo_filter = cfg["filtering"]["lowpass"]

raw_filtered = raw_filter.copy().filter(hi_filter, lo_filter)
raw_filtered.plot()
raw_filtered_25 = raw_filtered.copy()
raw_filtered_25.filter(l_freq=2, h_freq=25)
raw_filtered_25.plot()
# plot the filtering
grad_psd(raw_interp, raw_filter, raw_filtered, fig_path)
# grad_psd(raw_interp, raw_filter, raw_filtered_25, fig_path)
del raw, raw_interp, raw_filter


# 4. ICA
raw_ica = raw_filtered.copy()
ica = mne.preprocessing.ICA(n_components=cfg["ica"]["n_components"], method=cfg["ica"]["method"], random_state=99)
ica.fit(raw_ica)

ica.plot_components()
ica.plot_sources(raw_ica)
ica.apply(raw_ica)
raw_ica.plot()

# 5. REFERENCE TO THE AVERAGE

raw_reref = raw_ica.copy()
raw_reref.set_eeg_reference(ref_channels='average')
picks_eeg = mne.pick_types(raw_filtered.info, meg=False, eeg=True, eog=False, stim=False,
                       exclude='bads')

# 6. EPOCHING THE DATA

# sanity check: number of different types of epochs
Counter(events[:, 2])
s1 = {'s1_1': 1, 's1_2': 2, 's1_3': 3, 's1_4': 4, 's1_5': 5, 's1_6': 6, 's1_8': 8, 's1_9': 9}
response = {'R_1': 129, 'R_2': 130, 'R_3': 131, 'R_4': 132, 'R_5': 133, 'R_6': 134, 'R_8': 136, 'R_9': 137}
tmin = -0.2
tmax = 1.5
epoch_parameters = [-0.2,  # tmin
                    1.5,  # tmax
                    s1_events]
tmin, tmax, event_ids = epoch_parameters

epochs = mne.Epochs(raw_reref,
                    events,
                    event_id=event_ids,
                    tmin=tmin,
                    tmax=tmax,
                    detrend=0,
                    baseline=None,  # should we set it here?
                    preload=True)

epochs.plot()
# 7. SOPHISITICATED RANSAC GOES HERE
epochs_clean = epochs.copy()
ransac = Ransac(n_jobs=cfg["reref"]["ransac"]["n_jobs"], n_resample=cfg["reref"]["ransac"]["n_resample"], min_channels=cfg["reref"]["ransac"]["min_channels"],
                min_corr=cfg["reref"]["ransac"]["min_corr"], unbroken_time=cfg["reref"]["ransac"]["unbroken_time"])
ransac.fit(epochs_clean)
epochs_clean.average().plot(exclude=[])
bads = []
if len(bads) != 0 and bads not in ransac.bad_chs_:
    ransac.bad_chs_.extend(bads)
epochs_clean = ransac.transform(epochs_clean)

evoked = epochs.average()
evoked_clean = epochs_clean.average()

evoked.info['bads'] = ransac.bad_chs_
evoked_clean.info['bads'] = ransac.bad_chs_

fig, ax = plt.subplots(2, constrained_layout=True)
evoked.plot(exclude=[], axes=ax[0], show=False)
evoked_clean.plot(exclude=[], axes=ax[1], show=False)
ax[0].set_title(f"Before RANSAC (bad chs:{ransac.bad_chs_})")
ax[1].set_title("After RANSAC")
fig.savefig(fig_path / "RANSAC_results.pdf", dpi=800)
plt.close()

snr_pre = snr(epochs)
snr_post = snr(epochs_clean)
print(snr_pre, snr_post)

# 8. AUTOREJECT EPOCHS

ar = AutoReject(n_interpolate=cfg["autoreject"]["n_interpolate"], n_jobs=cfg["autoreject"]["n_jobs"])
ar.fit(epochs_clean)
epochs_ar, reject_log = ar.transform(epochs_clean, return_log=True)

epochs_ar.plot()
epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
reject_log.plot('horizontal', show=False)


epochs_ar.apply_baseline((-0.2, 0))

# plot and save the final results
fig, ax = plt.subplots(2, constrained_layout=True)
epochs_ar.average().plot_image(titles=f"SNR:{snr(epochs_ar):.2f}", show=False, axes=ax[0])
epochs_ar.average().plot(show=False, axes=ax[1])
plt.savefig(fig_path / "clean_evoked.pdf", dpi=800)
plt.close()
epochs_ar.save(results_path / 'epochs' / f"{sub}-{condition}-epo.fif", overwrite=True)

epochs = epochs_ar.copy()
del epochs_clean, evoked, evoked_clean, epochs_ar

# 9. EVOKEDS

event_ids = list(event_ids.values())
evokeds = []
for event_id in event_ids:
    evoked = epochs[list(np.where(epochs.events[:, 2] == event_id)[0])].average()
    evokeds.append(evoked)

fig, axes = plt.subplots(3, figsize=(30 * cm, 30 * cm))

# Plot the first three evoked responses
for i, evoked in enumerate(evokeds[:3]):
    mne.viz.plot_compare_evokeds(
        {f's{i+1}': evoked},
        picks=['Fz'],
        combine="mean",
        title=f'{sub} - Averaged evoked for condition {i+1}',
        colors={f's{i+1}': 'g'},
        linestyles={f's{i+1}': 'solid'},
        axes=axes[i],
        show=False
    )

# Adjust the layout
fig.tight_layout()

plt.show()

plt.savefig(fig_path / "evoked across conditions.pdf", dpi=800)
plt.close()
#todo: use baseline recording to remove noise from audio eeg
#todo: try with both eeg as well
# filter data, interpolate (audio eeg), remove bad channels, epoch, calculate mean of baseline epochs, subtract noise from each audio epoch
# check epochs, apply ICA, ransac, autoreject, evokeds





##################################################
##################################################
##################################################
##################################################

'''
raw.plot_psd_topomap()
std_dev = raw.get_data().std(axis=1)

# Plot these as a topomap
mne.viz.plot_topomap(std_dev, raw.info, show_names=True)
channel = 'Fz'
raw.plot(order=[raw.ch_names.index(channel)], n_channels=1, scalings='auto')

# Plot the PSD for a specific channel
raw.plot_psd(picks=[channel])

raw.plot_sensors(kind='topomap', show_names=True)

# view spectrum of specific channels:
channel_name = 'C1'
channel_index = raw.ch_names.index(channel_name)
# Get the data for just this channel
data, times = raw[channel_index, :]
# Compute the PSD using Welch's method
f, Pxx = signal.welch(data.flatten(), fs=raw.info['sfreq'], nperseg=2048, return_onesided=True)
# Plot the PSD
plt.figure(figsize=(10, 5))
plt.semilogy(f, Pxx, label=f'PSD of {channel_name}')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Power Spectral Density (V^2/Hz)')
plt.title(f'Power Spectral Density (PSD) {channel_name}')
plt.legend()
plt.show()
'''