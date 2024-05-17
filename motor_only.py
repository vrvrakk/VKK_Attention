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
fig_path = default_dir / 'eeg' / 'preprocessed' / 'results' / sub / 'figures'
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

condition1 = input('Please provide condition1 (baseline): ')
condition2 = input('Please provide condition2 (exp. EEG): ')
axis = input('Please provide axis (exp. EEG): ')


### STEP 0: Concatenate block files to one raw file in raw_folder
def choose_header_files(condition1=condition1, condition2=condition2 ,axis=axis):
    axis = None if axis.lower() == "none" or axis == "" else axis
    header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
    baseline_header_files = [file for file in header_files if condition1 in file]
    target_header_files = [file for file in header_files if condition2 in file]
    if axis is not None:
        target_header_files = [file for file in header_files if axis in file]
    return baseline_header_files, target_header_files, condition1, condition2, axis


baseline_header_files, target_header_files, condition1, condition2, axis = choose_header_files()


def get_raw_files(baseline_header_files, target_header_files, condition1, condition2, axis):
    def read_and_concatenate(header_files, condition, axis):
        raw_files = []
        for header_file in header_files:
            if len(header_files) > 1:
                raw_files.append(mne.io.read_raw_brainvision(os.path.join(sub_dir, header_file), preload=True))
                raw = mne.concatenate_raws(raw_files)  # read BrainVision files.
            else:
                print(f'Only one header file found: {header_file}. Cannot append.')
                raw = mne.io.read_raw_brainvision(os.path.join(sub_dir, header_file), preload=True)
        # append all files from a participant
        raw.rename_channels(mapping)
        # Use BrainVision montage file to specify electrode positions.
        montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")
        raw.set_montage(montage)
        raw.save(raw_figures / f"{sub}_{condition}_{axis}_raw.fif", overwrite=True)  # here the data is saved as raw
        print(f'{condition} raw data saved. If raw is empty, make sure axis and condition are filled in correctly.')
        return raw
    baseline_raw = read_and_concatenate(baseline_header_files, condition1, axis)
    target_raw = read_and_concatenate(target_header_files, condition2, axis)
    return baseline_raw, target_raw

baseline_raw, target_raw = get_raw_files(baseline_header_files, target_header_files, condition1, condition2, axis)

# rename channels due to the trigger change
def get_events(raw, s1_events, s2_events, response_events):
    events = mne.events_from_annotations(raw)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in s1_events.values()]
    events = np.array(filtered_events)
    return events

events = get_events(target_raw, s1_events, s2_events, response_events)

# get raw array, and info
# baseline_data = mne.io.RawArray(data=baseline_raw.get_data(), info=baseline_raw.info)
target_data = mne.io.RawArray(data=target_raw.get_data(), info=target_raw.info)
# mark bad channels
# baseline_raw.plot()
target_raw.plot()

# crop if necessary:
# start_time = int(input('Specify start_time: '))
# end_time = int(input('Specify start_time: '))
# raw.crop(tmin=start_time, tmax=end_time)

# 1. HIGH- AND LOW-PASS FILTER + POWER NOISE REMOVAL
def filtering(raw, data):
    cfg["filtering"]["notch"] = 50
    # remove the power noise
    raw_filter = raw.copy()
    raw_notch, iterations = dss.dss_line_iter(raw_filter.get_data().T, fline=cfg["filtering"]["notch"], sfreq=data.info["sfreq"],
                            nfft=cfg["filtering"]["nfft"])

    raw_filter._data = raw_notch.T

    hi_filter = cfg["filtering"]["highpass"]
    lo_filter = cfg["filtering"]["lowpass"]

    raw_filtered = raw_filter.copy().filter(hi_filter, lo_filter)
    raw_filtered.plot()
    # raw_filtered_25 = raw_filtered.copy()
    # raw_filtered_25.filter(l_freq=2, h_freq=25)
    # raw_filtered_25.plot()

    # plot the filtering
    grad_psd(raw, raw_filter, raw_filtered, fig_path)
    return raw, raw_filter, raw_filtered


# baseline_raw, baseline_filter, baseline_filtered = filtering(baseline_raw, baseline_data)
target_raw, target_filter, target_filtered = filtering(target_raw, target_data)


def interpolate(raw_filtered, condition):
    raw_interp = raw_filtered.copy().interpolate_bads(reset_bads=True)
    raw_interp.plot()
    raw_filtered.save(raw_figures / f"{sub}_{condition}_{axis}_interpolated.fif", overwrite=True)
    return raw_interp

# baseline_interp = interpolate(baseline_filtered, condition1)
target_interp = interpolate(target_filtered, condition2)

# 3b. Epoch baseline manually:
def epoch_baseline(baseline_interp):
    tmin, tmax = -0.2, 1.5
    epoch_duration = tmax - tmin
    n_samples = len(baseline_interp.times)
    epoch_length = int(epoch_duration * baseline_interp.info['sfreq'])
    step_size = epoch_length  # No overlap between epochs

    events_baseline = np.array([[i, 0, 1] for i in range(0, n_samples - epoch_length, step_size)])

    baseline_epochs = mne.Epochs(baseline_interp, events_baseline, event_id=1, tmin=0, tmax=epoch_duration, baseline=None, preload=True)
    baseline_epochs.plot()
    return baseline_epochs

# baseline_epochs = epoch_baseline(baseline_interp)

# 3c. Get noise profile from baseline EEG:
# baseline_data = baseline_epochs.get_data()
# noise_profile = np.mean(baseline_data, axis=0)

# 4. EPOCHING THE TARGET DATA

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

target_epochs = mne.Epochs(target_interp,
                    events,
                    event_id=event_ids,
                    tmin=tmin,
                    tmax=tmax,
                    detrend=0,
                    baseline=(-0.2, 0),  # should we set it here?
                    preload=True)

target_epochs.plot()
# 5. Apply noise profile on target EEG:
# stimuli_data = target_epochs.get_data()
# cleaned_stimuli_data = stimuli_data - noise_profile  # Remove noise by subtracting the noise profile
# # Create new Epochs object with cleaned data
# cleaned_target_epochs = mne.EpochsArray(cleaned_stimuli_data, target_epochs.info, events, tmin=tmin, event_id=event_ids, baseline=(None, 0))
# cleaned_target_epochs.plot()

# 7. SOPHISITICATED RANSAC GOES HERE
epochs_clean = target_epochs.copy()
ransac = Ransac(n_jobs=cfg["reref"]["ransac"]["n_jobs"], n_resample=cfg["reref"]["ransac"]["n_resample"], min_channels=cfg["reref"]["ransac"]["min_channels"],
                min_corr=cfg["reref"]["ransac"]["min_corr"], unbroken_time=cfg["reref"]["ransac"]["unbroken_time"])
ransac.fit(epochs_clean)
epochs_clean.average().plot(exclude=[])
target_epochs.average().plot(exclude=[])
bads = []
if len(bads) != 0 and bads not in ransac.bad_chs_:
    ransac.bad_chs_.extend(bads)
epochs_clean = ransac.transform(epochs_clean)

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

snr_pre = snr(target_epochs)
snr_post = snr(epochs_clean)
print(snr_pre, snr_post)


# 7. REFERENCE TO THE AVERAGE
epochs_reref = epochs_clean.copy()
epochs_reref.set_eeg_reference(ref_channels='average')
picks_eeg = mne.pick_types(target_filtered.info, meg=False, eeg=True, eog=False, stim=False,
                       exclude='bads')

snr_pre = snr(epochs_clean)
snr_post = snr(epochs_reref)
print(snr_pre, snr_post)
epochs_reref.plot()


# 6. ICA
epochs_ica = epochs_reref.copy()
# baseline_ica = baseline_epochs.copy()
ica = mne.preprocessing.ICA(n_components=cfg["ica"]["n_components"], method=cfg["ica"]["method"], random_state=99)
ica.fit(epochs_ica)

ica.plot_components()
ica.plot_sources(epochs_ica)
ica.apply(epochs_ica)
epochs_ica.plot()

snr_pre = snr(epochs_reref)
snr_post = snr(epochs_ica)
print(snr_pre, snr_post)
# 8. AUTOREJECT EPOCHS

ar = AutoReject(n_interpolate=cfg["autoreject"]["n_interpolate"], n_jobs=cfg["autoreject"]["n_jobs"])
ar.fit(epochs_ica)
epochs_ar, reject_log = ar.transform(epochs_ica, return_log=True)

epochs_ar.plot()
target_epochs[reject_log.bad_epochs].plot(scalings=dict(eeg=100e-6))
reject_log.plot('horizontal', show=False)

# epochs_ar.apply_baseline((None, 0))

# plot and save the final results
fig, ax = plt.subplots(2, constrained_layout=True)
epochs_ar.average().plot_image(titles=f"SNR:{snr(epochs_ar):.2f}", show=False, axes=ax[0])
epochs_ar.average().plot(show=False, axes=ax[1])
plt.savefig(fig_path / f"clean_evoked_{condition2}.pdf", dpi=800)
plt.close()
epochs_ar.save(results_path / 'epochs' / f"{sub}-{condition2}-epo.fif", overwrite=True)

epochs = epochs_ar.copy()
# bp filter:
epochs.filter(l_freq=2, h_freq=25)
fig, ax = plt.subplots(2, constrained_layout=True)
epochs.average().plot_image(titles=f"SNR:{snr(epochs):.2f}", show=False, axes=ax[0])
epochs.average().plot(show=False, axes=ax[1])
plt.savefig(fig_path / f"bp_filtered_clean_evoked_{condition2}.pdf", dpi=800)
plt.close()
epochs_ar.save(results_path / 'epochs' / f"{sub}-{condition2}-epo_bp_filt.fif", overwrite=True)

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
        picks=['CP4'],
        combine="mean",
        title=f'{sub} - Averaged evoked for stim {i+1}',
        colors={f's{i+1}': 'g'},
        linestyles={f's{i+1}': 'solid'},
        axes=axes[i],
        show=False
    )


plt.savefig(fig_path / f"evoked across conditions for {condition2}.pdf", dpi=800)
plt.close()

# Grand average response:
grand_average = mne.grand_average(evokeds)

fig, ax = plt.subplots(figsize=(30 * cm, 15 * cm))  # Adjust the figure size as needed
# Plot the grand average evoked response
mne.viz.plot_compare_evokeds(
    {'Grand Average': grand_average},
    picks=['Fp2', 'F2', 'F4', 'AF4'],
    combine="mean",
    title=f'{sub} - Grand Average Evoked Response',
    colors={'Grand Average': 'r'},
    linestyles={'Grand Average': 'solid'},
    axes=ax,
    show=True  # Set to True to display the plot immediately
)
plt.savefig(fig_path / f"evoked across conditions_{condition2}.pdf", dpi=800)
plt.close()
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