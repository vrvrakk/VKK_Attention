# libraries:
import mne
import pathlib
import os
import numpy as np
from autoreject import AutoReject, Ransac
from collections import Counter
import json
from meegkit import dss
from matplotlib import pyplot as plt, patches
from helper import grad_psd, snr
cm = 1 / 2.54

# 0. LOAD THE DATA

cwd = os.getcwd()
DIR = pathlib.Path(os.getcwd())
sub = "sub17"
condition = "main"

# specify participant's folder
data_DIR = DIR / "data" / sub
fig_path =DIR / "plots" / sub
results_path= DIR / "results" / sub
epochs_folder = DIR / "results" / sub / "epochs"
raw_folder = DIR / "results" / sub / "raw_data"
evokeds_folder = DIR / "results" / sub / "evokeds"

for folder in data_DIR, fig_path, results_path, epochs_folder, raw_folder, evokeds_folder:
    if not os.path.isdir(folder):
        os.makedirs(folder)

# config files
with open(DIR / "settings" / "preproc_config.json") as file:
    cfg = json.load(file)
with open(DIR / "settings" / "mapping.json") as file:
    mapping = json.load(file)

# find all .vhdr files in participant's folder
header_files = [file for file in os.listdir(data_DIR) if ".vhdr" in file]
header_files = [file for file in header_files if condition in file]
raw_files = []
for header_file in header_files:
    raw_files.append(mne.io.read_raw_brainvision(os.path.join(data_DIR, header_file), preload=True))  # read BrainVision files.

# append all files from a participant
raw = mne.concatenate_raws(raw_files)
raw.rename_channels(mapping)
# Use BrainVision montage file to specify electrode positions.
#montage_path = DIR / "settings" / cfg["montage"]["name"]
#montage = mne.channels.read_custom_montage(fname=montage_path)
montage = mne.channels.make_standard_montage("brainproducts-RNP-BA-128")
raw.set_montage(montage)
raw.save(raw_folder / f"{sub}-{condition}-raw.fif", overwrite=True) # here the data is saved as raw


# 1. CLEANING THE DATA

# rename channels due to the trigger change
events = mne.events_from_annotations(raw)[0]
if 68 in events[:, 2]:
    raw.annotations.rename(mapping={'Stimulus/S  4': 'Stimulus/S  1',
                                    'Stimulus/S 68': 'Stimulus/S  5',
                                    'Stimulus/S 16': 'Stimulus/S  2',
                                    'Stimulus/S 64': 'Stimulus/S  4',
                                    'Stimulus/S 20': 'Stimulus/S  3'
                                                 })

# epoch for pieces of music
data = mne.io.RawArray(data=raw.get_data(), info=raw.info)

# 2. REMOVE CRAZY CHANNELS
raw.plot() # mark bad channels
raw_interp = raw.copy().interpolate_bads(reset_bads=True)
raw_interp.plot()

# 3. HIGH- AND LOW-PASS FILTER + POWER NOISE REMOVAL

# remove the power noise
raw_filter = raw_interp.copy()
raw_notch, iterations = dss.dss_line_iter(raw_filter.get_data().T, fline=cfg["filtering"]["notch"],
                             sfreq=data.info["sfreq"],
                             nfft=cfg["filtering"]["nfft"])

raw_filter._data = raw_notch.T

hi_filter = cfg["filtering"]["highpass"]
lo_filter = cfg["filtering"]["lowpass"]

raw_filtered = raw_filter.copy().filter(hi_filter, lo_filter)
raw_filtered.plot()

# plot the filtering
grad_psd(raw_interp, raw_filter, raw_filtered, fig_path)

del raw, raw_interp, raw_filter


# 4. ICA
raw_ica = raw_filtered.copy()
ica = mne.preprocessing.ICA(n_components=cfg["ica"]["n_components"], method=cfg["ica"]["method"], random_state=99)
ica.fit(raw_ica)

ica.plot_components(picks = np.arange(0,30,1))
ica.plot_sources(raw_ica)
ica.apply(raw_ica)
raw_ica.plot()

# 5. REFERENCE TO THE AVERAGE

raw_ica.set_eeg_reference(ref_channels='average')
picks_eeg = mne.pick_types(raw_filtered.info, meg=False, eeg=True, eog=False, stim=False,
                       exclude='bads')

# 6. EPOCHING THE DATA

# sanity check: number of different types of epochs
Counter(events[:, 2])

tmin = -0.2
tmax = 0.5
event_id = {"note": 1,
            "note/change": 2,
            "boundary/change": 3,
            "boundary": 4,
            "start": 5,
            }
if 68 in events[:, 2]:
    event_id = {"note": 4,
                "note/change": 16,
                "boundary/change": 20,
                "boundary": 64,
                "start": 68,
                }

epochs = mne.Epochs(raw_ica,
                    events=events,
                    event_id=event_id,
                    tmin=tmin,
                    tmax=tmax,
                    detrend=0,
                    baseline=(-0.2, 0), # should we set it here?
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

fig, ax = plt.subplots(2)
evoked.plot(exclude=[], axes=ax[0], show=False)
evoked_clean.plot(exclude=[], axes=ax[1], show=False)
ax[0].set_title(f"Before RANSAC (bad chs:{ransac.bad_chs_})")
ax[1].set_title("After RANSAC")
fig.tight_layout()
fig.savefig(fig_path / pathlib.Path("RANSAC_results.pdf"), dpi=800)
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
reject_log.plot('horizontal', show = False)


epochs_ar.apply_baseline((None, 0))

# plot and save the final results
fig, ax = plt.subplots(2)
epochs_ar.average().plot_image(
    titles=f"SNR:{snr(epochs_ar):.2f}", show=False, axes=ax[0])
epochs_ar.average().plot(show=False, axes=ax[1])
plt.tight_layout()
plt.savefig(
    fig_path / pathlib.Path("clean_evoked.pdf"), dpi=800)
plt.close()
epochs_ar.save(results_path / 'epochs' / f"{sub}-{condition}-epo.fif", overwrite=True)

epochs = epochs_ar.copy()
del epochs_clean, evoked, evoked_clean, epochs_ar

# 9. EVOKEDS

ev_note = epochs[list((np.where(epochs.events[:, 2] == 1))[0])].average()
ev_boundary = epochs[list((np.where(epochs.events[:, 2] == 4))[0])].average()
ev_note_change = epochs[list((np.where(epochs.events[:, 2] == 2))[0])].average()
ev_boundary_change = epochs[list((np.where(epochs.events[:, 2] == 3))[0])].average()

if 68 in events[:, 2]:
    ev_note = epochs[list((np.where(epochs.events[:, 2] == 4))[0])].average()
    ev_boundary = epochs[list((np.where(epochs.events[:, 2] == 64))[0])].average()
    ev_note_change = epochs[list((np.where(epochs.events[:, 2] == 16))[0])].average()
    ev_boundary_change = epochs[list((np.where(epochs.events[:, 2] == 20))[0])].average()


evokeds_avr = dict(no_change=[ev_note, ev_boundary], change=[ev_note_change, ev_boundary_change])
evokeds_nobound = dict(no_change=ev_note, change=ev_note_change)
evokeds_boundary = dict(no_change=ev_boundary, change=ev_boundary_change)

fig, ax = plt.subplots(3, figsize=(30*cm, 30*cm))

mne.viz.plot_compare_evokeds(evokeds_avr, picks=['Fz'], combine="mean", title=f'{sub} - Averaged evokeds across two conditions', colors=['g', 'b'], linestyles=['solid', 'dotted'], axes=ax[0])
mne.viz.plot_compare_evokeds(evokeds_nobound, picks=['Fz'], combine="mean", title='Within unit', colors=['g', 'b'], linestyles=['solid', 'dotted'], axes=ax[1])
mne.viz.plot_compare_evokeds(evokeds_boundary, picks=['Fz'], combine="mean", title='At boundary', colors=['g', 'b'], linestyles=['solid', 'dotted'], axes=ax[2])


plt.savefig(
    fig_path / pathlib.Path("evoked across conditions.pdf"), dpi=800)
plt.close()



ev_note.save(results_path / 'evokeds' /f"{sub}-{condition}_note-ave.fif", overwrite=True)
ev_note_change.save(results_path / 'evokeds' / f"{sub}-{condition}_note_change-ave.fif", overwrite=True)
ev_boundary.save(results_path / 'evokeds' / f"{sub}-{condition}_boundary-ave.fif", overwrite=True)
ev_boundary_change.save(results_path / 'evokeds' / f"{sub}-{condition}_boundary_change-ave.fif", overwrite=True)





##################################################
##################################################
##################################################
##################################################

import matplotlib.pyplot as plt

import mne
from mne import io
from mne.datasets import sample
from mne.stats import permutation_cluster_test

print(__doc__)

data_path = sample.data_path()
meg_path = data_path / "MEG" / "sample"
raw_fname = meg_path / "sample_audvis_filt-0-40_raw.fif"
event_fname = meg_path / "sample_audvis_filt-0-40_raw-eve.fif"
tmin = -0.2
tmax = 0.5

#   Setup for reading the raw data
raw = io.read_raw_fif(raw_fname)
events = mne.read_events(event_fname)

channel = "MEG 1332"  # include only this channel in analysis
include = [channel]

picks = mne.pick_types(raw.info, meg=False, eog=True, include=include, exclude="bads")
event_id = 1
reject = dict(grad=4000e-13, eog=150e-6)
epochs1 = mne.Epochs(
    raw, events, event_id, tmin, tmax, picks=picks, baseline=(None, 0), reject=reject
)
condition1 = epochs1.get_data()  # as 3D matrix

event_id = 2
epochs2 = mne.Epochs(
    raw, events, event_id, tmin, tmax, picks=picks, baseline=(None, 0), reject=reject
)
condition2 = epochs2.get_data()  # as 3D matrix

condition1 = condition1[:, 0, :]  # take only one channel to get a 2D array
condition2 = condition2[:, 0, :]  # take only one channel to get a 2D array

threshold = 6.0
T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
    [condition1, condition2],
    n_permutations=1000,
    threshold=threshold,
    tail=1,
    n_jobs=None,
    out_type="mask",
)
times = epochs1.times
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(8, 4))
ax.set_title("Channel : " + channel)
ax.plot(
    times,
    condition1.mean(axis=0) - condition2.mean(axis=0),
    label="ERF Contrast (Event 1 - Event 2)",
)
ax.set_ylabel("MEG (T / m)")
ax.legend()

for i_c, c in enumerate(clusters):
    c = c[0]
    if cluster_p_values[i_c] <= 0.05:
        h = ax2.axvspan(times[c.start], times[c.stop - 1], color="r", alpha=0.3)
    else:
        ax2.axvspan(times[c.start], times[c.stop - 1], color=(0.3, 0.3, 0.3), alpha=0.3)

hf = plt.plot(times, T_obs, "g")
ax2.legend((h,), ("cluster p-value < 0.05",))
ax2.set_xlabel("time (ms)")
ax2.set_ylabel("f-values")

s1_epochs = epochs['R_1']
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