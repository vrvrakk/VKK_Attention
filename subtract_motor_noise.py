# todo: import motor-only ERPs
# todo: pad with 0s pre- and post-stimulus
# todo: harsh filer-> 1-25Hz
# todo: subtract motor-only ERP from actual epochs
# import libraries:
import numpy as np
from pathlib import Path
import mne
import json
import os
from collections import Counter
from helper import grad_psd, snr
from matplotlib import pyplot as plt, patches
from autoreject import Ransac

cm = 1 / 2.54

# create relevant paths:
path = Path.cwd()

data_path = path / 'data' / 'eeg'
results_path = data_path / 'preprocessed' / 'results'/ 'motor'
epochs_path = data_path / 'preprocessed' / 'results'/ 'motor' / 'epochs'
fig_path = results_path / 'figures'
evokeds_path = results_path / 'evokeds'
json_path = path / 'data' / 'misc'

# if path does not exist:
for folder in results_path, epochs_path, fig_path, evokeds_path:
    if not os.path.isdir(folder):
        os.makedirs((folder))

# load configuration files:
with open(json_path / "preproc_config.json") as file:
    cfg = json.load(file)
response_events = {'Stimulus/S129': 3,
         'Stimulus/S131': 5,
         'Stimulus/S130': 4,
         'Stimulus/S134': 8,
         'Stimulus/S132': 6,
         'Stimulus/S133': 7,
         'Stimulus/S136': 10,
         'Stimulus/S137': 11}
# load motor-only file:
# this data consists of grossly pre-processed motor-only eegs separately
# -> interpolated bad channels, manually selected noisy segments and removed, bp 1-40Hz, concatenated, ran ICA
motor_raw = mne.io.read_raw_fif(data_path / 'preprocessed'/'results'/'motor'/'motor-only concatenated raw.fif')
# get relevant events:
events = mne.events_from_annotations(motor_raw)[0]  # get events from annotations attribute of raw variable
events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
filtered_events = [event for event in events if event[2] in response_events.values()]
events = np.array(filtered_events)
# get epochs:
tmin = -0.2
tmax = 0.7
epoch_parameters = [tmin, tmax, response_events]
tmin, tmax, event_ids = epoch_parameters

motor_epochs = mne.Epochs(motor_raw,
                           events,
                           event_id=event_ids,
                           tmin=tmin,
                           tmax=tmax,
                           detrend=0,
                           baseline=(-0.2, 0),  # should we set it here?
                           preload=True)

motor_epochs.plot_psd()
bads = []
from autoreject import AutoReject, Ransac


def ransac(epochs, bads):
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

    evoked = epochs.average()
    evoked_clean = epochs_clean.average()

    evoked.info['bads'] = ransac.bad_chs_
    evoked_clean.info['bads'] = ransac.bad_chs_

    fig, ax = plt.subplots(2, constrained_layout=True)
    evoked.plot(exclude=[], axes=ax[0], show=False)
    evoked_clean.plot(exclude=[], axes=ax[1], show=False)
    ax[0].set_title(f"Before RANSAC (bad chs:{ransac.bad_chs_})")
    ax[1].set_title("After RANSAC")
    fig.savefig(results_path / f'motor-only epochs Ransac.pdf', dpi=800)
    plt.close()
    return epochs_clean, ransac


epochs_clean, ransac = ransac(motor_epochs, bads)
epochs_reref = epochs_clean.copy()
epochs_reref.set_eeg_reference(ref_channels='average')

def ar(epochs_reref):
    ar = AutoReject(n_interpolate=cfg["autoreject"]["n_interpolate"], n_jobs=cfg["autoreject"]["n_jobs"])
    ar = ar.fit(epochs_reref)
    epochs_ar, reject_log = ar.transform(epochs_reref, return_log=True)

    # target_epochs1[reject_log1.bad_epochs].plot(scalings=dict(eeg=100e-6))
    # reject_log1.plot('horizontal', show=False)

    # plot and save the final results
    fig, ax = plt.subplots(2, constrained_layout=True)
    epochs_ar.average().plot_image(titles=f"SNR:{snr(epochs_ar):.2f}", show=False, axes=ax[0])
    epochs_ar.average().plot(show=False, axes=ax[1])
    plt.savefig(fig_path / 'motor-only epochs AutoReject', dpi=800)
    plt.close()
    epochs_ar.save(results_path / 'epochs' / 'motor-only epochs AutoReject-epo.fif', overwrite=True)
    return epochs_ar

epochs_ar = ar(epochs_reref)

# make a copy and save:
epochs = epochs_ar.copy()
epochs.save(epochs_path/'motor-only Ransac-epo.fif')
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

# ERPs:
grand_average = mne.grand_average(evokeds)
grand_average.save(evokeds_path / 'ERP motor-only-ave.fif')
# add padding pre- and post- stimulus:
# 200ms padding samples in len -> 500 samples * 200ms / 1000ms = 100 samples
start_time = grand_average.times[0]
end_time = grand_average.times[-1]
padding_dur = 0.2
sfreq = grand_average.info['sfreq']
padding = np.pad(grand_average.data, ((0, 0), (100, 100)), 'constant')
pre_times = np.arange(-0.4, start_time, 1/sfreq)
post_times = np.arange(end_time + 1/sfreq, end_time + padding_dur, 1/sfreq)
padded_times = np.concatenate((pre_times, grand_average.times, post_times))
padded_evoked = mne.EvokedArray(padding, grand_average.info, tmin=padded_times[0])
padded_evoked.filter(l_freq=None, h_freq=25)
fig, ax = plt.subplots(figsize=(30 * cm, 15 * cm))  # Adjust the figure size as needed
# Plot the grand average evoked response
mne.viz.plot_compare_evokeds(
    {'Grand Average': padded_evoked},
    picks=['Cz', 'C1', 'C2', 'C3', 'C4'],
    combine='mean',
    title=f'Motor-only Padded ERPs, 1-25Hz',
    colors={'Grand Average': 'r'},
    linestyles={'Grand Average': 'solid'},
    axes=ax,
    show=True  # Set to True to display the plot immediately
)

plt.savefig(fig_path/'ERP padded motor-only, 1-25Hz.pdf', dpi=800)

padded_evoked.save(evokeds_path / 'Padded ERP 1-25Hz, motor-only-ave.fif')
concatenated_data = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/concatenated_data')
exp_raw = mne.io.read_raw_fif(concatenated_data/'EEG elevation with motor-only ERP subtracted-raw.fif', preload=True)


# Load the grand_average ERP data
padded_evoked = mne.read_evokeds(evokeds_path / 'Padded ERP 1-25Hz, motor-only-ave.fif')[0]

# get events (see previous lines for filtered_events)

sfreq = exp_raw.info['sfreq']
erp_duration = padded_evoked.times[-1] - padded_evoked.times[0]
n_samples_erp = len(padded_evoked.times)

# Subtract the ERP at each event time
for event in events:
    event_sample = event[0]  # sample number of the event
    start_sample = event_sample - int(padded_evoked.times[0] * sfreq)
    end_sample = start_sample + n_samples_erp

    # Check if the event is within the bounds of the raw data
    if start_sample >= 0 and end_sample <= len(exp_raw.times):
        # Subtract the ERP data from the raw data
        exp_raw._data[:, start_sample:end_sample] -= padded_evoked.data

exp_raw.save(concatenated_data/'EEG elevation with motor-only ERP subtracted-raw.fif')