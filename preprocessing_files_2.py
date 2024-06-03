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
from preprocessing_files_1 import cfg, events1, events2, s1_events, s2_events, response_events, cm

subs = ['sub06', 'sub07', 'sub08']
results_paths = []
condition = ['s1', 's2']
axis = ['azimuth', 'ele']

name = ''

for sub in subs:
    results_path = Path(f'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/{sub}')
    results_paths.append(results_path)

n = 0
raws = []
def get_files(condition, axis):
    for path in results_paths:
        for file in os.listdir(path):
            if '-raw.fif' in file:
                if condition in file:
                    if axis in file:
                        full_path = path/file
                        raw = mne.io.read_raw_fif(full_path, preload=True)
                        raws.append(raw)
    return raws

raws = get_files(condition=condition[n], axis=axis[n])
target_filtered = mne.concatenate_raws(raws)

# load 1-25Hz motor-only ERP:
padded_evoked = mne.read_evokeds( 'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/motor/evokeds/Padded ERP 1-25Hz, motor-only-ave.fif')

# 4. ICA
target_ica = target_filtered.copy()
ica = mne.preprocessing.ICA(n_components=cfg["ica"]["n_components"], method=cfg["ica"]["method"], random_state=99)
ica.fit(target_ica)
ica.plot_components()
# ica.save('motor-only ICA', overwrite=True)
ica.plot_sources(target_ica)
ica.apply(target_ica)
target_ica.save(results_path / 's1-ele ICA concatenated, motor-only subtraction-raw.fif')


sfreq = target_filtered.info['sfreq']
erp_duration = padded_evoked.times[-1] - padded_evoked.times[0]
n_samples_erp = len(padded_evoked.times)

events = mne.events_from_annotations(target_filtered)[0]  # get events from annotations attribute of raw variable
events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
filtered_events = [event for event in events if event[2] in response_events.values()]
events = np.array(filtered_events)

# Subtract the ERP at each event time
for event in events:
    event_sample = event[0]  # sample number of the event
    start_sample = event_sample - int(padded_evoked.times[0] * sfreq)
    end_sample = start_sample + n_samples_erp

    # Check if the event is within the bounds of the raw data
    if start_sample >= 0 and end_sample <= len(target_filtered.times):
        # Subtract the ERP data from the raw data
        target_filtered._data[:, start_sample:end_sample] -= padded_evoked.data


concatenated_data = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/concatenated_data')
target_filtered.save(concatenated_data/f'EEG {name}, conditions: {condition}, {axis}, motor-only subtraction.fif')


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


tmin, tmax = -0.2, 0.7
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
    fig.savefig(concatenated_data / 'figures' / f"{target} for RANSAC,{axis} {condition}.pdf", dpi=800)
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
    plt.savefig(concatenated_data/ 'figures' / f"{target} for clean_evoked {axis} {condition}.pdf", dpi=800)
    plt.close()
    epochs_ar.save(concatenated_data / 'epochs' / f"{target} from {name}-conditions: {axis},{condition}-epo.fif",
                   overwrite=True)
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
    plt.savefig(
        concatenated_data / 'evokeds' / f"{target} ERP from {name}, conditions: {condition}, {axis}, motor-only subtracted.pdf",
        dpi=800)
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
    plt.savefig(
        concatenated_data/ 'evokeds' / f"S1 vs S2 GRAND AVERAGE from {name}, conditions: {condition}, {axis}, motor-only subtracted.pdf",
        dpi=800)
    return evokeds_total


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
fig.savefig(concatenated_data / 'figures' / f"Re-referenced cleaned epochs, {axis} {condition}.pdf", dpi=800)

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
grand_average1.save(concatenated_data / '1-25Hz s1 target, elevation, motor-only subtraction stim1-ave.fif',
                    overwrite=True)
grand_average2.save(concatenated_data / '1-25Hz s1 target, elevation, motor-only subtraction stim2-ave.fif',
                    overwrite=True)

evokeds_total = s1_vs_s2(grand_average1, grand_average2, name=name)

