import json
import numpy
from scipy import signal
from matplotlib import pyplot as plt
from mne import events_from_annotations, compute_raw_covariance
import mne
from mne.io import read_raw_brainvision
from mne.epochs import Epochs
from autoreject import Ransac, AutoReject
from mne.preprocessing import ICA, read_ica, corrmap
from meegkit.dss import dss_line
from pathlib import Path
import os
import re

default_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data')
data_dir = default_dir
motor_dir = data_dir / 'eeg' / 'raw' / 'motor'

electrode_names = json.load(open(data_dir / 'misc' / "electrode_names.json"))

motor_dict = {'R_1': 129, 'R_2': 130, 'R_3': 131, 'R_4': 132, 'R_5': 133, 'R_6': 134, 'R_8': 136, 'R_9': 137}

epoch_parameters = [-0.5,  # tmin
                    1.5,  # tmax
                    motor_dict]

pattern = r'\d{6}_\w{2}'
regex = re.compile(pattern)
# get file:
motor_eeg = []
motor_header = []
header_paths = []
# get eeg and header files of motor-only condition:
for file_name in os.listdir(motor_dir):
    folder_path = Path(motor_dir/file_name)
    if folder_path.is_file() and regex.match(file_name):
        if 'motor' in file_name:
            if file_name.endswith('ot_motor.eeg'):
                motor_eeg.append(file_name)  # get eeg files
            elif file_name.endswith('ot_motor.vhdr'):
                motor_header.append(file_name)  # get header files
                header_path = Path(motor_dir / file_name)
                header_paths.append(header_path)

header_paths = [str(path) for path in header_paths]
output_dir = data_dir / 'eeg' / 'preprocessed'  # create output directory
motor_raw = []
# change directory:
os.chdir(motor_dir)

# for files in header_files_all[0]:
for files in motor_header:
    motor_raw.append(read_raw_brainvision(files))  # used to read info from eeg + vmrk files
raw = mne.concatenate_raws(motor_raw, preload=None, events_list=None, on_mismatch='raise', verbose=None)
# this should concatenate all raw eeg files within one subsubfolder
raw = raw.load_data()  # preload turns True under raw attributes
# assign channel names to the data
raw.rename_channels(electrode_names)
# set_montage to get spatial distribution of channels
raw.set_montage("standard_1020")  # ------ use brainvision montage instead?

# inspect raw data
raw.plot()
# todo: hz -> weird beginning??
# todo: ot -> 1 weird electrode -> FT9
# todo: ll -> OK
# todo: ls -> GARBAGE
# todo: ja -> OK
# todo: zh -> F2 electrode bad
# todo: ez -> FC2, C3
raw.compute_psd().plot(average=True)
fig = raw.plot_psd(xscale='linear', fmin=0, fmax=250)
fig.suptitle('all motor-only EEGs')  # can be added after plotting

print('STEP 1: Remove power line noise and apply minimum-phase highpass filter')
X = raw.get_data().T  # transpose -> create a 3D matrix-> Channels, Time, Voltage values
X, _ = dss_line(X, fline=50, sfreq=raw.info["sfreq"], nremove=5)



# plot changes made by the filter:
# plot before / after zapline denoising
# power line noise is not fully removed with 5 components, remove 10
f, ax = plt.subplots(1, 2, sharey=True)
f, Pxx = signal.welch(raw.get_data().T, 500, nperseg=500, axis=0, return_onesided=True)  # to get psd
ax[0].semilogy(f, Pxx)
f, Pxx = signal.welch(X, 500, nperseg=500, axis=0, return_onesided=True)
ax[1].semilogy(f, Pxx) # plot on a log scale
ax[0].set_xlabel("frequency [Hz]")
ax[1].set_xlabel("frequency [Hz]")
ax[0].set_ylabel("PSD [V**2/Hz]")
ax[0].set_title("before")
ax[1].set_title("after")
plt.show()
#
# # put the data back into raw
raw._data = X.T
# del X
#
# # remove line noise (eg. stray electromagnetic signals) -> high pass
raw = raw.filter(l_freq=.5, h_freq=None, phase="minimum")
raw.plot()
# # raw.plot_psd()
# Assuming `raw` is your MNE Raw object
bad_channels = ['FT9', 'PO7', 'TP8', 'TP7', 'PO10', 'TP9', 'T8']  # Replace with the actual names of bad channels in your data
raw.info['bads'] = bad_channels  # Add bad channels to the `bads` list

# Optionally, visualize the data to ensure the channels are marked correctly


# To drop the marked bad channels from further analysis:
raw_clean = raw.copy().drop_channels(bad_channels)
raw_clean.drop_channels('TP')
# Confirm removal:
raw_clean.plot()
raw_clean.plot_psd()
# # everything below 0.5Hz (electromagnetic drift)
# # minimum phase keeps temporal distortion to a minimum
#
# print('STEP 2: Epoch and downsample the data')
# # get events
events = events_from_annotations(raw)[0]  # get events from annotations attribute of raw variable
events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
filtered_events = [event for event in events if event[2] in motor_dict.values()]
events = numpy.array(filtered_events)


# reject_criteria = dict(eeg=90e-6)  # 100 µV  # 200 µV
# flat_criteria = dict(eeg=1e-6)
# remove all meaningless event codes, including post trial events
tmin, tmax, event_ids = epoch_parameters  # get epoch parameters
epochs = Epochs(
    raw,
    events,
    event_id=event_ids,
    tmin=tmin,
    tmax=tmax,
    # reject_tmax=0,
    # reject=reject_criteria,
    # flat=flat_criteria,
    baseline=None,
    preload=True,
)

# # extra: use raw data to compute the noise covariance  # for later analysis?
# tmax_noise = (events[0, 0] - 1) / raw.info["sfreq"]  # cut raw data before first stimulus
# raw.crop(0, tmax_noise)
# cov = compute_raw_covariance(raw)  # compute covariance matrix
# cov.save(outdir / f"{subfolder.name}_noise-cov.fif", overwrite=True)  # save to file
# del raw
#
# fs = 100  # resample data to effectively drop frequency components above fs / 3
# decim = int(epochs.info["sfreq"] / fs)
# epochs.filter(None, fs / 3, n_jobs=4)
# epochs.decimate(decim)


print('STEP 4: Blink rejection with ICA')  #todo Viola et al., 2009
# reference = read_ica(data_dir / 'misc' / 'reference-ica.fif')
# component = reference.labels_["blinks"]
ica = ICA(method="fastica")
ica.fit(epochs)
# corrmap([ica, ica], template=(0, component[0]), label="blinks", plot=False, threshold=0.75)
# ica.plot_components()  # first 10 independent components
ica.plot_sources(epochs)
# ica.plot_properties(epochs, picks=[8]) # 13, 14, 16, 18, 19
# ica.labels_["blinks"] = [1]
#
# ica.exclude = []
# ica.apply(epochs)
# ica.save(output_dir / f"{header_file}-ica.fif", overwrite=True)
# del ica
