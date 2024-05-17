"""Own version of preprocessing (based on Max script)"""

import glob
import json
from meegkit.dss import dss_line_iter
from mne.preprocessing import ICA
from autoreject import AutoReject, Ransac
from matplotlib import pyplot as plt, patches
import mne
import numpy as np
import pathlib
import os
# %matplotlib qt
_scaling = 10**6


def filtering(data, notch=None, highpass=None, lowpass=None):
    """
    Apply FIR filter to the raw dataset. Make a 2 by 2 plot with time
    series data and power spectral density before and after.
    """
    fig, ax = plt.subplots(2, sharex=True, sharey=True)
    fig.suptitle("Power Spectral Density")
    ax[0].set_title("before filtering")
    ax[1].set_title("after filtering")
    ax[1].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    ax[0].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")
    data.plot_psd(ax=ax[0], show=False, exclude=["FCz"])
    if notch is not None:  # ZapLine Notch filter
        X = data.get_data().T
        # remove power line noise with the zapline algorithm
        X, _ = dss_line_iter(X, fline=cfg["filtering"]["notch"],
                             sfreq=data.info["sfreq"],
                             nfft=cfg["filtering"]["nfft"])
        data._data = X.T  # put the data back into variable
        del X
    if lowpass is not None:
        data.filter(h_freq=lowpass, l_freq=None)
    if highpass is not None:
        data.filter(h_freq=None, l_freq=highpass)
    data.plot_psd(ax=ax[1], show=False, exclude=["FCz"])
    if lowpass is not None and highpass == None:
        fig.savefig(
            fig_folder / pathlib.Path("lowpassfilter.pdf"), dpi=800)
    if highpass is not None and lowpass == None:
        fig.savefig(
            fig_folder / pathlib.Path("highpassfilter.pdf"), dpi=800)
    if highpass and lowpass is not None:
        fig.savefig(
            fig_folder / pathlib.Path("bandpassfilter.pdf"), dpi=800)
    if notch is not None:
        fig.savefig(
            fig_folder / pathlib.Path("ZapLine_filter.pdf"), dpi=800)
    # plt.close()
    return data


def noise_rms(epochs):
    global scaling
    epochs_tmp = epochs.copy()
    n_epochs = epochs.get_data().shape[0]
    for i in range(n_epochs):
        if not i % 2:
            epochs_tmp.get_data()[i, :, :] = -epochs_tmp.get_data()[i, :, :]
    evoked = epochs_tmp.average().get_data()
    rms = np.sqrt(np.mean(evoked**2)) * _scaling
    del epochs_tmp
    return rms

def snr(epochs, signal_interval=(0.15, 0.2)):
    """
    Compute signal-to-noise ratio. Take root mean square of noise
    plus signal (interval where evoked activity is expected)
    and return the quotient.
    """
    signal = epochs.copy()
    signal.crop(signal_interval[0], signal_interval[1])
    n_rms = noise_rms(epochs)
    s_rms = np.sqrt(np.mean(signal.average().get_data()**2)) * _scaling
    snr = s_rms / n_rms  # signal rms divided by noise rms
    return snr


def autoreject_epochs(epochs,
                      n_interpolate=[1, 4, 8, 16],
                      consensus=None,
                      cv=10,
                      thresh_method="bayesian optimization",
                      n_jobs=-1,
                      random_state=None):
    """
    Automatically reject epochs via AutoReject algorithm:
    Computation of sensor-wise peak-to-peak-amplitude thresholds
    via cross-validation.
    """
    ar = AutoReject(n_interpolate=n_interpolate, n_jobs=n_jobs)
    ar.fit(epochs)
    epochs_ar, reject_log = ar.transform(epochs, return_log=True)
    fig, ax = plt.subplots(2)
    # plotipyt histogram of rejection thresholds
    ax[0].set_title("Rejection Thresholds")
    ax[0].hist(1e6 * np.array(list(ar.threshes_.values())), 30,
               color="g", alpha=0.4)
    ax[0].set(xlabel="Threshold (μV)", ylabel="Number of sensors")
    # plot cross validation error:
    loss = ar.loss_["eeg"].mean(axis=-1)  # losses are stored by channel type.
    im = ax[1].matshow(loss.T * 1e6, cmap=plt.get_cmap("viridis"))
    ax[1].set_xticks(range(len(ar.consensus)))
    ax[1].set_xticklabels(["%.1f" % c for c in ar.consensus])
    ax[1].set_yticks(range(len(ar.n_interpolate)))
    ax[1].set_yticklabels(ar.n_interpolate)
    # Draw rectangle at location of best parameters
    idx, jdx = np.unravel_index(loss.argmin(), loss.shape)
    rect = patches.Rectangle((idx - 0.5, jdx - 0.5), 1, 1, linewidth=2,
                             edgecolor="r", facecolor="none")
    ax[1].add_patch(rect)
    ax[1].xaxis.set_ticks_position("bottom")
    ax[1].set(xlabel=r"Consensus percentage $\kappa$",
              ylabel=r"Max sensors interpolated $\rho$",
              title="Mean cross validation error (x 1e6)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(fig_folder / pathlib.Path("autoreject_best_fit.pdf"), dpi=800)
    plt.close()
    evoked_bad = epochs[reject_log.bad_epochs].average()
    snr_ar = snr(epochs_ar)
    plt.plot(evoked_bad.times, evoked_bad.data.T * 1e06, 'r', zorder=-1)
    epochs_ar.average().plot(axes=plt.gca(), show=False,
                             titles=f"SNR: {snr_ar:.2f}")
    plt.savefig(
        fig_folder / pathlib.Path("autoreject_results.pdf"), dpi=800)
    plt.close()
    epochs_ar.plot_drop_log(show=False)
    plt.savefig(
        fig_folder / pathlib.Path("epochs_drop_log.pdf"), dpi=800)
    plt.close()
    return epochs_ar



### STEP 0: Define directories ###
DIR = pathlib.Path(os.getcwd())

with open(DIR / "data" / "misc" / "preproc_config.json") as file:
    cfg = json.load(file)
with open(DIR / "data" / "misc" / "electrode_names.json") as file:
    mapping = json.load(file)
sub = input('Please give sub number (as subn): ')
data_DIR = DIR / "data" / "eeg" / 'raw'
results_path= DIR/"data"/ "eeg" / "preprocessed" / 'results'

# Define directories for subject
folder_path = data_DIR / sub
epochs_folder = results_path / sub / "epochs"
raw_folder = results_path / sub / "raw_data"
fig_folder = results_path / sub / "figures"
evokeds_folder = results_path / sub / "evokeds"

for folder in epochs_folder, raw_folder, fig_folder, evokeds_folder:
    if not os.path.isdir(folder):
        os.makedirs(folder)



### STEP 1: Concatenate block files to one raw file in raw_folder ###
header_files = folder_path.glob("*.vhdr")
raw_files = []
for header_file in header_files:
    print(header_file)
    raw_files.append(mne.io.read_raw_brainvision(header_file, preload=True))
raw = mne.concatenate_raws(raw_files)
raw.rename_channels(mapping) #Map the electrodes
raw.add_reference_channels('FCz')# Add reference electrode.
montage_path = DIR / "analysis" / "settings" / cfg["montage"]["name"] # Get montage
montage = mne.channels.read_custom_montage(fname=montage_path)
#montage.plot()
raw.set_montage(montage)
raw.save(raw_folder / pathlib.Path(subj + "_raw.fif"), overwrite=True)
raw.plot() # To get a first impression of the data


### STEP 2: bandpass filtering of the data at 1=40 Hz ###
raw = filtering(raw,
                highpass=cfg["filtering"]["highpass"],
                lowpass=cfg["filtering"]["lowpass"],
                notch=cfg["filtering"]["notch"])


### STEP 3: Epoch the raw data and apply baseline ###
events = mne.events_from_annotations(raw)[0]  # Get events
mne.viz.plot_events(events) # Plot events
epochs = mne.Epochs(raw, events, tmin=cfg["epochs"]["tmin"], tmax=cfg["epochs"]["tmax"],
                    event_id=cfg["epochs"][f"event_id"], preload=True,
                    baseline=cfg["epochs"]["baseline"], detrend=cfg["epochs"]["detrend"])  # Epoch data and apply baseline
epochs.plot() #Plot epochs


del raw, raw_files  # del raw data to save working memory

### STEP 4: Run RANSAC/ exclude bad channels ###
epochs_clean = epochs.copy()
ransac = Ransac(n_jobs=cfg["reref"]["ransac"]["n_jobs"], n_resample=cfg["reref"]["ransac"]["n_resample"],
                min_channels=cfg["reref"]["ransac"]["min_channels"], min_corr=cfg["reref"]["ransac"]["min_corr"],
                unbroken_time=cfg["reref"]["ransac"]["unbroken_time"])

ransac.fit(epochs_clean)

epochs_clean.average().plot(exclude=[])

bads=["C1", "P5", "C6", "FC3", "CP3"] #Add channel names to exclude here
if len(bads) != 0:
    for bad in bads:
        if bad not in ransac.bad_chs_:
            ransac.bad_chs_.extend(bads)


epochs_clean = ransac.transform(epochs_clean)


# Plot RANSAC results
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
fig.savefig(
    fig_folder / pathlib.Path("RANSAC_results.pdf"), dpi=800)
plt.close()



### STEP 5: Rereference the epochs ###
epochs = epochs_clean.copy()
epochs_clean.set_eeg_reference(ref_channels="average", projection=True)
average_reference = epochs_clean.info["projs"]
epochs_clean.add_proj(average_reference)
epochs_clean.apply_proj()
snr_pre = snr(epochs)
snr_post = snr(epochs_clean)
epochs_reref = epochs_clean.copy()

# Plot the results of rereferencing
fig, ax = plt.subplots(2)
epochs.average().plot(axes=ax[0], show=False)
epochs_reref.average().plot(axes=ax[1], show=False)
ax[0].set_title(f"FCz, SNR={snr_pre:.2f}")
ax[1].set_title(f"{type}, SNR={snr_post:.2f}")
fig.tight_layout()
fig.savefig(
    fig_folder / pathlib.Path("average_reference.pdf"), dpi=800)
plt.close()


### STEP 6: Apply ICA ###
reference_path = DIR /"analysis"/ "settings" / cfg["ica"]["reference"]

epochs_ica = epochs_reref.copy()
snr_pre_ica = snr(epochs_ica)
ica = ICA(n_components=cfg["ica"]["n_components"], method=cfg["ica"]["method"])
ica.fit(epochs_ica)
ref = mne.preprocessing.read_ica(fname=reference_path)
ica.plot_components()


ica.exclude = [1] #List of integer components to exclude

# Plot excluded components
ica.plot_components(ica.exclude, show=False)
plt.savefig(fig_folder / pathlib.Path("ICA_components.pdf"), dpi=800)
plt.close()

# Apply ICA
ica.apply(epochs_ica)

# Plot ICA sources
ica.plot_sources(inst=epochs, show=False, start=0,
                 stop=10, show_scrollbars=False)
plt.savefig(fig_folder / pathlib.Path(f"ICA_sources.pdf"), dpi=800)
plt.close()
# Plot SNR change during ICA
snr_post_ica = snr(epochs_ica)
ica.plot_overlay(epochs.average(), exclude=ica.exclude,
                 show=False, title=f"SNR: {snr_pre_ica:.2f} (before), {snr_post_ica:.2f} (after)")
plt.savefig(fig_folder / pathlib.Path("ICA_results.pdf"), dpi=800)
plt.close()

### STEP 7: Apply AutoReject ###
epochs_clean = autoreject_epochs(
            epochs_ica, n_interpolate=cfg["autoreject"]["n_interpolate"],
            n_jobs=cfg["autoreject"]["n_jobs"],
            cv=cfg["autoreject"]["cv"],
            thresh_method=cfg["autoreject"]["thresh_method"])
epochs_clean.apply_baseline((None, 0))
epochs_clean.save(
    epochs_folder / pathlib.Path(id + "-epo.fif"), overwrite=True)

# Plot the AutoReject results
fig, ax = plt.subplots(2)
epochs_clean.average().plot_image(
    titles=f"SNR:{snr(epochs_clean):.2f}", show=False, axes=ax[0])

epochs_clean.average().plot(show=False, axes=ax[1])
plt.tight_layout()
plt.savefig(
    fig_folder / pathlib.Path("clean_evoked.pdf"), dpi=800)
plt.close()


### STEP 8: Average epochs and write evokeds###
evokeds = [epochs_clean[condition].average()
                   for condition in cfg["epochs"][f"event_id"].keys()]
mne.write_evokeds(evokeds_folder / pathlib.Path(subj + "-ave.fif"), evokeds, overwrite=True)
del epochs, epochs_reref, epochs_ica, epochs_clean, evokeds  # Delete data to save working memory