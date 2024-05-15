import json
import numpy
from scipy import signal
from matplotlib import pyplot as plt, patches
from mne import events_from_annotations, compute_raw_covariance
import mne
from mne.io import read_raw_brainvision
from autoreject import AutoReject, get_rejection_threshold
from mne.epochs import Epochs
from autoreject import Ransac, AutoReject
from mne.preprocessing import ICA, read_ica, corrmap
from meegkit.dss import dss_line_iter
import pathlib
from pathlib import Path
import os

# todo: get pipeline
default_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data')
data_dir = default_dir
motor_dir = data_dir / 'eeg' / 'raw' / 'motor' / 'motor_only'
fig_path = data_dir / 'eeg' / 'preprocessed' / 'motor' / 'figures'

electrode_names = json.load(open(data_dir / 'misc' / "electrode_names.json"))
config_params = json.load(open(data_dir / 'misc' / 'preproc_config.json'))
config_params["filtering"]["notch"] = 50
config_params["filtering"]["baseline"] = (-0.2, 0)

motor_dict = {'R_1': 129, 'R_2': 130, 'R_3': 131, 'R_4': 132, 'R_5': 133, 'R_6': 134, 'R_8': 136, 'R_9': 137}

epoch_parameters = [-0.5,  # tmin
                    1.5,  # tmax
                    motor_dict]

_scaling = 10 ** 6
conditions = {0: 'baseline', 1: 'audio', 2: 'forehead', 3: 'eyes', 4: 'blinks', 5: 'mouth', 6: 'jaw', 7: 'head',
              8: 'buttons_only', 9: 'buttons'}



def noise_rms(epochs):
    global scaling
    epochs_tmp = epochs.copy()
    n_epochs = epochs.get_data().shape[0]
    for i in range(n_epochs):
        if not i % 2:
            epochs_tmp.get_data()[i, :, :] = -epochs_tmp.get_data()[i, :, :]
    evoked = epochs_tmp.average().get_data()
    rms = numpy.sqrt(numpy.mean(evoked ** 2)) * _scaling
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
    s_rms = numpy.sqrt(numpy.mean(signal.average().get_data() ** 2)) * _scaling
    snr = s_rms / n_rms  # signal rms divided by noise rms
    return snr


# STEP 1: get files
# get eeg and header files of each condition:
def get_files(motor_dir, condition):
    eeg = []
    header = []
    header_paths = []
    for file_name in os.listdir(motor_dir):
        folder_path = Path(motor_dir / file_name)
        if folder_path.is_file():
            if condition in file_name:
                print(file_name)
                if file_name.endswith('.eeg'):
                    eeg.append(file_name)  # get eeg files
                elif file_name.endswith('.vhdr'):
                    header.append(file_name)  # get header files
                    header_path = Path(motor_dir / file_name)
                    header_paths.append(header_path)
    return eeg, header, header_paths


baseline_eeg, baseline_header, baseline_header_paths = get_files(motor_dir, condition=conditions[0])

output_dir = data_dir / 'eeg' / 'preprocessed'  # create output directory

# change directory:
os.chdir(motor_dir)


def get_subject():
    subject = input('Subject?: ')
    return subject


subject = get_subject() # works lol


def get_motor_raws(subject, baseline_header):
    motor_raw = []
    for files in baseline_header:
        motor_raw.append(read_raw_brainvision(files))  # used to read info from eeg + vmrk files
    raw = mne.concatenate_raws(motor_raw, preload=None, events_list=None, on_mismatch='raise', verbose=None)
    # this should concatenate all raw eeg files within one subsubfolder
    raw = raw.load_data()  # preload turns True under raw attributes
    # assign channel names to the data
    raw.rename_channels(electrode_names)
    # set_montage to get spatial distribution of channels
    raw.set_montage("standard_1020")  # ------ use brainvision montage instead?
    # raw.plot()
    save_path = motor_dir / f'{subject}_{conditions[0]}.fif'
    raw.save(save_path, overwrite=True)
    print(f"Saved raw file for {conditions[0]} to {save_path}")
    return raw


raw = get_motor_raws(subject, baseline_header)


# apply filter
def apply_filter(raw, subject):
    filt_raw = filtering(raw, subject, highpass=config_params["filtering"]["highpass"],
                    lowpass=config_params["filtering"]["lowpass"],
                    notch=config_params["filtering"]["notch"])
    return filt_raw


def filtering(data, subject, notch=None, highpass=None, lowpass=None):
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
        X, _ = dss_line_iter(X, fline=notch,
                             sfreq=data.info["sfreq"],
                             nfft=config_params["filtering"]["nfft"])
        data._data = X.T  # put the data back into variable
        del X
    if lowpass is not None:
        data.filter(h_freq=lowpass, l_freq=None)
    if highpass is not None:
        data.filter(h_freq=None, l_freq=highpass)

    data.plot_psd(ax=ax[1], show=False, exclude=["FCz"])

    # Save the figures with appropriate names
    filter_types = []
    if notch is not None:
        filter_types.append('notch')
    if lowpass is not None:
        filter_types.append('lowpass')
    if highpass is not None:
        filter_types.append('highpass')

    filter_name = '_'.join(filter_types)
    fig_filename = f'{subject}_{conditions[0]}_{filter_name}_filter.pdf'
    fig.savefig(fig_path / fig_filename, dpi=800)
    return data


filt_raw = apply_filter(raw, subject)  # works


# get baselined epochs
def get_epochs(filt_raw):
    baseline_len = 1.5
    sfreq = filt_raw.info['sfreq']
    event_interval = int(sfreq * baseline_len)
    n_events = int(filt_raw.n_times / event_interval)
    events = numpy.array([
        [i * event_interval, 0, 1] for i in range(n_events)
    ])
    # Create epochs
    epochs = mne.Epochs(filt_raw, events, event_id=1, tmin=-0.5, tmax=baseline_len - 1 / filt_raw.info['sfreq'],
                        baseline=config_params['epochs']['baseline'], detrend=config_params["epochs"]["detrend"],
                        preload=True)
    return epochs


epochs = get_epochs(filt_raw)  # works
epochs.plot()

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
    ax[0].hist(1e6 * numpy.array(list(ar.threshes_.values())), 30,
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
    idx, jdx = numpy.unravel_index(loss.argmin(), loss.shape)
    rect = patches.Rectangle((idx - 0.5, jdx - 0.5), 1, 1, linewidth=2,
                             edgecolor="r", facecolor="none")
    ax[1].add_patch(rect)
    ax[1].xaxis.set_ticks_position("bottom")
    ax[1].set(xlabel=r"Consensus percentage $\kappa$",
              ylabel=r"Max sensors interpolated $\rho$",
              title="Mean cross validation error (x 1e6)")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(fig_path / pathlib.Path("autoreject_best_fit.pdf"), dpi=800)
    plt.close()
    evoked_bad = epochs[reject_log.bad_epochs].average()
    snr_ar = snr(epochs_ar)
    plt.plot(evoked_bad.times, evoked_bad.data.T * 1e06, 'r', zorder=-1)
    epochs_ar.average().plot(axes=plt.gca(), show=False,
                             titles=f"SNR: {snr_ar:.2f}")
    plt.savefig(
        fig_path / pathlib.Path("autoreject_results.pdf"), dpi=800)
    plt.close()
    epochs_ar.plot_drop_log(show=False)
    plt.savefig(
        fig_path / pathlib.Path("epochs_drop_log.pdf"), dpi=800)
    plt.close()
    return epochs_ar


# Ransac
def ransac_clean(epochs, n_jobs=4, n_resample=50, min_channels=0.25, min_corr=0.75, unbroken_time=0.4):
    epochs_clean = epochs.copy()
    ransac = Ransac(n_jobs=n_jobs, n_resample=n_resample, min_channels=min_channels,
                    min_corr=min_corr, unbroken_time=unbroken_time)  # optimize speed
    ransac.fit(epochs_clean)
    epochs_clean.average().plot(exclude=[])
    bads = input("Visual inspection for bad sensors: ").split()
    if len(bads) != 0 and bads not in ransac.bad_chs_:
        ransac.bad_chs_.extend(bads)
    epochs_clean = ransac.transform(epochs_clean)
    return epochs_clean, ransac

epochs_clean, ransac = ransac_clean(epochs)
# plot before and after Ransac:
def plot_epochs_clean(ransac, epochs, epochs_clean, plot=True):
    evoked = epochs.average()
    evoked_clean = epochs_clean.average()
    evoked.info['bads'] = ransac.bad_chs_
    evoked_clean.info['bads'] = ransac.bad_chs_
    if plot==True:
        fig, ax = plt.subplots(2)
        evoked.plot(exclude=[], axes=ax[0], show=False)
        evoked_clean.plot(exclude=[], axes=ax[1], show=False)
        ax[0].set_title(f"Before RANSAC (bad chs:{ransac.bad_chs_})")
        ax[1].set_title("After RANSAC")
        fig.tight_layout()
        fig.savefig(
            fig_path/ pathlib.Path("RANSAC_results.pdf"), dpi=800)
        plt.close()
    return evoked, evoked_clean


def reref(snr, epochs_clean):
    """
    Type "average": Create a robust average reference by first interpolating the bad channels
    to exclude outliers. Take mean voltage over all inlier channels as reference.
    """
    epochs = epochs_clean.copy()
    epochs_clean.set_eeg_reference(ref_channels="average", projection=True)
    average_reference = epochs_clean.info["projs"]
    epochs_clean.add_proj(average_reference)
    epochs_clean.apply_proj()
    snr_pre = snr(epochs)
    snr_post = snr(epochs_clean)
    epochs_reref = epochs_clean.copy()

    return epochs_reref, snr_pre, snr_post


def plot_reref(epochs_reref, snr_pre, snr_post):
    # Plot the results of rereferencing
    fig, ax = plt.subplots(2)
    epochs.average().plot(axes=ax[0], show=False)
    epochs_reref.average().plot(axes=ax[1], show=False)
    ax[0].set_title(f"FCz, SNR={snr_pre:.2f}")
    ax[1].set_title(f"{type}, SNR={snr_post:.2f}")
    fig.tight_layout()
    fig.savefig(
        fig_path / pathlib.Path("average_reference.pdf"), dpi=800)
    plt.close()


def apply_ICA(epochs_reref):
    """
    Run independent component analysis. Fit all epochs to the mne.ICA class, use
    reference_ica.fif to show the algorithm how blinks and saccades look like.
    Apply ica and save components to keep track of the excluded component topography.
    """
    reference_path = data_dir / 'misc' / config_params["ica"]["reference"]

    epochs_ica = epochs_reref.copy()
    snr_pre_ica = snr(epochs_ica)
    ica = ICA(n_components=config_params["ica"]["n_components"], method=config_params["ica"]["method"])
    ica.fit(epochs_ica)
    ref = mne.preprocessing.read_ica(fname=reference_path)
    ica.plot_sources(epochs_ica)

    ica.exclude = [1]  # List of integer components to exclude

    # Plot excluded components
    ica.plot_components(ica.exclude, show=False)
    plt.savefig(fig_path / pathlib.Path("ICA_components.pdf"), dpi=800)
    plt.close()

    # Apply ICA
    ica.apply(epochs_ica)

    # Plot ICA sources
    ica.plot_sources(inst=epochs, show=False, start=0,
                     stop=10, show_scrollbars=False)
    plt.savefig(fig_path / pathlib.Path(f"ICA_sources.pdf"), dpi=800)
    plt.close()
    # Plot SNR change during ICA
    snr_post_ica = snr(epochs_ica)
    ica.plot_overlay(epochs.average(), exclude=ica.exclude,
                     show=False, title=f"SNR: {snr_pre_ica:.2f} (before), {snr_post_ica:.2f} (after)")
    plt.savefig(fig_path / pathlib.Path("ICA_results.pdf"), dpi=800)
    plt.close()

    return epochs_ica

# todo Viola et al., 2009


def bad_epochs(epochs_ica):
    epochs_ica_clean = autoreject_epochs(
        epochs_ica, n_interpolate=config_params["autoreject"]["n_interpolate"],
        n_jobs=config_params["autoreject"]["n_jobs"],
        cv=config_params["autoreject"]["cv"],
        thresh_method=config_params["autoreject"]["thresh_method"])
    epochs_ica_clean.apply_baseline((None, 0))
    epochs_ica_clean.save(
        fig_path / pathlib.Path(id + "-epo.fif"), overwrite=True)

    # Plot the AutoReject results
    fig, ax = plt.subplots(2)
    epochs_ica_clean.average().plot_image(
        titles=f"SNR:{snr(epochs_ica_clean):.2f}", show=False, axes=ax[0])

    epochs_ica_clean.average().plot(show=False, axes=ax[1])
    plt.tight_layout()
    plt.savefig(
        fig_path / pathlib.Path("clean_evoked.pdf"), dpi=800)
    plt.close()
    return epochs_ica_clean


# evoked responses:
def evokeds(epochs_ica_clean):
    evokeds = epochs_ica_clean.average()
    evokeds.plot()
    mne.write_evokeds(fig_path / pathlib.Path(subject + "-ave.fif"), evokeds, overwrite=True)

    return evokeds
