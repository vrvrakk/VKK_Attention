from mne.preprocessing import ICA
from autoreject import AutoReject, Ransac
from matplotlib import pyplot as plt, patches
import mne
import numpy as np
import pathlib
from meegkit import dss
from pathlib import Path
import os
import json
# %matplotlib qt

_scaling = 10**6
default_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data')
raw_dir = default_dir / 'eeg' / 'raw'
cfg_path = default_dir / 'misc'
fig_folder = default_dir / 'eeg' / 'preprocessed' / 'figures'

with open(cfg_path / "preproc_config.json") as file:

    cfg = json.load(file)


def append_items_to_dict(d_in: dict, d_out: dict):

    for k, v in d_in.items():

        d_out.setdefault(k, []).append(v)


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


def filtering(data, notch=None, highpass=None, lowpass=None, fig_folder=None):

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

        X, _ = dss.dss_line_iter(X, fline=cfg["filtering"]["notch"],

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


def autoreject_epochs(epochs,

                      n_interpolate=[1, 4, 8, 16],

                      consensus=None,

                      cv=10,

                      thresh_method="bayesian optimization",

                      n_jobs=-1,

                      random_state=None, fig_folder=None):

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


def reref(epochs, type="average", n_jobs=4, n_resample=50, min_channels=0.25,

          min_corr=0.95, unbroken_time=0.4, plot=True, fig_folder=None):

    """

    If type "average": Create a robust average reference by first interpolating the bad channels

    to exclude outliers. Take mean voltage over all inlier channels as reference.

    If type "rest": use reference electrode standardization technique (point at infinity).

    epochs: mne.Epoch object.

    type: string --> "average", "rest", "lm" (linked mastoids)

    """

    if type == "average":

        epochs_clean = epochs.copy()

        ransac = Ransac(n_jobs=n_jobs, n_resample=n_resample, min_channels=min_channels,

                        min_corr=min_corr, unbroken_time=unbroken_time)  # optimize speed

        ransac.fit(epochs_clean)

        epochs_clean.average().plot(exclude=[])

        bads = input("Visual inspection for bad sensors: ").split()

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

        fig.savefig(

            fig_folder / pathlib.Path("RANSAC_results.pdf"), dpi=800)

        plt.close()

        epochs = epochs_clean.copy()

        epochs_clean.set_eeg_reference(ref_channels="average", projection=True)

        average_reference = epochs_clean.info["projs"]

        epochs_clean.add_proj(average_reference)

        epochs_clean.apply_proj()

        snr_pre = snr(epochs)

        snr_post = snr(epochs_clean)

        epochs_reref = epochs_clean.copy()

    if type == "rest":

        sphere = mne.make_sphere_model("auto", "auto", epochs.info)

        src = mne.setup_volume_source_space(

            sphere=sphere, exclude=30., pos=5.)

        forward = mne.make_forward_solution(

            epochs.info, trans=None, src=src, bem=sphere)

        epochs_reref = epochs.copy().set_eeg_reference("REST", forward=forward)

        snr_pre = snr(epochs)

        snr_post = snr(epochs_reref)

    if type == "lm":

        epochs_reref = epochs.copy().set_eeg_reference(["TP9", "TP10"])

        snr_pre = snr(epochs)

        snr_post = snr(epochs_reref)

    if plot == True:

        fig, ax = plt.subplots(2)

        epochs.average().plot(axes=ax[0], show=False)

        epochs_reref.average().plot(axes=ax[1], show=False)

        ax[0].set_title(f"FCz, SNR={snr_pre:.2f}")

        ax[1].set_title(f"{type}, SNR={snr_post:.2f}")

        fig.tight_layout()

        fig.savefig(

            fig_folder / pathlib.Path(f"{type}_reference.pdf"), dpi=800)

        plt.close()

    return epochs_reref


def apply_ICA(epochs, reference, n_components=None, method="fastica",

              threshold="auto", n_interpolate=None):

    """

    Run independent component analysis. Fit all epochs to the mne.ICA class, use

    reference_ica.fif to show the algorithm how blinks and saccades look like.

    Apply ica and save components to keep track of the excluded component topography.

    """

    epochs_ica = epochs.copy()

    snr_pre_ica = snr(epochs_ica)

    # ar = AutoReject(n_interpolate=n_interpolate, n_jobs=n_jobs)

    # ar.fit(epochs_ica)

    # epochs_ar, reject_log = ar.transform(epochs_ica, return_log=True)

    ica = ICA(n_components=n_components, method=method)

    # ica.fit(epochs_ica[~reject_log.bad_epochs])

    ica.fit(epochs_ica)

    # reference ICA containing blink and saccade components.

    ref = mne.preprocessing.read_ica(fname=reference)

    # .labels_ dict must contain "blinks" key with int values.

    ica.plot_components()

    ica.exclude = [int(x) for x in input("ICA exclude: ").split()]

    ica.plot_components(ica.exclude, show=False)

    plt.savefig(fig_folder / pathlib.Path("ICA_components.pdf"), dpi=800)

    plt.close()

    # components = ref.labels_["blinks"]

    # for component in components:

    #     mne.preprocessing.corrmap([ref, ica], template=(0, components[component]),

    #                               label="blinks", plot=False, threshold=cfg["ica"]["threshold"])

    # ica.plot_components(ica.labels_["blinks"])

    # plt.savefig(fig_folder / pathlib.Path("ICA_components.pdf"), dpi=800)

    # plt.close()

    ica.apply(epochs_ica)  # apply ICA

    ica.plot_sources(inst=epochs, show=False, start=0,

                     stop=10, show_scrollbars=False)

    plt.savefig(fig_folder / pathlib.Path(f"ICA_sources.pdf"), dpi=800)

    plt.close()

    snr_post_ica = snr(epochs_ica)

    ica.plot_overlay(epochs.average(), exclude=ica.exclude,

                     show=False, title=f"SNR: {snr_pre_ica:.2f} (before), {snr_post_ica:.2f} (after)")

    plt.savefig(fig_folder / pathlib.Path("ICA_results.pdf"), dpi=800)

    plt.close()

    return epochs_ica


def grad_psd(data, data_filter, data_filtered, fig_folder):

    cm = 1 / 2.54

    fig, ax = plt.subplots(3, sharex=True, sharey=True, figsize=(30*cm, 30*cm))

    fig.suptitle("Power Spectral Density")

    ax[0].set_title("before filtering")

    ax[1].set_title("after power noise filtering")

    ax[2].set_title("after high and low pass filtering")

    ax[0].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")

    ax[1].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")

    ax[2].set(xlabel="Frequency (Hz)", ylabel="μV²/Hz (dB)")

    data.plot_psd(ax=ax[0], show=False)

    data_filter.plot_psd(ax=ax[1], show=False)

    data_filtered.plot_psd(ax=ax[2], show=False)

    fig.savefig(fig_folder / "filter.pdf", dpi = 1000)