from pathlib import Path
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

streams = ['s1', 's2']
axes = ['azimuth', 'elevation']

data_dir = Path.cwd() / 'data'
eeg_dir = data_dir / 'eeg' / 'raw'
subject_dir = eeg_dir / 's1' / 'azimuth'
electrode_names = json.load(open(data_dir / 'misc' / "electrode_names.json"))
# tmin, tmax and event_ids for both experiments

markers_dict = {'s1_1': 1, 's1_2': 2, 's1_3': 3, 's1_4': 4, 's1_5': 5, 's1_6': 6, 's1_8': 8, 's1_9': 9,  # stimulus 1 markers
's2_1': 65, 's2_2': 66, 's2_3': 67, 's2_4': 68, 's2_5': 69, 's2_6': 70, 's2_8': 72, 's2_9': 73,  # stimulus 2 markers
'R_1': 129, 'R_2': 130, 'R_3': 131, 'R_4': 132, 'R_5': 133, 'R_6': 134, 'R_8': 136, 'R_9': 137}  # response markers
epoch_parameters = [-0.5,  # tmin
                    1.5,  # tmax
                    markers_dict]

# get subject files
# header contains main eeg data: i.e. sampling freq
for stream_name in streams:
    for subfolder in eeg_dir.glob(f'*{stream_name}*'):
        # print(subfolder)
        for axis in axes:
            subsubfolder = subfolder / axis
            if subsubfolder.exists():
                # print(f"Found {axis} data in {subsubfolder}")
                output_dir = data_dir / 'eeg' / 'preprocessed' / stream_name / axis  # create output directory
                if not output_dir.exists():
                  output_dir.mkdir(parents=True)
#             # collect header files
                header_files = list(subject_dir.glob('*ll_azimuth.vhdr'))
                raws = []
                # concatenate blocks
                for header_file in header_files:
                    print(header_files)
                    raws.append(read_raw_brainvision(header_file))  # used to read info from eeg + vmrk files
                raw = mne.concatenate_raws(raws, preload=None, events_list=None, on_mismatch='raise', verbose=None)
                # this should concatenate all raw eeg files within one subsubfolder
                raw = raw.load_data()  # preload turns True under raw attributes

                # assign channel names to the data
                raw.rename_channels(electrode_names)
                # set_montage to get spatial distribution of channels
                raw.set_montage("standard_1020")  # ------ use brainvision montage instead?
                # montage_path = data_dir / 'misc' / "AS-96_REF.bvef"  # original version
                # montage = mne.channels.read_custom_montage(fname=montage_path)
                # raw.set_montage(montage)

                # inspect raw data
                raw.plot()
                raw.compute_psd().plot(average=True)
                fig = raw.plot_psd(xscale='linear', fmin=0, fmax=250)
                fig.suptitle(header_file.name)  # can be added after plotting

                print('STEP 1: Remove power line noise and apply minimum-phase highpass filter')
                X = raw.get_data().T # transpose -> create a 3D matrix-> Channels, Time, Voltage values
                X, _ = dss_line(X, fline=50, sfreq=raw.info["sfreq"], nremove=5) #todo PCA based (better than notch) Cheveigné, 2020

                # plot changes made by the filter:
                # plot before / after zapline denoising
                # power line noise is not fully removed with 5 components, remove 10
                f, ax = plt.subplots(1, 2, sharey=True)
                f, Pxx = signal.welch(raw.get_data().T, 500, nperseg=500, axis=0, return_onesided=True) # to get psd
                ax[0].semilogy(f, Pxx)
                f, Pxx = signal.welch(X, 500, nperseg=500, axis=0, return_onesided=True)
                ax[1].semilogy(f, Pxx) # plot on a log scale
                ax[0].set_xlabel("frequency [Hz]")
                ax[1].set_xlabel("frequency [Hz]")
                ax[0].set_ylabel("PSD [V**2/Hz]")
                ax[0].set_title("before")
                ax[1].set_title("after")
                plt.show()

                # put the data back into raw
                raw._data = X.T
                del X

                # remove line noise (eg. stray electromagnetic signals) -> high pass
                raw = raw.filter(l_freq=.5, h_freq=None, phase="minimum")
                # everything below 0.5Hz (electromagnetic drift)
                # minimum phase keeps temporal distortion to a minimum

                print('STEP 2: Epoch and downsample the data')
                # get events
                events = events_from_annotations(raw)[0] # get events from annotations attribute of raw variable
                events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
                # remove all meaningless event codes, including post trial events
                tmin, tmax, event_ids = epoch_parameters  # get epoch parameters
                epochs = Epochs(
                    raw,
                    events,
                    event_id=event_ids,
                    tmin=tmin,
                    tmax=tmax,
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

                print('STEP 4: interpolate bad channels and re-reference to average')  #todo Bigdely-Shamlo et al., 2015
                # todo new script for motor only eeg-> power line, drift + epoch + average
                # todo take into consideration movement + pre-motor potentials from eeg data
                # todo take motor template and subtract it from our button response data
                r = Ransac(n_jobs=4, min_corr=0.85)
                epochs_clean = epochs.copy()
                epochs_clean = r.fit_transform(epochs_clean)
                del r
                # 1 Interpolate all channels from a subset of channels (fraction denoted as min_channels),
                # repeat n_resample times.
                # 2 See if correlation of interpolated channels to original channel is above 75% per epoch (min_corr)
                # 3 If more than unbroken_time fraction of epochs have a lower correlation than that,
                # add channel to self.bad_chs_


                print('STEP 5: Blink rejection with ICA')  #todo Viola et al., 2009
                # reference = read_ica(data_dir / 'misc' / 'reference-ica.fif')
                # component = reference.labels_["blinks"]
                ica = ICA(n_components=0.999, method="fastica")
                ica.fit(epochs)
                # corrmap([ica, ica], template=(0, component[0]), label="blinks", plot=False, threshold=0.75)
                ica.plot_components(picks=range(10))  # first 10 independent components
                ica.plot_sources(epochs)
                ica.labels_["blinks"] = [0, 1]
                ica.apply(epochs, exclude=ica.labels_["blinks"])
                ica.save(output_dir / f"{subfolder.name}-ica.fif", overwrite=True)
                del ica

                epochs_clean.set_eeg_reference("average", projection=True)
                epochs.add_proj(epochs_clean.info["projs"][0])
                epochs.apply_proj()
                del epochs_clean

                print('STEP 6: Reject / repair bad epochs')  # Jas et al., 2017
                # ar = AutoReject(n_interpolate=[0, 1, 2, 4, 8, 16], n_jobs=4)
                ar = AutoReject(n_jobs=20)
                epochs = ar.fit_transform(epochs)  # Bigdely-Shamlo et al., 2015)?
                # apply threshold \tau_i to reject trials in the train set
                # calculate the mean of the signal( for each sensor and timepoint) over the GOOD (= not rejected)
                # trials in the train set
                # calculate the median of the signal(for each sensor and timepoint) over ALL trials in the test set
                # compare both of these signals and calculate the error
                # the candidate threshold with the lowest error is the best rejection threshold for a global rejection
                #todo: String: epochs['name'] will return an Epochs object comprising only the epochs labeled 'name'
                # (i.e., epochs created around events with the label 'name').
                s1_epochs = epochs['s1_1']


                # plot preprocessing results
                fig = epochs.plot_psd(xscale='linear', fmin=0, fmax=50)
                fig.suptitle(header_file.name)

                #  save peprocessed epochs to file
                epochs.save(outdir / f"{subfolder.name}-epo.fif", overwrite=True)
                ar.save(outdir / f"{subfolder.name}-autoreject.h5", overwrite=True)
