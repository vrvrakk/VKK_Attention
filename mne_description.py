LOADING FILES:
raw = mne.concatenate_raws(raws, preload=None, events_list=None, **on_mismatch='raise', verbose=None**)

Events List: In MNE-Python, an "events list" is a list containing arrays that represent events (markers) occurring in the EEG or MEG data.
Each event array has three columns: the sample number at which the event occurred, the previous event (if any),
and the event type (e.g., stimulus onset, response).

**on_mismatch**: Determines behavior when raw objects have different properties
(e.g., sampling frequency, channel names). Options are raise (raise an error), warn (issue a warning), or ignore

**verbose** parameter allows you to control how much information is printed to the console while the function is executing.
Setting verbose=True will print more information, such as progress updates or additional details about the concatenation process.

raw.compute_psd().plot(average=True)
average=False: see each channel separately

fig = raw.plot_psd(xscale='linear', fmin=0, fmax=250)
xscale can either be linear or logarithmic
fmin, fmax:
The lower- and upper-bound on frequencies of interest. Default is fmin=0, fmax=np.inf (spans all frequencies present in the data).

STEP 1: transposing raw data and removing power line noise!
    X = raw.get_data().T:
        raw.get_data(): This method retrieves the raw EEG data from the mne.io.Raw object raw.
        .T: This is the transpose operation, which rearranges the data matrix. In this context, it's used to swap the rows and columns of the data matrix. After transposing, the rows represent channels, columns represent time points, and the values represent voltage measurements.

    X, _ = dss_line(X, fline=50, sfreq=raw.info["sfreq"], nremove=20):
        dss_line(): This function is called with the transposed EEG data X as input.
        fline=50: This parameter specifies the frequency to target for artifact removal. In this case, it seems to be targeting line noise at 50 Hz.
        sfreq=raw.info["sfreq"]: This parameter specifies the sampling frequency of the EEG data, which is extracted from the raw object's info dictionary.
        nremove=20: This parameter specifies the number of PCA components to remove. It seems that 20 PCA components are targeted for removal.

Create Subplots:
    f, ax = plt.subplots(1, 2, sharey=True):
    This line creates a figure (f) and a pair of subplots (ax) arranged in a single row with two columns.
    Both subplots share the same y-axis scale (sharey=True).

Compute PSD with Welch's Method:
    f, Pxx = signal.welch(raw.get_data().T, 500, nperseg=500, axis=0, return_onesided=True):
    This line computes the PSD using Welch's method for the raw EEG data.
    raw.get_data().T: Retrieves the transposed EEG data matrix, where each row represents a channel and each column represents a time point.
    500: Sampling frequency, which is set to 500 Hz.
    nperseg=500: The length of each segment used for computing the PSD. Here, it's set to 500 samples.
    axis=0: Specifies that the frequency axis is along the 0th dimension (rows/channels) of the input data.
    return_onesided=True: Indicates that only one-sided PSD estimates are returned. This is suitable for real-valued signals, such as EEG data.

Plot PSD on the First Subplot:
    ax[0].semilogy(f, Pxx):
    This line plots the computed PSD (Pxx) on the first subplot (ax[0]) using a logarithmic scale on the y-axis.
    f, Pxx = signal.welch(X, 500, nperseg=500, axis=0, return_onesided=True): This line computes the PSD for the processed EEG data (X)
    after artifact removal using DSS.
    X: Processed EEG data after artifact removal, likely containing fewer artifacts and noise.
    The resulting PSD is plotted on the second subplot.


raw = raw.filter(l_freq=.5, h_freq=None, phase="minimum")
By minimizing phase distortion, minimum phase filters preserve the temporal characteristics of the signal as much as possible.
This is particularly important for tasks like event-related potential (ERP) analysis,
where the precise timing of neural responses is crucial for understanding cognitive processes.

STEP 4: interpolate bad channels + re-reference to average
Interpolation is a technique used to estimate missing or noisy data points in a signal by using information from neighboring data points.
In the context of EEG data pre-processing, interpolation is often used to replace "bad"
or noisy channels with estimates based on the surrounding channels.

Initialization of RANSAC Algorithm:
    Ransac(n_jobs=4, min_corr=0.85):
    This line initializes a RANSAC (Random Sample Consensus) algorithm object with certain parameters:
    n_jobs=4:
    This specifies the number of parallel jobs to run during the RANSAC algorithm.
    In this case, it indicates that the algorithm can utilize up to 4 CPU cores for faster computation if available.
    min_corr=0.85:
    This parameter sets the minimum correlation threshold.
    During the RANSAC algorithm, data points with correlations below this threshold are considered outliers and may be removed.

Copying the Epochs Data:
    epochs_clean = epochs.copy():
    This line creates a copy of the original epochs object and assigns it to a new variable epochs_clean.
    This step is important to ensure that the original epochs data is preserved and not modified during the cleaning process.

Applying RANSAC Algorithm:
    epochs_clean = r.fit_transform(epochs_clean):
    This line applies the RANSAC algorithm to the epochs_clean data.
    The .fit_transform() method of the RANSAC algorithm fits the model to the data and then transforms the data based on the fitted model.
    During the fitting process, RANSAC identifies and removes outliers
    or noisy data points from the epochs data based on the specified parameters (such as minimum correlation threshold).
    After the cleaning process, the epochs_clean data contains the cleaned epochs data, with outliers or noisy data points removed.

STEP 5: Blink rejection with ICA
