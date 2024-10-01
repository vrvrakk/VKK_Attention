import freefield
import slab
import time
import numpy
from pathlib import Path
from copy import deepcopy
import pickle
# import matplotlib
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt



"""
Equalize the loudspeaker array in two steps. First: equalize over all
level differences by a constant for each speaker. Second: remove spectral
difference by inverse filtering. For more details on how the
inverse filters are computed see the documentation of slab.Filter.equalizing_filterbank
"""

# initialize setup with standard samplerate (48824)
fs = 48828 # sampling frequency of the processors
freefield.initialize('dome', default='play_rec') # initialize using the 'play_red' mode (for calibrating I guess)
freefield.set_logger('warning') # this to make freefield Info warnings shut the hell up
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.

# dome parameters
reference_speaker = 23 # this is the reference speaker
azimuthal_angles = numpy.array([-17.5, 17.5]) # for the additional speakers -> for now they have no individual coordinates


# signal parameters
low_cutoff = 100 # lowest freq it picks up
high_cutoff = 20000
rec_repeat = 20  # how often to repeat measurement for averaging
# signal for loudspeaker calibration
signal_length = 1.0  # how long should the chirp be?
ramp_duration = signal_length/50
# use quadratic chirp to gain power in low freqs
signal = slab.Sound.chirp(duration=signal_length, level=85, from_frequency=low_cutoff, to_frequency=high_cutoff, kind='linear')
signal = slab.Sound.ramp(signal, when='both', duration=ramp_duration)

# equalization parameters
level_threshold = 0.3  # correct level only for speakers that deviate more than <threshold> dB from reference speaker
freq_bins = 1000  # can not be changed as of now
level_threshold = 0.3  # correct level only for speakers that deviate more than <threshold> dB from reference speaker
bandwidth = 1 / 50
alpha = 1.0

# obtain target signal by recording from reference speaker
reference_speaker = freefield.pick_speakers(reference_speaker)[0]
temp_recs = []
for i in range(rec_repeat):
    rec = freefield.play_and_record(speaker=reference_speaker, sound=signal, equalize=False, recording_samplerate=48828)
    # rec = slab.Sound.ramp(rec, when='both', duration=0.01)
    temp_recs.append(rec.data)
target = slab.Sound(data=numpy.mean(temp_recs, axis=0)) # get a Sound as the mean of all the 20 chirp repetitions



# # use original signal as reference - WARNING could result in unrealistic equalization filters,
baseline_amp = target.level  # at 10 db gain on preamp and signal level 90 dB!
dome_rec = []  # store all recordings from the dome for final spectral difference
equalization = dict()  # dictionary to hold equalization parameters


#------------------- hold on --------------------#

speakers = freefield.pick_speakers([(17.5, 0), (-17.5, 0)])
# place microphone 90° to source column at equal distance (recordings should be done in far field: > 1m)


# ---- START calibration ----#
# step 1: level equalization
"""
Record the signal from each speaker in the list and return the level of each
speaker relative to the target speaker(target speaker must be in the list)
"""

def level_equalization(speakers, signal):
    temp_recs = []
    for i in range(rec_repeat): # 20 repeats
        rec = freefield.play_and_record(speakers, signal, equalize=False, recording_samplerate=48828) # play chirp from selected speaker coordinates
        # rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data) # append recorded data of selected speaker into temp_recs
    recordings_leveled = slab.Sound(data=numpy.mean(temp_recs, axis=0))  # averaging along the rows, keeping the columns
    recordings_og = slab.Sound(recordings_leveled) # keep original data as well
    equalization_levels = target.level - recordings_leveled.level # get the equalization levels for the recording -> to match level of ref sound
    if not (target.level - level_threshold <= recordings_leveled.level <= target.level + level_threshold):
        # Calculate a scaling factor to match the target level
        scaling_factor = 10 ** ((target.level - recordings_leveled.level) / 20)
        # Adjust the amplitude of the recording to match the target level
        recordings_leveled.data *= scaling_factor  # Scale the amplitude values to match the target level
    return recordings_og, recordings_leveled, equalization_levels

# do this for both speakers -> level of recordings from speakers equalized based on ref recording from midline
recordings_og_left, recordings_leveled_left, equalization_levels_left = level_equalization(speakers=speakers[0], signal=signal)
recordings_og_right, recordings_leveled_right, equalization_levels_right = level_equalization(speakers=speakers[1], signal= signal)


# set up plot
def plot_recordings(recordings_og, recordings_leveled):
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(25, 10))
    ax[1].set_xlim(left=200, right=18000)
    ax[1].set_ylim(0, 50)
    ax[1].set_xlabel('Frequency (Hz)')
    # Set y-axis labels for both subplots
    for i in range(2):
        ax[i].set_ylabel('Power (dB/Hz)')
    # Plot original recording in the first subplot
    freefield.spectral_range(recordings_og, plot=ax[0], bandwidth=bandwidth)
    ax[0].set_title('Original Recording')
    # Plot leveled recording in the second subplot
    freefield.spectral_range(recordings_leveled, plot=ax[1], bandwidth=bandwidth)
    ax[1].set_title('Leveled Recording')
    plt.show()

plot_recordings(recordings_og_left, recordings_leveled_left)
plot_recordings(recordings_og_right, recordings_leveled_right)

# step 2: frequency equalization
"""
play the level-equalized signal, record and compute a bank of inverse filters
to equalize each speakers frequency response relative to the target speaker
"""


def freq_localization(recordings_leveled, target, freq_bins, low_cutoff, high_cutoff, bandwidth, alpha):
    """
    Use the already leveled recording and create an equalizing filter bank.
    recordings_leveled: Already leveled recording from the level_equalization function.
    target: The reference sound to which the filter is applied.
    """
    # Since `recordings_leveled` already has the desired level, use it directly
    # Create the equalizing filter bank using the leveled recording
    filter_bank = slab.Filter.equalizing_filterbank(
        target, recordings_leveled, length=freq_bins, low_cutoff=low_cutoff,
        high_cutoff=high_cutoff, bandwidth=bandwidth, alpha=alpha
    )

    return filter_bank


filter_bank_left = freq_localization(recordings_leveled_left, target, freq_bins, low_cutoff, high_cutoff, bandwidth, alpha)
filter_bank_right = freq_localization(recordings_leveled_right, target, freq_bins, low_cutoff, high_cutoff, bandwidth, alpha)

# plot
def plot_spectral_range(recordings_og, recordings_leveled):
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))  # Creates a figure with 2 subplots side by side
    # fig is the figure object, ax is an array of two axes (one for each subplot)

    # Now you can use ax[0], ax[1], etc., to refer to each subplot
    # Plot on the first subplot
    og = freefield.spectral_range(recordings_og, plot=ax[0])
    ax[0].set_title('Original Signal Level')

    # This is where you can use ax[1] for the second subplot
    # Replace the commented lines in your code like this:
    diff = freefield.spectral_range(recordings_leveled, plot=ax[1])
    ax[1].set_title('Level Equalized')

    # Show the plot
    plt.show()

plot_spectral_range(recordings_leveled_left, recordings_og_left)
plot_spectral_range(recordings_leveled_right, recordings_og_right)


def transfer_function(filter_bank):
    # check for notches in the filter:
    transfer_func = filter_bank.tf(show=False)[1][0:900, :]
    if (transfer_func < -30).sum() > 0:
        print("Some of the equalization filters contain deep notches - try adjusting the parameters.")

transfer_function(filter_bank_left)
transfer_function(filter_bank_right)

#
# step 3: filter bank
def filter_bank(recordings_leveled, target, dome_rec, filter_bank):
    recordings_filtered = filter_bank.apply(recordings_leveled)
    dome_rec.append(recordings_filtered)
    return dome_rec, recordings_filtered

recordings_left_equalized = filter_bank(recordings_leveled_left, dome_rec, filter_bank_left)
recordings_right_equalized = filter_bank(recordings_leveled_right, dome_rec, filter_bank_right)
# plot
def plot_equalized_recs(equalization, recordings, recordings_equalized, equalization_levels, speakers, filter_bank):
    """
    Function to plot the original and equalized recordings and save the equalization data.

    Parameters:
    - equalization: Dictionary to store the equalization results.
    - recordings: Original recording data (slab.Sound object).
    - recordings_equalized: Equalized recording data (slab.Sound object).
    - equalization_levels: Levels used to equalize the recordings.
    - speakers: Speaker object with properties like azimuth and index.
    - filter_bank: Filter bank applied to the recordings.

    Returns:
    - equalization: Updated dictionary with new equalization data.
    """
    # Step 1: Create the figure with two subplots
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))  # Increase figsize for better readability

    # Step 2: Plot the original recording's spectral range
    og = freefield.spectral_range(recordings, plot=ax[0])
    ax[0].set_title('Original Frequency Response')
    ax[0].set_xlabel('Frequency (Hz)')
    ax[0].set_ylabel('Power (dB/Hz)')

    # Step 3: Plot the equalized recording's spectral range
    diff = freefield.spectral_range(recordings_equalized, plot=ax[1])
    ax[1].set_title('Equalized Frequency Response')
    ax[1].set_xlabel('Frequency (Hz)')
    ax[1].set_ylabel('Power (dB/Hz)')

    # Step 4: Add an overall title for the figure
    fig.suptitle(
        f'Calibration for Dome Speaker Column at {speakers.azimuth:.1f}° Azimuth\n'
        f'Difference in Power Spectrum Before and After Equalization',
        fontsize=18
    )

    # Step 5: Adjust layout and show the plot
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust layout to make space for suptitle
    plt.show()

    # Step 6: Save the equalization data for this speaker
    array_equalization = {f"{speakers.index}": {"level": equalization_levels, "filter": filter_bank.channel(0)}}
    equalization.update(array_equalization)  # Update the main equalization dictionary

    return equalization

equalization = plot_equalized_recs(equalization, recordings_og_left, recordings_left_equalized, equalization_levels_left, speakers[0], filter_bank_left)

# Example call for right speaker
equalization = plot_equalized_recs(equalization, recordings_og_right, recordings_right_equalized, equalization_levels_right, speakers[1], filter_bank_right)


# write final equalization to pkl file
# freefield_path = freefield.DIR / 'data'
project_path = Path.cwd() / 'data' / 'misc' / 'calibration'  # todo varvara get your own path to your project
equalization_path = project_path / f'calibration_speakers'  # .10?
import os
if not os.path.exists(project_path):
    os.makedirs(project_path)
with open(equalization_path, 'wb') as f:  # save the newly recorded calibration
    pickle.dump(equalization, f, pickle.HIGHEST_PROTOCOL)


#todo varvara: in your experiment, load the pickle file (equalization dict), and apply the level and filter to the signal (wav files)
# todo this:
#     pick your speaker from the dict, and get level and filter_bank
#     attenuated = deepcopy(signal)  <- this is your .wav
#     attenuated = filter_bank.apply(attenuated)  <- choose the right speaker
#     attenuated.level += level  # which order? doesnt seem to matter much

