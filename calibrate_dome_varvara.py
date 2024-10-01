import numpy as np
import slab
import freefield
import matplotlib.pyplot as plt
import pickle
from copy import deepcopy
from pathlib import Path
import os

# --- Initialization and Parameters ---
fig_path = Path.cwd() / 'data' / 'misc' / 'calibration_figures'
os.makedirs(fig_path, exist_ok=True)  # Create directory if not exists
fs = 48828  # Sampling frequency of the processors
freefield.initialize('dome', default='play_rec')  # Initialize using 'play_rec' mode
freefield.set_logger('warning')  # Set logger to only show warnings
slab.Signal.set_default_samplerate(fs)  # Set default samplerate for generating sounds, filters, etc.

# Dome and signal parameters
reference_speaker = 23  # Reference speaker index
azimuthal_angles = np.array([-17.5, 17.5])  # Angles for the selected speakers
low_cutoff, high_cutoff = 100, 20000  # Frequency range for the chirp
rec_repeat = 20  # Number of repetitions for averaging
signal_length, ramp_duration = 1.0, 1.0 / 50  # Signal length and ramp duration
bandwidth, alpha = 1 / 50, 1.0  # Bandwidth and alpha parameters for filtering
level_threshold = 0.3  # Threshold to correct speaker levels (in dB)
freq_bins = 1000  # Number of frequency bins (fixed)

# Create the signal to be used for equalization
signal = slab.Sound.chirp(duration=signal_length, level=85, from_frequency=low_cutoff, to_frequency=high_cutoff,
                          kind='linear')
signal = slab.Sound.ramp(signal, when='both', duration=ramp_duration)

# --- Obtain Target Signal from the Reference Speaker ---
reference_speaker = freefield.pick_speakers(reference_speaker)[0]
temp_recs = [
    freefield.play_and_record(speaker=reference_speaker, sound=signal, equalize=False, recording_samplerate=fs).data for
    _ in range(rec_repeat)]
target = slab.Sound(data=np.mean(temp_recs, axis=0))

# Store baseline amplitude and initialize recording containers
baseline_amp = target.level
dome_rec = []  # Store all dome recordings
equalization = {}  # Dictionary to hold equalization parameters

# Pick speakers for calibration (based on their azimuthal angles)
speakers = freefield.pick_speakers([(17.5, 0), (-17.5, 0)])


# --- Step 1: Level Equalization Function ---
def level_equalization(speakers, signal):
    """
    Perform level equalization for a given speaker.

    Parameters:
    - speakers: Selected speaker for level equalization.
    - signal: Signal to play during equalization.

    Returns:
    - recordings_og: Original recording.
    - recordings_leveled: Leveled recording.
    - equalization_levels: Levels used for equalization.
    """
    temp_recs = [freefield.play_and_record(speakers, signal, equalize=False, recording_samplerate=fs).data for _ in
                 range(rec_repeat)]
    recordings_leveled = slab.Sound(data=np.mean(temp_recs, axis=0))  # Average across repetitions
    recordings_og = slab.Sound(recordings_leveled)  # Keep original data for comparison
    equalization_levels = target.level - recordings_leveled.level  # Calculate level difference

    # Apply scaling if recording level is outside the threshold range
    if not (target.level - level_threshold <= recordings_leveled.level <= target.level + level_threshold):
        print('Recording Level not within boundaries.')
        scaling_factor = 10 ** ((target.level - recordings_leveled.level) / 20)
        recordings_leveled.data *= scaling_factor  # Scale amplitude values to match target level
    else:
        print('Recording Level within boundaries.')

    return recordings_og, recordings_leveled, equalization_levels


# --- Step 2: Frequency Equalization Function ---
def freq_localization(recordings_leveled, target, freq_bins, low_cutoff, high_cutoff, bandwidth, alpha):
    """
    Create an equalizing filter bank using the leveled recording.

    Parameters:
    - recordings_leveled: Leveled recording data.
    - target: Reference sound to match.

    Returns:
    - filter_bank: Created filter bank for frequency equalization.
    """
    filter_bank = slab.Filter.equalizing_filterbank(target, recordings_leveled, length=freq_bins, low_cutoff=low_cutoff,
                                                    high_cutoff=high_cutoff, bandwidth=bandwidth, alpha=alpha)
    print('Filter bank created.')
    return filter_bank


# Step 3: Apply Filter Bank Directly
def filter_bank(recordings_leveled, dome_rec, filter_bank):
    recordings = filter_bank.channel(0).apply(recordings_leveled)
    recordings_filtered = slab.Sound(recordings)
    dome_rec.append(recordings_filtered)
    print('Filter bank applied.')
    return dome_rec, recordings_filtered


# --- Plotting Functions ---

def plot_equalized_recs(equalization, recordings, recordings_equalized, equalization_levels, speakers, filter_bank, target_rec):
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
    save_filename = f"{target_rec}_equalized_recordings_plot.png"
    plt.savefig(fig_path / save_filename)
    plt.close(fig)  # Close the figure to free up memory

    # Step 6: Save the equalization data for this speaker
    array_equalization = {f"{speakers.index}": {"level": equalization_levels, "filter": filter_bank.channel(0)}}
    equalization.update(array_equalization)  # Update the main equalization dictionary

    return equalization


def plot_recordings(recordings_og, recordings_leveled, target_rec, title_prefix=""):
    """
    Plot the original and leveled recordings side by side.

    Parameters:
    - recordings_og: Original recording (slab.Sound object).
    - recordings_leveled: Leveled recording after equalization (slab.Sound object).
    - title_prefix: Optional prefix for the plot titles.
    """
    fig, ax = plt.subplots(2, 1, sharex=True, sharey=True, figsize=(12, 8))
    ax[1].set_xlim(left=200, right=18000)
    ax[1].set_ylim(0, 50)
    ax[1].set_xlabel('Frequency (Hz)')

    for i in range(2):
        ax[i].set_ylabel('Power (dB/Hz)')

    # Plot original and leveled recordings
    freefield.spectral_range(recordings_og, plot=ax[0], bandwidth=bandwidth)
    ax[0].set_title(f'{title_prefix} Original Recording')
    freefield.spectral_range(recordings_leveled, plot=ax[1], bandwidth=bandwidth)
    ax[1].set_title(f'{title_prefix} Leveled Recording')
    save_filename = f"{target_rec}_og_vs_leveled_recordings.png"
    plt.savefig(fig_path / save_filename)
    plt.close(fig)  # Close the figure to free up memory



def plot_spectral_range(recordings_og, recordings_equalized, target_rec, title=""):
    """
    Plot the original and equalized recordings side by side.

    Parameters:
    - recordings_og: Original recording.
    - recordings_equalized: Equalized recording.
    - title: Title prefix for the plots.
    """
    fig, ax = plt.subplots(1, 2, figsize=(16, 8))
    freefield.spectral_range(recordings_og, plot=ax[0])
    ax[0].set_title(f'{title} Original Signal Level')
    freefield.spectral_range(recordings_equalized, plot=ax[1])
    ax[1].set_title(f'{title} Level Equalized')
    save_filename = f"{target_rec}_original_vs_equalized.png"
    plt.savefig(fig_path / save_filename)
    plt.close(fig)  # Close the figure to free up memory


# LEFT SPEAKER:
# Perform level equalization for both speakers
recordings_og_left, recordings_leveled_left, equalization_levels_left = level_equalization(speakers=speakers[0],signal=signal)
# Create equalizing filters for both speakers
filter_bank_left = freq_localization(recordings_leveled_left, target, freq_bins, low_cutoff, high_cutoff, bandwidth, alpha)
dome_rec, recordings_filtered_left = filter_bank(recordings_leveled_left, dome_rec, filter_bank_left)
equalization = plot_equalized_recs(equalization, recordings_og_left, recordings_filtered_left, equalization_levels_left, speakers[0], filter_bank_left, target_rec='left')
plot_recordings(recordings_og_left, recordings_leveled_left, title_prefix="Left Speaker", target_rec='left')
plot_spectral_range(recordings_og_left, recordings_filtered_left, target_rec='left', title="")
# RIGHT SPEAKER:
recordings_og_right, recordings_leveled_right, equalization_levels_right = level_equalization(speakers=speakers[1], signal=signal)
filter_bank_right = freq_localization(recordings_leveled_right, target, freq_bins, low_cutoff, high_cutoff, bandwidth, alpha)
dome_rec, recordings_filtered_right = filter_bank(recordings_leveled_right, dome_rec, filter_bank_right)
equalization = plot_equalized_recs(equalization, recordings_og_right, recordings_filtered_right, equalization_levels_right, speakers[1], filter_bank_right, target_rec='right')
plot_recordings(recordings_og_right, recordings_leveled_right, title_prefix="Right Speaker", target_rec='right')
plot_spectral_range(recordings_og_right, recordings_filtered_right, target_rec='right', title="")


# --- Plot Results ---


# --- Save Final Equalization to File ---
project_path = Path.cwd() / 'data' / 'misc' / 'calibration'
equalization_path = project_path / 'calibration_speakers.pkl'

os.makedirs(project_path, exist_ok=True)  # Create directory if not exists

# Save the equalization parameters using pickle
with open(equalization_path, 'wb') as f:
    pickle.dump(equalization, f, pickle.HIGHEST_PROTOCOL)

print(f"Equalization parameters saved successfully to {equalization_path}")

#todo varvara: in your experiment, load the pickle file (equalization dict), and apply the level and filter to the signal (wav files)
# todo this:
#     pick your speaker from the dict, and get level and filter_bank
#     attenuated = deepcopy(signal)  <- this is your .wav
#     attenuated = filter_bank.apply(attenuated)  <- choose the right speaker
#     attenuated.level += level  # which order? doesnt seem to matter much

