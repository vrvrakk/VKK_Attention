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
fs = 48828
freefield.initialize('dome', default='play_rec')

# initialize setup with modified samplerate (97656)
# fs = 97656
# proc_list = [['RP2', 'RP2', Path.cwd() / 'data' / 'rcx' / 'rec_buf.rcx'],
#              ['RX81', 'RX8', Path.cwd() / 'data' / 'rcx' / 'play_buf.rcx'],
#              ['RX82', 'RX8', Path.cwd() / 'data' / 'rcx' / 'play_buf.rcx']]
# freefield.initialize('dome', device=proc_list)
# freefield.PROCESSORS.mode = 'play_rec'


freefield.set_logger('warning')
slab.Signal.set_default_samplerate(fs)  # default samplerate for generating sounds, filters etc.

# dome parameters
reference_speaker = 23
azimuthal_angles = numpy.array([-17.5, 17.5])
# speaker_idx = [19,20,21,22,23,24,25,26,27]  # central array

# signal parameters
low_cutoff = 100
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
    rec = freefield.play_and_record(reference_speaker, signal, equalize=False, recording_samplerate=48828)
    # rec = slab.Sound.ramp(rec, when='both', duration=0.01)
    temp_recs.append(rec.data)
target = slab.Sound(data=numpy.mean(temp_recs, axis=0))

# # use original signal as reference - WARNING could result in unrealistic equalization filters,
#  can be used for HRTF measurement calibration to get really flat chirp spectra
baseline_amp = target.level  # at 10 db gain on preamp and signal level 90 dB!
# target = deepcopy(signal)
# target.level = baseline_amp

# get speaker id's for each column in the dome
# table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
# speaker_table = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
# speaker_list = []
# for az in azimuthal_angles:
#     speaker_list.append((speaker_table[speaker_table[:, 1] == az][:, 0]).astype('int'))
#
# speaker_list[3] = numpy.delete(speaker_list[3], [numpy.where(speaker_list[3] == 19), numpy.where(speaker_list[3] == 27)])



# dome_rec = []  # store all recordings from the dome for final spectral difference
equalization = dict()  # dictionary to hold equalization parameters


#------------------- hold on --------------------#
# pick single column to calibrate speaker_list[0] to speaker_list[6]
# speakers = freefield.pick_speakers(speaker_list[3])
# todo for varvara:
#  pick the right peakers: speakers = freefield.pick_speakers([[17.5,0], [-17.5, 0]])
speakers = freefield.pick_speakers([(17.5, 0), (-17.5, 0)])
# place microphone 90° to source column at equal distance (recordings should be done in far field: > 1m)


# ---- START calibration ----#
# step 1: level equalization
"""
Record the signal from each speaker in the list and return the level of each
speaker relative to the target speaker(target speaker must be in the list)
"""

def level_equalization(speakers, signal):
    recordings = []
    temp_recs = []
    for i in range(rec_repeat):
        rec = freefield.play_and_record(speakers, signal, equalize=False, recording_samplerate=48828)
        # rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data)
    recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
    # recordings.append(numpy.mean(temp_recs, axis=0))
    recordings = slab.Sound(recordings)
    equalization_levels = target.level - recordings.level
    if abs(recordings.level - target.level) > level_threshold:
        # Apply correction to the entire recordings.data array
        recordings.data = target.data  # Replace with the reference sound data

    return recordings, equalization_levels

recordings_left, equalization_levels_left = level_equalization(speakers=speakers[0], signal=signal)
recordings_right, equalization_levels_right = level_equalization(speakers=speakers[1], signal = signal)

# todo: this fucking bullshit does not work because paul gave me a shit ass script
# set up plot
def plot_recording(recordings):
    fig, ax = plt.subplots(3, 1, sharex=True, sharey=True, figsize=(25, 10))
    ax[2].set_xlim(left=200, right=18000)
    ax[2].set_ylim(0, 50)
    ax[2].set_xlabel('Frequency (Hz)')
    for i in range(3):
        ax[i].set_ylabel('Power (dB/Hz)')
    diff = freefield.spectral_range(recordings, plot=ax[0], bandwidth=bandwidth)
    ax[0].set_title('raw')

# step 2: frequency equalization
"""
play the level-equalized signal, record and compute a bank of inverse filters
to equalize each speakers frequency response relative to the target speaker
"""
def freq_localization(equalization_levels, signal, speakers):
    att_recordings = []
    attenuated = deepcopy(signal)
    attenuated.level += equalization_levels
    temp_recs = []
    for i in range(rec_repeat):
        rec = freefield.play_and_record(speakers[0], attenuated, equalize=False)
        # rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data)
    att_recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
    attenuated_recordings = slab.Sound(att_recordings)
    filter_bank = slab.Filter.equalizing_filterbank(target, attenuated_recordings, length=freq_bins, low_cutoff=low_cutoff,
                                                    high_cutoff=high_cutoff, bandwidth=bandwidth, alpha=alpha)
    return attenuated_recordings, filter_bank
# plot
# diff = freefield.spectral_range(attenuated_recordings, plot=ax[1])
# ax[1].set_title('level equalized')

# check for notches in the filter:
def transfer_function():
transfer_function = filter_bank.tf(show=False)[1][0:900, :]
if (transfer_function < -30).sum() > 0:
    print("Some of the equalization filters contain deep notches - try adjusting the parameters.")
#
# step 3: ole_test filter bankkk
recordings = []

attenuated = deepcopy(signal)
attenuated = filter_bank.channel(0).apply(attenuated)
attenuated.level += equalization_levels  # which order? doesnt seem to matter much
temp_recs = []
for i in range(rec_repeat):
    rec = freefield.play_and_record(speakers[0], attenuated, equalize=False)
    # rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
    temp_recs.append(rec.data)
recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
dome_rec.extend(recordings)  # collect equalized recordings from the whole dome for final evaluation
recordings = slab.Sound(recordings)

# plot
diff = freefield.spectral_range(recordings, plot=ax[2])
ax[2].set_title('frequency equalized')
fig.suptitle('Calibration for dome speaker column at %.1f° azimuth. \n Difference in power spectrum' % speakers[0].azimuth, fontsize=16)

# save equalization
array_equalization = {f"{speakers[i].index}": {"level": equalization_levels[i], "filter": filter_bank.channel(i)}
                for i in range(len(speakers))}
equalization.update(array_equalization)


# ----------  repeat for next speaker column ----------- #


# write final equalization to pkl file
# freefield_path = freefield.DIR / 'data'
project_path = Path.cwd() / 'data' / 'calibration' # todo varvara get your own path to your project
equalization_path = project_path / f'calibration_dome_100k_31.10'
with open(equalization_path, 'wb') as f:  # save the newly recorded calibration
    pickle.dump(equalization, f, pickle.HIGHEST_PROTOCOL)


#todo varvara: in your experiment, load the pickle file (equalization dict), and apply the level and filter to the signal (wav files)
# todo this:
#     pick your speaker from the dict, and get level and filter_bank
#     attenuated = deepcopy(signal)  <- this is your .wav
#     attenuated = filter_bank.apply(attenuated)  <- choose the right speaker
#     attenuated.level += level  # which order? doesnt seem to matter much


# check spectral difference across dome
# dome_recs = slab.Sound(dome_rec)
# diff = freefield.spectral_range(dome_recs)


# ------------------  test calibration  ---------------------#

import freefield
import slab
import numpy
import time
from pathlib import Path
freefield.initialize('dome', default='play_rec')  # initialize setup
# proc_list = [['RP2', 'RP2', Path.cwd() / 'data' / 'rcx' / 'bi_rec_buf.rcx'],
#              ['RX81', 'RX8', Path.cwd() / 'data' / 'rcx' / 'play_buf.rcx'],
#              ['RX82', 'RX8', Path.cwd() / 'data' / 'rcx' / 'play_buf.rcx']]
# freefield.initialize('dome', device=proc_list)
# freefield.PROCESSORS.mode = 'play_birec'

# manual testing: play broadband noise through all speakers
freefield.load_equalization(equalization_path)
signal = slab.Sound.pinknoise(duration=1.0, level=85)
speakers = freefield.pick_speakers(speaker_table[:, 0].astype('int'))
# speakers = freefield.pick_speakers(speaker_list[3])  # single column
time.sleep(10)
for speaker in speakers:
    print(speaker.level)
    freefield.set_signal_and_speaker(signal, speaker, equalize=True)
    freefield.play()
    freefield.wait_to_finish_playing()


freefield.set_logger('WARNING')
azimuthal_angles = numpy.array([-52.5, -35, -17.5, 0, 17.5, 35, 52.5])
table_file = freefield.DIR / 'data' / 'tables' / Path(f'speakertable_dome.txt')
speaker_table = numpy.loadtxt(table_file, skiprows=1, usecols=(0, 3, 4), delimiter=",", dtype=float)
speaker_list = []
for az in azimuthal_angles:
    speaker_list.append((speaker_table[speaker_table[:, 1] == az][:, 0]).astype('int'))
speaker_list[3] = numpy.delete(speaker_list[3], [numpy.where(speaker_list[3] == 19), numpy.where(speaker_list[3] == 27)])
# pick a column to ole_test calibration for
# signal parameters

fs = 48828
fs = 97656  # 97656.25, 195312.5
slab.set_default_samplerate(fs)
low_cutoff = 1000
high_cutoff = 17000
signal_length = 0.1  # how long should the chirp be?
rec_repeat = 30  # how often to repeat measurement for averaging
# signal for loudspeaker calibration
ramp_duration = signal_length/20
signal = slab.Sound.chirp(duration=signal_length, level=80, from_frequency=low_cutoff, to_frequency=high_cutoff,
                          kind='linear', samplerate=fs)
signal = slab.Sound.ramp(signal, when='both', duration=ramp_duration)
# freefield.load_equalization(file=Path.cwd() / 'data' / 'calibration' / 'calibration_central_cone_100k')

# measure spectral range across speakers of the selected column

speakers = freefield.pick_speakers(speaker_list[3])
recordings = []
for speaker in speakers:
    temp_recs = []  # <------------- (-.-) 2h bugfix
    for i in range(rec_repeat):
        rec = freefield.play_and_record(speaker, signal, equalize=False)
        # rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data)
    recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
recordings = slab.Sound(recordings)
# plot
diff = freefield.spectral_range(recordings)




noise = slab.Sound.pinknoise(duration=0.025, level=90)
noise = noise.ramp(when='both', duration=0.01)
silence = slab.Sound.silence(duration=0.025)
stim = slab.Sound.sequence(noise, silence, noise, silence, noise,
                           silence, noise, silence, noise)
stim = stim.ramp(when='both', duration=0.01)
for s in speaker_list:
    speakers = freefield.pick_speakers(s)
    for speaker in speakers:
        # for i in range(5):
        #     rec = freefield.play_and_record(speaker, noise, equalize=True)
        #     freefield.wait_to_finish_playing()
        freefield.set_signal_and_speaker(signal=stim, speaker=speaker, equalize=True)
        for i in range(5):
            freefield.play()
            freefield.wait_to_finish_playing()

#------ OPTIONAL -----#
# step 4: adjust level after freq equalization: (?) -- sometimes worth doing!
# level_threshold = 0.3  # correct level only for speakers that deviate more than <threshold> dB from reference speaker
# recordings.data[:, numpy.logical_and(recordings.level > target.level-level_threshold,
#                                      recordings.level < target.level+level_threshold)] = target.data
# final_equalization_levels = equalization_levels + (target.level - recordings.level)
# recordings = []
# for idx, (speaker, level) in enumerate(zip(speakers, final_equalization_levels)):
#     attenuated = deepcopy(signal)
#     attenuated = filter_bank.channel(idx).apply(attenuated)
#     attenuated.level += level  # which order? doesnt seem to matter much
#     temp_recs = []
#     for i in range(rec_repeat):
#         rec = freefield.play_and_record(speaker, attenuated, equalize=False)
#         rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
#         temp_recs.append(rec.data)
#     recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
# recordings = slab.Sound(recordings)
# # plot
# diff = freefield.spectral_range(recordings, plot=ax[3])
# ax[3].set_title('final level correction')


# load existing equalization pkl
from pathlib import Path
import pickle
project_path = Path.cwd() / 'data' / 'calibration'
file_name = project_path / f'central_arc_calibration_100k'
with open(file_name, "rb") as f:
    equalization = pickle.load(f)
# check spectral difference across dome
dome_recs = slab.Sound(dome_rec)
diff = freefield.spectral_range(dome_recs)

# - on human listener
time.sleep(10)
for speaker_id in speaker_ids:
    freefield.set_signal_and_speaker(signal, speaker_id, equalize=True)
    freefield.play()
    freefield.wait_to_finish_playing()

# - spectral range
recordings = []
for speaker_id in speaker_ids:
    temp_recs = []
    for i in range(20):
        rec = freefield.play_and_record(speaker_id, sound, equalize=False)
        rec = slab.Sound.ramp(rec, when='offset', duration=0.01)
        temp_recs.append(rec.data)
    recordings.append(slab.Sound(data=numpy.mean(temp_recs, axis=0)))
recordings = slab.Sound(recordings)
diff2 = freefield.spectral_range(recordings)


# build in ole_test function
raw, level, spectrum = freefield.test_equalization()
diff_raw = freefield.spectral_range(raw)
diff_level = freefield.spectral_range(level)
diff_spectrum = freefield.spectral_range(spectrum)

"""
### extra: arrange dome ####
import numpy as numpy
radius = 1.4 # meter
az_angles = numpy.radians((0, 17.5, 35, 52.5))
ele_angles = numpy.radians((12.5, 25, 37.5, 50))
# horizontal_dist = numpy.cos((numpy.pi / 2) - az_angles) * radius
horizontal_dist = numpy.sin(az_angles) * radius

# this would be the correct vertical distances for interaural speaker locations;
radii = numpy.sin(numpy.pi / 2 - az_angles) * radius
vertical_dist = []
for elevation in ele_angles:
    vertical_dist.append(numpy.sin(elevation) * radii)
vertical_dist = numpy.asarray(vertical_dist)

# but we are using these for simplicity (just think of head tracking based training):
vertical_dist = numpy.sin(ele_angles) * radius

vert_abs = []
for i in range(len(vertical_dist)):
    vert_abs.append(0.22 + vertical_dist[i])
    
    

"""