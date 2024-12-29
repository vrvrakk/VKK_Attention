'''
freefield.play_and_record(
    speaker,
    sound,
    compensate_delay=True,
    compensate_attenuation=False,
    equalize=True,
    recording_samplerate=48828,
)
Docstring:
Play the signal from a speaker and return the recording. Delay compensation
means making the buffer of the recording processor n samples longer and then
throwing the first n samples away when returning the recording so sig and
rec still have the same length. For this to work, the circuits rec_buf.rcx
and play_buf.rcx have to be initialized on RP2 and RX8s and the mic must
be plugged in.
Parameters:
    speaker: integer between 1 and 48, index number of the speaker
    sound: instance of slab.Sound, signal that is played from the speaker
    compensate_delay: bool, compensate the delay between play and record
    compensate_attenuation:
    equalize:
    recording_samplerate: samplerate of the recording
Returns:
    rec: 1-D array, recorded signal
'''


import freefield
from pathlib import Path
import slab
import pickle as pkl

freefield.initialize(setup='dome', default='play_rec')

sounds_path = Path('C:/projects/VKK_Attention/data/voices_english')



equalization_path = Path.cwd() / 'data' / 'misc' / 'calibration' / 'calibration_speakers.pkl'
with open(equalization_path, 'rb') as file:
    calibration_azimuth = pkl.load(file)

calibration_left = None
calibration_right = None

# Iterate over the dictionary and extract the first two sub-dictionaries
for index, (key, values) in enumerate(calibration_azimuth.items()):
    if index == 0:
        calibration_left = values  # Save the first sub-dictionary
    elif index == 1:
        calibration_right = values  # Save the second sub-dictionary
        break  # Stop after saving the first two sub-dictionaries
filter_left = calibration_left["filter"]
filter_right = calibration_right["filter"]

filename_list = []
wav_path_list = []
for sound_folder in sounds_path.iterdir():
    for sound_files in sound_folder.iterdir():
        filename = sound_files.name
        filename_list.append(filename)
        wav_path_list.append(sound_files.as_posix()) # cool
sounds_list = []
for sound_files in wav_path_list:
    sound = slab.Sound(sound_files)
    sounds_list.append(sound)

axes = ['azimuth', 'elevation']
azimuth_speaker_coordinates = (-17.5, 17.5)
elevation_speaker_coordinates = (-37.5, 37.5)
# azimuth coordinates
azimuth_left = freefield.pick_speakers((azimuth_speaker_coordinates[0], 0))
azimuth_right = freefield.pick_speakers((azimuth_speaker_coordinates[1], 0))
# elevation coordinates
elevation_bottom = freefield.pick_speakers((0, elevation_speaker_coordinates[0]))
elevation_top = freefield.pick_speakers((0, elevation_speaker_coordinates[1]))

def speaker_filters(filter_left, filter_right, speaker, axis=''):
    if axis == 'elevation':
        filter = speaker.filter
        filter.samplerate = 24414
    elif axis == 'azimuth_left':
        filter = filter_left
        filter.samplerate = 24414
    elif axis == 'azimuth_right':
        filter = filter_right
        filter.samplerate = 24414
    return filter


# todo: changed sfreq in the rcx files from 48k to 24k -> change back to default once done
# equalize sound
def equalize_sound(filter):
    filtered_sound_list = []
    for sound in sounds_list:
        sound.level = 80
        filt_sound = filter.apply(sound)
        filtered_sound_list.append(filt_sound)
    return filtered_sound_list


# todo: play and record all 32 sound files
def play_and_save_sounds(filtered_sound_list, speaker):
    recorded_sound_list = []
    for sound_filtered in filtered_sound_list:
        recorded_sound = freefield.play_and_record(speaker=speaker, sound=sound_filtered, compensate_delay=True, compensate_attenuation=False, equalize=False, recording_samplerate=24414)
        recorded_sound_list.append(recorded_sound)
    return recorded_sound_list

# apply correct filter depending on the speaker coordinates used:
axis = input('Please specify axis before proceeding (azimuth_left/azimuth_right, elevation: ')

elevation_bottom_filter = speaker_filters(filter_left, filter_right, elevation_bottom[0], axis='elevation')
elevation_top_filter = speaker_filters(filter_left, filter_right, elevation_top[0], axis='elevation')

# sounds sound normal when played with slab
filtered_sound_list_elevation_bottom = equalize_sound(elevation_bottom_filter)
filtered_sound_list_elevation_top = equalize_sound(elevation_top_filter)

recorded_sound_list_elevation_bottom = play_and_save_sounds(filtered_sound_list_elevation_bottom, elevation_bottom[0])
recorded_sound_list_elevation_top = play_and_save_sounds(filtered_sound_list_elevation_top, elevation_top[0])

# azimuth:
axis = input('Please specify axis before proceeding (azimuth_left/azimuth_right, elevation: ')

azimuth_left_filter = speaker_filters(filter_left, filter_right, azimuth_left[0], axis='azimuth_left')
azimuth_right_filter = speaker_filters(filter_left, filter_right, azimuth_right[0], axis='azimuth_right')

filtered_sound_list_azimuth_left = equalize_sound(azimuth_left_filter)
filtered_sound_list_azimuth_right = equalize_sound(azimuth_right_filter)

recorded_sound_list_azimuth_left = play_and_save_sounds(filtered_sound_list_azimuth_left, azimuth_left[0])
recorded_sound_list_azimuth_right = play_and_save_sounds(filtered_sound_list_azimuth_right, azimuth_right[0])


