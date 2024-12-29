import freefield
from pathlib import Path
import numpy
import random
import os
import slab
proc_list = [['RX81', 'RX8', Path.cwd() / 'experiment_jitter.rcx'],
             ['RX82', 'RX8', Path.cwd() / 'experiment_jitter.rcx']]
freefield.set_logger('info')
freefield.initialize('dome', device=proc_list)

# path:
data_path = Path.cwd() / 'data' / 'voices_english'
# params:
stim_dur_ms = 745  # duration in ms
isi = numpy.array((90, 70))
n_trials = 70
tlo1 = stim_dur_ms + isi[0]
tlo2 = stim_dur_ms + isi[1]
numbers = [1, 2, 3, 4, 5, 6, 8, 9]

voice_idx = list(range(1, 5))
folder_paths = []
wav_folders = [folder for folder in os.listdir(data_path)]
for i, folder in zip(voice_idx, wav_folders):
    folder_path = data_path / folder
    folder_paths.append(folder_path)  # absolute path of each voice folder
# Initialize the corresponding wav_files list
wav_files_lists = []
for i, folder_path in zip(voice_idx, folder_paths):
    wav_files_in_folder = list(folder_path.glob("*.wav"))
    wav_files_lists.append(wav_files_in_folder)

chosen_voice = wav_files_lists[1]  # select a voice folder

for number, file_path in zip(numbers, chosen_voice):
    # combine lists into a single iterable
    # elements from corresponding positions are paired together
    if os.path.exists(file_path):
        s = slab.Sound(data=file_path)
        freefield.write(f'{number}', s.data, ['RX81', 'RX82'])  # loads array on buffer
        freefield.write(f'{number}_n_samples1', s.n_samples, ['RX81', 'RX82'])
        freefield.write(f'{number}_n_samples2', s.n_samples, ['RX81', 'RX82'])

# get trial_sequences
speakers_coordinates = (17.5, -17.5)
trial_seq1 = []
trial_seq2 = []
random.shuffle(numbers)
for i in range(n_trials):
    while len(trial_seq1) > 0 and trial_seq1[-1] == numbers[0]:
        random.shuffle(numbers)
    trial_seq1.append(numbers[0])
    numbers.pop(0)
    if len(numbers) == 0:
        numbers = [1, 2, 3, 4, 5, 6, 8, 9]
        random.shuffle(numbers)
sequence1 = numpy.array(trial_seq1).astype('int32')
sequence1 = numpy.append(0, sequence1)
freefield.write('trial_seq1', sequence1, ['RX81', 'RX82'])
freefield.write('n_trials1', n_trials + 1, ['RX81', 'RX82'])
freefield.write('tlo1', tlo1, ['RX81', 'RX82'])
[speaker1] = freefield.pick_speakers((speakers_coordinates[0], 0))
freefield.write('channel1', speaker1.analog_channel, speaker1.analog_proc)
proc_list = ['RX81', 'RX82']
proc_list.remove(speaker1.analog_proc)
freefield.write('channel1', 25, proc_list)
numbers2 = [1, 2, 3, 4, 5, 6, 8, 9]
random.shuffle(numbers2)
for i in range(n_trials):
    while len(trial_seq2) > 0 and trial_seq2[-1] == numbers2[0]:
        random.shuffle(numbers2)
    trial_seq2.append(numbers2[0])
    numbers2.pop(0)
    if len(numbers2) == 0:
        numbers2 = [1, 2, 3, 4, 5, 6, 8, 9]
        random.shuffle(numbers2)
sequence2 = numpy.array(trial_seq2).astype('int32')
sequence2 = numpy.append(0, sequence2)
freefield.write('trial_seq2', sequence2, ['RX81', 'RX82'])
freefield.write('n_trials2', n_trials + 1, ['RX81', 'RX82'])
freefield.write('tlo2', tlo2, ['RX81', 'RX82'])
freefield.write('s2_delay', tlo1 * 3, ['RX81', 'RX82'])
[speaker2] = freefield.pick_speakers((speakers_coordinates[1], 0))
freefield.write('channel2', speaker2.analog_channel, speaker2.analog_proc)
proc_list = ['RX81', 'RX82']
proc_list.remove(speaker2.analog_proc)

# [speaker] = freefield.pick_speakers((speakers_coordinates[1], -37.5))
# [speaker] = freefield.pick_speakers((speakers_coordinates[1], 37.5))
freefield.play()
freefield.halt()
