import freefield
from pathlib import Path
import numpy
import random
import os
import slab

proc_list = [['RX81', 'RX8', Path.cwd() / 'experiment.rcx'],
             ['RX82', 'RX8', Path.cwd() / 'experiment.rcx']]
freefield.set_logger('info')

# path:
data_path = Path.cwd() / 'data' / 'voices_padded'

# params:
stim_dur_ms = 745  # duration in ms
isi = numpy.array((240, 180))
n_trials = 100
tlo = stim_dur_ms + isi[1]
numbers = [1, 2, 3, 4, 5, 6, 8, 9]
participant_id = ''
chosen_voice_path = Path.cwd() / 'data' / 'chosen_voice'

def select_voice():
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
    name_mapping = {0: 'Matilda',
                    1: 'Johanna',
                    2: 'Carsten',
                    3: 'Marc'}
    chosen_voice_name = name_mapping[wav_files_lists.index(chosen_voice)]
    chosen_voice_file = chosen_voice_path / f'training_{participant_id}_{chosen_voice_name}.txt'
    with open(chosen_voice_file, 'w') as file:
        file.write(str(chosen_voice))

    return chosen_voice


def write_buffer(chosen_voice):  # write voice data onto rcx buffer
    for number, file_path in zip(numbers, chosen_voice):
        # combine lists into a single iterable
        # elements from corresponding positions are paired together
        if os.path.exists(file_path):
            s = slab.Sound(data=file_path)
            freefield.write(f'{number}', s.data, ['RX81', 'RX82'])  # loads array on buffer
            freefield.write(f'{number}_n_samples', s.n_samples, ['RX81', 'RX82'])


# get trial_sequence
def get_trial_seq(n_trials, numbers, tlo):

    trial_seq = []
    random.shuffle(numbers)
    for i in range(n_trials):
        while len(trial_seq) > 0 and trial_seq[-1] == numbers[0]:
            random.shuffle(numbers)

        trial_seq.append(numbers[0])
        numbers.pop(0)
        if len(numbers) == 0:
            numbers = [1, 2, 3, 4, 5, 6, 8, 9]
            random.shuffle(numbers)
    sequence = numpy.array(trial_seq).astype('int32')
    sequence = numpy.append(0, sequence)
    return sequence


def play(sequence):
    speakers_coordinates = (17.5, 0)
    freefield.write('trial_seq1', sequence, ['RX81', 'RX82'])
    freefield.write('n_trials1', n_trials + 1, ['RX81', 'RX82'])
    freefield.write('tlo1', tlo, ['RX81', 'RX82'])
    [speaker] = freefield.pick_speakers((speakers_coordinates[0], 0))  # speaker 31, 17.5 az, 0.0 ele
    # or speaker 23, 0.0 az, 0.0 ele
    freefield.write('channel1', speaker.analog_channel, speaker.analog_proc)
    # [speaker] = freefield.pick_speakers((speakers_coordinates[1], -37.5))
    # [speaker] = freefield.pick_speakers((speakers_coordinates[1], 37.5))
    freefield.play()


if __name__ == "__main__":
    freefield.initialize('dome', device=proc_list)
    chosen_voice = select_voice()
    write_buffer(chosen_voice)
    sequence = get_trial_seq(n_trials, numbers, tlo)
    play(sequence)





