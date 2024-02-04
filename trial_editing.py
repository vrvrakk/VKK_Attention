import slab
import numpy
import os
import random
import freefield
from pathlib import Path
# import matplotlib.pyplot as plt

# Dai & Shinn-Cunningham (2018):
n_blocks = 10
n_trials1 = 56
isi = (664, 758)
# choose speakers:
speakers_coordinates = (-17.5, 17.5, 0)  # directions for each streams
s2_delay = 2000
sample_freq = 24414
numbers = [1, 2, 3, 4, 5, 6, 8, 9]
data_path = Path.cwd() / 'data' / 'voices'

proc_list = [['RX81', 'RX8', Path.cwd() / 'experiment.rcx'],
             ['RX82', 'RX8', Path.cwd() / 'experiment.rcx']]

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
n_samples = []
for wav_files_folder in wav_files_lists:
    for wav_file_path in wav_files_folder:
        if os.path.exists(wav_file_path):
            s = slab.Sound(data=wav_file_path)
            s = s.resample(sample_freq)
            n_samples.append(s.n_samples)

max_sample = int(numpy.max(n_samples))
# Equalize durations
equalized_sounds = []
altered_durations = []
for wav_files_folder in wav_files_lists:
    for wav_file_path in wav_files_folder:
        if wav_file_path.exists():
            s = slab.Sound(data=wav_file_path)
            s = s.resample(sample_freq)  # again..?
            if s.n_samples < max_sample:
                # Calculate the number of samples to add
                samples_to_add = max_sample - s.n_samples
                # Determine the number of channels in the sound
                n_channels = s.data.shape[1]
                # Create an array of zeros with the correct shape
                padding = numpy.zeros((samples_to_add, n_channels), dtype=s.data.dtype)
                # Append the zeros to the end of the sound data
                s.data = numpy.concatenate((s.data, padding), axis=0)

                # Track the file path and the amount of padding added
                altered_durations.append((wav_file_path, samples_to_add))

            equalized_sounds.append(s)
for item in altered_durations:
    print(f"File: {item[0]}, Padding Added: {item[1]} Samples")


for wav_files_folder in wav_files_lists:
    chosen_voice = random.choice(wav_files_lists)

n_samples_ms = []
for number, file_path in zip(numbers, chosen_voice):
    #freefield.write(f'{number}', s.data, ['RX81', 'RX82'])  # loads array on buffer
    #freefield.write(f'{number}_n_samples', s.n_samples,
    #                           ['RX81', 'RX82'])  # sets total buffer size according to numeration
    sound_duration_ms = int((s.n_samples / 24414) * 1000) # divide by 25k?
    n_samples_ms.append(sound_duration_ms)  # get list with duration of each sound event in ms
    n_samples_ms = list(n_samples_ms)

mean_n_samples = int(numpy.mean(n_samples_ms))
tlo1 = int(isi[0] + mean_n_samples)  # isi + (mean sample size of sound event / sample freq)
tlo2 = int(isi[1] + mean_n_samples)
t_end = n_trials1 * tlo1
trial_seq1 = slab.Trialsequence(conditions=numbers, n_reps=n_trials1 / len(numbers), kind='non_repeating')
for i in range(len(trial_seq1.trials)):
    if trial_seq1.trials[i] == 7:  # replace 7 with 9 in trial_seq.trials
        trial_seq1.trials[i] = 9

    n_trials2 = int(numpy.ceil((t_end - s2_delay) / tlo2))
trial_seq2 = slab.Trialsequence(conditions=numbers, n_reps=n_trials2 / len(numbers), kind='non_repeating')
# trial_seq2.trials = trial_seq2.trials[:-3]
for i in range(len(trial_seq2.trials)):
    if trial_seq2.trials[i] == 7:
        trial_seq2.trials[i] = 9

# get list with dur of each trial:
n_samples_ms_dict = dict(zip(numbers, n_samples_ms))

trials_dur1 = []
for index1, trial1 in enumerate(trial_seq1.trials):
    # print(index1, trial1)
    duration1 = n_samples_ms_dict.get(trial1)
    # print(duration1) # get dur of each trial in ms
    t1_onset = index1 * tlo1  # trial1 onsets
    t1_offset = t1_onset + tlo1
    if duration1 is not None:
        trials_dur1.append((index1, trial1, t1_onset, duration1, t1_offset))

# now for trial_seq2.trials:
trials_dur2 = []
for index2, trial2 in enumerate(trial_seq2.trials):
    duration2 = n_samples_ms_dict.get(trial2)
    t2_onset = (index2 * tlo2) + s2_delay
    t2_offset = t2_onset + tlo2
    if duration2 is not None:
        trials_dur2.append((index2, trial2, t2_onset, duration2, t2_offset))


overlapping_trials = []
for index2, trial2, t2_onset, duration2, t2_offset in trials_dur2:
    for index1, trial1, t1_onset, duration1, t1_offset in trials_dur1:
        # Check if the trials overlap in time and have the same number
        if not (t2_offset <= t1_onset or t2_onset >= t1_offset) and trial1 == trial2:
            overlapping_trials.append((index1, trial1, index2, trial2))
print(overlapping_trials)

# essentially: iterates through trials_dur 1 and 2-> if s1 sound event
for index1, trial1, index2, trial2 in overlapping_trials:  # TODO: something is OFF again
    prev_trial1 = trials_dur1[index1 - 1][1] if index1 > 0 else None  # index [1] entails the trial number
    next_trial1 = trials_dur1[index1 + 1][1] if index1 < len(trials_dur1) - 1 else None
    prev_trial2 = trials_dur2[index2 - 1][1] if index2 > 0 else None
    next_trial2 = trials_dur2[index2 + 1][1] if index2 < len(trials_dur2) - 1 else None
    # Find a replacement number for trial2
    exclude_numbers = {trial1, prev_trial1, next_trial1, prev_trial2, next_trial2}
    exclude_numbers.discard(None)  # Remove None values if they exist
    possible_numbers = [n for n in numbers if n not in exclude_numbers]
    if possible_numbers:
        new_number = random.choice(possible_numbers)
        trial_seq2.trials[index2] = new_number