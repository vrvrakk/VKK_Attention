import slab
import numpy
import os
import random
import freefield
from pathlib import Path
import matplotlib.pyplot as plt

participant_id = 'vk'
# Dai & Shinn-Cunningham (2018):
n_blocks = 10
n_trials1 = 56
isi = (664, 758)
# choose speakers:
speakers = (-17.5, 17.5)  # directions for each streams
s2_delay = 2000
sample_freq = 24414
numbers = [1, 2, 3, 4, 5, 6, 8, 9]
data_path = Path.cwd() / 'data' / 'voices'
responses_dir = Path.cwd() / 'data' / 'results'
proc_list = [['RX81', 'RX8', Path.cwd() / 'experiment.rcx'],
             ['RX82', 'RX8', Path.cwd() / 'experiment.rcx'],
             ['RP2', 'RP2', Path.cwd() / '9_buttons.rcx']]


def wav_list_select(data_path):  # create wav_list paths, and select a voice folder randomly
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
        chosen_voice = random.choice(wav_files_lists)
    return chosen_voice


def write_buffer(chosen_voice):  # write voice data onto rcx buffer
    n_samples = []
    n_samples_ms = []
    for number, file_path in zip(numbers,
                                 chosen_voice):  # combine lists into a single iterable->elements from corresponding positions are paired together
        if os.path.exists(file_path):
            s = slab.Sound(data=file_path)
            s = s.resample(sample_freq)
            n_samples.append(s.n_samples)
            freefield.write(f'{number}', s.data, ['RX81', 'RX82'])  # loads array on buffer
            freefield.write(f'{number}_n_samples', s.n_samples, ['RX81', 'RX82'])  # sets total buffer size according to numeration
            sound_duration_ms = int((s.n_samples / 50000) * 1000)
            n_samples_ms.append(sound_duration_ms)  # get list with duration of each sound event in ms
    # n_samples_ms = zip(numbers, n_samples_ms)
    n_samples_ms = list(n_samples_ms)
    return n_samples_ms


def get_trial_sequence(n_trials1, n_samples_ms):
    # get trial duration for both streams plus n_trials of lagging stream
    mean_n_samples = int(numpy.mean(n_samples_ms))  # mean duration of sound events in ms
    tlo1 = int(isi[0] + (mean_n_samples))  # isi + (mean sample size of sound event / sample freq)
    tlo2 = int(isi[1] + (mean_n_samples))
    t_end = n_trials1 * tlo1  # total length of BOTH streams
    n_trials2 = int(numpy.ceil((t_end - s2_delay) / tlo2))  # to estimate the total amount of s2 trials, for both streams to end simultaneously

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

    # make sure both streams have different numbers at concurrent trials:
    overlapping_trials = []
    for index2, trial2, t2_onset, duration2, t2_offset in trials_dur2:
        for index1, trial1, t1_onset, duration1, t1_offset in trials_dur1:
            if t2_onset < t1_offset and t2_offset > t1_onset and trial1 == trial2:
                overlapping_trials.append((index1, trial1, index2, trial2))
    print(overlapping_trials)

    for index1, trial1, index2, trial2 in overlapping_trials:
        prev_trial1 = trials_dur1[index1 - 1][1] if index1 > 0 else None  # index [1] entails the trial number
        next_trial1 = trials_dur1[index1 + 1][1] if index1 < len(trials_dur1) - 1 else None
        prev_trial2 = trials_dur2[index2 - 1][1] if index2 > 0 else None
        next_trial2 = trials_dur2[index2 + 1][1] if index2 < len(trials_dur2) - 2 else None
        # Find a replacement number for trial2
        exclude_numbers = {trial1, prev_trial1, next_trial1, prev_trial2, next_trial2} #TODO: check if prev and next trials_2 are actually correct
        exclude_numbers.discard(None)  # Remove None values if they exist
        possible_numbers = [n for n in numbers if n not in exclude_numbers]
        if possible_numbers:
            new_number = random.choice(possible_numbers)
            trial_seq2.trials[index2] = new_number

    return trial_seq1, trial_seq2, tlo1, tlo2


def run_block(trial_seq1, trial_seq2, tlo1, tlo2, speakers):
    [speaker1] = freefield.pick_speakers((speakers[0], 0))  # set speakers to play
    [speaker2] = freefield.pick_speakers((speakers[1], 0))
    sequence1 = numpy.array(trial_seq1.trials).astype('int32')
    sequence1 = numpy.append(0, sequence1)
    sequence2 = numpy.array(trial_seq2.trials).astype('int32')
    sequence2 = numpy.append(0, sequence2)
    # isi, s2_delay and n_trials only for calculating stuff->i.e. tlo
    # here we set tlo to RX8
    freefield.write('tlo1', tlo1, ['RX81', 'RX82'])
    freefield.write('tlo2', tlo2, ['RX81', 'RX82'])
    # set n_trials to pulse trains sheet0/sheet1
    freefield.write('n_trials1', trial_seq1.n_trials + 1,
                    ['RX81', 'RX82'])  # analog_proc attribute from speaker table dom txt file
    freefield.write('n_trials2', trial_seq2.n_trials + 1, ['RX81', 'RX82'])
    freefield.write('trial_seq1', sequence1, ['RX81', 'RX82'])
    freefield.write('trial_seq2', sequence2, ['RX81', 'RX82'])
    # set output speakers for both streams
    freefield.write('channel1', speaker1.analog_channel, speaker1.analog_proc)
    freefield.write('channel2', speaker2.analog_channel, speaker2.analog_proc)
    # convert sequence numbers to integers, add a 0 at the beginning and write to trial sequence buffers
    freefield.play()

def run_experiment(n_blocks, n_trials1, speakers):
    completed_blocks = 0  # initial number of completed blocks
    for block in range(n_blocks):  # iterate over the number of blocks
        chosen_voice = wav_list_select(data_path)
        n_samples_ms = write_buffer(chosen_voice)
        trial_seq1, trial_seq2, tlo1, tlo2 = get_trial_sequence(n_trials1, n_samples_ms)

        run_block(trial_seq1, trial_seq2, tlo1, tlo2, speakers)
        # Wait for user input to continue to the next block
        user_input = input('Press "1" to continue to the next block, or any other key to stop: ')
        if user_input != '1':
            print("Experiment stopped early by user.")
            break
        completed_blocks += 1  # Increment the count of completed blocks
    print(f"Experiment completed {completed_blocks} out of {n_blocks} blocks.")
    return completed_blocks


if __name__ == "__main__":
    freefield.initialize('dome', device=proc_list)

    run_experiment(n_blocks, n_trials1, speakers)


''' # PLOTTING TRIAL SEQUENCES OVER TIME
# Extracting trial numbers and their onsets from trials_dur1 and trials_dur2
trials_1 = [trial1[1] for trial1 in trials_dur1]  # Trial numbers for Stream 1
onsets_1 = [t1_onset[2] for t1_onset in trials_dur1]  # Onsets for Stream 1

trials_2 = [trial2[1] for trial2 in trials_dur2]  # Trial numbers for Stream 2
onsets_2 = [t2_onset[2] for t2_onset in trials_dur2]  # Onsets for Stream 2

# Create the plot
plt.figure(figsize=(15, 8))

# Plotting the trials with their onsets for both streams with lines
plt.plot(onsets_1, trials_1, label='Stream 1', marker='o', linestyle='-', alpha=0.7)
plt.plot(onsets_2, trials_2, label='Stream 2', marker='x', linestyle='-', alpha=0.7)

# Adding labels, title, and legend
plt.xlabel('Time Onset (ms)')
plt.ylabel('Trial Number')
plt.title('Trial Numbers Over Time for Stream 1 and Stream 2')
plt.legend()

# set the limits for better visibility if needed
plt.xlim(min(onsets_1 + onsets_2), max(onsets_1 + onsets_2))
plt.ylim(min(trials_1 + trials_2) - 1, max(trials_1 + trials_2) + 1)  # Adjusted for better y-axis visibility

# Show grid
plt.grid(True)

# Show the plot
plt.show()
'''