import slab
import numpy
import os
import random
import freefield
from pathlib import Path

n_blocks = 3  # total of 18 minutes per axis
n_trials1 = 200  # 200  # 4.95 min total
n_trials2 = 232  # 232  # 4.98 min total
# Dai & Shinn-Cunningham (2018):
isi = (741, 543)  # so that tlo1 = 1486, tlo2 = 1288
# choose speakers:
speakers_coordinates = (-17.5, 17.5, 0)  # directions for each streams
# s2_delay = 2000 # delay not necessary if target always on the right
sample_freq = 24414
numbers = [1, 2, 3, 4, 5, 6, 8, 9]
data_path = Path.cwd() / 'data' / 'voices_padded'
# n_samples = 18210
# sound_dur_ms = int((n_samples / 24414) * 1000)
proc_list = [['RX81', 'RX8', Path.cwd() / 'experiment.rcx'],
             ['RX82', 'RX8', Path.cwd() / 'experiment.rcx']]


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
    n_samples = 18210
    sound_dur_ms = int((n_samples / 24414) * 1000)
    n_samples_ms = []
    for number, file_path in zip(numbers, chosen_voice):
        # combine lists into a single iterable
        # elements from corresponding positions are paired together
        if os.path.exists(file_path):
            s = slab.Sound(data=file_path)
            n_samples_ms.append(sound_dur_ms)
            freefield.write(f'{number}', s.data, ['RX81', 'RX82'])  # loads array on buffer
            freefield.write(f'{number}_n_samples', s.n_samples,
                            ['RX81', 'RX82'])  # sets total buffer size according to numeration
            n_samples_ms = list(n_samples_ms)
    return n_samples_ms, sound_dur_ms


def get_trial_sequence(sound_dur_ms, n_samples_ms):
    # get trial duration for both streams plus n_trials of lagging stream
    tlo1 = int(isi[0] + sound_dur_ms)  # isi + (mean sample size of sound event / sample freq)
    tlo2 = int(isi[1] + sound_dur_ms)
    # here we set trial sequence
    trial_seq1 = slab.Trialsequence(conditions=numbers, n_reps=n_trials1 / len(numbers), kind='non_repeating')
    for i in range(len(trial_seq1.trials)):
        if trial_seq1.trials[i] == 7:  # replace 7 with 9 in trial_seq.trials
            trial_seq1.trials[i] = 9

    trial_seq2 = slab.Trialsequence(conditions=numbers, n_reps=n_trials2 / len(numbers), kind='non_repeating')
    for i in range(len(trial_seq2.trials)):
        if trial_seq2.trials[i] == 7:
            trial_seq2.trials[i] = 9
    n_samples_ms_dict = dict(zip(numbers, n_samples_ms))

    return trial_seq1, trial_seq2, n_samples_ms_dict, tlo1, tlo2


def trials_durations(trial_seq1, trial_seq2, tlo1, tlo2, n_samples_ms_dict):
    trials_dur1 = []
    for index1, trial1 in enumerate(trial_seq1.trials):
        duration1 = n_samples_ms_dict.get(trial1)
        # print(duration1)
        t1_onset = index1 * tlo1  # trial1 onsets
        t1_offset = t1_onset + tlo1
        if duration1 is not None:
            trials_dur1.append((index1, trial1, t1_onset, duration1, t1_offset))

    # now for trial_seq2.trials:
    trials_dur2 = []
    for index2, trial2 in enumerate(trial_seq2.trials):
        duration2 = n_samples_ms_dict.get(trial2)
        t2_onset = index2 * tlo2
        t2_offset = t2_onset + tlo2
        if duration2 is not None:
            trials_dur2.append((index2, trial2, t2_onset, duration2, t2_offset))
    return trials_dur1, trials_dur2


def categorize_conflicting_trials(trials_dur1, trials_dur2):
    # now creating the rolling window:
    equal_trials = []
    previous_trials = []
    next_trials = []

    # Populate the collections based on conditions
    for index1, trial1, t1_onset, _, t1_offset in trials_dur1:
        for index2, trial2, t2_onset, _, t2_offset in trials_dur2:
            if t1_onset <= t2_offset and t1_offset >= t2_onset:
                if trial1 == trial2:
                    equal_trials.append((index1, trial1, index2, trial2))
                if index2 > 0 and trial1 == trials_dur2[index2 - 1][1]:
                    previous_trials.append((index1, trial1, index2 - 1, trials_dur2[index2 - 1][1]))
                if index2 < len(trials_dur2) - 1 and trial1 == trials_dur2[index2 + 1][1]:
                    next_trials.append((index1, trial1, index2 + 1, trials_dur2[index2 + 1][1]))
    return equal_trials, previous_trials, next_trials


def get_conflict_lists(equal_trials, previous_trials, next_trials):
    # Containers for conflicts with additional information
    prev_equal_overlaps = []
    next_equal_overlaps = []
    previous_trials_conflicts = []
    next_trials_conflicts = []
    # Process previous trials conflicts
    for pt in previous_trials:
        if pt in equal_trials:
            prev_equal_overlaps.append(pt)
        else:
            previous_trials_conflicts.append(pt + ('previous trial2',))

    # Process next trials conflicts
    for nt in next_trials:
        if nt in equal_trials:
            next_equal_overlaps.append(nt)
        else:
            next_trials_conflicts.append(nt + ('next trial2',))
    # Combine previous and next equal overlaps and add 'equal trials' tag to each tuple
    equal_trials_conflicts = [et + ('equal trials',) for et in prev_equal_overlaps + next_equal_overlaps]

    return equal_trials_conflicts, previous_trials_conflicts, next_trials_conflicts


def replace_s2_conflicts(trials_dur2, trial_seq2, equal_trials_conflicts, previous_trials_conflicts,
                         next_trials_conflicts, trials_dur1):
    # replace conflicts for equal trials:
    exclude_numbers = set(numbers)
    for index1, trial1, index2, trial2, tag in equal_trials_conflicts:
        prev_trial1 = trials_dur1[index1 - 1][1] if index1 > 0 else None  # if trial1==trial2, exclude previous trial1
        next_trial1 = trials_dur1[index1 + 1][1] if index1 < len(
            trials_dur1) - 1 else None  # if trial1==trial2, exclude previous trial1
        exclude_numbers = {trial1, prev_trial1, next_trial1}  # remove trial1 from options
        # Add adjacent numbers from s2 to exclude list
        if index2 > 0:  # if trial2 is not in index 0
            exclude_numbers.add(trials_dur2[index2 - 1][1])
        if index2 < len(trials_dur2) - 1:  # if trial2 is not in final position
            exclude_numbers.add(trials_dur2[index2 + 1][1])
        possible_numbers = [n for n in numbers if n not in exclude_numbers]
        if possible_numbers:
            new_number = random.choice(possible_numbers)
            trial_seq2.trials[index2] = new_number
    # replace conflicts for previous s2 trials:
    for index1, trial1, index2, trial2, tag in previous_trials_conflicts:
        prev_trial1 = trials_dur1[index1 - 1][1] if index1 > 0 else None  # if trial1==trial2, exclude previous trial1
        under_prev_trial1 = trials_dur1[index1 - 1][
            1] if index1 > 1 else None  # necessary to avoid repetition of trial2
        exclude_numbers = {trial1, prev_trial1, under_prev_trial1}
        if index2 > 0:
            exclude_numbers.add(trials_dur2[index2 - 1][1])
        if index2 < len(trials_dur2) - 1:  # if trial2 is not in final position
            exclude_numbers.add(trials_dur2[index2 + 1][1])
            possible_numbers = [n for n in numbers if n not in exclude_numbers]
            if possible_numbers:
                new_number = random.choice(possible_numbers)
                trial_seq2.trials[index2] = new_number  # next_trial1 not excluded, as it is not neighboring
    # replace conflicts for next s2 trials:
    for index1, trial1, index2, trial2, tag in next_trials_conflicts:
        next_trial1 = trials_dur1[index1 + 1][1] if index1 < len(trials_dur1) - 1 else None
        over_next_trial1 = trials_dur1[index1 + 2][1] if index1 < len(
            trials_dur1) - 2 else None  # necessary to avoid repetition of trial1
        exclude_numbers = {trial1, next_trial1, over_next_trial1}
        if index2 > 0:
            exclude_numbers.add(trials_dur2[index2 - 1][1])
        if index2 < len(trials_dur2) - 1:  # if trial2 is not in final position
            exclude_numbers.add(trials_dur2[index2 + 1][1])
            possible_numbers = [n for n in numbers if n not in exclude_numbers]
            if possible_numbers:
                new_number = random.choice(possible_numbers)
                trial_seq2.trials[index2] = new_number  # next_trial1 not excluded, as it is not neighboring
    return trial_seq2


def check_updated_s2(trial_seq1, trial_seq2):
    # now check if updated trial_seq2 has repeated trials:
    for index2, trial2 in enumerate(trial_seq2.trials):
        if index2 > 0:  # Check if not the first trial
            prev_trial2 = trial_seq2.trials[index2 - 1]
            # Optionally handle the case where index2 matches the last index in trial_seq1
            prev_trial1 = trial_seq1.trials[index2 - 1] if index2 - 1 < len(trial_seq1.trials) else None
        else:
            prev_trial2 = None
            prev_trial1 = None

        if index2 < len(trial_seq2.trials) - 1:  # Check if not the last trial
            next_trial2 = trial_seq2.trials[index2 + 1]
            # Optionally handle the case where index2 + 1 exceeds the last index in trial_seq1
            next_trial1 = trial_seq1.trials[index2 + 1] if index2 + 1 < len(trial_seq1.trials) else None
        else:
            next_trial2 = None
            next_trial1 = None

        # Define a set of numbers to exclude based on adjacent trials in s1 and s2
        exclude_numbers = set()
        if prev_trial2 is not None:
            exclude_numbers.add(prev_trial2)
        if next_trial2 is not None:
            exclude_numbers.add(next_trial2)
        if prev_trial1 is not None:
            exclude_numbers.add(prev_trial1)
        if next_trial1 is not None:
            exclude_numbers.add(next_trial1)

        # Check for repetition and replace if necessary
        if (trial2 == prev_trial2) or (trial2 == next_trial2):
            possible_numbers = [n for n in numbers if n not in exclude_numbers]
            if possible_numbers:
                new_number = random.choice(possible_numbers)
                trial_seq2.trials[index2] = new_number  # Update the current trial in s2
    return trial_seq1, trial_seq2


def update_trials_dur2(trial_seq2, n_samples_ms_dict, tlo2):
    trials_dur2 = []
    for index2, trial2 in enumerate(trial_seq2.trials):
        duration2 = n_samples_ms_dict.get(trial2)
        t2_onset = index2 * tlo2
        t2_offset = t2_onset + tlo2
        if duration2 is not None:
            trials_dur2.append((index2, trial2, t2_onset, duration2, t2_offset))
    return trials_dur2

    # insert randomization script back here if necessary


def resolve_conflicts(trial_seq2, trial_seq1, trials_dur1, trials_dur2, tlo2, n_samples_ms_dict):
    # Initial categorization and conflict identification
    equal_trials, previous_trials, next_trials = categorize_conflicting_trials(trials_dur1, trials_dur2)
    equal_trials_conflicts, previous_trials_conflicts, next_trials_conflicts = get_conflict_lists(equal_trials,
                                                                                                  previous_trials,
                                                                                                  next_trials)

    # Loop until all conflicts are resolved
    while equal_trials_conflicts or previous_trials_conflicts or next_trials_conflicts:
        # Resolve conflicts and update trial_seq2 accordingly
        trial_seq2 = replace_s2_conflicts(trials_dur2, trial_seq2, equal_trials_conflicts, previous_trials_conflicts,
                                          next_trials_conflicts, trials_dur1)

        # Update trials_dur2 based on the new trial_seq2
        trials_dur2 = update_trials_dur2(trial_seq2, n_samples_ms_dict, tlo2)

        # Recategorize conflicts with the updated trials_dur2
        equal_trials, previous_trials, next_trials = categorize_conflicting_trials(trials_dur1, trials_dur2)
        equal_trials_conflicts, previous_trials_conflicts, next_trials_conflicts = get_conflict_lists(equal_trials,
                                                                                                      previous_trials,
                                                                                                      next_trials)

        # Optionally, check and update trial_seq2 to ensure no consecutive trials have the same number
        trial_seq1, trial_seq2 = check_updated_s2(trial_seq1, trial_seq2)

        trials_dur2 = update_trials_dur2(trial_seq2, n_samples_ms_dict, tlo2)

    return trial_seq1, trial_seq2, equal_trials_conflicts, previous_trials_conflicts, next_trials_conflicts, equal_trials, previous_trials, next_trials


def run_block(trial_seq1, trial_seq2, tlo1, tlo2):
    # [speaker1] = freefield.pick_speakers((speakers_coordinates[0], 0))  # speaker 15, -17.5 az, 0.0 ele
    # [speaker2] = freefield.pick_speakers((speakers_coordinates[1], 0))  # speaker 31, 17.5 az, 0.0 ele, target s1

    # elevation coordinates: works
    [speaker2] = freefield.pick_speakers((speakers_coordinates[2], -37.5))  # target s1
    [speaker1] = freefield.pick_speakers((speakers_coordinates[2], 37.5))

    sequence1 = numpy.array(trial_seq1.trials).astype('int32')
    sequence1 = numpy.append(0, sequence1)
    sequence2 = numpy.array(trial_seq2.trials).astype('int32')
    sequence2 = numpy.append(0, sequence2)
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
    freefield.write('channel1', speaker2.analog_channel, speaker2.analog_proc)
    freefield.write('channel2', speaker1.analog_channel, speaker1.analog_proc)
    freefield.play()


def run_experiment():  # works as desired
    completed_blocks = 0  # initial number of completed blocks
    for block in range(n_blocks):  # iterate over the number of blocks
        chosen_voice = wav_list_select(data_path)
        n_samples_ms, sound_dur_ms = write_buffer(chosen_voice)
        trial_seq1, trial_seq2, n_samples_ms_dict, tlo1, tlo2 = get_trial_sequence(sound_dur_ms, n_samples_ms)
        trials_dur1, trials_dur2 = trials_durations(trial_seq1, trial_seq2, tlo1, tlo2, n_samples_ms_dict)
        equal_trials, previous_trials, next_trials = categorize_conflicting_trials(trials_dur1, trials_dur2)
        equal_trials_conflicts, previous_trials_conflicts, next_trials_conflicts = get_conflict_lists(equal_trials,
                                                                                                      previous_trials,
                                                                                                      next_trials)
        trial_seq2 = replace_s2_conflicts(trials_dur2, trial_seq2, equal_trials_conflicts, previous_trials_conflicts,
                                          next_trials_conflicts, trials_dur1)
        trial_seq1, trial_seq2 = check_updated_s2(trial_seq1, trial_seq2)
        trials_dur2 = update_trials_dur2(trial_seq2, n_samples_ms_dict, tlo2)
        trial_seq1, trial_seq2, equal_trials_conflicts, previous_trials_conflicts, next_trials_conflicts, equal_trials, previous_trials, next_trials = resolve_conflicts(
            trial_seq2, trial_seq1, trials_dur1, trials_dur2, tlo2, n_samples_ms_dict)

        run_block(trial_seq1, trial_seq2, tlo1, tlo2)

        # Increment the count of completed blocks before user input to reflect the just-completed block
        completed_blocks += 1

        # Wait for user input to continue to the next block
        user_input = input('Press "1" to continue to the next block, or any other key to stop: ')
        if user_input == '1':
            print(f"Experiment completed {completed_blocks} out of {n_blocks} blocks.")
        else:
            print("Experiment stopped early by user.")
            break

    # If the loop completes without breaks, the final message will indicate all blocks are completed
    print(f"Experiment completed {completed_blocks} out of {n_blocks} blocks.")
    return completed_blocks


if __name__ == "__main__":
    freefield.initialize('dome', device=proc_list)

    run_experiment()
''' 

# PLOTTING TRIAL SEQUENCES OVER TIME
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
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

# Optionally, set the limits for better visibility if needed
plt.xlim(min(onsets_1 + onsets_2), max(onsets_1 + onsets_2))
plt.ylim(min(trials_1 + trials_2) - 1, max(trials_1 + trials_2) + 1)  # Adjusted for better y-axis visibility

# Show grid
plt.grid(True)

# Show the plot
plt.show()
'''