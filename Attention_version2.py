import slab
import numpy
import os
import random
import freefield
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


n_trials1 = 656  # 4.99 min total
n_trials2 = 896  # 5.00 min total
# s2_delay = 10
# Dai & Shinn-Cunningham (2018):
isi = (457, 335)  # so that tlo1 = 1202, tlo2 = 1080 (by factor * 1.364)
# choose speakers:
speakers_coordinates = (17.5, 0)  # directions for each streams
# s2_delay = 2000 # delay not necessary if target always on the right
sample_freq = 24414
numbers = [1, 2, 3, 4, 5, 6, 8, 9]
data_path = Path.cwd() / 'data' / 'voices_padded'
sequence_path = Path.cwd() / 'data' / 'generated_sequences'
participant_id = '240304_pf_azimuth'
# n_samples = 18210
# sound_dur_ms = int((n_samples / 24414) * 1000)
proc_list = [['RX81', 'RX8', Path.cwd() / 'experiment.rcx'],
             ['RX82', 'RX8', Path.cwd() / 'experiment.rcx']]

freefield.set_logger('warning')

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
            # freefield.write(f'{number}', s.data, ['RX81', 'RX82'])  # loads array on buffer
            # freefield.write(f'{number}_n_samples', s.n_samples,['RX81', 'RX82'])  # sets total buffer size according to numeration
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
            trials_dur1.append((index1, trial1, t1_onset, duration1, t1_offset, 's1'))

    # now for trial_seq2.trials:
    trials_dur2 = []
    for index2, trial2 in enumerate(trial_seq2.trials):
        duration2 = n_samples_ms_dict.get(trial2)
        t2_onset = (index2 * tlo2)  # + s2_delay
        t2_offset = t2_onset + tlo2
        if duration2 is not None:
            trials_dur2.append((index2, trial2, t2_onset, duration2, t2_offset, 's2'))

    return trials_dur1, trials_dur2


def events_table(trials_dur1, trials_dur2):
    # Assuming you have a list of tuples like this
    df1 = pd.DataFrame([(index, trial, onset, offset, stim) for index, trial, onset, _, offset, stim in trials_dur1],
                       columns=['index', 'trial', 't_onset', 't_offset', 'stimulus'])
    df2 = pd.DataFrame([(index, trial, onset, offset, stim) for index, trial, onset, _, offset, stim in trials_dur2],
                       columns=['index', 'trial', 't_onset', 't_offset', 'stimulus'])

    # Concatenate the two DataFrames
    combined_df = pd.concat([df1, df2], ignore_index=True)
    combined_df = combined_df.sort_values(by=['t_onset', 't_offset'])
    # Define your column headers
    combined_df.rename(
        columns={'index': 'Event IDs', 't_onset': 'Time onsets', 't_offset': 'Time offsets', 'stimulus': 'Stimulus',
                 'trial': 'Stream of Trials'}, inplace=True)
    combined_df.reset_index(drop=True, inplace=True)  # fixing pd indices
    combined_df['Conflicts'] = None  # Placeholders

    return combined_df


def find_conflicts(combined_df):
    for i in range(1, len(combined_df)):
        if combined_df.loc[i, 'Stream of Trials'] == combined_df.loc[i - 1, 'Stream of Trials']:
            combined_df.loc[i, 'Conflicts'] = True
            combined_df.loc[i - 1, 'Conflicts'] = True
    for i in range(len(combined_df) - 2):  # Iterate, stopping 2 trials before the end
        if combined_df.loc[i, 'Stimulus'] == 's1' and combined_df.loc[i + 2, 'Stimulus'] == 's2':
            if combined_df.loc[i, 'Stream of Trials'] == combined_df.loc[i + 2, 'Stream of Trials']:
                # Mark the conflicting 's1' and 's2' trials as True in the 'Conflicts' column
                combined_df.loc[i, 'Conflicts'] = True
                combined_df.loc[i + 2, 'Conflicts'] = True
    combined_df['Conflicts'].fillna(False, inplace=True)  # fill the rest with 'False'
    return combined_df


def replace_conflict(combined_df):
    possible_numbers = set(numbers)
    for current_trial in range(len(combined_df)):
        if combined_df.loc[current_trial, 'Conflicts']:
            # Define the span of trials to consider based on the current position
            span_start = max(current_trial - 3, 0)  # does not span below 0
            span_end = min(current_trial + 4, len(combined_df))  # does not span beyond length of df

            trials_window = {combined_df.loc[trial, 'Stream of Trials'] for trial in range(span_start, span_end) if trial != current_trial}

            # Calculate the current possible numbers by excluding surrounding trials
            current_possible_numbers = possible_numbers - trials_window

            # Ensure there are possible numbers available for replacement
            if current_possible_numbers:
                new_trial = random.choice(list(current_possible_numbers))  # Randomly choose a new trial number
                combined_df.loc[current_trial, 'Stream of Trials'] = new_trial  # Replace the trial number in the DataFrame

    return combined_df


def update_sequences(combined_df, trial_seq1, trial_seq2):
    for i in range(len(combined_df)):
        event_id = combined_df.loc[i, 'Event IDs']
        new_trial = combined_df.loc[i, 'Stream of Trials']
        stimulus = combined_df.loc[i, 'Stimulus']

        if stimulus == 's1':
            # Check if the trial number has been changed
            if trial_seq1.trials[event_id] != new_trial:
                trial_seq1.trials[event_id] = new_trial
        elif stimulus == 's2':
            # Check if the trial number has been changed
            if trial_seq2.trials[event_id] != new_trial:
                trial_seq2.trials[event_id] = new_trial

    return trial_seq1, trial_seq2


def save_sequence(participant_id, sequence_path, combined_df):
    file_name = f'{sequence_path}/{participant_id}.csv'
    index = 1
    while os.path.exists(file_name):
        file_name = f'{sequence_path}/{participant_id}_{index}.csv'
        index += 1
    combined_df.to_csv(file_name, index=False, sep=';')

    return file_name



def run_block(trial_seq1, trial_seq2, tlo1, tlo2):
    [speaker1] = freefield.pick_speakers((speakers_coordinates[1], 0))  # speaker 31, 17.5 az, 0.0 ele (target)
    [speaker2] = freefield.pick_speakers((speakers_coordinates[0], 0))  # speaker 23, 0.0 az, 0.0 ele

    # elevation coordinates: works
    # [speaker1] = freefield.pick_speakers((speakers_coordinates[2], -37.5))  # s1 target
    # [speaker2] = freefield.pick_speakers((speakers_coordinates[2], 37.5))  # s2 distractor

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
    freefield.write('channel1', speaker1.analog_channel, speaker1.analog_proc)  # s1 target
    freefield.write('channel2', speaker2.analog_channel, speaker2.analog_proc)  # s2 distractor
    freefield.play()


def run_experiment():  # works as desired
        chosen_voice = wav_list_select(data_path)
        n_samples_ms, sound_dur_ms = write_buffer(chosen_voice)
        trial_seq1, trial_seq2, n_samples_ms_dict, tlo1, tlo2 = get_trial_sequence(sound_dur_ms, n_samples_ms)
        trials_dur1, trials_dur2 = trials_durations(trial_seq1, trial_seq2, tlo1, tlo2, n_samples_ms_dict)
        combined_df = events_table(trials_dur1, trials_dur2)
        combined_df = find_conflicts(combined_df)
        combined_df = replace_conflict(combined_df)
        trial_seq1, trial_seq2 = update_sequences(combined_df, trial_seq1, trial_seq2)
        file_name = save_sequence(participant_id, sequence_path, combined_df)

        run_block(trial_seq1, trial_seq2, tlo1, tlo2)


if __name__ == "__main__":
    freefield.initialize('dome', device=proc_list)

    run_experiment()
''' 

# PLOTTING TRIAL SEQUENCES OVER TIME

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