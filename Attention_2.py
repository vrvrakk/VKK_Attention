import time
import slab
import numpy
import os
import random
import freefield
import matplotlib.pyplot as plt
import matplotlib.collections as collections
import datetime

from pathlib import Path

voice_idx=list(range(1,5))
n_trials1 = 96
n_trials2= 75
sources = (-17.5, 17.5)  # directions for each streams
isi=(500,750)
s2_delay = 2000  # delay for the lagging stream in ms

sample_freq = 48828
data_path = Path.cwd()/'data'/'voices'
responses_dir=Path.cwd()/'data'/'button_presses'
wav_folders = [folder for folder in os.listdir(data_path)]
folder_paths = []

wav_files_lists = []

def get_wav_list (data_path,voice_idx, folder_paths): # create wav_list paths
    [speaker1] = freefield.pick_speakers((sources[0], 0))  # set speakers to play
    [speaker2] = freefield.pick_speakers((sources[1], 0))

    for i, folder in zip(voice_idx,wav_folders):
        #print(i,folder)
        folder_path = data_path / folder
        #print(folder_path)
        folder_paths.append(folder_path) # absolute path of each voice folder
    # Initialize the corresponding wav_files list

    for i,folder_path in zip(voice_idx,folder_paths):
        wav_files_in_folder=list(folder_path.glob("*.wav"))
        wav_files_lists.append(wav_files_in_folder)
        chosen_voice = random.choice(wav_files_lists)
        numbers = [1, 2, 3, 4, 5, 6, 8, 9]
        n_samples = []
        n_samples_ms = []
        for number, file_path in zip(numbers, chosen_voice):  # combine lists into a single iterable->elements from corresponding positions are paired together
            # print(f'{number} {file_path}')
            if os.path.exists(file_path):
                s = slab.Sound(data=file_path)
                s = s.resample(48828)
                n_samples.append(s.n_samples)
                # freefield.write(f'{number}', s.data, ['RX81', 'RX82'])  # loads array on buffer
                # freefield.write(f'{number}_n_samples', s.n_samples,
                #               ['RX81', 'RX82'])  # sets total buffer size according to numeration
                sound_duration_ms = int((s.n_samples / 50000) * 1000)
                n_samples_ms.append(sound_duration_ms)
        n_samples_ms = zip(numbers, n_samples_ms)
        n_samples_ms = list(n_samples_ms)

    return wav_files_lists, n_samples_ms, chosen_voice, speaker1, speaker2  # absolute path of each wav file

def trials(n_trials1,n_trials2,numbers,n_samples_ms):

    # here we set trial sequence
    trial_seq1 = slab.Trialsequence(conditions=numbers, n_reps=n_trials1 / len(numbers), kind='non_repeating')
    for i in range(len(trial_seq1.trials)):
        if trial_seq1.trials[i] == 7:  # replace 7 with 9 in trial_seq.trials
            trial_seq1.trials[i] = 9
    trial_seq2 = slab.Trialsequence(conditions=numbers, n_reps=n_trials2 / len(numbers), kind='non_repeating')
    for i in range(len(trial_seq2.trials)):
        if trial_seq2.trials[i] == 7:
            trial_seq2.trials[i] = 9

    # get list with dur of each trial:
    n_samples_ms_dict = dict(n_samples_ms)

    trials_dur1 = []
    for trials in trial_seq1.trials:
        duration = n_samples_ms_dict.get(trials)
        if duration is not None:
            trials_dur1.append((trials,duration))
        else:
            # Handle the case where the trial number is not in the dictionary
            print(f"Warning: No duration found for number {trials}")
            trials_dur1.append((trials,0))  # Or handle this case as needed

    # now for trial_seq2.trials:
    trials_dur2 = []
    for trials in trial_seq2.trials:
        duration = n_samples_ms_dict.get(trials)
        if duration is not None:
            trials_dur2.append((trials, duration))
        else:
            # Handle the case where the trial number is not in the dictionary
            print(f"Warning: No duration found for number {trials}")
            trials_dur2.append((trials, 0))  # Or handle this case as needed

    # get trial duration for both streams plus n_trials of lagging stream
    mean_n_samples = 27330  # fixed samples mean based on ALL n_samples from all wav files
    tlo1 = int(isi[0] + (mean_n_samples / 50000) * 1000) # isi + (mean sample size of sound event / sample freq)* 1000 for ms
    tlo2 = int(isi[1] + (mean_n_samples / 50000) * 1000)
    t_end = n_trials1 * tlo1 # total length of s1
    # n_trials2 = int(numpy.floor((t_end - s2_delay) / tlo2)) # to estimate the total amount of s2 trials

    t1 = numpy.arange(0,trial_seq1.n_trials) * tlo1 # sound onsets across time for s1
    t2 = (numpy.arange(0,trial_seq2.n_trials) * tlo2) + s2_delay  # sound onsets across time for s2, with delay implemented

    # get list with time onsets of t1 + trials_dur + trial numbers
    t1_event_durs=[]
    for t1_val, (trial_number, dur1) in zip(t1, trials_dur1):
        offset = t1_val + dur1
        t1_event_durs.append((t1_val, offset, trial_number))
    # same for t2:
    t2_event_durs=[]
    for t2_val, (trial_number, dur2) in zip(t2, trials_dur2):
        offset=t2_val + dur2
        t2_event_durs.append((t2_val, offset,trial_number))

    # make sure both streams have different numbers at concurrent trials:
    overlapping_trials=[]
    # Loop through each trial in stream 1
    for t1_onset, t1_offset, t1_trial_number in t1_event_durs:
        # Check against each trial in stream 2
        for t2_onset, t2_offset, t2_trial_number in t2_event_durs:
            # Check if there is any overlap
            if t1_onset < t2_offset and t2_onset < t1_offset:
                # There is an overlap if stream 1 onset is before stream 2 offset,
                # and stream 2 onset is before stream 1 offset
                overlapping_trials.append((t1_trial_number, t2_trial_number))

    for i, (t1_number, t2_number) in enumerate(overlapping_trials):
        if t1_number == t2_number:
            # Exclude the current number from the options
            exclude_numbers = {t1_number}

            # Find the index of t2_number in trial_seq2.trials
            t2_index = trial_seq2.trials.index(t2_number)

            # Check and add the previous trial's number if not the first trial
            if t2_index > 0:
                exclude_numbers.add(trial_seq2.trials[t2_index - 1])

            # Check and add the next trial's number if not the last trial
            if t2_index < len(trial_seq2.trials) - 1:
                exclude_numbers.add(trial_seq2.trials[t2_index + 1])

            # Generate a list of possible replacement numbers
            possible_numbers = [n for n in numbers if n not in exclude_numbers]

            # Choose a new number and replace it in trial_seq2.trials
            if possible_numbers:
                new_number = random.choice(possible_numbers)
                trial_seq2.trials[t2_index] = new_number
                # Update the overlapping_trials list with the new number
                overlapping_trials[i] = (t1_number, new_number)
            else:
                # Handle the case where no possible number is found
                print(f"No alternative number available to replace trial {t2_index + 1} in Stream 2.")

    sequence1 = numpy.array(trial_seq1.trials).astype('int32')
    sequence1 = numpy.append(0, sequence1)
    sequence2 = numpy.array(trial_seq2.trials).astype('int32')
    sequence2 = numpy.append(0, sequence2)

    return sequence1, sequence2, trial_seq1, trial_seq2, tlo1, tlo2

def run_block(sequence1, sequence2, trial_seq1, trial_seq2, tlo1, tlo2, speaker1, speaker2):
    # isi, s2_delay and n_trials only for calculating stuff->i.e. tlo
    # here we set tlo to RX8
    freefield.write('tlo1', tlo1, speaker1.analog_proc)
    freefield.write('tlo2', tlo2, speaker2.analog_proc)
    # set n_trials to pulse trains sheet0/sheet1
    freefield.write('n_trials1', trial_seq1.n_trials + 1,speaker1.analog_proc)  # analog_proc attribute from speaker table dom txt file
    freefield.write('n_trials2', trial_seq2.n_trials + 1, speaker2.analog_proc)
    freefield.write('trial_seq1', sequence1,speaker1.analog_proc)
    freefield.write('trial_seq2', sequence2,speaker2.analog_proc)
    # set output speakers for both streams
    freefield.write('channel1',speaker1.analog_channel, speaker1.analog_proc)
    freefield.write('channel2', speaker2.analog_channel, speaker2.analog_proc)
    # convert sequence numbers to integers, add a 0 at the beginning and write to trial sequence buffers

    freefield.play()
    responses_table=[]
    while True:
        trial_count = freefield.read('trial_idx', speaker1.analog_proc) # get index values of trial_seq1 trials
        while not True: # if trial index is equal or larger than 96
            break
        trial_start_time = trial_count * tlo1 # otherwise calculate sound onsets for each trial
        s1_number = freefield.read('s1_number', speaker1.analog_proc) # get number played from speaker 1
        s2_number = 0 if trial_start_time <= s2_delay else freefield.read('s2_number', speaker2.analog_proc)
        # if trial_start_time is smaller or equal to the delay, s2 number is 0, otherwise, get s2 number played by speaker 2
        last_response = None # initialize last_response
        last_response_time = -1 # last_response_time starts from -1, as we need to capture the very first response
        current_time_within_trial = 0

        while current_time_within_trial < tlo1:
            response = freefield.read('button', 'RP2')
            if response != 0 and current_time_within_trial > last_response_time: # if response is not 0 + if there is a later response time
                last_response = response
                last_response_time = current_time_within_trial
            current_time_within_trial += 1 # check for every ms from 0 to 1046

        responses_table.append([trial_start_time, last_response, s1_number, s2_number])
        # use time function to record button press times and compare to stimulus times (from the lists created)
    completed_blocks += 1
    user_input = input('Press "1" for next block or "0" to stop: ')
    print(f"Block {completed_blocks} completed.")
    # Construct the file name
    if user_input == '0':
        print("Experiment stopped early by user.")
        break
    # to save:
    date_str = datetime.datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD
    file_name = responses_dir / f"{participant_initials}_{date_str}_block_{block + 1}.csv"
    # Save responses_table to a file
    with open(file_name, 'w') as file:
        for entry in responses_table:
            file.write(','.join(map(str, entry)) + '\n')

    else:
        print(f"Experiment completed all {n_blocks} blocks.")

    freefield.halt()
# trial_seq has responses attribute-> could be used to save responses for each trial
# and write trial_seq.responses as a pickle file
    return numbers, n_samples_ms, responses_table


def run_experiment( s2_delay, n_trials1):

# add n_blcoks as arg in function
# for block in n_blocks
# run run_block function
# if they are ready, press 1 to run next block (user response)
# could return responses and save them here

    n_blocks = 5 # total number of blocks per experiment ran
    completed_blocks = 0 # initial number of completed blocks
    for block in range(n_blocks): # run function run_block 5 times
        sequence1, sequence2, trial_seq1, trial_seq2, tlo1, tlo2 = run_block(wav_files_lists)
        freefield.play(kind='zBusA') # play block
        responses_table = [] # create empty responses table


    return responses_table, completed_blocks


if __name__ == "__main__":
    proc_list=[['RX81','RX8',  Path.cwd()/'test2.rcx'],
               ['RX82','RX8',  Path.cwd()/'test2.rcx'],
               ['RP2','RP2',  Path.cwd()/'9_buttons.rcx']]
    freefield.initialize('dome',device=proc_list)
    run_experiment()

