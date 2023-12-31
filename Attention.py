import time
import slab
import numpy
import os
import random
import freefield
import datetime
import pickle
from pathlib import Path

participant_id = 'vk'
n_blocks = 1
n_trials1 = 20
speakers = (-17.5, 17.5)  # directions for each streams
isi = (500, 750)


sample_freq = 48828
numbers = [1, 2, 3, 4, 5, 6, 8, 9]
data_path = Path.cwd()/'data'/'voices'
responses_dir=Path.cwd()/'data'/'results'
proc_list = [['RX81', 'RX8', Path.cwd() / 'test2.rcx'],
             ['RX82', 'RX8', Path.cwd() / 'test2.rcx'],
             ['RP2', 'RP2', Path.cwd() / '9_buttons.rcx']]

def get_wav_list(data_path):  # create wav_list paths
    voice_idx = list(range(1, 5))
    folder_paths = []
    wav_folders = [folder for folder in os.listdir(data_path)]
    for i, folder in zip(voice_idx,wav_folders):
        #print(i,folder)
        folder_path = data_path / folder
        #print(folder_path)
        folder_paths.append(folder_path) # absolute path of each voice folder
    # Initialize the corresponding wav_files list
    wav_files_lists = []
    for i,folder_path in zip(voice_idx,folder_paths):
        wav_files_in_folder=list(folder_path.glob("*.wav"))
        wav_files_lists.append(wav_files_in_folder)
        chosen_voice = random.choice(wav_files_lists)
    return chosen_voice

def write_buffer(chosen_voice):
    n_samples = []
    n_samples_ms = []
    for number, file_path in zip(numbers, chosen_voice):  # combine lists into a single iterable->elements from corresponding positions are paired together
        # print(f'{number} {file_path}')
        if os.path.exists(file_path):
            s = slab.Sound(data=file_path)
            s = s.resample(48828)
            n_samples.append(s.n_samples)
            freefield.write(f'{number}', s.data, ['RX81', 'RX82'])  # loads array on buffer
            freefield.write(f'{number}_n_samples', s.n_samples,
                          ['RX81', 'RX82'])  # sets total buffer size according to numeration
            sound_duration_ms = int((s.n_samples / 50000) * 1000)
            n_samples_ms.append(sound_duration_ms)
    n_samples_ms = zip(numbers, n_samples_ms)
    n_samples_ms = list(n_samples_ms)
    return n_samples_ms

def get_trial_sequence(n_trials1, n_samples_ms):
    # get trial duration for both streams plus n_trials of lagging stream
    mean_n_samples = numpy.mean(n_samples_ms)  # fixed samples mean based on ALL n_samples from all wav files
    tlo1 = int(isi[0] + (mean_n_samples / 50000) * 1000)  # isi + (mean sample size of sound event / sample freq)* 1000 for ms
    tlo2 = int(isi[1] + (mean_n_samples / 50000) * 1000)
    t_end = n_trials1 * tlo1 # total length of s1
    n_trials2 = int(numpy.floor((t_end - s2_delay) / tlo2)) # to estimate the total amount of s2 trials

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
    for trials1 in trial_seq1.trials:
        duration = n_samples_ms_dict.get(trials1) # get dur of each trial, that corresponds to trial from n_samples_dict
        if duration is not None:
            trials_dur1.append((trials1, duration))

    # now for trial_seq2.trials:
    trials_dur2 = []
    for trials2 in trial_seq2.trials:
        duration = n_samples_ms_dict.get(trials2)
        if duration is not None:
            trials_dur2.append((trials2, duration))



    t1 = numpy.arange(0,trial_seq1.n_trials) * tlo1 # sound onsets across time for s1
    t2 = (numpy.arange(0,trial_seq2.n_trials) * tlo2) + s2_delay  # sound onsets across time for s2, with delay implemented

    # get list with time onsets of t1 + trials_dur + trial numbers
    t1_event_durs=[]
    for t1_onset, (trial_number, dur1) in zip(t1, trials_dur1):
        offset = t1_onset + dur1
        t1_event_durs.append((t1_onset, offset, trial_number))
    # same for t2:
    t2_event_durs=[]
    for t2_onset, (trial_number, dur2) in zip(t2, trials_dur2):
        offset=t2_onset + dur2
        t2_event_durs.append((t2_onset, offset, trial_number))

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

    for i, (t1_trial_number, t2_trial_number) in enumerate(overlapping_trials):
        if t1_trial_number == t2_trial_number:
            # Exclude the current number from the options
            exclude_numbers = {t1_trial_number}

            # Find the index of t2_number in trial_seq2.trials
            t2_index = trial_seq2.trials.index(t2_trial_number)

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
                overlapping_trials[i] = (t1_trial_number, new_number)

    return trial_seq1, trial_seq2, tlo1, tlo2

def run_block(trial_seq1, trial_seq2, tlo1, tlo2, speakers, n_trials1, participant_initials = 'vkk'):
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
    freefield.write('n_trials1', trial_seq1.n_trials + 1, ['RX81', 'RX82'])  # analog_proc attribute from speaker table dom txt file
    freefield.write('n_trials2', trial_seq2.n_trials + 1, ['RX81', 'RX82'])
    freefield.write('trial_seq1', sequence1 ,['RX81', 'RX82'])
    freefield.write('trial_seq2', sequence2, ['RX81', 'RX82'])
    # set output speakers for both streams
    freefield.write('channel1',speaker1.analog_channel, speaker1.analog_proc)
    freefield.write('channel2', speaker2.analog_channel, speaker2.analog_proc)
    # convert sequence numbers to integers, add a 0 at the beginning and write to trial sequence buffers
    freefield.play()
    # responses_table = []
    # start_time = time.time()
    # while True:
    #     trial_count = freefield.read('trial_idx', speaker1.analog_proc)
    #     trial_start_time = start_time + trial_count * tlo1
    #     s1_number = freefield.read('s1_number', speaker1.analog_proc)
    #     s2_number = 0 if trial_start_time <= s2_delay else freefield.read('s2_number', speaker2.analog_proc)
    #     current_time_within_trial = start_time - trial_start_time
    #     while current_time_within_trial < tlo1 / 1000:
    #         response = freefield.read('button', 'RP2')
    #         if response != 0:  # Record every non-zero response
    #             responses_table.append([trial_start_time, current_time_within_trial, response, s1_number, s2_number])
    #         current_time_within_trial = time.time() - trial_start_time
    #     if trial_count >= trial_seq1.n_trials:
    #         break
    #     # use time function to record button press times and compare to stimulus times (from the lists created)
    # # to save:
    # date_str = datetime.datetime.now().strftime("%Y%m%d")  # Format: YYYYMMDD
    # pickle_file_name = responses_dir / f"{participant_initials}_{date_str}.pkl"
    # # Save responses_table to a pickle file
    # with open(f"results/{pickle_file_name}", 'wb') as file:  # Note the 'wb' mode for writing binary files
    #     pickle.dump(responses_table, file)
    # return responses_table

def run_experiment(participant_id, n_blocks, n_trials1, speakers, isi):
    global s2_delay
    s2_delay = 2000  # delay for the lagging stream in ms
    completed_blocks = 0  # initial number of completed blocks
    for block in range(n_blocks):  # iterate over the number of blocks
        chosen_voice = get_wav_list(data_path)
        n_samples_ms = write_buffer(chosen_voice)
        trial_seq1, trial_seq2, tlo1, tlo2 = get_trial_sequence(n_trials1, n_samples_ms)

        run_block(trial_seq1, trial_seq2, tlo1, tlo2, speakers, n_trials1, participant_id)
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

    # run_experiment(participant_id, n_blocks, n_trials1, speakers, isi)

