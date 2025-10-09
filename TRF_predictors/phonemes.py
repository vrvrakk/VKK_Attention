'''
script to create the phonemes predictor arrays, for each condition
phonemes extracted using Montreal Forced Aligner
use stream events to align phonemes
samplerate: 125 Hz
mask segments in array based on EEG marked bad segments
repeat for every sub, concatenate all of one condition (az left, az right, ele top, ele bottom)

'''

# import libraries
from pathlib import Path  # specifying paths / directories
import os  # for saving/loading
import numpy as np  # well for many things
import pandas as pd  # handling dataframes
import mne  # for EEG data
import textgrid  # for MFA data
from EEG.extract_events import sub_list  # import sub list


# define conditions
conditions = ['a1', 'a2', 'e1', 'e2']
# for conds a1 and e1, the stream 1 is the target
# for conds a2 and e2, stream 2 is target
sfreq = 125

for condition in conditions:
    # pick subjects depending on condition
    if condition in ['e1', 'e2']:
        valid_subs = sub_list[6:]  # only subs with elevation data
    else:
        valid_subs = sub_list  # all subs for azimuth

    for sub in valid_subs:
        print(f"Processing subject {sub}, condition {condition}")
        # define directories for extracting files / saving data
        default_dir = Path.cwd()
        data_dir = default_dir / 'data'
        predictors_dir = data_dir / 'eeg' / 'predictors'
        # stream events array and bad segments array for all EEG files of sub-condition concatenated
        stream_events_dir = predictors_dir / 'streams_events'/sub/condition
        # this will be needed at the very end of the script:
        bad_segments_dir = predictors_dir / 'bad_segments'/sub/condition

        # Montreal Forced Aligner phonemes for each voice:
        mfa_dir = data_dir / 'voices_english' / 'MFA'

        # import the block sequence information:
        block_sequences_dir = data_dir / 'params' / 'block_sequences' / f'{sub}.csv'
        block_csv = pd.read_csv(block_sequences_dir)

        # filter csv and keep rows with relevant condition:
        def filter_blocks(condition):
            if condition == 'a1':
                blocks_cond = block_csv[(block_csv['block_condition'] == 'azimuth') & (block_csv['block_seq'] == 's1')]
            elif condition == 'a2':
                blocks_cond = block_csv[(block_csv['block_condition'] == 'azimuth') & (block_csv['block_seq'] == 's2')]
            elif condition == 'e1':
                blocks_cond = block_csv[(block_csv['block_condition'] == 'elevation') & (block_csv['block_seq'] == 's1')]
            elif condition == 'e2':
                blocks_cond = block_csv[(block_csv['block_condition'] == 'elevation') & (block_csv['block_seq'] == 's2')]
            return blocks_cond

        blocks_cond = filter_blocks(condition)

        # now import stream events for selected condition and sub:
        def label_streams(condition):
            target_stream_event_arrays = []
            distractor_stream_event_arrays = []
            for files in stream_events_dir.iterdir():
                if condition in ['a1', 'e1']:
                    if 'stream1' in files.name:
                        stream_event = np.load(files, allow_pickle=True)
                        target_stream_event_arrays.append(stream_event)  # already in samplerate 125 Hz
                    elif 'stream2' in files.name:
                        stream_event = np.load(files, allow_pickle=True)
                        distractor_stream_event_arrays.append(stream_event)
                elif condition in ['a2', 'e2']:
                    if 'stream2' in files.name:
                        stream_event = np.load(files, allow_pickle=True)
                        target_stream_event_arrays.append(stream_event)  # already in samplerate 125 Hz
                    elif 'stream1' in files.name:
                        stream_event = np.load(files, allow_pickle=True)
                        distractor_stream_event_arrays.append(stream_event)
            return target_stream_event_arrays, distractor_stream_event_arrays


        target_stream_event_arrays, distractor_stream_event_arrays = label_streams(condition)
        # ok I now need the EEG files, to properly create aligned predictors:
        eeg_path = Path(f'D:/VKK_Attention/data/eeg/preprocessed/results/{sub}/ica')
        eeg_list = []
        phonemes_arrays = []

        for eeg_files in eeg_path.iterdir():
            if condition in eeg_files.name:
                eeg_file = mne.io.read_raw_fif(eeg_files, preload=True)
                eeg_file.resample(sfreq)
                eeg_len = len(eeg_file._data[1])
                array = np.zeros(eeg_len)
                phonemes_arrays.append(array)
                eeg_list.append(eeg_file)

        # get the voices of each block of selected condition:
        blocks_voices = blocks_cond['Voices'].reset_index(drop=True)

        import json
        events_dict_dir = 'D:/VKK_Attention/data/misc/eeg_events.json'
        with open(events_dict_dir, 'r') as f:
            eeg_events_dict = json.load(f)

        # define stream type:
        if condition in ['a1', 'e1']:
            target = 'stream1'
            distractor = 'stream2'
        elif condition in ['a2', 'e2']:
            target = 'stream2'
            distractor = 'stream1'


        def extract_phoneme_array(phonemes_arrays, stream_events_array, stream_type=''):
            # iterate over the voices with indexing:
            for index, row in enumerate(blocks_voices):
                # get the voice of the working block:
                voice = row
                # go to corresponding phoneme folder
                voice_folder = mfa_dir / f'aligned_{voice}'
                # great, now iterate over the events of this block:
                stream = stream_events_array[index]
                # Only keep TextGrid files, sorted by filename
                textgrid_files = sorted([f for f in voice_folder.iterdir() if f.suffix == ".TextGrid"])
                # now iterate over the events:
                for events in stream:
                    samplepoint = events[0]
                    number = events[-1]
                    if stream_type == 'stream2':
                        print(f'OG number: {number}')
                        number = number - 64
                        print(f'converted number: {number}')
                    if number == 7:  # number 7, if present, is a deviant, not a number
                        continue
                    # adjust index if > 7
                    if number > 7:
                        folder_index = number - 2
                    else:
                        folder_index = number - 1
                    # this is to properly get the textgrid file of the current stimulus number
                    txt_grd = textgrid_files[folder_index]
                    print("Using:", txt_grd.name, "for number", number)
                    tg = textgrid.TextGrid.fromFile(txt_grd)
                    # where binary impulses will be assigned at phoneme onsets
                    for intervals in tg[1]:
                        if intervals.mark not in (None, ''):
                            phoneme = intervals.mark
                            phoneme_onset = int(intervals.minTime * sfreq)
                            # phoneme_offset = int(intervals.maxTime * sfreq)
                            print(f'stimulus:{number}, text grid: {txt_grd}, phonemes: {phoneme}')
                            # asbolute onset: safety check when bounds don't match
                            abs_onset = samplepoint + phoneme_onset
                            if abs_onset < len(phonemes_arrays[index]):
                                # add 1s in phoneme_array, for phoneme impulses (binary):
                                phonemes_arrays[index][abs_onset] = 1
                            else:
                                print(
                                    f"Skipping out-of-bounds onset {abs_onset} (array length {len(phonemes_arrays[index])})")

            # check lens of ones:
            for i, array in enumerate(phonemes_arrays):
                arrays = phonemes_arrays[i]
                ones = [x for x in arrays if x == 1]
                print(f'Total number of impulses per phoneme array: {len(ones)}')
            # concatenate the arrays:
            phonemes_concat = np.concatenate(phonemes_arrays)
            print(f'final phonemes length: {len(phonemes_concat)}')
            return phonemes_concat


        target_phonemes_concat = extract_phoneme_array(phonemes_arrays, target_stream_event_arrays, stream_type=target)
        distractor_phonemes_concat = extract_phoneme_array(phonemes_arrays, distractor_stream_event_arrays, stream_type=distractor)

        for files in bad_segments_dir.iterdir():
            if 'bad_series_concat.npy.npz' in files.name:
                bad_segments = np.load(files, allow_pickle=True)
                bad_series = bad_segments['bad_series']  # great, also 125 Hz
                # with this series, bad segments in the phoneme arrays will be masked

        # mask phonemes_array:
        # first, check if lengths match:
        print(f'target array len: {len(target_phonemes_concat)}, bad series len: {len(bad_series)}')
        print(f'distractor array len: {len(distractor_phonemes_concat)}, bad series len: {len(bad_series)}')

        # create saving dir:
        save_dir = predictors_dir / 'phonemes' / condition
        os.makedirs(save_dir, exist_ok=True)
        target_filename = save_dir / f'{sub}_target_phonemes_concat.npz'
        target_phonemes_masked = target_phonemes_concat[bad_series == 0]
        np.savez(target_filename, phonemes=target_phonemes_masked)
        print(f'Saved target phoneme array of {sub}, condition {condition}, '
              f'as {target_filename}')
        distractor_filename = save_dir / f'{sub}_distractor_phonemes_concat.npz'
        distractor_phonemes_masked = distractor_phonemes_concat[bad_series == 0]
        np.savez(distractor_filename, phonemes=distractor_phonemes_masked)
        print(f'Saved distractor phoneme array of {sub}, condition {condition}, '
              f'as {distractor_filename}')

# ultimate concatenation:

for condition in conditions:
    target_arrays = []
    distractor_arrays = []
    concat_dir = predictors_dir / 'phonemes' / 'concat' / condition
    saved_phonemes = predictors_dir / 'phonemes' / condition
    concat_dir.mkdir(parents=True, exist_ok=True)
    for files in saved_phonemes.iterdir():
        if 'target' in files.name:
            tarray = np.load(files)
            tarray = tarray['phonemes']
            target_arrays.append(tarray)
        elif 'distractor' in files.name:
            print(files)
            darray = np.load(files)
            darray = darray['phonemes']
            distractor_arrays.append(darray)
    # concatenate all files:
    ultimate_target_phoneme = np.concatenate(target_arrays)
    target_filename = concat_dir/f'{condition}_concat_target_phonemes.npz'
    np.savez(target_filename, phonemes=ultimate_target_phoneme)
    ultimate_distractor_phoneme = np.concatenate(distractor_arrays)
    distractor_filename = concat_dir/f'{condition}_concat_distractor_phonemes.npz'
    np.savez(distractor_filename, phonemes=ultimate_distractor_phoneme)
    print(f'Saved across-sub concatenated phoneme arrays for:'
          f'{target_filename} and'
          f'{distractor_filename}')

