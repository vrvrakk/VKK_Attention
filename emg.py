'''
Pre-processing EMG data: baseline recording,
motor-only responses and epoching our signal around target
and distractor events respectively.
'''
# libraries:
from copy import deepcopy
from collections import Counter
import mne
from pathlib import Path
import os
import numpy as np
from scipy.stats import mode
from scipy.signal import butter, filtfilt
from collections import Counter
import json
from meegkit import dss
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt, patches
from meegkit.dss import dss_line
import pandas as pd
from helper import grad_psd, snr

def get_target_blocks():
    # specify axis:
    if condition in ['a1', 'a2']:
        axis = 'azimuth'
    elif condition in ['e1', 'e2']:
        axis = 'elevation'
    else:
        print('No valid condition defined.')
    # define target stream: s1 or s2?
    # we know this from the condition
    if condition in ['a1', 'e1']:
        target_stream = 's1'
        distractor_stream = 's2'
        target_mapping = s1_mapping
        distractor_mapping = s2_mapping
    elif condition in ['a2', 'e2']:
        target_stream = 's2'
        distractor_stream = 's1'
        target_mapping = s2_mapping
        distractor_mapping = s1_mapping
    else:
        raise ValueError("Invalid condition provided.")  # Handle unexpected conditions
    target_blocks = []
    # iterate through the values of the csv path; first enumerate to get the indices of each row
    for idx, items in enumerate(csv.values):
        block_seq = items[0]  # from first column, save info as variable block_seq
        block_condition = items[1]  # from second column, save info as var block_condition
        if block_seq == target_stream and block_condition == axis:
            block = csv.iloc[idx]
            target_blocks.append(block)  # Append relevant rows
    target_blocks = pd.DataFrame(target_blocks).reset_index(drop=True)  # convert condition list to a dataframe
    return target_stream, distractor_stream, target_blocks, axis, target_mapping, distractor_mapping

def define_events(target_stream):
    if target_stream in ['s1']:
        target_events = s1_events
    elif target_stream in ['s2']:
        target_events = s2_events
    else:
        print('No valid target stream defined.')
    if target_events == s1_events:
        distractor_events = s2_events
    elif target_events == s2_events:
        distractor_events = s1_events
    return target_events, distractor_events


def create_events(chosen_events, events_mapping):
    target_number = target_block[3]
    events = mne.events_from_annotations(emg)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in chosen_events.values()]
    event_value = [int(key) for key, val in events_mapping.items() if val == target_number]  # specify key as integer,
    # otherwise event_value will be empty (as keys are strings normally)
    event_value = event_value[0]
    final_events = [event for event in filtered_events if event[2] == event_value]
    return final_events

def baseline_events(chosen_events, events_mapping):
    target_number = target_block[3]
    events = mne.events_from_annotations(emg)[0]  # get events from annotations attribute of raw variable
    events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
    filtered_events = [event for event in events if event[2] in chosen_events.values()]
    event_value = [int(key) for key, val in events_mapping.items() if val == target_number]  # specify key as integer,
    # otherwise event_value will be empty (as keys are strings normally)
    event_value = event_value[0]
    final_events = [event for event in filtered_events if event[2] != event_value]
    return final_events

# band-pass filter 20-150 Hz
# Remove low-frequency noise: Eliminates motion artifacts or baseline drift that occur below 20 Hz.
# Remove high-frequency noise: Filters out high-frequency noise (e.g., electrical noise or other non-EMG signals) above 150-450 Hz,
# which isn't part of typical muscle activity.
def filter_emg(emg):
    emg_filt = emg.copy().filter(l_freq=1, h_freq=150)
    '''The typical frequency range for EMG signals from the Flexor Digitorum Profundus (FDP) muscle is around 20 to 450 Hz. 
    Most muscle activity is concentrated in the 50-150 Hz range, but high-frequency components can go up to 450 Hz.'''
    # Notch filter at 50 Hz to remove power line noise
    print('Remove power line noise and apply minimum-phase highpass filter')  # Cheveigné, 2020
    emg_filt.copy().notch_filter(freqs=[50, 100, 150], method='fir')
    emg_filt.append(emg_filt)
    return emg_filt


# rectify the EMG data:
def rectify(emg_filt):
    emg_rectified = emg_filt.copy()
    emg_rectified._data = np.abs(emg_rectified._data)
    # emg_rectified._data *= 1e6
    return emg_rectified

# apply smoothing on EMG data:
# def smoothing(emg_rectified):
#     emg_smoothed = emg_rectified.copy().filter(l_freq=None, h_freq=5)
#     emg_smoothed._data *= 1e6  # Convert from μV to mV for better scale
#     return emg_smoothed



# get baseline and motor recording:
def get_helper_recoring(recording):
    for files in sub_dir.iterdir():
        if files.is_file and recording in files.name and files.suffix == '.vhdr':
            helper = files
    helper = mne.io.read_raw_brainvision(helper, preload=True)
    helper.rename_channels(mapping)
    helper.set_montage('standard_1020')
    helper = helper.pick_channels(['A2', 'M2', 'A1'])
    helper.set_eeg_reference(ref_channels=['A1'])
    helper = mne.set_bipolar_reference(helper, anode='A2', cathode='M2', ch_name='EMG_bipolar')
    helper.drop_channels(['A1'])
    helper_filt = helper.copy().filter(l_freq=1, h_freq=150)
    print('Remove power line noise and apply minimum-phase highpass filter')  # Cheveigné, 2020
    helper_filt.copy().notch_filter(freqs=[50, 100, 150], method='fir')
    helper_rectified = helper.copy()
    helper_rectified._data = np.abs(helper_rectified._data)
    # helper_rectified._data *= 1e6
    # create helper epochs manually:
    helper_n_samples = len(helper_rectified.times)
    return helper_rectified, helper_n_samples


# Create baseline events and epochs
def baseline_epochs(events_emg, emg_rectified, events_dict, target):
    tmin = -0.2
    tmax = 0.9
    event_ids = {key: val for key, val in events_dict.items() if any(event[2] == val for event in events_emg)}
    epochs_baseline = mne.Epochs(emg_rectified, events_emg, event_id=event_ids, tmin=tmin, tmax=tmax,
                        baseline=None, preload=True)
    epochs_baseline.save(fif_path / f'{sub_input}_condition_{condition}_{index}_{target}-epo.fif', overwrite=True)
    return epochs_baseline


# get emg epochs, around the target and distractor events:
def epochs(events_dict, emg_rectified, events_emg, target='', tmin=0, tmax=0, baseline=None):
    event_ids = {key: val for key, val in events_dict.items() if any(event[2] == val for event in events_emg)}
    epochs = mne.Epochs(emg_rectified, events_emg, event_id=event_ids, tmin=tmin, tmax=tmax, baseline=baseline, preload=True)
    epochs.save(fif_path / f'{sub_input}_condition_{condition}_{index}_{target}-epo.fif', overwrite=True)
    return epochs


# Create a function to save results to CSV for all blocks combined
def save_results_to_csv(all_target_counts, all_distractor_counts):
    csv_filename = f"{results_path}/emg_classifications_{sub_input}_{condition}.csv"

    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['Condition', 'Type', 'True Response', 'Partial Response', 'No Response']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()  # Write the header

        # Write the accumulated results for the target epochs
        writer.writerow({
            'Condition': condition,
            'Type': 'Target',
            'True Response': all_target_counts.get('True Response', 0),
            'Partial Response': all_target_counts.get('Partial Response', 0),
            'No Response': all_target_counts.get('No Response', 0)
        })

        # Write the accumulated results for the distractor epochs
        writer.writerow({
            'Condition': condition,
            'Type': 'Distractor',
            'True Response': all_distractor_counts.get('True Response', 0),
            'Partial Response': all_distractor_counts.get('Partial Response', 0),
            'No Response': all_distractor_counts.get('No Response', 0)
        })

    print(f"Results saved to {csv_filename}")


if __name__ == '__main__':
    # matplotlib.use('TkAgg')
    sub_input = input("Give sub number as subn (n for number): ")
    sub = [sub.strip() for sub in sub_input.split(',')]
    cm = 1 / 2.54
    # 0. LOAD THE DATA
    sub_dirs = []
    for subs in sub:
        default_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data')
        data_dir = default_dir / 'eeg' / 'raw'  # to get the raw EEG files (which will we 'convert' to EMG
        sub_dir = data_dir / subs
        emg_dir = default_dir / 'emg' / subs # creating a folder dedicated to EMG files we will create
        results_path = emg_dir / 'preprocessed' / 'results'  # where results will be saved
        fig_path = emg_dir / 'preprocessed' / 'figures' # where figures from pre-processing and classification will be stored
        erp_path = fig_path / 'ERPs'
        z_figs = fig_path / 'z_distributions'
        class_figs = fig_path / 'classifications'
        df_path = default_dir / 'performance' / subs / 'tables'  # csv tables containing important information from vmrk files
        sub_dirs.append(sub_dir)
        json_path = default_dir / 'misc'
        psd_path = fig_path / 'spectral_power'
        fif_path = emg_dir / 'fif files'  # pre-processed .eeg files of the EMG data will be stored
        combined_epochs = fif_path / 'combined_epochs'
        for folder in sub_dir, fif_path, results_path, fig_path, erp_path, z_figs, class_figs, df_path, combined_epochs, psd_path:
            if not os.path.isdir(folder):
                os.makedirs(folder)

    # Load necessary JSON files
    with open(json_path / "preproc_config.json") as file:
        cfg = json.load(file)
    with open(json_path / "electrode_names.json") as file:
        mapping = json.load(file)
    with open(json_path / 'eeg_events.json') as file:
        markers_dict = json.load(file)

    s1_events = markers_dict['s1_events'] # stimulus 1 markers
    s2_events = markers_dict['s2_events']  # stimulus 2 markers
    response_events = markers_dict['response_events']  # response markers

    # mapping events needed for creating epochs with only target-number events, for target, distractor AND response epochs
    mapping_path = json_path / 'events_mapping.json'
    with open(mapping_path, 'r') as file:
        events_mapping = json.load(file)
    s1_mapping = events_mapping[0]
    s2_mapping = events_mapping[1]
    response_mapping = events_mapping[2]
    # conditions: a1, a2, e1 or e2
    condition = input('Please provide condition (exp. EEG): ')  # choose a condition of interest for processing
    ### Get target header files for all participants
    target_header_files_list = []
    for sub_dir in sub_dirs:
        header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
        filtered_files = [file for file in header_files if condition in file]
        if filtered_files:
            target_header_files_list.append(filtered_files)
    # TO CALCULATE SNR:
    for files in sub_dir.iterdir():
        if files.is_file and 'baseline.vhdr' in files.name:
            baseline = mne.io.read_raw_brainvision(files, preload=True)

    baseline.rename_channels(mapping)
    baseline.set_montage('standard_1020')
    baseline = baseline.pick_channels(['A2', 'M2', 'A1'])  # select EMG-relevant files
    baseline.set_eeg_reference(ref_channels=['A1'])  # set correct reference channel
    baseline = mne.set_bipolar_reference(baseline, anode='A2', cathode='M2', ch_name='EMG_bipolar')  # change ref to EMG bipolar
    baseline.drop_channels(['A1'])  # drop reference channel (don't need to see it)
    # pre-process baseline:
    baseline_filt = filter_emg(baseline)
    baseline_rect = baseline_filt.copy()
    baseline_rectified = rectify(baseline_rect)
    # Calculate the number of samples in the recording
    '''
    - epoch_length_s: Total length of each epoch in seconds.
    - tmin: Start of the epoch in seconds relative to each synthetic event.
    - tmax: End of the epoch in seconds relative to each synthetic event.
    '''
    n_samples = baseline_rectified.n_times
    duration_s = n_samples / 500
    epoch_length_s = 1.1
    tmin = -0.2
    tmax = 0.9
    # Calculate the step size in samples for each epoch (based on the epoch length)
    step_size = int(epoch_length_s * 500)
    # Generate synthetic events spaced at regular intervals across the continuous data
    # The first column is the sample index, second is filler (e.g., 0), third is the event ID (e.g., 1)
    events = np.array([[i, 0, 1] for i in range(0, n_samples - step_size, step_size)])

    # Create epochs based on these synthetic events
    epochs_baseline = mne.Epochs(baseline_rectified, events, event_id={'arbitrary': 1}, tmin=tmin, tmax=tmax, baseline=None, preload=True)
    epochs_baseline_erp = epochs_baseline.average().plot()
    epochs_baseline_erp.savefig(erp_path / f'{sub_input}_baseline_ERP.png')
    plt.close(epochs_baseline_erp)
    # # Assuming you have response_data and baseline_data from your epochs
    # response_data = combined_response_epochs.get_data()  # Get data from response epochs
    # baseline_data = combined_non_target_stim.get_data()  # Get data from baseline epochs

    all_response_epochs = []
    all_target_response_epochs = []
    all_target_no_response_epochs = []
    all_distractor_response_epochs = []
    all_distractor_no_response_epochs = []
    all_non_target_stim_epochs = []
    all_invalid_non_target_epochs = []
    all_invalid_distractor_epochs = []
    all_invalid_target_epochs = []

    ### Process Each File under the Selected Condition ###
    for index in range(len(target_header_files_list[0])):
        # Loop through all the files matching the condition
        emg_file = target_header_files_list[0][index]
        full_path = os.path.join(sub_dir, emg_file)
        emg = mne.io.read_raw_brainvision(full_path, preload=True)

        # Set montage for file:
        emg.rename_channels(mapping)
        emg.set_montage('standard_1020')
        emg = emg.pick_channels(['A2', 'M2', 'A1']) # select EMG-relevant files
        emg.set_eeg_reference(ref_channels=['A1']) # set correct reference channel
        emg = mne.set_bipolar_reference(emg, anode='A2', cathode='M2', ch_name='EMG_bipolar')  # change ref to EMG bipolar
        emg.drop_channels(['A1'])  # drop reference channel (don't need to see it)

        # Get Target Block information
        csv_path = default_dir / 'params' / 'block_sequences' / f'{sub_input}.csv'
        csv = pd.read_csv(csv_path)
        # get target stream, blocks for selected condition, axis relevant to condition (az or ele)
        target_stream, distractor_stream, target_blocks, axis, target_mapping, distractor_mapping = get_target_blocks()

        # Define target and distractor events based on target number and stream
        target_block = [target_block for target_block in target_blocks.values[index]]
        target_events, distractor_events = define_events(target_stream)
        combined_events = {**distractor_events, **target_events}
        # Get events for EMG analysis, based on filtered target and distractor, and response events
        targets_emg_events = create_events(target_events, target_mapping)
        distractors_emg_events = create_events(distractor_events, distractor_mapping)
        responses_emg_events = create_events(response_events, response_mapping)

        # Filter and rectify the EMG data
        emg_filt = filter_emg(emg)
        emg_rect = emg_filt.copy()
        emg_rectified = rectify(emg_rect)
        target_emg_rectified = rectify(emg_filt)
        distractor_emg_rectified = rectify(emg_filt)
        responses_emg_rectified = rectify(emg_filt)

        # # epoch target data: with responses, without:
        # # get response epochs separately first:
        # response_epochs = epochs(response_events, responses_emg_rectified, responses_emg_events, target='responses', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))

        # Create non-target Stimuli Epochs:
        # for distractor:
        non_target_events_distractor = baseline_events(distractor_events, distractor_mapping)
        # for target:
        non_target_events_target = baseline_events(target_events, target_mapping)

        # Initialize the lists for categorization
        target_response_events = []
        target_no_response_events = []
        distractor_response_events = []
        distractor_no_response_events = []
        non_target_stimulus_events = []
        invalid_non_target_response_events = []
        invalid_target_response_events = []
        invalid_distractor_response_events = []
        invalid_response_epochs = []
        response_only_epochs = []  # Pure response events not linked to other events

        # Track which response events have been assigned (initialize as NumPy array)
        unassigned_response_events = responses_emg_events  # This should already be a NumPy array

        # Step 4: Process each target event
        for event in targets_emg_events:
            stim_timepoint = event[0] / 500  # Replace 500 with your actual sampling frequency if different
            time_start = stim_timepoint  # Start of the time window for a target event
            time_end = stim_timepoint + 0.9  # End of the time window for a target event

            # Find responses that fall within this target's time window
            response_found = False
            for response_event in unassigned_response_events:
                response_timepoint = response_event[0] / 500
                if time_start < response_timepoint < time_end:
                    if response_timepoint - stim_timepoint < 0.2:
                        invalid_target_response_events.append(np.append(event, 'target'))  # Classify as invalid response
                    else:
                        # If a valid response is found within the time window, assign it to target_response_events
                        target_response_events.append(event)

                    # Remove the response_event row using np.delete
                    idx_to_remove = np.where((unassigned_response_events == response_event).all(axis=1))[0][0]
                    unassigned_response_events = np.delete(unassigned_response_events, idx_to_remove, axis=0)

                    response_found = True
                    break  # Stop after finding the first response (adjust if multiple responses per target are needed)

            if not response_found:
                # If no response is found, add the target event to target_no_response_events
                target_no_response_events.append(event)
        # Step 5: Process leftover responses for distractor events
        for event in distractors_emg_events:
            stim_timepoint = event[0] / 500
            time_start = stim_timepoint
            time_end = stim_timepoint + 0.9

            # Check for unassigned responses that fall within the distractor's time window
            response_found = False
            for response_event in unassigned_response_events:
                response_timepoint = response_event[0] / 500
                if time_start < response_timepoint < time_end:
                    if response_timepoint - stim_timepoint < 0.2:
                        invalid_distractor_response_events.append(np.append(event, 'distractor'))  # Classify as invalid response
                    else:
                        # If a valid response is found within the time window, assign it to target_response_events
                        distractor_response_events.append(event)

                    idx_to_remove = np.where((unassigned_response_events == response_event).all(axis=1))[0][0]
                    unassigned_response_events = np.delete(unassigned_response_events, idx_to_remove, axis=0)
                    response_found = True
                    break

            if not response_found:
                # If no response is found, add the distractor to no_response list
                distractor_no_response_events.append(event)

        # Step 6: Process non-target events for any remaining responses
        non_target_combined_events = np.concatenate([non_target_events_target, non_target_events_distractor])
        non_target_combined_events = non_target_combined_events[np.argsort(non_target_combined_events[:, 0])]
        for event in non_target_combined_events:
            stim_timepoint = event[0] / 500
            time_start = stim_timepoint
            time_end = stim_timepoint + 0.9

            response_found = False
            for response_event in unassigned_response_events:
                response_timepoint = response_event[0] / 500
                if time_start < response_timepoint < time_end:
                    # If response falls in a non-target window, add to invalid responses
                    invalid_non_target_response_events.append(np.append(event, 'non-target'))
                    idx_to_remove = np.where((unassigned_response_events == response_event).all(axis=1))[0][0]
                    unassigned_response_events = np.delete(unassigned_response_events, idx_to_remove, axis=0)
                    response_found = True
                    break

            if not response_found:
                non_target_stimulus_events.append(event)

        # Step 7: Remaining responses are pure response events (response_only_epochs)
        response_only_epochs = unassigned_response_events  # should be empty
        invalid_non_target_events = [np.array(event[:3], dtype=int) for event in invalid_non_target_response_events]
        invalid_target_events = [np.array(event[:3], dtype=int) for event in invalid_target_response_events]
        invalid_distractor_events = [np.array(event[:3], dtype=int) for event in invalid_distractor_response_events]


        # now get target epochs with responses:
        if target_response_events:
            target_response_epochs = epochs(target_events, target_emg_rectified, target_response_events, target='target_with_responses', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))
            target_response_data = target_response_epochs.get_data() # save target responses data
        # target epochs without responses:
        if target_no_response_events and len(target_no_response_events) > 0:
            target_no_responses_epochs = epochs(target_events, target_emg_rectified,
                                                target_no_response_events, target='target_without_responses',
                                                tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))
            # target_no_responses_data = target_no_responses_epochs.get_data()
        # now distractor stimuli:
        # with responses:
        if distractor_response_events and len(distractor_response_events) > 0:
            distractor_responses_epochs = epochs(distractor_events, distractor_emg_rectified,
                                                 distractor_response_events, target='distractor_with_responses',
                                                 tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))
        if distractor_no_response_events:
            distractor_no_responses_epochs = epochs(distractor_events, distractor_emg_rectified,
                                                 distractor_no_response_events, target='distractor_without_responses',
                                                    tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))
            distractor_no_responses_data = distractor_no_responses_epochs.get_data()

        # now for non-target stimuli:
        if non_target_stimulus_events and len(non_target_stimulus_events) > 0:
            non_target_stim_epochs = epochs(combined_events, emg_rect, non_target_stimulus_events, target='non_target_stimuli', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))
            non_target_stim_data = non_target_stim_epochs.get_data()


        if invalid_non_target_events and len(invalid_non_target_events) > 0:
            invalid_non_target_epochs = epochs(combined_events, responses_emg_rectified, invalid_non_target_events, target='invalid_responses', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))

        if invalid_target_events and len(invalid_target_events) > 0:
            invalid_target_epochs = epochs(combined_events, responses_emg_rectified, invalid_target_events, target='invalid_target_responses', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))

        if invalid_distractor_events and len(invalid_distractor_events) > 0:
            invalid_distractor_epochs = epochs(combined_events, responses_emg_rectified, invalid_distractor_events,
                                           target='invalid_distractor_responses', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))


        # if 'response_epochs' in locals() and response_epochs is not None:
        #     all_response_epochs.append(response_epochs)
        if 'target_response_epochs' in locals() and target_response_epochs is not None:
            all_target_response_epochs.append(target_response_epochs)
        if 'target_no_responses_epochs' in locals() and target_no_responses_epochs is not None:
            all_target_no_response_epochs.append(target_no_responses_epochs)
        if 'distractor_responses_epochs' in locals() and distractor_responses_epochs is not None:
            all_distractor_response_epochs.append(distractor_responses_epochs)
        if 'distractor_no_responses_epochs' in locals() and distractor_no_responses_epochs is not None:
            all_distractor_no_response_epochs.append(distractor_no_responses_epochs)
        if 'invalid_response_epochs' in locals() and invalid_response_epochs is not None:
            all_invalid_non_target_epochs.append(invalid_non_target_epochs)
        if 'invalid_target_epochs' in locals() and invalid_target_epochs is not None:
            all_invalid_target_epochs.append(invalid_target_epochs)
        if 'invalid_distractor_epochs' in locals() and invalid_distractor_epochs is not None:
            all_invalid_distractor_epochs.append(invalid_distractor_epochs)
        if 'baseline_epochs' in locals() and baseline_epochs is not None:
            all_non_target_stim_epochs.append(non_target_stim_epochs)


    def is_valid_epoch_list(epoch_list):
        return [epochs for epochs in epoch_list if isinstance(epochs, mne.Epochs) and len(epochs) > 0]

    if all_target_response_epochs:
        valid_target_response_epochs = is_valid_epoch_list(all_target_response_epochs)
        if valid_target_response_epochs:
            combined_target_response_epochs = mne.concatenate_epochs(valid_target_response_epochs)
            combined_target_response_epochs.save(
                combined_epochs / f'{sub_input}_condition_{condition}_combined_target_response_epochs-epo.fif',
                overwrite=True)
            target_response_erp = combined_target_response_epochs.average().plot()
            target_response_erp.savefig(erp_path / f'{sub_input}_condition_{condition}_target_response_erp.png')
            response_data = combined_target_response_epochs.get_data(copy=True)
            combined_target_response_events = combined_target_response_epochs.events
            plt.close(target_response_erp)

    if all_target_no_response_epochs:
        valid_target_no_response_epochs = is_valid_epoch_list(all_target_no_response_epochs)
        if valid_target_no_response_epochs:
            combined_target_no_response_epochs = mne.concatenate_epochs(valid_target_no_response_epochs)
            combined_target_no_response_epochs.save(
                combined_epochs / f'{sub_input}_condition_{condition}_combined_no_response_epochs-epo.fif',
                overwrite=True)
            target_no_response_erp = combined_target_no_response_epochs.average().plot()
            target_no_response_erp.savefig(erp_path / f'{sub_input}_condition_{condition}_target_no_response_erp.png')
            target_no_response_data = combined_target_no_response_epochs.get_data(copy=True)
            combined_target_no_response_events = combined_target_no_response_epochs.events
            plt.close(target_no_response_erp)

    if all_distractor_response_epochs:
        valid_distractor_response_epochs = is_valid_epoch_list(all_distractor_response_epochs)
        if valid_distractor_response_epochs:
            combined_distractor_response_epochs = mne.concatenate_epochs(valid_distractor_response_epochs)
            combined_distractor_response_epochs.save(
                combined_epochs / f'{sub_input}_condition_{condition}_combined_distractor_response_epochs-epo.fif',
                overwrite=True)
            distractor_response_erp = combined_distractor_response_epochs.average().plot()
            distractor_response_erp.savefig(erp_path / f'{sub_input}_condition_{condition}_distractor_response_erp.png')
            distractor_responses_data = combined_distractor_response_epochs.get_data(copy=True)
            combined_distractor_responses_events = combined_distractor_response_epochs.events
            plt.close(distractor_response_erp)

    if all_distractor_no_response_epochs:
        valid_distractor_no_response_epochs = is_valid_epoch_list(all_distractor_no_response_epochs)
        if valid_distractor_no_response_epochs:
            combined_distractor_no_response_epochs = mne.concatenate_epochs(valid_distractor_no_response_epochs)
            combined_distractor_no_response_epochs.save(
                combined_epochs / f'{sub_input}_condition_{condition}_combined_distractor_no_response_epochs-epo.fif',
                overwrite=True)
            distractor_no_response_erp = combined_distractor_no_response_epochs.average().plot()
            distractor_no_response_erp.savefig(
                erp_path / f'{sub_input}_condition_{condition}_distractor_no_response_erp.png')
            plt.close(distractor_no_response_erp)
            distractor_data = combined_distractor_no_response_epochs.get_data(copy=True)
            combined_distractor_no_response_events = combined_distractor_no_response_epochs.events

    if all_non_target_stim_epochs:
        valid_non_target_stim_epochs = is_valid_epoch_list(all_non_target_stim_epochs)
        if valid_non_target_stim_epochs:
            combined_non_target_stim = mne.concatenate_epochs(valid_non_target_stim_epochs)
            combined_non_target_stim.save(
                combined_epochs / f'{sub_input}_condition_{condition}_combined_non_target_stim_epochs-epo.fif',
                overwrite=True)
            non_target_stim_erp = combined_non_target_stim.average().plot()
            non_target_stim_erp.savefig(erp_path / f'{sub_input}_condition_{condition}_non_target_stim_erp.png')
            plt.close(non_target_stim_erp)
            non_target_data = combined_non_target_stim.get_data(copy=True)
            combined_non_target_events = combined_non_target_stim.events

    if all_invalid_non_target_epochs:
        valid_invalid_non_target_epochs = is_valid_epoch_list(all_invalid_non_target_epochs)
        if valid_invalid_non_target_epochs:
            combined_invalid_non_target_epochs = mne.concatenate_epochs(valid_invalid_non_target_epochs)
            combined_invalid_non_target_epochs.save(
                combined_epochs / f'{sub_input}_condition_{condition}_combined_invalid_response_epochs-epo.fif',
                overwrite=True)
            invalid_non_target_erp = combined_invalid_non_target_epochs.average().plot()
            invalid_non_target_erp.savefig(erp_path / f'{sub_input}_condition_{condition}_invalid_response_erp.png')
            invalid_non_target_data = combined_invalid_non_target_epochs.get_data(copy=True)
            combined_invalid_non_target_events = combined_invalid_non_target_epochs.events
            plt.close(invalid_non_target_erp)

    if all_invalid_target_epochs:
        valid_invalid_target_epochs = is_valid_epoch_list(all_invalid_target_epochs)
        if valid_invalid_target_epochs:
            combined_invalid_target_epochs = mne.concatenate_epochs(valid_invalid_target_epochs)
            combined_invalid_target_epochs.save(
                combined_epochs / f'{sub_input}_condition_{condition}_combined_invalid_response_epochs-epo.fif',
                overwrite=True)
            invalid_target__erp = combined_invalid_target_epochs.average().plot()
            invalid_target__erp.savefig(erp_path / f'{sub_input}_condition_{condition}_invalid_response_erp.png')
            invalid_target_data = combined_invalid_target_epochs.get_data(copy=True)
            combined_invalid_target_events = combined_invalid_target_epochs.events
            plt.close(invalid_target__erp)
            
    if all_invalid_distractor_epochs:
        valid_invalid_distractor_epochs = is_valid_epoch_list(all_invalid_target_epochs)
        if valid_invalid_distractor_epochs:
            combined_invalid_distractor_epochs = mne.concatenate_epochs(valid_invalid_distractor_epochs)
            combined_invalid_distractor_epochs.save(
                combined_epochs / f'{sub_input}_condition_{condition}_combined_invalid_response_epochs-epo.fif',
                overwrite=True)
            invalid_distractor_erp = combined_invalid_distractor_epochs.average().plot()
            invalid_distractor_erp.savefig(erp_path / f'{sub_input}_condition_{condition}_invalid_response_erp.png')
            invalid_distractor_data = combined_invalid_distractor_epochs.get_data(copy=True)
            combined_invalid_distractor_events = combined_invalid_distractor_epochs.events
            plt.close(invalid_distractor_erp)

    all_invalid_events = []
    if 'invalid_target_response_events' in locals() and len(invalid_target_response_events) > 0:
        all_invalid_events.append(invalid_target_response_events)
    if 'invalid_distractor_response_events' in locals() and len(invalid_distractor_response_events) > 0:
        all_invalid_events.append(invalid_distractor_response_events)
    if 'invalid_non_target_response_events' in locals() and len(invalid_non_target_response_events) > 0:
        all_invalid_events.append(invalid_non_target_response_events)
    flattened_list = []
    for sublist in all_invalid_events:
        for event in sublist:
            flattened_list.append(event)
    all_invalid_events = np.concatenate(all_invalid_events)
    all_invalid_events = pd.DataFrame(all_invalid_events)
    all_invalid_events = all_invalid_events.drop(columns=1)
    all_invalid_events.columns = ['Timepoints', 'Stimulus', 'Type']
    all_invalid_events['Timepoints'] = all_invalid_events['Timepoints'].astype(int)
    all_invalid_events['Timepoints'] = all_invalid_events['Timepoints'] / 500
    all_invalid_events['Stimulus'] = all_invalid_events['Stimulus'].astype(int)
    all_invalid_events = all_invalid_events.sort_values(by='Timepoints')
    combined_invalid_events = all_invalid_events


    def baseline_normalization(emg_epochs, tmin=0.2, tmax=0.9):
        """Calculate the derivative and z-scores for the baseline period."""
        emg_epochs.load_data()  # Ensure data is available
        emg_window = emg_epochs.copy().crop(tmin, tmax)  # Crop to window of interest
        emg_data = emg_window.get_data(copy=True)  # Extract data as a NumPy array

        # Compute the derivative along the time axis
        emg_derivative = np.diff(emg_data, axis=-1)

        # Z-score normalization within each epoch (derivative normalized by its own mean/std)
        emg_derivative_z = (emg_derivative - np.mean(emg_derivative, axis=-1, keepdims=True)) / np.std(
            emg_derivative, axis=-1, keepdims=True)
        emg_var = np.var(emg_derivative_z, axis=-1)  # Variance of baseline
        emg_rms = np.sqrt(np.mean(np.square(emg_derivative_z), axis=-1))  # RMS of baseline

        return emg_data, emg_derivative, emg_derivative_z, emg_var, emg_rms


    # Apply baseline normalization to non-target epochs
    baseline_data, baseline_derivative, baseline_z_scores, baseline_var, baseline_rms = baseline_normalization(
        epochs_baseline, tmin=0.2, tmax=0.9)


    # Calculate global baseline mean and std
    baseline_mean = np.mean(baseline_derivative)
    baseline_std = np.std(baseline_derivative)


    def z_normalization(emg_data, baseline_mean, baseline_std):
        """Calculate z-scores using baseline statistics for normalization."""
        # Compute the derivative across the time axis
        emg_derivative = np.diff(emg_data, axis=-1)

        # Normalize using the global baseline mean and std
        emg_derivative_z = (emg_derivative - baseline_mean) / baseline_std

        # Calculate features: variance, RMS, and slopes
        emg_var = np.var(emg_derivative_z, axis=-1)
        emg_rms = np.sqrt(np.mean(np.square(emg_derivative_z), axis=-1))
        slopes = np.diff(emg_derivative_z, axis=-1)
        slope_mean = np.mean(slopes, axis=-1)
        slope_ratio = np.sum(slopes[slopes > 0]) / np.abs(np.sum(slopes[slopes < 0]))

        return emg_derivative, emg_derivative_z, emg_var, emg_rms, slope_mean, slope_ratio


    # Baseline:
    if 'epochs_baseline' in locals() and len(epochs_baseline.events) > 0:
        baseline_derivatives, baseline_z_scores, baseline_var, baseline_rms, baseline_slope_mean, baseline_slope_ratio = z_normalization(baseline_data, baseline_mean, baseline_std)
    # Readiness:
    if 'combined_non_target_stim' in locals() and len(combined_non_target_stim.events) > 0:
        readiness_derivatives, readiness_z_scores, readiness_var, readiness_rms, readiness_slope_mean, readiness_slope_ratio = z_normalization(
            non_target_data, baseline_mean, baseline_std)
    # Target Response:
    if 'combined_target_response_epochs' in locals() and len(combined_target_response_epochs.events) > 0:
        response_derivatives, response_z_scores, response_var, response_rms, response_slope_mean, response_slope_ratio = z_normalization(response_data, baseline_mean, baseline_std)
    # Distractors:
    if 'combined_distractor_no_response_epochs' in locals() and len(combined_distractor_no_response_epochs.events) > 0:
        distractor_derivatives, distractor_z_scores, distractor_var, distractor_rms, distractor_slope_mean, distractor_slope_ratio = z_normalization(
            distractor_data, baseline_mean, baseline_std)

    # Additionally:
    if 'combined_invalid_non_target_response_epochs' in locals() and len(combined_invalid_non_target_epochs.events) > 0:
        invalid_non_target_derivatives, invalid_non_target_z_scores, invalid_non_target_var, invalid_non_target_rms, invalid_non_target_slope_mean, invalid_non_target_slope_ratio = z_normalization(
            invalid_non_target_data, baseline_mean, baseline_std)

    if 'combined_invalid_target_response_epochs' in locals() and len(combined_invalid_target_epochs.events) > 0:
        invalid_target_derivatives, invalid_target_z_scores, invalid_target_var, invalid_target_rms, invalid_target_slope_mean, invalid_target_slope_ratio = z_normalization(
            invalid_target_data, baseline_mean, baseline_std)

    if 'combined_invalid_distractor_response_epochs' in locals() and len(combined_invalid_distractor_epochs.events) > 0:
        invalid_distractor_derivatives, invalid_distractor_z_scores, invalid_distractor_var, invalid_distractor_rms, invalid_distractor_slope_mean, invalid_distractor_slope_ratio = z_normalization(
            invalid_distractor_data, baseline_mean, baseline_std)
        
        
    if 'combined_target_no_response_epochs' in locals() and len(combined_target_no_response_epochs.events) >0:
        target_no_response_derivatives, target_no_response_z_scores, target_no_response_var, target_no_response_rms, target_no_response_slope_mean, target_no_response_slope_ratio = z_normalization(target_no_response_data, baseline_mean, baseline_std)

    if 'combined_distractor_response_epochs' in locals() and len(combined_distractor_response_epochs.events) > 0:
        distractor_response_derivatives, distractor_response_z_scores, distractor_response_var, distractor_response_rms, distractor_response_slope_mean, distractor_response_slope_ratio = z_normalization(distractor_responses_data, baseline_mean, baseline_std)
    

    # # Step 1: Compute the signal power (variance of the response data)
    def SNR(data, baseline_data):
        P_signal = np.mean(np.var(data, axis=-1))  # Variance of response data

        # Step 2: Compute the noise power (variance of the baseline data)
        P_noise = np.mean(np.var(baseline_data, axis=-1))  # Variance of baseline data

        # Step 3: Calculate the Signal-to-Noise Ratio (SNR)
        SNR = P_signal / P_noise

        print(f"SNR: {SNR}")
    SNR(non_target_data, baseline_data)
    SNR(response_data, baseline_data)
    SNR(distractor_data, baseline_data)

    # TFA:
    # Define frequency range of interest (from 1 Hz to 30 Hz)
    def tfa_heatmap(epochs, target):
        frequencies = np.logspace(np.log10(1), np.log10(30), num=30)  # Frequencies from 1 to 30 Hz
        n_cycles = np.minimum(frequencies / 2, 7)  # Number of cycles in Morlet wavelet (adapts to frequency)

        # Compute the Time-Frequency Representation (TFR) using Morlet wavelets
        epochs = epochs.copy().crop(tmin=-0.2, tmax=0.9)
        power = mne.time_frequency.tfr_morlet(epochs, freqs=frequencies, n_cycles=n_cycles, return_itc=False)

        # Plot TFR as a heatmap (power across time and frequency)
        power_plot = power.plot([0], mode='logratio', title='TFR (Heatmap)', show=True)
        for i, fig in enumerate(power_plot):
            fig.savefig(
                psd_path / f'{sub_input}_{condition}_{target}_plot.png')  # Save with a unique name for each figure
            plt.close(fig)
        return power
    response_power = tfa_heatmap(combined_target_response_epochs, target='target_response')
    baseline_power = tfa_heatmap(epochs_baseline, target='baseline')
    non_target_power = tfa_heatmap(combined_non_target_stim, target='non_target_stim')
    distractor_power = tfa_heatmap(combined_distractor_no_response_epochs, target='distractor')


    bands = {
        "low_band": (1, 10),
        "mid_band": (10, 20),
        "high_band": (20, 30)
    }

    def get_avg_band_power(power, bands, fmin, fmax, tmin=0.0, tmax=0.9):
        """Compute the average power for specified frequency bands within a time window."""
        time_mask = (power.times >= tmin) & (power.times <= tmax)  # Mask for the time window
        early_phase = (power.times >= 0.0) & (power.times <= 0.2)
        late_phase = (power.times >= 0.2) & (power.times <= 0.9)
        low_band = bands['low_band']
        middle_band = bands['mid_band']
        high_band = bands['high_band']
        
        # investigate entire epoch:
        for band_name, (fmin, fmax) in bands.items():
            freq_mask = (power.freqs >= fmin) & (power.freqs <= fmax)
            band_power = power.data[:, freq_mask, :][:, :, time_mask]  # Data within frequency band and time window
            early_band_power = power.data[:, freq_mask, :][:, :, early_phase]  # band powers in early phase
            late_band_power = power.data[:, freq_mask, :][:, :, late_phase]  # band powers in late phase

            # Compute the average power within the frequency band and time window
            avg_band_power = band_power.mean()  # Mean across all epochs, frequencies, and time within the window
            early_avg_band_power = early_band_power.mean()
            late_avg_band_power = late_band_power.mean()
            early_vs_late = late_avg_band_power / early_avg_band_power # overall avg powe trend of each epoch
        
        # investigate the three bands separately, and get ratios of early vs late phase:
        low_freq_mask = (power.freqs >= low_band[0]) & (power.freqs <= low_band[1])
        low_band_power = power.data[:, low_freq_mask, :][:, :, time_mask]  # Data within frequency band and time window
        low_early_band_power = power.data[:, low_freq_mask, :][:, :, early_phase] # band powers in early phase
        low_late_band_power = power.data[:, low_freq_mask, :][:, :, late_phase] # band powers in late phase

        # Compute the average power within the frequency band and time window
        low_avg_band_power = low_band_power.mean()  # Mean across all epochs, frequencies, and time within the window
        low_early_avg_band_power = low_early_band_power.mean()
        low_late_avg_band_power = low_late_band_power.mean()
        low_early_vs_late = low_late_avg_band_power / low_early_avg_band_power

        middle_freq_mask = (power.freqs >= middle_band[0]) & (power.freqs <= middle_band[1])
        middle_band_power = power.data[:, middle_freq_mask, :][:, :, time_mask]  # Data within frequency band and time window
        middle_early_band_power = power.data[:, middle_freq_mask, :][:, :, early_phase]  # band powers in early phase
        middle_late_band_power = power.data[:, middle_freq_mask, :][:, :, late_phase]  # band powers in late phase

        # Compute the average power within the frequency band and time window
        middle_avg_band_power = middle_band_power.mean()  # Mean across all epochs, frequencies, and time within the window
        middle_early_avg_band_power = middle_early_band_power.mean()
        middle_late_avg_band_power = middle_late_band_power.mean()
        middle_early_vs_late = middle_late_avg_band_power / middle_early_avg_band_power

        high_freq_mask = (power.freqs >= high_band[0]) & (power.freqs <= high_band[1])
        high_band_power = power.data[:, high_freq_mask, :][:, :, time_mask]  # Data within frequency band and time window
        high_early_band_power = power.data[:, high_freq_mask, :][:, :, early_phase]  # band powers in early phase
        high_late_band_power = power.data[:, high_freq_mask, :][:, :, late_phase]  # band powers in late phase

        # Compute the average power within the frequency band and time window
        high_avg_band_power = high_band_power.mean()  # Mean across all epochs, frequencies, and time within the window
        high_early_avg_band_power = high_early_band_power.mean()
        high_late_avg_band_power = high_late_band_power.mean()
        high_early_vs_late = high_late_avg_band_power / high_early_avg_band_power

        # find dominant band:
        bands_avg = {'low_band_avg': low_avg_band_power, 'mid_band_avg': middle_avg_band_power, 'high_band_avg': high_avg_band_power}
        dominant_band = max(bands_avg, key=bands_avg.get)
        # find dominant frequency in dominant band:
        if dominant_band == 'low_band_avg':
            mean_power_across_time = low_band_power.mean(axis=2)
            dominant_freq_index = mean_power_across_time.argmax(axis=1)
            dominant_freq = int(power.freqs[dominant_freq_index])
        elif dominant_band == 'mid_band_avg':
            mean_power_across_time = middle_band_power.mean(axis=2)
            dominant_freq_index = mean_power_across_time.argmax(axis=1)
            dominant_freq = int(power.freqs[dominant_freq_index])
        else:
            mean_power_across_time = high_band_power.mean(axis=2)
            dominant_freq_index = mean_power_across_time.argmax(axis=1)
            dominant_freq = int(power.freqs[dominant_freq_index])
        vals = {'dominant band': dominant_band,
                'dominant freq': dominant_freq,
                'avg power': avg_band_power,
                'low avg power': low_avg_band_power,
                'mid avg power': middle_avg_band_power,
                'high avg power': high_avg_band_power,
                'LTER': early_vs_late,
                'low LTER': low_early_vs_late,
                'mid LTER': middle_early_vs_late,
                'high LTER': high_early_vs_late}

        return vals


    response_vals = get_avg_band_power(response_power, bands, fmin=1, fmax=30)
    distractor_vals = get_avg_band_power(distractor_power, bands, fmin=1, fmax=30)
    non_target_vals = get_avg_band_power(non_target_power, bands, fmin=1, fmax=30)
    baseline_vals = get_avg_band_power(baseline_power, bands, fmin=1, fmax=30)


    def get_ref_features(response_vals, non_target_vals, distractor_vals):
        """
        Calculate reference features from response, non-target, distractor, and baseline values.
        """

        # Extract values from response, non-target, distractor, and baseline dictionaries
        response_low, response_mid, response_high = response_vals['low avg power'], response_vals['mid avg power'], \
                                                    response_vals['high avg power']
        non_target_low, non_target_mid, non_target_high = non_target_vals['low avg power'], non_target_vals[
            'mid avg power'], non_target_vals['high avg power']
        distractor_low, distractor_mid, distractor_high = distractor_vals['low avg power'], distractor_vals[
            'mid avg power'], distractor_vals['high avg power']

        # Create the features dictionary with the specified values from each band
        features = {
            'dominant_band': [
                {'response': response_vals['dominant band']},
                {'non_target': non_target_vals['dominant band']},
                {'distractor': distractor_vals['dominant band']}
            ],
            'dominant_freq': [
                {'response': response_vals['dominant freq']},
                {'non_target': non_target_vals['dominant freq']},
                {'distractor': distractor_vals['dominant freq']}
            ],
            'overall_avg_power': [
                {'response': response_vals['avg power']},
                {'non_target': non_target_vals['avg power']},
                {'distractor': distractor_vals['avg power']}
            ],
            'low_band': {
                'avg power': [
                    {'response': response_low},
                    {'non_target': non_target_low},
                    {'distractor': distractor_low}
                ],
                'LTER': [
                    {'response': response_vals['low LTER']},
                    {'non_target': non_target_vals['low LTER']},
                    {'distractor': distractor_vals['low LTER']}
                ]
            },
            'mid_band': {
                'avg power': [
                    {'response': response_mid},
                    {'non_target': non_target_mid},
                    {'distractor': distractor_mid}
                ],
                'LTER': [
                    {'response': response_vals['mid LTER']},
                    {'non_target': non_target_vals['mid LTER']},
                    {'distractor': distractor_vals['mid LTER']}
                ]
            },
            'high_band': {
                'avg power': [
                    {'response': response_high},
                    {'non_target': non_target_high},
                    {'distractor': distractor_high}
                ],
                'LTER': [
                    {'response': response_vals['high LTER']},
                    {'non_target': non_target_vals['high LTER']},
                    {'distractor': distractor_vals['high LTER']}
                ]
            }
        }

        return features


    # Call with the correct argument order:
    features = get_ref_features(response_vals, non_target_vals, distractor_vals)

    print(features)


    def adaptive_thresholds(features):
        """
        Calculate adaptive thresholds for each band and metric from features.
        """

        # Initialize dictionaries to store thresholds for each band and condition
        response_threshold = {}
        non_target_threshold = {}
        distractor_threshold = {}

        # Loop through each feature set to handle dominant_band, dominant_freq, overall_avg_power, and each band
        for feature, values in features.items():
            if feature in ['dominant_band', 'dominant_freq', 'overall_avg_power']:
                # For these overall metrics, directly set thresholds
                response_threshold[feature] = next(item['response'] for item in values if 'response' in item)
                non_target_threshold[feature] = next(item['non_target'] for item in values if 'non_target' in item)
                distractor_threshold[feature] = next(item['distractor'] for item in values if 'distractor' in item)
            else:
                # For band-specific metrics (low_band, mid_band, high_band)
                response_band_threshold = {}
                non_target_band_threshold = {}
                distractor_band_threshold = {}

                # Extract avg power and LTER metrics within each band
                for metric, metric_values in values.items():
                    response_value = next(item['response'] for item in metric_values if 'response' in item)
                    non_target_value = next(item['non_target'] for item in metric_values if 'non_target' in item)
                    distractor_value = next(item['distractor'] for item in metric_values if 'distractor' in item)

                    # Set thresholds for each metric in the current band
                    response_band_threshold[metric] = response_value
                    non_target_band_threshold[metric] = non_target_value
                    distractor_band_threshold[metric] = distractor_value

                # Assign thresholds for each specific band metric
                response_threshold[feature] = response_band_threshold
                non_target_threshold[feature] = non_target_band_threshold
                distractor_threshold[feature] = distractor_band_threshold

        return response_threshold, non_target_threshold, distractor_threshold


    response_threshold, non_target_threshold, distractor_threshold = adaptive_thresholds(features)


    def epochs_vals(epochs):
        frequencies = np.logspace(np.log10(1), np.log10(30), num=30)
        n_cycles = frequencies / 2

        # Define frequency bands
        bands = {
            'low_band': (1, 10),
            'mid_band': (10, 20),
            'high_band': (20, 30)
        }

        # Initialize list to store power and metrics for each epoch
        epochs_vals = []

        # Loop over each epoch and compute TFA independently
        for epoch_index in range(len(epochs)):
            # Extract single epoch as an Epochs object for TFR computation
            epoch_data = epochs.get_data(copy=True)[epoch_index][np.newaxis, :, :]
            epoch_info = epochs.info
            single_epoch = mne.EpochsArray(epoch_data, epoch_info, tmin=epochs.tmin)

            # Compute TFR for the individual epoch
            power = mne.time_frequency.tfr_morlet(single_epoch, freqs=frequencies, n_cycles=n_cycles, return_itc=False)

            # Mask for early and late phases
            time_mask = (power.times >= 0.0) & (power.times <= 0.9)
            early_phase = (power.times >= 0.0) & (power.times <= 0.2)
            late_phase = (power.times >= 0.2) & (power.times <= 0.9)

            # Dictionary to store metrics for each band in the current epoch
            epoch_vals = {}
            bands_avg = {}

            # Iterate through each frequency band and calculate metrics
            for band_name, (fmin, fmax) in bands.items():
                freq_mask = (frequencies >= fmin) & (frequencies <= fmax)
                band_power = power.data[:, freq_mask, :][:, :, time_mask]

                # Calculate average power across the time dimension
                avg_band_power = band_power.mean()  # Average power over time
                early_band_power = power.data[:, freq_mask, :][:, :, early_phase].mean()
                late_band_power = power.data[:, freq_mask, :][:, :, late_phase].mean()
                early_vs_late_ratio = late_band_power / early_band_power  # Ratio for phase power changes

                # Store the average power for later determination of dominant band
                bands_avg[band_name] = avg_band_power

                # Store values in dictionary for each band
                epoch_vals[band_name] = {
                    'avg power': avg_band_power,
                    'early power': early_band_power,
                    'late power': late_band_power,
                    'early_vs_late_ratio': early_vs_late_ratio
                }

            # Determine the dominant band based on the highest avg power across all bands
            dominant_band = max(bands_avg, key=bands_avg.get)
            dominant_band_freq_mask = (frequencies >= bands[dominant_band][0]) & (
                    frequencies <= bands[dominant_band][1]
            )
            dominant_band_power = power.data[:, dominant_band_freq_mask, :][:, :, time_mask]

            # Find the dominant frequency within the dominant band
            mean_power_across_time = dominant_band_power.mean(axis=2)
            dominant_freq_index = mean_power_across_time.argmax(axis=1)
            dominant_freq = frequencies[dominant_band_freq_mask][dominant_freq_index[0]]

            # Correct calculation for overall LTER across the entire epoch
            overall_early_power = power.data[:, :, early_phase].mean()
            overall_late_power = power.data[:, :, late_phase].mean()
            overall_lter = overall_late_power / overall_early_power

            # Append overall metrics and dominant band/freq to epoch values
            epoch_vals.update({
                'overall_avg_power': sum(bands_avg.values()) / len(bands_avg),
                'LTER': overall_lter,
                'dominant_band': dominant_band,
                'dominant_freq': dominant_freq
            })

            # Append metrics for the current epoch
            epochs_vals.append(epoch_vals)

        return epochs_vals

        # Epoch the EMG data for classification based on features:
        target_epochs = epochs(target_events, target_emg_rectified, targets_emg_events, target='target', tmin=-0.2,
                               tmax=0.9, baseline=(-0.2, 0.0))
        distractor_epochs = epochs(distractor_events, distractor_emg_rectified, distractors_emg_events,
                                   target='distractor', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))

        target_epochs_vals = epochs_vals(target_epochs)
        distractor_epochs_vals = epochs_vals(distractor_epochs)
        non_target_stim_vals = epochs_vals(combined_non_target_stim)

        from collections import Counter

        def count_dominant_bands_and_frequencies(epochs_vals):
            # Extract dominant bands and dominant frequencies from each epoch
            dominant_bands = [epoch['dominant_band'] for epoch in epochs_vals]
            dominant_frequencies = [epoch['dominant_freq'] for epoch in epochs_vals]

            # Count occurrences of each dominant band
            band_counts = Counter(dominant_bands)
            low_band_count = band_counts.get('low_band', 0)
            mid_band_count = band_counts.get('mid_band', 0)
            high_band_count = band_counts.get('high_band', 0)

            # Count occurrences of each dominant frequency
            freq_counts = Counter(dominant_frequencies)

            # Print dominant band counts
            print("Dominant Band Counts:")
            print({
                'low_band': low_band_count,
                'mid_band': mid_band_count,
                'high_band': high_band_count
            })

            # Print dominant frequency counts
            print("Dominant Frequency Counts:")
            for freq, count in sorted(freq_counts.items()):
                print(f"{freq} Hz: {count} times")

            # Return the counts for potential further use
            return {
                'band_counts': {
                    'low_band': low_band_count,
                    'mid_band': mid_band_count,
                    'high_band': high_band_count
                },
                'frequency_counts': dict(freq_counts)
            }

        # Example usage with target_epochs_vals
        dominant_counts = count_dominant_bands_and_frequencies(target_epochs_vals)
        dominant_counts = count_dominant_bands_and_frequencies(distractor_epochs_vals)
        dominant_counts = count_dominant_bands_and_frequencies(non_target_stim_vals)

        def classify_epochs(epochs_vals, response_threshold, distractor_threshold, non_target_threshold):
            classifications = []

            response_score = 0
            distractor_score = 0
            non_target_score = 0

            for epoch_vals in epochs_vals:
                overall_lter = epoch_vals['LTER'] # todo: check if overall LTER is in thresholds
                if overall_lter < 1:
                    non_target_score += 2
                if 1 < overall_lter < 2:
                    distractor_score += 2
                if overall_lter > 2:
                    response_score += 2
                dominant_band = epoch_vals['dominant_band']
                dominant_freq = epoch_vals['dominant_freq']
                # Unpack values for each band
                low_band_vals = epoch_vals['low_band']
                mid_band_vals = epoch_vals['mid_band']
                high_band_vals = epoch_vals['high_band']

                if dominant_band == 'low_band':
                    response_score += 2
                    distractor_score += 1
                    non_target_score += 2
                    band_lter = low_band_vals['early_vs_late_ratio']
                    band_avg_power = low_band_vals['avg power']
                    response_score += 2
                    non_target_score += 2
                    distractor_score += 2
                elif dominant_band == 'mid_band':
                    distractor_score += 2
                    response_score += 1
                    non_target_score += 1
                    band_lter = mid_band_vals['early_vs_late_ratio']
                    band_avg_power = mid_band_vals['avg power']
                else:
                    non_target_score += 2
                    distractor_score += 1
                    band_lter = high_band_vals['early_vs_late_ratio']
                    band_avg_power = high_band_vals['avg power']

                # Calculate differences for each threshold avg power
                diff_avg_power_response = abs(band_avg_power - response_threshold[f'{dominant_band}']['avg power'])
                diff_avg_power_distractor = abs(band_avg_power - distractor_threshold[f'{dominant_band}']['avg power'])
                diff_avg_power_non_target = abs(band_avg_power - non_target_threshold[f'{dominant_band}']['avg power'])

                diffs = {
                    'response': diff_avg_power_response,
                    'distractor': diff_avg_power_distractor,
                    'no response': diff_avg_power_non_target
                }

                # Find the key with the smallest difference
                smallest_diff_avg_power_label = min(diffs, key=diffs.get)
                if smallest_diff_avg_power_label == 'response':
                    response_score += 2
                elif smallest_diff_avg_power_label == 'distractor':
                    distractor_score += 2
                else:
                    non_target_score += 2

                diff_avg_power_response = abs(band_lter - response_threshold[f'{dominant_band}']['LTER'])
                diff_avg_power_distractor = abs(band_lter - distractor_threshold[f'{dominant_band}']['LTER'])
                diff_avg_power_non_target = abs(band_lter - non_target_threshold[f'{dominant_band}']['LTER'])

                lter_diffs = {'response': diff_avg_power_response, 'distractor': diff_avg_power_distractor, 'non-target': diff_avg_power_non_target}

                smallest_diff_lter_label = min(lter_diffs, key=lter_diffs.get)
                if smallest_diff_lter_label == 'response':
                    response_score += 2
                elif smallest_diff_lter_label == 'distractor':
                    distractor_score += 2
                else:
                    non_target_score += 2

                if dominant_freq == 1:
                    response_score += 2
                    non_target_score += 2
                    distractor_score += 1
                if 1 < dominant_freq < 11:
                    response_score += 2
                    distractor_score += 2
                    non_target_score += 1
                if 10 < dominant_freq < 20:
                    distractor_score += 2
                    response_score += 1
                    non_target_score += 1
                if 20 < dominant_freq < 30:
                    distractor_score += 1
                    non_target_score += 2

                # Determine classification based on the highest score
                if response_score > distractor_score and response_score > non_target_score:
                    classifications.append("Target")
                elif distractor_score > response_score and distractor_score > non_target_score:
                    classifications.append("Distractor")
                elif non_target_score > response_score and non_target_score > distractor_score:
                    classifications.append("Non-Target")
                else:
                    classifications.append("Uncertain")  # If scores are tied or unclear

            return classifications

        # Classify the different epochs
        targets_classifications = classify_epochs(target_epochs_vals, response_threshold, distractor_threshold, non_target_threshold)
        distractor_classifications = classify_epochs(distractor_epochs_vals, response_threshold, distractor_threshold, non_target_threshold)
        non_target_classifications = classify_epochs(non_target_stim_vals, response_threshold, distractor_threshold, non_target_threshold)

        # add labels:
        def add_labels(data, label, event_times, type):
            squeezed_data = data.squeeze(axis=1)  # (16, 551)
            data_df = pd.DataFrame(squeezed_data)
            data_df['Label'] = label
            data_df['Timepoints'] = event_times
            data_df['Type'] = type
            return data_df

        # Assuming you have arrays of event times for each type of event
        if 'combined_target_response_events' in locals():
            target_response_events = np.array(combined_target_response_events)
            target_response_times = target_response_events[:, 0] / 500  # Adjust 500 to your actual sampling rate
            target_response_df = add_labels(response_data, label='Response', event_times=target_response_times,
                                            type='target')

        if 'combined_distractor_no_response_events' in locals():
            distractor_no_response_events = np.array(combined_distractor_no_response_events)
            distractor_no_response_times = distractor_no_response_events[:, 0] / 500
            distractor_no_responses_df = add_labels(distractor_data, label='No Response',
                                                    event_times=distractor_no_response_times, type='distractor')

        if 'combined_invalid_events' in locals():
            combined_invalid_events['Label'] = 'invalid response'
            invalid_responses_df = combined_invalid_events

        if 'combined_target_no_response_events' in locals():
            target_no_response_events = np.array(combined_target_no_response_events)
            target_no_response_times = target_no_response_events[:, 0] / 500
            target_no_responses_df = add_labels(target_no_response_data, label='No Response',
                                                event_times=target_no_response_times, type='target')

        if 'combined_distractor_responses_events' in locals():
            distractor_response_events = np.array(combined_distractor_responses_events)
            distractor_response_times = distractor_response_events[:, 0] / 500
            distractor_responses_df = add_labels(distractor_responses_data, label='Response',
                                                 event_times=distractor_response_times, type='distractor')

        if 'combined_non_target_events' in locals():
            non_target_stimulus_events = np.array(combined_non_target_events)
            non_target_stim_times = non_target_stimulus_events[:, 0] / 500
            non_target_stim_df = add_labels(non_target_data, label='No Response', event_times=non_target_stim_times,
                                            type='non-target')

        # plot percentages of each response type:
        # calculate percentages for plotting:
        # total target and distractor stimuli
        total_target_count = len(target_epochs)
        total_distractor_count = len(distractor_epochs)
        # total true target responses:
        target_response_count = len(target_response_df)
        # total true distractor responses:
        if 'distractor_responses_df' in locals() and len(distractor_responses_df) > 0:
            distractor_response_count = len(distractor_responses_df)
        else:
            distractor_response_count = 0
        # total no-response target epochs:
        target_no_response_count = len(target_no_responses_df)
        # total target invalid response epochs:s
        target_invalid_response_count = len(invalid_target_response_events)

        # total no-response distractor epochs:
        distractor_no_response_count = len(distractor_no_responses_df)
        # invalid distractor response epochs:
        distractor_invalid_response_count = len(invalid_distractor_events)

        # total no-response non-target epochs:
        non_target_stimulus_count = len(non_target_stim_df)
        # total invalid non-target response epochs:
        invalid_non_target_response_count = len(invalid_non_target_response_events)
        # total non-target stimulus epochs:
        total_non_target_stim_count = non_target_stimulus_count + invalid_non_target_response_count

        # get percentages for performance:
        correct_responses = target_response_count * 100 / total_target_count
        distractor_responses = distractor_response_count * 100 / total_distractor_count
        # invalid target and distractor responses:
        invalid_target_responses = target_invalid_response_count * 100 / total_target_count
        invalid_distractor_responses = distractor_invalid_response_count * 100 / total_distractor_count
        # missed targets:
        misses_count = total_target_count - (target_response_count + target_invalid_response_count)
        missed_targets = misses_count * 100 / total_target_count

        invalid_non_target_responses = invalid_non_target_response_count * 100 / total_non_target_stim_count

        total_invalid_response_count = invalid_non_target_response_count + distractor_invalid_response_count + target_invalid_response_count
        # total percentage of all invalid responses, based on total sum of stimuli
        invalid_responses = total_invalid_response_count * 100 / (
                    total_target_count + total_distractor_count + total_non_target_stim_count)
        # total errors:
        target_errors = (target_invalid_response_count + misses_count) * 100 / total_target_count
        total_errors = (total_invalid_response_count + misses_count + distractor_response_count) * 100 / (
                    total_non_target_stim_count + total_target_count + total_distractor_count)

        # plot performance: correct responses, distractor, target misses, invalid target resp, total target error
        categories = ['Correct Target Responses', 'Distractor Responses', 'Targets Missed', 'Invalid Target Responses',
                      'Total Target Error']
        colors = ['blue', 'red', 'yellow', 'orange', 'black']
        values = [correct_responses, distractor_responses, missed_targets, invalid_target_responses, target_errors]
        plt.figure(figsize=(12, 6))
        plt.bar(categories, values, color=colors)
        plt.xlabel('Response Type')
        plt.ylabel('Performance (in %)')
        plt.title(f'{condition}_{index} performance of {sub_input}')
        plt.savefig(fig_path / f'{condition}_{index}_performance_{sub_input}.png')
        plt.close()


    # Concatenate all the DataFrames into one DataFrame
    # Initialize an empty list to hold DataFrames
    dataframes = []

    # Append each DataFrame to the list only if it exists
    if 'target_response_df' in locals():
        dataframes.append(target_response_df)
    if 'distractor_no_responses_df' in locals():
        dataframes.append(distractor_no_responses_df)
    if 'invalid_responses_df' in locals():
        dataframes.append(combined_invalid_events)
    if 'target_no_responses_df' in locals():
        dataframes.append(target_no_responses_df)
    if 'distractor_responses_df' in locals():
        dataframes.append(distractor_responses_df)
    if 'non_target_stim_df' in locals():
        dataframes.append(non_target_stim_df)







    # def plot_emg_derivative_z(emg_derivative_z, target):
    #             # Remove extra dimensions if necessary
    #             emg_derivative_z = np.squeeze(emg_derivative_z)  # This will reduce dimensions like (1, ...) to (...)
    #
    #             # Create a figure
    #             plt.figure(figsize=(12, 6))
    #
    #             # Plot each epoch individually without averaging
    #             for i in range(emg_derivative_z.shape[0]):
    #                 plt.plot(emg_derivative_z[i].T, label=f'Epoch {i + 1}')
    #
    #             # Set labels and title
    #             plt.title(f'EMG Derivative Z-Score (Individual Epochs) for {target}')
    #             plt.xlabel('Time (samples)')
    #             plt.ylabel('Z-Score')
    #             plt.legend(loc='upper right')
    #             plt.savefig(z_figs / f'{sub_input}_{condition}_{target}_{index}_z_scores.png')
    #             plt.close()
    #
    # plot_emg_derivative_z(baseline_z_scores, target='baseline')
    # plot_emg_derivative_z(response_z_scores, target='responses')
    # plot_emg_derivative_z(readiness_z_scores, target='non_target_stim')
    # plot_emg_derivative_z(distractor_z_scores, target='distractor')
