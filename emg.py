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



    def get_avg_band_power(power, fmin, fmax):
        """Compute the average power within a specific frequency band."""
        freq_mask = (power.freqs >= fmin) & (power.freqs <= fmax)

        band_power = power.data[:, freq_mask, :] # data within frequency band
        avg_power = band_power.mean(axis=2).mean()  # Average across all epochs
        # Find the maximum power across frequencies and time
        max_power_idx_time = band_power.mean(axis=1).argmax()  # Max across frequencies, returning index for time
        max_power_idx_freq = band_power.mean(axis=2).argmax()  # Max across time, returning index for frequency
        max_power = band_power[:, :, :].max()  # Find the actual max power
        # Get the corresponding frequency and time for the maximum power
        max_freq = power.freqs[freq_mask][max_power_idx_freq]  # Frequency corresponding to max power
        max_time = power.times[max_power_idx_time]  # Time corresponding to max power

        # Calculate the first derivative (velocity) across time for all frequencies
        velocity = np.diff(band_power, axis=-1)  # First derivative along the time axis (change rate of power)
        max_velocity = velocity[:, max_power_idx_freq, max_power_idx_time]
        # Calculate the second derivative (acceleration) across time for all frequencies
        acceleration = np.diff(velocity, axis=-1)  # Second derivative along the time axis (change rate of velocity)
        max_acceleration = acceleration[:, max_power_idx_freq, max_power_idx_time]

        vals = [max_time, max_freq, max_power, avg_power, max_velocity[0], max_acceleration[0]]
        return vals  # Average across all epochs


    response_vals = get_avg_band_power(response_power, fmin=1, fmax=30)
    non_target_vals = get_avg_band_power(non_target_power, fmin=1, fmax=30)
    distractor_vals = get_avg_band_power(distractor_power, fmin=1, fmax=30)
    baseline_vals = get_avg_band_power(baseline_power, fmin=1, fmax=30)


    def get_ref_features(response_vals, non_target_vals, distractor_vals, baseline_vals):
        """
        Calculate variance, RMS, max z scores from baseline and response epochs.
        """

        # Assign response, readiness, and distractor values
        response_time, response_freq, response_max_power, response_avg_power, response_avg_acceleration = response_vals
        non_target_time, non_target_freq, non_target_max_power, non_target_avg_power, non_target_avg_acceleration = non_target_vals
        distractor_time, distractor_freq, distractor_max_power, distractor_avg_power, distractor_avg_acceleration = distractor_vals
        baseline_time, baseline_freq, baseline_max_power, baseline_avg_power, baseline_avg_acceleration = baseline_vals

        # Return a dictionary of features
        features = {
            'max time': (response_time, non_target_time, distractor_time, baseline_time),
            'max frequency': (response_freq, non_target_freq, distractor_freq, baseline_freq),
            'max power': (response_max_power, non_target_max_power, distractor_max_power, baseline_max_power),
            'avg power': (response_avg_power, non_target_avg_power, distractor_avg_power, baseline_avg_power),
            'avg_acceleration': (response_avg_acceleration, non_target_avg_acceleration, distractor_avg_acceleration, baseline_avg_acceleration)
        }

        return features


    # Call with the correct argument order:
    features = get_ref_features(response_vals, non_target_vals, distractor_vals, baseline_vals)

    print(features)

    def adaptive_thresholds(features):
        response_max_frequency = features['max frequency'][0]
        non_target_max_frequency = features['max frequency'][1]
        distractor_max_frequency = features['max frequency'][2]
        baseline_max_frequency = features['max frequency'][3]

        response_max_time = features['max time'][0]
        non_target_max_time = features['max time'][1]
        distractor_max_time = features['max time'][2]
        baseline_max_time = features['max time'][3]

        response_max_power = features['max power'][0]
        non_target_max_power = features['max power'][1]
        distractor_max_power = features['max power'][2]
        baseline_max_power = features['max power'][3]

        # Extract average power for each condition
        response_avg_power = features['avg power'][0]
        non_target_avg_power = features['avg power'][1]
        distractor_avg_power = features['avg power'][2]
        baseline_avg_power = features['avg power'][3]

        # Extract average acceleration for each condition
        response_avg_acceleration = features['avg_acceleration'][0]
        non_target_avg_acceleration = features['avg_acceleration'][1]
        distractor_avg_acceleration = features['avg_acceleration'][2]
        baseline_avg_acceleration = features['avg_acceleration'][3]
        
        response_threshold = {'max frequency': response_max_frequency, 'max time': response_max_time, 'max power': response_max_power, 'avg power': response_avg_power, 'avg acceleration': response_avg_acceleration}
        non_target_threshold = {'max frequency': non_target_max_frequency, 'max time': non_target_max_time, 'max power': non_target_max_power, 'avg power': non_target_avg_power, 'avg acceleration': non_target_avg_acceleration}
        distractor_threshold = {'max frequency': distractor_max_frequency, 'max time': distractor_max_time, 'max power': distractor_max_power, 'avg power': distractor_avg_power, 'avg acceleration': distractor_avg_acceleration}
        baseline_threshold = {'max frequency': baseline_max_frequency, 'max time': baseline_max_time, 'max power': baseline_max_power, 'avg power': baseline_avg_power, 'avg acceleration': baseline_avg_acceleration}
        return response_threshold, non_target_threshold, distractor_threshold, baseline_threshold


    response_threshold, non_target_threshold, distractor_threshold, baseline_threshold = adaptive_thresholds(features)

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
    # Epoch the EMG data
    target_epochs = epochs(target_events, target_emg_rectified, targets_emg_events, target='target', tmin=-0.2,
                           tmax=0.9, baseline=(-0.2, 0.0))
    distractor_epochs = epochs(distractor_events, distractor_emg_rectified, distractors_emg_events,
                               target='distractor', tmin=-0.2, tmax=0.9, baseline=(-0.2, 0.0))
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
    invalid_responses = total_invalid_response_count * 100 / (total_target_count + total_distractor_count + total_non_target_stim_count)
    # total errors:
    target_errors = (target_invalid_response_count + misses_count) * 100 / total_target_count
    total_errors = (total_invalid_response_count + misses_count + distractor_response_count) * 100 / (total_non_target_stim_count + total_target_count + total_distractor_count)

    # plot performance: correct responses, distractor, target misses, invalid target resp, total target error
    categories = ['Correct Target Responses', 'Distractor Responses', 'Targets Missed', 'Invalid Target Responses', 'Total Target Error']
    colors = ['blue', 'red', 'yellow', 'orange', 'black']
    values = [correct_responses, distractor_responses, missed_targets, invalid_target_responses, target_errors]
    plt.figure(figsize=(12, 6))
    plt.bar(categories, values, color=colors)
    plt.xlabel('Response Type')
    plt.ylabel('Performance (in %)')
    plt.title(f'{condition}_{index} performance of {sub_input}')
    plt.savefig(fig_path / f'{condition}_{index}_performance_{sub_input}.png')
    plt.close()

    # todo: classify epochs based on features.
    # target epochs: true responses, no response, readiness
    # distractor epochs: same
    # non_target stim: same
    # target epochs with response:
    # Define frequency range and number of cycles for Morlet wavelets
    frequencies = np.logspace(np.log10(1), np.log10(30), num=30)
    n_cycles = frequencies / 2

    # Frequency mask for the desired frequency range
    freq_mask = (frequencies >= 1) & (frequencies <= 30)

    # Initialize lists to store power and metrics for each epoch
    power_list = []
    epochs_vals = []
    sampling_rate = 500

    # Loop over each epoch and compute TFA independently
    for epoch_index in range(len(target_epochs)):
        # Extract single epoch as an Epochs object for TFR computation
        epoch_data = target_epochs.get_data(copy=True)[epoch_index][np.newaxis, :, :]
        epoch_info = target_epochs.info
        single_epoch = mne.EpochsArray(epoch_data, epoch_info, tmin=target_epochs.tmin)

        # Compute TFR for the individual epoch
        power = mne.time_frequency.tfr_morlet(single_epoch, freqs=frequencies, n_cycles=n_cycles, return_itc=False)

        # Extract data within the frequency band
        band_power = power.data[:, freq_mask, :]  # Shape: (1, n_frequencies, n_times)

        # Find max power, time, and frequency
        max_power = band_power.max()
        avg_power = band_power.mean(axis=2).mean()  # Average power over time for each epoch

        # Find the time and frequency indices for max power
        max_time_idx = band_power.mean(axis=1).argmax(axis=-1)
        max_freq_idx = band_power.mean(axis=2).argmax(axis=-1)

        # Map max frequency and time indices to actual values
        max_freq = power.freqs[freq_mask][max_freq_idx[0]]
        max_time = power.times[max_time_idx[0]]

        # Calculate velocity (first derivative) and acceleration (second derivative)
        velocity = np.diff(band_power, axis=-1)
        acceleration = np.diff(velocity, axis=-1)

        # Retrieve velocity and acceleration at the max power point
        max_velocity = velocity[0, max_freq_idx[0], max_time_idx[0]]
        max_acceleration = acceleration[0, max_freq_idx[0], max_time_idx[0]]

        # Store values for each epoch
        epoch_vals = [max_time, max_freq, max_power, avg_power, max_velocity, max_acceleration]
        epochs_vals.append(epoch_vals)
        # todo finish getting each epoch's features. refine avg features function
        # todo: compare features to threshold and classify epochs -> see if it gets it correctly
        # todo: repeat for distractors, and non-target stim
        # todo: plot pie charts of bar plots of true responses, readiness and no resp of target vs distractor




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
