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
    for index, items in enumerate(csv.values):
        block_seq = items[0]  # from first column, save info as variable block_seq
        block_condition = items[1]  # from second column, save info as var block_condition
        if block_seq == target_stream and block_condition == axis:
            block = csv.iloc[index]
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
    emg_filt = emg.copy().filter(l_freq=20, h_freq=150)
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


'''Smoothing the rectified EMG signal helps reduce high-frequency noise and makes the muscle activation patterns more visible and interpretable. 
It helps to create an EMG envelope, showing overall trends in muscle activity rather than every small fluctuation.
Noise reduction: It minimizes random fluctuations in the signal, making it easier to identify meaningful muscle activity.
Highlight activation: Smoothing reveals broader trends in muscle activity, which is useful for detecting partial errors 
or pre-activation in EMG signals.'''

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
    helper_filt = helper.copy().filter(l_freq=20, h_freq=150)
    print('Remove power line noise and apply minimum-phase highpass filter')  # Cheveigné, 2020
    helper_filt.copy().notch_filter(freqs=[50, 100, 150], method='fir')
    helper_rectified = helper.copy()
    helper_rectified._data = np.abs(helper_rectified._data)
    # helper_rectified._data *= 1e6
    # create helper epochs manually:
    helper_n_samples = len(helper_rectified.times)
    return helper_rectified, helper_n_samples


# Create baseline events every 2 seconds
def baseline_epochs(events_emg, emg_rectified, events_dict, target):
    tmin = -1.5
    tmax = 1.0
    event_ids = {key: val for key, val in events_dict.items() if any(event[2] == val for event in events_emg)}
    epochs_baseline = mne.Epochs(emg_rectified, events_emg, event_id=event_ids, tmin=tmin, tmax=tmax,
                        baseline=None)
    epochs_baseline.save(fif_path / f'{sub_input}_condition_{condition}_{index}_{target}-epo.fif', overwrite=True)
    return epochs_baseline


# get emg epochs, around the target and distractor events:
def epochs(events_dict, emg_rectified, events_emg, target, baseline=None):
    tmin = -1.5
    tmax = 1.0
    event_ids = {key: val for key, val in events_dict.items() if any(event[2] == val for event in events_emg)}
    epochs = mne.Epochs(emg_rectified, events_emg, event_id=event_ids, tmin=tmin, tmax=tmax, baseline=baseline)
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
        df_path = default_dir / 'performance' / subs / 'tables'  # csv tables containing important information from vmrk files
        sub_dirs.append(sub_dir)
        json_path = default_dir / 'misc'
        fif_path = emg_dir / 'fif files'  # pre-processed .eeg files of the EMG data will be stored
        for folder in sub_dir, fif_path, results_path, fig_path, df_path:
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
    target = ['target', 'distractor', 'responses', 'distractor_baseline', 'target_baseline'] # tags needed for the epoching section

    ### Get target header files for all participants
    target_header_files_list = []
    for sub_dir in sub_dirs:
        header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
        filtered_files = [file for file in header_files if condition in file]
        if filtered_files:
            target_header_files_list.append(filtered_files)

    all_blocks_data = []  # Initialize cumulative counts for all blocks
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

        # Get events for EMG analysis, based on filtered target and distractor, and response events
        targets_emg_events = create_events(target_events, target_mapping)
        distractors_emg_events = create_events(distractor_events, distractor_mapping)
        responses_emg_events = create_events(response_events, response_mapping)

        # Filter and rectify the EMG data
        emg_filt = filter_emg(emg)
        target_emg_rectified = rectify(emg_filt)
        distractor_emg_rectified = rectify(emg_filt)
        responses_emg_rectified = rectify(emg_filt)

        # Create Baseline Epochs:
        # for distractor:
        b_events_distractor = baseline_events(distractor_events, distractor_mapping)
        distractor_epochs_baseline = baseline_epochs(b_events_distractor, distractor_emg_rectified, distractor_events, target=target[3])
        # for target:
        b_events_target = baseline_events(target_events, target_mapping)
        target_epochs_baseline = baseline_epochs(b_events_target, target_emg_rectified, target_events, target=target[4])
        # concatenate all the baseline epochs:
        epochs_baseline = mne.concatenate_epochs([distractor_epochs_baseline, target_epochs_baseline])
        # Epoch the EMG data
        target_epochs = epochs(target_events, target_emg_rectified, targets_emg_events, target[0], baseline=(-0.2, 0.0))
        distractor_epochs = epochs(distractor_events, distractor_emg_rectified, distractors_emg_events, target[1], baseline=(-0.2, 0.0))
        response_epochs = epochs(response_events, responses_emg_rectified, responses_emg_events, target[2], baseline=(-0.2, 0.0))

        # load the csv tables, related to the condition and index: they contain vmrk information
        vmrk_files = []
        for files in df_path.iterdir():
            if files.is_file and f'{condition}_{index}' in files.name:
                vmrk_files.append(files)
        vmrk_dfs = {}
        for index, dfs in enumerate(vmrk_files):
            vmrk_df = pd.read_csv(dfs, delimiter=',')
            vmrk_dfs[dfs.name[:-4]] = vmrk_df
            # here we read each csv file: target resposnes, distractor responses, invalid responses
            # and target and distractor events without response
        # correct responses, distractor responses: 1
        # no responses: 0
        # invalid responses: 3
        # use baseline epochs as, well, baseline

        concatenated_vmrk_dfs = pd.concat(vmrk_dfs)
        vmrk_dfs_sorted = concatenated_vmrk_dfs.sort_values(by=['Unnamed: 0'], ascending=True)
        vmrk_dfs = vmrk_dfs_sorted.rename(columns={'Unnamed: 0': 'Index'})
        del concatenated_vmrk_dfs, vmrk_dfs_sorted

        # # get ERPs of each and save as fig:
        # plt.ioff()  # turn off interactive plotting
        # target_erp = target_epochs.average().plot()
        # target_erp.savefig(fig_path / f'{sub_input}_condition_{condition}_{index}_target_ERP.png')

        # Create thresholds:
        # Calculate the baseline derivative:
        def baseline_normalization(emg_epochs, tmin, tmax):
            emg_epochs.load_data()  # This ensures data is available for manipulation
            emg_window = emg_epochs.copy().crop(tmin, tmax)  # select a time window within each epoch, where a response to stimulus is expected
            emg_data = emg_window.get_data(copy=True)  # Now it's a NumPy array
            emg_derivative = np.diff(emg_data, axis=-1)  # Derivative across time
            # The np.diff() function calculates the derivative of emg_data along the time axis (axis=-1).
            # This results in a new array, emg_derivative, with a shape of (n_epochs, n_channels, n_samples - 1).
            # Z-score normalization
            emg_derivative_z = (emg_derivative - np.mean(emg_derivative, axis=-1, keepdims=True)) / np.std(
                emg_derivative, axis=-1, keepdims=True)
            return emg_data, emg_derivative, emg_derivative_z

        baseline_emg_data, baseline_emg_derivative, baseline_emg_derivative_z = baseline_normalization(epochs_baseline, tmin=0.5, tmax=1.0)
        baseline_mean = np.mean(baseline_emg_derivative, axis=-1, keepdims=True)
        baseline_std = np.std(baseline_emg_derivative, axis=-1, keepdims=True)

        '''axis=0: Refers to the first dimension (rows or the outermost dimension).
           axis=1: Refers to the second dimension (columns or the second layer of dimensions).
           axis=-1: Refers to the last dimension (often time samples or individual values within a feature).'''

        # Extract data from MNE Epochs object (shape will be [n_epochs, n_channels, n_samples])
        def z_normalization(emg_epochs, baseline_mean, baseline_std, tmin, tmax, target):
            emg_epochs.load_data()  # Ensure data is loaded for manipulation
            emg_window = emg_epochs.copy().crop(tmin, tmax)  # Crop epochs to desired window
            emg_data = emg_window.get_data(copy=True)  # Extract data as a NumPy array

            # Compute the derivative across the time axis
            emg_derivative = np.diff(emg_data, axis=-1)

            # Ensure baseline_mean and baseline_std are properly aligned to the current data shape
            baseline_mean = np.mean(baseline_mean, axis=0) # use global baseline mean
            baseline_std = np.mean(baseline_std, axis=0)  # Use global baseline std
            # Collapse over epochs axis to get a single mean and std

            # Apply normalization using the correct broadcasting for single channel data
            emg_derivative_z = (emg_derivative - baseline_mean) / baseline_std
            emg_reduced = np.squeeze(emg_derivative_z)

            event_times = emg_window.times
            timepoints = []
            for idx, row in enumerate(emg_window.events):
                epoch_times = int(emg_window.events[idx][0]) / 500
                absolute_times = epoch_times + event_times
                timepoints.append(absolute_times)
            absolute_timepoints = [times[:-1] for times in timepoints]
            epochs_z_scores_dfs = {}
            # Iterate through each epoch and its corresponding time points
            for epoch_idx, (epoch_data, time_array) in enumerate(zip(emg_reduced, absolute_timepoints)):
                # Create a DataFrame for each epoch
                epoch_df = pd.DataFrame([epoch_data], columns=time_array)

                # Add the epoch index as an identifier (optional)
                epoch_df['Epoch'] = epoch_idx + 1

                # Set 'Epoch' as the index (optional, depends on your requirements)
                epoch_df.set_index('Epoch', inplace=True)

                # Append the DataFrame to the list
                epochs_z_scores_dfs[epoch_idx] = epoch_df
                # todo: save excel file with z scores of each epoch
            return epochs_z_scores_dfs, emg_reduced

        # for a window of 350 samples (0.7s)
        target_z_scores_dfs, target_emg_z = z_normalization(target_epochs, baseline_mean, baseline_std, tmin=0.2, tmax=0.9, target=target[0])
        distractor_z_score_dfs, distractor_emg_z = z_normalization(distractor_epochs, baseline_mean, baseline_std, tmin=0.2, tmax=0.9, target=target[1])
        response_z_scores_dfs, response_emg_z = z_normalization(response_epochs, baseline_mean, baseline_std, tmin=-0.3, tmax=0.4, target=target[2])

        # True Response Threshold:
        # You can use the z-transformed EMG derivative from response epochs
        # to establish a threshold for classifying true responses.
        # Typically, you might select a threshold that is 1 standard deviation above the baseline
        # (i.e., a z-score > 1.0).
        # No Response Threshold: The baseline epochs can help define a threshold
        # for identifying no responses (i.e., z-scores close to 0).
        # Partial Response: Anything in between could be considered a partial response,
        # where the z-scores are not high enough to be classified as a full true response,
        # but there is still some activity.
        # Upper threshold for true responses (from response epochs)#
        response_std = np.std(response_emg_z)
        b_std = np.std(baseline_emg_derivative_z)  # should be 1.0
        response_threshold = np.abs(np.max(response_emg_z)) / (1.25 * response_std)
        # Lower threshold for no responses (from baseline epochs)
        no_response_threshold = np.abs(np.min(baseline_emg_derivative_z)) / 2 * b_std

        def classify_emg_epochs(emg_z_scores_dfs, response_threshold, no_response_threshold):
            classifications_list = []
            for epochs_df, df in emg_z_scores_dfs.items():
                # Step 1: Calculate the absolute values of the z-scores
                abs_vals = np.abs(df.values)  # Calculate absolute values for the single row

                # Step 2: Use np.where to find the index of the max absolute z-score
                max_pos = np.where(abs_vals == np.max(abs_vals))  # Returns a tuple of (row indices, column indices)

                # Step 3: Since there is only one row, get the max column index
                max_column_index = max_pos[1][0]  # Extract the first column index from the tuple

                # Step 4: Use the column index to get the corresponding timepoint
                timepoint = df.columns[max_column_index]  # Retrieve timepoint using column index

                # Step 5: Extract the exact z-score value for printing (optional, since you have abs_vals)
                z = abs_vals[0, max_column_index]
                if z >= response_threshold:
                    classifications_list.append((epochs_df, "True Response", z, timepoint))
                elif z <= no_response_threshold:
                    classifications_list.append((epochs_df, "No Response", z, timepoint))
                else:
                    classifications_list.append((epochs_df, "Partial Response", z, timepoint))
            classifications = pd.DataFrame(classifications_list, columns=['Epoch', 'Response', 'Z-score', 'Timepoint'])

            return classifications

        # Classify target and distractor epochs separately
        target_classifications = classify_emg_epochs(target_z_scores_dfs, response_threshold, no_response_threshold)
        distractor_classifications = classify_emg_epochs(distractor_z_score_dfs, response_threshold, no_response_threshold)


        def verify_and_refine_classification_all_conditions(classifications, vmrk_dfs, time_window=0.9, target=''):
            """
            Verifies classifications by checking if their timepoints match the corresponding
            vmrk_dfs responses within a given time window.

            Args:
            - classifications: DataFrame with initial classifications.
            - vmrk_dfs: DataFrame with event markers.
            - time_window: Time difference tolerance for matching (default is 0.9 seconds).
            - response: The type of response to verify in the classifications DataFrame.
            - vmrk_response: The corresponding vmrk_df response (e.g., 1 for True Response, 0 for No Response).

            Returns:
            - Updated classification DataFrame with a 'Match' column indicating matched rows.
            """

            # Create a copy of the initial classification DataFrame to modify
            refined_classification_df = classifications.copy()

            # Define mappings for classification types and corresponding `vmrk_response` values
            classification_conditions = {
                'True Response': [0, 1, 2],  # True Response can match both target (1) and distractor (2) responses
                'No Response': [0, 1, 2],
                'Partial Response': [0, 1, 2],  # Partial Responses can match with 0, 1, 2
            }

            # Define the match labels based on the `vmrk_response` value and the classification type
            match_labels = {
                # Solid Cases
                (1, 'True Response'): 'target button press',  # A correct button press to the target stimulus
                (0, 'No Response'): 'no response',  # No response to a presented stimulus (target or distractor)
                (2, 'True Response'): 'distractor button press',  # A button press response to a distractor stimulus

                # Partial Responses:
                (0, 'Partial Response'): 'response temptation',  # A partial response to a target stimulus
                (1, 'Partial Response'): 'weak target button press',  # A partial response to a distractor stimulus
                (2, 'Partial Response'): 'weak distractor button press',  # A partial response classified as invalid

                # Other:
                (0, 'True Response'): 'invalid response',
                (1, 'No Response'): 'weak target button press',
                (2, 'True Response'): 'weak distractor button press',  # A strong response but considered invalid
            }
            # Initialize the 'Match' column if not already present
            if 'Match' not in refined_classification_df.columns:
                refined_classification_df['Match'] = 'Unmatched'  # Default value for unmatched rows

            # Step 1: Iterate through each condition type and corresponding `vmrk_response` values
            for classification_type, vmrk_response_values in classification_conditions.items():
                # true, no and partial response, 0,1,2,3 in dict:
                # Extract rows from refined_classification_df matching the given classification type
                condition_responses_df = refined_classification_df[
                    refined_classification_df['Response'] == classification_type]
                # print(condition_responses_df)

                # Step 2: Iterate over the vmrk_response_values for the given classification_type
                # Apply proper parentheses to the logical condition
                for vmrk_response_value in vmrk_response_values:
                    # Apply proper parentheses to the logical condition -> rows of interest
                    vmrk_condition_responses_df = vmrk_dfs[
                        (vmrk_dfs['Response'] == vmrk_response_value) & (vmrk_dfs['Stimulus Type'].str.strip() == target)
                        ]
                    # Step 3: Match `condition_responses_df` in `refined_classification_df` with `vmrk_dfs` based on timepoints
                    match_results = []  # Store results of each verification for review

                    # Step 4: Iterate over each response in `condition_responses_df`
                    for idx, row in condition_responses_df.iterrows():
                        # Get the timepoint of the current Response in refined_classification_df
                        response_time = row['Timepoint']

                        # Check if there's a matching timepoint in vmrk_condition_responses_df within the allowed tolerance
                        matching_rows = vmrk_condition_responses_df[
                            (vmrk_condition_responses_df['Timepoints'] >= response_time - time_window) &
                            (vmrk_condition_responses_df['Timepoints'] <= response_time + time_window)
                            ]
                        # Step 5: If at least one match is found, update with the corresponding match label
                        if not matching_rows.empty:
                            match_label = match_labels.get((vmrk_response_value, classification_type), 'Matched')
                            match_results.append((idx, match_label))
                        else:
                            match_results.append((idx, 'Unmatched'))

                    # Step 6: Update the existing 'Match' column with the new match labels
                    for idx, match_status in match_results:
                        if refined_classification_df.at[idx, 'Match'] in ['Unmatched',
                                                                          'Not Matched']:  # Update only if not already matched
                            refined_classification_df.at[idx, 'Match'] = match_status

            return refined_classification_df


        target_refined_classifications = verify_and_refine_classification_all_conditions(target_classifications, vmrk_dfs, target=target_stream)
        distractor_refined_classifications = verify_and_refine_classification_all_conditions(distractor_classifications, vmrk_dfs, target=distractor_stream)


        def summarize_classifications(target_refined_classifications, distractor_refined_classifications, vmrk_dfs):
            # Step 1: Summarize the counts for each match label
            total_stim = vmrk_dfs[vmrk_dfs['Stimulus Type'] == target_stream]
            total_target_stim_count = len(total_stim)
            total_distractor_stim_count = len(vmrk_dfs[vmrk_dfs['Stimulus Type'] == distractor_stream])
            target_summary_counts = target_refined_classifications['Match'].value_counts()
            distractor_summary_counts = distractor_refined_classifications['Match'].value_counts()

            # Step 2: Define main categories based on match labels
            target_no_response = target_summary_counts.get('no response', 0)
            target_button_presses = target_summary_counts.get(f'target button press', 0) + target_summary_counts.get('weak target button press', 0)
            target_invalid_responses = target_summary_counts.get('invalid response', 0)
            target_response_temptation = target_summary_counts.get('response temptation', 0)
            distractor_response_temptation = distractor_summary_counts.get('response temptation')
            distractor_button_presses = distractor_summary_counts.get(f'distractor button press', 0) + distractor_summary_counts.get(f'weak distractor button press', 0)
            distractor_invalid_responses = distractor_summary_counts.get(f'invalid response', 0)
            total_button_presses = target_button_presses + target_invalid_responses + distractor_button_presses + distractor_invalid_responses


            # Step 3: Create categories and percentages
            summary_data = {
                'Target No-Responses': target_no_response,
                'Target Button-Presses': target_button_presses,
                'Distractor Button-Presses': distractor_button_presses,
                'Total Invalid-Responses (target & distractor)': target_invalid_responses + distractor_invalid_responses,
                'Target Response-Temptation': target_response_temptation,
                'Distractor Response-Temptation': distractor_response_temptation
            }

            # Calculate percentages for each category
            correct_responses = (target_button_presses / total_target_stim_count) * 100
            invalid_responses = ((target_invalid_responses + distractor_invalid_responses) / total_button_presses) * 100
            distractor_responses = (distractor_button_presses / total_button_presses) * 100
            target_temptations = (target_response_temptation / total_target_stim_count) * 100
            distractor_temptations = (distractor_response_temptation / total_distractor_stim_count) * 100
            total_errors = ((target_invalid_responses + distractor_invalid_responses + distractor_button_presses) / total_button_presses) * 100
            total_misses = ((total_target_stim_count - target_button_presses) * 100) / total_target_stim_count

            summary_percentages = {'Correct Responses': correct_responses, 'Distractor Responses': distractor_responses, 'Invalid Responses (all)': invalid_responses,
                                   'Target Response-Readiness': target_temptations, 'Distractor Response-Readiness': distractor_temptations, 'Total Errors (all responses)': total_errors,
                                   'Total Target Misses': total_misses}

            return summary_data, summary_percentages

    # Step 4: Summarize target and distractor classifications


        summary_data, summary_percentages = summarize_classifications(target_refined_classifications, distractor_refined_classifications, vmrk_dfs)


        def plot_summary_percentages(summary_percentages):
            """
            Plots a bar chart to visualize the summarized classification percentages.

            Args:
            - summary_percentages: Dictionary containing calculated percentages for each classification type.
            """
            # Step 1: Extract categories and their corresponding percentages
            categories = list(summary_percentages.keys())
            percentages = list(summary_percentages.values())

            # Step 2: Create a bar plot
            plt.figure(figsize=(14, 8))
            bars = plt.bar(categories, percentages, color=['#66b3ff', '#ff9999', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#8c8c8c'])

            # Step 3: Add labels and values to each bar
            for bar, percentage in zip(bars, percentages):
                yval = bar.get_height()
                plt.text(bar.get_x() + bar.get_width() / 2.0, yval + 0.5, f'{percentage:.1f}%', ha='center', va='bottom')

            # Step 4: Customize the plot
            plt.xlabel('Response Type')
            plt.ylabel('Percentage (%)')
            plt.title('Summary of EMG Z-score Classifications')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(0, max(percentages) + 5)  # Adjust y-axis limit for better label placement

            # Step 5: Show the plot
            plt.tight_layout()
            plt.close()
            plt.savefig(fig_path / f'{sub}_{condition}_{index}_EMG_classifications.png')

        # Run the plotting function using your calculated summary percentages
        plot_summary_percentages(summary_percentages)
'''
        def plot_emg_derivative_z(emg_derivative_z, target, epoch_idx=None):
            # Remove extra dimensions if necessary
            emg_derivative_z = np.squeeze(emg_derivative_z)  # This will reduce dimensions like (1, ...) to (...)

            # Create a figure
            plt.figure(figsize=(12, 6))

            # Plot each epoch individually without averaging
            for i in range(emg_derivative_z.shape[0]):
                plt.plot(emg_derivative_z[i].T, label=f'Epoch {i + 1}')

            # Set labels and title
            plt.title(f'EMG Derivative Z-Score (Individual Epochs) for {target}')
            plt.xlabel('Time (samples)')
            plt.ylabel('Z-Score')
            plt.legend(loc='upper right')
            # else:
            #     # If no epoch is specified, flatten and plot all epochs
            #     emg_data_flat = np.mean(emg_derivative_z, axis=1)  # Mean across channels
            #     plt.figure(figsize=(10, 6))
            #     for i in range(emg_data_flat.shape[0]):  # Iterate over epochs
            #         plt.plot(emg_data_flat[i], label=f'Epoch {i + 1}')
            #     plt.title('EMG Derivative Z-Score (All Epochs)')
            #     plt.xlabel('Time (samples)')
            #     plt.ylabel('Z-Score')
            #     plt.legend(loc='best')
            plt.show()
            plt.savefig(fig_path / f'{sub_input}_{condition}_{target}_{index}_z_derivatives')

        #
        # # Call the function to plot all epochs
        plot_emg_derivative_z(target_emg_z, target=target[0])
        plot_emg_derivative_z(distractor_emg_z, target=target[1])
        plot_emg_derivative_z(response_emg_z, target=target[2])
        plot_emg_derivative_z(baseline_emg_derivative_z, target='baseline')
        
                    def count_true_responses(df):
                """Helper function to count the number of True Responses in a DataFrame."""
                return len(df[df['Response'] == 'True Response'])

            initial_true_responses = count_true_responses(refined_classification_df)
            # Get the true response count from the VMRK DataFrame (markers)
            marker_true_responses = len((vmrk_dfs[(vmrk_dfs['Stimulus Type'] == target_stream) & (vmrk_dfs['Response'] == 1)]))
            print(f"Initial True Responses: {initial_true_responses}, Marker True Responses: {marker_true_responses}")
            # Iteratively refine the classification by adjusting the z-score threshold
            iterations = 0
            response_threshold = 10  # Start with an initial response threshold (adjust as needed)
            no_response_threshold = 2
            while initial_true_responses != marker_true_responses and iterations < 20:
                iterations += 1
                print(f"\nIteration: {iterations}")

                # Adjust thresholds based on mismatch (simple example: adjust response threshold down if too few True Responses)
                if initial_true_responses < marker_true_responses:
                    response_threshold *= 0.95  # Decrease threshold to get more True Responses
                    no_response_threshold *= 1.05  # Increase threshold to reduce False Positives
                else:
                    response_threshold *= 1.05  # Increase threshold to be stricter
                    no_response_threshold *= 0.95  # Decrease threshold for more No Responses

                print(
                    f"New Response Threshold: {response_threshold}, New No-Response Threshold: {no_response_threshold}")
                # Re-classify using the new thresholds
                refined_classification_df['Response'] = refined_classification_df['Z-score'].apply(
                    lambda z: "True Response" if z >= response_threshold else
                    "No Response" if z <= no_response_threshold else
                    "Partial Response"
                )
                # Re-count the number of True Responses after classification
                initial_true_responses = count_true_responses(refined_classification_df)
                print(f"Updated True Responses: {initial_true_responses}")
                # Final comparison to see if matching
            if initial_true_responses == marker_true_responses:
                print("\nRefinement successful! The number of True Responses matches the markers.")
            else:
                print("\nReached maximum iterations. The number of True Responses still does not match the markers.")
'''