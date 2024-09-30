'''
Pre-processing EMG data: baseline recording,
motor-only responses and epoching our signal around target
and distractor events respectively.
All .fif files created are saved in a 'fif files' folder,
for the selected subject, and named after selected condition,
and events selected (target, distractor or response events)
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
        target_mapping = s1_mapping
        distractor_mapping = s2_mapping
    elif condition in ['a2', 'e2']:
        target_stream = 's2'
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
    return target_stream, target_blocks, axis, target_mapping, distractor_mapping

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
        data_dir = default_dir / 'eeg' / 'raw'
        sub_dir = data_dir / subs
        emg_dir = default_dir / 'emg' / subs
        results_path = emg_dir / 'preprocessed' / 'results' / 'z_score_data'
        fig_path = emg_dir / 'preprocessed' / 'figures'
        sub_dirs.append(sub_dir)
        json_path = default_dir / 'misc'
        fif_path = emg_dir / 'fif files'
        for folder in sub_dir, fif_path, results_path, fig_path:
            if not os.path.isdir(folder):
                os.makedirs(folder)

    # Load necessary JSON files
    with open(json_path / "preproc_config.json") as file:
        cfg = json.load(file)
    with open(json_path / "electrode_names.json") as file:
        mapping = json.load(file)
    with open(json_path / 'eeg_events.json') as file:
        markers_dict = json.load(file)

    s1_events = markers_dict['s1_events']
    s2_events = markers_dict['s2_events']  # stimulus 2 markers
    response_events = markers_dict['response_events']  # response markers

    mapping_path = json_path / 'events_mapping.json'
    with open(mapping_path, 'r') as file:
        events_mapping = json.load(file)
    s1_mapping = events_mapping[0]
    s2_mapping = events_mapping[1]
    response_mapping = events_mapping[2]

    condition = input('Please provide condition (exp. EEG): ')  # choose a condition of interest for processing
    target = ['target', 'distractor', 'responses', 'distractor_baseline', 'distractor_baseline', 'target_baseline' ]
    ### Get target header files for all participants
    target_header_files_list = []
    for sub_dir in sub_dirs:
        header_files = [file for file in os.listdir(sub_dir) if ".vhdr" in file]
        filtered_files = [file for file in header_files if condition in file]
        if filtered_files:
            target_header_files_list.append(filtered_files)
    # Initialize cumulative counts for all blocks
    # List to store counts for each block
    all_blocks_data = []
    ### Process Each File under the Selected Condition ###
    for index in range(len(target_header_files_list[0])):
        # Loop through all the files matching the condition
        emg_file = target_header_files_list[0][index]
        full_path = os.path.join(sub_dir, emg_file)
        emg = mne.io.read_raw_brainvision(full_path, preload=True)

        # Set montage for file:
        emg.rename_channels(mapping)
        emg.set_montage('standard_1020')
        emg = emg.pick_channels(['A2', 'M2', 'A1'])
        emg.set_eeg_reference(ref_channels=['A1'])
        emg = mne.set_bipolar_reference(emg, anode='A2', cathode='M2', ch_name='EMG_bipolar')
        emg.drop_channels(['A1'])

        # Get Target Block information
        csv_path = default_dir / 'params' / 'block_sequences' / f'{sub_input}.csv'
        csv = pd.read_csv(csv_path)
        target_stream, target_blocks, axis, target_mapping, distractor_mapping = get_target_blocks()

        # Define target and distractor events
        target_block = [target_block for target_block in target_blocks.values[index]]
        target_events, distractor_events = define_events(target_stream)

        # Get events for EMG analysis
        targets_emg_events = create_events(target_events, target_mapping)
        distractors_emg_events = create_events(distractor_events, distractor_mapping)
        responses_emg_events = create_events(response_events, response_mapping)

        # Filter, rectify, (and smooth) the EMG data
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
        epochs_baseline = mne.concatenate_epochs([distractor_epochs_baseline, target_epochs_baseline])
        # Epoch the EMG data
        target_epochs = epochs(target_events, target_emg_rectified, targets_emg_events, target[0], baseline=(-0.1, 0.1))
        distractor_epochs = epochs(distractor_events, distractor_emg_rectified, distractors_emg_events, target[1], baseline=(-0.1, 0.1))
        response_epochs = epochs(response_events, responses_emg_rectified, responses_emg_events, target[2], baseline=(-0.2, 0.0))

        # get ERPs of each and save as fig:
        plt.ioff()  # turn off interactive plotting
        target_erp = target_epochs.average().plot()
        target_erp.savefig(fig_path / f'{sub_input}_condition_{condition}_{index}_target_ERP.png')

        distractor_erp = distractor_epochs.average().plot()
        distractor_erp.savefig(fig_path / f'{sub_input}_condition_{condition}_{index}_distractor_ERP.png')

        response_erp = response_epochs.average().plot()
        response_erp.savefig(fig_path / f'{sub_input}_condition_{condition}_{index}_responses_ERP.png')

        baseline_erp = epochs_baseline.average().plot()
        baseline_erp.savefig(fig_path / f'{sub_input}_condition_{condition}_{index}_{target[3]}ERP.png')

        # Create thresholds:
        # Calculate the baseline derivative:
        def baseline_normalization(emg_epochs, tmin, tmax):
            emg_epochs.load_data()  # This ensures data is available for manipulation
            emg_window = emg_epochs.copy().crop(tmin, tmax)  # select a time window within each epoch, where a response to stimulus is expected
            emg_data = emg_window.get_data(copy=True)  # Now it's a NumPy array
            emg_derivative = np.diff(emg_data, axis=-1)  # Derivative across time
            # Z-score normalization
            emg_derivative_z = (emg_derivative - np.mean(emg_derivative, axis=-1, keepdims=True)) / np.std(
                emg_derivative, axis=-1, keepdims=True)
            return emg_data, emg_derivative, emg_derivative_z

        baseline_emg_data, baseline_emg_derivative, baseline_emg_derivative_z = baseline_normalization(epochs_baseline, tmin=-0.5, tmax=0.1)
        baseline_mean = np.mean(baseline_emg_derivative, axis=-1, keepdims=True)
        baseline_std = np.std(baseline_emg_derivative, axis=-1, keepdims=True)

        # Extract data from MNE Epochs object (shape will be [n_epochs, n_channels, n_samples])
        def z_normalization(emg_epochs, baseline_mean, baseline_std, tmin, tmax):
            emg_epochs.load_data()  # This ensures data is available for manipulation
            emg_window = emg_epochs.copy().crop(tmin, tmax)
            emg_data = emg_window.get_data(copy=True)  # Now it's a NumPy array
            emg_derivative = np.diff(emg_data, axis=-1)  # Derivative across time
            baseline_mean = baseline_mean[:, np.newaxis, np.newaxis]  # Add extra dimensions for broadcasting
            baseline_std = baseline_std[:, np.newaxis, np.newaxis]
            # Z-score normalization
            emg_derivative_z = (emg_derivative - baseline_mean[:, np.newaxis]) / baseline_std[:, np.newaxis]
            return emg_data, emg_derivative, emg_derivative_z

        # for a window of 300 samples (0.6s)
        target_emg_data, target_emg_derivative, target_emg_derivative_z = z_normalization(target_epochs, baseline_mean, baseline_std, tmin=0.3, tmax=0.9)

        distractor_emg_data, distractor_emg_derivative, distractor_emg_derivative_z = z_normalization(distractor_epochs, baseline_mean, baseline_std, tmin=0.3, tmax=0.9)

        response_emg_data, response_emg_derivative, response_emg_derivative_z = z_normalization(response_epochs, baseline_mean, baseline_std, tmin=-0.3, tmax=0.3)
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
        # Upper threshold for true responses (from response epochs)
        response_std = np.std(response_emg_derivative_z)
        b_std = np.std(baseline_emg_derivative_z)
        response_threshold = np.abs(np.max(response_emg_derivative_z)) / (1.25 * response_std)
        # Lower threshold for no responses (from baseline epochs)
        no_response_threshold = np.abs(np.min(baseline_emg_derivative_z))
        def classify_emg_epochs(emg_derivative_z, response_threshold, no_response_threshold, sample_window=(150, 250)):
            classifications = []
            for epoch in emg_derivative_z:
                # Extract the samples of interest (e.g., 150-250)
                epoch_window = epoch[..., sample_window[0]:sample_window[1]]  # Slicing the desired sample range
                # Calculate the max z-score within this window
                z = np.abs(np.max(epoch_window))
                if z >= response_threshold:
                    classifications.append("True Response")
                elif z <= no_response_threshold:
                    classifications.append("No Response")
                else:
                    classifications.append("Partial Response")
            return classifications

        # Classify target and distractor epochs separately
        target_classifications = classify_emg_epochs(target_emg_derivative_z, response_threshold, no_response_threshold, sample_window=(50, 200))
        distractor_classifications = classify_emg_epochs(distractor_emg_derivative_z, response_threshold, no_response_threshold, sample_window=(50, 200))


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
            plt.show()
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
            plt.savefig(fig_path/ f'{sub_input}_{condition}_{target}_z_derivatives')


        # Call the function to plot all epochs
        plot_emg_derivative_z(target_emg_derivative_z, target=target[0])
        plot_emg_derivative_z(distractor_emg_derivative_z, target=target[1])
        plot_emg_derivative_z(response_emg_derivative_z, target=target[2])
        plot_emg_derivative_z(baseline_emg_derivative_z, target='baseline')


        # Count the number of each response type for targets and distractors
        target_counts = Counter(target_classifications)
        distractor_counts = Counter(distractor_classifications)

        print(f"Target Responses: {target_counts}")
        print(f"Distractor Responses: {distractor_counts}")
        # Store the results in a dictionary for each block
        block_data = {
            'Block': index,
            'True_Response_Target': target_counts.get('True Response', 0),
            'Partial_Response_Target': target_counts.get('Partial Response', 0),
            'No_Response_Target': target_counts.get('No Response', 0),
            'True_Response_Distractor': distractor_counts.get('True Response', 0),
            'Partial_Response_Distractor': distractor_counts.get('Partial Response', 0),
            'No_Response_Distractor': distractor_counts.get('No Response', 0)
        }
        # Append the data to the list
        all_blocks_data.append(block_data)
        print(f"Processing completed for file {index + 1}/{len(target_header_files_list[0])}")
    # After processing all blocks, save the accumulated results into a CSV
    # Convert the list of dictionaries to a DataFrame
    df = pd.DataFrame(all_blocks_data)

    # Save the DataFrame to a CSV file
    filename = f'{sub_input}_condition_{condition}_emg_classifications.csv'
    df.to_csv(f'{sub_input}_condition_{condition}_emg_classifications.csv', index=False)


def plot_response_pie_charts(df):
    # Sum the counts for True, Partial, and No Responses across all blocks for both Target and Distractor
    target_total_responses = df[['True_Response_Target', 'Partial_Response_Target', 'No_Response_Target']].sum()
    distractor_total_responses = df[['True_Response_Distractor', 'Partial_Response_Distractor', 'No_Response_Distractor']].sum()

    # Calculate percentages for target
    target_total = target_total_responses.sum()
    target_percentages = [
        (target_total_responses['True_Response_Target'] / target_total) * 100,
        (target_total_responses['Partial_Response_Target'] / target_total) * 100,
        (target_total_responses['No_Response_Target'] / target_total) * 100
    ]

    # Calculate percentages for distractor
    distractor_total = distractor_total_responses.sum()
    distractor_percentages = [
        (distractor_total_responses['True_Response_Distractor'] / distractor_total) * 100,
        (distractor_total_responses['Partial_Response_Distractor'] / distractor_total) * 100,
        (distractor_total_responses['No_Response_Distractor'] / distractor_total) * 100
    ]

    # Labels for the pie charts
    labels = ['True Response', 'Partial Response', 'No Response']

    # Create subplots for side-by-side pie charts
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    # Plot Target Pie Chart
    axes[0].pie(target_percentages, labels=labels, autopct='%1.1f%%', startangle=90, colors=['green', 'orange', 'red'])
    axes[0].set_title('Target Responses')

    # Plot Distractor Pie Chart
    axes[1].pie(distractor_percentages, labels=labels, autopct='%1.1f%%', startangle=90, colors=['green', 'orange', 'red'])
    axes[1].set_title('Distractor Responses')

    # Adjust layout and display
    plt.tight_layout()
    plt.savefig(fig_path/ f'{sub_input}_{condition}_distractor_vs_target_percentages')

plot_response_pie_charts(df)


