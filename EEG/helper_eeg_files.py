import copy
import os
from pathlib import Path
import mne
import numpy as np
import EEG.extract_events
from EEG.extract_events import default_path, eeg_path, blocks_path, sub_list
import EEG.preprocessing_eeg
from EEG.preprocessing_eeg import mapping, dss

results_path = Path(default_path / 'data' / 'eeg' / 'preprocessed' / 'results' )
condition = 'baseline'
# extract and load helper eeg files:
helper_header_files = EEG.extract_events.extract_eeg_files(condition=f'{condition}.vhdr')
helper_eeg_files = EEG.extract_events.load_eeg_files(helper_header_files)

def add_montage(helper_eeg_files, condition=''):
    for index, (sub, eeg) in enumerate(zip(sub_list, helper_eeg_files)):
        eeg.resample(sfreq=500)  # downsample from 1000Hz to 500Hz
        eeg.rename_channels(mapping)
        eeg.add_reference_channels('FCz')  # add reference channel
        eeg.set_montage('standard_1020')  # apply standard montage
        eeg.drop_channels(['A1', 'A2', 'M2'])
        helper_path = results_path / sub / condition
        os.makedirs(helper_path, exist_ok=True)
        eeg.save(helper_path
                 / f'{sub}_{condition}_downsampled-raw.fif', overwrite=True)
        helper_eeg_files[index] = eeg
    return helper_eeg_files


helper_eeg_files = add_montage(helper_eeg_files, condition=condition)

for helper_eeg in helper_eeg_files:
    helper_eeg.plot()
    # helper_eeg.plot_psd()

for sub, helper_eeg in zip(sub_list, helper_eeg_files):
    helper_path = results_path / sub / condition
    os.makedirs(helper_path, exist_ok=True)
    helper_eeg.save(helper_path/f'{sub}_{condition}-raw.fif', overwrite=True)

helper_interpolate = copy.deepcopy(helper_eeg_files)
for index, (sub, helper_eeg) in enumerate(zip(sub_list, helper_interpolate)):
    helper_interpolated = helper_eeg.interpolate_bads(reset_bads=True)
    helper_interpolated.save(results_path / sub /
                               condition / f'{condition}_interpolated_{sub}-raw.fif', overwrite=True)
    helper_interpolate[index] = helper_interpolated


helper_filter = copy.deepcopy(helper_interpolate)

def filter_eeg(helper_filter, freq_range=(1, 30, 1), condition=condition):
    for index, (sub, eeg_files) in enumerate(zip(sub_list, helper_filter)):
        eeg_filter = eeg_files.copy()
        data = mne.io.RawArray(data=eeg_files.get_data(), info=eeg_files.info)
        eeg_notch, iterations = dss.dss_line(eeg_files.get_data().T, fline=50,
                                             sfreq=data.info["sfreq"],
                                             nfft=400)

        eeg_filter._data = eeg_notch.T
        hi_filter = freq_range[0]
        lo_filter = freq_range[1]

        eeg_filtered = eeg_filter.copy().filter(hi_filter, lo_filter)
        eeg_filtered.save( results_path/ sub/ condition/ f'{sub}_{condition}_filtered_{freq_range[1]}-raw.fif', overwrite=True)
        helper_filter[index] = eeg_filtered
    return helper_filter


helper_filtered = filter_eeg(helper_filter, freq_range=(1, 30, 1), condition=condition)
helper_concat = mne.concatenate_raws(helper_filtered)

# a. fit ICA:
eeg_file = helper_concat  # change variable according to condition
eeg_ica = eeg_file.copy()
eeg_ica.info['bads'].append('FCz')  # Add FCz to the list of bad channels
ica = mne.preprocessing.ICA(n_components=0.999, method='picard', random_state=99)
ica.fit(eeg_ica)  # bad segments that were marked in the EEG signal will be excluded.
# b. investigate...:
ica.plot_components()
ica.plot_sources(eeg_ica)
# c. apply ICA to remove selected components: blinks, eye movements etc.
ica.apply(eeg_ica)
# d. re-reference with average:
eeg_ica.info['bads'].remove('FCz')
eeg_ica.set_eeg_reference(ref_channels='average')
# e. save
concat_ica_path = results_path / 'concatenated_data'/'continuous'/'ica'
eeg_ica.save(concat_ica_path / f'concatenated_{condition}_ica-raw.fif', overwrite=True)
# create epochs and ERPs smoothed:
events = mne.events_from_annotations(eeg_ica)[0]  # get events from annotations attribute of raw variable
helper_unique_annotations = np.unique(eeg_ica.annotations.description)
helper_reject_annotation = None
for unique_annotation in helper_unique_annotations:
    if 'BAD' in unique_annotation:
        helper_reject_annotation = unique_annotation
    elif 'bad' in unique_annotation:
        helper_reject_annotation = unique_annotation
helper_epochs_path = results_path / 'concatenated_data' / 'epochs' / f'{condition}_epochs-epo.fif'
if condition == 'baseline':
    helper_len = eeg_ica.n_times
    helper_sfreq = eeg_ica.info['sfreq']
    helper_time_window = (-0.2, 0.5)
    helper_time_window_len = 1.2  # s
    helper_epoch_len = int(helper_time_window_len * helper_sfreq)
    helper_n_events = round(helper_len / helper_epoch_len)
    helper_event_times = np.linspace(0, helper_len - helper_epoch_len, helper_n_events)
    # If we put events at the very end of the EEG, there wouldn’t be enough data to create full epochs of -0.2s to 0.9s
    helper_events = np.column_stack([helper_event_times, np.zeros(helper_n_events, dtype=int), np.ones(helper_n_events, dtype=int)])
    # [event[0], event[1], event[2]] -> time, 0, stimulus id
    helper_epochs = mne.Epochs(eeg_ica, helper_events.astype(int), event_id=1, tmin=-0.2, tmax=0.2, baseline=(-0.2, 0), reject_by_annotation=helper_reject_annotation, preload=True)
    helper_epochs.set_eeg_reference(['FCz'])
    helper_epochs.save(helper_epochs_path, overwrite=True)
elif condition == 'motor':
    helper_events, helper_ids = mne.events_from_annotations(eeg_ica)
    helper_ids = {keys: values for keys, values in helper_ids.items() if keys not in {'New Segment/'}}
    helper_events = [event for event in helper_events if event[2] in helper_ids.values()]
    helper_epochs = mne.Epochs(eeg_ica, helper_events, helper_ids, tmin=-0.1, tmax=0.9, baseline=(-0.1, 0.0), reject_by_annotation=helper_reject_annotation, preload=True)
    helper_epochs.set_eeg_reference(['FCz'])
    helper_epochs.save(helper_epochs_path, overwrite=True)

# now time for padding the ERP:
helper_erp = helper_epochs.average()
helper_erp.plot()
duration = 0.2  # Length of ramp in seconds
sfreq = helper_erp.info['sfreq']  # Sampling frequency (e.g., 500 Hz)
size = int(duration * sfreq)  # Ramp size in data samples

# Step 2: Define cosine envelope function
envelope = lambda t: 0.5 * (1 + np.cos(np.pi * (t - 1)))  # Cosine envelope
ramp_multiplier = envelope(np.linspace(0.0, 1.0, size))  # Generate ramp multiplier

# Step 3: Smooth edges of the ERP data
helper_erp_smooth = helper_erp.copy()  # Create a copy to preserve the original ERP
for ch in range(helper_erp_smooth.data.shape[0]):  # Loop over each channel
    helper_erp_smooth.data[ch, :size] *= ramp_multiplier  # Apply ramp at the beginning
    helper_erp_smooth.data[ch, -size:] *= ramp_multiplier[::-1]  # Apply ramp at the end
# Step 8: Plot the padded ERP
helper_erp_smooth.plot()
concat_helper_epochs_path = results_path / 'erp'
os.makedirs(concat_helper_epochs_path, exist_ok=True)
helper_erp_smooth.save(concat_helper_epochs_path / f'{condition}_smooth_erp-ave.fif', overwrite=True)
print(f'{condition} ERP saved in: {concat_helper_epochs_path}')




# events = events[[not e in [99999] for e in events[:, 2]]]  # remove specific events, if in 2. column
# filtered_events = [event for event in events if str(event[2]) in response_mapping.keys()] # get motor-response events only
# events = np.array(filtered_events)
#
#
# motor_events = get_motor_events(motor_ica, response_mapping)
# motor_markers = motor_events[:, 2]
# response_markers = np.array(list(response_mapping.keys()), dtype='int32')
# filtered_response_markers = motor_markers[np.isin(motor_markers, response_markers)]
# filtered_response_events = [event for event in response_markers if event in filtered_response_markers]