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
# extract and load baseline eeg files:
baseline_header_files = EEG.extract_events.extract_eeg_files(condition='baseline.vhdr')
baseline_eeg_files = EEG.extract_events.load_eeg_files(baseline_header_files)

def add_montage(baseline_eeg_files, condition=''):
    for index, (sub, eeg) in enumerate(zip(sub_list, baseline_eeg_files)):
        eeg.resample(sfreq=500)  # downsample from 1000Hz to 500Hz
        eeg.rename_channels(mapping)
        eeg.add_reference_channels('FCz')  # add reference channel
        eeg.set_montage('standard_1020')  # apply standard montage
        eeg.drop_channels(['A1', 'A2', 'M2'])
        eeg.save(results_path / sub / 'baseline'
                 / f'{sub}_{condition}_downsampled-raw.fif', overwrite=True)
        baseline_eeg_files[index] = eeg
    return baseline_eeg_files


baseline_eeg_files = add_montage(baseline_eeg_files, condition='baseline')

for baseline_eeg in baseline_eeg_files:
    baseline_eeg.plot()
    baseline_eeg.plot_psd()

for sub, baseline_eeg in zip(sub_list, baseline_eeg_files):
    baseline_path = results_path / sub / 'baseline'
    os.makedirs(baseline_path, exist_ok=True)
    baseline_eeg.save(baseline_path/f'{sub}_baseline-raw.fif', overwrite=True)



baseline_interpolate = copy.deepcopy(baseline_eeg_files)
for index, (sub, baseline_eeg) in enumerate(zip(sub_list, baseline_interpolate)):
    baseline_interpolated = baseline_eeg.interpolate_bads(reset_bads=True)
    baseline_interpolated.save(results_path/ sub /
                               'baseline' / f'baseline_interpolated{sub}-raw.fif', overwrite=True)
    baseline_interpolate[index] = baseline_interpolated


baseline_filter = copy.deepcopy(baseline_interpolate)

def filter_eeg(baseline_filter, freq_range=(1, 30, 1), condition='baseline'):
    for index, (sub, eeg_files) in enumerate(zip(sub_list, baseline_filter)):
        eeg_filter = eeg_files.copy()
        data = mne.io.RawArray(data=eeg_files.get_data(), info=eeg_files.info)
        eeg_notch, iterations = dss.dss_line(eeg_files.get_data().T, fline=50,
                                             sfreq=data.info["sfreq"],
                                             nfft=400)

        eeg_filter._data = eeg_notch.T
        hi_filter = freq_range[0]
        lo_filter = freq_range[1]

        eeg_filtered = eeg_filter.copy().filter(hi_filter, lo_filter)
        eeg_filtered.save( results_path/ sub/ 'baseline'/  f'{sub}_{condition}_filtered_{freq_range[1]}-raw.fif', overwrite=True)
        baseline_filter[index] = eeg_filtered
    return baseline_filter


baseline_filtered = filter_eeg(baseline_filter, freq_range=(1, 30, 1), condition='baseline')

baseline_ica = copy.deepcopy(baseline_filtered)

index = 18  # repeat ICA application for all subs
sub = sub_list[index]
condition = 'baseline'
# a. fit ICA:
eeg_file = baseline_ica[index]  # change variable according to condition
eeg_ica = eeg_file.copy()
eeg_ica.info['bads'].append('FCz')  # Add FCz to the list of bad channels
ica = mne.preprocessing.ICA(n_components=0.999, method='fastica', random_state=99)
ica.fit(eeg_ica)  # bad segments that were marked in the EEG signal will be excluded.
# b. investigate...:
ica.plot_components()
# ica.plot_sources(eeg_ica)
# c. apply ICA to remove selected components: blinks, eye movements etc.
ica.apply(eeg_ica)
# d. re-reference with average:
eeg_ica.info['bads'].remove('FCz')
eeg_ica.set_eeg_reference(ref_channels='average')
# e. save
baseline_ica[index] = eeg_ica
eeg_ica.save(results_path / sub / 'baseline' / f'{sub}_{condition}_{index}_ica-raw.fif', overwrite=True)

# once done:
del index


# create epochs and ERPs smoothed:
for index, (sub, eeg_ica) in enumerate(zip(sub_list, baseline_ica)):
    events = mne.events_from_annotations(eeg_ica)[0]  # get events from annotations attribute of raw variable
    baseline_unique_annotations = np.unique(eeg_ica.annotations.description)
    baseline_reject_annotation = None
    for unique_annotation in baseline_unique_annotations:
        if 'BAD' in unique_annotation:
            baseline_reject_annotation = unique_annotation
        elif 'bad' in unique_annotation:
            baseline_reject_annotation = unique_annotation

    baseline_len = eeg_ica.n_times
    baseline_sfreq = eeg_ica.info['sfreq']
    baseline_time_window = (-0.2, 0.9)
    baseline_time_window_len = 1.2 #s
    baseline_epoch_len = int(baseline_time_window_len * baseline_sfreq)
    baseline_n_events = round(baseline_len / baseline_epoch_len)
    baseline_event_times = np.linspace(0, baseline_len - baseline_epoch_len, baseline_n_events)
    # If we put events at the very end of the EEG, there wouldn’t be enough data to create full epochs of -0.2s to 0.9s
    baseline_events = np.column_stack([baseline_event_times, np.zeros(baseline_n_events, dtype=int), np.ones(baseline_n_events, dtype=int)])
    # [event[0], event[1], event[2]] -> time, 0, stimulus id
    baseline_epochs = mne.Epochs(eeg_ica, baseline_events.astype(int), event_id=1, tmin=-0.2, tmax=0.9, baseline=(None, 0), reject_by_annotation=baseline_reject_annotation, preload=True)
    # now time for padding the ERP:
    baseline_erp = baseline_epochs.average()
    duration = 0.2  # Length of ramp in seconds
    sfreq = baseline_erp.info['sfreq']  # Sampling frequency (e.g., 500 Hz)
    size = int(duration * sfreq)  # Ramp size in data samples

    # Step 2: Define cosine envelope function
    envelope = lambda t: 0.5 * (1 + np.cos(np.pi * (t - 1)))  # Cosine envelope
    ramp_multiplier = envelope(np.linspace(0.0, 1.0, size))  # Generate ramp multiplier

    # Step 3: Smooth edges of the ERP data
    baseline_erp_smooth = baseline_erp.copy()  # Create a copy to preserve the original ERP
    for ch in range(baseline_erp_smooth.data.shape[0]):  # Loop over each channel
        baseline_erp_smooth.data[ch, :size] *= ramp_multiplier  # Apply ramp at the beginning
        baseline_erp_smooth.data[ch, -size:] *= ramp_multiplier[::-1]  # Apply ramp at the end
    # Step 8: Plot the padded ERP
    baseline_erp_smooth.plot()
    baseline_erp_smooth.save(results_path / sub / 'baseline' / f'{sub}_baseline_erp-ave.fif', overwrite=True)
    print(f'{sub} baseline ERP saved in: results_path / {sub} / baseline')




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