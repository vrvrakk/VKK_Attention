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

for baseline_eeg in baseline_eeg_files:
    baseline_eeg.plot()

for sub, baseline_eeg in zip(sub_list, baseline_eeg_files):
    baseline_path = results_path / sub / 'baseline'
    os.makedirs(baseline_path, exist_ok=True)
    baseline_eeg.save(baseline_path/f'{sub}_baseline-raw.fif')

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

baseline_interpolate = copy.deepcopy(baseline_eeg_files)
for index, (sub, baseline_eeg) in enumerate(zip(sub_list, baseline_interpolate)):
    baseline_interpolated = baseline_eeg.interpolate_bads(reset_bads=True)
    baseline_interpolated.save(results_path/ sub /
                               'baseline' / f'baseline_interpolated{sub}-raw.fif')
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
        eeg_filtered.save( results_path/ f'{sub}_{condition}_filtered_{freq_range[1]}-raw.fif', overwrite=True)
        baseline_filter[index] = eeg_filtered
    return baseline_filter


baseline_filtered = filter_eeg(baseline_filter, freq_range=(1, 30, 1), condition='baseline')

baseline_ica = copy.deepcopy(baseline_filtered)

index = 4  # repeat ICA application for all subs
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


