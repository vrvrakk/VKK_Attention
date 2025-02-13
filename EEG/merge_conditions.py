import mne
import matplotlib.pyplot as plt
import os
from pathlib import Path

default_path = Path.cwd()
data_path = default_path / 'data' / 'eeg' / 'preprocessed' / 'results' / 'concatenated_data' / 'epochs' / 'all_subs'
conditions = ['a1', 'a2', 'e1', 'e2']
planes = ['Azimuth', 'Elevation']
selected_ch = 'attention'
event_type = 'targets_with_valid_responses'
# 0: animal sounds, 1: targets_with_valid_responses, 5: distractors_with_valid_responses (+ 6 + 7)
# 8: distractors_without_responses, 12: non_targets_targets_no_response, 17: non_targets_distractor_no_response
selected_plane = planes[0]
if selected_plane == 'Azimuth':
    a1 = conditions[0]
    a2 = conditions[1]
    data_name_a1 = f'{a1}_{selected_ch}_{event_type}-epo.fif'
    data_name_a2 = f'{a2}_{selected_ch}_{event_type}-epo.fif'

    a1_target_resp_epochs = mne.read_epochs(data_path / data_name_a1, preload=True)
    a2_target_resp_epochs = mne.read_epochs(data_path / data_name_a2, preload=True)

    num_subjects = 15
    azimuth_epochs = mne.concatenate_epochs([a2_target_resp_epochs, a2_target_resp_epochs])
    total_epochs = len(azimuth_epochs.events)
    azimuth_erp = azimuth_epochs.average()
    mne.viz.plot_compare_evokeds(
        azimuth_erp,
        title=f"Azimuth | {total_epochs} epochs | {num_subjects} subjects - ERP ({azimuth_erp.info['highpass']}-{azimuth_erp.info['lowpass']} Hz)",
        combine='mean',
        show_sensors='upper right',
        cmap='viridis'
    )
    # azimuth_low_freq_erp = azimuth_erp.filter(l_freq=None, h_freq=6)
    # mne.viz.plot_compare_evokeds(
    #     azimuth_low_freq_erp,
    #     title=f"Azimuth across subjects - ERP ({azimuth_low_freq_erp.info['highpass']}-{azimuth_low_freq_erp.info['lowpass']} Hz)",
    #     combine='mean',
    #     show_sensors='upper right',
    #     cmap='viridis'
    # )