import mne
import matplotlib.pyplot as plt
import os
from pathlib import Path
import numpy as np
from EEG.extract_events import sub_list
from EEG.params import conditions, event_types
default_path = Path.cwd()
data_path = default_path / 'data' / 'eeg' / 'preprocessed' / 'results' / 'concatenated_data' / 'epochs' / 'all_subs'
evokeds_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/concatenated_data/evokeds')
planes = ['Azimuth', 'Elevation']
selected_ch = 'attention'
# 0: animal sounds, 1: targets_with_valid_responses
# 5: distractors_with_valid_responses (+ 6 + 7)
# 8: distractors_without_responses, 12: non_targets_targets_no_response
# 16: non_targets_distractor_no_response

event_type = event_types[0]
fig_path = default_path / 'data/eeg/preprocessed/results/concatenated_data/figures'
selected_plane = planes[0]
fmax = 10
if selected_plane == 'Azimuth':
    a1 = conditions[0]
    a2 = conditions[1]
    data_name_a1 = f'{a1}_{selected_ch}_{event_type}-epo.fif'
    data_name_a2 = f'{a2}_{selected_ch}_{event_type}-epo.fif'

    a1_target_resp_epochs = mne.read_epochs(data_path / data_name_a1, preload=True)
    a2_target_resp_epochs = mne.read_epochs(data_path / data_name_a2, preload=True)

    azimuth_epochs = mne.concatenate_epochs([a2_target_resp_epochs, a2_target_resp_epochs])
    azimuth_epochs.drop_channels(['FCz'])
    total_epochs = len(azimuth_epochs.events)
    # freqs = np.linspace(1, fmax, num=100)  # 30 log-spaced frequencies
    # n_cycles = freqs / 2  # Define cycles per frequency (standard approach)
    # power = mne.time_frequency.tfr_multitaper(azimuth_epochs,
    #                                           freqs=freqs,
    #                                           n_cycles=n_cycles,
    #                                           average=True,
    #                                           return_itc=False,
    #                                           decim=1,
    #                                           n_jobs=1)
    #
    # # === Plot Time-Frequency Heatmap ===
    # freq_range = (1, fmax)
    # fig, ax = plt.subplots(figsize=(8, 6))
    # power.plot(picks=['Cz'], baseline=(-0.2, 0), mode='percent',
    #            title=f'TFA Heatmap: {event_type.replace("_", " ")} | {total_epochs} epochs | all subjects',
    #            axes=ax, cmap='viridis', fmin=freq_range[0], fmax=freq_range[1], vmin=-0.2)
    # fig.savefig(fig_path / f'tfa_{selected_plane}_all_subs_{selected_ch}_{event_type}_{freq_range[1]}_Hz.png')
    # plt.show()
    # plt.close(fig)


# get ERPs with std error:
ave_fif_list1 = []
ave_fif_list2 = []
for sub in sub_list:
    if sub == 'sub16':
        continue
    if selected_plane == 'Azimuth':
        ave_path1 = evokeds_path / conditions[0] / event_type
        ave_path2 = evokeds_path / conditions[1] / event_type
    elif selected_plane == 'Elevation':
        ave_path1 = evokeds_path / conditions[2] / event_type
        ave_path2 = evokeds_path / conditions[3] / event_type
for ave_fif1, ave_fif2 in zip(ave_path1.iterdir(), ave_path2.iterdir()):
    evoked1 = mne.read_evokeds(ave_fif1)[0]
    evoked1.filter(l_freq=None, h_freq=fmax)
    evoked2 = mne.read_evokeds(ave_fif2)[0]
    evoked2.filter(l_freq=None, h_freq=fmax)
    ave_fif_list1.append(evoked1)
    ave_fif_list2.append(evoked2)

ave_concat_list = (np.concatenate((ave_fif_list1, ave_fif_list2))).tolist()
total_erps = len(ave_concat_list)
grand_average = mne.grand_average(ave_concat_list)  # Averages all evokeds
mne.viz.plot_compare_evokeds([ave_concat_list],
                             title=f"Azimuth | | {event_type.replace('_', ' ')} |{total_erps} subjects | ERP ({ave_concat_list[0].info['highpass']}-{ave_concat_list[0].info['lowpass']} Hz)",
                             combine='mean',
                             show_sensors='upper right',
                             legend=False,
                             ci=0.95
                             )
plt.savefig(fig_path / f'erp_{selected_plane}_{selected_ch}_{event_type}_{fmax}_Hz_all_subs.png')
# plt.close()

#
# h_freq = 6
# ave_concat_list_copy = ave_concat_list.copy()
# for ave_fifs in ave_concat_list_copy:
#     ave_fifs.filter(l_freq=None, h_freq=6)
# mne.viz.plot_compare_evokeds(
#     [ave_concat_list_copy],
#     title=f"Azimuth across subjects | {event_type.replace('_', ' ')} - ERP ({ave_concat_list_copy[0].info['highpass']}-{ave_concat_list_copy[0].info['lowpass']} Hz)",
#     combine='mean',
#     show_sensors='upper right',
#     legend=False,
#     ci=0.95
# )
# plt.savefig(fig_path / f'erp_{selected_plane}_{selected_ch}_{event_type}_{h_freq}_Hz_all_subs.png')
# plt.close()
