from pathlib import Path
import os
import matplotlib.pyplot as plt
import numpy as np
import mne
from EEG.extract_events import default_path
from EEG.params import sub_list, exceptions, event_types, conditions, channels, occipital_channels, motor_channels

results_path = default_path / 'data' / 'eeg' / 'preprocessed' / 'results'
epochs_path = results_path / 'concatenated_data' / 'epochs'
concat_path = results_path / 'concatenated_data' / 'epochs'
# time to subtract baseline from ERPs:
fig_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/concatenated_data/figures')

def get_merged_epochs(event_type):
    all_concat_epochs = {}

    for sub in sub_list:
        all_concat_epochs[sub] = {}
        if sub in exceptions or sub == 'sub08':
            continue
        else:
            print(sub)
            all_concat_epochs[sub][event_type] = {}
            sub_path = concat_path / sub / selected_ch / event_type / f'{sub}_{condition}_{event_type}_{selected_ch}_concatenated-epo.fif'
            if not sub_path.exists():
                print(f'{sub} path for {event_type} does not exist.')
                continue
            selected_epochs = mne.read_epochs(sub_path, preload=True)

            selected_epochs_corrected = selected_epochs.copy()
            all_concat_epochs[sub][event_type] = selected_epochs_corrected
    # Initialize dictionary to store all epochs of each event type from all subjects
    all_concat_epoch_types = {event_type: []}

    # Loop through subjects and collect epochs per event type
    for sub, epochs_dict in all_concat_epochs.items():
        if epochs_dict:
            for event_type, epoch in epochs_dict.items():
                if len(epoch) > 0:  # Ensure epochs exist for this event type
                    n_epochs = len(epoch.events)
                    all_concat_epoch_types[event_type].append((n_epochs, epoch))  # Append without overwriting

    for event_type in all_concat_epoch_types.keys():
        epochs_info = all_concat_epoch_types[event_type]

        if epochs_info:  # Ensure there is data
            # Extract only the second element (epochs) from all tuples
            epochs_list = [ep_tuple[1] for ep_tuple in epochs_info]
            total_epochs = sum(ep_tuple[0] for ep_tuple in epochs_info)  # Sum total trials
            num_subjects = len(epochs_info)  # Count number of subjects

            # Concatenate epochs from all subjects
            merged_epochs = mne.concatenate_epochs(epochs_list)
            merged_epochs_path = Path(results_path / 'concatenated_data' / 'epochs' / 'all_subs')
            merged_epochs.save(merged_epochs_path / f'{condition}_{selected_ch}_{event_type}-epo.fif', overwrite=True)

            # Compute and plot ERP
            merged_epochs.drop_channels(['FCz'])
    return merged_epochs, total_epochs, num_subjects


def get_erp(merged_epochs, total_epochs, num_subjects, fmax=None):
    whole_erp = merged_epochs.average()
    whole_erp.apply_baseline((-0.2, 0.0))
    fig_sub_path = fig_path / selected_ch / event_type
    os.makedirs(fig_sub_path, exist_ok=True)
    whole_erp.plot_topo()
    plt.savefig(fig_sub_path / f'{condition}_ERP_topo_{event_type}.png', dpi=300)
    plt.close()
    whole_erp.plot()
    plt.savefig(fig_sub_path / f'{condition}_ERP_{event_type}.png', dpi=300)
    plt.close()
    titles = f'ERP for {event_type.replace("_", " ")} | {total_epochs} epochs | {num_subjects} subjects'
    mne.viz.plot_compare_evokeds(whole_erp, title=titles, legend=False, combine='mean')
    plt.savefig(fig_sub_path / f'{condition}_ERP_average_{event_type}.png', dpi=300)
    plt.close()
    if fmax != 30:
        erp_filt = whole_erp.copy().filter(l_freq=None, h_freq=fmax)
        titles = f'ERP for {event_type.replace("_", " ")} | {total_epochs} epochs | {num_subjects} subjects'
        mne.viz.plot_compare_evokeds(erp_filt, title=titles, legend=False, combine='mean')
        plt.savefig(fig_sub_path / f'{condition}_ERP_average_{event_type}_{fmax}.png', dpi=300)
        plt.close()
    return whole_erp


def get_tfa(merged_epochs, total_epochs, num_subjects, fmax=None):
    freqs = np.linspace(1, fmax, num=100)  # 30 log-spaced frequencies
    n_cycles = freqs / 2  # Define cycles per frequency (standard approach)
    power = mne.time_frequency.tfr_multitaper(merged_epochs, freqs=freqs,
                                              n_cycles=n_cycles,
                                              average=True,
                                              return_itc=False,
                                              decim=1, n_jobs=1)

    # === Plot Time-Frequency Heatmap ===
    fig, ax = plt.subplots(figsize=(8, 6))
    power.plot(picks=['Cz'], baseline=(-0.2, 0), mode='percent',
               title=f'TFA Heatmap: {event_type.replace("_", " ")} | '
                     f'{total_epochs} epochs | {num_subjects} subjects', axes=ax, cmap='viridis')
    fig_sub_path = fig_path / selected_ch / event_type
    os.makedirs(fig_sub_path, exist_ok=True)
    fig.savefig(fig_sub_path / f'{condition}_tfa_{event_type}_{fmax}_Hz.png', dpi=300)
    plt.show()
    plt.close(fig)
    return power


if __name__ == '__main__':
    condition = conditions[0]
    selected_ch = channels[1]
    event_type = event_types[0]
    # 0: animal sounds, 1: targets_with_valid_responses
    # 5: distractors_with_valid_responses (+ 6 + 7)
    # 8: distractors_without_responses, 12: non_targets_targets_no_response
    # 16: non_targets_distractor_no_response

    merged_epochs, total_epochs, num_subjects = get_merged_epochs(event_type)
    # for 1-30 Hz
    # whole_erp_all_bands = get_erp(merged_epochs, total_epochs, num_subjects, fmax=30)
    # power_all_bands = get_tfa(merged_epochs, total_epochs, num_subjects, fmax=30)
    # # for 1-7 Hz:
    whole_erp_theta = get_erp(merged_epochs, total_epochs, num_subjects, fmax=8)
    power_theta = get_tfa(merged_epochs, total_epochs, num_subjects, fmax=8)