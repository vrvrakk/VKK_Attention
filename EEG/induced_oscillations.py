from pathlib import Path
import mne
import matplotlib.pyplot as plt
import numpy as np

epochs_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/concatenated_data/epochs/all_subs')



if __name__ == '__main__':
    epochs_list = []
    epochs_names = []
    for fif_files in epochs_path.iterdir():
        if 'fif' in fif_files.name:
            file_name = fif_files.name[:-8]
            epochs = mne.read_epochs(fif_files, preload=True)
            epochs.set_eeg_reference('average') # apply avg ref before getting TFA
            epochs_list.append(epochs)
            epochs_names.append(file_name)


    index = 22
    epoch_type = epochs_names[index]
    # TFA each epoch:
    freqs = np.logspace(np.log10(1), np.log10(30), num=100)  # 30 log-spaced frequencies
    n_cycles = freqs / 2  # Define cycles per frequency (standard approach)
    power = mne.time_frequency.tfr_multitaper(epochs_list[index], freqs=freqs,
                                              picks=['Cz'],
                                              n_cycles=n_cycles,
                                              average=False,  # to get TFA of EACH trial, not averaged
                                              return_itc=False,
                                              decim=1, n_jobs=1)
    power_data = power.data.mean(axis=0)  # it needs n channels, freqs, samples (reducing 0th dimension - n epochs)
    # 2. Make an AverageTFR object
    induced = mne.time_frequency.AverageTFR(
        info=power.info,
        data=power_data,
        times=power.times,
        freqs=power.freqs,
        nave=power.data.shape[0],  # number of averaged epochs
        method='multitaper'
    )

    fig = induced.plot(picks='Cz', baseline=(-0.2, 0), mode='logratio', fmin=1, fmax=30)
    save_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/concatenated_data/epochs/all_subs/induced_tfa')
    print(epoch_type)
    fig[0].savefig(save_path/f'{epoch_type}_induced_TFR.png')


    #todo: induced per sub
    single_sub_epochs = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/eeg/preprocessed/results/concatenated_data/epochs')
    for sub_folders in single_sub_epochs.iterdir():
        if 'sub' in sub_folders.name:
            for folders in sub_folders.iterdir():
                if folders.name == 'attention':
                    for epochs_folders in folders.iterdir():
                        if epochs_folders.name == epoch_type[13:]:
                            for fif_files in epochs_folders.iterdir():
                                print(fif_files.name[:-8]




