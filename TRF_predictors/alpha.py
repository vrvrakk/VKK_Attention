'''
get absolute alpha power from occipitoparietal electrodes
get relative alpha from: (occipito − rest) / (occipito + rest)
Steps:
1. Bandpass filter or compute PSD (e.g. Welch) in 8–12 Hz.
2. Split electrodes into:
    Occipito-parietal set (PO, O, P, CP etc.)
    Rest (all other electrodes).
3. Compute log-power for both sets.
4. Take median occipito-parietal power ÷ median rest power → relative alpha ratio.
5. That gives you one value per subject (classical occipital-dominance measure).
'''

# 1. import libraries:
import os
from pathlib import Path
import pickle as pkl

import numpy as np
import mne
from copy import deepcopy
import logging
from scipy.signal import hilbert


# 2: define directories and params:

# conditions dict:
conditions = {
    'a1': {'target': 'stream1', 'distractor': 'stream2'},
    'e1': {'target': 'stream1', 'distractor': 'stream2'},
    'a2': {'target': 'stream2', 'distractor': 'stream1'},
    'e2': {'target': 'stream2', 'distractor': 'stream1'}}

base_dir = Path.cwd()
data_dir = base_dir / 'data' / 'eeg'
predictor_dir = data_dir / 'predictors'
bad_segments_dir = predictor_dir / 'bad_segments'
eeg_dir = Path('D:/VKK_Attention/data/eeg/preprocessed/results')


sfreq = 125


def load_eeg(condition):
    eeg_list = []
    for sub_folders in eeg_dir.iterdir():
        if 'sub' in sub_folders.name:
            eeg_folder = sub_folders / 'ica'
            for eeg_files in eeg_folder.iterdir():
                if condition in eeg_files.name:
                    eeg = mne.io.read_raw_fif(eeg_files, preload=True)
                    # filter up to 8 Hz, resample if necessary and drop occipitoparietal channels + avg ref
                    eeg_filt = eeg.filter(l_freq=None, h_freq=20) #todo: changed bp
                    eeg_resamp = eeg_filt.resample(sfreq)
                    eeg_avg = eeg_resamp.set_eeg_reference('average')
                    # eeg_ch = eeg_filt.pick([ch for ch in eeg_avg.ch_names if not ch.startswith(('O', 'PO'))])
                    eeg_list.append(eeg_avg) # todo: change back
    return eeg_list


def mask_eeg():
    eeg_masked_dict = {}
    for sub, eeg_arr in eeg_dict.items():
        bad_series = bads_dict[sub][0] # lol -.-
        # concatenate EEG data per subject:
        eeg_concat = deepcopy(eeg_arr)
        eeg_concat = mne.concatenate_raws(eeg_concat)
        # length check prior masking:
        logging.basicConfig(level=logging.INFO)
        logging.info(f"EEG length: {len(eeg_concat)}, bads length: {len(bad_series)}")
        assert len(eeg_concat) == len(bad_series), "Mismatch between EEG and bad segments length!"
        # mask along the array of corresponding sub:
        eeg_data = eeg_concat.get_data()
        eeg_masked = eeg_data[:, bad_series == 0]
        # z-score EEG data:
        eeg_clean = (eeg_masked - eeg_masked.mean(axis=1, keepdims=True)) / eeg_masked.std(axis=1, keepdims=True)

        eeg_masked_dict[sub] = eeg_clean
    return eeg_masked_dict


def filter_alpha(data):
    # 1. band-pass filter 8–12 Hz
    data_filt = mne.filter.filter_data(data, sfreq, l_freq=8, h_freq=12)
    # 2. Hilbert transform to get amplitude envelope
    analytic = hilbert(data_filt, axis=-1)
    alpha_power = np.abs(analytic)**2
    # 3. log-transform + z-score
    alpha_log = np.log(alpha_power + 1e-6)
    alpha_z = (alpha_log - alpha_log.mean()) / alpha_log.std()
    alpha_avg = np.mean(alpha_z, axis=0)
    return alpha_avg


if __name__ == '__main__':

    for condition in list(conditions.keys()):
        eeg_list = load_eeg(condition)

        sub_list = []
        for index, eeg_file in enumerate(eeg_list):
            filename = eeg_list[index].filenames[0]
            subject = os.path.basename(filename).split('_')[0]  # get base name of fif file (without path), split by _,
            # and get first value (sub)
            if subject not in sub_list:
                sub_list.append(subject)

        # placeholder for EEG data / subject
        eeg_dict = {}
        for subject in sub_list:
            eeg_arrays = []
            for eeg_file in eeg_list:
                eeg_name = eeg_file.filenames[0]
                if subject in eeg_name:
                    eeg_arrays.append(eeg_file)
                eeg_dict[subject] = eeg_arrays

        bads_dict = {}
        for sub_folders in bad_segments_dir.iterdir():
            bads = []
            for cond_folders in sub_folders.iterdir():
                if condition in cond_folders.name:
                    for files in cond_folders.iterdir():
                        if 'concat.npy.npz' in files.name:
                            bad_array = np.load(files, allow_pickle=True)
                            bad_array = bad_array['bad_series']
                            bads.append(bad_array)
            if sub_folders.name in sub_list:
                bads_dict[sub_folders.name] = bads

        eeg_masked_dict = mask_eeg()

        eeg_chs = np.array(eeg_list[0].ch_names)
        occ_chs = np.array([ch for ch in eeg_chs if ch.startswith(('O', 'PO'))])

        # Boolean mask: True for channels to keep (non-occipital)
        keep_mask = ~np.isin(eeg_chs, occ_chs)

        # Example usage
        eeg_masked = eeg_chs[keep_mask]   # EEG without occipito-parietal channels
        eeg_occ = eeg_chs[~keep_mask]  # only occipito-parietal channels (for alpha)

        alpha_dir = data_dir / 'journal' / 'alpha' / condition
        alpha_dir.mkdir(parents=True, exist_ok=True)

        alpha_dict = {}
        for sub, eeg_arr in eeg_masked_dict.items():
            # get alpha
            print(eeg_arr.shape)
            occ_eeg_data = eeg_arr[~keep_mask, :]
            rest_eeg_data = eeg_arr[keep_mask, :]
            occ_z = filter_alpha(occ_eeg_data)
            rest_z = filter_alpha(rest_eeg_data)
            alpha_ratio = (occ_z - rest_z) / (occ_z + rest_z)
            alpha_dict[sub] = {'occ_alpha': occ_z, 'alpha_ratio': alpha_ratio}

        # save alpha_dict
        filename = f'{condition}_alpha.pkl'

        with open(alpha_dir/filename, 'wb') as f:
            pkl.dump(alpha_dict, f)
            print(f'Saved {filename} in {alpha_dir}.')



