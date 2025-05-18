from pathlib import Path
import mne
import os
import numpy as np

default_path = Path.cwd()
motor_erp = default_path/ 'data/eeg/preprocessed/results/erp'

for files in motor_erp.iterdir():
    if 'motor_smooth_erp-ave.fif' in files.name:
        motor_erp = mne.read_evokeds(files)
        motor_erp[0].resample(125)


rt_path = default_path / 'data/eeg/predictors/RTs'

condition = 'a1'
rt_targets_subs = {}
rt_distractors_subs = {}
for sub in rt_path.iterdir():
    for sub_folder in sub.iterdir():
        if condition in sub_folder.name:
            for files in sub_folder.iterdir():
                if 'targets' in files.name:
                    for np_files in files.iterdir():
                        if 'concat' in np_files.name:
                            rt_targets = np.load(np_files)
                            rt_targets = rt_targets['RTs']
                            rt_targets_subs[sub.name] = rt_targets
                elif 'distractors' in files.name:
                    for np_files in files.iterdir():
                        if 'concat' in np_files.name:
                            rt_distractors = np.load(np_files)
                            rt_distractors = rt_distractors['RTs']
                            rt_distractors_subs[sub.name] = rt_distractors

# RTs are also based on sfreq of 125Hz
# remove bad segments:
bad_segments_path = default_path / 'data/eeg/predictors/bad_segments'
bad_segments_dict = {}
for sub in bad_segments_path.iterdir():
    for folders in sub.iterdir():
        if condition in folders.name:
            for files in folders.iterdir():
                if 'concat.npy' in files.name:
                    bad_segments = np.load(files)
                    bad_segments = bad_segments['bad_series']
                    bad_segments_dict[sub.name] = bad_segments

def mask_rt_arrays(rt_dict, bad_segments_dict):
    rt_masked = {}
    for sub_rt, rt_array in rt_dict.items():
        bad_array = bad_segments_dict[sub_rt]
        good_samples = bad_array == 0  # Boolean mask, keep zeros (-999 are bad segments)
        rt_masked[sub_rt] = rt_array[good_samples]
    return rt_masked

rt_targets_masked = mask_rt_arrays(rt_targets_subs, bad_segments_dict)
rt_distractors_masked = mask_rt_arrays(rt_distractors_subs, bad_segments_dict)

# get motor_erp data:
motor_data = motor_erp[0].get_data().T
# convolve rt_masked array with motor_erp
for sub, rt_array in rt_targets_masked.items():
    rt_samples = rt_array == 1


