from pathlib import Path

import matplotlib.pyplot as plt
import mne
import os
import numpy as np

default_path = Path.cwd()
motor_erp = default_path / 'data/eeg/preprocessed/results/erp'

for files in motor_erp.iterdir():
    if 'motor_smooth_erp-ave.fif' in files.name:
        motor_erp = mne.read_evokeds(files)
        motor_erp[0].resample(125)


rt_path = default_path / 'data/eeg/predictors/RTs'

condition = 'e2'
rt_targets_subs = {}
rt_distractors_subs = {}
def get_rt_pred(rt_subs_dict, stream_type):
    for sub in rt_path.iterdir():
        for sub_folder in sub.iterdir():
            if condition in sub_folder.name:
                for files in sub_folder.iterdir():
                    if stream_type in files.name:
                        for np_files in files.iterdir():
                            if 'concat' in np_files.name:
                                rt_targets = np.load(np_files)
                                rt_targets = rt_targets['RTs']
                                rt_subs_dict[sub.name] = rt_targets
    return rt_subs_dict

rt_targets_subs = get_rt_pred(rt_targets_subs, stream_type='stream1')
rt_distractors_subs = get_rt_pred(rt_distractors_subs, stream_type='stream2')

def count_nonzero_segments(array):
    count = 0
    in_segment = False
    for i in range(len(array)):
        if array[i] != 0 and not in_segment:
            in_segment = True
            count += 1
        elif array[i] == 0 and in_segment:
            in_segment = False
    return count

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


def concat_rt_arrays(rt_masked):
    rt_masked_list = [array for array in rt_masked.values()]
    rt_list = np.concatenate(rt_masked_list)
    print(len(rt_list))
    return rt_list


rt_targets_concat = concat_rt_arrays(rt_targets_masked)
rt_distractors_concat = concat_rt_arrays(rt_distractors_masked)

# get motor_erp data:
motor_data = motor_erp[0].get_data().T
motor_data_avg = np.mean(motor_data, axis=1)
# import matplotlib.pyplot as plt
# plt.figure(figsize=(10, 4), dpi=300)
# time_len = np.arange(len(motor_data_avg))
# plt.plot(time_len, motor_data_avg)
# plt.xlabel('Time samples')
# plt.ylabel('Amplitude (uV)')
# plt.title('Average motor ERP')
# plt.tight_layout()

# convolve rt_masked array with motor_erp
rt_targets_concat_tagged = rt_targets_concat != 0
rt_distractors_concat_tagged = rt_distractors_concat != 0

# assign motor_erps for True values:
motor_pred_target = np.zeros(len(rt_targets_concat_tagged))
motor_pred_distractor = np.zeros(len(rt_distractors_concat_tagged))
def assign_motor_erp(rt_targets_tagged, motor_pred):
    in_segment = False
    for i in range(len(rt_targets_tagged) - 1):
        current = rt_targets_tagged[i]
        next = rt_targets_tagged[i + 1]
        if current and not in_segment:
            in_segment = True
        if in_segment and current and not next:
            response_idx = i
            segment_dur = response_idx + len(motor_data_avg)
            if segment_dur <= len(motor_pred):
                motor_pred[response_idx:segment_dur] += motor_data_avg
                # Using += ensures values get added instead of overwritten.
            else:
                samples_left = len(motor_pred) - response_idx
                motor_pred[response_idx:] += motor_data_avg[:samples_left]
            in_segment = False
    return motor_pred

target_motor_pred = assign_motor_erp(rt_targets_concat_tagged, motor_pred_target)
distractor_motor_pred = assign_motor_erp(rt_distractors_concat_tagged, motor_pred_distractor)

n_insertions = np.count_nonzero(motor_pred_distractor) // 125
print(f"Motor ERP inserted {n_insertions} times in target predictor.")


# plt.figure(figsize=(12, 4), dpi=150)
# plt.plot(motor_pred_target)
# plt.xlabel("Time (samples)")
# plt.ylabel("Motor ERP Predictor")
# plt.title("Motor ERP Predictor for Distractor Responses")
# plt.tight_layout()
# plt.show()

# save files:
save_path = rt_path / 'concatenated' / condition
save_path.mkdir(parents=True, exist_ok=True)
stream_type = 'stream1_stream2'
filename = f'{condition}_{stream_type}_motor_pred_concat.npz'
np.savez(save_path/filename,
         motor_pred_target=target_motor_pred,
         motor_pred_distractor=distractor_motor_pred)