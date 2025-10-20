'''
This is a very caveman-like way to select the ROI:
common channels with a corr r value >= 0.1 across all conditions
'''
import os
import pickle as pkl
from pathlib import Path
import numpy as np


base_dir = Path.cwd()
channels_dir = base_dir / 'data' / 'eeg' / 'journal' / 'common_channels'
common_channels_dict = {}

for files in channels_dir.iterdir():
    filename = files.stem
    with open(files, 'rb') as f:
        common_channels = pkl.load(f)
        common_channels_vals = common_channels.values()
    common_channels_dict[filename] = common_channels_vals

# flatten one level: collect all arrays from all sub-dictionaries
all_channels = []
for ch_lists in common_channels_dict.values():
    all_channels.extend(ch_lists)  # extend with each subject's array of channels

# make sure each element is a set
channels_as_sets = [set(ch) for ch in all_channels]
# intersection across all sets
common_all = set.intersection(*channels_as_sets)
print("Common across all files:", common_all)
common_roi = np.array(list(common_all))


# fronto-temporal electrodes and fronto-central electrodes based on lit
lit_roi = np.array(['F1', 'F2', 'F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                    'FC1', 'FC2', 'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8',
                    'FCz', 'Fz', 'Cz'])

# filter out channels not in literature roi:
final_roi = [ch for ch in lit_roi if ch in common_roi]
