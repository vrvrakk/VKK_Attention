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
    common_channels_dict[filename] = common_channels

channels = list(common_channels_dict.values())

# make sure each element is a set
channels_as_sets = [set(ch) for ch in channels]

# intersection across all sets
common_all = set.intersection(*channels_as_sets)

print("Common across all files:", common_all)

final_roi = np.array(list(common_all))