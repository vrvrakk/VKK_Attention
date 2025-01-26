# 1. Import libraries:
import os
from pathlib import Path
import numpy as np
import pandas as pd
import mne
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt

# 2. define params and paths:
default_path = Path.cwd()
eeg_path = default_path / 'data' / 'eeg' / 'raw'
blocks_path = default_path / 'data' / 'params' / 'block_sequences'

condition_list = ['a1', 'a2', 'e1', 'e2']
# 3. create sub_list:
sub_list = []
for i in range(1, 30, 1):
    # .zfill(2):
    # Adds leading zeros to the string until its length is 2 characters.
    string = f'sub{str(i).zfill(2)}'
    if string in ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub07', 'sub08', 'sub09', 'sub12']:
        continue
    else:
        sub_list.append(string)

# 4. extract eeg files:
a1_eeg_header_files = []
a2_eeg_header_files = []
e1_eeg_header_files = []
e2_eeg_header_files = []
for sub in sub_list:
    for folders in eeg_path.iterdir():
        for files in folders.iterdir():
            if files.is_file() and 'vhdr' in files.name and sub in files.name:
                if 'a1' in files.name:
                    a1_eeg_header_files.append(files)
                elif 'a2' in files.name:
                    a2_eeg_header_files.append(files)
                elif 'e1' in files.name:
                    e1_eeg_header_files.append(files)
                elif 'e2' in files.name:
                    e2_eeg_header_files.append(files)
# 5. load eeg files:
def load_eeg_files(eeg_header_files):
    eeg_list = []
    for files in eeg_header_files:
        eeg = mne.io.read_raw_brainvision(files, preload=True)
        eeg_list.append(eeg)
    return eeg_list

# run function: I guess I have to process each file manually :/
a1_eeg = load_eeg_files(a1_eeg_header_files)
# a2_eeg = load_eeg_files(a2_eeg_header_files)
# e1_eeg = load_eeg_files(e1_eeg_header_files)
# e2_eeg = load_eeg_files(e2_eeg_header_files)

# 6. load block sequences: matching target stimuli in each eeg file
# csv files from params path:


