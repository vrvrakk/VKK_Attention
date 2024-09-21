import matplotlib

matplotlib.use('Agg')
from pathlib import Path
import os
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

sub = input('Give subject number:')
condition = input('Select a condition (a1, a2, e1, or e2):')
default_dir = Path.cwd()
sub_dir = default_dir / 'data' / 'emg' / f'{sub}' / 'preprocessed' / 'results' / 'z_score_data'

csv_files = []
for files in sub_dir.iterdir():
    if files.is_file:
        if condition in files.name:
            csv_files.append(files)

csv_dfs = {}
for csv_file in csv_files:
    csv = pd.read_csv(csv_file, header=None)
    csv_dfs[csv_file.name] = csv



csv_concatenated = pd.concat(csv_dfs.values())




