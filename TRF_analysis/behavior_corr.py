import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
from pathlib import Path
import pickle as pkl
from scipy.stats import ttest_rel
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats import norm

base_dir = Path.cwd()
data_dir = base_dir / 'eeg' / 'data'

planes = {'azimuth': ['a1', 'a2'],
          'elevation':['e1', 'e2']}


plane = planes['azimuth']

alphas = {}
for condition in plane:
    alpha_dir = data_dir / 'journal' / 'alpha' / condition
    for folders in alpha_dir.iterdir():
        with open(folders, 'rb') as a:
            alpha = pkl.load(a)
            alphas[condition] = alpha

# keep alpha ratio as a metric:
from EEG.preprocessing_eeg import sub_list

sub_list = sub_list[6:]

alpha_ratios = {cond : {} for cond in plane}
for cond in alphas.keys():
    alpha_cond = alphas[cond]
    for sub in alpha_cond.keys():
        if sub in sub_list:
            alpha_ratio = alpha_cond[sub]['alpha_ratio']
            alpha_ratios[cond][sub] = np.mean(alpha_ratio)


def collapse_df(data_dict):
    subjects = list(next(iter(data_dict.values())).keys())  # all subjects
    collapsed_df = {}
    for subj in subjects:
        vals = []
        for cond in data_dict.values():
            vals.append(cond[subj])
            mean_vals = np.mean(vals)
        collapsed_df[subj] = mean_vals
    return collapsed_df


alpha_collapsed_df = collapse_df(alpha_ratios)

# reaction times:
predictor_dir = data_dir / 'predictors'
rt_dict = {cond: {} for cond in plane}
for condition in plane:
    rt_dir = predictor_dir / 'responses' / condition / 'all'
    for folders in rt_dir.iterdir():
        rt = np.load(folders, allow_pickle=True)
        rt_arr = rt['responses']

