import pickle as pkl
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plane = 'elevation'
stim_type = 'all'
base_dir = Path.cwd()
data_dir = base_dir / 'data' / 'eeg'
concat_dir = data_dir / 'journal' / 'TRF' / 'results' / plane

with open(concat_dir/f'{plane}_results_average.pkl', 'rb') as res:
    avg_results_dict = pkl.load(res)


weights = avg_results_dict['predictions_dict']['weights']

col_names = ['onsets_target', 'envelopes_target', 'phonemes_target', 'responses_target',
             'onsets_distractor', 'envelopes_distractor', 'phonemes_distractor', 'alpha']

all_ch = ['Fp1', 'Fp2', 'F7','F3','Fz','F4','F8','FC5','FC1','FC2','FC6','T7','C3','Cz','C4','T8',
          'TP9','CP5','CP1','CP2','CP6','TP10','P7','P3','Pz','P4','P8','PO9','O1','Oz','O2','PO10',
          'AF7','AF3','AF4','AF8','F5','F1','F2','F6','FT9','FT7','FC3','FC4','FT8','FT10','C5','C1',
          'C2','C6','TP7','CP3','CPz','CP4','TP8','P5','P1','P2','P6','PO7','PO3','POz','PO4','PO8','FCz']

roi = np.array(['Fp1', 'F3', 'Fz', 'F4', 'FC1', 'C3', 'Cz', 'C4', 'C1', 'C2', 'CP5',
                   'CP1', 'CP3', 'P7', 'P3', 'Pz', 'P4', 'P5', 'P1', 'F1', 'F2', 'AF3', 'FCz'])


ch_mask = np.isin(all_ch, roi)

weights_filt = weights[:, :, :, ch_mask]

weights_avg = np.average(weights_filt, axis=-1)  # avg across those channels
time = np.array([-0.104, -0.096, -0.088, -0.08 , -0.072, -0.064, -0.056, -0.048,
       -0.04 , -0.032, -0.024, -0.016, -0.008,  0.   ,  0.008,  0.016,
        0.024,  0.032,  0.04 ,  0.048,  0.056,  0.064,  0.072,  0.08 ,
        0.088,  0.096,  0.104,  0.112,  0.12 ,  0.128,  0.136,  0.144,
        0.152,  0.16 ,  0.168,  0.176,  0.184,  0.192,  0.2  ,  0.208,
        0.216,  0.224,  0.232,  0.24 ,  0.248,  0.256,  0.264,  0.272,
        0.28 ,  0.288,  0.296,  0.304,  0.312,  0.32 ,  0.328,  0.336,
        0.344,  0.352,  0.36 ,  0.368,  0.376,  0.384,  0.392,  0.4  ,
        0.408,  0.416,  0.424,  0.432,  0.44 ,  0.448,  0.456,  0.464,
        0.472,  0.48 ,  0.488,  0.496,  0.504,  0.512,  0.52 ,  0.528,
        0.536,  0.544,  0.552,  0.56 ,  0.568,  0.576,  0.584,  0.592,
        0.6  ,  0.608,  0.616,  0.624,  0.632,  0.64 ,  0.648,  0.656,
        0.664,  0.672,  0.68 ,  0.688,  0.696,  0.704,  0.712,  0.72 ,
        0.728,  0.736,  0.744,  0.752,  0.76 ,  0.768,  0.776,  0.784,
        0.792,  0.8  ,  0.808,  0.816,  0.824,  0.832,  0.84 ,  0.848,
        0.856,  0.864,  0.872,  0.88 ,  0.888,  0.896,  0.904,  0.912,
        0.92 ,  0.928,  0.936,  0.944,  0.952,  0.96 ,  0.968,  0.976,
        0.984,  0.992,  1.])


weights_dict = {}
for index, predictor in enumerate(col_names):
    weights_dict[predictor] = weights_avg[:, index, :]

# predictors of interest
predictors = ["onsets", "envelopes", "phonemes"]

paired_weights = {}

for pred in predictors:
    target_key = f"{pred}_target"
    distractor_key = f"{pred}_distractor"

    if target_key in weights_dict and distractor_key in weights_dict:
        paired_weights[pred] = (weights_dict[target_key],
                                weights_dict[distractor_key])


from mne.stats import permutation_cluster_test


def cluster_perm(target_data, distractor_data, predictor, time, condition, stim_type, data_dir, n=18):
    """
    Run cluster-based permutation test on paired target vs distractor TRFs.

    Parameters
    ----------
    target_data : array, shape (n_subjects, n_times)
    distractor_data : array, shape (n_subjects, n_times)
    predictor : str
        Predictor name (e.g. 'phonemes')
    time : array, shape (n_times,)
        Time vector
    condition, stim_type : str
        Labels for plotting/saving
    data_dir : Path
        Root directory for saving figures
    sub_list : list
        List of subjects (used for SEM)
    """

    # Compute summary stats
    target_std = np.std(target_data, axis=0)
    target_mean = np.mean(target_data, axis=0)
    distractor_std = np.std(distractor_data, axis=0)
    distractor_mean = np.mean(distractor_data, axis=0)

    target_sem = target_std / np.sqrt(n)
    distractor_sem = distractor_std / np.sqrt(n)

    # Run cluster permutation test
    X = [target_data, distractor_data]  # list of arrays (subjects × time)
    T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
        X, n_permutations=10000, tail=1, n_jobs=1
    )

    # Plot
    plt.figure(figsize=(8, 4))
    plt.plot(time, target_mean, 'b-', linewidth=2, label='Target')
    plt.fill_between(time,
                     target_mean - target_sem,
                     target_mean + target_sem,
                     color='b', alpha=0.3)
    plt.plot(time, distractor_mean, 'r-', linewidth=2, label='Distractor')
    plt.fill_between(time,
                     distractor_mean - distractor_sem,
                     distractor_mean + distractor_sem,
                     color='r', alpha=0.3)

    # Highlight significant clusters
    for cl, pval in zip(clusters, cluster_p_values):
        if pval < 0.05:
            time_inds = cl[0]
            plt.axvspan(time[time_inds[0]], time[time_inds[-1]],
                        color='gray', alpha=0.3)

    plt.title(f'TRF Comparison - {condition} - {predictor}')
    plt.legend()

    fig_path = data_dir / 'journal' / 'figures' / 'TRF' / condition / stim_type
    fig_path.mkdir(parents=True, exist_ok=True)
    filename = f'{predictor}_{stim_type}_{condition}.png'
    # plt.savefig(fig_path / filename, dpi=300)
    plt.show()


for predictor, (target_arr, distractor_arr) in paired_weights.items():
    cluster_perm(target_arr, distractor_arr, predictor,
                 time=time, condition=plane,
                 stim_type=stim_type,
                 data_dir=data_dir, n=18)




