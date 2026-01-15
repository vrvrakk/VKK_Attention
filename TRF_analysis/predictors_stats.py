import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import os
from scipy.stats import ttest_rel
import pickle as pkl
import pandas as pd
from scipy.stats import ttest_rel, pearsonr

'''
a lil script to create stats about the across subject SD of phoneme counts, and stim onsets,
as well as the overal distribution of the predictors across streams.
Justin Case
'''

# Define directories
default_dir = Path.cwd()
data_dir = default_dir / 'data' / 'eeg'
counts_dir = data_dir / 'predictors' / 'sanity_check'
stats_dir = counts_dir / 'stats'
fig_dir = counts_dir / 'figures'
# make sure paths exist or are created
stats_dir.mkdir(parents=True, exist_ok=True)
fig_dir.mkdir(parents=True, exist_ok=True)


# load zip files:
def load_zip(stim_type=''):
    if plane == 'elevation':
        conditions = ['e1', 'e2']
    elif plane == 'azimuth':
        conditions = ['a1', 'a2']
    else:
        raise ValueError('Please give a valid plane as input.')
    predictor_filename1 = f'{predictor}_counts_{conditions[0]}.npz'
    predictor_filename2 = f'{predictor}_counts_{conditions[1]}.npz'
    pred1 = np.load(counts_dir/stim_type/predictor_filename1, allow_pickle=True)
    pred2 = np.load(counts_dir/stim_type/predictor_filename2, allow_pickle=True)
    pred1_dict = pred1[predictor].item()
    pred2_dict = pred2[predictor].item()
    all_targets = {}
    all_distractors = {}
    for sub in pred1_dict.keys():
        targets1 = pred1_dict[sub]['target']
        targets2 = pred2_dict[sub]['target']
        targets_mean = int(np.mean([targets1, targets2]))
        all_targets[sub] = targets_mean
        distr1 = pred1_dict[sub]['distractor']
        distr2 = pred2_dict[sub]['distractor']
        distr_mean = int(np.mean([distr1, distr2]))
        all_distractors[sub] = distr_mean
    return all_targets, all_distractors


def get_stats(all_targets, all_distractors, stim_type=''):
    target_vals = list(all_targets.values())
    distractor_vals = list(all_distractors.values())
    target_mean_subs = int(np.mean(target_vals))
    target_sd_subs = int(np.std(target_vals))
    target_sd_percentage = (target_sd_subs / target_mean_subs) * 100
    distractor_mean_subs = int(np.mean(distractor_vals))
    distractor_sd_subs = (np.std(distractor_vals))
    distractor_sd_percentage = (distractor_sd_subs / distractor_mean_subs) * 100
    dataframe = pd.DataFrame(columns=['Stream', 'Stimulus Type', f'{predictor.capitalize()}',  'Mean Count', 'SD', 'SD in %'])
    dataframe['Stream'] = ['Target', 'Distractor']  # stream order
    if stim_type == 'target_nums':
        stim_type = 'target_numbers'
    else:
        stim_type = stim_type
    dataframe['Stimulus Type'] = stim_type.capitalize().replace('_', ' ')  # stim type same in all rows per run
    dataframe[f'{predictor.capitalize()}'] = predictor
    dataframe['Mean Count'] = [target_mean_subs, distractor_mean_subs]
    dataframe['SD'] = np.round([target_sd_subs, distractor_sd_subs], decimals=1)
    dataframe['SD in %'] = np.round([target_sd_percentage, distractor_sd_percentage], decimals=1)
    return dataframe


def plot_hist_counts(all_targets, all_distractors, stim_type='', save_dir=fig_dir):
    """
    Plot histograms of target and distractor counts across subjects
    for a given plane, predictor, and stim_type.
    """
    target_vals = np.array(list(all_targets.values()))
    distractor_vals = np.array(list(all_distractors.values()))

    plt.figure()
    bins = np.linspace(
        min(target_vals.min(), distractor_vals.min()),
        max(target_vals.max(), distractor_vals.max()),
        10)

    plt.hist(target_vals, bins=bins, alpha=0.6, label='Target')
    plt.hist(distractor_vals, bins=bins, alpha=0.6, label='Distractor')
    plt.xlabel(f'{predictor.capitalize()} count')
    plt.ylabel('Number of subjects')

    if stim_type == 'target_nums':
        stim_label = 'Target numbers'
    else:
        stim_label = stim_type.replace('_', ' ').capitalize()

    plt.title(f'{plane.capitalize()} – {predictor.capitalize()} – {stim_label}')
    plt.legend()
    fname = save_dir / f'{plane}_{predictor}_{stim_type}_hist.png'
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()


if __name__ == '__main__':
    predictor = 'phonemes'  # 'phonemes' or 'onsets'
    planes = ['azimuth', 'elevation']
    plane = planes[0]

    # all stimuli
    target_all, distractor_all = load_zip(stim_type='all')
    df_all = get_stats(target_all, distractor_all, stim_type='all')
    plot_hist_counts(target_all, distractor_all, stim_type='all')

    # non-targets
    nt_target, nt_distractor = load_zip(stim_type='non_targets')
    df_nt = get_stats(nt_target, nt_distractor, stim_type='non_targets')
    plot_hist_counts(nt_target, nt_distractor, stim_type='non_targets')

    # target-numbers only
    target_nums, distractor_nums = load_zip(stim_type='target_nums')
    df_target_nums = get_stats(target_nums, distractor_nums, stim_type='target_nums')
    plot_hist_counts(target_nums, distractor_nums, stim_type='target_nums')

    # combine stats for this plane and save
    df_ultimate = pd.concat([df_all, df_nt, df_target_nums])
    filename = stats_dir / f'{plane}_{predictor}_stats_ultimate.csv'
    df_ultimate.to_csv(filename, index=False)

    # phoneme count and delta r corr across subs:
    nsi_dir = data_dir / 'journal' / 'TRF' / 'results' / 'r' / 'NSI'

    with open(nsi_dir / f'{predictor}_{plane}_r_diffs.pkl', 'rb') as az:
        r_ztranformed = pkl.load(az)
        # merge azimuth and elevation dicts
    r_zscored_all = {}
    r_zscored_all.update(r_ztranformed)  # contains keys 'a1', 'a2', or 'e1', 'e2'

    # build r_diff_z_arrays from merged dict
    r_diff_z_arrays = {}
    for cond, sub in r_zscored_all.items():
        r_diff_z_array = np.array([d['r_diff'] for d in sub.values()])
        r_diff_z_arrays[cond] = r_diff_z_array
    # get delta r mean:
    sub_list = ['sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17', 'sub18', 'sub19', 'sub20',
                'sub21', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29']
    for cond in r_diff_z_arrays:
        old_list = r_diff_z_arrays[cond]  # list of 18 arrays
        new_dict = {sub: np.mean(arr) for sub, arr in zip(sub_list, old_list)}  # get mean of each sub in all 4 conds
        r_diff_z_arrays[cond] = new_dict

    if predictor == 'phonemes':
        if plane == 'azimuth':
            cond1 = 'a1'
            cond2 = 'a2'
        else:
            cond1 = 'e1'
            cond2 = 'e2'

        plane_avg = {}
        for sub in sub_list:
            val1 = r_diff_z_arrays[cond1][sub]
            val2 = r_diff_z_arrays[cond2][sub]
            plane_avg[sub] = (val1 + val2) / 2
        # run peason corr
        r_vals = list(plane_avg.values())
        # is there a correlation between delta r and phoneme counts across subs?
        pearsonr(r_vals, list(target_all.values()))
        pearsonr(r_vals, list(distractor_all.values()))

    ''' Above code already ran and dfs are saved. 
    I am lazy rn so below I will re-load them and combine the DFs across planes'''

    df_azimuth = pd.read_csv(stats_dir / f'azimuth_{predictor}_stats_ultimate.csv')
    df_azimuth.insert(0, 'Plane', 'Azimuth')

    df_elevation = pd.read_csv(stats_dir / f'elevation_{predictor}_stats_ultimate.csv')
    df_elevation.insert(0, 'Plane', 'Elevation')

    df_across_planes = pd.concat([df_azimuth, df_elevation])
    df_filename = f'both_planes_{predictor}_counts.csv'
    df_across_planes.to_csv(stats_dir / df_filename, index=False)








