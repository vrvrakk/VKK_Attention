from pathlib import Path
import os
import pickle as pkl

import numpy as np

plane = 'azimuth'
stim_type = 'all'

# directories:
base_dir = Path.cwd()
data_dir = base_dir / 'data' / 'eeg'
predictor_dir = data_dir / 'predictors'
bad_segments_dir = predictor_dir / 'bad_segments'
eeg_dir = Path('D:/VKK_Attention/data/eeg/preprocessed/results')


def cond_selection(plane=''):
    if plane == 'azimuth':
        cond1 = 'a1'
        cond2 = 'a2'
    elif plane == 'elevation':
        cond1 = 'e1'
        cond2 = 'e2'
    return cond1, cond2


def load_cond(cond, stim_type="all"):
    save_dir = data_dir / 'journal' / 'TRF' / 'results' / cond / stim_type
    predictions_dir = save_dir / 'predictions'
    weights_dir = save_dir / 'weights'

    result = {}
    # load TRFs
    with open(weights_dir/'target_onset_trfs.pkl', 'rb') as f:
        result["target_onset_trfs"] = pkl.load(f)
    with open(weights_dir/'distractor_onset_trfs.pkl', 'rb') as f:
        result["distractor_onset_trfs"] = pkl.load(f)

    with open(weights_dir/'target_phoneme_trfs.pkl', 'rb') as f:
        result["target_phoneme_trfs"] = pkl.load(f)
    with open(weights_dir/'distractor_phoneme_trfs.pkl', 'rb') as f:
        result["distractor_phoneme_trfs"] = pkl.load(f)

    with open(weights_dir/'target_env_trfs.pkl', 'rb') as f:
        result["target_env_trfs"] = pkl.load(f)
    with open(weights_dir/'distractor_env_trfs.pkl', 'rb') as f:
        result["distractor_env_trfs"] = pkl.load(f)

    # predictions (composite model)
    with open(predictions_dir/'predictions.pkl', 'rb') as f:
        result["predictions_dict"] = pkl.load(f)

    return result


cond1, cond2 = cond_selection(plane=plane)

cond1_results = load_cond(cond=cond1, stim_type=stim_type)
cond2_results = load_cond(cond=cond2, stim_type=stim_type)

dictionaries = list(cond1_results.keys())


cond1_name = "cond1"
cond2_name = "cond2"

combined = {}
for key in cond1_results.keys():
    combined[key] = {
        cond1_name: cond1_results[key],
        cond2_name: cond2_results[key]
    }

results_to_compare = {}
for key in dictionaries:
    if 'trfs' in key:
        data_cond1 = np.array(list(combined[key][cond1_name].values()))
        data_cond2 = np.array(list(combined[key][cond2_name].values()))
        results_to_compare[key] = (data_cond1, data_cond2)
    elif 'predictions' in key:
        subdict1 = combined[key][cond1_name]
        subdict2 = combined[key][cond2_name]

        # Collect r-values across subs
        r_vals1, r_vals2 = [], []
        weights1, weights2 = [], []
        for sub_id in subdict1.keys():
            r_vals1.append(subdict1[sub_id]['r'])
            r_vals2.append(subdict2[sub_id]['r'])
            weights1.append(subdict1[sub_id]['weights'])
            weights2.append(subdict2[sub_id]['weights'])

        results_to_compare[key] = {
            cond1_name: {"r": np.array(r_vals1), "weights": np.array(weights1)},
            cond2_name: {"r": np.array(r_vals2), "weights": np.array(weights2)}
        }


from scipy.stats import ttest_rel, wilcoxon, shapiro

p_vals_dict = {}
for key in dictionaries:
    if 'predictions' in key:
        group1 = np.mean(results_to_compare[key][cond1_name]['r'], axis=-1) # avg along ch
        group2 = np.mean(results_to_compare[key][cond2_name]['r'], axis=-1)
    else:
        group1 = np.mean(results_to_compare[key][0], axis=-1) # avg along axis
        group2 = np.mean(results_to_compare[key][1], axis=-1)

    _, p1 = shapiro(group1)
    _, p2 = shapiro(group2)
    significance = False
    if p1 and p2 > 0.05:
        print('Distribution normal. T-test applied')
        # distribution normal, proceed with ttest:
        stat, p = ttest_rel(group1, group2)
        if p < 0.05:
            significance = True
            print(f'significant differences between {cond1} and {cond2} detected! '
                  f'Concatenation cannot be considered.')
    else:
        print('Distribution deviates from normal. Wilcoxon test applied.')
        stat, p = wilcoxon(group1, group2)
        if p < 0.05:
            significance = True
            print(f'significant differences between {cond1} and {cond2} detected! '
                  f'Concatenation cannot be considered.')
    p_vals_dict[key] = {'stat': stat, 'p': p, 'shapiro': (p1, p2), 'significance': significance}

# save dicts:
save_dir = data_dir / 'journal' / 'TRF' / 'results' / 'sub_conditions_comparisons' / plane
save_dir.mkdir(parents=True, exist_ok=True)
filename = f'{plane}_p_vals_dict.pkl'
with open(save_dir/filename, 'wb') as file:
    pkl.dump(p_vals_dict, file)

concat_dir = data_dir / 'journal' / 'TRF' / 'results' / plane
concat_dir.mkdir(parents=True, exist_ok=True)

avg_results_dict = {}
for keys in dictionaries:
    if 'predictions' in keys:
        subdict1 = results_to_compare[keys][cond1_name]
        subdict2 = results_to_compare[keys][cond2_name]
        # make sure nested dict exists
        avg_results_dict[keys] = {}
        for sub_keys in subdict1.keys(): # keys: r and weights
            group1 = subdict1[sub_keys]
            group2 = subdict2[sub_keys]
            group_avg = np.average((group1, group2), axis=0)
            avg_results_dict[keys][sub_keys] = group_avg
    else:
        group1 = results_to_compare[keys][0]
        group2 = results_to_compare[keys][1]
        group_avg = np.average((group1, group2), axis=0)
        avg_results_dict[keys] = group_avg


with open(concat_dir/f'{plane}_results_average.pkl', 'wb') as res:
    pkl.dump(avg_results_dict, res)