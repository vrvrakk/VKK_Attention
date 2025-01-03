from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

subs = input('Sub name: ')
sub_names = [sub.strip() for sub in subs.split(',')]
conditions = input('Specify conditions: ')
condition_list = []
for condition in conditions.split(','):
    condition_list.append(condition.strip())


default_path = Path.cwd()

for sub in sub_names:
    metrics_dict = {'band_metrics': None,
                    'frequency_metrics': None,
                    'power_metrics': None}

    data_path = default_path / 'data' / 'emg' / sub / 'preprocessed' / 'results'

    for file in data_path.iterdir():
        if 'band' in file.name:
            filename = file
            band_metrics = pd.read_csv(filename)
            metrics_dict['band_metrics'] = band_metrics
        elif 'frequency' in file.name:
            frequency_metrics = pd.read_table(file, delimiter=',')
            metrics_dict['frequency_metrics'] = frequency_metrics
        elif 'power' in file.name:
            power_metrics = pd.read_table(file, delimiter=',')
            metrics_dict['power_metrics'] = power_metrics

# possible trends: for later use
trends = {1: 'Target > Distractor < Non-Target',
          2: 'Target > Non-Target > Distractor',
          3: 'Distractor > Target > Non-Target',
          4: 'Distractor > Non-Target > Target',
          5: 'Non-Target > Target > Distractor',
          6: 'Non-target > Distractor > Target'}


# now band metrics extraction:
band_chi_p_values = band_metrics['p-value']
band_conditions = band_metrics['Unnamed: 0']
# change Cramer's Value Column name:
band_metrics.rename(columns={f'{band_metrics.columns[4]}': 'Cramer Value'}, inplace=True)

band_effect_size = band_metrics['Cramer Value']
band_counts = band_metrics['Observed Counts']
expected_band_counts = band_metrics['Expected Frequencies']
# Convert each string in 'Observed Counts' to an actual dictionary
import ast
# Convert strings to dictionaries if necessary
band_counts = band_counts.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
expected_band_counts = expected_band_counts.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

bands_table = pd.DataFrame(data=None, columns=['Condition', 'Significance', 'Association', 
                                               'Target Low-Band Dominance', 'Target Mid-Band Dominance', 'Target High-Band Dominance',
                                               'Distractor Low-Band Dominance', 'Distractor Mid-Band Dominance', 'Distractor High-Band Dominance',
                                               'Non-Target Low-Band Dominance', 'Non-Target Mid-Band Dominance', 'Non-Target High-Band Dominance'])
bands_table['Condition'] = band_conditions


for condition, (index, row) in zip(condition_list, enumerate(bands_table.iterrows())):

    chi_p_value = band_chi_p_values.iloc[index]
    cramer_val = band_effect_size.iloc[index]

    # significance:
    if chi_p_value < 0.05:
       bands_table.loc[index, 'Significance'] = True
    else:
        bands_table.loc[index, 'Significance'] = False
    # effect of size:
    if cramer_val == 0:
        bands_table.loc[index, 'Association'] = 'No Association'
    elif 0 < cramer_val <= 0.1:
        bands_table.loc[index, 'Association'] = 'Weak Association'
    elif 0.1 < cramer_val <= 0.3:
        bands_table.loc[index, 'Association'] = 'Moderate Association'
    elif 0.3 < cramer_val:
        bands_table.loc[index, 'Association'] = 'Strong Association'
    # expected vs observed counts:
    target_low_band_count = band_counts.iloc[index]['Low Band']['Target']
    target_mid_band_count = band_counts.iloc[index]['Mid Band']['Target']
    target_high_band_count = band_counts.iloc[index]['High Band']['Target']

    distractor_low_band_count = band_counts.iloc[index]['Low Band']['Distractor']
    distractor_mid_band_count = band_counts.iloc[index]['Mid Band']['Distractor']
    distractor_high_band_count = band_counts.iloc[index]['High Band']['Distractor']

    non_target_low_band_count = band_counts.iloc[index]['Low Band']['Non-Target']
    non_target_mid_band_count = band_counts.iloc[index]['Mid Band']['Non-Target']
    non_target_high_band_count = band_counts.iloc[index]['High Band']['Non-Target']
    
    total_target_band_count = sum((target_low_band_count, target_mid_band_count, target_high_band_count))
    target_low_band_percentage = (target_low_band_count * 100) / total_target_band_count
    bands_table.loc[index, 'Target Low-Band Dominance'] = target_low_band_percentage
    target_mid_band_percentage = (target_mid_band_count * 100) / total_target_band_count
    bands_table.loc[index, 'Target Mid-Band Dominance'] = target_mid_band_percentage
    target_high_band_percentage = (target_high_band_count * 100) / total_target_band_count
    bands_table.loc[index, 'Target High-Band Dominance'] = target_high_band_percentage
    
    #  distractor percentages:
    total_distractor_band_count = sum((distractor_low_band_count, distractor_mid_band_count, distractor_high_band_count))
    distractor_low_band_percentage = (distractor_low_band_count * 100) / total_distractor_band_count
    bands_table.loc[index, 'Distractor Low-Band Dominance'] = distractor_low_band_percentage
    distractor_mid_band_percentage = (distractor_mid_band_count * 100) / total_distractor_band_count
    bands_table.loc[index, 'Distractor Mid-Band Dominance'] = distractor_mid_band_percentage
    distractor_high_band_percentage = (distractor_high_band_count * 100) / total_distractor_band_count
    bands_table.loc[index, 'Distractor High-Band Dominance'] = distractor_high_band_percentage
    # non-target percentages:
    total_non_target_band_count = sum(
        (non_target_low_band_count, non_target_mid_band_count, non_target_high_band_count))
    non_target_low_band_percentage = (non_target_low_band_count * 100) / total_non_target_band_count
    bands_table.loc[index, 'Non-Target Low-Band Dominance'] = non_target_low_band_percentage
    non_target_mid_band_percentage = (non_target_mid_band_count * 100) / total_non_target_band_count
    bands_table.loc[index, 'Non-Target Mid-Band Dominance'] = non_target_mid_band_percentage
    non_target_high_band_percentage = (non_target_high_band_count * 100) / total_non_target_band_count
    bands_table.loc[index, 'Non-Target High-Band Dominance'] = non_target_high_band_percentage
    


# now to the dominant frequencies distributions:
freqs_table = pd.DataFrame(data=None, columns=['Condition', 'Significance', 'Effect Size',
                                               'Target vs Distractor', 'Target vs Non-Target',
                                               'Distractor vs Non-Target'])

frequency_keys = list(frequency_metrics.keys())

# overall significance and eta-value:
freq_kruskal_p_key = frequency_keys[2]  # whether the group differences are statistically significant
freq_effect_size_key = frequency_keys[3]  # magnitude of group differences across all groups (global effect size)
freq_kruskal_p_vals = frequency_metrics[freq_kruskal_p_key]
freq_effect_sizes = frequency_metrics[freq_effect_size_key]
# pairwise comparison
target_vs_distractor_p_value_key = frequency_keys[4]  # pairwise comparison of Target vs. Distractor groups, indicating significance
target_vs_no_target_p_value_key = frequency_keys[5]
distractor_vs_non_target_p_value_key = frequency_keys[6]

target_vs_distractor_p_values = frequency_metrics[target_vs_distractor_p_value_key]
target_vs_no_target_p_values = frequency_metrics[target_vs_no_target_p_value_key]
distractor_vs_non_target_p_values = frequency_metrics[distractor_vs_non_target_p_value_key]


frequency_conditions = frequency_metrics['Unnamed: 0']
freqs_table['Condition'] = frequency_conditions



for condition, (index, row) in zip(condition_list, enumerate(freqs_table.iterrows())):
    # extract freqs stats for interpretation:
    freq_kruskal_p = freq_kruskal_p_vals.iloc[index]
    freq_effect_size = freq_effect_sizes.iloc[index]

    # dunn p-vals:
    target_vs_distractor_p_value = target_vs_distractor_p_values.iloc[index]
    target_vs_no_target_p_value = target_vs_no_target_p_values.iloc[index]
    distractor_vs_non_target_p_value = distractor_vs_non_target_p_values.iloc[index]

    # assign significance:
    freqs_table.loc[index, 'Significance'] = freq_kruskal_p < 0.05

    if freq_effect_size <= 0.01:
        freqs_table.loc[index, 'Effect Size'] = 'Negligible'
    elif 0.01 < freq_effect_size <= 0.06:
        freqs_table.loc[index, 'Effect Size'] = 'Small'
    elif 0.06 < freq_effect_size <= 0.14:
        freqs_table.loc[index, 'Effect Size'] = 'Medium'
    elif 0.14 < freq_effect_size:
        freqs_table.loc[index, 'Effect Size'] = 'Large'

    # determine significance of each pair:
    if target_vs_distractor_p_value <= 0.001:
        freqs_table.loc[index, 'Target vs Distractor'] = 'Highly Strong'
    elif target_vs_distractor_p_value <= 0.01:
        freqs_table.loc[index, 'Target vs Distractor'] = 'Strong'
    elif target_vs_distractor_p_value <= 0.05:
        freqs_table.loc[index, 'Target vs Distractor'] = 'Weak'
    else:
        freqs_table.loc[index, 'Target vs Distractor'] = 'No Significance'

    if target_vs_no_target_p_value <= 0.001:
        freqs_table.loc[index, 'Target vs Non-Target'] = 'Highly Strong'
    elif target_vs_no_target_p_value <= 0.01:
        freqs_table.loc[index, 'Target vs Non-Target'] = 'Strong'
    elif target_vs_no_target_p_value <= 0.05:
        freqs_table.loc[index, 'Target vs Non-Target'] = 'Weak'
    else:
        freqs_table.loc[index, 'Target vs Non-Target'] = 'No Significance'

    if distractor_vs_non_target_p_value <= 0.001:
        freqs_table.loc[index, 'Distractor vs Non-Target'] = 'Highly Strong'
    elif distractor_vs_non_target_p_value <= 0.01:
        freqs_table.loc[index, 'Distractor vs Non-Target'] = 'Strong'
    elif distractor_vs_non_target_p_value <= 0.05:
        freqs_table.loc[index, 'Distractor vs Non-Target'] = 'Weak'
    else:
        freqs_table.loc[index, 'Distractor vs Non-Target'] = 'No Significance'


# now power metrics:
power_table = pd.DataFrame(data=None, columns=['Condition', 'Effect Size', 'Significance', 'Target vs Distractor',
                                               'Target vs Non-Target', 'Distractor vs Non-Target', 'Trend'])

power_table['Condition'] = power_metrics['Unnamed: 0']
power_metrics.keys()
power_p_vals = power_metrics['Kruskal-Wallis p-value']
power_eta_squared = power_metrics['Effect Size (eta-squared)']
power_variance = power_metrics['Levene statistic (variance)']
power_dunn_p_target_vs_distractor = power_metrics['Dunn Posthoc Target vs. Distractor']
power_dunn_p_target_vs_non_target = power_metrics['Dunn Posthoc Target vs. Non-Target']
power_dunn_p_distractor_vs_non_target = power_metrics['Dunn Posthoc Distractor vs. Non-Target']
power_delta_target_vs_distractor = power_metrics['Cliff Delta Target vs Distractor']
power_delta_target_vs_non_target = power_metrics['Cliff Delta Target vs Non-Target']
power_delta_distractor_vs_non_target = power_metrics['Cliff Delta Distractor vs Non-Target']


trend_dict = pd.DataFrame({'a1': {'target': 0,
                     'distractor': 0,
                     'non_target': 0},
              'a2': {'target': 0,
                     'distractor': 0,
                     'non_target': 0},
              'e1': {'target': 0,
                     'distractor': 0,
                     'non_target': 0},
              'e2': {'target': 0,
                     'distractor': 0,
                     'non_target': 0}})

trend_dict = trend_dict.transpose()

for condition, (index, row) in zip(condition_list, enumerate(power_table.iterrows())):

    target = trend_dict.iloc[index]['target']
    distractor = trend_dict.iloc[index]['distractor']
    non_target = trend_dict.iloc[index]['non_target']

    # extract power stats for interpretation:
    power_p_val = power_p_vals.iloc[index]


    power_table.loc[index, 'Significance'] = power_p_val < 0.05
    power_eta_val = power_eta_squared.iloc[index]
    
    if power_eta_val <= 0.01:
        power_table.loc[index, 'Effect Size'] = 'Negligible'
    elif 0.01 < power_eta_val <= 0.06:
        power_table.loc[index, 'Effect Size'] = 'Small'
    elif 0.06 < power_eta_val <= 0.14:
        power_table.loc[index, 'Effect Size'] = 'Medium'
    elif 0.14 < power_eta_val:
        power_table.loc[index, 'Effect Size'] = 'Large'
    # pairwise significance:
    power_dunn_p_target_vs_distractor_val = power_dunn_p_target_vs_distractor.iloc[index]
    power_dunn_p_target_vs_non_target_val = power_dunn_p_target_vs_non_target.iloc[index]
    power_dunn_p_distractor_vs_non_target_val = power_dunn_p_distractor_vs_non_target.iloc[index]

    if power_dunn_p_target_vs_distractor_val <= 0.001:
        power_table.loc[index, 'Target vs Distractor'] = 'Highly Strong'
    elif power_dunn_p_target_vs_distractor_val <= 0.01:
        power_table.loc[index, 'Target vs Distractor'] = 'Strong'
    elif power_dunn_p_target_vs_distractor_val <= 0.05:
        power_table.loc[index, 'Target vs Distractor'] = 'Weak'
    else:
        power_table.loc[index, 'Target vs Distractor'] = 'No Significance'

    if power_dunn_p_target_vs_non_target_val <= 0.001:
        power_table.loc[index, 'Target vs Non-Target'] = 'Highly Strong'
    elif power_dunn_p_target_vs_non_target_val <= 0.01:
        power_table.loc[index, 'Target vs Non-Target'] = 'Strong'
    elif power_dunn_p_target_vs_non_target_val <= 0.05:
        power_table.loc[index, 'Target vs Non-Target'] = 'Weak'
    else:
        power_table.loc[index, 'Target vs Non-Target'] = 'No Significance'

    if power_dunn_p_distractor_vs_non_target_val <= 0.001:
        power_table.loc[index, 'Distractor vs Non-Target'] = 'Highly Strong'
    elif power_dunn_p_distractor_vs_non_target_val <= 0.01:
        power_table.loc[index, 'Distractor vs Non-Target'] = 'Strong'
    elif power_dunn_p_distractor_vs_non_target_val <= 0.05:
        power_table.loc[index, 'Distractor vs Non-Target'] = 'Weak'
    else:
        power_table.loc[index, 'Distractor vs Non-Target'] = 'No Significance'
    # so far so good.
    # cliff delta comparison:
    power_delta_target_vs_distractor_val = power_delta_target_vs_distractor.iloc[index]
    power_delta_target_vs_non_target_val = power_delta_target_vs_non_target.iloc[index]
    power_delta_distractor_vs_non_target_val = power_delta_distractor_vs_non_target.iloc[index]

    if 0 < power_delta_target_vs_distractor_val <= 0.5:
        target =+ 1
    elif 0.5 < power_delta_target_vs_distractor_val <= 1:
        target =+ 2
    elif -0.5 <= power_delta_target_vs_distractor_val <= 0:
        distractor =+ 1
    elif -0.9 <= power_delta_target_vs_distractor_val < -0.5:
        distractor =+ 2

    if 0 <= power_delta_target_vs_non_target_val <= 0.5:
        target = + 1
    elif 0.5 < power_delta_target_vs_non_target_val <= 1:
        target = + 2
    elif -0.5 <= power_delta_target_vs_non_target_val <= 0:
        non_target =+ 1
    elif -0.9 <= power_delta_target_vs_non_target_val < -0.5:
        non_target =+ 1

    if 0 <= power_delta_distractor_vs_non_target_val <= 0.5:
        distractor = + 1
    elif 0.5 < power_delta_distractor_vs_non_target_val <= 1:
        distractor = + 2
    elif -0.5 <= power_delta_distractor_vs_non_target_val <= 0:
        non_target = + 1
    elif -0.9 <= power_delta_distractor_vs_non_target_val < -0.5:
        non_target =+ 2
    # determine trend of power across epochs in each condition:
    if target > distractor and target > non_target:
        print(f'for {condition} target power is strongest.')
        if distractor > non_target:
            print(f'for {condition} distractor power is second.')
            print(f'for {condition} non_target power is third.')
        elif non_target > distractor:
            print(f'for {condition} non_target power is second.')
            print(f'for {condition} distractor power is third.')
        elif non_target == distractor:
            print(f'for {condition} distractor and non-target are equal.')
    elif distractor > target and distractor > non_target:
         print(f'for {condition} distractor power is strongest.')
         if target > non_target:
             print(f'for {condition} target power is second.')
             print(f'for {condition} non_target power is third.')
         elif non_target > target:
             print(f'for {condition} non_target power is second.')
             print(f'for {condition} target power is third.')
         elif non_target == target:
              print(f'for {condition} target and non-target are equal.')
    elif non_target > target and non_target > distractor:
        print(f'for {condition} non_target power is strongest.')
        if target > distractor:
            print(f'for {condition} target power is second.')
            print(f'for {condition} distractor power is third.')
        elif distractor > target:
            print(f'for {condition} distractor power is second.')
            print(f'for {condition} target power is third.')
        elif target == distractor:
            print(f'for {condition} target and distractor are equal.')
    elif target == distractor and target == non_target:
        print(f'for {condition} all epoch types have equal power. No differences observed.')
    elif target == distractor and target > non_target:
        print(f'for {condition} target and distractor have equal power. Non-target is weaker.')
    elif target == non_target and target < distractor:
        print(f'for {condition} target and non_target have equal power. Distractor is weaker.')
    elif non_target == distractor and non_target < target:
        print(f'for {condition} distractor and non_target have equal power. Target is weaker.')
    trend_dict.iloc[index]['target'] = target
    trend_dict.iloc[index]['distractor'] = distractor
    trend_dict.iloc[index]['non_target'] = non_target





