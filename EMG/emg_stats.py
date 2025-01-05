from pathlib import Path
import os
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
subs_band_metrics = {}
subs_frequency_metrics = {}
subs_power_metrics = {}

for sub in sub_names:
    metrics_dict = {'band_metrics': None,
                    'frequency_metrics': None,
                    'power_metrics': None}

    data_path = default_path / 'data' / 'emg' / sub / 'preprocessed' / 'results'

    for file in data_path.iterdir():
        if 'band' in file.name:
            filename = file
            band_metrics = pd.read_csv(filename)
        elif 'frequency' in file.name:
            frequency_metrics = pd.read_table(file, delimiter=',')
        elif 'power' in file.name:
            power_metrics = pd.read_table(file, delimiter=',')
    subs_band_metrics[sub] = band_metrics
    subs_frequency_metrics[sub] = frequency_metrics
    subs_power_metrics[sub] = power_metrics

# now band metrics extraction:
bands_tables = {}
for sub, metrics in subs_band_metrics.items():
    band_chi_p_values = metrics['p-value']
    band_conditions = metrics['Unnamed: 0']
    # change Cramer's Value Column name:
    metrics.rename(columns={f'{metrics.columns[4]}': 'Cramer Value'}, inplace=True)

    band_effect_size = metrics['Cramer Value']
    band_counts = metrics['Observed Counts']
    expected_band_counts = metrics['Expected Frequencies']
    # Convert each string in 'Observed Counts' to an actual dictionary
    import ast

    # Convert strings to dictionaries if necessary
    band_counts = band_counts.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    expected_band_counts = expected_band_counts.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    bands_tables[sub] = pd.DataFrame(data=None, columns=['Condition', 'Significance', 'Association',
                                                         'Target Low-Band Dominance', 'Target Mid-Band Dominance',
                                                         'Target High-Band Dominance',
                                                         'Distractor Low-Band Dominance',
                                                         'Distractor Mid-Band Dominance',
                                                         'Distractor High-Band Dominance',
                                                         'Non-Target Low-Band Dominance',
                                                         'Non-Target Mid-Band Dominance',
                                                         'Non-Target High-Band Dominance'])
    bands_tables[sub]['Condition'] = band_conditions

for sub, bands_table in bands_tables.items():
    metrics = subs_band_metrics[sub]  # Retrieve metrics for the current subject
    band_chi_p_values = metrics['p-value']
    band_effect_size = metrics['Cramer Value']
    band_counts = metrics['Observed Counts']
    band_counts = band_counts.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    expected_band_counts = expected_band_counts.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    for condition, (index, row) in zip(condition_list, enumerate(bands_table.iterrows())):

        chi_p_value = band_chi_p_values.iloc[index]
        cramer_val = band_effect_size.iloc[index]

        # significance:
        if chi_p_value < 0.05:
            bands_tables[sub].loc[index, 'Significance'] = True
        else:
            bands_tables[sub].loc[index, 'Significance'] = False
        # effect of size:
        if cramer_val == 0:
            bands_tables[sub].loc[index, 'Association'] = 'No Association'
        elif 0 < cramer_val <= 0.1:
            bands_tables[sub].loc[index, 'Association'] = 'Weak Association'
        elif 0.1 < cramer_val <= 0.3:
            bands_tables[sub].loc[index, 'Association'] = 'Moderate Association'
        elif 0.3 < cramer_val:
            bands_tables[sub].loc[index, 'Association'] = 'Strong Association'
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
        bands_tables[sub].loc[index, 'Target Low-Band Dominance'] = target_low_band_percentage
        target_mid_band_percentage = (target_mid_band_count * 100) / total_target_band_count
        bands_tables[sub].loc[index, 'Target Mid-Band Dominance'] = target_mid_band_percentage
        target_high_band_percentage = (target_high_band_count * 100) / total_target_band_count
        bands_tables[sub].loc[index, 'Target High-Band Dominance'] = target_high_band_percentage

        #  distractor percentages:
        total_distractor_band_count = sum(
            (distractor_low_band_count, distractor_mid_band_count, distractor_high_band_count))
        distractor_low_band_percentage = (distractor_low_band_count * 100) / total_distractor_band_count
        bands_tables[sub].loc[index, 'Distractor Low-Band Dominance'] = distractor_low_band_percentage
        distractor_mid_band_percentage = (distractor_mid_band_count * 100) / total_distractor_band_count
        bands_tables[sub].loc[index, 'Distractor Mid-Band Dominance'] = distractor_mid_band_percentage
        distractor_high_band_percentage = (distractor_high_band_count * 100) / total_distractor_band_count
        bands_tables[sub].loc[index, 'Distractor High-Band Dominance'] = distractor_high_band_percentage
        # non-target percentages:
        total_non_target_band_count = sum(
            (non_target_low_band_count, non_target_mid_band_count, non_target_high_band_count))
        non_target_low_band_percentage = (non_target_low_band_count * 100) / total_non_target_band_count
        bands_tables[sub].loc[index, 'Non-Target Low-Band Dominance'] = non_target_low_band_percentage
        non_target_mid_band_percentage = (non_target_mid_band_count * 100) / total_non_target_band_count
        bands_tables[sub].loc[index, 'Non-Target Mid-Band Dominance'] = non_target_mid_band_percentage
        non_target_high_band_percentage = (non_target_high_band_count * 100) / total_non_target_band_count
        bands_tables[sub].loc[index, 'Non-Target High-Band Dominance'] = non_target_high_band_percentage

# now to the dominant frequencies distributions:
freqs_tables = {}
for sub, frequency_metrics in subs_frequency_metrics.items():
    freqs_tables[sub] = pd.DataFrame(data=None, columns=['Condition', 'Significance', 'Effect Size',
                                                         'Target vs Distractor', 'Target vs Non-Target',
                                                         'Distractor vs Non-Target'])

    frequency_keys = list(frequency_metrics.keys())

    # overall significance and eta-value:
    freq_kruskal_p_key = frequency_keys[2]  # whether the group differences are statistically significant
    freq_effect_size_key = frequency_keys[3]  # magnitude of group differences across all groups (global effect size)
    freq_kruskal_p_vals = frequency_metrics[freq_kruskal_p_key]
    freq_effect_sizes = frequency_metrics[freq_effect_size_key]
    # pairwise comparison
    target_vs_distractor_p_value_key = frequency_keys[
        4]  # pairwise comparison of Target vs. Distractor groups, indicating significance
    target_vs_no_target_p_value_key = frequency_keys[5]
    distractor_vs_non_target_p_value_key = frequency_keys[6]

    target_vs_distractor_p_values = frequency_metrics[target_vs_distractor_p_value_key]
    target_vs_no_target_p_values = frequency_metrics[target_vs_no_target_p_value_key]
    distractor_vs_non_target_p_values = frequency_metrics[distractor_vs_non_target_p_value_key]

    frequency_conditions = frequency_metrics['Unnamed: 0']
    freqs_tables[sub]['Condition'] = frequency_conditions

for sub, freqs_table in freqs_tables.items():
    frequency_metrics = subs_frequency_metrics[sub]  # Retrieve metrics for the current subject
    freq_kruskal_p_vals = frequency_metrics[frequency_keys[2]]
    freq_effect_sizes = frequency_metrics[frequency_keys[3]]
    target_vs_distractor_p_values = frequency_metrics[frequency_keys[4]]
    target_vs_no_target_p_values = frequency_metrics[frequency_keys[5]]
    distractor_vs_non_target_p_values = frequency_metrics[frequency_keys[6]]
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
power_tables = {}
trends_dict = {}
for sub, power_metrics in subs_power_metrics.items():
    power_tables[sub] = pd.DataFrame(data=None,
                                     columns=['Condition', 'Effect Size', 'Significance', 'Target vs Distractor',
                                              'Target vs Non-Target', 'Distractor vs Non-Target', 'Trend'])

    power_tables[sub]['Condition'] = power_metrics['Unnamed: 0']

    trends_dict[sub] = pd.DataFrame({'a1': {'target': 0,
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
                                            'non_target': 0}}).transpose()

for sub, power_table in power_tables.items():
    trend_dict = trends_dict[sub]
    power_metrics = subs_power_metrics[sub]
    power_p_vals = power_metrics['Kruskal-Wallis p-value']
    power_eta_squared = power_metrics['Effect Size (eta-squared)']
    power_dunn_p_target_vs_distractor = power_metrics['Dunn Posthoc Target vs. Distractor']
    power_dunn_p_target_vs_non_target = power_metrics['Dunn Posthoc Target vs. Non-Target']
    power_dunn_p_distractor_vs_non_target = power_metrics['Dunn Posthoc Distractor vs. Non-Target']
    power_delta_target_vs_distractor = power_metrics['Cliff Delta Target vs Distractor']
    power_delta_target_vs_non_target = power_metrics['Cliff Delta Target vs Non-Target']
    power_delta_distractor_vs_non_target = power_metrics['Cliff Delta Distractor vs Non-Target']
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


        # Cliff delta comparison:
        def update_trend(value, pos, neg):
            if 0 < value <= 0.5:
                return pos + 1
            elif 0.5 < value <= 1:
                return pos + 2
            elif -0.5 <= value <= 0:
                return neg + 1
            elif -0.9 <= value < -0.5:
                return neg + 2
            return 0


        target += update_trend(power_delta_target_vs_distractor.iloc[index], target, distractor)
        target += update_trend(power_delta_target_vs_non_target.iloc[index], target, non_target)
        distractor += update_trend(power_delta_distractor_vs_non_target.iloc[index], distractor, non_target)

        trend_dict.iloc[index]['target'] = target
        trend_dict.iloc[index]['distractor'] = distractor
        trend_dict.iloc[index]['non_target'] = non_target

        trends_dict[sub] = trend_dict


for sub, df in trends_dict.items():

    df.to_csv(f'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/emg/subject_results/{sub}_emg_trends.csv')



