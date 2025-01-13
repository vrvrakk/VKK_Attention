from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

# subs = input('Sub name: ')
sub_names = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub08', 'sub20', 'sub21', 'sub10', 'sub11','sub13', 'sub14', 'sub15','sub16', 'sub17', 'sub18', 'sub19', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27']
conditions = input('Specify conditions: ')

condition_list = []
for condition in conditions.split(','):
    condition_list.append(condition.strip())

default_path = Path.cwd()
subs_band_metrics = {}
subs_frequency_metrics = {}
subs_power_metrics = {}

for sub in sub_names:
    band_metrics, frequency_metrics, power_metrics = None, None, None

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

band_types = [f'Band {i}' for i in range(1, 151, 10)]  # Correct band names: Band 1, Band 11, Band 21, ...

bands_tables = {}
for sub, metrics in subs_band_metrics.items():
    band_chi_p_values = metrics['p-value']
    band_conditions = metrics['Unnamed: 0']
    # Rename Cramer's Value column
    metrics.rename(columns={f'{metrics.columns[4]}': 'Cramer Value'}, inplace=True)

    band_effect_size = metrics['Cramer Value']
    band_counts = metrics['Observed Counts']
    expected_band_counts = metrics['Expected Frequencies']

    # Convert each string in 'Observed Counts' to an actual dictionary
    import ast
    band_counts = band_counts.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
    expected_band_counts = expected_band_counts.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

    # Create table structure
    bands_tables[sub] = pd.DataFrame(data=None, columns=['Condition', 'Significance', 'Association'] +
                                     [f'Target {band} Dominance' for band in band_types] +
                                     [f'Distractor {band} Dominance' for band in band_types] +
                                     [f'Non-Target {band} Dominance' for band in band_types])
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

        # Significance
        bands_tables[sub].loc[index, 'Significance'] = chi_p_value < 0.05

        # Association
        if cramer_val == 0:
            bands_tables[sub].loc[index, 'Association'] = 'No Association'
        elif 0 < cramer_val <= 0.1:
            bands_tables[sub].loc[index, 'Association'] = 'Weak Association'
        elif 0.1 < cramer_val <= 0.3:
            bands_tables[sub].loc[index, 'Association'] = 'Moderate Association'
        elif 0.3 < cramer_val:
            bands_tables[sub].loc[index, 'Association'] = 'Strong Association'

        # Dynamically filter bands present in `band_counts`
        present_bands = [band for band in band_types if band in band_counts.iloc[index]]

        # Dominance percentages for each band
        for band in present_bands:
            # Target
            target_band_count = band_counts.iloc[index][band]['Target']
            total_target_band_count = sum(band_counts.iloc[index][b]['Target'] for b in present_bands)
            target_band_percentage = (target_band_count * 100) / total_target_band_count
            bands_tables[sub].loc[index, f'Target {band} Dominance'] = target_band_percentage

            # Distractor
            distractor_band_count = band_counts.iloc[index][band]['Distractor']
            total_distractor_band_count = sum(band_counts.iloc[index][b]['Distractor'] for b in present_bands)
            distractor_band_percentage = (distractor_band_count * 100) / total_distractor_band_count
            bands_tables[sub].loc[index, f'Distractor {band} Dominance'] = distractor_band_percentage

            # Non-Target
            non_target_band_count = band_counts.iloc[index][band]['Non-Target']
            total_non_target_band_count = sum(band_counts.iloc[index][b]['Non-Target'] for b in present_bands)
            non_target_band_percentage = (non_target_band_count * 100) / total_non_target_band_count
            bands_tables[sub].loc[index, f'Non-Target {band} Dominance'] = non_target_band_percentage

power_tables = {}
trends_dict = {}
for sub, power_metrics in subs_power_metrics.items():
    power_tables[sub] = pd.DataFrame(data=None,
                                     columns=['Condition', 'Effect Size', 'Significance', 'Target vs Distractor',
                                              'Target vs Non-Target', 'Distractor vs Non-Target', 'Trend'])

    power_tables[sub]['Condition'] = power_metrics['Unnamed: 0']

    trends_dict[sub] = pd.DataFrame({'a1': {'target': 1,
                                            'distractor': 1,
                                            'non_target': 1},
                                     'a2': {'target': 1,
                                            'distractor': 1,
                                            'non_target': 1},
                                     'e1': {'target': 1,
                                            'distractor': 1,
                                            'non_target': 1},
                                     'e2': {'target': 1,
                                            'distractor': 1,
                                            'non_target': 1}}).transpose()

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
        def update_trend(value):
            pos_update, neg_update = 0, 0  # Default updates are 0
            if 0 < value <= 0.5:
                pos_update = 1
            elif 0.5 < value <= 1:
                pos_update = 2
            elif value <= 0:  # Negative value
                if 0.5 >= abs(value) >= 0:
                    neg_update = 1
                elif 0.9 >= abs(value) > 0.5:
                    neg_update = 2
            return pos_update, neg_update


        target_update, distractor_update = update_trend(power_delta_target_vs_distractor.iloc[index])
        target += target_update
        distractor += distractor_update

        target_update, non_target_update = update_trend(power_delta_target_vs_non_target.iloc[index])
        target += target_update
        non_target += non_target_update
        distractor_update, non_target_update = update_trend(power_delta_distractor_vs_non_target.iloc[index])
        distractor += distractor_update
        non_target += non_target_update

        trend_dict.iloc[index]['target'] = target
        trend_dict.iloc[index]['distractor'] = distractor
        trend_dict.iloc[index]['non_target'] = non_target

    trends_dict[sub] = trend_dict


for sub, df in trends_dict.items():

    df.to_csv(f'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/emg/subject_results/{sub}_emg_trends.csv')


categories = ['Target', 'Distractor', 'Non-Target']

a1_trends = {}
a2_trends = {}
e1_trends = {}
e2_trends = {}

for sub, df in trends_dict.items():
    for index, rows in enumerate(df.iterrows()):
        if index == 0:
            condition_a1 = df.iloc[index]
            a1_trends[sub] = pd.DataFrame(condition_a1).T
        elif index == 1:
            condition_a2 = df.iloc[index]
            a2_trends[sub] = pd.DataFrame(condition_a2).T
        elif index == 2:
            e1_condition = df.iloc[index]
            e1_trends[sub] = pd.DataFrame(e1_condition).T
        elif index == 3:
            e2_condition = df.iloc[index]
            e2_trends[sub] = pd.DataFrame(e2_condition).T

def concat_dfs(dict):
    dataframes = []
    for name, df in dict.items():
        df['Subject'] = name
        df = df.rename(columns={'index': 'Condition'})
        dataframes.append(df)
    # Concatenate all DataFrames into one
    df_combined = pd.concat(dataframes, ignore_index=True)
    return df_combined

a1_df_combined = concat_dfs(a1_trends)
a2_df_combined = concat_dfs(a2_trends)
e1_df_combined = concat_dfs(e1_trends)
e2_df_combined = concat_dfs(e2_trends)

# plot trends for each condition:


# Step 2: Define a function to categorize trends
def trends_percentages(trends):
    def categorize_trend(row):
        values = {'Target': row['target'], 'Distractor': row['distractor'], 'Non-Target': row['non_target']}
        sorted_values = sorted(values.items(), key=lambda x: x[1], reverse=True)  # Sort by scores in descending order
        return " > ".join([key for key, _ in sorted_values])  # Create trend string
    
    # Step 3: Apply the function to the combined DataFrame
    trends['Trend'] = trends.apply(categorize_trend, axis=1)
    
    # Step 4: Calculate percentages for each trend
    trend_counts = trends['Trend'].value_counts()  # Count occurrences of each trend
    trend_percentages = (trend_counts / trend_counts.sum()) * 100  # Convert to percentages
    
    # Display the results
    print("Trend Percentages:")
    print(trend_percentages)
    
    # Optional: Save the percentages to a CSV file
    trend_percentages.to_csv('trend_percentages_combined.csv', index=True, header=['Percentage'])
    return trend_percentages
a1_trends_percentages = trends_percentages(a1_df_combined)
a2_trends_percentages = trends_percentages(a2_df_combined)
e1_trends_percentages = trends_percentages(e1_df_combined)
e2_trends_percentages = trends_percentages(e2_df_combined)

# Combine all percentages into a single DataFrame
all_trends = pd.DataFrame({
    "a1": a1_trends_percentages,
    "a2": a2_trends_percentages,
    "e1": e1_trends_percentages,
    "e2": e2_trends_percentages
}).fillna(0)  # Fill NaN with 0 for missing trends in some conditions

# Plot the trends as a grouped bar chart
all_trends.T.plot(kind='bar', figsize=(12, 8), width=0.8)
plt.title('Power Trend Percentages Across Conditions')
plt.ylabel('Percentage (%)')
plt.xlabel('Conditions')
plt.legend(title='Trends', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()

# Save and show the plot
plt.savefig(f'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/emg/subject_results/figures/all_power_emg_trends_across_conditions.png')
plt.show()

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

significant_power_subs = {}
for sub, df in power_tables.items():
    for index, row in df.iterrows():
        if row['Target vs Distractor'] != 'No Significance' and row['Target vs Non-Target'] != 'No Significance' and row['Distractor vs Non-Target'] != 'No Significance':
            significant_power_subs[sub] = True
        else:
            significant_power_subs[sub] = False

from collections import Counter

sub_power_counts = Counter(significant_power_subs.values())

significant_freq_subs = {}
for sub, df in freqs_tables.items():
    for index, row in df.iterrows():
        if row['Target vs Distractor'] != 'No Significance' and row['Target vs Non-Target'] != 'No Significance' and row['Distractor vs Non-Target'] != 'No Significance':
            significant_freq_subs[sub] = True
        else:
            significant_freq_subs[sub] = False

sub_freq_counts = Counter(significant_freq_subs.values())
all_freq_counts = sub_freq_counts[0] + sub_freq_counts[1]
significant_freq_counts = sub_freq_counts[1]
# percentage of significant counts:
percentage_significant_freqs = (significant_freq_counts * 100) / all_freq_counts

# percentage of significant power counts:
all_power_counts = sub_power_counts[0] + sub_power_counts[1]
significant_power_counts = sub_power_counts[1]
percentage_significant_power = (significant_power_counts * 100) / all_power_counts

# get the über vals now for the über stats table:
from scipy.stats import combine_pvalues
''' combining p-values from independent tests that bear upon the same hypothesis.
    only for continuous distributions.
    pvalues: array, 1D
    method: fisher
    total=−2⋅(ln(p1)+ln(p2)+ln(p3)+⋯+ln(pn))
    Degrees of Freedom:

    The degrees of freedom (dfdf) is 2 times the number of p-values combined.

    Find the Combined P-value:

    Use the chi-squared distribution to calculate the combined p-value based on:
        total (the test statistic you calculated).
        df (the degrees of freedom).
    For each category or group:

    Find the difference between the observed and expected counts.
    Square the difference (to make it positive).
    Divide by the expected count to account for scaling.

Add up these values across all categories to get the chi-squared statistic.
    '''
# for power:
all_kruskal_p_values = []
all_dunn_p_values_target_vs_distractor = []
all_dunn_p_values_target_vs_non_target = []
all_dunn_p_values_distractor_vs_non_target = []
all_size_effects = []
all_levene_variances = []

for sub, df in subs_power_metrics.items():
    power_metrics = subs_power_metrics[sub]
    power_p_vals = power_metrics['Kruskal-Wallis p-value']
    all_kruskal_p_values.append(power_p_vals)
    eta_squared = power_metrics['Effect Size (eta-squared)']
    all_size_effects.append(eta_squared)
    levene_variance = power_metrics['Levene statistic (variance)']
    all_levene_variances.append(levene_variance)
    # power_eta_squared = power_metrics['Effect Size (eta-squared)']
    power_dunn_p_target_vs_distractor = power_metrics['Dunn Posthoc Target vs. Distractor']
    all_dunn_p_values_target_vs_distractor.append(power_dunn_p_target_vs_distractor)
    power_dunn_p_target_vs_non_target = power_metrics['Dunn Posthoc Target vs. Non-Target']
    all_dunn_p_values_target_vs_non_target.append(power_dunn_p_target_vs_non_target)
    power_dunn_p_distractor_vs_non_target = power_metrics['Dunn Posthoc Distractor vs. Non-Target']
    all_dunn_p_values_distractor_vs_non_target.append(power_dunn_p_distractor_vs_non_target)

a1_eta_squared = [vals[0] for vals in all_size_effects]
a2_eta_squared = [vals[1] for vals in all_size_effects]
e1_eta_squared = [vals[2] for vals in all_size_effects if len(vals) > 2]
e2_eta_squared = [vals[3] for vals in all_size_effects if len(vals) > 2]

a1_overall_eta_squared = np.mean(a1_eta_squared)
a1_eta_squared_std = np.std(a1_eta_squared)
print(f"Overall Effect Size (eta-squared): {a1_overall_eta_squared:.3f} +- {a1_eta_squared_std:.3f}")

a2_overall_eta_squared = np.mean(a2_eta_squared)
a2_eta_squared_std = np.std(a2_eta_squared)
print(f"Overall Effect Size (eta-squared): {a2_overall_eta_squared:.3f} +- {a2_eta_squared_std:.3f}")

e1_overall_eta_squared = np.mean(e1_eta_squared)
e1_eta_squared_std = np.std(e1_eta_squared)
print(f"Overall Effect Size (eta-squared): {e1_overall_eta_squared:.3f} +- {e1_eta_squared_std:.3f}")

e2_overall_eta_squared = np.mean(e2_eta_squared)
e2_eta_squared_std = np.std(e2_eta_squared)
print(f"Overall Effect Size (eta-squared): {e2_overall_eta_squared:.3f} +- {e2_eta_squared_std:.3f}")


a1_variance = [vals[0] for vals in all_levene_variances]
a1_levene_stats = [float(val.split(',')[0].strip('[')) for val in a1_variance]
a1_levene_p_values = [float(val.split(',')[1].strip(']')) for val in a1_variance]
a2_variance = [vals[1] for vals in all_levene_variances]
a2_levene_stats = [float(val.split(',')[0].strip('[')) for val in a2_variance]
a2_levene_p_values = [float(val.split(',')[1].strip(']')) for val in a2_variance]
e1_variance = [vals[2] for vals in all_levene_variances if len(vals) > 2]
e1_levene_stats = [float(val.split(',')[0].strip('[')) for val in e1_variance]
e1_levene_p_values = [float(val.split(',')[1].strip(']')) for val in e1_variance]
e2_variance = [vals[3] for vals in all_levene_variances if len(vals) > 2]
e2_levene_stats = [float(val.split(',')[0].strip('[')) for val in e2_variance]
e2_levene_p_values = [float(val.split(',')[1].strip(']')) for val in e2_variance]

# get overall variance p val and and variance vals:
def overall_var(levene_stats):
    overall_levene_stat = np.mean(levene_stats)
    levene_stat_std = np.std(levene_stats)
    print(f"Overall Levene Statistic (variance): {overall_levene_stat:.3f} +- {levene_stat_std:.3f}")
    return overall_levene_stat, levene_stat_std

a1_overall_levene_stat, a1_levene_stat_std = overall_var(a1_levene_stats)
a2_overall_levene_stat, a2_levene_stat_std = overall_var(a2_levene_stats)
e1_overall_levene_stat, e1_levene_stat_std = overall_var(e1_levene_stats)
e2_overall_levene_stat, e2_levene_stat_std = overall_var(e2_levene_stats)

def get_var_p_vals(levene_p_values):
    combined_stat, combined_p_value = combine_pvalues(levene_p_values, method='fisher')
    print(f"Combined Levene p-value: {combined_p_value:.3e}")
    return combined_stat, combined_p_value

a1_combined_stat, a1_combined_p_value= get_var_p_vals(a1_levene_p_values)
a2_combined_stat, a2_combined_p_value = get_var_p_vals(a2_levene_p_values)
e1_combined_stat, e1_combined_p_value = get_var_p_vals(e1_levene_p_values)
e2_combined_stat, e2_combined_p_value = get_var_p_vals(e2_levene_p_values)


a1_kruskal = [vals[0] for vals in all_kruskal_p_values]
a2_kruskal = [vals[1] for vals in all_kruskal_p_values]
e1_kruskal = [vals[2] for vals in all_kruskal_p_values if len(vals) > 2]
e2_kruskal = [vals[3] for vals in all_kruskal_p_values if len(vals) > 2]
all_kruskal_vals = [a1_kruskal, a2_kruskal, e1_kruskal, e2_kruskal]

def combined_p_vals(all_vals, condition_list):
    combined_p_vals_dict = {}
    for condition, vals in zip(condition_list, all_vals):
        combined_stat, combined_p = combine_pvalues(vals, method='fisher')
        print(f"Condition: {condition}, Combined p-value: {combined_p}")
        combined_p_vals_dict[condition] = {'combined_stat': combined_stat, 'combined_p': combined_p}
    return combined_p_vals_dict
combined_kruskal_p_vals_dict = combined_p_vals(all_kruskal_vals, condition_list)

# indicates that the p-values are consistently small, providing strong evidence against the null hypothesis

# Combine Dunn Posthoc p-values for each comparison
# Target vs. Distractor
target_vs_distractor_a1 = [vals[0] for vals in all_dunn_p_values_target_vs_distractor]
target_vs_distractor_a2 = [vals[1] for vals in all_dunn_p_values_target_vs_distractor]
target_vs_distractor_e1 = [vals[2] for vals in all_dunn_p_values_target_vs_distractor if len(vals) > 2]
target_vs_distractor_e2 = [vals[3] for vals in all_dunn_p_values_target_vs_distractor if len(vals) > 2]
combined_target_vs_distractor_vals = [target_vs_distractor_a1, target_vs_distractor_a2, target_vs_distractor_e1, target_vs_distractor_e2]
combined_target_vs_distractor_dict = combined_p_vals(combined_target_vs_distractor_vals, condition_list)

# Target vs. Non-Target
target_vs_non_target_a1 = [vals[0] for vals in all_dunn_p_values_target_vs_non_target]
target_vs_non_target_a2 = [vals[1] for vals in all_dunn_p_values_target_vs_non_target]
target_vs_non_target_e1 = [vals[2] for vals in all_dunn_p_values_target_vs_non_target if len(vals) > 2]
target_vs_non_target_e2 = [vals[3] for vals in all_dunn_p_values_target_vs_non_target if len(vals) > 2]
combined_target_vs_non_target_vals = [target_vs_non_target_a1, target_vs_non_target_a2, target_vs_non_target_e1, target_vs_non_target_e2]
combined_target_vs_non_target_dict = combined_p_vals(combined_target_vs_non_target_vals, condition_list)

# Distractor vs. Non-Target
distractor_vs_non_target_a1 = [vals[0] for vals in all_dunn_p_values_distractor_vs_non_target]
distractor_vs_non_target_a2 = [vals[1] for vals in all_dunn_p_values_distractor_vs_non_target]
distractor_vs_non_target_e1 = [vals[2] for vals in all_dunn_p_values_distractor_vs_non_target if len(vals) > 2]
distractor_vs_non_target_e2 = [vals[3] for vals in all_dunn_p_values_distractor_vs_non_target if len(vals) > 2]
distractor_vs_non_target_vals = [distractor_vs_non_target_a1, distractor_vs_non_target_a2, distractor_vs_non_target_e1, distractor_vs_non_target_e2]
combined_distractor_vs_non_target_dict = combined_p_vals(distractor_vs_non_target_vals, condition_list)

# for frequencies:
