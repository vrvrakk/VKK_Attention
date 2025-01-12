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

# def plot_trends(df_combined, condition=''):
#     plt.figure(figsize=(12, 10))
#     for sub in df_combined['Subject']:
#         data = df_combined[df_combined['Subject'] == sub]
#         jitter = (np.random.rand(len(categories)) - 0.5) * 0.1
#         x_jittered = [0, 1, 2] + jitter
#         plt.plot(x_jittered, data[['target', 'distractor', 'non_target']].values[0],
#                  marker='o',
#                  label=sub)
#     plt.title(f'Trends in {condition} condition')
#     # Set X-axis labels to categories
#     plt.xticks(ticks=[0, 1, 2], labels=categories)
#     plt.yticks(ticks=[1, 2, 3, 4, 5])
#     plt.ylabel('Score')
#     plt.xlabel('Epoch Type')
#     plt.legend(title='Subjects')
#     plt.grid(alpha=0.3)
#     plt.show()
# 
# 
# plot_trends(a1_df_combined, condition='a1')
# plot_trends(a2_df_combined, condition='a2')
# plot_trends(e1_df_combined, condition='e1')
# plot_trends(e2_df_combined, condition='e2')



# # now to the dominant frequencies distributions:
# freqs_tables = {}
# for sub, frequency_metrics in subs_frequency_metrics.items():
#     freqs_tables[sub] = pd.DataFrame(data=None, columns=['Condition', 'Significance', 'Effect Size',
#                                                          'Target vs Distractor', 'Target vs Non-Target',
#                                                          'Distractor vs Non-Target'])
#
#     frequency_keys = list(frequency_metrics.keys())
#
#     # overall significance and eta-value:
#     freq_kruskal_p_key = frequency_keys[2]  # whether the group differences are statistically significant
#     freq_effect_size_key = frequency_keys[3]  # magnitude of group differences across all groups (global effect size)
#     freq_kruskal_p_vals = frequency_metrics[freq_kruskal_p_key]
#     freq_effect_sizes = frequency_metrics[freq_effect_size_key]
#     # pairwise comparison
#     target_vs_distractor_p_value_key = frequency_keys[
#         4]  # pairwise comparison of Target vs. Distractor groups, indicating significance
#     target_vs_no_target_p_value_key = frequency_keys[5]
#     distractor_vs_non_target_p_value_key = frequency_keys[6]
#
#     target_vs_distractor_p_values = frequency_metrics[target_vs_distractor_p_value_key]
#     target_vs_no_target_p_values = frequency_metrics[target_vs_no_target_p_value_key]
#     distractor_vs_non_target_p_values = frequency_metrics[distractor_vs_non_target_p_value_key]
#
#     frequency_conditions = frequency_metrics['Unnamed: 0']
#     freqs_tables[sub]['Condition'] = frequency_conditions
#
# for sub, freqs_table in freqs_tables.items():
#     frequency_metrics = subs_frequency_metrics[sub]  # Retrieve metrics for the current subject
#     freq_kruskal_p_vals = frequency_metrics[frequency_keys[2]]
#     freq_effect_sizes = frequency_metrics[frequency_keys[3]]
#     target_vs_distractor_p_values = frequency_metrics[frequency_keys[4]]
#     target_vs_no_target_p_values = frequency_metrics[frequency_keys[5]]
#     distractor_vs_non_target_p_values = frequency_metrics[frequency_keys[6]]
#     for condition, (index, row) in zip(condition_list, enumerate(freqs_table.iterrows())):
#         # extract freqs stats for interpretation:
#         freq_kruskal_p = freq_kruskal_p_vals.iloc[index]
#         freq_effect_size = freq_effect_sizes.iloc[index]
#
#         # dunn p-vals:
#         target_vs_distractor_p_value = target_vs_distractor_p_values.iloc[index]
#         target_vs_no_target_p_value = target_vs_no_target_p_values.iloc[index]
#         distractor_vs_non_target_p_value = distractor_vs_non_target_p_values.iloc[index]
#
#         # assign significance:
#         freqs_table.loc[index, 'Significance'] = freq_kruskal_p < 0.05
#
#         if freq_effect_size <= 0.01:
#             freqs_table.loc[index, 'Effect Size'] = 'Negligible'
#         elif 0.01 < freq_effect_size <= 0.06:
#             freqs_table.loc[index, 'Effect Size'] = 'Small'
#         elif 0.06 < freq_effect_size <= 0.14:
#             freqs_table.loc[index, 'Effect Size'] = 'Medium'
#         elif 0.14 < freq_effect_size:
#             freqs_table.loc[index, 'Effect Size'] = 'Large'
#
#         # determine significance of each pair:
#         if target_vs_distractor_p_value <= 0.001:
#             freqs_table.loc[index, 'Target vs Distractor'] = 'Highly Strong'
#         elif target_vs_distractor_p_value <= 0.01:
#             freqs_table.loc[index, 'Target vs Distractor'] = 'Strong'
#         elif target_vs_distractor_p_value <= 0.05:
#             freqs_table.loc[index, 'Target vs Distractor'] = 'Weak'
#         else:
#             freqs_table.loc[index, 'Target vs Distractor'] = 'No Significance'
#
#         if target_vs_no_target_p_value <= 0.001:
#             freqs_table.loc[index, 'Target vs Non-Target'] = 'Highly Strong'
#         elif target_vs_no_target_p_value <= 0.01:
#             freqs_table.loc[index, 'Target vs Non-Target'] = 'Strong'
#         elif target_vs_no_target_p_value <= 0.05:
#             freqs_table.loc[index, 'Target vs Non-Target'] = 'Weak'
#         else:
#             freqs_table.loc[index, 'Target vs Non-Target'] = 'No Significance'
#
#         if distractor_vs_non_target_p_value <= 0.001:
#             freqs_table.loc[index, 'Distractor vs Non-Target'] = 'Highly Strong'
#         elif distractor_vs_non_target_p_value <= 0.01:
#             freqs_table.loc[index, 'Distractor vs Non-Target'] = 'Strong'
#         elif distractor_vs_non_target_p_value <= 0.05:
#             freqs_table.loc[index, 'Distractor vs Non-Target'] = 'Weak'
#         else:
#             freqs_table.loc[index, 'Distractor vs Non-Target'] = 'No Significance'

# now power metrics: