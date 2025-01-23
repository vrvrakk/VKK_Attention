from pathlib import Path
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
#
sub_names = ['sub01', 'sub02', 'sub03', 'sub10' ,
             'sub04', 'sub05', 'sub06', 'sub08',
             'sub11','sub13', 'sub14', 'sub15',
             'sub16',  'sub17', 'sub18', 'sub19', 'sub20', 'sub21',
             'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27', 'sub28', 'sub29']


conditions = input('Specify conditions: ')

condition_list = []
for condition in conditions.split(','):
    condition_list.append(condition.strip())

default_path = Path.cwd()
data_path = default_path /'data'/'emg'/'subject_results'

all_band_metrics = {}
all_frequency_metrics = {}
all_power_metrics = {}

for file in data_path.iterdir():
    print(file)
    if 'all_chi2' in file.name:
        chi2_filename = file
        band_metrics = pd.read_csv(chi2_filename)
        # Update the band metrics dictionary
        all_band_metrics[chi2_filename.name] = band_metrics
    elif 'all_power' in file.name:
        print(file)
        power_filename = file
        power_metrics = pd.read_table(power_filename, delimiter=',')
        # Update the power metrics dictionary
        all_power_metrics[power_filename.name] = power_metrics
    elif 'all_frequency' in file.name:
        freq_filename = file
        frequency_metrics = pd.read_table(freq_filename, delimiter=',')
        # Update the frequency metrics dictionary
        all_frequency_metrics[freq_filename.name] = frequency_metrics
# get band metrics:
# chi2 statistic:
import ast
def extract_band_stats(all_band_metrics, condition=''):
    chi2_columns = list(all_band_metrics[f'{condition}_all_chi2_metrics.csv'].keys())
    chi2 = all_band_metrics[f'{condition}_all_chi2_metrics.csv']['Chi-Square Statistic'][0]
    chi2_df = all_band_metrics[f'{condition}_all_chi2_metrics.csv']['Degrees of Freedom'][0]
    chi2_size_effect = all_band_metrics[f'{condition}_all_chi2_metrics.csv'].iloc[0, 4]
    observed_counts = all_band_metrics[f'{condition}_all_chi2_metrics.csv']['Observed Counts'].dropna()
    # Convert strings to dictionaries
    observed_counts_dicts = [ast.literal_eval(entry) for entry in observed_counts]
    # Frequency range labels
    freq_ranges = [f"{i}-{i + 9} Hz" for i in range(1, 151, 10)]
    # Create DataFrame
    observed_counts_df = pd.DataFrame(observed_counts_dicts, index=freq_ranges)
    expected_counts = all_band_metrics[f'{condition}_all_chi2_metrics.csv']['Expected Frequencies'].dropna()
    expected_counts_dicts = [ast.literal_eval(entry) for entry in expected_counts]
    expected_counts_df = pd.DataFrame(expected_counts_dicts, index=freq_ranges)
    return chi2, chi2_df, chi2_size_effect, observed_counts_df, expected_counts_df

a1_chi2, a1_chi2_df, a1_chi2_size_effect, a1_observed_counts_df, a1_expected_counts_df = extract_band_stats(all_band_metrics, condition='a1')
a2_chi2, a2_chi2_df, a2_chi2_size_effect, a2_observed_counts_df, a2_expected_counts_df = extract_band_stats(all_band_metrics, condition='a2')
e1_chi2, e1_chi2_df, e1_chi2_size_effect, e1_observed_counts_df, e1_expected_counts_df = extract_band_stats(all_band_metrics, condition='e1')
e2_chi2, e2_chi2_df, e2_chi2_size_effect, e2_observed_counts_df, e2_expected_counts_df = extract_band_stats(all_band_metrics, condition='e2')

# extract frequency distribution stats:
def extract_frequency_stats(all_frequency_metrics, condition=''):
    kruskal_h_value = all_frequency_metrics[f'{condition}_all_frequency_metrics.csv']['Kruskal-Wallis H'].dropna()
    kruskal_p_value = all_frequency_metrics[f'{condition}_all_frequency_metrics.csv']['Kruskal-Wallis p-value'].dropna()
    kruskal_eta_squared = all_frequency_metrics[f'{condition}_all_frequency_metrics.csv']['Kruskal Effect Size (eta-squared)'].dropna()
    normality_stats = pd.DataFrame({'H-value': kruskal_h_value, 'p-value': kruskal_p_value, 'eta-squared': kruskal_eta_squared})
    dunn_p_target_distractor = all_frequency_metrics[f'{condition}_all_frequency_metrics.csv']['Dunn Posthoc Target vs. Distractor'].dropna()
    dunn_p_target_non_target = all_frequency_metrics[f'{condition}_all_frequency_metrics.csv']['Dunn Posthoc Target vs. Non-Target'].dropna()
    dunn_p_distractor_non_target = all_frequency_metrics[f'{condition}_all_frequency_metrics.csv']['Dunn Posthoc Distractor vs. Non-Target'].dropna()
    dunn_p_vals = pd.DataFrame({'Target vs Distractor': dunn_p_target_distractor, 'Target vs Non-Target': dunn_p_target_non_target, 'Distractor vs Non-Target': dunn_p_distractor_non_target})
    delta_target_distractor = all_frequency_metrics[f'{condition}_all_frequency_metrics.csv']['Cliff Delta Target vs Distractor'].dropna()
    delta_target_non_target = all_frequency_metrics[f'{condition}_all_frequency_metrics.csv']['Cliff Delta Target vs Non-Target'].dropna()
    delta_distractor_non_target = all_frequency_metrics[f'{condition}_all_frequency_metrics.csv']['Cliff Delta Distractor vs Non-Target'].dropna()
    delta_size_effects = pd.DataFrame({'Target vs Distractor': delta_target_distractor, 'Target vs Non-Target': delta_target_non_target, 'Distractor vs Non-Target': delta_distractor_non_target})
    target_stats = all_frequency_metrics[f'{condition}_all_frequency_metrics.csv']['Target'].dropna()
    target_dicts = [ast.literal_eval(entry) for entry in target_stats]
    target_df = pd.DataFrame(target_dicts)
    distractor_stats = all_frequency_metrics[f'{condition}_all_frequency_metrics.csv']['Distractor'].dropna()
    distractor_dicts = [ast.literal_eval(entry) for entry in distractor_stats]
    distractor_df = pd.DataFrame(distractor_dicts)
    non_target_stats = all_frequency_metrics[f'{condition}_all_frequency_metrics.csv']['Non-Target'].dropna()
    non_target_dicts = [ast.literal_eval(entry) for entry in non_target_stats]
    non_target_df = pd.DataFrame(non_target_dicts)
    index_names = ['Target', 'Distractor', 'Non-Target']
    statistics_df = pd.concat((target_df, distractor_df, non_target_df))
    statistics_df.index = index_names
    return statistics_df, normality_stats, dunn_p_vals, delta_size_effects
    
a1_statistics_df, a1_normality_stats, a1_dunn_p_vals, a1_delta_size_effects = extract_frequency_stats(all_frequency_metrics, condition='a1')
a2_statistics_df, a2_normality_stats, a2_dunn_p_vals, a2_delta_size_effects = extract_frequency_stats(all_frequency_metrics, condition='a2')
e1_statistics_df, e1_normality_stats, e1_dunn_p_vals, e1_delta_size_effects = extract_frequency_stats(all_frequency_metrics, condition='e1')
e2_statistics_df, e2_normality_stats, e2_dunn_p_vals, e2_delta_size_effects = extract_frequency_stats(all_frequency_metrics, condition='e2')

def extract_power_stats(all_power_metrics, condition=''):
    stats = all_power_metrics[f'{condition}_all_power_metrics.csv']['Descriptive Statistics'].dropna()
    stats_dict = [ast.literal_eval(entry) for entry in stats]
    descriptive_stats_df = pd.DataFrame(stats_dict, index=['Target', 'Distractor', 'Non-Target'])
    stat_tests = all_power_metrics[f'{condition}_all_power_metrics.csv']['Statistical Tests'].dropna()
    stat_tests_keys = list(stat_tests.keys())
    kruskal_h_val = stat_tests[stat_tests_keys[0]]
    kruskal_p_val = stat_tests[stat_tests_keys[1]]
    kruskal_h_squared = stat_tests[stat_tests_keys[2]]
    kruskal_vals = pd.DataFrame({
        'H-value': [kruskal_h_val],
        'p-value': [kruskal_p_val],
        'h-squared': [kruskal_h_squared]
    }, index=['Kruskal-Wallis'])
    dunn_p_val_target_distractor = stat_tests[stat_tests_keys[3]]
    dunn_p_val_target_non_target = stat_tests[stat_tests_keys[4]]
    dunn_p_val_distractor_non_target = stat_tests[stat_tests_keys[5]]
    dunn_stats = pd.DataFrame({
        'Target vs Distractor': [dunn_p_val_target_distractor],
        'Target vs Non-Target': [dunn_p_val_target_non_target],
        'Distractor vs Non-Target': [dunn_p_val_distractor_non_target]
    })

    # Extract Cliff's Delta values
    delta_target_vs_distractor = stat_tests[stat_tests_keys[6]]
    delta_target_vs_non_target = stat_tests[stat_tests_keys[7]]
    delta_distractor_vs_non_target = stat_tests[stat_tests_keys[8]]
    size_effects = pd.DataFrame({
        'Target vs Distractor': [delta_target_vs_distractor],
        'Target vs Non-Target': [delta_target_vs_non_target],
        'Distractor vs Non-Target': [delta_distractor_vs_non_target]
    })
    return descriptive_stats_df, kruskal_vals, dunn_stats, size_effects


a1_descriptive_stats_df, a1_kruskal_vals, a1_dunn_stats, a1_size_effects = extract_power_stats(all_power_metrics, condition='a1')
a2_descriptive_stats_df, a2_kruskal_vals, a2_dunn_stats, a2_size_effects = extract_power_stats(all_power_metrics, condition='a2')
e1_descriptive_stats_df, e1_kruskal_vals, e1_dunn_stats, e1_size_effects = extract_power_stats(all_power_metrics, condition='e1')
e2_descriptive_stats_df, e2_kruskal_vals, e2_dunn_stats, e2_size_effects = extract_power_stats(all_power_metrics, condition='e2')


# Cliff delta comparison: across individual subs
def assign_significance(p_value):
    if p_value <= 0.001:
        return 'Highly Strong'
    elif p_value <= 0.01:
        return 'Strong'
    elif p_value <= 0.05:
        return 'Weak'
    else:
        return 'No Significance'

subs_power_metrics_a1 = {}
subs_power_metrics_a2 = {}
subs_power_metrics_e1 = {}
subs_power_metrics_e2 = {}
for sub_name in sub_names:
    sub_data_path = default_path / 'data' / 'emg' / sub_name / 'preprocessed' / 'results'

    for file in sub_data_path.iterdir():
        if 'a1_avg_power_metrics' in file.name:
            subs_power_metrics_a1[sub_name] = pd.read_table(file, delimiter=',')
        elif 'a2_avg_power_metrics' in file.name:
            subs_power_metrics_a2[sub_name] = pd.read_table(file, delimiter=',')
        elif 'e1_avg_power_metrics' in file.name:
            subs_power_metrics_e1[sub_name] = pd.read_table(file, delimiter=',')
        elif 'e2_avg_power_metrics' in file.name:
            subs_power_metrics_e2[sub_name] = pd.read_table(file, delimiter=',')



def create_trends_dict(subs_power_metrics, idx_key):
    trends_dict = {}
    power_tables = {}
    for sub, power_metrics in subs_power_metrics.items(): # could be any sub_power_metrics_a1, or a2, or e1 and e2
        # it's to initialize the trends_dict[sub]
        power_tables[sub] = pd.DataFrame(data=None,
                                         columns=['Target vs Distractor',
                                                  'Target vs Non-Target', 'Distractor vs Non-Target'], index=[idx_key])

        trends_dict[sub] = pd.DataFrame({'target': [1],
                                                'distractor': [1],
                                                'non_target': [1]}).transpose()
    return trends_dict, power_tables

trends_dict_a1, power_tables_a1 = create_trends_dict(subs_power_metrics_a1, idx_key='a1')
trends_dict_a2, power_tables_a2 = create_trends_dict(subs_power_metrics_a2, idx_key='a2')
trends_dict_e1, power_tables_e1 = create_trends_dict(subs_power_metrics_e1, idx_key='e1')
trends_dict_e2, power_tables_e2 = create_trends_dict(subs_power_metrics_e2, idx_key='e2')

def get_trend_vals(trends_dict, power_tables, subs_power_metrics, condition):
    print(f'getting trends for condition {condition}')
    if condition in ['e1', 'e2']:
        # Exclude keys 'sub01' to 'sub08'
        filtered_power_tables = {
            key: value for key, value in power_tables.items()
            if key not in [f"sub{str(i).zfill(2)}" for i in [1, 2, 3, 4, 5, 6, 8]]
        }
    else:
        filtered_power_tables = power_tables

    for sub, power_table in filtered_power_tables.items():
        trend_dict = trends_dict[sub].transpose()
        power_metrics = subs_power_metrics[sub]

        # Map Statistical Tests using the 'Unnamed: 0' column
        statistical_tests = power_metrics.set_index('Unnamed: 0')['Statistical Tests'].dropna()

        # Extract statistical values for the given condition
        dunn_target_vs_distractor = statistical_tests['Dunn Posthoc Target vs. Distractor']
        dunn_target_vs_non_target = statistical_tests['Dunn Posthoc Target vs. Non-Target']
        dunn_distractor_vs_non_target = statistical_tests['Dunn Posthoc Distractor vs. Non-Target']

        delta_target_vs_distractor = statistical_tests['Cliff Delta Target vs Distractor']
        delta_target_vs_non_target = statistical_tests['Cliff Delta Target vs Non-Target']
        delta_distractor_vs_non_target = statistical_tests['Cliff Delta Distractor vs Non-Target']
        print({'delta_target_vs_distractor': delta_target_vs_distractor, 'delta_target_vs_non_target':delta_target_vs_non_target, 'delta_distractor_vs_non_target':delta_distractor_vs_non_target})

        # Retrieve current trend values
        target = trend_dict['target']
        distractor = trend_dict['distractor']
        non_target = trend_dict['non_target']

        # Pairwise significance
        power_table['Target vs Distractor'] = assign_significance(
            dunn_target_vs_distractor)
        power_table['Target vs Non-Target'] = assign_significance(
            dunn_target_vs_non_target)
        power_table['Distractor vs Non-Target'] = assign_significance(
            dunn_distractor_vs_non_target)

        # Update trends
        def update_trend(value):
            pos_update, neg_update = 0, 0
            if 0 < value <= 0.5:
                pos_update = 1
            elif 0.5 < value <= 1:
                pos_update = 2
            elif value <= 0:
                if 0.5 >= abs(value) >= 0:
                    neg_update = 1
                elif 0.9 >= abs(value) > 0.5:
                    neg_update = 2
            return pos_update, neg_update

        target_update, distractor_update = update_trend(delta_target_vs_distractor)
        target += target_update
        distractor += distractor_update

        target_update, non_target_update = update_trend(delta_target_vs_non_target)
        target += target_update
        non_target += non_target_update

        distractor_update, non_target_update = update_trend(delta_distractor_vs_non_target)
        distractor += distractor_update
        non_target += non_target_update

        # Update trend_dict
        trend_dict['target'] = target
        trend_dict['distractor'] = distractor
        trend_dict['non_target'] = non_target

        # Update trends_dict for the current subject
        trends_dict[sub] = trend_dict

    return trends_dict

trends_dict_a1 = get_trend_vals(trends_dict_a1, power_tables_a1, subs_power_metrics_a1, condition='a1')
trends_dict_a2 = get_trend_vals(trends_dict_a2, power_tables_a2, subs_power_metrics_a2, condition='a2')
trends_dict_e1 = get_trend_vals(trends_dict_e1, power_tables_e1, subs_power_metrics_e1, condition='e1')
trends_dict_e2 = get_trend_vals(trends_dict_e2, power_tables_e2, subs_power_metrics_e2, condition='e2')


def save_trends(trends_dict, condition=''):
    for sub, df in trends_dict.items():

        df.to_csv(f'C:/Users/vrvra/PycharmProjects/VKK_Attention/data/emg/subject_results/{sub}_{condition}emg_trends.csv')


save_trends(trends_dict_a1, condition='a1')
save_trends(trends_dict_a2, condition='a2')
save_trends(trends_dict_e1, condition='e1')
save_trends(trends_dict_e2, condition='e2')


categories = ['Target', 'Distractor', 'Non-Target']

def concat_dfs(dict):
    dataframes = []
    for name, df in dict.items():
        df['Subject'] = name
        df = df.rename(columns={'index': 'Condition'})
        dataframes.append(df)
    # Concatenate all DataFrames into one
    df_combined = pd.concat(dataframes, ignore_index=True)
    return df_combined

a1_df_combined = concat_dfs(trends_dict_a1)
a2_df_combined = concat_dfs(trends_dict_a2)
e1_df_combined = concat_dfs(trends_dict_e1)
e2_df_combined = concat_dfs(trends_dict_e2)

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

