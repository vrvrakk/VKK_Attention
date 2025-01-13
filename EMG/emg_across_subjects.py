import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from scipy.stats import f_oneway, kruskal, shapiro
import scikit_posthocs as sp
from pathlib import Path
from scipy.stats import chi2_contingency

# Directory containing saved subject results
default_dir = Path.cwd()
subject_results_dir = default_dir / 'data'/ 'emg'/ 'subject_results'
fig_path = subject_results_dir / 'figures'
os.makedirs(fig_path, exist_ok=True)
#
sub_list = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub08', 'sub20', 'sub21', 'sub10', 'sub11','sub13', 'sub14', 'sub15','sub16', 'sub17', 'sub18', 'sub19', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26', 'sub27']
# Prompt user for conditions and convert input into a list
conditions_input = input("Enter conditions separated by commas (e.g., 'a1, e1'): ")
conditions_list = [condition.strip() for condition in conditions_input.split(',')]

epoch_types = ['Target', 'Distractor', 'Non-Target']
tfa_categories = ['Target', 'Distractor', 'Non-Target']

# Initialize an empty dictionary to hold averaged results for each category and condition
avg_tfa_results = {category: {condition: None for condition in conditions_list} for category in tfa_categories}

# Load and aggregate TFA results for each subject
for sub in sub_list:
    with open(subject_results_dir / 'tfa' /  f'tfa_dict_{sub}.pkl', 'rb') as f:
        tfa_results = pickle.load(f)

        for category in tfa_categories:
            for condition in conditions_list:
                if condition in tfa_results[category]:
                    power_data = tfa_results[category][condition]
                    if avg_tfa_results[category][condition] is None:
                        avg_tfa_results[category][condition] = power_data
                    else:
                        avg_tfa_results[category][condition] += power_data

# Average the results by dividing by the number of subjects
num_subjects = len(sub_list)
for category in tfa_categories:
    for condition in conditions_list:
        if avg_tfa_results[category][condition] is not None:
            avg_tfa_results[category][condition] /= num_subjects

print(f"Individual heatmaps have been saved to {fig_path}.")
# Function to load and combine results from multiple .pkl files
# Load all subjects' data from multiple .pkl files
def load_all_subject_results(subject_results_dir, conditions_list):
    """
    Load and combine results from multiple .pkl files in a directory.

    Args:
        subject_results_dir (Path): Path to the directory with subject .pkl files.
        conditions (list): List of conditions (e.g., ['target', 'distractor', 'non_target']).

    Returns:
        dict: Dictionary with combined results from all subjects.
    """
    all_results = {epoch_type: {cond: [] for cond in conditions_list} for epoch_type in epoch_types}

    for filepath in subject_results_dir.glob("*.pkl"):
        with open(filepath, 'rb') as f:
            subject_data = pickle.load(f)
            for epoch_type in epoch_types:
                for condition in conditions_list:
                    if epoch_type in subject_data:
                        if condition in subject_data[epoch_type]:
                            all_results[epoch_type][condition].append(subject_data[epoch_type][condition])

    return all_results
# Function to add significance labels
def significance_label(p_val):
    if p_val < 0.0001:
        return "****"
    elif p_val < 0.001:
        return "***"
    elif p_val < 0.01:
        return "**"
    elif p_val < 0.05:
        return "*"
    else:
        return "ns"


def plot_aggregated_dominant_band_distributions(all_results):
    band_types = ['band_1_10', 'band_11_20', 'band_21_30', 'band_31_40', 'band_41_50', 'band_51_60', 'band_61_70',
                  'band_71_80', 'band_81_90', 'band_91_100',
                  'band_101_110', 'band_111_120', 'band_121_130', 'band_131_140', 'band_141_150']
    epoch_label_map = {
        'Target': 'Target',
        'Distractor': 'Distractor',
        'Non-Target': 'Non-Target'
    }
    for cond in conditions_list:  # Iterate through each condition
        # Initialize aggregated counts for each epoch type and band
        aggregated_counts = {epoch_type: Counter({band: 0 for band in band_types}) for epoch_type in epoch_types}

        # Aggregate counts of dominant bands for each condition
        for epoch_type in epoch_types:
            dominant_bands = [epoch['dominant_band'] for subject_data in all_results[epoch_type][cond] for epoch in subject_data]
            for band in band_types:
                aggregated_counts[epoch_type][band] += dominant_bands.count(band)

        # Filter out bands with 0 counts across all epoch types
        filtered_band_types = [band for band in band_types if any(aggregated_counts[epoch_type][band] > 0 for epoch_type in epoch_types)]

        # Update aggregated counts to include only filtered bands
        filtered_counts = {epoch_type: {band: aggregated_counts[epoch_type][band] for band in filtered_band_types}
                           for epoch_type in epoch_types}

        # Normalize counts for visualization
        normalized_counts = {
            epoch_type: {band: count / sum(filtered_counts[epoch_type].values())
                         for band, count in filtered_counts[epoch_type].items()}
            for epoch_type in epoch_types
        }

        # Convert to DataFrame for plotting
        df = pd.DataFrame(normalized_counts).T.rename(index=epoch_label_map)  # Rows as epoch types, columns as band types

        # Perform Chi-Square test
        raw_df = pd.DataFrame(filtered_counts).T.rename(index=epoch_label_map)
        chi2, p, _, _ = chi2_contingency(raw_df)
        significance_label_text = significance_label(p)

        # Plot
        filtered_band_types_legend = [band.replace('band_', '').replace('_', '-') + ' Hz' for band in filtered_band_types]
        ax = df.plot(kind='bar', stacked=True, figsize=(12, 10))
        plt.title(f'Dominant Band Distribution by Condition Across Subjects - {cond} {significance_label_text}')
        plt.ylabel('Count of Dominant Bands')
        plt.xlabel('Epoch Type')
        plt.legend(labels=filtered_band_types_legend, title="Band Type", loc='center left',  bbox_to_anchor=(1.0, 0.5))
        ax.set_xticklabels(ax.get_xticklabels(), rotation='horizontal')
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space on the right for the legend
        # Save and show the plot
        plt.savefig(fig_path / f'{cond}_group_dominant_band_distribution.png')
        plt.close()

def add_bootstrapped_ci(data, group_col, value_col, ax):
    """
    Adds bootstrapped confidence intervals to violin plots.
    """
    groups = data[group_col].unique()
    for group in groups:
        group_data = data[data[group_col] == group][value_col]
        bootstrapped_means = [np.mean(np.random.choice(group_data, size=len(group_data), replace=True)) for _ in range(1000)]
        ci_lower, ci_upper = np.percentile(bootstrapped_means, [2.5, 97.5]) # getting the margin error of the CI, lower and upper lims
        x_pos = list(groups).index(group)
        ax.errorbar(x_pos, np.mean(group_data), yerr=[[np.mean(group_data) - ci_lower], [ci_upper - np.mean(group_data)]],
                    fmt='o', color=(1.0, 0.8509803921568627, 0.1843137254901961), capsize=5)

def plot_aggregated_avg_power_bar(all_results, fig_path):
    fig_path = Path(fig_path)
    fig_path.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists

    for condition in conditions_list:  # Iterate over each condition (e.g., a1, a2, e1, e2)
        # Initialize lists to store overall average power values across subjects
        avg_powers = {epoch_type: [] for epoch_type in epoch_types}

        # Collect individual epoch power values for each epoch type across subjects
        for epoch_type in epoch_types:
            avg_powers[epoch_type].extend(
                [epoch['overall_avg_power'] for subject_data in all_results[epoch_type][condition] for epoch in subject_data]
            )

        target_powers = avg_powers['Target']
        distractor_powers = avg_powers['Distractor']
        non_target_powers = avg_powers['Non-Target']
        power_df = pd.DataFrame({
            'avg_power': target_powers + distractor_powers + non_target_powers,
            'epoch_type': (['Target'] * len(target_powers)) +
                          (['Distractor'] * len(distractor_powers)) +
                          (['Non-Target'] * len(non_target_powers))
        })


        log_target_avg_powers = [np.log1p(value) for value in target_powers]
        log_distractor_avg_powers = [np.log1p(value) for value in distractor_powers]
        log_non_target_avg_powers = [np.log1p(value) for value in non_target_powers]
        all_powers = log_target_avg_powers + log_distractor_avg_powers + log_non_target_avg_powers
        max_power = np.max(all_powers)
        min_power = np.min(all_powers)
        def normalize_values(values, min_power, max_power):
            return [(val - min_power) / (max_power - min_power) for val in values]
        # normalize now the vals:
        normalized_target_avg_powers = normalize_values([value for value in log_target_avg_powers], min_power, max_power)
        normalized_distractor_avg_powers = normalize_values([value for value in log_distractor_avg_powers], min_power, max_power)
        normalized_non_target_avg_powers = normalize_values([value for value in log_non_target_avg_powers], min_power, max_power)

        # **Filter outliers**
        threshold = 1
        normalized_target_avg_powers = [val for val in normalized_target_avg_powers if val <= threshold]
        normalized_distractor_avg_powers = [val for val in normalized_distractor_avg_powers if val <= threshold]
        normalized_non_target_avg_powers = [val for val in normalized_non_target_avg_powers if val <= threshold]

        # Gather data for bar plot (calculate mean power values per epoch type)
        epoch_types_labels = ['Target', 'Distractor', 'Non-Target']

        # Statistical Testing
        is_normal = all(shapiro(avg_list)[1] > 0.05 for avg_list in avg_powers.values())
        significance_labels = {}
        if is_normal:
            # Perform One-way ANOVA on individual values
            anova_p_val = f_oneway(*avg_powers.values()).pvalue
            if anova_p_val < 0.05:
                df = pd.DataFrame({
                    'log_avg_power': log_target_avg_powers + log_distractor_avg_powers + log_non_target_avg_powers,
                    'epoch_type': (['Target'] * len(log_target_avg_powers)) +
                                  (['Distractor'] * len(log_distractor_avg_powers)) +
                                  (['Non-Target'] * len(log_non_target_avg_powers))
                })
                normalized_df = pd.DataFrame({
                    'normalized_avg_power': normalized_target_avg_powers + normalized_distractor_avg_powers +
                                            normalized_non_target_avg_powers,
                    'epoch_type': (['Target'] * len(normalized_target_avg_powers)) +
                                  (['Distractor'] * len(normalized_distractor_avg_powers)) +
                                  (['Non-Target'] * len(normalized_non_target_avg_powers))
                })
                posthoc = sp.posthoc_ttest(df, val_col='log_avg_power', group_col='epoch_type', p_adjust='bonferroni')
                significance_labels = {
                    ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                    ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
                    ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target'])
                }
        else:
            kruskal_p_val = kruskal(*avg_powers.values()).pvalue
            if kruskal_p_val < 0.05:
                df = pd.DataFrame({
                    'log_avg_power': log_target_avg_powers + log_distractor_avg_powers + log_non_target_avg_powers,
                    'epoch_type': (['Target'] * len(log_target_avg_powers)) +
                                  (['Distractor'] * len(log_distractor_avg_powers)) +
                                  (['Non-Target'] * len(log_non_target_avg_powers))
                })
                normalized_df = pd.DataFrame({
                    'normalized_avg_power': normalized_target_avg_powers + normalized_distractor_avg_powers +
                                            normalized_non_target_avg_powers,
                    'epoch_type': (['Target'] * len(normalized_target_avg_powers)) +
                                  (['Distractor'] * len(normalized_distractor_avg_powers)) +
                                  (['Non-Target'] * len(normalized_non_target_avg_powers))})
                posthoc = sp.posthoc_dunn(df, val_col='log_avg_power', group_col='epoch_type', p_adjust='bonferroni')
                significance_labels = {
                    ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                    ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
                    ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target'])
                }

        # Plot
        plt.figure(figsize=(12, 10))
        colors = ['darkviolet','gold', 'royalblue']
        ax = sns.violinplot(data=normalized_df, x='epoch_type', y='normalized_avg_power', hue='epoch_type',
                            palette=colors, legend=False)
        # Optionally add a strip plot to show individual data points
        # sns.stripplot(data=power_df, x="epoch_type", y="avg_power", color="black", alpha=0.5,
        #               jitter=False)
        add_bootstrapped_ci(normalized_df, 'epoch_type', 'normalized_avg_power', ax)
        # plt.ylim(None, 0.2)
        plt.legend(title=f'Sample Size: {len(normalized_target_avg_powers)}', loc='upper right')
        plt.title(f"{condition} Total Average Power by Epoch Type")
        plt.xlabel("Epoch Type")
        plt.ylabel("Overall Average Power (normalized)")


        # Add significance labels
        y_max = normalized_df['normalized_avg_power'].max()
        y_offset = y_max * 0.01  # Offset for significance text above bars
        pairs = [('Target', 'Distractor'), ('Target', 'Non-Target'), ('Distractor', 'Non-Target')]
        for (group1, group2) in pairs:
            if group1 in epoch_types_labels and group2 in epoch_types_labels:
                x1, x2 = epoch_types_labels.index(group1), epoch_types_labels.index(group2)
                y = y_max + y_offset
                label = significance_labels.get((group1, group2), "ns")
                if label != "ns":
                    plt.plot([x1, x2], [y, y], color='black', linestyle='solid')
                    plt.text((x1 + x2) / 2, y + y_offset * 1.1, label, ha='center', va='bottom', fontsize=12)
                    y_max += y_offset * 1.5

        plt.savefig(fig_path / f'{condition}_group_avg_power.png')
        plt.close()



import matplotlib.pyplot as plt
import numpy as np

def plot_all_tfa_heatmaps(avg_tfa_results, tfa_categories, conditions_list, fig_path):
    # Determine grid size for subplots
    n_rows = len(tfa_categories)
    n_cols = len(conditions_list)

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3), constrained_layout=True)

    # Flatten axes for simpler indexing (if there's only one row or column)
    axes = np.atleast_2d(axes)

    # Loop through each category and condition
    for i, category in enumerate(tfa_categories):
        for j, condition in enumerate(conditions_list):
            avg_tfr = avg_tfa_results.get(category, {}).get(condition, None)

            if avg_tfr is not None:
                # Extract power data from AverageTFR object
                power_data = avg_tfr.data[0]  # Assuming single-channel data, use [0] to select the channel
                times = avg_tfr.times
                freqs = avg_tfr.freqs

                # Plot the TFA heatmap on the current axis
                im = axes[i, j].imshow(power_data, aspect='auto', cmap='viridis',
                                       extent=[times[0], times[-1], freqs[0], freqs[-1]], origin='lower', vmin=-0.4, vmax=0.4
                                        ) #vmin=-0.2, vmax=0.2
                axes[i, j].set_title(f'{category} - {condition}', fontsize=10)
                axes[i, j].set_xlabel('Time (s)', fontsize=8)
                axes[i, j].set_ylabel('Frequency (Hz)', fontsize=8)
                axes[i, j].tick_params(axis='both', which='major', labelsize=7)

            else:
                # If no data is available, hide the subplot
                axes[i, j].axis('off')

    # Add a single color bar for the entire figure
    cbar = fig.colorbar(im, ax=axes, orientation='vertical', shrink=0.8, label='Power')

    # Save and show the plot
    plt.savefig(fig_path / f'{conditions_input}all_tfa_heatmaps.png', dpi=300)
    plt.close()



def plot_aggregated_dominant_frequency_distributions(all_results):
    for cond in conditions_list:  # Iterate through each condition (e.g., a1, a2, e1, e2)
        aggregated_data = []

        # Collect frequency data from all epoch types for the current condition
        for epoch_type in epoch_types:
            if cond in all_results[epoch_type]:  # Ensure the condition exists in each epoch type
                # Gather dominant frequencies from each subject and epoch
                dominant_freqs = [
                    epoch['dominant_freq']
                    for subject_data in all_results[epoch_type][cond]
                    for epoch in subject_data
                ]
                print(dominant_freqs)
                aggregated_data.extend([(epoch_type, freq) for freq in dominant_freqs])

        # Create DataFrame for plotting and statistical analysis
        df = pd.DataFrame(aggregated_data, columns=['Condition', 'Frequency'])

        # Perform statistical testing
        target_freqs = df[df['Condition'] == 'Target']['Frequency']
        distractor_freqs = df[df['Condition'] == 'Distractor']['Frequency']
        non_target_freqs = df[df['Condition'] == 'Non-Target']['Frequency']

        # Normality check
        is_normal = all(shapiro(freq)[1] > 0.05 for freq in [target_freqs, distractor_freqs, non_target_freqs])

        significance_labels = {}
        if is_normal:
            # One-way ANOVA
            anova_p_val = f_oneway(target_freqs, distractor_freqs, non_target_freqs).pvalue
            if anova_p_val < 0.05:
                posthoc = sp.posthoc_ttest(df, val_col='Frequency', group_col='Condition', p_adjust='bonferroni')
                significance_labels = {
                    ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                    ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
                    ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target'])
                }
        else:
            # Kruskal-Wallis
            kruskal_p_val = kruskal(target_freqs, distractor_freqs, non_target_freqs).pvalue
            if kruskal_p_val < 0.05:
                posthoc = sp.posthoc_dunn(df, val_col='Frequency', group_col='Condition', p_adjust='bonferroni')
                significance_labels = {
                    ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                    ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
                    ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target'])}

        # Plot
        plt.figure(figsize=(12, 10))
        ax = sns.violinplot(x='Condition', y='Frequency', data=df, palette=['blue', 'red', 'yellow'])
        plt.title(f'Dominant Frequency Distribution Across Subjects - {cond}')
        plt.ylabel("Dominant Frequency (Hz)")
        plt.legend(title=f'Sample Size: {len(target_freqs)}', loc='upper right')
        add_bootstrapped_ci(df, 'Condition', 'Frequency', ax)

        # Add significance labels
        y_max = df['Frequency'].max()
        y_offset = y_max * 0.01  # Offset to prevent overlap with plot elements
        pairs = [('Target', 'Distractor'), ('Target', 'Non-Target'), ('Distractor', 'Non-Target')]

        for (group1, group2) in pairs:
            x1, x2 = df['Condition'].unique().tolist().index(group1), df['Condition'].unique().tolist().index(group2)
            y = y_max + y_offset
            label = significance_labels.get((group1, group2), "ns")
            if label != "ns":
                plt.plot([x1, x2], [y, y], color='black', linestyle='solid')
                plt.text((x1 + x2) / 2, y + y_offset * 1.1, label, ha='center', va='bottom', fontsize=12)
                y_max += y_offset * 1.1  # Space out labels to prevent overlap

        plt.savefig(fig_path / f'{cond}_group_dominant_frequency_distributions.png')
        plt.close()

plot_all_tfa_heatmaps(avg_tfa_results, tfa_categories, conditions_list, fig_path)

all_results = load_all_subject_results(subject_results_dir, conditions_list)
plot_aggregated_dominant_band_distributions(all_results)
plot_aggregated_avg_power_bar(all_results, fig_path)
# plot_aggregated_dominant_frequency_counts(all_results, fig_path)
plot_aggregated_dominant_frequency_distributions(all_results)


'''
1.1 Fixed Effects
consistent across all subjects (no variability allowed):
Epoch Type (Target, Distractor, Non-Target) – Primary variable.

1.2 Random Effects
Subject ID: Each subject can have their unique baseline (random intercept).

1.3 Response Variable
    The dependent variables to analyze:
        Dominant Frequency (Hz) and Average Power (each separate)
1.4 Model Family
-  non-parametric data
- log-transformed power data
- skewed power and frequency data
- by independence of random effects, we assume that individual's EMG activity is independed from one another

Frequency ~ Epoch Type + (Epoch Type | Subject ID)
Why: we allow different individuals to have different patterns of change based on the fixed variable;
model allows the effect of Epoch Type to vary across subjects. 
For example, one subject might show a large difference in frequency between Target and Distractor, 
while another subject shows only a small difference
I guess a bayesian model would work, thing is that would require a shit-ton of work for now.
'''
# from statsmodels.formula.api import mixedlm
# import ast
# # prepare the data:
# target_a1_dfs = {}
# target_data = all_results['Target']
# target_a1 = target_data['a1']
# for sub_idx, epochs in enumerate(target_a1):  # `sub_idx` is the subject index, and `epochs` is the list of dictionaries
#     # Convert the list of epoch dictionaries into a DataFrame
#     target_a1_dfs[sub_idx] = pd.DataFrame(epochs)
# target_a2 = target_data['a2']
# target_e1 = target_data['e1']
# target_e2 = target_data['e2']
# # Fit the model
# model = mixedlm(
#     "Frequency ~ Epoch_Type",  # Fixed effects
#     data=df,                   # DataFrame with combined data
#     groups=df["Subject_ID"],   # Random effect: Subject ID
# )
# result = model.fit()
#
# # Summary of the results
# print(result.summary())
#
# sns.histplot(result.resid, kde=True)
# plt.title("Residuals Distribution")
#
# print(result.random_effects)
#
# print(result.params)
#
# # if Epoch Type is significant, test for pair-wise differences:
# from statsmodels.stats.multicomp import pairwise_tukeyhsd
#
# # for normal data:
# posthoc = pairwise_tukeyhsd(df['Frequency'], df['Epoch_Type'])
# print(posthoc.summary())
#
# # for non-parametric data: Dunn Posthoc
#
#
#
# sns.boxplot(data=df, x="Epoch_Type", y="Frequency", palette="Set2")
# plt.title("Dominant Frequency Distribution by Epoch Type")


# Plot aggregated dominant band distributions for each condition

# def plot_aggregated_dominant_frequency_counts(all_results, fig_path):
#     fig_path = Path(fig_path)
#     fig_path.mkdir(parents=True, exist_ok=True)  # Ensure the output directory exists
#
#     for cond in conditions_list:  # Iterate through each condition (e.g., a1, a2, e1, e2)
#         aggregated_data = []
#
#         # Collect frequency data from all epoch types for the current condition
#         for epoch_type in epoch_types:
#             if cond in all_results[epoch_type]:  # Ensure the condition exists in each epoch type
#                 # Gather dominant frequencies from each subject and epoch
#                 dominant_freqs = [
#                     epoch['dominant_freq']
#                     for subject_data in all_results[epoch_type][cond]
#                     for epoch in subject_data
#                 ]
#                 aggregated_data.extend([(epoch_type, freq) for freq in dominant_freqs])
#
#         # Create DataFrame for counts
#         df = pd.DataFrame(aggregated_data, columns=['Epoch Type', 'Frequency'])
#
#         # Count frequencies for each epoch type
#         counts = df.groupby(['Epoch Type', 'Frequency']).size().reset_index(name='Count')
#
#         # Pivot table for plotting
#         counts_pivot = counts.pivot(index='Frequency', columns='Epoch Type', values='Count').fillna(0)
#
#         # Plot counts
#         plt.figure(figsize=(12, 8))
#         counts_pivot.plot(kind='bar', stacked=False, figsize=(12, 8),
#                           color=['darkviolet', 'gold', 'royalblue'])
#
#         # Set plot details
#         plt.title(f'Dominant Frequency Counts Across Epoch Types - {cond}')
#         plt.xlabel('Frequency (Hz)')
#         plt.ylabel('Count')
#         plt.legend(title='Epoch Type')
#         plt.xticks(ticks=np.arange(150), labels=range(1, 151, 1), fontsize=6, rotation=0)  # Rotate x-axis labels for better readability
#         plt.grid(axis='y', alpha=0.3)
#
#         # Save the plot
#         plt.tight_layout()
#         plt.savefig(fig_path / f'{cond}_dominant_frequency_counts.png')
#         plt.close()


# Plot aggregated dominant frequency distributions for each condition