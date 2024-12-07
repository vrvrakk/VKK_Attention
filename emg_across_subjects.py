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

import glob

# Directory containing saved subject results
default_dir = Path.cwd()
subject_results_dir = default_dir / 'data'/ 'emg'/ 'subject_results'
fig_path = subject_results_dir / 'figures'
os.makedirs(fig_path, exist_ok=True)
sub_input = input('Give subjects:')
sub_list = [sub.strip() for sub in sub_input.split(',')]
# Prompt user for conditions and convert input into a list
conditions_input = input("Enter conditions separated by commas (e.g., 'a1, e1'): ")
conditions_list = [condition.strip() for condition in conditions_input.split(',')]

epoch_types = ['target', 'distractor', 'non_target']
tfa_categories = ['Target', 'Distractor', 'Non-Target']

# Initialize an empty dictionary to hold averaged results for each category and condition
avg_tfa_results = {category: {condition: None for condition in conditions_list} for category in tfa_categories}

# Load and aggregate TFA results for each subject
for sub in sub_list:
    with open(subject_results_dir / f'tfa_dict_{sub}.pkl', 'rb') as f:
        tfa_results = pickle.load(f)

        for category in tfa_categories:
            for condition in conditions_list:
                if condition in tfa_results[category]:
                    power_data = tfa_results[category][condition]

                    # Initialize or add to the running sum for averaging
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

# Plot the averaged TFA heatmaps
for category in tfa_categories:
    for condition in conditions_list:
        avg_tfr = avg_tfa_results[category][condition]

        if avg_tfr is not None:
            # Extract power data from AverageTFR object
            power_data = avg_tfr.data[0]  # Assuming single-channel data, use [0] to select the channel
            times = avg_tfr.times
            freqs = avg_tfr.freqs

            # Plot the TFA heatmap
            plt.figure(figsize=(8, 6))
            plt.imshow(power_data, aspect='auto', cmap='viridis',
                       extent=[times[0], times[-1], freqs[0], freqs[-1]], origin='lower') # vmin=-1, vmax=1
            plt.colorbar(label='Power')
            plt.title(f'TFA Heatmap - {category} - {condition}')
            plt.xlabel('Time (s)')
            plt.ylabel('Frequency (Hz)')

            # Save each plot separately
            plt.savefig(fig_path / f'{category}_{condition}_heatmap.png')
            plt.close()  # Close the figure after saving to avoid memory issues

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


# Plot aggregated dominant frequency distributions for each condition
# Plot aggregated dominant frequency distributions for each condition
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
                aggregated_data.extend([(epoch_type.capitalize(), freq) for freq in dominant_freqs])

        # Create DataFrame for plotting and statistical analysis
        df = pd.DataFrame(aggregated_data, columns=['Condition', 'Frequency'])

        # Perform statistical testing
        target_freqs = df[df['Condition'] == 'Target']['Frequency']
        distractor_freqs = df[df['Condition'] == 'Distractor']['Frequency']
        non_target_freqs = df[df['Condition'] == 'Non_target']['Frequency']

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
                    ('Target', 'Non_target'): significance_label(posthoc.loc['Target', 'Non_target']),
                    ('Distractor', 'Non_target'): significance_label(posthoc.loc['Distractor', 'Non_target'])
                }
        else:
            # Kruskal-Wallis
            kruskal_p_val = kruskal(target_freqs, distractor_freqs, non_target_freqs).pvalue
            if kruskal_p_val < 0.05:
                posthoc = sp.posthoc_dunn(df, val_col='Frequency', group_col='Condition', p_adjust='bonferroni')
                significance_labels = {
                    ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                    ('Target', 'Non_target'): significance_label(posthoc.loc['Target', 'Non_target']),
                    ('Distractor', 'Non_target'): significance_label(posthoc.loc['Distractor', 'Non_target'])
                }

        # Plot
        plt.figure(figsize=(12, 10))
        sns.violinplot(x='Condition', y='Frequency', data=df, palette=['blue', 'orange', 'green'])
        plt.title(f'Dominant Frequency Distribution Across Subjects - {cond}')
        plt.ylabel("Dominant Frequency (Hz)")

        # Add significance labels
        y_max = df['Frequency'].max()
        y_offset = y_max * 0.01  # Offset to prevent overlap with plot elements
        pairs = [('Target', 'Distractor'), ('Target', 'Non_target'), ('Distractor', 'Non_target')]

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
# Plot aggregated dominant band distributions for each condition


def plot_aggregated_dominant_band_distributions(all_results):
    band_types = ['low_band', 'mid_band', 'high_band']

    epoch_label_map = {
        'target': 'Target',
        'distractor': 'Distractor',
        'non_target': 'Non-Target'
    }
    for cond in conditions_list:  # Iterate through each condition
        # Initialize aggregated counts for each epoch type and band
        aggregated_counts = {epoch_type: Counter({band: 0 for band in band_types}) for epoch_type in epoch_types}

        # Aggregate counts of dominant bands for each condition
        for epoch_type in epoch_types:
            dominant_bands = [epoch['dominant_band'] for subject_data in all_results[epoch_type][cond] for epoch in subject_data]
            for band in band_types:
                aggregated_counts[epoch_type][band] += dominant_bands.count(band)

        # Convert to DataFrame for plotting
        df = pd.DataFrame(aggregated_counts).T.rename(index=epoch_label_map)  # Rows as epoch types, columns as band types

        # Perform Chi-Square test
        chi2, p, _, _ = chi2_contingency(df)
        significance_label_text = significance_label(p)

        # Plot
        ax = df.plot(kind='bar', stacked=True, color=['skyblue', 'salmon', 'lightgreen'], figsize=(10, 8))
        plt.title(f'Dominant Band Distribution by Condition Across Subjects - {cond} {significance_label_text}')
        plt.ylabel('Count of Dominant Bands')
        plt.xlabel('Epoch Type')
        plt.legend(title="Band Type", loc='upper right')
        ax.set_xticklabels(ax.get_xticklabels(), rotation='horizontal')

        # Save and show the plot
        plt.savefig(fig_path / f'{cond}_group_dominant_band_distribution.png')
        plt.close()


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

        # Gather data for bar plot (calculate mean power values per epoch type)
        epoch_types_labels = ['Target', 'Distractor', 'Non-Target']
        avg_values = [np.mean(avg_powers[et]) for et in epoch_types]
        unit_label = 'Overall Average Power (W)'

        # Statistical Testing
        is_normal = all(shapiro(avg_list)[1] > 0.05 for avg_list in avg_powers.values())
        significance_labels = {}
        if is_normal:
            # Perform One-way ANOVA on individual values
            anova_p_val = f_oneway(*avg_powers.values()).pvalue
            if anova_p_val < 0.05:
                df = pd.DataFrame({
                    'overall_avg_power': sum(avg_powers.values(), []),
                    'epoch_type': (['Target'] * len(avg_powers['target'])) +
                                  (['Distractor'] * len(avg_powers['distractor'])) +
                                  (['Non-Target'] * len(avg_powers['non_target']))
                })
                posthoc = sp.posthoc_ttest(df, val_col='overall_avg_power', group_col='epoch_type', p_adjust='bonferroni')
                significance_labels = {
                    ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                    ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
                    ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target'])
                }
        else:
            kruskal_p_val = kruskal(*avg_powers.values()).pvalue
            if kruskal_p_val < 0.05:
                df = pd.DataFrame({
                    'overall_avg_power': sum(avg_powers.values(), []),
                    'epoch_type': (['Target'] * len(avg_powers['target'])) +
                                  (['Distractor'] * len(avg_powers['distractor'])) +
                                  (['Non-Target'] * len(avg_powers['non_target']))
                })
                posthoc = sp.posthoc_dunn(df, val_col='overall_avg_power', group_col='epoch_type', p_adjust='bonferroni')
                significance_labels = {
                    ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                    ('Target', 'Non-Target'): significance_label(posthoc.loc['Target', 'Non-Target']),
                    ('Distractor', 'Non-Target'): significance_label(posthoc.loc['Distractor', 'Non-Target'])
                }

        # Plot
        plt.figure(figsize=(12, 10))
        bars = plt.bar(epoch_types_labels, avg_values, color=['blue', 'orange', 'green'])
        plt.ylabel(unit_label)
        plt.title(f'{condition} Comparison of Overall Average Power by Epoch Type')

        # Add significance labels
        y_max = max(avg_values)
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


def plot_aggregated_LTER_distribution(all_results):
    for cond in conditions_list:
        aggregated_data = []

        for epoch_type in epoch_types:
            LTER_vals = [epoch['LTER'] for subject_data in all_results[epoch_type][cond] for epoch in subject_data]
            aggregated_data.extend([(epoch_type.capitalize(), LTER) for LTER in LTER_vals])

        df = pd.DataFrame(aggregated_data, columns=['Condition', 'LTER'])

        # Statistical Testing
        target_LTER = df[df['Condition'] == 'Target']['LTER']
        distractor_LTER = df[df['Condition'] == 'Distractor']['LTER']
        non_target_LTER = df[df['Condition'] == 'Non_target']['LTER']

        is_normal = all(shapiro(lter)[1] > 0.05 for lter in [target_LTER, distractor_LTER, non_target_LTER])
        significance_labels = {}
        if is_normal:
            anova_p_val = f_oneway(target_LTER, distractor_LTER, non_target_LTER).pvalue
            if anova_p_val < 0.05:
                posthoc = sp.posthoc_ttest(df, val_col='LTER', group_col='Condition', p_adjust='bonferroni')
                significance_labels = {
                    ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                    ('Target', 'Non_target'): significance_label(posthoc.loc['Target', 'Non_target']),
                    ('Distractor', 'Non_target'): significance_label(posthoc.loc['Distractor', 'Non_target'])
                }
        else:
            kruskal_p_val = kruskal(target_LTER, distractor_LTER, non_target_LTER).pvalue
            if kruskal_p_val < 0.05:
                posthoc = sp.posthoc_dunn(df, val_col='LTER', group_col='Condition', p_adjust='bonferroni')
                significance_labels = {
                    ('Target', 'Distractor'): significance_label(posthoc.loc['Target', 'Distractor']),
                    ('Target', 'Non_target'): significance_label(posthoc.loc['Target', 'Non_target']),
                    ('Distractor', 'Non_target'): significance_label(posthoc.loc['Distractor', 'Non_target'])
                }

        # Plot
        plt.figure(figsize=(12, 10))
        sns.violinplot(x='Condition', y='LTER', data=df, palette=['blue', 'orange', 'green'])
        plt.title(f'LTER Distribution by Condition Across Subjects - {cond}')
        plt.ylabel('LTER')

        y_max = df['LTER'].max()
        y_offset = y_max * 0.01
        pairs = [('Target', 'Distractor'), ('Target', 'Non_target'), ('Distractor', 'Non_target')]
        for (group1, group2) in pairs:
            x1, x2 = df['Condition'].unique().tolist().index(group1), df['Condition'].unique().tolist().index(group2)
            y = y_max + y_offset
            label = significance_labels.get((group1, group2), "ns")
            if label != "ns":
                plt.plot([x1, x2], [y, y], color='black', linestyle='solid')
                plt.text((x1 + x2) / 2, y + y_offset * 1.1, label, ha='center', va='bottom', fontsize=12)
                y_max += y_offset * 1.2

        plt.savefig(fig_path / f'{cond}_group_LTER_distribution.png')
        plt.close()



import matplotlib.pyplot as plt
import numpy as np

def plot_all_tfa_heatmaps(avg_tfa_results, tfa_categories, conditions_list, fig_path):
    # Determine grid size for subplots
    n_rows = len(tfa_categories)
    n_cols = len(conditions_list)

    # Create a figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), constrained_layout=True)

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
                                       extent=[times[0], times[-1], freqs[0], freqs[-1]], origin='lower',
                                        ) # vmin=-1, vmax=1
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
    plt.savefig(fig_path / 'all_tfa_heatmaps.png', dpi=300)
    plt.close()

plot_all_tfa_heatmaps(avg_tfa_results, tfa_categories, conditions_list, fig_path)

all_results = load_all_subject_results(subject_results_dir, conditions_list)
plot_aggregated_dominant_frequency_distributions(all_results)
plot_aggregated_dominant_band_distributions(all_results)
plot_aggregated_avg_power_bar(all_results, fig_path)
plot_aggregated_LTER_distribution(all_results)
