import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import matplotlib
matplotlib.use('Agg')

def load_and_aggregate_all_dfs(sub_list, condition_list, default_dir):
    """Load and aggregate all DataFrames for each type across all subjects, grouped by condition."""
    # Initialize a dictionary to hold concatenated DataFrames for each type by condition
    aggregated_dfs = {condition: {
        'valid_target_responses': [],
        'invalid_target_responses': [],
        'valid_distractor_responses': [],
        'invalid_distractor_responses': [],
        'target_stimuli': [],
        'distractor_stimuli': [],
        'missed_target_stimuli': [],
        'missed_distractor_stimuli': [],
        'non_target_invalid_responses': []
    } for condition in condition_list}

    for sub in sub_list:
        df_path = default_dir / 'data' / 'performance' / f'{sub}' / 'tables'

        for condition in condition_list:
            for file in df_path.glob(f"df_{sub}_{condition}_*_*.csv"):
                df = pd.read_csv(file)

                # Append DataFrame to the appropriate list in `aggregated_dfs[condition]` based on filename
                if 'valid_target_responses' in file.name:
                    aggregated_dfs[condition]['valid_target_responses'].append(df)
                elif 'invalid_target_responses' in file.name:
                    aggregated_dfs[condition]['invalid_target_responses'].append(df)
                elif 'valid_distractor_responses' in file.name:
                    aggregated_dfs[condition]['valid_distractor_responses'].append(df)
                elif 'invalid_distractor_responses' in file.name:
                    aggregated_dfs[condition]['invalid_distractor_responses'].append(df)
                elif 'target_stimuli' in file.name:
                    aggregated_dfs[condition]['target_stimuli'].append(df)
                elif 'distractor_stimuli' in file.name:
                    aggregated_dfs[condition]['distractor_stimuli'].append(df)
                elif 'missed_target_stimuli' in file.name:
                    aggregated_dfs[condition]['missed_target_stimuli'].append(df)
                elif 'missed_distractor_stimuli' in file.name:
                    aggregated_dfs[condition]['missed_distractor_stimuli'].append(df)
                elif 'non_target_invalid_responses' in file.name:
                    aggregated_dfs[condition]['non_target_invalid_responses'].append(df)

    # Concatenate lists into single DataFrames for each type within each condition
    for condition, data_types in aggregated_dfs.items():
        for key in data_types.keys():
            aggregated_dfs[condition][key] = pd.concat(data_types[key], ignore_index=True) if data_types[
                key] else pd.DataFrame()

    return aggregated_dfs


def calculate_condition_performance(aggregated_dfs):
    """Calculate performance metrics for each condition based on aggregated DataFrames."""
    condition_performance = {}

    for condition, dfs in aggregated_dfs.items():
        # Total counts for stimuli
        total_target_stimuli = len(dfs['target_stimuli'])
        total_distractor_stimuli = len(dfs['distractor_stimuli'])
        total_stimuli = total_target_stimuli + total_distractor_stimuli

        # Valid and invalid responses counts
        valid_target_responses = len(dfs['valid_target_responses'])
        invalid_target_responses = len(dfs['invalid_target_responses'])
        valid_distractor_responses = len(dfs['valid_distractor_responses'])
        invalid_distractor_responses = len(dfs['invalid_distractor_responses'])
        non_target_invalid_responses = len(dfs['non_target_invalid_responses'])

        # Missed stimuli
        missed_target_stimuli = len(dfs['missed_target_stimuli'])
        missed_distractor_stimuli = len(dfs['missed_distractor_stimuli'])

        # Calculate rates
        hit_rate = (valid_target_responses / total_target_stimuli) * 100 if total_target_stimuli > 0 else 0
        distractor_hit_rate = (
                                          valid_distractor_responses / total_distractor_stimuli) * 100 if total_distractor_stimuli > 0 else 0
        miss_rate = (missed_target_stimuli / total_target_stimuli) * 100 if total_target_stimuli > 0 else 0
        target_invalid_response_rate = (
                                                   invalid_target_responses / total_target_stimuli) * 100 if total_target_stimuli > 0 else 0
        overall_invalid_response_rate = (
                                                (
                                                            invalid_target_responses + invalid_distractor_responses + non_target_invalid_responses) / total_stimuli
                                        ) * 100 if total_stimuli > 0 else 0

        # Store performance metrics for the condition
        condition_performance[condition] = {
            'hit_rate': hit_rate,
            'distractor_hit_rate': distractor_hit_rate,
            'miss_rate': miss_rate,
            'target_invalid_response_rate': target_invalid_response_rate,
            'overall_invalid_response_rate': overall_invalid_response_rate
        }

    return pd.DataFrame(condition_performance).T  # Transpose to make conditions rows


def plot_condition_performance(performance_df, save_dir):
    """Plot the performance metrics per condition as a bar chart."""
    # Melt DataFrame to long format for easy plotting with seaborn
    metrics_long = performance_df.reset_index().melt(id_vars="index", var_name="Metric", value_name="Rate")
    metrics_long.rename(columns={"index": "Condition"}, inplace=True)

    # Set up the plot
    sns.set(style="whitegrid")
    plt.figure(figsize=(16, 4))

    # Bar plot for performance metrics per condition
    sns.barplot(data=metrics_long, x='Condition', y='Rate', hue='Metric', palette="viridis")
    plt.title("Performance Metrics by Condition")
    plt.ylabel("Rate (%)")
    plt.xlabel("Condition")
    plt.ylim(0, 100)
    plt.legend(title='Metric')

    # Save the plot
    save_path = save_dir / "performance_metrics_by_condition.png"
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Performance metrics plot saved to {save_path}")

def plot_rt_distributions(aggregated_dfs, save_dir):
    """Plot the distribution of RTs from valid target responses across all subjects for each condition."""
    for condition, data in aggregated_dfs.items():
        # Extract the 'Time Difference' column from all valid target response DataFrames
        if 'valid_target_responses' in data and not data['valid_target_responses'].empty:
            rt_values = data['valid_target_responses']['Time Difference'].dropna().values
            rt_values = rt_values[(rt_values > 0.2) & (rt_values < 0.9)]

            # Check if there are enough RT values to plot
            if len(rt_values) < 5:
                print(f"Insufficient RT data to plot for condition {condition}.")
                continue

            # Plot the histogram
            plt.figure(figsize=(10, 6))
            sns.histplot(rt_values, bins=int(len(rt_values) / 5), kde=True, color='skyblue', edgecolor='black')

            # Calculate and plot mean and median
            mean_rt = np.mean(rt_values)
            median_rt = np.median(rt_values)
            plt.axvline(mean_rt, color='red', linestyle='dashed', linewidth=1, label=f'Mean: {mean_rt:.3f}s')
            plt.axvline(median_rt, color='green', linestyle='dashed', linewidth=1, label=f'Median: {median_rt:.3f}s')

            # Labels and title
            plt.xlabel('Reaction Time (seconds)')
            plt.ylabel('Frequency')
            plt.title(f'Aggregated RTs for Valid Target Responses - Condition {condition}')

            # Display min and max RT values
            plt.gca().text(0.98, 0.98, f'Min: {min(rt_values):.3f}s\nMax: {max(rt_values):.3f}s', fontsize=10,
                           verticalalignment='top', horizontalalignment='right', transform=plt.gca().transAxes,
                           bbox=dict(facecolor='white', alpha=0.6))

            plt.legend(loc='upper left')
            plt.savefig(save_dir / f"Aggregated_RTs_{condition}_valid_target_responses.png")
            plt.close()
            print(f"Saved RT distribution plot for condition {condition}.")
        else:
            print(f"No valid target responses data available for condition {condition}.")


def main():
    # Define the main directory, subjects, and conditions
    default_dir = Path.cwd()
    subs = input('Enter subject numbers separated by commas: ')
    sub_list = [sub.strip() for sub in subs.split(',')]

    conditions = input('Enter conditions to analyze (a1, a2, e1, e2), separated by commas: ')
    condition_list = [cond.strip() for cond in conditions.split(',')]

    # Load and aggregate DataFrames across all subjects, separated by condition
    aggregated_dfs = load_and_aggregate_all_dfs(sub_list, condition_list, default_dir)

    # Calculate performance metrics for each condition
    performance_df = calculate_condition_performance(aggregated_dfs)

    # Print the performance metrics
    print("Performance Metrics by Condition:")
    print(performance_df)

    # Plot and save the performance metrics
    save_dir = default_dir / 'data' / 'performance' / 'aggregated_results'
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_condition_performance(performance_df, save_dir)
    plot_rt_distributions(aggregated_dfs, save_dir)

if __name__ == "__main__":
    main()
