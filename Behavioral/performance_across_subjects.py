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
                if '_valid_target_responses' in file.name:
                    aggregated_dfs[condition]['valid_target_responses'].append(df)
                elif 'invalid_target_responses' in file.name:
                    aggregated_dfs[condition]['invalid_target_responses'].append(df)
                elif '_valid_distractor_responses' in file.name:
                    aggregated_dfs[condition]['valid_distractor_responses'].append(df)
                elif 'invalid_distractor_responses' in file.name:
                    aggregated_dfs[condition]['invalid_distractor_responses'].append(df)
                elif 'all_target_stimuli' in file.name:
                    aggregated_dfs[condition]['target_stimuli'].append(df)
                elif 'all_distractor_stimuli' in file.name:
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
            'Target Hit-rate': hit_rate,
            'Distractor Hit-rate': distractor_hit_rate,
            'Missed Targets': miss_rate,
            'Invalid Target Responses': target_invalid_response_rate,
            'Overall Invalid Responses': overall_invalid_response_rate
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
    sns.barplot(data=metrics_long, x='Condition', y='Rate', hue='Metric', palette=['darkviolet', 'gold', 'royalblue', 'forestgreen', 'orange'])
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


sub_list = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub08',
            'sub10', 'sub11', 'sub13', 'sub14', 'sub15', 'sub17',\
            'sub18', 'sub19', 'sub20', 'sub21', 'sub22', 'sub23', 'sub24',
            'sub25', 'sub26', 'sub27', 'sub28', 'sub29']
def main(sub_list):
    # Define the main directory, subjects, and conditions
    default_dir = Path.cwd()
    sub_list = sub_list

    conditions = input('Enter conditions to analyze (a1, a2, e1, e2), separated by commas: ')
    condition_list = [cond.strip() for cond in conditions.split(',')]

    # Load and aggregate DataFrames across all subjects, separated by condition
    aggregated_dfs = load_and_aggregate_all_dfs(sub_list, condition_list, default_dir)
    azimuth_dfs = {**aggregated_dfs['a1'], **aggregated_dfs['a2']}
    elevation_dfs = {**aggregated_dfs['e1'], **aggregated_dfs['e2']}
    concatenated_dfs = {'azimuth': azimuth_dfs, 'elevation': elevation_dfs}
    # Calculate performance metrics for each condition
    performance_df = calculate_condition_performance(concatenated_dfs)

    # Print the performance metrics
    print("Performance Metrics by Condition:")
    print(performance_df)

    # Plot and save the performance metrics
    save_dir = default_dir / 'data' / 'performance' / 'aggregated_results'
    save_dir.mkdir(parents=True, exist_ok=True)
    plot_condition_performance(performance_df, save_dir)



if __name__ == "__main__":
    main(sub_list=sub_list)
