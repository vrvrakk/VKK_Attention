from pathlib import Path
import mne
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import pandas as pd
from scipy.stats import kruskal, shapiro, levene, f_oneway, tukey_hsd
import statsmodels.api as sm
import scikit_posthocs as sp
from itertools import combinations
import seaborn as sns

sub_list = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub08',
            'sub20', 'sub21', 'sub10', 'sub11','sub13', 'sub14', 'sub15','sub16',
            'sub17', 'sub18', 'sub19', 'sub22', 'sub23', 'sub24', 'sub25', 'sub26',
            'sub27', 'sub28', 'sub29']

epoch_types = ['combined_target_response_epochs', 'combined_distractor_no_response_epochs',
               'combined_non_target_target_epochs', 'combined_non_target_distractor_epochs']

epoch_categories = ['Target', 'Distractor', 'Non-Target Target', 'Non-Target Distractor']

conditions = ['a1', 'a2', 'e1', 'e2']

data_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/emg')
results_path = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/emg/subject_results/figures')


def get_epochs(condition=''):
    excluded_subs = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub06', 'sub08']
    target_epochs = {}
    distractor_epochs = {}
    non_target_target_epochs = {}
    non_target_distractor_epochs = {}
    if condition in ['e1', 'e2']:
        filtered_sub_list = [sub for sub in sub_list if sub not in excluded_subs]
    else:
        filtered_sub_list = sub_list
    for sub in filtered_sub_list:
        target_epochs[sub] = {}
        distractor_epochs[sub] = {}
        non_target_target_epochs[sub] = {}
        non_target_distractor_epochs[sub] = {}
        folderpath = data_path / sub / 'fif files' / 'combined_epochs'
        for file in folderpath.iterdir():
            if epoch_types[0] in file.name and condition in file.name:
                epoch = mne.read_epochs(file, preload=True)
                target_epochs[sub] = epoch
            elif epoch_types[1] in file.name and condition in file.name:
                epoch = mne.read_epochs(file, preload=True)
                distractor_epochs[sub] = epoch
            elif epoch_types[2] in file.name and condition in file.name:
                epoch = mne.read_epochs(file, preload=True)
                non_target_target_epochs[sub] = epoch
            elif epoch_types[3] in file.name and condition in file.name:
                non_target_distractor_epochs[sub] = epoch
    return target_epochs, distractor_epochs, non_target_target_epochs, non_target_distractor_epochs


a1_target_epochs, a1_distractor_epochs, a1_non_target_target_epochs, a1_non_target_distractor_epochs = get_epochs(condition=conditions[0])
a2_target_epochs, a2_distractor_epochs, a2_non_target_target_epochs, a2_non_target_distractor_epochs = get_epochs(condition=conditions[1])
e1_target_epochs, e1_distractor_epochs, e1_non_target_target_epochs, e1_non_target_distractor_epochs = get_epochs(condition=conditions[2])
e2_target_epochs, e2_distractor_epochs, e2_non_target_target_epochs, e2_non_target_distractor_epochs = get_epochs(condition=conditions[3])


# concatenate all epochs:
def concatenate_epochs(*epochs_dicts, plane='', epoch_name=''):
    all_epochs = []
    # Loop through each dictionary and extract the Epochs objects
    for epochs_dict in epochs_dicts:
        for epochs in epochs_dict.values():
            all_epochs.append(epochs)

    # Concatenate all epochs into a single Epochs object
    concatenated_epochs = mne.concatenate_epochs(all_epochs)
    plot = concatenated_epochs.plot_psd()
    plot.savefig(results_path / f'{plane}_{epoch_name}_psd.png')
    plt.close()
    return concatenated_epochs


# azimuth:
concatenated_epochs_azimuth_target = concatenate_epochs(a1_target_epochs, a2_target_epochs, plane='azimuth', epoch_name='target')
concatenated_epochs_azimuth_distractor = concatenate_epochs(a1_distractor_epochs, a2_distractor_epochs, plane='azimuth', epoch_name='distractor')
concatenated_epochs_azimuth_non_target_target = concatenate_epochs(a1_non_target_target_epochs, a2_non_target_target_epochs, plane='azimuth', epoch_name='non_target_target')
concatenated_epochs_azimuth_non_target_distractor = concatenate_epochs(a1_non_target_distractor_epochs, a2_non_target_distractor_epochs, plane='azimuth', epoch_name='non_target_distractor')

# elevation
concatenated_epochs_elevation_target = concatenate_epochs(e1_target_epochs, e2_target_epochs, plane='elevation', epoch_name='target')
concatenated_epochs_elevation_distractor = concatenate_epochs(e1_distractor_epochs, e2_distractor_epochs, plane='elevation', epoch_name='distractor')
concatenated_epochs_elevation_non_target_target = concatenate_epochs(e1_non_target_target_epochs, e2_non_target_target_epochs, plane='elevation', epoch_name='non_target_target')
concatenated_epochs_elevation_non_target_distractor = concatenate_epochs(e1_non_target_distractor_epochs, e2_non_target_distractor_epochs, plane='elevation', epoch_name='non_target_distractor')

# get minimum length of events from ALL concatenated epochs combined:
concatenated_epochs = [concatenated_epochs_azimuth_target,
                      concatenated_epochs_azimuth_distractor,
                      concatenated_epochs_azimuth_non_target_target,
                      concatenated_epochs_azimuth_non_target_distractor,
                      concatenated_epochs_elevation_target,
                      concatenated_epochs_elevation_distractor,
                      concatenated_epochs_elevation_non_target_target,
                      concatenated_epochs_elevation_non_target_distractor]

np.random.seed(42)
all_lens = []

# Calculate the minimum number of events across all provided epochs
for epochs in concatenated_epochs:
    length = len(epochs.events)
    all_lens.append(length)
min_length = min(all_lens)  # Find the minimum length

# make all concatenated epochs of same length:
def crop_lengths(min_length, concat_epochs):
    # Crop each epochs object to the minimum length
    cropped_epochs = concat_epochs[np.random.choice(len(concat_epochs), min_length, replace=False)]
    return cropped_epochs


# azimuth cropping:
concatenated_epochs_azimuth_target = crop_lengths(min_length, concatenated_epochs_azimuth_target)
concatenated_epochs_azimuth_distractor = crop_lengths(min_length, concatenated_epochs_azimuth_distractor)
concatenated_epochs_azimuth_non_target_target = crop_lengths(min_length, concatenated_epochs_azimuth_non_target_target)
concatenated_epochs_azimuth_non_target_distractor = crop_lengths(min_length, concatenated_epochs_azimuth_non_target_distractor)

concatenated_epochs_elevation_target = crop_lengths(min_length, concatenated_epochs_elevation_target)
concatenated_epochs_elevation_distractor = crop_lengths(min_length, concatenated_epochs_elevation_distractor)
concatenated_epochs_elevation_non_target_target = crop_lengths(min_length, concatenated_epochs_elevation_non_target_target)
concatenated_epochs_elevation_non_target_distractor = crop_lengths(min_length, concatenated_epochs_elevation_non_target_distractor)

# check non-target streams overlaps:

# plot heatmaps:
def plot_heatmaps(epochs, epoch_name='', plane=''):
    frequencies = np.logspace(np.log10(1), np.log10(150), num=150)  # Frequencies from 1 to 30 Hz
    n_cycles = np.minimum(frequencies / 2, 7)  # Number of cycles in Morlet wavelet (adapts to frequency)

    # Compute the TFR using Morlet wavelets
    tfa = mne.time_frequency.tfr_morlet(
        epochs,
        freqs=frequencies,
        n_cycles=n_cycles,
        return_itc=False,
        average=True,
        decim=1,
        n_jobs=1
    )

    # Apply baseline correction
    tfa.apply_baseline(baseline=(-0.2, 0.0), mode='logratio')

    # Plot and save the TFA heatmap
    fig = tfa.plot(
        mode='logratio',
        tmin=-0.2,
        tmax=0.9,
        fmin=1,
        fmax=150,
        vmin=-0.5,
        vmax=0.5,
        cmap='viridis',
        show=True,
        title=f'{epoch_name} Epochs TFA Heatmap ({plane})'
    )
    fig[0].savefig(results_path / f'tfa_{plane}_{epoch_name}.png', dpi=300)
    plt.close()
    return tfa


# azimuth TFA:
tfa_azimuth_target = plot_heatmaps(concatenated_epochs_azimuth_target, epoch_name='Target', plane='azimuth')
tfa_azimuth_distractor = plot_heatmaps(concatenated_epochs_azimuth_distractor, epoch_name='Distractor', plane='azimuth')
tfa_azimuth_non_target_target = plot_heatmaps(concatenated_epochs_azimuth_non_target_target, epoch_name='Non-Target Target', plane='azimuth')
tfa_azimuth_non_target_distractor = plot_heatmaps(concatenated_epochs_azimuth_non_target_distractor, epoch_name='Non-Target Distractor', plane='azimuth')


# elevation TFA:
tfa_elevation_target = plot_heatmaps(concatenated_epochs_elevation_target, epoch_name='Target', plane='elevation')
tfa_elevation_distractor = plot_heatmaps(concatenated_epochs_elevation_distractor, epoch_name='Distractor', plane='elevation')
tfa_elevation_non_target_target = plot_heatmaps(concatenated_epochs_elevation_non_target_target, epoch_name='Non-Target Target', plane='elevation')
tfa_elevation_non_target_distractor = plot_heatmaps(concatenated_epochs_elevation_non_target_distractor, epoch_name='Non-Target Distractor', plane='elevation')


def normality_test(epoch_categories, ax=0, data_list=[]):
    alpha = 0.05
    descriptive_statistics = {}
    mean_data_list = []
    for category, tfa_data in zip(epoch_categories, data_list):
        mean_data = np.mean(tfa_data.data, axis=ax)
        mean_data_list.append(mean_data)
        mean_value = np.mean(mean_data)
        median_value = np.median(mean_data)
        std_value = np.std(mean_data)
        shapiro_data = shapiro(mean_data.flatten())
        normality = shapiro_data.pvalue >= alpha
        descriptive_statistics[category] = {'mean': mean_value, 'median': median_value, 'std': std_value,
                                            'shapiro': shapiro_data.pvalue, 'normality': normality}
    mean_data_list = [data.flatten() for data in mean_data_list]
    variance = levene(*mean_data_list)
    return descriptive_statistics, variance


azimuth_data_list = [tfa_azimuth_target, tfa_azimuth_distractor, tfa_azimuth_non_target_target,
                     tfa_azimuth_non_target_distractor]

elevation_data_list = [tfa_elevation_target, tfa_elevation_distractor,
                       tfa_elevation_non_target_target, tfa_elevation_non_target_distractor]

azimuth_descriptive_stats, azimuth_variance = normality_test(epoch_categories, data_list=azimuth_data_list, ax=1)

elevation_descriptive_stats, elevation_variance = normality_test(epoch_categories, data_list=elevation_data_list, ax=1)


# verify distribution of data visually:
def q_q_plot(epoch_categories, data_list=[], plane='', ax=0):
    cols = 2
    rows = 2
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 12))
    axes = axes.flatten()  # Flatten the axes array for easy indexing
    for idx, tfa_data in enumerate(data_list):
        data = np.mean(tfa_data.data, axis=ax)
        sm.qqplot(data.flatten(), line='45', ax=axes[idx])
        # axes[idx]: refers to the specific subplot (axis)
        # uses the corresponding title from the epoch_categories list
        axes[idx].set_title(epoch_categories[idx] if idx < len(epoch_categories) else f'Plot {idx + 1}')
    # Adjust layout
    plt.tight_layout()
    plt.show()
    plt.savefig(results_path / f'q_q_plots_{plane}.png')


q_q_plot(epoch_categories, data_list=azimuth_data_list, plane='azimuth', ax=1)

q_q_plot(epoch_categories, data_list=elevation_data_list, plane='elevation', ax=1)

# overall effect size after kruskal:
def kruskal_effect_size(H, k, N):
    '''H: Kruskal-Wallis H-statistic.
    k: Number of groups.
    N: Total number of observations.
    eta = H-k+1 / N-k'''
    eta_squared = (H - k + 1) / (N - k)
    return eta_squared


# now pairwise size effect:
def cliffs_delta(x, y):
    n_x, n_y = len(x), len(y)
    greater = sum(1 for i in x for j in y if i > j)
    less = sum(1 for i in x for j in y if i < j)
    delta = (greater - less) / (n_x * n_y)
    return delta


def significance_testing(descriptive_statistics, data_list, epoch_categories):
    data_grouped = []
    normality = []
    for category, tfa_data in zip(epoch_categories, data_list):
        data = np.mean(tfa_data.data, axis=1).flatten()
        data_grouped.append(data)
    for category, (key, data_dict) in zip(epoch_categories, descriptive_statistics.items()):
        if data_dict['normality'] == False:
            normality.append(data_dict['normality'])
    data_dicts = {}  # Create a dictionary to store data with group labels
    for i, (category, data) in enumerate(zip(epoch_categories, data_grouped)):
        data_dicts[category] = data
    # Assuming labels is a list containing category labels (same order as epoch_categories)
    df = pd.DataFrame(data_dicts)  # Create a DataFrame from the dictionary
    df_long = pd.melt(df, var_name='category', value_name='data')
    if False in normality:
        print('At least one category has non-parametric data. Kruskal Wallis is used for significance testing...')
        significance_vals = kruskal(*data_grouped)
        if significance_vals.pvalue < 0.05:  # Check for significance
            print(f'Kruskal-Wallis test p-value: {significance_vals.pvalue}, There might be significant differences between groups.')
            print('Pairwise comparison using Dunn testing...')
            print('Getting overall size effect..')
            N = len(data_grouped[0]) + len(data_grouped[1]) + len(data_grouped[2] + len(data_grouped[3]))
            overall_size_effect = kruskal_effect_size(significance_vals.statistic, 4, N)
            print(f'Overall size effect (eta-squared): {overall_size_effect}')
            # Perform post-hoc test (Dunn's test) for pairwise comparisons
            posthoc_results = sp.posthoc_dunn(df_long, group_col='category', val_col='data')  # Assuming data_list contains data for each category
            print(f'Dunn results: {posthoc_results}')
            print('Getting pairwise size effect...')
            target_distractor_delta = cliffs_delta(data_grouped[0], data_grouped[1])
            target_non_target_delta = cliffs_delta(data_grouped[0], data_grouped[2])
            distractor_non_target_delta = cliffs_delta(data_grouped[1], data_grouped[3])
            non_targets_delta = cliffs_delta(data_grouped[2], data_grouped[3])
            target_non_target_distractor_delta = cliffs_delta(data_grouped[0], data_grouped[3])
            distractor_non_target_target_delta = cliffs_delta(data_grouped[1], data_grouped[2])
            pairwise_size_effects = pd.DataFrame({'target_distractor': [target_distractor_delta],
                                                  'target_non_target_target':[target_non_target_delta],
                                                  'distractor_non_target_distractor':[distractor_non_target_delta],
                                                  'non_targets': [non_targets_delta],
                                                  'target_non_target_distractor': [target_non_target_distractor_delta],
                                                  'distractor_non_target_target': [distractor_non_target_target_delta]})
            # Interpret Dunn's test results (p-values for pairwise comparisons)
        else:
            print(f'Kruskal-Wallis test p-value: {significance_vals.pvalue}, No significant differences between groups.')
    elif False not in normality:
        print(f'All data is parametric. Using one-way ANOVA for significance testing..')
        significance_vals = f_oneway(df_long)
        # Extract sum of squares and total sum of squares
        df = len(df_long.columns) - 1  # Degrees of freedom between groups
        total_n = sum([len(df_long[group]) for group in df_long.columns])  # Total sample size
        total_df = total_n - 1  # Total degrees of freedom

        ss_between = df * significance_vals.statistic * (total_n / (df + 1))
        ss_total = ss_between + (significance_vals.statistic * total_df / (df + 1))

        # Calculate eta-squared (η²)
        overall_size_effect = ss_between / ss_total
        if significance_vals.pvalue < 0.05:
            print("One-way ANOVA: Significant differences found between groups.")
            posthoc_results = tukey_hsd(df_long)
            print(f'Tukey results: {posthoc_results}')
            # Calculate pairwise effect sizes (Cohen's d)
            print("\nPairwise Effect Sizes (Cohen's d):")
            pairwise_size_effects = {}
            groups = df_long['group'].unique()
            for g1, g2 in combinations(groups, 2):
                # Extract values for each group
                group1 = df_long[df_long['group'] == g1]['value']
                group2 = df_long[df_long['group'] == g2]['value']

                # Calculate means and pooled standard deviation
                mean1, mean2 = np.mean(group1), np.mean(group2)
                std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
                n1, n2 = len(group1), len(group2)
                pooled_std = np.sqrt(((n1 - 1) * std1 ** 2 + (n2 - 1) * std2 ** 2) / (n1 + n2 - 2))

                # Calculate Cohen's d
                cohen_d = (mean1 - mean2) / pooled_std
                # Add result to the dictionary
                pairwise_size_effects[(g1, g2)] = {
                    'Cohen_d': cohen_d,
                    'Group1_mean': mean1,
                    'Group2_mean': mean2,
                    'Pooled_std': pooled_std,
                    'Group1_size': n1,
                    'Group2_size': n2
                }

        else:
            print("One-way ANOVA: No significant differences found between groups.")
    return significance_vals, posthoc_results, overall_size_effect, pairwise_size_effects


azimuth_significance_vals, azimuth_posthoc_results, azimuth_overall_size_effect, azimuth_pairwise_size_effects = significance_testing(azimuth_descriptive_stats, azimuth_data_list, epoch_categories)
elevation_significance_vals, elevation_posthoc_results, elevation_overall_size_effect, elevation_pairwise_size_effects = significance_testing(elevation_descriptive_stats, elevation_data_list, epoch_categories)

def plot_average_powers(epoch_categories, data_list, time_window=(0.0, 0.9), freq_range=(1, 150), plane=''):
    # Plot Power vs. Time
    cols = 2
    rows = 2
    fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(10, 12))
    axes = axes.flatten()  # Flatten the axes array for easy indexing

    for i, (category, tfa_data) in enumerate(zip(epoch_categories, data_list)):
        time = tfa_data.times  # Extract time points
        freqs = tfa_data.freqs  # Extract frequency points

        # Find the indices for the desired frequency range
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        time_mask = (time >= time_window[0]) & (time <= time_window[1])

        # Average power over the selected frequency range
        power = tfa_data.data[:, freq_mask, :][:, :, time_mask].mean(axis=1).flatten()  # Shape: (n_channels, freqs, time)
        ax = axes[i]
        ax.plot(time[time_mask], power)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Power')
        ax.set_title(f'{category} - Power x Time ({freq_range[0]}-{freq_range[1]} Hz)')
        ax.grid()
    plt.tight_layout()
    plt.grid(True)
    plt.savefig(results_path / f'power_over_time_{plane}.png')

plot_average_powers(epoch_categories, azimuth_data_list, time_window=(0.0, 0.9), freq_range=(1, 150), plane='azimuth')
plot_average_powers(epoch_categories, elevation_data_list, time_window=(0.0, 0.9), freq_range=(1, 150), plane='elevation')


# todo: create a statistics table, and save.
# todo: save all.





# def get_avg_power_distributions():
# mean_tfa_target = np.mean(tfa_azimuth_target.data, axis=1).flatten()
# mean_tfa_distractor = np.mean(tfa_azimuth_distractor.data, axis=1).flatten()
# mean_tfa_non_target = np.mean(tfa_azimuth_non_target.data, axis=1).flatten()

# length = len(mean_tfa_distractor)
# mean_tfa_target = mean_tfa_target[np.random.choice(len(mean_tfa_target), length, replace=False)]
# mean_tfa_non_target = mean_tfa_non_target[np.random.choice(len(mean_tfa_non_target), length, replace=False)]
# tfa_df = pd.DataFrame({'Target': mean_tfa_target, 'Distractor': mean_tfa_distractor, 'Non-Target': mean_tfa_non_target})
# tfa_df_long = pd.melt(tfa_df, var_name='Condition', value_name='Mean Power')
# # test variance:
#
# levene_var = levene(mean_tfa_target, mean_tfa_distractor, mean_tfa_non_target)
# target_shapiro = shapiro(mean_tfa_target)
# distractor_shapiro = shapiro(mean_tfa_distractor)
# non_target_shapiro = shapiro(mean_tfa_non_target)
#
# kruskal_test = kruskal(mean_tfa_target,mean_tfa_distractor, mean_tfa_non_target)
#
# # Assuming your data is combined into one DataFrame with a 'Group' column for categories
# posthoc_results = posthoc_dunn(tfa_df_long, val_col='Mean Power', group_col='Condition', p_adjust='holm')  # Holm-Bonferroni
# print(posthoc_results)
# # Access specific p-values
# pval_distractor_non_target = posthoc_results.loc["Distractor", "Non-Target"]
# pval_target_distractor = posthoc_results.loc["Target", "Distractor"]
# pval_target_non_target = posthoc_results.loc['Target', 'Non-Target']
#
# def cliffs_delta(x, y):
#     n1, n2 = len(x), len(y)
#     more = sum([1 for xi in x for yi in y if xi > yi])
#     less = sum([1 for xi in x for yi in y if xi < yi])
#     return (more - less) / (n1 * n2)
#
# delta_target_distractor = cliffs_delta(mean_tfa_target, mean_tfa_distractor)
# delta_target_non_target = cliffs_delta(mean_tfa_target, mean_tfa_non_target)
# delta_distractor_non_target = cliffs_delta(mean_tfa_distractor, mean_tfa_non_target)
#
# def significance_label(p_val):
#     if p_val < 0.0001:
#         return "****"
#     elif p_val < 0.001:
#         return "***"
#     elif p_val < 0.01:
#         return "**"
#     elif p_val < 0.05:
#         return "*"
#     else:
#         return "ns"
#
# pval1 = significance_label(pval_target_distractor)
# pval2 = significance_label(pval_target_non_target)
# pval3 = significance_label(pval_distractor_non_target)
#
#
# sns.violinplot(tfa_df_long, x='Condition', y='Mean Power', palette=['Royalblue', 'Darkviolet', 'Gold'])
# # Add significance annotations
# # Example positions (adjust based on your plot data range)
# x_positions = [0, 1, 2]  # x-axis positions for the conditions (0: Target, 1: Distractor, 2: Non-Target)
# y_max = tfa_df_long['Mean Power'].max()  # Maximum y-value to determine where to place the annotations
# y_offsets = [y_max + 0.05, y_max + 0.10, y_max + 0.15]  # Offsets for better visibility
#
# # Add p-value labels
# plt.text((x_positions[0] + x_positions[1]) / 2, y_offsets[0], pval1, ha='center', fontsize=10)
# plt.text((x_positions[0] + x_positions[2]) / 2, y_offsets[1], pval2, ha='center', fontsize=10)
# plt.text((x_positions[1] + x_positions[2]) / 2, y_offsets[2], pval3, ha='center', fontsize=10)
#
# # Optionally add lines to connect groups
# plt.plot([x_positions[0], x_positions[1]], [y_offsets[0] - 0.02, y_offsets[0] - 0.02], color='black', lw=1)
# plt.plot([x_positions[0], x_positions[2]], [y_offsets[1] - 0.02, y_offsets[1] - 0.02], color='black', lw=1)
# plt.plot([x_positions[1], x_positions[2]], [y_offsets[2] - 0.02, y_offsets[2] - 0.02], color='black', lw=1)
#
# # Add titles and labels
# plt.title("Mean Power Distribution Across Conditions")
# plt.ylabel("Mean Power")
# plt.xlabel("Condition")