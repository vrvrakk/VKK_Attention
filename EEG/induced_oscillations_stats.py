import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scipy.stats import ttest_rel
from statannotations.Annotator import Annotator

# === PATHS === #
default_path = Path.cwd()
epochs_path = default_path / 'data/eeg/preprocessed/results/concatenated_data/epochs'
concat_epochs = epochs_path / 'all_subs'
csv_path = concat_epochs / 'stats'

# === LOAD AND CONCATENATE CSVs === #
def load_all_theta_metrics(csv_dir):
    all_dfs = []
    for file in csv_dir.glob('*_theta_frontal.csv'):
        parts = file.stem.split('_')
        condition = parts[0]
        epoch_type = '_'.join(parts[1:-2])  # e.g., 'nontarget_distractor'
        df = pd.read_csv(file, sep=';')
        df['condition'] = condition
        df['epoch_type'] = epoch_type
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

long_theta_df = load_all_theta_metrics(csv_path)


from scipy.stats import shapiro

def test_normality_per_condition(df, metric):
    results = []

    for cond in df['condition'].unique():
        sub_df = df[df['condition'] == cond]
        pivot = sub_df.pivot(index='subject', columns='epoch_type', values=metric).dropna()

        if {'non_targets_target_no_response', 'non_targets_distractor_no_response'}.issubset(pivot.columns):
            diff = pivot['non_targets_distractor_no_response'] - pivot['non_targets_target_no_response']
            stat, pval = shapiro(diff)
            results.append({
                'condition': cond,
                'W_stat': stat,
                'p_value': pval,
                'normal': pval > 0.05
            })

    return pd.DataFrame(results)

normality_itc = test_normality_per_condition(long_theta_df, 'mean_theta_ITC')
normality_power = test_normality_per_condition(long_theta_df, 'mean_theta_power')
normality_freq = test_normality_per_condition(long_theta_df, 'peak_theta_freq')

print(normality_itc)

# === DESCRIPTIVE SUMMARY STATS === #
def get_summary_stats(df, metric):
    return (
        df.groupby(['condition', 'epoch_type'])[metric]
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'mean': f'{metric}_mean', 'std': f'{metric}_std', 'count': 'N'})
    )

summary_itc = get_summary_stats(long_theta_df, 'mean_theta_ITC')
summary_power = get_summary_stats(long_theta_df, 'mean_theta_power')
summary_freq = get_summary_stats(long_theta_df, 'peak_theta_freq')

# === TUKEY POST HOC (PER CONDITION) === #
def run_tukey_per_condition(df, metric):
    results = {}
    for cond in df['condition'].unique():
        cond_df = df[df['condition'] == cond]
        tukey = pairwise_tukeyhsd(cond_df[metric], cond_df['epoch_type'], alpha=0.05)
        results[cond] = tukey.summary()
    return results

tukey_results_itc = run_tukey_per_condition(long_theta_df, 'mean_theta_ITC')
tukey_results_power = run_tukey_per_condition(long_theta_df, 'mean_theta_power')
tukey_results_freq = run_tukey_per_condition(long_theta_df, 'peak_theta_freq')

# === COHEN'S D (PAIRED TARGET VS DISTRACTOR PER CONDITION) === #
def compute_cohens_d(df, metric):
    results = []
    for cond in df['condition'].unique():
        sub_df = df[df['condition'] == cond]
        pivot = sub_df.pivot(index='subject', columns='epoch_type', values=metric).dropna()
        target = pivot['non_targets_target_no_response']
        distractor = pivot['non_targets_distractor_no_response']
        diff = distractor - target
        d = diff.mean() / diff.std(ddof=1)
        tstat, pval = ttest_rel(distractor, target)
        results.append({
            'condition': cond,
            'Cohen_d': d,
            't': tstat,
            'p': pval
        })
    return pd.DataFrame(results)

effectsize_itc = compute_cohens_d(long_theta_df, 'mean_theta_ITC')
effectsize_power = compute_cohens_d(long_theta_df, 'mean_theta_power')
effectsize_freq = compute_cohens_d(long_theta_df, 'peak_theta_freq')

# === MIXED EFFECTS MODEL (ALL DATA) === #
model = smf.mixedlm("mean_theta_ITC ~ epoch_type * condition",
                    data=long_theta_df,
                    groups=long_theta_df["subject"])
result = model.fit()
print(result.summary())

# === PLOT 1: VIOLIN PLOT BY CONDITION === #
def plot_metric(metric, title, y_label):
    plt.figure(figsize=(8, 5))
    sns.violinplot(
        data=long_theta_df,
        x='epoch_type',
        y=metric,
        hue='condition',
        split=True,
        inner='box'
    )
    plt.title(title)
    plt.ylabel(y_label)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()

plot_metric('mean_theta_ITC', 'Theta ITC by Epoch Type and Condition', 'ITC (4–8 Hz)')
plot_metric('mean_theta_power', 'Theta Power by Epoch Type and Condition', 'Power (4–8 Hz)')
plot_metric('peak_theta_freq', 'Peak Theta Frequency by Epoch Type and Condition', 'Peak Frequency (Hz)')

# === PLOT 2: BOXPLOT WITH STARS === #
from scipy.stats import shapiro

def plot_with_stats(metric, y_label, title, save_name):
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=long_theta_df, x='epoch_type', y=metric, hue='condition', ax=ax)

    # Define epoch type pair
    pair = ('non_targets_target_no_response', 'non_targets_distractor_no_response')

    for cond in long_theta_df['condition'].unique():
        cond_data = long_theta_df[long_theta_df['condition'] == cond]
        pivot = cond_data.pivot(index='subject', columns='epoch_type', values=metric).dropna()

        if pair[0] in pivot.columns and pair[1] in pivot.columns:
            # Calculate paired differences and test for normality
            diff = pivot[pair[1]] - pivot[pair[0]]
            _, pval = shapiro(diff)
            normal = pval > 0.05

            # Choose appropriate test
            test_type = 't-test_paired' if normal else 'Wilcoxon'
            print(f"{metric} | {cond}: using {test_type} (p = {pval:.3f})")

            # Apply annotation
            annotator = Annotator(ax, [pair],
                                  data=cond_data,
                                  x='epoch_type', y=metric)
            annotator.configure(test=test_type, text_format='star', loc='outside')
            annotator.apply_and_annotate()

    ax.set_ylabel(y_label)
    ax.set_title(title)
    plt.tight_layout()

    # Save figure
    fig_path = csv_path / 'figures'
    fig_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path / f'{save_name}.png')
    plt.show()

plot_with_stats('mean_theta_ITC', 'Theta ITC (4–8 Hz)', 'Theta ITC by Epoch Type', 'theta_itc_boxplot')
plot_with_stats('mean_theta_power', 'Theta Power (4–8 Hz)', 'Theta Power by Epoch Type', 'theta_power_boxplot')
plot_with_stats('peak_theta_freq', 'Peak Theta Frequency (Hz)', 'Peak Theta Frequency by Epoch Type', 'theta_peak_freq_boxplot')

# === EXPORT TO EXCEL === #
excel_path = csv_path / 'summary_theta_metrics.csv'
long_theta_df.to_csv(excel_path, index=False)
print(f"Saved to {excel_path}")
