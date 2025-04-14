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
def load_all_band_metrics(csv_dir):
    all_dfs = []
    for file in csv_dir.glob('*_bands_frontal.csv'):
        parts = file.stem.split('_')
        condition = parts[0]
        epoch_type = '_'.join(parts[1:-3])
        df = pd.read_csv(file, sep=';')
        df['condition'] = condition
        df['epoch_type'] = epoch_type
        all_dfs.append(df)
    return pd.concat(all_dfs, ignore_index=True)

long_band_df = load_all_band_metrics(csv_path)
long_band_df['epoch_type'] = long_band_df['epoch_type'].replace({
    'non_targets_distractor_no': 'non_targets_distractor_no_response',
    'non_targets_target_no': 'non_targets_target_no_response'
})

from scipy.stats import shapiro

def test_normality_per_band(df, metric):
    results = []

    for band in df['band'].unique():
        for cond in df['condition'].unique():
            sub_df = df[(df['band'] == band) & (df['condition'] == cond)]
            pivot = sub_df.pivot(index='subject', columns='epoch_type', values=metric).dropna()

            if {'non_targets_target_no_response', 'non_targets_distractor_no_response'}.issubset(pivot.columns) and len(pivot) >= 3:
                diff = pivot['non_targets_distractor_no_response'] - pivot['non_targets_target_no_response']
                stat, pval = shapiro(diff)
                results.append({
                    'band': band,
                    'condition': cond,
                    'N': len(diff),
                    'W_stat': stat,
                    'p_value': pval,
                    'normal': pval > 0.05
                })

    return pd.DataFrame(results)


normality_itc = test_normality_per_band(long_band_df, 'mean_ITC')
normality_power = test_normality_per_band(long_band_df, 'mean_power')
normality_freq = test_normality_per_band(long_band_df, 'peak_freq')

# === DESCRIPTIVE SUMMARY STATS === #
def get_summary_stats(df, metric):
    return (
        df.groupby(['band', 'condition', 'epoch_type'])[metric]
        .agg(['mean', 'std', 'count'])
        .reset_index()
        .rename(columns={'mean': f'{metric}_mean', 'std': f'{metric}_std', 'count': 'N'})
    )

summary_itc = get_summary_stats(long_band_df, 'mean_ITC')
summary_power = get_summary_stats(long_band_df, 'mean_power')
summary_freq = get_summary_stats(long_band_df, 'peak_freq')


# === TUKEY POST HOC (PER CONDITION) === #
def run_tukey_per_condition_and_band(df, metric):
    results = {}
    for band in df['band'].unique():
        for cond in df['condition'].unique():
            sub_df = df[(df['band'] == band) & (df['condition'] == cond)]
            if sub_df['epoch_type'].nunique() > 1:
                tukey = pairwise_tukeyhsd(sub_df[metric], sub_df['epoch_type'], alpha=0.05)
                results[(band, cond)] = tukey.summary()
    return results

tukey_results_itc = run_tukey_per_condition_and_band(long_band_df, 'mean_ITC')
tukey_results_power = run_tukey_per_condition_and_band(long_band_df, 'mean_power')
tukey_results_freq = run_tukey_per_condition_and_band(long_band_df, 'peak_freq')

# === COHEN'S D (PAIRED TARGET VS DISTRACTOR PER CONDITION) === #
def compute_cohens_d_per_band(df, metric):
    results = []
    for band in df['band'].unique():
        for cond in df['condition'].unique():
            sub_df = df[(df['band'] == band) & (df['condition'] == cond)]
            pivot = sub_df.pivot(index='subject', columns='epoch_type', values=metric).dropna()

            if {'non_targets_target_no_response', 'non_targets_distractor_no_response'}.issubset(pivot.columns):
                target = pivot['non_targets_target_no_response']
                distractor = pivot['non_targets_distractor_no_response']
                diff = distractor - target
                d = diff.mean() / diff.std(ddof=1)
                tstat, pval = ttest_rel(distractor, target)
                results.append({
                    'band': band,
                    'condition': cond,
                    'Cohen_d': d,
                    't': tstat,
                    'p': pval
                })
    return pd.DataFrame(results)

effectsize_itc = compute_cohens_d_per_band(long_band_df, 'mean_ITC')
effectsize_power = compute_cohens_d_per_band(long_band_df, 'mean_power')
effectsize_freq = compute_cohens_d_per_band(long_band_df, 'peak_freq')

# === MIXED EFFECTS MODEL (ALL DATA) === #
def run_mixedlm_per_band(df, metric):
    models = {}
    for band in df['band'].unique():
        band_df = df[df['band'] == band]
        model = smf.mixedlm(f"{metric} ~ epoch_type * condition", data=band_df, groups=band_df["subject"])
        result = model.fit()
        models[band] = result
    return models

mixed_models_itc = run_mixedlm_per_band(long_band_df, 'mean_ITC')
mixed_models_power = run_mixedlm_per_band(long_band_df, 'mean_power')
mixed_models_freq = run_mixedlm_per_band(long_band_df, 'peak_freq')


# === PLOT 1: VIOLIN PLOT BY CONDITION === #
def plot_metric_per_band(metric, y_label, title_prefix):
    for band in long_band_df['band'].unique():
        plt.figure(figsize=(8, 5))
        sns.violinplot(
            data=long_band_df[long_band_df['band'] == band],
            x='epoch_type', y=metric, hue='condition',
            split=True, inner='box'
        )
        plt.title(f"{title_prefix} – {band}")
        plt.ylabel(y_label)
        plt.xticks(rotation=30)
        plt.tight_layout()
        plt.show()

plot_metric_per_band('mean_ITC', 'ITC', 'ITC by Epoch Type and Condition')
plot_metric_per_band('mean_power', 'Power', 'Band Power by Epoch Type and Condition')
plot_metric_per_band('peak_freq', 'Peak Frequency (Hz)', 'Peak Frequency by Epoch Type and Condition')


# === PLOT 2: BOXPLOT WITH STARS === #
from scipy.stats import shapiro

def plot_with_stats_per_band(df, metric, y_label, title_prefix, save_prefix):
    from scipy.stats import shapiro

    pair = ('non_targets_target_no_response', 'non_targets_distractor_no_response')
    fig_path = csv_path / 'figures'
    fig_path.mkdir(parents=True, exist_ok=True)

    for band in df['band'].unique():
        band_df = df[df['band'] == band]
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.boxplot(data=band_df, x='epoch_type', y=metric, hue='condition', ax=ax)

        for cond in band_df['condition'].unique():
            cond_data = band_df[band_df['condition'] == cond]
            pivot = cond_data.pivot(index='subject', columns='epoch_type', values=metric).dropna()

            if pair[0] in pivot.columns and pair[1] in pivot.columns:
                diff = pivot[pair[1]] - pivot[pair[0]]
                _, pval = shapiro(diff)
                normal = pval > 0.05
                test_type = 't-test_paired' if normal else 'Wilcoxon'
                print(f"{metric} | {band} | {cond}: using {test_type} (p = {pval:.3f})")

                annotator = Annotator(ax, [pair],
                                      data=cond_data,
                                      x='epoch_type', y=metric)
                annotator.configure(test=test_type, text_format='star', loc='outside')
                annotator.apply_and_annotate()

        ax.set_ylabel(y_label)
        ax.set_title(f"{title_prefix} – {band}")
        plt.tight_layout()
        fig.savefig(fig_path / f"{save_prefix}_{band}.png")
        plt.show()

plot_with_stats_per_band(long_band_df, 'mean_ITC', 'ITC', 'ITC by Epoch Type', 'itc_boxplot')
plot_with_stats_per_band(long_band_df, 'mean_power', 'Power', 'Power by Epoch Type', 'power_boxplot')
plot_with_stats_per_band(long_band_df, 'peak_freq', 'Peak Frequency (Hz)', 'Peak Frequency by Epoch Type', 'freq_boxplot')


# === EXPORT TO EXCEL === #
excel_path = csv_path / 'summary_band_metrics.csv'
long_band_df.to_csv(excel_path, index=False)
print(f"Saved to {excel_path}")


from scipy.stats import ttest_rel, wilcoxon, shapiro

def test_epoch_type_difference_per_band(df, metric):
    results = []

    pair = ('non_targets_target_no_response', 'non_targets_distractor_no_response')

    for band in df['band'].unique():
        for cond in df['condition'].unique():
            sub_df = df[(df['band'] == band) & (df['condition'] == cond)]
            pivot = sub_df.pivot(index='subject', columns='epoch_type', values=metric).dropna()

            if set(pair).issubset(pivot.columns) and len(pivot) >= 3:
                diff = pivot[pair[1]] - pivot[pair[0]]
                w_stat, p_norm = shapiro(diff)
                normal = p_norm > 0.05

                if normal:
                    stat, pval = ttest_rel(pivot[pair[1]], pivot[pair[0]])
                    test = 't-test_paired'
                else:
                    stat, pval = wilcoxon(pivot[pair[1]], pivot[pair[0]])
                    test = 'Wilcoxon'

                results.append({
                    'band': band,
                    'condition': cond,
                    'test': test,
                    'N': len(pivot),
                    'stat': stat,
                    'p': pval,
                    'normal': normal
                })

    return pd.DataFrame(results)

epoch_type_results_itc = test_epoch_type_difference_per_band(long_band_df, 'mean_ITC')
epoch_type_results_power = test_epoch_type_difference_per_band(long_band_df, 'mean_power')
epoch_type_results_freq = test_epoch_type_difference_per_band(long_band_df, 'peak_freq')
