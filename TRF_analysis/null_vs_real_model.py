# === Final integrated block: stats + visualization ===
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import ttest_rel, wilcoxon, shapiro
import seaborn as sns
import matplotlib.pyplot as plt

# === Directories ===
base_dir = Path.cwd()
data_dir = base_dir / 'data' / 'eeg' / 'journal' / 'TRF' / 'results' / 'diagnostics'
out_plot_dir = data_dir / 'summary_plots'
out_plot_dir.mkdir(parents=True, exist_ok=True)

planes = ['azimuth', 'elevation']
predictors = ['phonemes', 'envelopes']

all_results = []

sns.set(style="whitegrid", context="talk")

for plane in planes:
    main_dir = data_dir / 'main' / plane / 'all'
    null_dir = data_dir / 'null' / plane / 'all'

    dfs = []

    for pred in predictors:
        # --- Load data ---
        real_file = main_dir / f'prediction_accuracy_{pred}_mean_r.csv'
        null_file = null_dir / f'prediction_accuracy_{pred}_mean_r.csv'
        if not (real_file.exists() and null_file.exists()):
            print(f"[WARNING] Missing files for {plane} – {pred}")
            continue

        real = pd.read_csv(real_file, sep=';')
        null = pd.read_csv(null_file, sep=';')

        # --- Match by subject ---
        merged = pd.merge(real[['subject', 'roi_mean_r']],
                          null[['subject', 'roi_mean_r']],
                          on='subject', suffixes=('_real', '_null'))

        diff = merged['roi_mean_r_real'] - merged['roi_mean_r_null']
        w_stat, w_p = shapiro(diff)
        normal = w_p > 0.05

        if normal:
            t, p = ttest_rel(merged['roi_mean_r_real'], merged['roi_mean_r_null'])
            test_name = 'paired t-test'
            mean_diff = np.mean(diff)
            sd_diff = np.std(diff, ddof=1)
            dz = mean_diff / sd_diff
            effect_size = dz
            stat_value = t
        else:
            w, p = wilcoxon(merged['roi_mean_r_real'], merged['roi_mean_r_null'])
            test_name = 'Wilcoxon signed-rank'
            n_pos = np.sum(diff > 0)
            n_neg = np.sum(diff < 0)
            r_rb = (n_pos - n_neg) / len(diff)
            effect_size = r_rb
            stat_value = w

        result = {
            'plane': plane,
            'predictor': pred,
            'n': len(diff),
            'test': test_name,
            'normality_p': round(w_p, 5),
            'statistic': round(stat_value, 3),
            'p_value': f"{p:.1e}" if p < 0.001 else round(p, 5),
            'effect_size': round(effect_size, 3),
            'mean_real': round(np.mean(merged["roi_mean_r_real"]), 3),
            'mean_null': round(np.mean(merged["roi_mean_r_null"]), 3),
            'delta_r': round(np.mean(diff), 3)
        }
        all_results.append(result)

        # For plotting
        real['model'] = 'Real'
        real['predictor'] = pred
        null['model'] = 'Shuffled'
        null['predictor'] = pred
        dfs.append(pd.concat([real, null], ignore_index=True))

    # --- Combine for plotting ---
    if dfs:
        df_plane = pd.concat(dfs, ignore_index=True)
        plt.figure(figsize=(8, 6))
        sns.violinplot(
            data=df_plane,
            x='predictor', y='roi_mean_r', hue='model',
            inner=None, split=True, cut=0, linewidth=1
        )
        sns.stripplot(
            data=df_plane,
            x='predictor', y='roi_mean_r', hue='model',
            dodge=True, alpha=0.6, size=5
        )

        # Connect each subject with lines
        for pred in predictors:
            subs = df_plane[df_plane['predictor'] == pred]['subject'].unique()
            for s in subs:
                sub_data = df_plane[(df_plane['predictor'] == pred) & (df_plane['subject'] == s)]
                if len(sub_data) == 2:
                    plt.plot([pred, pred], sub_data['roi_mean_r'], color='gray', alpha=0.4, linewidth=0.8)

        plt.title(f'Model Validation – {plane.capitalize()}')
        plt.ylabel('Prediction accuracy (r)')
        plt.xlabel('Predictor')
        plt.legend(title='Model', loc='upper left')
        plt.tight_layout()
        plt.savefig(out_plot_dir / f'{plane}_model_validation.png', dpi=300)
        plt.close()
        print(f"[SAVED] Plot: {out_plot_dir / f'{plane}_model_validation.png'}")

# --- Save statistics ---
results_df = pd.DataFrame(all_results)
for plane in planes:
    out_dir = data_dir / plane / 'all'
    out_dir.mkdir(parents=True, exist_ok=True)
    results_df_plane = results_df[results_df['plane'] == plane]
    results_df_plane.to_csv(out_dir / 'statistics_model_vs_null.csv', sep=';', index=False, encoding='utf-8')

results_df.to_csv(data_dir / 'statistics_model_vs_null_ALL.csv', sep=';', index=False, encoding='utf-8')
print("\n=== Summary ===")
print(results_df)
