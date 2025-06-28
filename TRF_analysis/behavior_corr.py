import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
from pathlib import Path
from scipy.stats import ttest_rel
import seaborn as sns
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from scipy.stats import norm


def plot_comparison_r_distribution(dict_target, dict_distractor, cond, color_t='mediumseagreen',
                                   color_d='darkorange'):
    """
    Compare the r-value distributions for target vs. distractor in one condition.

    Parameters:
    - dict_target: dict of {subject_id: r_value} for target stream
    - dict_distractor: dict of {subject_id: r_value} for distractor stream
    - cond: condition label (e.g., 'a1')
    - predictor_short: string used for the plot title
    - color_t, color_d: colors for target and distractor
    """
    r_target = np.array(list(dict_target.values()))
    r_distractor = np.array(list(dict_distractor.values()))

    mean_t, std_t = r_target.mean(), r_target.std(ddof=1)
    mean_d, std_d = r_distractor.mean(), r_distractor.std(ddof=1)

    print(f"[{cond.upper()}] Target mean r:     {mean_t:.4f} ± {std_t:.4f}")
    print(f"[{cond.upper()}] Distractor mean r: {mean_d:.4f} ± {std_d:.4f}")

    plt.figure(figsize=(8, 5))
    plt.hist(r_target, bins=10, alpha=0.7, label=f'Target (mean={mean_t:.2f})', color=color_t, edgecolor='black')
    plt.hist(r_distractor, bins=10, alpha=0.7, label=f'Distractor (mean={mean_d:.2f})', color=color_d,
             edgecolor='black')

    plt.axvline(mean_t, color=color_t, linestyle='--')
    plt.axvline(mean_d, color=color_d, linestyle='--')

    plt.xlabel('Cross-validated r')
    plt.ylabel('Number of Subjects')
    plt.title(f'r Distribution — {plane.capitalize()}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

def compute_r_difference(dict_target, dict_distractor):
    """
    Compute the subjectwise difference in cross-validated r values (target - distractor).

    Parameters:
    - dict_target: dict of {subject_id: r_value} for target stream
    - dict_distractor: dict of {subject_id: r_value} for distractor stream

    Returns:
    - r_diff_dict: dict of {subject_id: r_target - r_distractor}
    - r_diff_array: numpy array of the differences
    """
    common_subs = set(dict_target.keys()) & set(dict_distractor.keys())
    r_diff_dict = {sub: dict_target[sub] - dict_distractor[sub] for sub in common_subs}
    r_diff_array = np.array(list(r_diff_dict.values()))

    print(f"Included {len(common_subs)} subjects in r-difference computation.")
    print(f"Mean difference: {r_diff_array.mean():.4f}, SD: {r_diff_array.std(ddof=1):.4f}")

    return r_diff_dict, r_diff_array


def extract_rts(cond, stream1, stream2):
    RTs_dict = {}
    for sub in subjects:
        stream1_path1 = Path(default_path / f'data/eeg/predictors/RTs/{sub}/{cond}/stream1/{sub}_{cond}_stream1_RTs_series_concat.npz')
        stream1_npz = np.load(stream1_path1)
        stream1_RTs = stream1_npz['RTs']
        stream1_RTs = np.array(stream1_RTs)
        stream1_RTs = stream1_RTs[stream1_RTs != 0]
        # Keep only one RT value per response (i.e., when RT changes)
        # Compare each value to the previous one
        unique_RTs_stream1 = stream1_RTs[np.insert(np.diff(stream1_RTs) != 0, 0, True)] if len(stream1_RTs) > 0 else []

        stream2_path1 = Path(default_path / f'data/eeg/predictors/RTs/{sub}/{cond}/stream2/{sub}_{cond}_stream2_RTs_series_concat.npz')
        stream2_npz = np.load(stream2_path1)
        stream2_RTs = stream2_npz['RTs']
        stream2_RTs = np.array(stream2_RTs)
        stream2_RTs = stream2_RTs[stream2_RTs != 0]
        # Keep only one RT value per response (i.e., when RT changes)
        # Compare each value to the previous one
        unique_RTs_stream2 = stream2_RTs[np.insert(np.diff(stream2_RTs) != 0, 0, True)] if len(stream2_RTs) > 0 else []

        RTs_dict[sub] = {f'{stream1}' : unique_RTs_stream1,
                         f'{stream2}': unique_RTs_stream2}
    save_dir = default_path / f'data/eeg/behaviour/{plane}/{cond}'
    save_dir.mkdir(parents=True, exist_ok=True)
    filename = f'RTs_{cond}.npz'
    np.savez(save_dir / filename, **RTs_dict)
    return RTs_dict

def plot_clean_violin(RTs_dict1, RTs_dict2, cond1_label='', cond2_label='', stream_order=('target', 'distractor')):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import pandas as pd

    sns.set(style="whitegrid", font_scale=1.2)

    # Prepare DataFrame
    all_data = []

    def collect_data(RTs_dict, cond_label):
        for sub, rt_data in RTs_dict.items():
            for stream in stream_order:
                rts = rt_data.get(stream, [])
                for rt in rts:
                    all_data.append({
                        'Condition': cond_label,
                        'Stream': stream.capitalize(),
                        'RT (s)': rt
                    })

    collect_data(RTs_dict1, cond1_label)
    collect_data(RTs_dict2, cond2_label)

    df = pd.DataFrame(all_data)

    plt.figure(figsize=(8, 5))
    ax = sns.violinplot(x='Condition', y='RT (s)', hue='Stream', data=df,
                        palette='muted', split=True, inner='quart', linewidth=1.2)

    # Remove scatter/stripplot
    ax.set_title('Reaction Time Distributions by Condition and Stream', fontsize=14)
    ax.set_ylabel('RT (s)', fontsize=12)
    ax.set_xlabel('')
    sns.despine(trim=True)

    # Add mean ± SD + n
    for (cond, stream), group in df.groupby(['Condition', 'Stream']):
        x_base = [cond1_label, cond2_label].index(cond)
        x_shift = -0.15 if stream == 'Target' else 0.15
        x_pos = x_base + x_shift

        n = len(group)
        mean = group['RT (s)'].mean()
        std = group['RT (s)'].std()

        annotation = f"n = {n}\nμ = {mean:.2f}\nσ = {std:.2f}"
        ax.text(x_pos, group['RT (s)'].max() + 0.1,
                annotation,
                ha='center', va='bottom', fontsize=9.5, color='black')

    ax.legend(title='Stream')
    plt.tight_layout()
    plt.show()
    save_dir = default_path / f'data/eeg/behaviour/figures'
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'{plane}_RTs_distribution.png', dpi=300)

def compute_subject_performance(target_dfs, distractor_dfs):
    """
    Given a list of dataframes for target and distractor stimuli,
    compute hits, misses, false alarms, and correct rejections.
    """
    # Concatenate all blocks
    target_all = pd.concat(target_dfs, ignore_index=True)
    # Drop rows that contain column headers as actual data (e.g., 'Response')
    target_all = target_all[target_all['Response'] != 'Response'].copy()

    # Convert to numeric (inplace) — this will convert strings like '1' -> 1
    target_all['Response'] = pd.to_numeric(target_all['Response'], errors='coerce')
    distractor_all = pd.concat(distractor_dfs, ignore_index=True)

    # Drop rows that contain column headers as actual data (e.g., 'Response')
    distractor_all = distractor_all[distractor_all['Response'] != 'Response'].copy()

    # Convert to numeric (inplace) — this will convert strings like '1' -> 1
    distractor_all['Response'] = pd.to_numeric(distractor_all['Response'], errors='coerce')

    # Target metrics
    hits = (target_all['Response'] == 1).sum()
    misses = (target_all['Response'] == 0).sum()
    n_targets = hits + misses

    # Distractor metrics
    false_alarms = (distractor_all['Response'] == 2).sum()
    correct_rejections = (distractor_all['Response'] == 0).sum()
    n_distractors = false_alarms + correct_rejections

    # Rates with protection against 0 or 1
    def corrected_rate(numerator, denominator):
        if denominator == 0:
            return np.nan
        rate = numerator / denominator
        # Avoid perfect 0 or 1 for d-prime
        rate = min(max(rate, 1e-5), 1 - 1e-5)
        return rate

    hit_rate = corrected_rate(hits, n_targets)
    fa_rate = corrected_rate(false_alarms, n_distractors)

    # Optional: d-prime (Z(hit) - Z(FA))
    d_prime = norm.ppf(hit_rate) - norm.ppf(fa_rate) if not np.isnan(hit_rate) and not np.isnan(fa_rate) else np.nan

    return {
        'hits': hits,
        'misses': misses,
        'hit_rate': hit_rate,
        'false_alarms': false_alarms,
        'correct_rejections': correct_rejections,
        'fa_rate': fa_rate,
        'd_prime': d_prime,
        'n_targets': n_targets,
        'n_distractors': n_distractors
    }


def load_stimulus_csv(file_path, expected_cols=8):
    """
    Loads a stimulus CSV (comma-separated, no header).
    Returns a cleaned DataFrame or None if empty/invalid.
    """
    try:
        df = pd.read_csv(file_path, header=None)
        if df.shape[1] != expected_cols:
            df = df[0].str.split(',', expand=True)
        if df.shape[0] == 0 or df.shape[1] < expected_cols:
            return None
        df.columns = ['Row', 'Stimulus Type', 'Stimulus Stream', 'Numbers',
                      'Position', 'Timepoints', 'Time Difference', 'Response']
        return df
    except Exception as e:
        print(f"[ERROR] Could not load {file_path.name}: {e}")
        return None

def collect_stimulus_data(subjects, cond=''):
    """
    Collects valid all_target_stimuli and all_distractor_stimuli CSVs for each subject and condition.
    Returns a dict with {sub: {'target': df, 'distractor': df}} structure.
    """
    base_dir = Path("C:/Users/pppar/PycharmProjects/VKK_Attention/data/performance")
    collected = {}

    for sub in subjects:
        subj_dir = base_dir / sub / "tables"
        collected[sub] = {'target': [], 'distractor': []}

        for file in subj_dir.glob(f"*{cond}*"):
            fname = file.name.lower()
            if 'all_target_stimuli.csv' in fname:
                df = load_stimulus_csv(file)
                if df is not None:
                    collected[sub]['target'].append(df)

            elif 'all_distractor_stimuli.csv' in fname:
                df = load_stimulus_csv(file)
                if df is not None:
                    collected[sub]['distractor'].append(df)

    return collected

# --- Set these values ---
plane = 'azimuth'
if plane == 'azimuth':
    cond1 = 'a1'
elif plane == 'elevation':
    cond1 = 'e1'  # Example condition

folder_type = 'all_stims'
predictor_short = 'on_en_ov_RT'  # e.g., 'env', 'full', etc.
selected_streams = ['target_stream', 'distractor_stream']  # or 'distractor'

# --- Load r values from .npy ---

default_path = Path.cwd()  # Set your base path here if needed
save_dir = default_path / f'data/eeg/behaviour/figures'
perf_path = save_dir / plane
perf_path.mkdir(parents=True, exist_ok=True)
input_path1 = default_path / f"data/eeg/trf/trf_testing/results/single_sub/{plane}/{cond1}/{folder_type}/{predictor_short}"
rval_path_target1 = input_path1 / f"subjectwise_crossval_rvals_{plane}_target_stream_{folder_type}_{predictor_short}.npy"
rval_path_distractor1 = input_path1 / f"subjectwise_crossval_rvals_{plane}_distractor_stream_{folder_type}_{predictor_short}.npy"


rvals_target1 = np.load(rval_path_target1, allow_pickle=True).item()
rvals_distractor1 = np.load(rval_path_distractor1, allow_pickle=True).item()

subjects = list(rvals_target1.keys())

# Call comparison plot
plot_comparison_r_distribution(rvals_target1, rvals_distractor1, cond=cond1)

target_r_vals = list(rvals_target1.values())
distractor_r_vals = list(rvals_distractor1.values())

t_stat1, p_val1 = ttest_rel(target_r_vals, distractor_r_vals)
print(f"T-test: t = {t_stat1:.3f}, p = {p_val1:.4f}")


from sklearn.utils import resample

# Compute TRF r NSI for each subject
nsi_vals = (np.array(target_r_vals) - np.array(distractor_r_vals)) / (np.array(target_r_vals) + np.array(distractor_r_vals))

# Compute mean and standard deviation
nsi_mean = np.mean(nsi_vals)
nsi_std = np.std(nsi_vals)

cohen_d = np.mean(nsi_vals) / np.std(nsi_vals, ddof=1)

bootstrap_means = []

for _ in range(10000):
    sample = resample(nsi_vals)
    bootstrap_means.append(np.mean(sample))

ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
print(f"Bootstrap 95% CI of mean difference: [{ci_lower:.4f}, {ci_upper:.4f}]")

# and `nsi_vals` is a NumPy array of their corresponding average NSI values

nsi_dict = {sub: val for sub, val in zip(subjects, nsi_vals)}


# Test correlation with performance and RT

# For target stream
# Subjects and conditions
conds = [cond1]


RTs_dict1 = extract_rts(cond1, stream1 = 'target', stream2 = 'distractor')


alpha1 = Path.cwd() / f'data/eeg/alpha/{plane}/{cond1}/alpha_metrics{cond1}.npz'
alpha1 = np.load(alpha1, allow_pickle=True)



# === performance and R diff === #

stim_data1 = collect_stimulus_data(subjects, cond=cond1)

performance_dict1 = {}

for sub, stim_data in stim_data1.items():
    if stim_data['target'] and stim_data['distractor']:
        perf = compute_subject_performance(stim_data['target'], stim_data['distractor'])
        performance_dict1[sub] = perf

# Your input: performance_dict1
# Weights: You can tune these (default = 1 for all)
w_hit = 1
w_miss = 1
w_fa = 1

# Step 1: Compute raw composite score per subject
composite_scores_raw = {}

for sub, data in performance_dict1.items():
    n_targets = data['n_targets']
    n_distractors = data['n_distractors']

    hit_rate = data['hits'] / n_targets
    miss_rate = data['misses'] / n_targets
    fa_rate = data['false_alarms'] / n_distractors

    # Composite: reward hits, penalize misses and FAs
    raw_score = (w_hit * hit_rate) - (w_miss * miss_rate) - (w_fa * fa_rate)
    composite_scores_raw[sub] = raw_score

# Step 2: Normalize the scores to 0–1 range
min_score = min(composite_scores_raw.values())
max_score = max(composite_scores_raw.values())

composite_scores_norm = {
    sub: (score - min_score) / (max_score - min_score)
    for sub, score in composite_scores_raw.items()
}

# === RTs === #
target_rts = {}
for sub, sub_dict in RTs_dict1.items():
    target_rt = sub_dict['target']
    target_rts[sub] = target_rt

# correlate target rt with target r values:
mean_rts = []
for sub in subjects:
    target_rt = target_rts[sub]
    if len(target_rt) > 0:
        mean_rt = np.mean(target_rt)
        mean_rts.append(mean_rt)

mean_rts = list(mean_rts)

# === plot RT trends === #
# === Setup ===
n_blocks = 5
subject_ids = list(stim_data1.keys())
renamed_ids = [f"sub{str(i + 1).zfill(2)}" for i in range(len(subject_ids))]

# === Collect block-level means ===
all_subject_block_rts = []

for sub in subject_ids:
    block_means = []
    for block_df in stim_data1[sub]['target']:
        block_df['Response'] = pd.to_numeric(block_df['Response'], errors='coerce').astype('Int64')
        valid_block_df = block_df[block_df['Response'] == 1]
        rts = valid_block_df['Time Difference'].astype(float)
        mean_rt = rts.mean() if not rts.empty else np.nan
        block_means.append(mean_rt)
    all_subject_block_rts.append(block_means)

# === Grand summary stats ===
all_rts_flat = [rt for block in all_subject_block_rts for rt in block if not np.isnan(rt)]
overall_mean_rt = np.mean(all_rts_flat)
overall_std_rt = np.std(all_rts_flat)
overall_sem_rt = overall_std_rt / np.sqrt(len(all_rts_flat))

# === Plot ===
plt.figure(figsize=(10, 6))

# Plot individual subject traces
for block_means in all_subject_block_rts:
    plt.plot(range(1, n_blocks + 1), block_means, color='lightgray', alpha=0.5, linewidth=1)

# Compute group-level means
all_subject_block_rts = np.array(all_subject_block_rts, dtype=np.float64)
block_means = np.nanmean(all_subject_block_rts, axis=0)
block_sems = np.nanstd(all_subject_block_rts, axis=0) / np.sqrt(np.sum(~np.isnan(all_subject_block_rts), axis=0))

# Overlay mean ± SEM line
plt.errorbar(
    range(1, n_blocks + 1), block_means, yerr=block_sems,
    fmt='-o', color='midnightblue', ecolor='steelblue', elinewidth=2, capsize=4,
    linewidth=2.5, label=f'Group Mean ± SEM\nOverall: {overall_mean_rt:.3f}s ± {overall_sem_rt:.3f}'
)

# === Aesthetics ===
plt.xlabel('Block Number', fontsize=12)
plt.ylabel('Reaction Time (s)', fontsize=12)
plt.title(f'{plane.capitalize()} — RT Trend Across Blocks (Target Trials)', fontsize=14, weight='bold')
plt.xticks(range(1, n_blocks + 1), fontsize=10)
plt.yticks(fontsize=10)
plt.grid(True, alpha=0.3)
plt.legend(frameon=False, fontsize=10, loc='upper right')
plt.tight_layout()

# Save
plt.savefig(perf_path / f'{plane}_{cond1}_rt_trends_clean.png', dpi=300)
plt.show()



# RT big picture:
# Collect all valid target RTs
all_rts = []

for sub_data in stim_data1.values():
    for block in sub_data['target']:
        block['Response'] = pd.to_numeric(block['Response'], errors='coerce').astype('Int64')
        valid_rts = block[block['Response'] == 1]['Time Difference'].astype(float)
        all_rts.extend(valid_rts)

# Summary stats
mean_rt = np.mean(all_rts)
std_rt = np.std(all_rts)

# Plot RT distribution only
plt.figure(figsize=(8, 5))
sns.histplot(all_rts, bins=30, kde=True, color='royalblue')
plt.axvline(mean_rt, color='black', linestyle='--', label=f"Mean = {mean_rt:.3f}s ± {std_rt:.3f}")
plt.xlabel('Reaction Time (s)')
plt.ylabel('Count')
plt.title(f'{plane.capitalize()}\nDistribution of Target Reaction Times')
plt.legend()
plt.tight_layout()
plt.savefig(perf_path / f'{plane}_{cond1}_rt_hist.png')
plt.show()

# == plot performance == #
# Extract and rename
subject_ids = list(performance_dict1.keys())
n_subs = len(subject_ids)
renamed_ids = [f"sub{str(i+1).zfill(2)}" for i in range(n_subs)]  # sub01, sub02, ...

hit_rates = []
fa_rates = []

for sub in subject_ids:
    perf = performance_dict1[sub]
    hit_rate = perf['hits'] / perf['n_targets'] * 100
    fa_rate = perf['false_alarms'] / perf['n_distractors'] * 100
    hit_rates.append(hit_rate)
    fa_rates.append(fa_rate)

# Sort by hit rate
sorted_indices = np.argsort(hit_rates)
hit_rates_sorted = [hit_rates[i] for i in sorted_indices]
fa_rates_sorted = [fa_rates[i] for i in sorted_indices]
renamed_ids_sorted = [renamed_ids[i] for i in sorted_indices]

# Plot
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
x = np.arange(n_subs)

ax.bar(x - bar_width / 2, hit_rates_sorted, width=bar_width, label='Hit Rate (%)', color='royalblue')
ax.bar(x + bar_width / 2, fa_rates_sorted, width=bar_width, label='False Alarm Rate (%)', color='red')

ax.set_xticks(x)
ax.set_xticklabels(renamed_ids_sorted, rotation=45, ha='right')
ax.set_ylabel('Percentage (%)')
ax.set_title('Subject-wise Hit and False Alarm Rates')
ax.legend()
plt.tight_layout()
plt.grid(alpha=0.3)
plt.savefig(perf_path/'performance.png', dpi=300)
plt.show()

# === Group Performance === #
# === Compute group-level metrics ===
total_hits = sum([perf['hits'] for perf in performance_dict1.values()])
total_targets = sum([perf['n_targets'] for perf in performance_dict1.values()])
group_hit_rate = total_hits / total_targets * 100

total_fa = sum([perf['false_alarms'] for perf in performance_dict1.values()])
total_distractors = sum([perf['n_distractors'] for perf in performance_dict1.values()])
group_fa_rate = total_fa / total_distractors * 100

# === Plot ===
fig, ax = plt.subplots(figsize=(6, 6))
bars = ax.bar(['Hit Rate', 'False Alarm Rate'], [group_hit_rate, group_fa_rate],
              color=['royalblue', 'red'])

# Add text annotations
for bar in bars:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom', fontsize=12)

# Styling
ax.set_ylim(0, 110)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title(f'{plane.capitalize()}\nGroup-Level Performance Summary', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()

# Optional: Save
plt.savefig(perf_path / 'group_performance.png', dpi=300)
plt.show()

# === multivariate model === #
# # Initialize dict to store all subject DataFrames
phase_nsi_dict = np.load(f'C:/Users/pppar/PycharmProjects/VKK_Attention/data/eeg/behaviour/{plane}/{cond1}/phase_nsi.npz', allow_pickle=True)
phase_nsi_dict = phase_nsi_dict['arr_0']
phase_nsi_dict = phase_nsi_dict.item()
# relative alpha
relative_alphas = {}
for sub in subjects:
    alpha_metrics = alpha1[sub].item()
    relative_alpha = alpha_metrics['relative_alpha']
    relative_alphas[sub] = relative_alpha

subject_dfs = {}

for sub in subjects:
    phase_nsi = phase_nsi_dict[sub]
    performance = composite_scores_norm[sub]
    subject_dfs[sub] = {'phase_nsi': phase_nsi, 'performance': performance}


# Initialize lists
subjects = []
phase_nsi = []
performance = []
relative_alphas_list = []

# Choose correct index for phase NSI
n = 4 if plane == 'azimuth' else 6

# Extract data from subject dictionary
for sub, data in subject_dfs.items():
    subjects.append(sub)
    phase_nsi.append(data['phase_nsi'][n])
    performance.append(data['performance'])
    relative_alphas_list.append(relative_alphas[sub])

# Create dataframe
df = pd.DataFrame({
    'subjects': subjects,
    'r_nsi': list(nsi_dict.values()),
    'phase_nsi': phase_nsi,
    'performance': performance,
    'mean_rts': mean_rts,
    'relative_alphas': relative_alphas_list,
})

# Log-transform alpha power to improve normality
df['log_alphas'] = np.log(df['relative_alphas'] + 1e-6)


import statsmodels.api as sm

# Define predictors and target
X = df[['r_nsi', 'phase_nsi', 'mean_rts', 'log_alphas']]
y = df['performance']
X = sm.add_constant(X)

# Fit model
model_ols = sm.OLS(y, X).fit()
print(model_ols.summary())


from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Fit sklearn model for plotting
X_all = df[['r_nsi', 'phase_nsi', 'mean_rts', 'log_alphas']]
y_all = df['performance']
model = LinearRegression().fit(X_all, y_all)
y_pred = model.predict(X_all)

# Plot actual vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_all, y_pred, color='navy', alpha=0.7)
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('Actual Performance')
plt.ylabel('Predicted Performance')
plt.title('Regression: Actual vs Predicted Performance')
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import r2_score

features = ['r_nsi', 'phase_nsi', 'mean_rts', 'log_alphas']
r2_values = {}

for feature in features:
    X_feat = df[[feature]]
    model = LinearRegression().fit(X_feat, y)
    y_pred = model.predict(X_feat)
    r2_values[feature] = r2_score(y, y_pred)

# Display R²
print("\nR² values for single-predictor models:")
for feat, r2 in r2_values.items():
    print(f"{feat}: R² = {r2:.3f}")

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

# Create binary label (median split)
df['high_perf'] = (df['performance'] >= df['performance'].median()).astype(int)

# Scale predictors
X_bin = df[['r_nsi', 'phase_nsi', 'mean_rts', 'log_alphas']]
y_bin = df['high_perf']
X_bin_scaled = StandardScaler().fit_transform(X_bin)

# Logistic regression with CV
clf = LogisticRegression()
acc_scores = cross_val_score(clf, X_bin_scaled, y_bin, cv=5, scoring='accuracy')

print(f"\nLogistic Regression Accuracy (5-fold): {acc_scores}")
print(f"Mean Accuracy: {acc_scores.mean():.3f} ± {acc_scores.std():.3f}")


from scipy.stats import shapiro
import numpy as np

residuals = model_ols.resid

# Histogram
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=10, edgecolor='black')
plt.title("Histogram of Residuals")
plt.xlabel("Residual")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# Q-Q Plot
sm.qqplot(residuals, line='s')
plt.title("QQ Plot of Residuals")
plt.tight_layout()
plt.show()

# Shapiro-Wilk test
shapiro_stat, shapiro_p = shapiro(residuals)
print(f"\nShapiro-Wilk test p-value = {shapiro_p:.4f}")
if shapiro_p > 0.05:
    print("Residuals appear normally distributed.")
else:
    print("Residuals are likely not normal.")


# Cook's distance
influence = model_ols.get_influence()
cooks_d = influence.cooks_distance[0]

# Plot Cook’s Distance
plt.figure(figsize=(8, 4))
markerline, stemlines, baseline = plt.stem(np.arange(len(cooks_d)), cooks_d)
plt.setp(stemlines, linewidth=1)
plt.setp(markerline, markersize=5)
plt.axhline(4 / len(df), color='red', linestyle='--', label='Cutoff (4/n)')
plt.title("Cook's Distance for Each Subject")
plt.xlabel("Subject Index")
plt.ylabel("Cook's Distance")
plt.legend()
plt.tight_layout()
plt.show()

# Print influential outliers
outliers = np.where(cooks_d > 4 / len(df))[0]
print(f"\nPotential influential outliers (index): {outliers}")

df_clean = df.drop(index=outliers)
# Refit model with df_clean...

# Refit model
X_int = df_clean[['r_nsi', 'phase_nsi', 'mean_rts', 'log_alphas']]
y = df_clean['performance']
X_int = sm.add_constant(X_int)
model_int = sm.OLS(y, X_int).fit()
print(model_int.summary())


# Refit model final- remove dead regressors
X_final = df_clean[['r_nsi', 'mean_rts', 'log_alphas']]
y_final = df_clean['performance']
X_final = sm.add_constant(X_final)
model_final = sm.OLS(y_final, X_final).fit()
print(model_final.summary())

import matplotlib.ticker as ticker

sns.set(style="whitegrid", font_scale=1.2)

# Define predictors and titles
predictors = ['r_nsi', 'mean_rts', 'log_alphas']
titles = ['Neural Selectivity Index (r)', 'Mean Reaction Time (s)', 'Log Alpha Power']

# Loop through and create one plot per figure
for pred, title in zip(predictors, titles):
    plt.figure(figsize=(6, 5), dpi=100)
    sns.regplot(data=df, x=pred, y='performance',
                scatter_kws={'s': 50, 'alpha': 0.8},
                line_kws={'color': 'black', 'linewidth': 2})

    plt.title(title, fontweight='bold')
    plt.xlabel(pred.replace('_', ' ').title(), fontweight='bold')
    plt.ylabel('Performance (d′)', fontweight='bold')
    plt.gca().yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    plt.tight_layout()
    plt.show()
