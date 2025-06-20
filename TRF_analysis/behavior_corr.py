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


def correlate_rdiff_with_targetRT(RTs_dict, r_diff_dict, color='steelblue', cond=''):
    subs = list(r_diff_dict.keys())
    r_diffs = []
    mean_rts = []
    for sub in subs:
        if sub in RTs_dict and 'target' in RTs_dict[sub]:
            target_rts = RTs_dict[sub]['target']
            if len(target_rts) > 0:
                mean_rt = np.mean(target_rts)
                mean_rts.append(mean_rt)
                r_diffs.append(r_diff_dict[sub])
    r_diffs = np.array(r_diffs)
    mean_rts = np.array(mean_rts)
    # Spearman correlation
    r, p = spearmanr(mean_rts, r_diffs)
    r2 = r ** 2
    # Plot
    plt.figure(figsize=(6, 5))
    plt.scatter(mean_rts, r_diffs, color=color, edgecolor='k', s=80, alpha=0.8)
    m, b = np.polyfit(mean_rts, r_diffs, 1)
    x_vals = np.linspace(min(mean_rts), max(mean_rts), 100)
    plt.plot(x_vals, m * x_vals + b, color='black', linestyle='--')
    plt.xlabel('Mean RT (s)', fontsize=12)
    plt.ylabel('Target - Distractor r', fontsize=12)
    plt.title(f"R Diff vs Target RT", fontsize=13)
    # Annotation box
    annotation = f"ρ = {r:.2f}\np = {p:.4f}\n$r^2$ = {r2:.2f}"
    plt.gca().text(0.98, 0.02, annotation,
                   transform=plt.gca().transAxes,
                   fontsize=11,
                   verticalalignment='bottom',
                   horizontalalignment='right',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"Spearman rho = {r:.3f}, p = {p:.4f}, r² = {r2:.3f}")
    save_dir = default_path / f'data/eeg/behaviour/figures'
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'{plane}_{cond}_r_diff_RTs_corr.png', dpi=300)

def correlate_rt_with_relative_alpha(RTs_dict, alpha_npz, stream='target', cond=''):
    """
    Correlate mean RT (per subject) with relative alpha power.
    Parameters:
    - RTs_dict: dict with subject keys and RT arrays per stream (e.g. {'target': [...], 'distractor': [...]})
    - alpha_npz: loaded .npz object with per-subject alpha metrics
    - stream: 'target' or 'distractor'
    - cond_label: label for the condition, used in plot title
    """
    rt_vals = []
    alpha_vals = []
    sub_ids = []
    for sub in RTs_dict:
        if sub not in alpha_npz:
            continue
        rt_array = RTs_dict[sub].get(stream, [])
        if len(rt_array) == 0:
            continue
        rel_alpha = alpha_npz[sub].item().get('relative_alpha', None)
        if rel_alpha is None:
            continue
        mean_rt = np.mean(rt_array)
        rt_vals.append(mean_rt)
        alpha_vals.append(rel_alpha)
        sub_ids.append(sub)
    rt_vals = np.array(rt_vals)
    alpha_vals = np.array(alpha_vals)
    # Spearman correlation
    rho, p = spearmanr(alpha_vals, rt_vals)
    r_squared = rho**2
    # --- Plot ---
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(alpha_vals, rt_vals, color='mediumslateblue', edgecolor='k', s=80, alpha=0.8)
    m, b = np.polyfit(alpha_vals, rt_vals, 1)
    x_vals = np.linspace(min(alpha_vals), max(alpha_vals), 100)
    plt.plot(x_vals, m * x_vals + b, color='black', linestyle='-', linewidth=1.5)
    # Annotation
    plt.text(0.97, 0.03, f"ρ = {rho:.2f}\np = {p:.4f}\n$r^2$ = {r_squared:.2f}",
             ha='right', va='bottom', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))
    plt.xlabel('Relative Alpha Power', fontsize=13)
    plt.ylabel('Mean Reaction Time (s)', fontsize=13)
    plt.title(f'RT vs Relative Alpha ({stream.capitalize()}s)', fontsize=14)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    print(f"\n[INFO] Correlation results - {stream} RT):")
    print(f"Spearman ρ = {rho:.3f}, p = {p:.4f}, r² = {r_squared:.3f}")
    save_dir = default_path / f'data/eeg/behaviour/figures'
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'{plane}_{cond}_r_diff_relative_alpha_corr.png', dpi=300)


def correlate_alpha_with_dprime(performance_dict, alpha_npz, color='mediumpurple', cond='', metric=''):
    alpha_vals = []
    dprimes = []
    for sub in performance_dict:
        if sub not in alpha_npz:
            continue
        rel_alpha = alpha_npz[sub].item().get(metric, None)
        dprime = performance_dict[sub]['d_prime']
        if rel_alpha is not None and not np.isnan(dprime):
            alpha_vals.append(rel_alpha)
            dprimes.append(dprime)

    alpha_vals = np.array(alpha_vals)
    dprimes = np.array(dprimes)

    rho, p = spearmanr(alpha_vals, dprimes)
    r_squared = rho**2

    # --- Plot ---
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(alpha_vals, dprimes, color=color, edgecolor='k', s=80, alpha=0.85)
    m, b = np.polyfit(alpha_vals, dprimes, 1)
    x_vals = np.linspace(min(alpha_vals), max(alpha_vals), 100)
    plt.plot(x_vals, m * x_vals + b, color='black', linestyle='-', linewidth=1.3)

    plt.xlabel(f'{metric.capitalize().replace('_', ' ')} Power', fontsize=13)
    plt.ylabel("d'", fontsize=13)
    plt.title(f'{plane.capitalize()} — d\' vs {metric.capitalize().replace('_', ' ')}', fontsize=14)

    plt.text(0.97, 0.03, f"ρ = {rho:.2f}\np = {p:.4f}\n$r^2$ = {r_squared:.2f}",
             ha='right', va='bottom', transform=plt.gca().transAxes,
             bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"\n[INFO] Correlation results ({plane.capitalize()} — d' vs {metric}):")
    print(f"Spearman ρ = {rho:.3f}, p = {p:.4f}, r² = {r_squared:.3f}")

    save_dir = default_path / f'data/eeg/behaviour/figures'
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'{plane}_{cond}_dprime_{metric}_corr.png', dpi=300)

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


def correlate_rdiff_with_dprime(performance_dict, r_diff_dict, color='slategray', cond=''):
    dprimes = []
    r_diffs = []
    for sub in r_diff_dict:
        perf = performance_dict.get(sub)
        if perf is not None and not np.isnan(perf['d_prime']):
            dprimes.append(perf['d_prime'])
            r_diffs.append(r_diff_dict[sub])
    dprimes = np.array(dprimes)
    r_diffs = np.array(r_diffs)

    # Correlation
    rho, p = spearmanr(dprimes, r_diffs)
    r2 = rho ** 2

    # Plot
    plt.figure(figsize=(6.5, 5.5))
    plt.scatter(dprimes, r_diffs, color=color, edgecolor='k', s=80, alpha=0.8)
    m, b = np.polyfit(dprimes, r_diffs, 1)
    x_vals = np.linspace(min(dprimes), max(dprimes), 100)
    plt.plot(x_vals, m * x_vals + b, color='black', linestyle='--', linewidth=1.5)
    plt.xlabel("d'", fontsize=13)
    plt.ylabel('Target - Distractor r', fontsize=13)
    plt.title(f"R Diff vs d'", fontsize=14)

    annotation = f"ρ = {rho:.2f}\np = {p:.4f}\n$r^2$ = {r2:.2f}"
    plt.text(0.97, 0.03, annotation, transform=plt.gca().transAxes,
             ha='right', va='bottom', bbox=dict(facecolor='white', edgecolor='gray', boxstyle='round,pad=0.3'))

    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

    print(f"[{plane.capitalize()}] Correlation: Spearman ρ = {rho:.3f}, p = {p:.4f}, r² = {r2:.3f}")

    save_dir = default_path / f'data/eeg/behaviour/figures'
    save_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_dir / f'{plane}_{cond}_r_diff_dprime_corr.png', dpi=300)


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

default_path = Path('.')  # Set your base path here if needed
input_path1 = default_path / f"data/eeg/trf/trf_testing/results/single_sub/{plane}/{cond1}/{folder_type}/{predictor_short}"
rval_path_target1 = input_path1 / f"subjectwise_crossval_rvals_{plane}_target_stream_{folder_type}_{predictor_short}.npy"
rval_path_distractor1 = input_path1 / f"subjectwise_crossval_rvals_{plane}_distractor_stream_{folder_type}_{predictor_short}.npy"


rvals_target1 = np.load(rval_path_target1, allow_pickle=True).item()
rvals_distractor1 = np.load(rval_path_distractor1, allow_pickle=True).item()

subjects = list(rvals_target1.keys())

# Call comparison plot
plot_comparison_r_distribution(rvals_target1, rvals_distractor1, cond=cond1)

r_diff_dict1, r_diff_array1 = compute_r_difference(rvals_target1, rvals_distractor1)

target_r_vals = list(rvals_target1.values())
distractor_r_vals = list(rvals_distractor1.values())

t_stat1, p_val1 = ttest_rel(target_r_vals, distractor_r_vals)
print(f"T-test: t = {t_stat1:.3f}, p = {p_val1:.4f}")


from sklearn.utils import resample

diff = np.array(target_r_vals) - np.array(distractor_r_vals)
cohen_d = np.mean(diff) / np.std(diff, ddof=1)

bootstrap_means = []

for _ in range(10000):
    sample = resample(diff)
    bootstrap_means.append(np.mean(sample))

ci_lower = np.percentile(bootstrap_means, 2.5)
ci_upper = np.percentile(bootstrap_means, 97.5)
print(f"Bootstrap 95% CI of mean difference: [{ci_lower:.4f}, {ci_upper:.4f}]")



# Test correlation with performance and RT

# For target stream
# Subjects and conditions
conds = [cond1]


RTs_dict1 = extract_rts(cond1, stream1 = 'target', stream2 = 'distractor')



correlate_rdiff_with_targetRT(RTs_dict1, r_diff_dict1, color='teal', cond=cond1)


alpha1 = Path.cwd() / f'data/eeg/alpha/{plane}/{cond1}/alpha_metrics{cond1}.npz'
alpha1 = np.load(alpha1, allow_pickle=True)



correlate_rt_with_relative_alpha(RTs_dict1, alpha1, stream='target', cond=cond1)

# === performance and R diff === #

stim_data1 = collect_stimulus_data(subjects, cond=cond1)

performance_dict1 = {}

for sub, stim_data in stim_data1.items():
    if stim_data['target'] and stim_data['distractor']:
        perf = compute_subject_performance(stim_data['target'], stim_data['distractor'])
        performance_dict1[sub] = perf


correlate_rdiff_with_dprime(performance_dict1, r_diff_dict1, color='tomato', cond=cond1)
correlate_alpha_with_dprime(performance_dict1, alpha1, color='mediumpurple', cond=cond1, metric='relative_alpha')
# correlate_alpha_with_dprime(performance_dict1, alpha1, color='orange', cond=cond1, metric='alpha_lateralization')
# Positive ALI → greater alpha power in the right hemisphere
# → Inhibition of the right (target) hemisphere

target_rts = {}
for sub, sub_dict in RTs_dict1.items():
    target_rt = sub_dict['target']
    target_rts[sub] = target_rt

# correlate target rt with target r values:
subs = list(rvals_target1.keys())
mean_rts = []
for sub in subs:
    target_rt = target_rts[sub]
    if len(target_rt) > 0:
        mean_rt = np.mean(target_rt)
        mean_rts.append(mean_rt)

mean_rts = list(mean_rts)
# Spearman correlation
r, p = spearmanr(mean_rts, diff)
r2 = r ** 2
# Plot
plt.figure(figsize=(6, 5))
plt.scatter(mean_rts, diff, edgecolor='k', s=80, alpha=0.8)
m, b = np.polyfit(mean_rts, diff, 1)
x_vals = np.linspace(min(mean_rts), max(mean_rts), 100)
plt.plot(x_vals, m * x_vals + b, color='black', linestyle='--')
plt.xlabel('Mean RT (s)', fontsize=12)
plt.ylabel('Target r', fontsize=12)
plt.title(f"Target R values vs Target RTs", fontsize=13)
# Annotation box
annotation = f"ρ = {r:.2f}\np = {p:.4f}\n$r^2$ = {r2:.2f}"
plt.gca().text(0.98, 0.02, annotation,
               transform=plt.gca().transAxes,
               fontsize=11,
               verticalalignment='bottom',
               horizontalalignment='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
print(f"Spearman rho = {r:.3f}, p = {p:.4f}, r² = {r2:.3f}")
save_dir = default_path / f'data/eeg/behaviour/figures'
save_dir.mkdir(parents=True, exist_ok=True)
plt.savefig(save_dir / f'{plane}_{cond1}_r_vals_RTs_corr.png', dpi=300)


# === ITC === #
itc_path = default_path / f'data/eeg/trf/trf_testing/results/single_sub/ITC/{plane}/{cond1}'
for files in itc_path.iterdir():
    if 'npz' in files.name:
        itc_dict = np.load(files)

itc_target = itc_dict['itc_target']
itc_distractor = itc_dict['itc_distractor']

# Assume freqs is a 1D array of shape (100,)
freqs = np.logspace(np.log10(1), np.log10(8), num=100)
itc_t = itc_target.mean(axis=1)  # shape (18,)
itc_d = itc_distractor.mean(axis=1)
itc_diff = itc_t - itc_d

r, p = theta_corr = pearsonr(itc_diff, diff)


# trf metrics path:

metrics_path = default_path/ f'data/eeg/trf/trf_testing/results/single_sub/{plane}/{cond1}/all_stims/on_en_ov_RT/weights/metrics_summary'
for files in metrics_path.iterdir():
    if 'subject_metrics' in files.name:
        trf_metrics = pd.read_csv(files)

trf_keys = trf_metrics.keys()
target_rms_early = trf_metrics['target_rms_early']
distractor_rms_early = trf_metrics['distractor_rms_early']

target_rms_late = trf_metrics['target_rms_late']
distractor_rms_late = trf_metrics['distractor_rms_late']

ttest_rel(target_rms_early, distractor_rms_early)
ttest_rel(target_rms_late, distractor_rms_late)