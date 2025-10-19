# Import libraries:
# for plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
plt.ion()
import seaborn as sns

# for importing and saving
from pathlib import Path
import pickle as pkl

# for general data stuff / handling
import pandas as pd
import numpy as np

# for stats
from scipy.stats import pearsonr, spearmanr
from scipy.stats import norm
from scipy.stats import ttest_rel


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
    Returns a dict with {subject: {'target': df, 'distractor': df}}
    """
    base_dir = Path(data_dir/"performance")
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


def compute_subject_performance(target_dfs, distractor_dfs):
    """
    Given a list of dataframes for target and distractor stimuli,
    compute hits, misses, false alarms, and correct rejections.
    """
    # Concatenate all blocks
    target_all = pd.concat(target_dfs, ignore_index=True)
    # Drop rows that contain column headers as actual data (e.g., 'Response')
    target_all = target_all[target_all['Response'] != 'Response'].copy()

    # Convert to numeric (inplace) —> this will convert strings like '1' -> 1
    # stupid problems still require a solution
    target_all['Response'] = pd.to_numeric(target_all['Response'], errors='coerce')
    distractor_all = pd.concat(distractor_dfs, ignore_index=True)

    # Drop rows that contain column headers as actual data i.e. 'Response'
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


if __name__ == '__main__':

    base_dir = Path.cwd()
    data_dir = base_dir / 'data'

    planes = {'azimuth': ['a1', 'a2'],
              'elevation': ['e1', 'e2'],
              'all': ['a1', 'a2', 'e1', 'e2']}

    plane = planes['all']

    alphas = {}
    for condition in plane:
        alpha_dir = data_dir / 'eeg' / 'journal' / 'alpha' / condition
        for folders in alpha_dir.iterdir():
            with open(folders, 'rb') as a:
                alpha = pkl.load(a)
                alphas[condition] = alpha

    # keep alpha ratio as a metric:
    from EEG.preprocessing_eeg import sub_list

    sub_list = sub_list[6:]

    alpha_ratios = {cond: {} for cond in plane}
    for cond in alphas.keys():
        alpha_cond = alphas[cond]
        for sub in alpha_cond.keys():
            if sub in sub_list:
                alpha_ratio = alpha_cond[sub]['alpha_ratio']
                alpha_ratios[cond][sub] = np.mean(alpha_ratio)

    norm_scores = {cond: {} for cond in plane}
    speed_scores_norm = {cond: {} for cond in plane}
    for condition in plane:
        stim_data = collect_stimulus_data(sub_list, cond=condition)

        performance_dict = {}

        for sub, dict_data in stim_data.items():
            if dict_data['target'] and dict_data['distractor']:
                perf = compute_subject_performance(dict_data['target'], dict_data['distractor'])
                performance_dict[sub] = perf

        # Your input: performance_dict1
        # Weights: You can tune these (default = 1 for all)
        w_hit = 1
        w_miss = 1
        w_fa = 1

        # Step 1: Compute raw composite score per subject

        composite_scores_raw = {}

        for sub, data in performance_dict.items():
            n_targets = data['n_targets']
            n_distractors = data['n_distractors']

            hit_rate = data['hits'] / n_targets
            miss_rate = data['misses'] / n_targets
            fa_rate = data['false_alarms'] / n_distractors

            # Composite: reward hits, penalize misses and FAs
            raw_score = (w_hit * hit_rate) - (w_miss * miss_rate) - (w_fa * fa_rate)
            composite_scores_raw[sub] = raw_score

        # z-score raw values across subjects:
        from scipy.stats import zscore
        z_scored_vals = zscore(list(composite_scores_raw.values()))

        composite_scores_norm = {
            sub: score
            for sub, score in zip(composite_scores_raw.keys(), z_scored_vals)}
        norm_scores[condition] = composite_scores_norm

        # and finally RTs (valid window 0.2-0.9; pre and post window responses are 'errors')
        rts = {}
        for sub, stream_data in stim_data.items():
            time_diffs = []
            target_data = stream_data['target']
            for block_df in target_data:
                time_diff = block_df['Time Difference']
                time_diff = time_diff.drop(index=[0])
                time_diff_int = [float(t) for t in time_diff.values]
                time_diff_filt = [t for t in time_diff_int if 0.2 <= t <= 0.9]
                time_diffs.append(time_diff_filt)
            time_diffs_concat = np.concatenate(time_diffs)
            time_diffs_inverse = 1 / np.mean(time_diffs_concat)  # speed score
            rts[sub] = time_diffs_inverse
        # z-score across subs:
        rts_zscored = zscore(list(rts.values()))
        speed_scores_norm[condition] = rts_zscored

    # so now we have:
    # speec scores z-scored
    # alpha-ratios z-scored
    # performance z-scored

    # now load also r NSI:
    save_dir = data_dir / 'eeg' / 'journal' / 'TRF' / 'results' / 'r'
    nsi_dir = save_dir / 'NSI'

    if plane == ['a1', 'a2']:
        plane_name = 'azimuth'
        with open(nsi_dir / f'{plane_name}_r_diffs.pkl', 'rb') as az:
            r_zscored = pkl.load(az)
    elif plane == ['e1', 'e2']:
        plane_name = 'elevation'
        with open(nsi_dir / f'{plane_name}_r_diffs.pkl', 'rb') as el:
            r_zscored = pkl.load(el)
    elif plane == ['a1', 'a2', 'e1', 'e2']:
        with open(nsi_dir / f'azimuth_r_diffs.pkl', 'rb') as az:
            az_r_zscored = pkl.load(az)
        with open(nsi_dir / f'elevation_r_diffs.pkl', 'rb') as el:
            ele_r_zscored = pkl.load(el)

    # keep z-scored values:
    r_diff_z_arrays = {}
    for cond, sub in r_zscored.items():
        r_diff_z_array = np.array([d['r_diff_z'] for d in sub.values()])
        r_diff_z_arrays[cond] = r_diff_z_array

    '''
    now run correlation:
    z-scored metrics:
    1. performance accuracy: norm_scores
    2. speed scores: speed_scores_norm
    3. alpha ratios: alpha_ratios
    4. r NSI: r_diff_z_arrays
    Each dictionary contains the values of all 18 subjects, z-scored, for both conditions that a plane consists of
    i.e. azimuth (a1, a2) and elevation (e1, e2)
    
    Run Ridge Regression
    '''

    from sklearn.linear_model import RidgeCV
    import statsmodels.api as sm

    # Collect y (accuracy) and X (speed, alpha, r_nsi) across both conditions
    y_all = []
    X_all = []

    for cond in plane:  # e.g., plane = ["a1", "a2"]
        # acc = np.array(list(norm_scores[cond].values()))  # dependent variable
        # spd = speed_scores_norm[cond]  # speed (z-scored inverse RT)
        alp = np.array(list(alpha_ratios[cond].values()))  # alpha (z/log-transformed)
        rns = r_diff_z_arrays[cond]  # r_nsi (z-scored TRF metric)

        y_all.append(rns)
        X_all.append(np.column_stack([alp]))

    # Stack into single arrays
    y_all = np.concatenate(y_all)
    X_all = np.vstack(X_all)  # now shape = (n_subjects*2, 3 predictors)

    # Add intercept
    X_const = sm.add_constant(X_all)

    # Fit OLS
    model_full = sm.OLS(y_all, X_const).fit()
    print(model_full.summary(xname=["const", "alpha"]))
    # Add intercept
    X_const = sm.add_constant(X_all)

    # Initial OLS for influence stats
    model_full = sm.OLS(y_all, X_const).fit()
    influence = model_full.get_influence()

    # Cook’s distance
    cooks_d, _ = influence.cooks_distance
    outlier_idx = np.where(cooks_d > 4 / len(y_all))[0]
    print("Potential influential points:", outlier_idx)

    # Remove outliers if needed
    if len(outlier_idx) > 0:
        X_clean = np.delete(X_const, outlier_idx, axis=0)
        y_clean = np.delete(y_all, outlier_idx, axis=0)
    else:
        X_clean, y_clean = X_const, y_all

    # Final cleaned model
    model_clean = sm.OLS(y_clean, X_clean).fit()
    print(model_clean.summary(xname=["const", "alpha"]))

    #
    import statsmodels.formula.api as smf

    # merge azimuth and elevation dicts
    r_zscored_all = {}
    r_zscored_all.update(az_r_zscored)  # contains keys 'a1', 'a2'
    r_zscored_all.update(ele_r_zscored)  # contains keys 'e1', 'e2'

    # build r_diff_z_arrays from merged dict
    r_diff_z_arrays = {}
    for cond, sub in r_zscored_all.items():
        r_diff_z_array = np.array([d['r_diff_z'] for d in sub.values()])
        r_diff_z_arrays[cond] = r_diff_z_array

    # Now build long dataframe
    rows = []
    subjects = sub_list  # make sure this has all 18 subjects in correct order

    for cond in plane:
        acc = np.array([norm_scores[cond][s] for s in subjects])
        spd = speed_scores_norm[cond]
        alp = np.array([alpha_ratios[cond][s] for s in subjects])
        rns = r_diff_z_arrays[cond]

        for i, sub in enumerate(subjects):
            rows.append({
                "subject": sub,
                "condition": cond,
                "accuracy": acc[i],
                "speed": spd[i],
                "alpha": alp[i],
                "r_nsi": rns[i]
            })

    df = pd.DataFrame(rows)

    # Re-standardize predictors globally across all conditions
    for col in ["speed", "alpha", "r_nsi"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=1)

    # Mixed-effects regression: random intercept per subject
    m = smf.mixedlm("accuracy ~ speed + alpha + r_nsi + C(condition)",
                    df, groups=df["subject"])
    res = m.fit(method="powell", reml=False)
    print(res.summary())

    # test only with r-nsi:
    # A slope of 0.27 is similar to a correlation of ~0.27,
    # which would mean about 7% of the variance explained (since r² ≈ 0.27² = 0.073)
    m_simple = smf.mixedlm("accuracy ~ r_nsi + C(condition)",
                           df, groups=df["subject"])
    res_simple = m_simple.fit(method="powell", reml=False)
    print(res_simple.summary())

    # interaction between rnsi and conditions
    m_interaction = smf.mixedlm("accuracy ~ r_nsi * C(condition)",
                    df, groups=df["subject"])
    res_interaction = m_interaction.fit(method="powell", reml=False)
    print(res_interaction.summary())
    '''
    A mixed-effects regression was conducted with accuracy (z-scored) as the dependent variable, 
    rNSI and condition as fixed effects, and subject as a random intercept. 
    The analysis showed a significant positive effect of rNSI on accuracy 
    (β = 0.27, SE = 0.14, z = 1.96, p = .050), 
    indicating that participants with stronger target–distractor neural separation performed more accurately. 
    Condition had no effect on accuracy (all p = 1.00). 
    The random intercept variance (0.28) indicated some between-subject variability in accuracy.
    '''