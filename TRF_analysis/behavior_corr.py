# Import libraries:
# for plotting
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')  # non-interactive backend, no Tkinterplt.ion()
import seaborn as sns

# for importing and saving
from pathlib import Path
import pickle as pkl

# for general data stuff / handling
import pandas as pd
import numpy as np

# for stats
from scipy.stats import norm
from scipy.stats import zscore
import statsmodels.formula.api as smf
from scipy.stats import shapiro, normaltest, levene
from scipy.stats import chi2


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
    # nevermind
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
        'n_distractors': n_distractors}


def build_plane_df(norm_scores_plane, r_z_plane, sub_list, plane_name):
    """
    Build long-format DataFrame for a given spatial plane (azimuth or elevation).

    norm_scores_plane : dict, e.g. {'a1': {...}, 'a2': {...}}
    r_z_plane         : dict, e.g. {'a1': {...}, 'a2': {...}}
    sub_list          : list of all 18 subjects
    plane_name        : 'azimuth' or 'elevation'
    """
    rows = []
    for cond, sub_dict in r_z_plane.items():
        # collect z-scored accuracy values for all subjects
        acc = np.array([norm_scores_plane[cond][s] for s in sub_list])

        # collect z-scored neural r_diff values for all subjects
        r_diff_z_array = np.array([d['r_diff'] for d in sub_dict.values()])

        for i, sub in enumerate(sub_list):
            rows.append({
                "subject": sub,
                "plane": plane_name,
                "condition": cond,
                "accuracy": acc[i],
                "r_nsi": np.mean(r_diff_z_array[i])
            })
    df = pd.DataFrame(rows)

    # Re-standardize predictors within the plane (so scales are comparable)
    for col in ["r_nsi"]:
        df[col] = (df[col] - df[col].mean()) / df[col].std(ddof=1)
    return df
    # not really necessary, I think


def run_mixed_model(df, plane_name):
    print(f"\n===== Mixed model for {plane_name.upper()} =====")
    m = smf.mixedlm("accuracy ~ r_nsi + C(condition)", df, groups=df["subject"])
    res = m.fit( reml=False)
    print(res.summary())
    return res


def get_lm_summary(res):
    # Safely extract scalar arrays (avoid pandas indexing bugs)
    params = np.asarray(res.params, dtype=float).flatten()
    bse = np.asarray(res.bse, dtype=float).flatten()
    zvals = np.asarray(res._results.tvalues, dtype=float).flatten()  # <-- direct internal access
    pvals = np.asarray(res._results.pvalues, dtype=float).flatten()
    ci = np.asarray(res.conf_int(), dtype=float)

    lm_summary = pd.DataFrame({
        "term": res.params.index,
        "coef": params,
        "se": bse,
        "z": zvals,
        "pval": pvals,
        "ci_low": ci[:, 0],
        "ci_high": ci[:, 1]
    })
    return lm_summary.round(3)


def plot_model_diagnostics(model, title=f"Model Diagnostics", predictor='', plane_name=''):
    import statsmodels.api as sm
    # Extract residuals and fitted values
    residuals = model.resid
    fitted = model.fittedvalues
    sub_names = model.random_effects.keys()
    subs = list(sub_names)
    stacked_subs = subs * 2

    # Compute outlier threshold and count
    resid_mean = np.mean(residuals)
    resid_std = np.std(residuals)
    outlier_mask = np.abs(residuals - resid_mean) > 3 * resid_std
    n_outliers = np.sum(outlier_mask)
    perc_outliers = (n_outliers / len(residuals)) * 100

    # Print outlier info
    print(f"\n=== {title} ({predictor}, {plane_name}) ===")
    print(f"Total residuals: {len(residuals)}")
    print(f"Outliers (>3 SD): {n_outliers} ({perc_outliers:.1f}%)")

    if n_outliers > 0:
        outlier_indices = np.where(outlier_mask)[0]
        print(f"Indices of outliers: {outlier_indices.tolist()}")
        for index, name in enumerate(stacked_subs):
            if index in outlier_indices.tolist():
                print(f"Subjects flagged: {stacked_subs[index]}")

    # Create plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    sns.histplot(residuals, kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title("Residuals distribution")

    sm.qqplot(residuals, line='s', ax=axes[1])
    axes[1].set_title("QQ-plot")

    sns.scatterplot(x=fitted, y=residuals, ax=axes[2], color='teal')
    axes[2].axhline(0, ls='--', color='gray')
    axes[2].set_title("Residuals vs Fitted")

    plt.suptitle(
        f"{title}\n{predictor.capitalize()} – {plane_name.capitalize()} | Outliers: {n_outliers} ({perc_outliers:.1f}%)",
        fontsize=13, fontweight='bold')
    plt.tight_layout()
    fig_dir = Path(data_dir / 'eeg' / 'journal' / 'figures' / 'LMM' / 'diagnostics')
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir/f'{predictor}_{plane_name}_diagnostics.png', dpi=300)
    plt.savefig(fig_dir/f'{predictor}_{plane_name}_diagnostics.pdf', dpi=300)
    plt.close()


def remove_lmm_outliers(df, threshold=3, plane_name=''):
    """
    Detects and removes statistical outliers from LMM data based on z-scores
    of both r_nsi and accuracy columns.

    :param df: DataFrame containing 'r_nsi' and 'accuracy'
    :param threshold: Z-score cutoff for defining outliers (default = 3)
    :param plane_name: Optional, used only for printed messages
    :return: Cleaned DataFrame (outliers removed)
    """
    z_r = np.abs(zscore(df['r_nsi'], nan_policy='omit'))
    z_a = np.abs(zscore(df['accuracy'], nan_policy='omit'))
    mask = (z_r > threshold) | (z_a > threshold)
    n_out = mask.sum()
    if n_out > 0:
        print(f"[WARNING] {plane_name.upper()}: {n_out} outlier(s) detected (|z|>{threshold})")
        print(df.loc[mask, ['subject', 'condition', 'r_nsi', 'accuracy']])
        df_clean = df.loc[~mask].copy()
    else:
        df_clean = df.copy()
    return df_clean


def check_residuals_normality(model, plane_name='', predictor=''):
    """
    Checks residual normality and homoscedasticity after outlier exclusion.

    Parameters
    ----------
    model : fitted statsmodels MixedLMResults object
        The model whose residuals to inspect.
    plane_name : str
        Name of the spatial plane (e.g., 'azimuth', 'elevation') for labeling.
    predictor : str
        Optional, specifies which predictor this model belongs to.
    """
    residuals = model.resid
    fitted = model.fittedvalues

    print(f"\n=== Residual Diagnostics: {predictor.capitalize()} – {plane_name.capitalize()} ===")
    print(f"Number of residuals: {len(residuals)}")

    # --- Normality tests ---
    shapiro_stat, shapiro_p = shapiro(residuals)

    print(f"Shapiro–Wilk test: W = {shapiro_stat:.3f}, p = {shapiro_p:.4f}")

    # --- Homoscedasticity (Levene’s test on split halves of fitted values) ---
    median_split = np.median(fitted)
    group1 = residuals[fitted <= median_split]
    group2 = residuals[fitted > median_split]
    lev_stat, lev_p = levene(group1, group2)
    print(f"Levene’s test for equal variances: W = {lev_stat:.3f}, p = {lev_p:.4f}")

    # --- Visualization ---
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.histplot(residuals, kde=True, ax=axes[0], color='steelblue')
    axes[0].set_title('Residual Distribution')
    axes[0].set_xlabel('Residuals')

    import statsmodels.api as sm
    sm.qqplot(residuals, line='s', ax=axes[1])
    axes[1].set_title('Q–Q Plot')

    plt.suptitle(f"{predictor.capitalize()} – {plane_name.capitalize()} | Normality Check", fontsize=13, fontweight='bold')
    plt.tight_layout()

    # --- Save figure ---
    fig_dir = Path(data_dir / 'eeg' / 'journal' / 'figures' / 'LMM' / 'diagnostics')
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f'{predictor}_{plane_name}_normality_check.png', dpi=300)
    plt.savefig(fig_dir / f'{predictor}_{plane_name}_normality_check.pdf', dpi=300)
    plt.close()

    print(f"Figure saved to: {fig_dir / f'{predictor}_{plane_name}_normality_check.png'}")


def compare_models(model_restricted, model_full):
    """
    Compare two nested mixed models using AIC and Likelihood Ratio Test (LRT).
    """
    aic_restricted = model_restricted.aic
    aic_full = model_full.aic
    lr_stat = 2 * (model_full.llf - model_restricted.llf)
    df_diff = model_full.df_modelwc - model_restricted.df_modelwc
    p_value = chi2.sf(lr_stat, df_diff)

    print("\n=== Model Comparison (LRT + AIC) ===")
    print(f"AIC (restricted): {aic_restricted:.2f}")
    print(f"AIC (full):       {aic_full:.2f}")
    print(f"Delta AIC:             {aic_restricted - aic_full:.2f}")
    print(f"Likelihood Ratio chi**2({df_diff:.0f}) = {lr_stat:.2f}, p = {p_value:.4f}")

    if p_value < 0.05:
        print("The full model provides a significantly better fit.")
    else:
        print("No significant improvement from adding the interaction.")

    return {"AIC_diff": aic_restricted - aic_full, "LR_stat": lr_stat, "p": p_value}


# def test_condition_interaction(df_plane, plane_name=''):
#     """
#     Test whether the r_nsi–accuracy relationship differs across conditions within a plane.
#     """
#     print(f"\n=== Testing condition interaction for {plane_name.upper()} ===")
#     m_restricted = smf.mixedlm("accuracy ~ r_nsi + C(condition)", df_plane, groups=df_plane["subject"]).fit( reml=False)
#     m_full = smf.mixedlm("accuracy ~ r_nsi * C(condition)", df_plane, groups=df_plane["subject"]).fit( reml=False)
#
#     compare_models(m_restricted, m_full)
#     print('Full model:')
#     print(m_full.summary())
#     print('Restricted model:')
#     print(m_full.summary())
#     return m_full


def run_lmm_analysis(predictor=''):
    with open(nsi_dir / f'{predictor}_azimuth_r_diffs.pkl', 'rb') as az:
        az_r_ztranformed = pkl.load(az)
    with open(nsi_dir / f'{predictor}_elevation_r_diffs.pkl', 'rb') as el:
        ele_r_ztranformed = pkl.load(el)
    # merge azimuth and elevation dicts
    r_zscored_all = {}
    r_zscored_all.update(az_r_ztranformed)  # contains keys 'a1', 'a2'
    r_zscored_all.update(ele_r_ztranformed)  # contains keys 'e1', 'e2'

    # build r_diff_z_arrays from merged dict
    r_diff_z_arrays = {}
    for cond, sub in r_zscored_all.items():
        r_diff_z_array = np.array([d['r_diff'] for d in sub.values()])
        r_diff_z_arrays[cond] = r_diff_z_array
    # separate analysis by plane:
    df_az = build_plane_df(
        norm_scores_plane={k: composite_scores_zscored[k] for k in ['a1', 'a2']},
        r_z_plane=az_r_ztranformed,
        sub_list=sub_list,
        plane_name='azimuth')
    df_ele = build_plane_df(
        norm_scores_plane={k: composite_scores_zscored[k] for k in ['e1', 'e2']},
        r_z_plane=ele_r_ztranformed,
        sub_list=sub_list,
        plane_name='elevation')
    lm_dif_dir = save_dir / 'LMM'
    lm_dif_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== RUNNING INITIAL (RAW) MODELS ===")
    res_az_raw = run_mixed_model(df_az, "azimuth")
    res_ele_raw = run_mixed_model(df_ele, "elevation")

    plot_model_diagnostics(res_az_raw, "Azimuth – Raw Model Diagnostics", predictor=predictor, plane_name='azimuth_raw')
    plot_model_diagnostics(res_ele_raw, "Elevation – Raw Model Diagnostics", predictor=predictor,
                           plane_name='elevation_raw')

    df_az_clean = remove_lmm_outliers(df_az, threshold=3, plane_name='azimuth')
    df_ele_clean = remove_lmm_outliers(df_ele, threshold=3, plane_name='elevation')

    print("\n=== RUNNING CLEANED MODELS ===")
    res_az = run_mixed_model(df_az_clean, "azimuth")
    res_ele = run_mixed_model(df_ele_clean, "elevation")

    res_az_summary = get_lm_summary(res_az)
    res_ele_summary = get_lm_summary(res_ele)

    # Post-clean diagnostics
    # plot_model_diagnostics(res_az, "Azimuth – Cleaned Model Diagnostics", predictor=predictor,
    #                        plane_name='azimuth_clean')
    # plot_model_diagnostics(res_ele, "Elevation – Cleaned Model Diagnostics", predictor=predictor,
    #                        plane_name='elevation_clean')

    df_all = pd.concat([df_az_clean, df_ele_clean])

    # Base model without interaction
    # Condition shifts accuracy up or down
    m_no_interaction = smf.mixedlm("accuracy ~ r_nsi + plane", df_all, groups=df_all["subject"]).fit(
                                                                                                     reml=False)

    # Full model with interaction (plane × r_nsi)
    # Condition changes how r nsi relates to accuracy
    res_interaction = smf.mixedlm("accuracy ~ r_nsi * plane", df_all, groups=df_all["subject"]).fit(
                                                                                                  reml=False)
    # plot_model_diagnostics(res_interaction, predictor=predictor, plane_name='interaction')

    # --- Compare AIC and LRT between models ---
    # compare_models(m_no_interaction, res_interaction)

    res_interaction_summary = get_lm_summary(res_interaction)
    no_interaction_summary = get_lm_summary(m_no_interaction)

    # check_residuals_normality(res_az, plane_name='azimuth', predictor=predictor)
    # check_residuals_normality(res_ele, plane_name='elevation', predictor=predictor)
    # check_residuals_normality(res_interaction, plane_name='interaction', predictor=predictor)

    # optional: test condition-level interactions within each plane
    # test_condition_interaction(df_az_clean, plane_name='azimuth')
    # test_condition_interaction(df_ele_clean, plane_name='elevation')

    res_az_summary.to_csv(lm_dif_dir / f'{predictor}_az_df.csv', sep=';', encoding='utf-8')
    res_ele_summary.to_csv(lm_dif_dir / f'{predictor}_ele_df.csv', sep=';', encoding='utf-8')
    res_interaction_summary.to_csv(lm_dif_dir / f'{predictor}_int_df.csv', sep=';', encoding='utf-8')

    # Print slope summaries
    print(f"\n{predictor.capitalize()} Azimuth slope for r_nsi: {res_az.params['r_nsi']:.3f}")
    print(f"{predictor.capitalize()} Elevation slope for r_nsi: {res_ele.params['r_nsi']:.3f}")
    # Store models in dictionaries for summary
    pre_models = {
        'azimuth': res_az_raw,
        'elevation': res_ele_raw}

    post_models = {
        'azimuth': res_az,
        'elevation': res_ele,
        'interaction': res_interaction}

    # Save residual diagnostic comparison
    summary_df = summarize_residual_diagnostics(pre_models, post_models, predictor=predictor)

    return res_az_summary, res_ele_summary, df_az_clean, df_ele_clean


def plot_group_performance(perf_dict, plane_name=''):
    # === Group Performance === #
    # === Compute group-level metrics ===
    total_hits = sum([perf[0]['hits'] for perf in perf_dict])
    total_targets = sum([perf[0]['n_targets'] for perf in perf_dict])
    group_hit_rate = total_hits / total_targets * 100

    total_fa = sum([perf[0]['false_alarms'] for perf in perf_dict])
    total_distractors = sum([perf[0]['n_distractors'] for perf in perf_dict])
    group_fa_rate = total_fa / total_distractors * 100

    # === Plot ===
    fig, ax = plt.subplots(figsize=(6, 6))
    bars = ax.bar(['Hit Rate', 'False Alarm Rate'], [group_hit_rate, group_fa_rate],
                  color=['royalblue', 'red'])

    # Add text annotations
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, yval + 1, f"{yval:.2f}%", ha='center', va='bottom',
                fontsize=12)

    # Styling
    ax.set_ylim(0, 110)
    ax.set_ylabel('Percentage (%)', fontsize=12)
    ax.set_title(f'{plane_name}\nGroup-Level Performance Summary', fontsize=14)
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    fig_path = data_dir / 'eeg' / 'journal' / 'figures' / 'performance'
    fig_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path / f'{plane_name}_group_performance.png', dpi=300)
    plt.close()


def summarize_residual_diagnostics(pre_models, post_models, predictor=''):
    """
    Compare residual diagnostics (normality, variance equality, outlier counts)
    before and after data cleaning for LMMs.

    Saves a CSV summary in:
        data_dir / 'eeg' / 'journal' / 'TRF' / 'results' / 'diagnostics'

    :param pre_models: dict with {'azimuth': model, 'elevation': model, 'interaction': model} before cleaning
    :param post_models: dict with same structure after cleaning
    :param predictor: name of predictor (string)
    """

    def run_tests(model):
        resid = model.resid
        sh_w, sh_p = shapiro(resid)
        dag_k, dag_p = normaltest(resid)
        # Levene's test comparing first vs second half of residuals as pseudo groups
        half = len(resid) // 2
        lev_w, lev_p = levene(resid[:half], resid[half:])
        return sh_p, dag_p, lev_p

    #  Collect results
    rows = []
    for stage, models in zip(["Pre-clean", "Post-clean"], [pre_models, post_models]):
        for plane, model in models.items():
            sh_p, dag_p, lev_p = run_tests(model)
            resid = model.resid
            resid_mean = np.mean(resid)
            resid_std = np.std(resid)
            outliers = np.sum(np.abs(resid - resid_mean) > 3 * resid_std)
            perc_out = (outliers / len(resid)) * 100

            rows.append({
                "Stage": stage,
                "Plane": plane,
                "N residuals": len(resid),
                "Outliers (>|3SD|)": outliers,
                "Outlier %": round(perc_out, 1),
                "Shapiro p": round(sh_p, 4),
                "D’Agostino p": round(dag_p, 4),
                "Levene p": round(lev_p, 4)
            })

    df_summary = pd.DataFrame(rows)

    # --- Save CSV ---
    diag_dir = Path(data_dir / 'eeg' / 'journal' / 'TRF' / 'results' / 'diagnostics')
    diag_dir.mkdir(parents=True, exist_ok=True)
    csv_path = diag_dir / f'{predictor}_residual_diagnostics_summary.csv'
    df_summary.to_csv(csv_path, sep=';', encoding='utf-8', index=False)

    print(f"\nResidual diagnostics summary saved to:\n{csv_path}")
    return df_summary


def plot_subject_performance(azimuth_perf, elevation_perf, data_dir):
    """
    Plot per-subject performance (hit rate and false alarm rate) for Azimuth and Elevation.
    azimuth_perf, elevation_perf: each is a list/array of dicts or rows with fields:
        ['hits', 'n_targets', 'false_alarms', 'n_distractors']
    """
    # === Compute per-subject performance ===
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    def extract_perf(perf_array, plane):
        data = []
        for i, perf in enumerate(perf_array):
            if isinstance(perf, dict):
                hits = perf['hits']
                n_targets = perf['n_targets']
                false_alarms = perf['false_alarms']
                n_distractors = perf['n_distractors']
            else:
                # handle case where perf is a structured array or similar
                hits = perf[0]['hits']
                n_targets = perf[0]['n_targets']
                false_alarms = perf[0]['false_alarms']
                n_distractors = perf[0]['n_distractors']

            hit_rate = hits / n_targets * 100 if n_targets > 0 else np.nan
            fa_rate = false_alarms / n_distractors * 100 if n_distractors > 0 else np.nan

            data.append({'Subject': i + 1,
                         'Plane': plane,
                         'Hit Rate (%)': hit_rate,
                         'False Alarm Rate (%)': fa_rate})
        return pd.DataFrame(data)

    df_az = extract_perf(azimuth_perf, 'Azimuth')
    df_el = extract_perf(elevation_perf, 'Elevation')
    df = pd.concat([df_az, df_el], ignore_index=True)

    # sns.set(style="whitegrid", context="talk", font_scale=1)
    fig, axes = plt.subplots(2, 1, figsize=(3.35, 4.5), sharex=True) # figsize of one column

    palettes = {'Hit Rate (%)': '#3366cc', 'False Alarm Rate (%)': '#cc3333'}
    metrics = ['Hit Rate (%)', 'False Alarm Rate (%)']

    for ax, metric in zip(axes, metrics):
        sns.boxplot(data=df, x='Plane', y=metric, color=palettes[metric],
                    width=0.6, fliersize=0, linewidth=1, ax=ax)
        sns.swarmplot(data=df, x='Plane', y=metric, color='black',
                      alpha=0.7, size=1.8, ax=ax)

        # Dynamically expand y-axis limits so no points are clipped
        ymin, ymax = df[metric].min(), df[metric].max()
        yrange = ymax - ymin if ymax > ymin else 1
        ax.set_ylim(ymin - 0.05 * yrange, ymax + 0.05 * yrange)

        ax.set_ylabel(metric)
        ax.tick_params()
        ax.grid(axis='y', alpha=0.3, linewidth=0.8)
        ax.set_xlabel('')

    # axes[0].set_title('Subject-Level Performance: Azimuth vs Elevation', fontsize=15)
    plt.tight_layout()
    sns.despine(top=True, right=False, left=False, bottom=False)
    fig.subplots_adjust(bottom=0.12)
    # fig.supxlabel('Plane', fontsize=8)
    fig_path = Path(data_dir) / 'eeg' / 'journal' / 'figures' / 'performance'
    fig_path.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_path / 'subject_performance.png', dpi=300, bbox_inches='tight')
    plt.savefig(fig_path / 'subject_performance_pdf.pdf', dpi=300, bbox_inches='tight')
    plt.close()


def plot_interactive_diagnostics(df_clean_az, df_clean_ele, predictor=''):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12

    if predictor == 'envelopes':
        limits = [-1.7, 2.1, -2.5, 1.5]
    else:
        limits = [-1.45, 2.4, -2, 2.5]
    sns.regplot(
        data=df_clean_az, x='r_nsi', y='accuracy',
        scatter=False, line_kws={'color': 'red', 'linewidth': 2}, ci=95
    )
    plt.scatter(df_clean_az['r_nsi'], df_clean_az['accuracy'], color='red', s=60, alpha=0.8, label='Azimuth')

    # --- Condition 2 ---
    sns.regplot(
        data=df_clean_ele, x='r_nsi', y='accuracy',
        scatter=False, line_kws={'color': 'blue', 'linewidth': 2}, ci=95)  # adds confidence interval shade
    plt.scatter(df_clean_ele['r_nsi'], df_clean_ele['accuracy'], color='blue', s=60, alpha=0.8,
                label="Elevation")  # add scatterpoints

    # --- Labels, title, legend ---
    plt.title(f"{predictor.capitalize()} Neural–Behavioral Relationship aross Planes",
              fontweight='bold', pad=15)
    plt.xlabel(r'Neural Selectivity Index ($r_{nsi}$)', labelpad=10)  # rnsi in latex form
    plt.ylabel('Behavioral Performance (z-scored accuracy)', labelpad=10)
    plt.legend(title='Condition', frameon=True, loc='lower right')
    plt.gca()
    plt.axis(limits)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    fig_dir = data_dir / 'eeg' / 'journal' / 'figures' / 'LMM'
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f'{predictor}_interactive_LMM_plot.png', dpi=300)
    plt.savefig(fig_dir / f'{predictor}_interactive_LMM_plot.pdf', dpi=300)
    plt.close()


def plot_plane_diagnostics(df_clean, predictor='', plane_name=''):
    '''
    A function to plot the correlation analysis results: scatter plot
        - x: r-nsi | y: performance accuracy
        - one dot per subject -> regression line + 95% CI ribbon
        - one regression line per condition (so 2 lines per panel)
    If interaction == False:
        one regression line (collapsed across the two conditions in that plane)
        optionally light gray lines connecting the same subject’s two points
        to hint at within-subject paired structure

    :param df: the dataframe with the performance accuracy and r-NSI values of each sub, within one plane
    :param predictor: specifies the type of predictor correlation scores to be plotted
    :param plane_name: which plane is the focus? if interaction == True, then all conditions involved
    :return: None. Save plots in correct path
    '''
    if plane_name == 'azimuth':
        condition1 = 'a1'
        condition2 = 'a2'
        cond_name1 = 'Right'
        cond_name2 = 'Left'
    else:
        condition1 = 'e1'
        condition2 = 'e2'
        cond_name1 = 'Bottom'
        cond_name2 = 'Top'
    if predictor == 'envelopes':
        limits = [-1.7, 2.1, -2.5, 1.5]
    else:
        limits = [-1, 2.4, -2, 2.5]

    # === Split by condition === #
    df_cond1 = df_clean[df_clean['condition'] == condition1].copy()
    df_cond2 = df_clean[df_clean['condition'] == condition2].copy()

    # === Labels and formatting === #
    # --- Condition 1 ---
    sns.regplot(
        data=df_cond1, x='r_nsi', y='accuracy',
        scatter=False, line_kws={'color': 'red', 'linewidth': 2}, ci=95)
    plt.scatter(df_cond1['r_nsi'], df_cond1['accuracy'], color='red', s=60, alpha=0.8,
                label=f"{condition1.upper()} – {cond_name1.capitalize()}")

    # --- Condition 2 ---
    sns.regplot(
        data=df_cond2, x='r_nsi', y='accuracy',
        scatter=False, line_kws={'color': 'blue', 'linewidth': 2}, ci=95)  # adds confidence interval shade
    plt.scatter(df_cond2['r_nsi'], df_cond2['accuracy'], color='blue', s=60, alpha=0.8,
                label=f"{condition2.upper()} – {cond_name2.capitalize()}")  # add scatterpoints

    # --- Labels, title, legend ---
    plt.title(f"{predictor.capitalize()} Neural–Behavioral Relationship in {plane_name.capitalize()} Plane",
              fontweight='bold', fontsize=12, pad=12)
    plt.xlabel(r'Neural Selectivity Index ($r_{nsi}$)', fontsize=12, labelpad=10)  # rnsi in latex form
    plt.ylabel('Behavioral Performance (z-scored accuracy)', fontsize=12, labelpad=10)
    plt.legend(title='Condition', frameon=True, fontsize=11, title_fontsize=12, loc='lower left',
               bbox_to_anchor=(0.02, 0.02))
    plt.gca()
    plt.axis(limits)
    plt.grid(alpha=0.3)
    fig_dir = data_dir / 'eeg' / 'journal' / 'figures' / 'LMM'
    fig_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(fig_dir / f'{predictor}_{plane_name}_LMM_per_cond.png', dpi=300)
    plt.savefig(fig_dir / f'{predictor}_{plane_name}_LMM_per_cond.pdf', dpi=300)
    plt.show()
    plt.close()


if __name__ == '__main__':

    base_dir = Path.cwd()
    data_dir = base_dir / 'data'

    planes = {'all': ['a1', 'a2', 'e1', 'e2']}

    plane = planes['all']

    # keep alpha ratio as a metric:
    from EEG.preprocessing_eeg import sub_list

    sub_list = sub_list[6:]

    zscored_scores = {cond: {} for cond in plane}
    performance_dict = {cond: {} for cond in plane}
    for condition in plane:
        stim_data = collect_stimulus_data(sub_list, cond=condition)

        for sub, dict_data in stim_data.items():
            if dict_data['target'] and dict_data['distractor']:
                perf = compute_subject_performance(dict_data['target'], dict_data['distractor'])
                performance_dict[condition][sub] = perf

    # merge sub conds:
    # Merge azimuth conditions (a1 + a2)
    azimuth_perf = np.vstack([
        np.vstack(list(performance_dict['a1'].values())),
        np.vstack(list(performance_dict['a2'].values()))])
    elevation_perf = np.vstack([np.vstack(list(performance_dict['e1'].values())),
                                np.vstack(list(performance_dict['e2'].values()))])
    # plot_group_performance(azimuth_perf, 'Azimuth')
    # plot_group_performance(elevation_perf, 'Elevation')

    # plot_subject_performance(azimuth_perf, elevation_perf, data_dir)

    # input: performance_dict
    # Step 1: Compute raw composite score per subject
    composite_scores_raw = {cond:{} for cond in plane}

    for condition, sub_dicts in performance_dict.items():
        for sub in sub_dicts.keys():
            n_targets = sub_dicts[sub]['n_targets']
            n_distractors = sub_dicts[sub]['n_distractors']

            hit_rate = sub_dicts[sub]['hits'] / n_targets
            miss_rate = sub_dicts[sub]['misses'] / n_targets
            fa_rate = sub_dicts[sub]['false_alarms'] / n_distractors

            # Composite: reward hits, penalize misses and FAs
            raw_score = hit_rate - (miss_rate + fa_rate)
            composite_scores_raw[condition][sub] = raw_score

            def analyze_plane_performance(composite_scores):
                from scipy.stats import ttest_rel
                """
                Takes a dict with keys 'a1','a2','e1','e2'
                Each contains accuracy values for 18 subjects.

                Returns means, SDs, paired t-test, and effect size (Cohen's d for paired samples).
                """

                # Convert dict structure into arrays aligned by subject
                subs = sorted(composite_scores['a1'].keys())

                azimuth = np.array([
                    (composite_scores['a1'][sub] + composite_scores['a2'][sub]) / 2
                    for sub in subs])

                elevation = np.array([
                    (composite_scores['e1'][sub] + composite_scores['e2'][sub]) / 2
                    for sub in subs])

                # Basic stats
                mean_az = np.mean(azimuth)
                sd_az = np.std(azimuth, ddof=1)

                mean_el = np.mean(elevation)
                sd_el = np.std(elevation, ddof=1)

                # Paired t-test
                tval, pval = ttest_rel(azimuth, elevation)

                # Paired Cohen's d (mean diff / SD of diff)
                diffs = azimuth - elevation
                cohen_d = np.mean(diffs) / np.std(diffs, ddof=1)

                # Print results cleanly
                print("=== Accuracy Comparison: Azimuth vs Elevation ===")
                print(f"Azimuth:   mean = {mean_az:.4f}, SD = {sd_az:.4f}")
                print(f"Elevation: mean = {mean_el:.4f}, SD = {sd_el:.4f}")
                print("")
                print(f"Paired t-test: t = {tval:.4f}, p = {pval:.6f}")
                print(f"Cohen's d (paired): d = {cohen_d:.4f}")
                print("=================================================")

                return {"azimuth": {"mean": mean_az, "sd": sd_az, "values": azimuth},
                        "elevation": {"mean": mean_el, "sd": sd_el, "values": elevation},
                        "t-test": {"t": tval, "p": pval}, "effect_size": cohen_d}


            pref_stats = analyze_plane_performance(composite_scores_raw)

    # z-score raw values across subjects:
    composite_scores_zscored = {cond: {} for cond in plane}
    for condition in plane:
        # Extract all subject raw scores for this condition
        raw_vals = np.array(list(composite_scores_raw[condition].values()))
        z_scored_vals = zscore(raw_vals)
        for sub, val in zip(sub_list, z_scored_vals):
            composite_scores_zscored[condition][sub] = val

    # so now we have: performance accuracy + r-NSI

    # now load also r NSI:
    save_dir = data_dir / 'eeg' / 'journal' / 'TRF' / 'results' / 'r'
    nsi_dir = save_dir / 'NSI'

    '''
        now run correlation:
        z-scored metrics:
        1. performance accuracy: norm_scores -> actually just z-scored, not normalized, ignore naming
        2. r NSI: r_diff_z_arrays # z-transformed, using Fischer's formula.
        Each dictionary contains the values of all 18 subjects, z-scored, for both conditions that a plane consists of
        i.e. azimuth (a1, a2) and elevation (e1, e2)

    '''
    env_res_az_summary, env_res_ele_summary, df_env_az, df_env_ele = \
        run_lmm_analysis(predictor='envelopes')

    phonemes_res_az_summary, phonemes_res_ele_summary, df_phonemes_az, \
        df_phonemes_ele = run_lmm_analysis(predictor='phonemes')

    plot_plane_diagnostics(df_env_az, predictor='envelopes', plane_name='azimuth')
    plot_plane_diagnostics(df_env_ele, predictor='envelopes', plane_name='elevation')

    plot_plane_diagnostics(df_phonemes_az, predictor='phonemes', plane_name='azimuth')
    plot_plane_diagnostics(df_phonemes_ele, predictor='phonemes', plane_name='elevation')

    plot_interactive_diagnostics(df_env_az, df_env_ele, predictor='envelopes')
    plot_interactive_diagnostics(df_phonemes_az, df_phonemes_ele, predictor='phonemes')
