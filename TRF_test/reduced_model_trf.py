import copy
# 1: Import Libraries
# for defining directories and loading/saving
import os
from pathlib import Path
import pickle as pkl

# for designing the matrix
import numpy as np
import pandas as pd
from mtrf import TRF
import random

# eeg:
import mne

# stats etc:
from scipy.stats import shapiro, ttest_rel, wilcoxon, friedmanchisquare
from statsmodels.stats.multitest import multipletests
import pingouin as pg

# plotting:
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.ion()

'''
Recommended approach for r-values:
1. run reduced TRF models for each stream: envelopes, phoenems (intial phoneme of each stimulus excluded) 
+ button presses
2. keep average = False, so that you get r-values for each channel
3. transform r-values with Fischer's z formula: 1/2 * ln( 1+r / 1-r )
    It’s not a standardized score; it’s a transformed version of r
    Stabilize variance of r before averaging or comparing.
    Used when combining correlations across channels, subjects, or conditions.
4. Subtract z-transformed r-target - r-distractor
5. average across ROI
6. Use Δz on mixed models with performance accuracy
7. If you want to report in r units, back-transform after averaging/contrasting:
    r=e^(2z)−1/e^(2z)+1
'''


def matrix_vif(matrix):
    X = sm.add_constant(matrix)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    print(vif)
    return vif


def run_model(X_folds, Y_folds, sub_list):
    if condition in ['a1', 'a2']:
        X_folds_filt = X_folds[6:]  # only filter for the conditions that are from the old design (elevation)
        Y_folds_filt = Y_folds[6:]
        sub_list = sub_list[6:]
    else:
        X_folds_filt = X_folds
        Y_folds_filt = Y_folds
        sub_list = sub_list

    predictions_dict = {}
    for sub, pred_fold, eeg_fold in zip(sub_list, X_folds_filt, Y_folds_filt):
        trf = TRF(direction=1, method='ridge')  # forward model
        trf.train(stimulus=pred_fold, response=eeg_fold, fs=sfreq, tmin=tmin, tmax=tmax, regularization=best_lambda, average=True, seed=42)
        # Do I want one TRF across all the data? → average=True
        predictions, r = trf.predict(stimulus=pred_fold, response=eeg_fold, average=False)
        # get r vals for each channel bish
        weights = trf.weights
        predictions_dict[sub] = {'predictions': predictions, 'r': r, 'weights': weights}

    time = trf.times

    return time, predictions_dict


if __name__ == '__main__':

    all_ch = np.array(['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
              'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz',
              'C4', 'T8', 'TP9', 'CP5', 'CP1', 'CP2', 'CP6',
              'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1',
              'Oz', 'O2', 'PO10', 'AF7', 'AF3', 'AF4', 'AF8', 'F5',
              'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8',
              'FT10', 'C5', 'C1', 'C2', 'C6', 'TP7', 'CP3', 'CPz',
              'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3',
              'POz', 'PO4', 'PO8', 'FCz'])

    phoneme_roi = np.array(['F3', 'F4', 'F5', 'F6', 'F7', 'F8',
                            'FC3', 'FC4', 'FC5', 'FC6', 'FT7', 'FT8'])  # supposedly phoneme electrodes
    env_roi = np.array(['Cz'])
    # specify condition:
    conditions = ['a1', 'a2', 'e1', 'e2']

    stim_type = 'all'

    r_values_dict = {}

    for condition in conditions:
        r_values = {}
        # directories:
        base_dir = Path.cwd()
        data_dir = base_dir / 'data' / 'eeg'
        predictor_dir = data_dir / 'predictors'
        bad_segments_dir = predictor_dir / 'bad_segments'
        eeg_dir = Path('D:/VKK_Attention/data/eeg/preprocessed/results')

        # load matrices of chosen stimulus and condition:
        dict_dir = data_dir / 'journal' / 'TRF' / 'matrix' / condition / stim_type
        with open(dict_dir / f'{condition}_matrix_target.pkl', 'rb') as f:
            target_dict = pkl.load(f)

        with open(dict_dir / f'{condition}_matrix_distractor.pkl', 'rb') as f:
            distractor_dict = pkl.load(f)

        sub_list = list(distractor_dict.keys())

        X_folds_target = []
        X_folds_distractor = []
        Y_folds = []
        # Stack predictors for the target stream
        for sub, target_data, distractor_data in zip(sub_list, target_dict.values(), distractor_dict.values()):
            eeg = target_data['eeg']
            X_target = np.column_stack(
                [target_data['envelopes'], target_data['phonemes'], target_data['responses']])
            X_distractor = np.column_stack(
                [distractor_data['envelopes'], distractor_data['phonemes'], distractor_data['responses']])

            Y_eeg = eeg
            print("X_target shape:", X_target.shape)
            print("X_distractor shape:", X_distractor.shape)
            print("EEG shape:", Y_eeg.shape)
            # checking collinearity:
            import statsmodels.api as sm
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            # Build combined DataFrame
            col_names = ['envelopes', 'phonemes', 'responses']

            # Build combined DataFrame
            target_df = pd.DataFrame(
                np.column_stack([X_target]),
                columns=[f'{name}_target' for name in col_names])

            distractor_df = pd.DataFrame(
                np.column_stack([X_distractor]),
                columns=[f'{name}_distractor' for name in col_names])

            # Add constant for VIF calculation
            target_vif = matrix_vif(target_df)
            distractor_vif = matrix_vif(distractor_df)

            # split into trials:
            target_predictors_stacked = target_df.values  # ← ready for modeling
            distractor_predictors_stacked = distractor_df.values
            X_folds_target.append(target_predictors_stacked)
            X_folds_distractor.append(distractor_predictors_stacked)
            Y_folds.append(Y_eeg)

        random.seed(42)

        best_lambda = 0.01

        tmin = - 0.1
        tmax = 1.0
        sfreq = 125

        # threshold = 0.1  # e.g., keep channels with r >= 0.1 - not using rn

        time, target_predictions_dict = run_model(X_folds_target, Y_folds, sub_list)
        _, distractor_predictions_dict = run_model(X_folds_distractor, Y_folds, sub_list)

        for sub in target_predictions_dict.keys():
            target_r_val = target_predictions_dict[sub]['r']
            distractor_r_val = distractor_predictions_dict[sub]['r']
            r_values[sub] = {'target': target_r_val, 'distractor': distractor_r_val}
        r_values_dict[condition] = r_values

    # now z-transform r values of each stream, all electrodes:
    r_values_transformed_dict = {cond: {} for cond in conditions}
    for cond, sub_dict in r_values_dict.items():
        transformed_r_vals = {}
        for sub_name in sub_dict.keys():
            target_r_values = np.arctanh(sub_dict[sub_name]['target'])
            distractor_r_values = np.arctanh(sub_dict[sub_name]['distractor'])
            transformed_r_vals[sub_name] = {'target': target_r_values, 'distractor': distractor_r_values}
        r_values_transformed_dict[cond] = transformed_r_vals

    # now compare r values:
    def compare_r_values(r_values_transformed_dict, predictor=''):
        if predictor == 'phonemes':
            roi = phoneme_roi
            masking = np.isin(all_ch, roi)
        elif predictor == 'envelopes':
            roi = env_roi
            masking = np.isin(all_ch, roi)
        # extract z-transformed r-diff per condition
        azimuth = {'a1': {}, 'a2': {}}
        elevation = {'e1': {}, 'e2': {}}
        for cond, sub_dict in r_values_transformed_dict.items():
            for sub, values in sub_dict.items():
                if cond in ['a1', 'a2']:
                    azimuth[cond][sub] = {'r_diff': values['target'][masking] - values['distractor'][masking]}
                elif cond in ['e1', 'e2']:
                    elevation[cond][sub] = {'r_diff': values['target'][masking] - values['distractor'][masking]}

        # --- Build tidy DataFrame ---
        rows = []
        for cond, sub_dict in r_values_transformed_dict.items():
            for sub, vals in sub_dict.items():
                rows.append({
                    'subject': sub,
                    'condition': cond,
                    'target': np.mean(vals['target'][masking]),
                    'distractor': np.mean(vals['distractor'][masking]),
                    'diff': np.mean(vals['target'][masking] - vals['distractor'][masking])})
        df = pd.DataFrame(rows)

        # Long format for ANOVA / Friedman
        df_long = df.melt(
            id_vars=['subject', 'condition'],
            value_vars=['target', 'distractor'],
            var_name='stream',
            value_name='r')

        print("Preview of wide DF:\n", df.head(), "\n")
        print("Preview of long DF:\n", df_long.head(), "\n")

        # --- Normality check per condition ---
        normality_flags = {}
        for cond in df['condition'].unique():
            delta = df.loc[df['condition'] == cond, 'diff']
            stat, p = shapiro(delta)
            normality_flags[cond] = (p > 0.05)
            print(f"{cond} Shapiro-Wilk p={p:.3f} -> {'normal' if p > 0.05 else 'non-normal'}")

        all_normal = all(normality_flags.values())

        # --- Condition-wise tests ---
        results = []
        for cond in df['condition'].unique():
            cond_df = df[df['condition'] == cond]
            delta = cond_df['diff']

            if normality_flags[cond]:
                # Paired t-test
                t_stat, p_val = ttest_rel(cond_df['target'], cond_df['distractor'], alternative='greater')
                dz = np.mean(delta) / np.std(delta, ddof=1)  # Cohen’s dz
                J = 1 - (3 / (4 * 18 - 9)) # n = 18
                gz = J * dz
                results.append([cond, "t-test", delta.mean(), p_val, gz, np.nan])
            else:
                # Wilcoxon signed-rank
                t_val, p_val = wilcoxon(cond_df['target'], cond_df['distractor'], alternative='greater')
                n_pos = np.sum(delta > 0)
                n_neg = np.sum(delta < 0)
                rbc = (n_pos - n_neg) / (n_pos + n_neg) if (n_pos + n_neg) > 0 else np.nan
                results.append([cond, "Wilcoxon", delta.mean(), p_val, np.nan, rbc])

        results_df = pd.DataFrame(results, columns=["condition", "test", "mean_diff", "p_raw", "gz", "r_rank_biserial"])

        # Holm-Bonferroni correction on raw p-values
        reject, p_holm, _, _ = multipletests(results_df["p_raw"], alpha=0.05, method="holm")
        results_df["p_holm"] = p_holm
        results_df["significant"] = reject

        print("\n--- Condition-wise results ---")
        print(results_df)

        # --- Global repeated-measures / Friedman ---
        print("\n--- Global test ---")
        if all_normal:
            aov = pg.rm_anova(dv='r', within=['stream', 'condition'], subject='subject',
                              data=df_long, detailed=True)
            print(aov)
        else:
            cond_diffs = [df.loc[df['condition'] == cond, 'diff'].values for cond in df['condition'].unique()]
            stat, p_val = friedmanchisquare(*cond_diffs)
            print(f"Friedman test across conditions: x$^2$={stat:.2f}, p={p_val:.3g}")
            # extract r diff (target-distractor) -> azimuth and elevation

        return df, df_long, results_df, azimuth, elevation


    phonemes_df, phonemes_df_long, phonemes_results_df, phonemes_az, phonemes_ele\
        = compare_r_values(r_values_transformed_dict, predictor='phonemes')

    envelopes_df, envelopes_df_long, envelopes_results_df, envelopes_az, envelopes_ele,\
        = compare_r_values(r_values_transformed_dict, predictor='envelopes')

    def save_data(results_df, df, azimuth, elevation, predictor=''):
        save_dir = data_dir / 'journal' / 'TRF' / 'results' / 'r' / 'main'
        save_dir.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(save_dir/f'{predictor}_r_df.csv', index=False)
        df.to_csv(save_dir/f'{predictor}_subs_r_df.csv', index=False)
        nsi_dir = save_dir / 'NSI'
        nsi_dir.mkdir(parents=True, exist_ok=True)

        with open(nsi_dir / f'{predictor}_azimuth_r_diffs.pkl', 'wb') as az:
            pkl.dump(azimuth, az)
        with open(nsi_dir / f'{predictor}_elevation_r_diffs.pkl', 'wb') as el:
            pkl.dump(elevation, el)

    save_data(phonemes_results_df, phonemes_df, phonemes_az, phonemes_ele, predictor='phonemes')
    save_data(envelopes_results_df, envelopes_df, envelopes_az, envelopes_ele, predictor='envelopes')

    # post-hoc comparison for phonemes:
    def post_hoc_wilcoxon(df):
        import itertools, pingouin as pg
        conds = df['condition'].unique()
        pairs = list(itertools.combinations(conds, 2))
        posthoc = pg.pairwise_tests(dv='diff', within='condition', subject='subject',
                                    padjust='holm', data=df, parametric=False)
        print(posthoc)
        return posthoc

    posthoc_phonemes = post_hoc_wilcoxon(phonemes_df)
    '''
    Post-hoc Wilcoxon signed-rank tests (Holm-corrected) 
    revealed a significant difference between A1 and E2 (W=21, p=0.02, Hedges’ g=0.50), 
    indicating a stronger phoneme-level attention effect in the A1 condition.
    The A1–A2 comparison showed a trend toward significance (W=29, p=0.06, g=0.55), 
    whereas all other condition pairs did not differ significantly (all p > 0.8, |g| < 0.3).
    '''

