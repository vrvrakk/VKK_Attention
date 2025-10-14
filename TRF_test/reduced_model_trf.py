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


def matrix_vif(matrix):
    X = sm.add_constant(matrix)
    vif = pd.Series([variance_inflation_factor(X.values, i) for i in range(X.shape[1])], index=X.columns)
    print(vif)
    return vif


def run_model(X_folds, Y_folds, sub_list):
    if condition in ['a1', 'a2']:
        X_folds_filt = X_folds[6:]  # only filter for the conditions that are affected
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
        predictions, r = trf.predict(stimulus=pred_fold, response=eeg_fold, average=True)
        weights = trf.weights
        predictions_dict[sub] = {'predictions': predictions, 'r': r, 'weights': weights}

    time = trf.times

    return time, predictions_dict


if __name__ == '__main__':

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
        alpha_dir = data_dir / 'journal' / 'alpha' / condition

        # load matrices of chosen stimulus and condition:
        dict_dir = data_dir / 'journal' / 'TRF' / 'matrix' / condition / stim_type
        with open(dict_dir / f'{condition}_matrix_target.pkl', 'rb') as f:
            target_dict = pkl.load(f)

        with open(dict_dir / f'{condition}_matrix_distractor.pkl', 'rb') as f:
            distractor_dict = pkl.load(f)

        sub_list = list(distractor_dict.keys())

        # load alpha:
        for files in alpha_dir.iterdir():
            if condition in files.name:
                with open(files, 'rb') as f:
                    alpha_dict = pkl.load(f)
        # keep occ_alpha:
        alpha_arrays = {}
        for sub, rows in alpha_dict.items():
            alpha_arr = rows['occ_alpha']
            if sub in sub_list:
                alpha_arrays[sub] = alpha_arr


        X_folds_target = []
        X_folds_distractor = []
        Y_folds = []
        # Stack predictors for the target stream
        for sub, target_data, distractor_data, alpha_arr in zip(sub_list, target_dict.values(), distractor_dict.values(),
                                                                alpha_arrays.values()):
            eeg = target_data['eeg']
            X_target = np.column_stack(
                [target_data['onsets'], target_data['envelopes'], target_data['phonemes'], target_data['responses']])
            X_distractor = np.column_stack(
                [distractor_data['onsets'], distractor_data['envelopes'], distractor_data['phonemes'],
                 distractor_data['responses']])

            Y_eeg = eeg
            print("X_target shape:", X_target.shape)
            print("X_distractor shape:", X_distractor.shape)
            print("EEG shape:", Y_eeg.shape)
            # checking collinearity:
            import statsmodels.api as sm
            from statsmodels.stats.outliers_influence import variance_inflation_factor

            # Build combined DataFrame
            col_names = ['onsets', 'envelopes', 'phonemes', 'responses']

            # Build combined DataFrame
            # Ensure alpha is 2D for stacking
            alpha_arr = np.array(alpha_arr).reshape(-1, 1)

            target_df = pd.DataFrame(
                np.column_stack([X_target, alpha_arr]),
                columns=[f'{name}_target' for name in col_names] + ['alpha'])

            distractor_df = pd.DataFrame(
                np.column_stack([X_distractor, alpha_arr]),
                columns=[f'{name}_distractor' for name in col_names] + ['alpha'])

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

        threshold = 0.1  # e.g., keep channels with r >= 0.05

        all_ch = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'FC5', 'FC1', 'FC2', 'FC6', 'T7', 'C3', 'Cz', 'C4', 'T8',
                  'TP9', 'CP5', 'CP1', 'CP2', 'CP6', 'TP10', 'P7', 'P3', 'Pz', 'P4', 'P8', 'PO9', 'O1', 'Oz', 'O2', 'PO10',
                  'AF7', 'AF3', 'AF4', 'AF8', 'F5', 'F1', 'F2', 'F6', 'FT9', 'FT7', 'FC3', 'FC4', 'FT8', 'FT10', 'C5', 'C1',
                  'C2', 'C6', 'TP7', 'CP3', 'CPz', 'CP4', 'TP8', 'P5', 'P1', 'P2', 'P6', 'PO7', 'PO3', 'POz', 'PO4', 'PO8',
                  'FCz']

        common_roi = np.array(['Fp1', 'F3', 'Fz', 'F4', 'FC1', 'C3', 'Cz', 'C4', 'C1', 'C2', 'CP5',
                               'CP1', 'CP3', 'P7', 'P3', 'Pz', 'P4', 'P5', 'P1', 'F1', 'F2', 'AF3', 'FCz'])

        ch_mask = np.isin(all_ch, common_roi)

        time, target_predictions_dict = run_model(X_folds_target, Y_folds, sub_list)
        _, distractor_predictions_dict = run_model(X_folds_distractor, Y_folds, sub_list)

        for sub in target_predictions_dict.keys():
            target_r_val = target_predictions_dict[sub]['r']
            distractor_r_val = distractor_predictions_dict[sub]['r']
            r_values[sub] = {'target': target_r_val, 'distractor':distractor_r_val}
        r_values_dict[condition] = r_values

    # now compare r values:
    def compare_r_values(r_values_dict):
        from scipy.stats import shapiro, ttest_rel, wilcoxon, friedmanchisquare
        from statsmodels.stats.multitest import multipletests
        import pingouin as pg

        # --- Step 1. Build tidy DataFrame ---
        rows = []
        for cond, sub_dict in r_values_dict.items():
            for sub, vals in sub_dict.items():
                rows.append({
                    'subject': sub,
                    'condition': cond,
                    'target': vals['target'],
                    'distractor': vals['distractor'],
                    'diff': vals['target'] - vals['distractor']
                })
        df = pd.DataFrame(rows)

        # Long format for ANOVA / Friedman
        df_long = df.melt(
            id_vars=['subject', 'condition'],
            value_vars=['target', 'distractor'],
            var_name='stream',
            value_name='r'
        )

        print("Preview of wide DF:\n", df.head(), "\n")
        print("Preview of long DF:\n", df_long.head(), "\n")

        # --- Step 2. Normality check per condition ---
        normality_flags = {}
        for cond in df['condition'].unique():
            delta = df.loc[df['condition'] == cond, 'diff']
            stat, p = shapiro(delta)
            normality_flags[cond] = (p > 0.05)
            print(f"{cond} Shapiro-Wilk p={p:.3f} -> {'normal' if p > 0.05 else 'non-normal'}")

        all_normal = all(normality_flags.values())

        # --- Step 3. Condition-wise tests ---
        results = []
        for cond in df['condition'].unique():
            cond_df = df[df['condition'] == cond]
            delta = cond_df['diff']

            if normality_flags[cond]:
                # Paired t-test
                t_stat, p_val = ttest_rel(cond_df['target'], cond_df['distractor'], alternative='greater')
                dz = np.mean(delta) / np.std(delta, ddof=1)  # Cohen’s dz
                results.append([cond, "t-test", delta.mean(), p_val, dz, np.nan])
            else:
                # Wilcoxon signed-rank
                stat, p_val = wilcoxon(cond_df['target'], cond_df['distractor'], alternative='greater')
                n_pos = np.sum(delta > 0)
                n_neg = np.sum(delta < 0)
                rbc = (n_pos - n_neg) / (n_pos + n_neg) if (n_pos + n_neg) > 0 else np.nan
                results.append([cond, "Wilcoxon", delta.mean(), p_val, np.nan, rbc])

        results_df = pd.DataFrame(results, columns=["condition", "test", "mean_diff", "p_raw", "dz", "r_rank_biserial"])

        # Holm-Bonferroni correction on raw p-values
        reject, p_holm, _, _ = multipletests(results_df["p_raw"], alpha=0.05, method="holm")
        results_df["p_holm"] = p_holm
        results_df["significant"] = reject

        print("\n--- Condition-wise results ---")
        print(results_df)

        # --- Step 4. Global repeated-measures / Friedman ---
        print("\n--- Global test ---")
        if all_normal:
            aov = pg.rm_anova(dv='r', within=['stream', 'condition'], subject='subject',
                              data=df_long, detailed=True)
            print(aov)
        else:
            cond_diffs = [df.loc[df['condition'] == cond, 'diff'].values for cond in df['condition'].unique()]
            stat, p_val = friedmanchisquare(*cond_diffs)
            print(f"Friedman test across conditions: x^2={stat:.2f}, p={p_val:.3g}")

        return df, df_long, results_df


    df, df_long, results_df = compare_r_values(r_values_dict)

    save_dir = data_dir / 'journal' / 'TRF' / 'results' / 'r'
    save_dir.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(save_dir/'r_df.csv', index=False)
    df.to_csv(save_dir/'subs_r_df.csv', index=False)
