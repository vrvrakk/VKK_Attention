import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel, wilcoxon, zscore, sem
from statsmodels.stats.multitest import fdrcorrection
import pandas as pd
matplotlib.use('TkAgg')
plt.ion()

# === Configuration ===

plane = 'azimuth'
if plane == 'azimuth':
    conds = ['a1']
elif plane == 'elevation':
    conds = ['e1']

folder_types = ['all_stims', 'non_targets', 'target_nums', 'deviants']
folder_type = folder_types[0]
for cond in conds:
    if cond == 'a1':
        target = 'Right'
    elif cond == 'e1':
        target = 'Bottom'

    weights_dir = rf"C:/Users/pppar/PycharmProjects/VKK_Attention/data/eeg/trf/trf_testing/results/single_sub/{plane}/{cond}/{folder_type}/on_en_ov_RT/weights"
    window_len = 11  # Hamming window length
    sfreq = 125  # Sampling rate (Hz)
    time_lags = np.linspace(-0.1, 1.0, 139)  # time axis

    # === Helper: Smooth with Hamming window ===
    def smooth_weights(weights):
        """
        Smooths each predictor (column) over time using a Hamming window.
        Expects weights of shape (n_lags, n_predictors)
        Returns smoothed weights of the same shape.
        """
        hamming_win = np.hamming(window_len)
        hamming_win /= hamming_win.sum()
        return np.array([
            np.convolve(weights[:, i], hamming_win, mode='same')
            for i in range(weights.shape[1])
        ]).T  # transpose back to (n_lags, n_predictors)

    # === Load, smooth per subject, and stack all weights ===
    def load_and_smooth_weights(stream_type):
        subject_ids = ['sub01', 'sub02', 'sub03', 'sub04', 'sub05', 'sub08']
        files = sorted([f for f in os.listdir(weights_dir) if stream_type in f and 'weights' in f])
        smoothed_all = []
        for f in files:
            if any(sub in f for sub in subject_ids):
                continue
            w = np.load(os.path.join(weights_dir, f), allow_pickle=True).squeeze().T  # (139, 4)
            smoothed = smooth_weights(w)
            smoothed_all.append(smoothed)
        smoothed_all = np.stack(smoothed_all, axis=0)  # (n_subjects, 139, 4)
        avg = np.mean(smoothed_all, axis=0).T  # (4, 139)
        return avg, smoothed_all  # average, per-subject

    # Load data
    smoothed_target, smoothed_target_all = load_and_smooth_weights("target_stream")
    smoothed_target_all = smoothed_target_all[:, :, 1]
    smoothed_distractor, smoothed_distractor_all = load_and_smooth_weights("distractor_stream")
    smoothed_distractor_all = smoothed_distractor_all[:, :, 1]

    # === Z-score data over time (axis=1) ===
    # smoothed_target_all_z = zscore(smoothed_target_all, axis=1)
    # smoothed_distractor_all_z = zscore(smoothed_distractor_all, axis=1)

    # === Plotting ===
    def plot_smoothed_response(data, title_prefix, colors, filename_prefix, predictor_labels=None):
        # Restrict to time lags between 0.1 and 0.4 seconds
        time_mask = (time_lags >= 0.0) & (time_lags <= 0.4)
        time_trimmed = time_lags[time_mask]

        for i in range(data.shape[0]):  # i = predictor index
            label = predictor_labels[i] if predictor_labels else f'Predictor {i+1}'
            plt.figure(figsize=(6, 3))
            plt.plot(time_trimmed, data[i, time_mask], color=colors[i], linewidth=2)
            plt.title(f'{title_prefix}: {label}')
            plt.xlabel('Time lag (s)')
            plt.ylabel('Amplitude (a.u.)')
            plt.tight_layout()
            plt.savefig(os.path.join(weights_dir, f'{filename_prefix}_{label}.png'), dpi=300)
            plt.show()

    predictor_labels = ['Onsets', 'Envelopes', 'RTs', 'Overlap']
    colors = ['royalblue', 'seagreen', 'firebrick', 'goldenrod']

    # plot_smoothed_response(smoothed_target, "Target Stream", colors, 'avg_trf_target', predictor_labels)
    # plot_smoothed_response(smoothed_distractor, "Distractor Stream", colors, 'avg_trf_distractor', predictor_labels)

    ######################

    def compare_trf_metrics(start, end, time_lags, sfreq, data_target_all, distractor_data_all, label='envelope'):
        """
        Compare TRF metrics between target and distractor within a specific time window.

        Parameters
        ----------
        start : float
            Start time of the window (in seconds).
        end : float
            End time of the window (in seconds).
        time_lags : np.ndarray
            Time vector (e.g., from TRF model).
        sfreq : float
            EEG sampling frequency.
        smoothed_target_all : ndarray (n_subjects x n_times x n_channels)
        smoothed_distractor_all : ndarray (n_subjects x n_times x n_channels)
        label : str
            Label for the predictor (e.g., 'envelope', 'semantic').

        Returns
        -------
        result_dict : dict
            Dictionary of computed stats and p-values.
        """
        # === Time window mask ===
        window_mask = (time_lags >= start) & (time_lags <= end)
        time_resolution = 1 / sfreq

        # === Extract window data ===
        t_full = data_target_all[:, window_mask]
        d_full = distractor_data_all[:, window_mask]


        # === Compute metrics ===
        t_rms, d_rms = np.sqrt(np.mean(t_full**2, axis=1)), np.sqrt(np.mean(d_full**2, axis=1))

        pos_peak_t = np.max(t_full, axis=1)

        neg_peak_t = np.min(t_full, axis=1)


        pos_peak_d = np.max(d_full, axis=1)
        neg_peak_d = np.min(d_full, axis=1)

        ptp_t = pos_peak_t - neg_peak_t
        ptp_d = pos_peak_d - neg_peak_d


        # create metrics dictionary:
        metrics_dict = {

            'target_rms': t_rms,
            'distractor_rms': d_rms,
            'target_ptp': ptp_t,
            'distractor_ptp': ptp_d}

        # === Stat helpers ===
        def safe_wilcoxon(x, y):
            try:
                return wilcoxon(x, y).pvalue
            except:
                return np.nan

        def cohen_d(x, y):
            return (np.mean(x) - np.mean(y)) / np.sqrt((np.std(x) ** 2 + np.std(y) ** 2) / 2)

        # === Package results ===
        result = {
            'Window': f'{start*1000:.0f}–{end*1000:.0f} ms',
            'Predictor': label,

            # Descriptives
            'Target RMS ± SD': f'{np.mean(t_rms):.4f} ± {np.std(t_rms):.4f}',
            'Distractor RMS ± SD': f'{np.mean(d_rms):.4f} ± {np.std(d_rms):.4f}',
            'Target PTP ± SD': f'{np.mean(ptp_t):.4f} ± {np.std(ptp_t):.4f}',
            'Distractor PTP ± SD': f'{np.mean(ptp_d):.4f} ± {np.std(ptp_d):.4f}',

            # Tests

            't-test p (RMS)': ttest_rel(t_rms, d_rms).pvalue,
            'Wilcoxon p (RMS)': safe_wilcoxon(t_rms, d_rms),
            'Cohen d (RMS)': cohen_d(t_rms, d_rms),


            't-test p (PTP)': ttest_rel(ptp_t, ptp_d).pvalue,
            'Wilcoxon p (PTP)': safe_wilcoxon(ptp_t, ptp_d),
            'Cohen d (PTP)': cohen_d(ptp_t, ptp_d)
        }

        return result, metrics_dict

    # === Save and report ===
    # Full window
    result_full_raw, metrics_full_raw = compare_trf_metrics(0.0, 0.5, time_lags, sfreq, smoothed_target_all, smoothed_distractor_all)
    # Early window
    result_early_raw, metrics_early_raw = compare_trf_metrics(0.0, 0.2, time_lags, sfreq, smoothed_target_all, smoothed_distractor_all)

    # Late window
    result_late_raw, metrics_late_raw = compare_trf_metrics(0.2, 0.5, time_lags, sfreq, smoothed_target_all, smoothed_distractor_all)

    # Combine into DataFrame
    result_dict_raw_all = {
        'full': result_full_raw,
        'early': result_early_raw,
        'late': result_late_raw
    }


    metrics_dict_raw_all = {
        'full': metrics_full_raw,
        'early': metrics_early_raw,
        'late': metrics_late_raw
    }


    # === Testing for normality ===
    from scipy.stats import shapiro
    # === Define all metric arrays ===

    def run_normality_tests(metrics_dict_all, window='full', save_path=None, label=''):
        assert window in ['full', 'early', 'late']
        metrics_dict = metrics_dict_all[window]

        normality_results = {
            'Metric': [],
            'Shapiro p-value': [],
            'Normally Distributed': []
        }

        for name, values in metrics_dict.items():
            try:
                stat, p = shapiro(values)
                normal = p > 0.05
            except Exception:
                p, normal = np.nan, False
            normality_results['Metric'].append(name)
            normality_results['Shapiro p-value'].append(p)
            normality_results['Normally Distributed'].append(normal)

        df = pd.DataFrame(normality_results)
        if save_path:
            filename = f'normality_test_results_{label}_{window}.csv'
            df.to_csv(os.path.join(save_path, filename), index=False)

        return df


    normality_df_full_raw = run_normality_tests(metrics_dict_raw_all, window='full', save_path=weights_dir, label='raw')
    normality_df_early_raw = run_normality_tests(metrics_dict_raw_all, window='early', save_path=weights_dir, label='raw')
    normality_df_late_raw = run_normality_tests(metrics_dict_raw_all, window='late', save_path=weights_dir, label='raw')


    def run_tests_and_apply_fdr(metrics_dict_all, window='full', normality_df=None):
        assert window in ['full', 'early', 'late']
        metrics_dict = metrics_dict_all[window]

        results = {}
        ttest_keys = []
        wilcoxon_keys = []
        ttest_pvals = []
        wilcoxon_pvals = []

        # Collect all base names from target_ keys
        metric_bases = [
            k.replace('target_', '') for k in metrics_dict.keys()
            if k.startswith('target_') and f'distractor_{k.replace("target_", "")}' in metrics_dict
        ]

        for base in metric_bases:
            t_key = f'target_{base}'
            d_key = f'distractor_{base}'

            tvals = metrics_dict[t_key]
            dvals = metrics_dict[d_key]

            # Skip known problematic distractor-only metrics
            if base in ['pos_amp_at_target_peak', 'neg_amp_at_target_peak']:
                continue

            # Check normality from provided DataFrame
            try:
                t_norm = normality_df.loc[normality_df['Metric'] == t_key, 'Normally Distributed'].values[0]
            except IndexError:
                t_norm = False
            try:
                d_norm = normality_df.loc[normality_df['Metric'] == d_key, 'Normally Distributed'].values[0]
            except IndexError:
                d_norm = False

            # Choose test
            if t_norm and d_norm:
                stat, p = ttest_rel(tvals, dvals)
                label = f't-test p ({base.replace("_", " ").title()})'
                ttest_keys.append(label)
                ttest_pvals.append(p)
            else:
                try:
                    stat, p = wilcoxon(tvals, dvals)
                except Exception:
                    p = np.nan
                label = f'Wilcoxon p ({base.replace("_", " ").title()})'
                wilcoxon_keys.append(label)
                wilcoxon_pvals.append(p)

            results[label] = p

        # FDR corrections
        if ttest_pvals:
            _, corrected = fdrcorrection(ttest_pvals)
            for k, p_corr in zip(ttest_keys, corrected):
                results[f'FDR-corrected {k}'] = p_corr

        if wilcoxon_pvals:
            _, corrected = fdrcorrection(wilcoxon_pvals)
            for k, p_corr in zip(wilcoxon_keys, corrected):
                results[f'FDR-corrected {k}'] = p_corr
        return results


    fdr_results_full_raw = run_tests_and_apply_fdr(metrics_dict_raw_all, window='full', normality_df=normality_df_full_raw)
    fdr_results_early_raw = run_tests_and_apply_fdr(metrics_dict_raw_all, window='early', normality_df=normality_df_early_raw)
    fdr_results_late_raw = run_tests_and_apply_fdr(metrics_dict_raw_all, window='late', normality_df=normality_df_late_raw)


    ################

    # === Plot Comparison: Target vs Distractor for Each Predictor ===
    comparison_dir = os.path.join(weights_dir, 'comparison_plots')
    os.makedirs(comparison_dir, exist_ok=True)

    import seaborn as sns


    def get_sig_star(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return 'n.s.'


    def parse_pval(p):
        """
        Convert a p-value to float for comparison:
        - If already a float, returns it.
        - If string like '< 0.01', returns 0.01.
        - Otherwise returns np.nan.
        """
        if isinstance(p, str):
            p = p.strip()
            if p.startswith('<'):
                try:
                    return float(p[1:].strip())
                except ValueError:
                    return np.nan
            try:
                return float(p)
            except ValueError:
                return np.nan
        elif isinstance(p, (float, int, np.floating)):
            return float(p)
        return np.nan


    def compute_cohens_d(x, y):
        """Compute Cohen's d for paired samples."""
        diff = np.array(x) - np.array(y)
        return np.mean(diff) / np.std(diff, ddof=1)


    def summarize_tests(fdr_results, alpha=0.05):
        """
        Summarize which test was used and FDR-corrected p-values.

        Parameters
        ----------
        fdr_results : dict
            Output from run_tests_and_apply_fdr.
        alpha : float
            Significance threshold.

        Returns
        -------
        pd.DataFrame
        """
        records = []

        for key in fdr_results:
            if key.startswith('FDR-corrected'):
                raw_key = key.replace('FDR-corrected ', '')
                test_type = 't-test' if 't-test' in raw_key else 'Wilcoxon'
                metric = raw_key.replace('t-test p (', '').replace('Wilcoxon p (', '').replace(')', '')

                corrected_p = fdr_results[key]
                raw_p = fdr_results.get(raw_key, np.nan)

                records.append({
                    'Metric': metric,
                    'Test': test_type,
                    'Raw p': f"{raw_p:.4f}" if not np.isnan(raw_p) else 'NaN',
                    'FDR-corrected p': f"{corrected_p:.4f}",
                    'Significance': get_sig_star(corrected_p) if not np.isnan(corrected_p) else 'n/a'
                })

        return pd.DataFrame(records).sort_values('FDR-corrected p')


    summary_full = summarize_tests(fdr_results_full_raw)
    summary_early = summarize_tests(fdr_results_early_raw)
    summary_late = summarize_tests(fdr_results_late_raw)


    def plot_metrics(metrics_df_dict, summary_df, window='full', plane='EEG', target='both'):
        """
        Plot boxplots of FDR-significant metrics for a given time window.

        Parameters
        ----------
        metrics_df_dict : dict
            Dict with time window keys mapping to metric dicts (e.g. metrics_dict_raw_all).
        summary_df : pd.DataFrame
            DataFrame returned by summarize_tests() with FDR-corrected results.
        window : str
            'full', 'early', or 'late'.
        plane : str
            e.g., 'EEG'.
        target : str
            e.g., 'both'.
        """
        metrics_dict = metrics_df_dict[window]
        summary_subset = summary_df[summary_df['Significance'] != 'n.s.']

        if summary_subset.empty:
            print(f"No FDR-significant metrics for window: {window}")
            return

        # === Time window labels ===
        window_labels = {
            'full': '0.0–0.5',
            'early': '0.0–0.2',
            'late': '0.2–0.5'
        }
        window_time = window_labels.get(window, '')

        # === Plotting ===
        cols = len(summary_subset)
        rows = int(np.ceil(len(summary_subset) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 4 * rows))
        if isinstance(axes, np.ndarray):
            axes = axes.flatten()
        else:
            axes = [axes]

        nice_colors = {'Target': '#4169E1', 'Distractor': '#8B0000'}  # royal blue & wine red

        for i, row in enumerate(summary_subset.itertuples()):
            base = row.Metric.lower().replace(' ', '_')
            tkey = f'target_{base}'
            dkey = f'distractor_{base}'
            tvals, dvals = np.array(metrics_dict[tkey]), np.array(metrics_dict[dkey])

            df = pd.DataFrame({
                'Value': np.concatenate([tvals, dvals]),
                'Stream': ['Target'] * len(tvals) + ['Distractor'] * len(dvals)
            })

            ax = axes[i]
            sns.boxplot(x='Stream', y='Value', data=df, hue='Stream',
                        palette=nice_colors, ax=ax, showfliers=False, width=0.5, dodge=False, legend=False)
            sns.stripplot(x='Stream', y='Value', data=df, color='black',
                          alpha=0.5, size=3, jitter=0.12, ax=ax)

            # Compute effect size
            d = compute_cohens_d(tvals, dvals)

            # Means and SDs
            t_mean, t_std = np.mean(tvals), np.std(tvals)
            d_mean, d_std = np.mean(dvals), np.std(dvals)

            legend_text = [
                f"Target: M={t_mean:.2f}, SD={t_std:.2f}",
                f"Distractor: M={d_mean:.2f}, SD={d_std:.2f}",
                f"Cohen’s d = {d:.2f}, p < 0.05"
            ]
            ax.legend(handles=[
                plt.Line2D([0], [0], label=legend_text[0], color='none'),
                plt.Line2D([0], [0], label=legend_text[1], color='none'),
                plt.Line2D([0], [0], label=legend_text[2], color='none')
            ], loc='upper right', fontsize=8, frameon=False)

            # Significance line
            y_max = max(np.max(tvals), np.max(dvals))
            y_min = min(np.min(tvals), np.min(dvals))
            y_range = y_max - y_min
            y_text = y_max + 0.1 * y_range
            ax.plot([0.25, 0.75], [y_text] * 2, color='black', lw=1)
            ax.text(0.5, y_text + 0.03 * y_range, row.Significance,
                    ha='center', va='bottom', fontsize=11)

            # Axis & title
            ax.set_title(
                f"{row.Metric.capitalize()} | {plane.capitalize()}-{target.capitalize()} | ({window_time}s\n{folder_type.capitalize().replace('_', ' ')})",
                fontsize=10)
            ax.set_ylabel('Value', fontsize=9)
            ax.set_xlabel('')
            ax.tick_params(labelsize=8)

        for j in range(len(summary_subset), len(axes)):
            fig.delaxes(axes[j])

        fig.tight_layout()
        fig.savefig(os.path.join(comparison_dir, f'trf_sig_metrics_{window}_{plane}_{target}.png'), dpi=300)
        plt.show()


    plot_metrics(metrics_dict_raw_all, summary_early, window='early', plane=plane, target=target)
    plot_metrics(metrics_dict_raw_all, summary_late, window='late', plane=plane, target=target)

    # Define base save directory
    csv_dir = r"C:\Users\pppar\PycharmProjects\VKK_Attention\data\eeg\trf\trf_testing\results\single_sub"

    # Construct full output path
    output_dir = os.path.join(csv_dir, plane, cond, folder_type, "on_en_ov_RT", "tables")
    os.makedirs(output_dir, exist_ok=True)

    # Save each summary
    summary_full.to_csv(os.path.join(output_dir, f'summary_full_{plane}_{cond}.csv'), index=False)
    summary_early.to_csv(os.path.join(output_dir, f'summary_early_{plane}_{cond}.csv'), index=False)
    summary_late.to_csv(os.path.join(output_dir, f'summary_late_{plane}_{cond}.csv'), index=False)

    print("Summaries saved to:", output_dir)


    from scipy.stats import sem, ttest_rel
    from statsmodels.stats.multitest import fdrcorrection
    from mne.stats import permutation_cluster_test

    n_samples = 18  # Set this explicitly

    # === Time window for plotting ===
    plot_mask = (time_lags >= 0.0) & (time_lags <= 0.5)
    time_plot = time_lags[plot_mask]

    # === Slice TRFs ===
    target_trfs = smoothed_target_all[:, plot_mask]
    distractor_trfs = smoothed_distractor_all[:, plot_mask]

    # === FDR-corrected pointwise t-tests ===
    p_vals = np.array([
        ttest_rel(target_trfs[:, i], distractor_trfs[:, i]).pvalue
        for i in range(target_trfs.shape[1])
    ])
    _, p_fdr = fdrcorrection(p_vals)
    sig_mask = p_fdr < 0.05

    # === Mean ± SEM ===
    target_mean = target_trfs.mean(axis=0)
    target_sem = sem(target_trfs, axis=0)
    distractor_mean = distractor_trfs.mean(axis=0)
    distractor_sem = sem(distractor_trfs, axis=0)

    # === Effect size (average Cohen's d over sig timepoints) ===
    if sig_mask.any():
        d_val = compute_cohens_d(target_trfs[:, sig_mask].mean(axis=1), distractor_trfs[:, sig_mask].mean(axis=1))
    else:
        d_val = compute_cohens_d(target_trfs.mean(axis=1), distractor_trfs.mean(axis=1))  # fallback

    # === Permutation cluster test on full 0–0.5s ===
    X = [smoothed_target_all[:, plot_mask], smoothed_distractor_all[:, plot_mask]]
    T_obs, clusters, cluster_p_values, _ = permutation_cluster_test(
        X,
        n_permutations=100000,
        tail=0,
        threshold=None,
        out_type='mask',
        verbose=True
    )

    # === PLOT ===
    plt.figure(figsize=(8, 4))

    # Plot means ± SEM
    plt.plot(time_plot, target_mean, color='#4C72B0', label='Target', linewidth=2)
    plt.fill_between(time_plot, target_mean - target_sem, target_mean + target_sem, color='#4C72B0', alpha=0.3)

    plt.plot(time_plot, distractor_mean, color='#DD5C5C', label='Distractor', linewidth=2)
    plt.fill_between(time_plot, distractor_mean - distractor_sem, distractor_mean + distractor_sem, color='#DD5C5C',
                     alpha=0.3)

    # Overlay FDR-significant regions (gray)
    in_sig = False
    for i in range(len(sig_mask)):
        if sig_mask[i] and not in_sig:
            in_sig = True
            start_idx = i
            start = time_plot[i]
        elif not sig_mask[i] and in_sig:
            in_sig = False
            end_idx = i
            end = time_plot[i]
            plt.axvspan(start, end, color='gray', alpha=0.15)
            center = (start + end) / 2
            y_max = max(np.max(target_mean[start_idx:end_idx]), np.max(distractor_mean[start_idx:end_idx]))
            plt.text(center, y_max + 0.05, '*', ha='center', va='bottom', fontsize=11)

    # Overlay cluster permutation (light red)
    for idx in np.where(cluster_p_values < 0.05)[0]:
        cluster_mask = clusters[idx]
        cluster_time = time_plot[cluster_mask]
        if cluster_time.size > 0:
            plt.axvspan(cluster_time[0], cluster_time[-1], color='red', alpha=0.12, zorder=0)

    # === Final touches ===
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.xlabel('Time lag (s)', fontsize=11)
    plt.ylabel('TRF Amplitude (a.u.)', fontsize=11)
    plt.title(f'Envelope TRF: {plane.capitalize()}-{target.capitalize()} | Target vs Distractor (n={n_samples})\n {folder_type.capitalize().replace('_', ' ')}\nFDR & Permutation Sig. | Cohen\'s d = {d_val:.2f}',
              fontsize=11)
    plt.legend(fontsize=9)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(comparison_dir, 'envelope_trf_combined_significance.png'), dpi=300)
    plt.show()











