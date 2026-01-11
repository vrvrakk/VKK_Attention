import numpy as np
import matplotlib.pyplot as plt
from mne.stats import permutation_cluster_test
import seaborn as sns
from pathlib import Path

component_windows = {
    "P1": (0.05, 0.15),  # early sensory
    "N1": (0.15, 0.25),  # robust first attention effects; frontocentral and temporal
    "P2": (0.25, 0.35),  # conflict monitoring / categorization of stimulus
    "N2": (0.35, 0.50)}  # late attention-driven decision making

time = np.array([-0.104, -0.096, -0.088, -0.08 , -0.072, -0.064, -0.056, -0.048,
       -0.04 , -0.032, -0.024, -0.016, -0.008,  0.   ,  0.008,  0.016,
        0.024,  0.032,  0.04 ,  0.048,  0.056,  0.064,  0.072,  0.08 ,
        0.088,  0.096,  0.104,  0.112,  0.12 ,  0.128,  0.136,  0.144,
        0.152,  0.16 ,  0.168,  0.176,  0.184,  0.192,  0.2  ,  0.208,
        0.216,  0.224,  0.232,  0.24 ,  0.248,  0.256,  0.264,  0.272,
        0.28 ,  0.288,  0.296,  0.304,  0.312,  0.32 ,  0.328,  0.336,
        0.344,  0.352,  0.36 ,  0.368,  0.376,  0.384,  0.392,  0.4  ,
        0.408,  0.416,  0.424,  0.432,  0.44 ,  0.448,  0.456,  0.464,
        0.472,  0.48 ,  0.488,  0.496,  0.504,  0.512,  0.52 ,  0.528,
        0.536,  0.544,  0.552,  0.56 ,  0.568,  0.576,  0.584,  0.592,
        0.6  ,  0.608,  0.616,  0.624,  0.632,  0.64 ,  0.648,  0.656,
        0.664,  0.672,  0.68 ,  0.688,  0.696,  0.704,  0.712,  0.72 ,
        0.728,  0.736,  0.744,  0.752,  0.76 ,  0.768,  0.776,  0.784,
        0.792,  0.8  ,  0.808,  0.816,  0.824,  0.832,  0.84 ,  0.848,
        0.856,  0.864,  0.872,  0.88 ,  0.888,  0.896,  0.904,  0.912,
        0.92 ,  0.928,  0.936,  0.944,  0.952,  0.96 ,  0.968,  0.976,
        0.984,  0.992,  1.   ])

default_dir = Path.cwd()
trf_dir = default_dir / 'data/eeg/journal/TRF'


def cluster_effect_size(az_data, ele_data, time, time_sel, cl):
    """
    Compute Cohen's dz and Hedges' gz for a given cluster.

    target_data, distractor_data : arrays (n_subjects, n_times)
    time : full time vector
    time_sel : the time points used in this component window
    cl : cluster indices relative to time_sel (from MNE)
    """
    # cluster indices relative to time_sel
    ti = cl[0]
    cluster_times = time_sel[ti]  # actual time values

    # build mask for the full time axis
    cluster_mask = np.isin(time, cluster_times)

    # subject-wise averages
    T_vals = az_data[:, cluster_mask].mean(axis=1)
    D_vals = ele_data[:, cluster_mask].mean(axis=1)
    delta = T_vals - D_vals

    # effect sizes
    mean_diff = delta.mean()
    sd_diff = delta.std(ddof=1)
    dz = mean_diff / sd_diff
    n = len(delta)
    J = 1 - (3 / (4*n - 9))
    gz = J * dz

    return mean_diff, dz, gz


def cluster_perm(predictor, stim_type):
    az = trf_dir / 'azimuth' / f'{predictor}_diff_wave_{stim_type}.npz'
    az_trfs = np.load(az, allow_pickle=True)
    az_trfs = az_trfs['diff_waves'].item()

    ele = trf_dir / 'elevation' / f'{predictor}_diff_wave_{stim_type}.npz'
    ele_trfs = np.load(ele, allow_pickle=True)
    ele_trfs = ele_trfs['diff_waves'].item()

    # stack into arrays (n_subjects, n_times)
    from mne.stats import fdr_correction
    az_data = np.vstack(list(az_trfs.values()))
    ele_data = np.vstack(list(ele_trfs.values()))

    # compute means/SEMs for plotting full time
    az_mean = az_data.mean(axis=0)
    ele_mean = ele_data.mean(axis=0)
    az_sem = az_data.std(axis=0) / np.sqrt(az_data.shape[0])
    ele_sem = ele_data.std(axis=0) / np.sqrt(ele_data.shape[0])

    # plot full responses
    plt.plot(time, az_mean, color='blue', linewidth=2, label='Azimuth')
    plt.fill_between(time, az_mean - az_sem, az_mean + az_sem,
                     color='blue', alpha=0.3)

    plt.plot(time, ele_mean, color='red', linewidth=2, label='Elevation')
    plt.fill_between(time, ele_mean - ele_sem, ele_mean + ele_sem,
                     color='red', alpha=0.3)

    all_pvals = []
    all_clusters = []
    all_labels = []
    all_times = []

    # loop windows
    for comp, (tmin, tmax) in component_windows.items():
        tmask = (time >= tmin) & (time <= tmax)
        if not tmask.any():
            continue
        time_sel = time[tmask]
        X = [az_data[:, tmask], ele_data[:, tmask]]
        T_obs, clusters, cluster_p_values, H0 = permutation_cluster_test(
            X, n_permutations=5000, tail=1, n_jobs=1)

        for cl, pval in zip(clusters, cluster_p_values):
            all_pvals.append(pval)
            all_labels.append(comp)
            all_clusters.append(cl)
            all_times.append(time_sel)

    # apply FDR once across all windows
    reject, pvals_fdr = fdr_correction(all_pvals, alpha=0.05)

    # highlight significant clusters after correction
    for comp, cl, pval, pval_corr, rej, time_sel in zip(
            all_labels, all_clusters, all_pvals, pvals_fdr, reject, all_times):
        if rej:
            ti = cl[0]  # time indices relative to time_sel
            mean_diff, dz, gz = cluster_effect_size(az_data, ele_data, time, time_sel, cl)
            plt.axvspan(time_sel[ti[0]], time_sel[ti[-1]],
                        color='gray', alpha=0.2)
            plt.axvline(x=0, color='gray', linestyle='--', linewidth=0.8, alpha=0.7)
            t_start, t_end = time_sel[ti[0]], time_sel[ti[-1]]
            print(f"{comp}: {t_start * 1000:.0f}-{t_end * 1000:.0f} ms, g={gz:.3f}, pFDR={pval_corr:.3f}")

    # plt.title(f'TRF Comparison - {plane} - {predictor}')
    plt.xlim([time[0], 0.6])
    if predictor == 'phonemes':
        plt.ylim([-0.6, 0.7])
    else:
        plt.ylim([-1, 1.5])
    plt.legend(loc='upper right')
    plt.xlabel('Time (s)')
    plt.ylabel('TRF amplitude (a.u.)')
    sns.despine(top=True, right=True)
    plt.show()


# phonemes
cluster_perm(predictor='phonemes', stim_type='all')
cluster_perm(predictor='phonemes', stim_type='non_targets')
cluster_perm(predictor='phonemes', stim_type='target_nums')

# envelopes
cluster_perm(predictor='envelopes', stim_type='all')
cluster_perm(predictor='envelopes', stim_type='non_targets')
cluster_perm(predictor='envelopes', stim_type='target_nums')