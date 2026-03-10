"""
Plot TRF topomaps using ONLY significant cluster time ranges.

Conditions included:

WITHIN elevation (frontocentral exploratory ROI):
    phonemes:
        target_numbers: 96–144 ms, 152–200 ms, 280–336 ms
        non_targets:    56–112 ms, 304–344 ms, 352–496 ms

ACROSS plane:
    envelopes (main ROI), non_targets:
        128–144 ms, 152–184 ms, 256–280 ms

    phonemes (frontocentral exploratory ROI), non_targets:
        56–144 ms, 360–480 ms

Topomaps plotted using ALL electrodes.

Author: You
"""

import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from pathlib import Path
import mne

from mtrf import TRF

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 8


# =============================================================================
# USER SETTINGS
# =============================================================================

sfreq = 125
tmin = -0.1
tmax = 1.0
best_lambda = 0.01

base_dir = Path.cwd()
data_dir = base_dir / 'data' / 'eeg'


# =============================================================================
# SIGNIFICANT TIME WINDOWS (SECONDS)
# =============================================================================

SIGNIFICANT_WINDOWS = {

    "within_elevation": {

        "phonemes_target_numbers": [
            (0.096, 0.200),
            (0.280, 0.336)
        ],

        "phonemes_non_targets": [
            (0.056, 0.112),
            (0.304, 0.496)
        ]
    },

    "across_plane": {

        "envelopes_non_targets": [
            (0.128, 0.144),
            (0.152, 0.184),
            (0.256, 0.280)
        ],

        "phonemes_non_targets": [
            (0.056, 0.144),
            (0.360, 0.480)
        ]
    }
}


# =============================================================================
# ELECTRODES (ALL)
# =============================================================================

ALL_CHANNELS = np.array([
    'Fp1','Fp2','F7','F3','Fz','F4','F8',
    'FC5','FC1','FC2','FC6','T7','C3','Cz',
    'C4','T8','TP9','CP5','CP1','CP2','CP6',
    'TP10','P7','P3','Pz','P4','P8','PO9','O1',
    'Oz','O2','PO10','AF7','AF3','AF4','AF8','F5',
    'F1','F2','F6','FT9','FT7','FC3','FC4','FT8',
    'FT10','C5','C1','C2','C6','TP7','CP3','CPz',
    'CP4','TP8','P5','P1','P2','P6','PO7','PO3',
    'POz','PO4','PO8','FCz'
])


# =============================================================================
# HELPERS
# =============================================================================

def create_info():

    info = mne.create_info(
        ch_names=list(ALL_CHANNELS),
        sfreq=sfreq,
        ch_types='eeg'
    )

    montage = mne.channels.make_standard_montage('standard_1020')
    info.set_montage(montage)

    return info


def run_trf(X_list, Y_list, sub_list):

    results = {}

    for sub, X, Y in zip(sub_list, X_list, Y_list):

        trf = TRF(direction=1, method='ridge')

        trf.train(
            stimulus=X,
            response=Y,
            fs=sfreq,
            tmin=tmin,
            tmax=tmax,
            regularization=best_lambda,
            average=True
        )

        results[sub] = trf.weights

    return trf.times, results


def compute_difference(target_weights, distractor_weights, predictor_index):

    diffs = []

    for sub in target_weights.keys():

        t = target_weights[sub][predictor_index]
        d = distractor_weights[sub][predictor_index]

        diffs.append(t - d)

    return np.stack(diffs)


def plot_topomap(mean_map, info, save_path, title):

    fig, ax = plt.subplots(figsize=(4,4))

    im, _ = mne.viz.plot_topomap(
        mean_map,
        info,
        axes=ax,
        cmap="magma",
        contours=0,
        show=False
    )

    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("ΔTRF amplitude (a.u.)")

    plt.tight_layout()

    plt.savefig(save_path.with_suffix(".png"), dpi=300)
    plt.savefig(save_path.with_suffix(".pdf"), dpi=300)

    print("Saved:", save_path)

    plt.close()


def extract_window(times, diffs, window):

    t1, t2 = window

    idx = np.where((times >= t1) & (times <= t2))[0]

    mean_sub = diffs[:, idx, :].mean(axis=1)

    return mean_sub.mean(axis=0)


# =============================================================================
# LOAD DATA
# =============================================================================

def load_condition(condition, stim_type="all"):

    dict_dir = data_dir / "journal" / "TRF" / "matrix" / condition / stim_type

    with open(dict_dir / f"{condition}_matrix_target.pkl","rb") as f:
        target = pkl.load(f)

    with open(dict_dir / f"{condition}_matrix_distractor.pkl","rb") as f:
        distractor = pkl.load(f)

    sub_list = list(target.keys())

    X_target = []
    X_distractor = []
    Y = []

    for sub in sub_list:

        X_target.append(
            np.column_stack([
                target[sub]["envelopes"],
                target[sub]["phonemes"],
                target[sub]["responses"]
            ])
        )

        X_distractor.append(
            np.column_stack([
                distractor[sub]["envelopes"],
                distractor[sub]["phonemes"],
                distractor[sub]["responses"]
            ])
        )

        Y.append(target[sub]["eeg"])

    return sub_list, X_target, X_distractor, Y


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    info = create_info()

    save_dir = data_dir / "journal" / "figures" / "TRF" / "topomap" / "significant_clusters"
    save_dir.mkdir(parents=True, exist_ok=True)


    # ============================================================
    # WITHIN elevation
    # ============================================================

    sub_list, Xt_e1, Xd_e1, Y_e1 = load_condition("e1")
    _, Xt_e2, Xd_e2, Y_e2 = load_condition("e2")

    Xt = [np.concatenate([a,b]) for a,b in zip(Xt_e1, Xt_e2)]
    Xd = [np.concatenate([a,b]) for a,b in zip(Xd_e1, Xd_e2)]
    Y  = [np.concatenate([a,b]) for a,b in zip(Y_e1, Y_e2)]

    times, target_weights = run_trf(Xt,Y,sub_list)
    _, distractor_weights = run_trf(Xd,Y,sub_list)


    phoneme_diff = compute_difference(target_weights, distractor_weights, predictor_index=1)


    for stim_type, windows in SIGNIFICANT_WINDOWS["within_elevation"].items():

        for window in windows:

            mean_map = extract_window(times, phoneme_diff, window)

            fname = f"phonemes_within_elevation_{stim_type}_{int(window[0]*1000)}_{int(window[1]*1000)}ms"

            plot_topomap(
                mean_map,
                info,
                save_dir / fname,
                fname
            )


    # ============================================================
    # ACROSS plane
    # ============================================================

    sub_list, Xt_a1, Xd_a1, Y_a1 = load_condition("a1")
    _, Xt_a2, Xd_a2, Y_a2 = load_condition("a2")

    Xt = [np.concatenate([a,b]) for a,b in zip(Xt_a1, Xt_a2)]
    Xd = [np.concatenate([a,b]) for a,b in zip(Xd_a1, Xd_a2)]
    Y  = [np.concatenate([a,b]) for a,b in zip(Y_a1, Y_a2)]

    times, target_weights = run_trf(Xt,Y,sub_list)
    _, distractor_weights = run_trf(Xd,Y,sub_list)


    envelope_diff = compute_difference(target_weights, distractor_weights, predictor_index=0)
    phoneme_diff  = compute_difference(target_weights, distractor_weights, predictor_index=1)


    for window in SIGNIFICANT_WINDOWS["across_plane"]["envelopes_non_targets"]:

        mean_map = extract_window(times, envelope_diff, window)

        fname = f"envelopes_across_plane_non_targets_{int(window[0]*1000)}_{int(window[1]*1000)}ms"

        plot_topomap(info=info, mean_map=mean_map, save_path=save_dir/fname, title=fname)


    for window in SIGNIFICANT_WINDOWS["across_plane"]["phonemes_non_targets"]:

        mean_map = extract_window(times, phoneme_diff, window)

        fname = f"phonemes_across_plane_non_targets_{int(window[0]*1000)}_{int(window[1]*1000)}ms"

        plot_topomap(info=info, mean_map=mean_map, save_path=save_dir/fname, title=fname)


print("\nDONE.")


