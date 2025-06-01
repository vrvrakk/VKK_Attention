import matplotlib
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from mtrf import TRF
from mtrf.stats import crossval
from tqdm import tqdm
matplotlib.use('TkAgg')

default_path = Path.cwd()


plane = 'azimuth'
cond1 = 'a1'
cond2 = 'a2'
predictors = ['envelopes', 'binary_weights', 'overlap_ratios'] # removed proxies
preds =['envelopes', 'onsets', 'overlap_ratios']

def get_preds(cond1='', cond2='', stim_type=''):
    pred_dict = {}
    for pred, predictor in zip(preds, predictors):
        pred_list1 = []
        pred_list2 = []
        predictor_path = default_path / f'data/eeg/predictors/{predictor}'
        for folders in predictor_path.iterdir():
            if 'sub' in folders.name:
                for sub_folders in folders.iterdir():
                    if cond1 in sub_folders.name:
                        for files in sub_folders.iterdir():
                            if stim_type in files.name:
                                for pred_files1 in files.iterdir():
                                    if 'concat' in pred_files1.name:
                                        pred_array1 = np.load(pred_files1)
                                        pred_array1 = pred_array1[pred]
                                        pred_list1.append(pred_array1)
                    if cond2 in sub_folders.name:
                        for files in sub_folders.iterdir():
                            if stim_type in files.name:
                                for pred_files2 in files.iterdir():
                                    if 'concat'  in pred_files2.name:
                                        pred_array2 = np.load(pred_files2)
                                        pred_array2 = pred_array2[pred]
                                        pred_list2.append(pred_array2)
        pred_dict[pred] = (pred_list1, pred_list2)
    return pred_dict

# Get predictor dictionaries for each stimulus type
preds_targets = get_preds(cond1='a1', cond2='a2', stim_type='targets')
preds_distractors = get_preds(cond1='a1', cond2='a2', stim_type='distractors')
preds_deviants = get_preds(cond1='a1', cond2='a2', stim_type='deviants')
preds_nt_target = get_preds(cond1='a1', cond2='a2', stim_type='nt_target')
preds_nt_distractor = get_preds(cond1='a1', cond2='a2', stim_type='nt_distractor')

def concat_arrays(pred_list1, pred_list2):
    pred_list1_concat = np.concatenate(pred_list1, axis=0)
    pred_list2_concat = np.concatenate(pred_list2, axis=0)
    pred_list_concat = np.concatenate((pred_list1_concat, pred_list2_concat))
    return pred_list_concat

def pred_concat_arrays():
    concat_dict = {}
    for pred in preds:
        concat_dict[f'{pred}_targets'] = concat_arrays(preds_targets[pred][0], preds_targets[pred][1])
        concat_dict[f'{pred}_distractors'] = concat_arrays(preds_distractors[pred][0], preds_distractors[pred][1])
        concat_dict[f'{pred}_deviants'] = concat_arrays(preds_deviants[pred][0], preds_deviants[pred][1])
        concat_dict[f'{pred}_nt_target'] = concat_arrays(preds_nt_target[pred][0], preds_nt_target[pred][1])
        concat_dict[f'{pred}_nt_distractor'] = concat_arrays(preds_nt_distractor[pred][0], preds_nt_distractor[pred][1])
    return concat_dict

concat_preds = pred_concat_arrays()


# load eeg file
eeg_path = default_path / f'data/eeg/trf/model_inputs/{plane}_raw/{plane}_raw_eeg_all.npy'
eeg = np.load(eeg_path)

# load bad segments:
def get_bads(cond1, cond2):
    bad_arrays_list1 = []
    bad_arrays_list2 = []
    bad_segments = default_path/'data/eeg/predictors/bad_segments'
    for folders in bad_segments.iterdir():
        if 'sub' in folders.name:
            for sub_folders in folders.iterdir():
                if cond1 in sub_folders.name:
                    for files1 in sub_folders.iterdir():
                        if 'concat.npy' in files1.name:
                            bad_array1 = np.load(files1)
                            bad_array1 = bad_array1['bad_series']
                            bad_arrays_list1.append(bad_array1)
                if cond2 in sub_folders.name:
                    for files2 in sub_folders.iterdir():
                        if 'concat.npy' in files2.name:
                            bad_array2 = np.load(files2)
                            bad_array2 = bad_array2['bad_series']
                            bad_arrays_list2.append(bad_array2)
    return bad_arrays_list1, bad_arrays_list2

bad_arrays_list1, bad_arrays_list2 = get_bads(cond1=cond1, cond2=cond2)
bad_arrays_list1_concat = np.concatenate(bad_arrays_list1, axis=0)
bad_arrays_list2_concat = np.concatenate(bad_arrays_list2, axis=0)
bads_concat = np.concatenate((bad_arrays_list1_concat, bad_arrays_list2_concat), axis=0)

# mask preds:
good_samples = bads_concat == 0

masked_preds = {}
for key, arr in concat_preds.items():
    if arr.shape[0] == good_samples.shape[0]:
        masked_preds[key] = arr[good_samples]
    else:
        print(f"⚠️ Skipping {key} due to mismatched shape: {arr.shape} vs {good_samples.shape}")


import pandas as pd

preds_stacked = pd.DataFrame({k: v for k, v in masked_preds.items()})


eeg_T = eeg.T
fs = 125

trf = TRF(direction=1)

n_samples = fs * 60
total_samples = len(preds_stacked.values)
n_folds = total_samples // n_samples
# Split predictors and EEG into subject chunks
X_folds = np.array_split(preds_stacked.values, n_folds)
Y_folds = np.array_split(eeg_T, n_folds)

trf.train(X_folds, Y_folds, fs=fs, tmin=-0.1, tmax=1.0, regularization=1.0, seed=42)


predictions = []
r_vals = []

for i in tqdm(range(n_folds)):
    start = i * n_samples
    end = start + n_samples
    X_chunk = preds_stacked.values[start:end]
    Y_chunk = eeg_T[start:end]

    pred_chunk, r_chunk = trf.predict(X_chunk, Y_chunk)
    predictions.append(pred_chunk)
    r_vals.append(r_chunk)

r = np.mean(r_vals)

n_total = len(X_folds)
n_split = n_total // 5  # or any chunk size you want

X_subsample = X_folds[:n_split]
Y_subsample = Y_folds[:n_split]
X_subsample2 = X_folds[n_split:2 * n_split]
Y_subsample2 = Y_folds[n_split:2 * n_split]
X_subsample3 = X_folds[2 * n_split:]
Y_subsample3 = Y_folds[2 * n_split:]
X_subsamples = [X_subsample, X_subsample2, X_subsample3]
Y_subsamples = [Y_subsample, Y_subsample2, Y_subsample3]

r_crossvals = []
for x_subsamples, y_subsamples in zip(X_subsamples, Y_subsamples):
    r_crossval = crossval(
        trf,
        x_subsamples,
        y_subsamples,
        fs=fs,
        tmin=-0.1,
        tmax=1.0,
        regularization=1.0
    )
    r_crossvals.append(r_crossval)

r_crossval_mean = np.mean(r_crossvals)
print("Avg r_crossval across all chunks:", np.round(r_crossval_mean, 3))

weights = trf.weights  # shape: (n_features, n_lags, n_channels)
time_lags = np.linspace(-0.1, 1.0, weights.shape[1])  # time axis

# Loop and plot
# Define your lag window of interest
tmin_plot = 0.0
tmax_plot = 1.0

# Create a mask for valid time lags
lag_mask = (time_lags >= tmin_plot) & (time_lags <= tmax_plot)
time_lags_trimmed = time_lags[lag_mask]

trf_preds = list(preds_stacked.columns)

for i, name in enumerate(trf_preds):
    filename = name
    plt.ion()
    plt.figure(figsize=(8, 4))

    trf_weights = weights[i].T[:, lag_mask]  # shape: (n_channels, n_lags_selected)

    # Smooth for aesthetics
    window_len = 11
    hamming_win = np.hamming(window_len)
    hamming_win /= hamming_win.sum()
    smoothed_weights = np.array([
        np.convolve(trf_weights[ch], hamming_win, mode='same')
        for ch in range(trf_weights.shape[0])
    ])

    # Plot per channel
    for ch in range(smoothed_weights.shape[0]):
        plt.plot(time_lags_trimmed, smoothed_weights[ch], alpha=0.4)

    plt.title(f'TRF for {name}')
    plt.xlabel('Time lag (s)')
    plt.ylabel('Amplitude (a.u.)')
    plt.plot([], [], ' ', label=f'λ = 1.0, r = {r_crossval_mean:.2f}')
    plt.legend(loc='upper right', fontsize=8)
    plt.tight_layout()

    # # Save and show
    # plt.savefig(save_path / f'{filename}.png', dpi=300)
    plt.show()



