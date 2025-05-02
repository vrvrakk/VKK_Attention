import numpy
import os
import random
import pandas
from analysis.helper import normalize_within_bin
from mtrf.model import TRF
from mtrf.stats import pearsonr
import pickle

"""
Using this script we are going to determine the lambda, by drawing random 20 participants across trials and combine the data together, to compute the lambda value for the further analysis.
"""

# directories

DIR = os.getcwd()
DATA_DIR = f'{DIR}/data/combined_data'
ANALYSIS_DIR = f'{DIR}/analysis'

# bin edges
with open(f'{ANALYSIS_DIR}/variables/bin_edges.pkl', 'rb') as f:
    bin_edges = pickle.load(f)

# randomly select 20 trials for the analysis

subjects = [f for f in os.listdir(DATA_DIR) if 'sub' in f]

all_trials = []

for subject in subjects:
    subject_trials = [f for f in os.listdir(f'{DATA_DIR}/{subject}') if 'sub' in f]
    for trial in subject_trials:
        all_trials.append(f'{subject}/{trial}')

random.seed(42)
selected_trials = random.sample(all_trials, 20)

# Store all trials separately
stimulus = []
response = []

for idx, trial in enumerate(selected_trials):

    print('|||||||||||||||||||')
    print(f'Trial: {idx + 1} / {len(selected_trials)}')
    print(trial)
    print('|||||||||||||||||||')

    subject, stim = trial.split('/')

    data = pandas.read_csv(f'{DATA_DIR}/{subject}/{stim}', index_col=0)
    start_time = 2
    end_time = data.index[-1] - 1
    data = data.loc[start_time:end_time]

    # Extract EEG channels
    eeg_channels = [col for col in data.columns if col.startswith(('Fp', 'F', 'T', 'C', 'P', 'O', 'AF', 'PO')) and col not in ['Prob', 'FCz']]
    print(len(eeg_channels))

    # Extract relevant columns
    envelope = data['envelope'].values
    loc_change = data['loc_change'].values
    H = data['H'].values

    binned_H = numpy.digitize(H, bin_edges, right=True)

    H_df = pandas.DataFrame({'H': H, 'bin': binned_H})
    H_df['normalized_H'] = H_df.groupby('bin')['H'].transform(normalize_within_bin)
    H_df.loc[H_df['bin'] != 0, 'normalized_H'] = H_df['normalized_H'].replace(0, 0.001)

    # Prepare stimulus features
    temp = H_df.pivot(columns='bin', values='normalized_H').fillna(0)
    stim_H = temp.iloc[:, 1:].to_numpy()

    resp = data[eeg_channels].values

    # Ensure stim_H and resp are the same length
    min_length = min(stim_H.shape[0], resp.shape[0])
    stim_H = stim_H[:min_length, :]
    resp = resp[:min_length, :]
    print(stim_H.shape, resp.shape)

    # Pack stimulus and response for cross-validation
    stim_combined = numpy.column_stack([envelope[:min_length], loc_change[:min_length], stim_H])
    stimulus.append(stim_combined)
    response.append(resp)

# --------- TRF TRAINING ---------
fs = 500  # Sampling rate
tmin = -0.1
tmax = 0.5
nfold = 6  # Number of folds for cross-validation
regularization = numpy.logspace(-6, 2, 20)

# ----- TRF MODEL for optimization ---------

trf = TRF(metric=pearsonr)

r = trf.train(stimulus, response, fs, tmin, tmax, regularization)

print(regularization[numpy.where(r == r.max())]) # 0.04281332
lbd = regularization[numpy.where(r == r.max())]

with open(f'{ANALYSIS_DIR}/variables/lambda.pkl', 'wb') as f:
    pickle.dump(lbd, f)
