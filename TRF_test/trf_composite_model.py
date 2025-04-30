from pathlib import Path
import os
import mne
import numpy as np
import pandas as pd
import mtrf
from mtrf import TRF

default_path = Path.cwd()
sub = 'sub10'
condition = 'a1'
predictors_list = ['binary_weights', 'envelopes', 'events_proximity', 'overlap_ratios']
predictors_path = default_path / 'data' / 'eeg' / 'predictors'
stim_type = 'stream1'


all_predictors = {}
for current_predictor in predictors_list:
    predictor_path = predictors_path / current_predictor / sub / condition / stim_type
    for files in predictor_path.iterdir():
        if 'concat' in files.name:
            predictor_loaded = np.load(files)
            predictor_keys = list(predictor_loaded.keys())
            if current_predictor == 'events_proximity':
                current_key = predictor_keys[0]

                predictor_array_pre = predictor_loaded[predictor_keys[0]]
                predictor_array_post = predictor_loaded[predictor_keys[1]]
            else:
                predictor_array = predictor_loaded[predictor_keys[0]]
        all_predictors[current_predictor] = predictor_loaded
