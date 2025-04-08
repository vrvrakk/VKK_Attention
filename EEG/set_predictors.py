'''
TRF predictors:
1. load eeg files
2. load events
3. extract predictors: onsets, envelopes
4. set binary predictors: (1) target stream, target stimulus, distractor stimulus, deviant sound
                          (0) distractor stream
5. add weights to binary predictors/features: target stream > distractor stream
                                              target stimulus > distractor stimulus > deviant sound
                                              non-target stimuli target > non-target stimuli distractor
'''

# import libraries:
import os
from pathlib import Path
import pandas as pd
import numpy as np
import mtrf
import mne
import copy
from params import sub_list, response_mapping, actual_mapping, stimuli_dict, conditions, event_types, exceptions, conditions
# import eelbrain
from EEG.preprocessing_eeg import single_eeg_path

class TRF:
    def __init__(self):
        self.param1 = None
        self.param2 = None
        self.param3 = None
    def load_events(self):


    def load_eeg(self):

    def extract_onsets(self):

    def extract_envelopes(self):

    def set_binary_predictors(self):






