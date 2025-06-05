# === TFA on predicted EEG === #
import numpy as np
import matplotlib.pyplot as plt
import os
from pathlib import Path
import mne
import pandas


plane = 'azimuth'
folder_types = ['all_stims', 'non_targets', 'target_nums', 'deviants']
folder_type = folder_types[0]
predictions_dir = fr'C:\Users\pppar\PycharmProjects\VKK_Attention\data\eeg\trf\trf_testing\composite_model\single_sub\{plane}\{folder_type}\on_en_RT_ov\weights\predictions'

target_preds_dict = {}
distractor_preds_dict = {}
for pred_files in os.listdir(predictions_dir):
    if 'target_stream' in pred_files:
        target_predictions = np.load(os.path.join(predictions_dir, pred_files))
        sub = str(target_predictions['subject'])
        target_preds_dict[sub] = target_predictions['prediction'].squeeze()
    elif 'distractor_stream' in pred_files:
        distractor_predictions = np.load(os.path.join(predictions_dir, pred_files))
        sub = str(distractor_predictions['subject'])
        distractor_preds_dict[sub] = distractor_predictions['prediction'].squeeze()

# === Load relevant events and mask the bad segments === #
stream_events_dir = r'C:\Users\pppar\PycharmProjects\VKK_Attention\data\eeg\predictors\streams_events'

for sub_folders in os.listdir(stream_events_dir):
    for cond_folders in os.listdir(os.path.join(stream_events_dir, sub_folders)):
        if plane == 'azimuth':
            if 'a1' in cond_folders:
                for files1 in os.listdir(os.path.join(stream_events_dir, sub_folders, cond_folders)):
                    stream1_events =
                    # todo: load stream events and concat, and remove bad segmetns :S