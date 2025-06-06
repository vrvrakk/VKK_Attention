import os
import numpy as np
from pathlib import Path
import os
from scipy.stats import wilcoxon, ttest_rel, shapiro
import pandas as pd

def load_itc_files(save_dir, stream):
    """
    Loads all .npy ITC files from the given directory that match folder_type, plane, and stream.

    Parameters:
        save_dir (str): Path to the main save directory.
        folder_type (str): Subfolder category (e.g., 'tfr_data').
        plane (str): Spatial plane (e.g., 'a1' or 'e2').
        stream (str): Stream identity (e.g., 'S1' or 'S2').

    Returns:
        itc_data_list (list of np.ndarray): List of ITC arrays from all matching files.
        subjects (list of str): Corresponding subject IDs extracted from filenames.
    """
    folder_path = save_dir
    itc_data_list = []
    subjects = []

    for fname in os.listdir(folder_path):
        if 'npy' in fname:
            print(fname)
            sub_id = fname.split('_')[0]  # assumes filename starts with e.g., 'sub-01'
            full_path = os.path.join(folder_path, fname)
            data = np.load(full_path)
            itc_data_list.append(data)
            subjects.append(sub_id)

    return itc_data_list, subjects






if __name__ == '__main__':

    plane = 'elevation'
    folder_types = ['all_stims', 'non_targets', 'target_nums', 'deviants']
    folder_type  = folder_types[0]
    default_path = Path.cwd()
    save_dir = default_path / 'data' / f'eeg/trf/trf_comparison/{plane}/{folder_type}/ITC'

    itc_list, sub_ids = load_itc_files(save_dir, stream='target')
    print(f"Loaded ITC data from {len(itc_list)} subjects.")

    # todo: correlated with RT speed and performance of each sub
