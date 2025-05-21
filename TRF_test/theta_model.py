import os
from pathlib import Path
import numpy as np
import pandas as pd

predictors_list = ['binary_weights', 'envelopes', 'overlap_ratios', 'events_proximity', 'RTs']
pred_type = 'scores'
stream_type1 = 'stream1'
stream_type2 = 'stream2'
default_path = Path.cwd()

def get_scores_dict(plane, stream_type1, stream_type2):
    scores_dict = {}
    for predictor_name in predictors_list:
        if 'RTs_motor' in predictor_name:
            continue
        save_path = default_path / f'data/eeg/trf/trf_testing/{predictor_name}/{plane}/data'
        for files in save_path.iterdir():
            if f'{stream_type1}_{stream_type2}' in files.name:
                print(files)
                pred_scores = np.load(files, allow_pickle=True)
                pred_dict = pred_scores['scores']
                scores_dict[predictor_name] = pred_dict
    # convert to DataFrame for better handling
    scores_df = pd.DataFrame()
    for pred_key, scores in scores_dict.items():
        scores = scores.item()
        scores_df[pred_key] = pd.Series(scores)
    return scores_df


scores_df_azimuth = get_scores_dict(plane='azimuth', stream_type1=stream_type1, stream_type2=stream_type2)
scores_df_azimuth['mean_r'] = scores_df_azimuth.mean(axis=1)

scores_df_elevation = get_scores_dict(plane='elevation', stream_type1=stream_type1, stream_type2=stream_type2)
scores_df_elevation['mean_r'] = scores_df_elevation.mean(axis=1)


import matplotlib.pyplot as plt

def visualize_scores(scores_df, plane):
    # X-axis: lambda values (from index)
    lambdas = scores_df.index.values
    # Y-axis: mean r across predictors
    mean_r = scores_df['mean_r'].values

    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, mean_r, marker='o')
    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('Mean r across predictors')
    plt.title(f'Global Lambda Optimization – {plane}')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

visualize_scores(scores_df_azimuth, plane='azimuth')
visualize_scores(scores_df_elevation, plane='elevation')


def visualize_pred_scores(scores_df, plane):

    # Remove 'mean_r' column for individual predictors
    predictor_scores = scores_df.drop(columns='mean_r')

    # Plot each predictor's r-values across lambdas
    plt.figure(figsize=(10, 6))

    for predictor in predictor_scores.columns:
        plt.plot(predictor_scores.index, predictor_scores[predictor], marker='o', label=predictor)

    plt.xscale('log')
    plt.xlabel('Lambda (log scale)')
    plt.ylabel('r-value')
    plt.title(f'TRF performance per predictor – {plane}')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

visualize_pred_scores(scores_df_azimuth, plane='azimuth')
visualize_pred_scores(scores_df_elevation, plane='elevation')