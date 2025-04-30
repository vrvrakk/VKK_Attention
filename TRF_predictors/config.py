from pathlib import Path

sub = 'sub08'
condition = 'a1'
sfreq = 125
stim_dur = 0.745

if condition in ['a1', 'e1']:
    stream1_label = 'target_stream'
    stream2_label = 'distractor_stream'
elif condition in ['a2', 'e2']:
    stream2_label = 'target_stream'
    stream1_label = 'distractor_stream'

default_path = Path.cwd()
results_path = default_path / 'data/eeg/preprocessed/results'
events_path = default_path / 'data/eeg/predictors/streams_events'
params_path = default_path / 'data' / 'params'
voices_path = default_path / 'data' / 'voices_english' / 'downsampled'
predictors_path = default_path / 'data/eeg/predictors'


if condition in ['a1', 'e1']:
    base_target = 's1'
    base_distractor = 's2'
else:
    base_target = 's2'
    base_distractor = 's1'

# assign binary predictors:
stream1_nums = {1, 2, 3, 4, 5, 6, 7, 8, 9}
stream2_nums = {65: 1, 66: 2, 67: 3, 68: 4, 69: 5, 70: 6, 71: 7, 72: 8, 73: 9}
response_nums = {129: 1, 130: 2, 131: 3, 132: 4, 133: 5, 134: 6, 136: 8, 137: 9}