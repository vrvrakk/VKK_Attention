import random
import numpy
import pandas as pd
import matplotlib.pyplot as plt

numbers = [1, 2, 3, 4, 5, 6, 8, 9]
isi = numpy.array((240, 180))
duration_s = 300  # 5 min total
stim_dur_ms = 745  # duration in ms
n_trials1 = int(numpy.floor((duration_s) / ((isi[0] + stim_dur_ms) / 1000)))
n_trials2 = int(numpy.floor((duration_s) / ((isi[1] + stim_dur_ms) / 1000)))

tlo1 = stim_dur_ms + isi[0]
tlo2 = stim_dur_ms + isi[1]

t1_total = tlo1 * n_trials1
t2_total = tlo2 * n_trials2

# create stimulus streams:

t1_timepoints = []
for t1 in range(0, t1_total, tlo1):
    t1_timepoints.append((t1, 's1'))
t1_df = pd.DataFrame(t1_timepoints, columns=['Timepoints', 'Stimulus Type'])

t2_timepoints = []
for t2 in range(0, t2_total, tlo2):
    t2_timepoints.append((t2, 's2'))
t2_df = pd.DataFrame(t2_timepoints, columns=['Timepoints', 'Stimulus Type'])

streams_df = pd.concat((t1_df, t2_df))
streams_df = streams_df.sort_values(by='Timepoints', ascending=True).reset_index(drop=True)
streams_df['Numbers'] = None

# rolling window:
random.shuffle(numbers)
used_numbers = set()
for index, row in streams_df.iterrows():
    window_start = row['Timepoints'] - tlo1
    window_end = row['Timepoints'] + tlo1

    window_data = streams_df[(streams_df['Timepoints'] >= window_start) & (streams_df['Timepoints'] <= window_end)]
    possible_numbers = [x for x in numbers if x not in window_data['Numbers'].tolist()]
    if possible_numbers:
        assigned_number = possible_numbers[0]
        streams_df.at[index, 'Numbers'] = assigned_number
        used_numbers.add(assigned_number)
        numbers.remove(assigned_number)
    else:
        if len(numbers) == 0:
            numbers = [1, 2, 3, 4, 5, 6, 8, 9]
            random.shuffle(numbers)
        possible_numbers = [x for x in numbers if x not in window_data['Numbers'].tolist()]
        assigned_number = possible_numbers[0]
        streams_df.at[index, 'Numbers'] = assigned_number
        used_numbers.add(assigned_number)
        numbers.remove(assigned_number)

plt.hist(streams_df['Numbers'].values)