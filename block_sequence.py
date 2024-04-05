import random
import numpy
import pandas as pd
from pathlib import Path


def block_sequence():
    # azimuth
    block_duration = 120  # 2 min
    conditions = ['s1', 's2']
    repetitions = 1  # 10 blocks total azimuth

    block_seq_azimuth = conditions * repetitions
    random.shuffle(block_seq_azimuth)

    # elevation
    block_seq_ele = conditions * repetitions
    random.shuffle(block_seq_ele)  # 10 blocks total elevation

    block_seqs = numpy.concatenate((block_seq_azimuth, block_seq_ele))

    block_conditions = ['azimuth', 'ele']
    repetitions = 2
    block_seq_conditions = block_conditions * repetitions
    random.shuffle(block_seq_conditions)

    block_seqs_df = pd.DataFrame({'block_seq': block_seqs, 'block_condition': block_seq_conditions})
    return block_seqs_df



