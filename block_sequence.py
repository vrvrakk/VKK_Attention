import random
import pandas as pd
from generate_voice_list import voice_names


def block_sequence():
    # azimuth
    block_duration = 120  # 2 min
    target_conditions = ['s1', 's2']
    repetitions = 5  # 10 blocks total azimuth

    block_seq_azimuth = target_conditions * repetitions
    random.shuffle(block_seq_azimuth)

    # elevation
    block_seq_ele = target_conditions * repetitions
    random.shuffle(block_seq_ele)  # 10 blocks total elevation

    block_seqs = block_seq_azimuth + block_seq_ele

    plane_conditions = ['azimuth', 'ele']
    repetitions = 10
    block_seq_conditions = plane_conditions * repetitions
    random.shuffle(block_seq_conditions)

    block_seqs_df = pd.DataFrame({'block_seq': block_seqs, 'block_condition': block_seq_conditions})
    # todo: fixed coordinates for each stream

    block_seqs_df['Voices'] = voice_names

    return block_seqs_df


# save sequence:

