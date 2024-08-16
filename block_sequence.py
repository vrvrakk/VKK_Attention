import random
import pandas as pd
from generate_voice_list import voice_names


def get_target_number_seq():
    target_number_seq = []
    numbers = [1, 2, 3, 4, 5, 6, 8, 9]
    while len(target_number_seq) < 20:
        # make sure wav_list is not empty
        if len(numbers) == 0:
            numbers = [1, 2, 3, 4, 5, 6, 8, 9]
        used_number = []
        target_number = random.choice(numbers)
        used_number.append(target_number)
        target_number_seq.append(target_number)
        numbers.remove(target_number)

    return target_number_seq

def block_sequence(target_number_seq):  # ONLY AZIMUTH FOR NOW #
    # azimuth
    target_conditions = ['s1', 's2']
    repetitions = 10  # 10 blocks total each axis

    block_seq_azimuth = target_conditions * repetitions
    random.shuffle(block_seq_azimuth)

    # elevation
    block_seq_ele = target_conditions * repetitions
    random.shuffle(block_seq_ele)  # 10 blocks total elevation

    block_seqs = block_seq_azimuth #+ block_seq_ele

    plane_conditions = ['azimuth', 'ele']
    repetitions = 10
    block_seq_conditions = plane_conditions * repetitions
    random.shuffle(block_seq_conditions)

    block_seqs_df = pd.DataFrame({'block_seq': block_seqs, 'block_condition': block_seq_conditions})

    block_seqs_df['Voices'] = voice_names
    block_seqs_df['Target Number'] = target_number_seq

    return block_seqs_df



# so far, so good. All seems to work as expected
