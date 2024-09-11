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


def block_sequence(target_number_seq):  # BOTH PLANES #
    # azimuth
    azimuth = ['azimuth']
    repetitions = 10  # 10 blocks total each axis
    azimuth = azimuth * repetitions
    # elevation
    elevation = ['elevation']
    elevation = elevation * repetitions
    block_seq_planes = azimuth + elevation

    # target streams
    streams = ['s1', 's2']
    streams = streams * 10

    block_seqs_df = pd.DataFrame({'block_seq': streams, 'block_condition': block_seq_planes})

    block_seqs_df['Voices'] = voice_names
    block_seqs_df['Target Number'] = target_number_seq
    block_seqs_df = block_seqs_df.sample(frac=1).reset_index(drop=True)
    return block_seqs_df


# so far, so good. All seems to work as expected
