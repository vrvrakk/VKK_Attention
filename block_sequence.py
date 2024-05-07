import random
import pandas as pd
from generate_voice_list import voice_names

speakers_coordinates = (17.5, 0)  # directions for each streams
azimuth = ((speakers_coordinates[0], 0), (speakers_coordinates[1], 0))
elevation = ((speakers_coordinates[1], -37.5), (speakers_coordinates[1], -12.5))

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
    block_seqs_df['Target Coordinates'] = None
    block_seqs_df['Distractor Coordinates'] = None

    for index, row in block_seqs_df.iterrows():
        if row['block_condition'] == 'azimuth':
            target_coordinates = random.choice(azimuth)
            distractor_coordinates = [coord for coord in azimuth if coord != target_coordinates][0]
        else:  # elevation block condition
            target_coordinates = random.choice(elevation)
            distractor_coordinates = [coord for coord in elevation if coord != target_coordinates][0]

        block_seqs_df.at[index, 'Target Coordinates'] = target_coordinates
        block_seqs_df.at[index, 'Distractor Coordinates'] = distractor_coordinates
    block_seqs_df['Voices'] = voice_names

    return block_seqs_df


# save sequence:

