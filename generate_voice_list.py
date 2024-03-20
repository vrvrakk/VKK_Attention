import random
from pathlib import Path
import datetime
data_path = Path.cwd() / 'data' / 'voices_padded'

# params:
datetime = datetime.datetime.now().date().strftime('%Y%m%d')
subject_id = 'vkk'
participant_id = datetime + '_' + subject_id

wav_list = []
for folder in data_path.iterdir():
    wav_list.append(list(folder.iterdir()))

voice_seq = []
voice_names = []
while len(voice_seq) < 20:
    # make sure wav_list is not empty
    if len(wav_list) == 0:
        wav_list = []
        for folder in data_path.iterdir():
            wav_list.append(list(folder.iterdir()))
    used_voice = []
    chosen_voice = random.choice(wav_list)
    chosen_voice_name = chosen_voice[0].parent.name
    voice_names.append(chosen_voice_name)
    used_voice.append(chosen_voice)
    voice_seq.append(chosen_voice)
    wav_list.remove(chosen_voice)


def save_voice_seq(voice_names):
    voice_dir = Path('C:/Users/vrvra/PycharmProjects/VKK_Attention/data/misc/voice_per_block.txt')
    voice_names = str(voice_names)
    with open(voice_dir, 'a') as file:
        file.write(participant_id)
        file.write(':')
        file.write(voice_names)
        file.write("\n----\n")
    return file

file = save_voice_seq(voice_names)