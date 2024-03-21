import random
from pathlib import Path

data_path = Path.cwd() / 'data' / 'voices_padded'

wav_list = []
for folder in data_path.iterdir():
    wav_list.append(list(folder.iterdir()))

voice_seq = []
while len(voice_seq) < 20:
    # make sure wav_list is not empty
    if len(wav_list) == 0:
        wav_list = []
        for folder in data_path.iterdir():
            wav_list.append(list(folder.iterdir()))
    used_voice = []
    voice = random.choice(wav_list)
    used_voice.append(voice)
    voice_seq.append(voice)
    wav_list.remove(voice)

