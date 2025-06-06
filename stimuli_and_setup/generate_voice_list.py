import random
from pathlib import Path
data_path = Path.cwd() / 'data' / 'voices_english'


wav_list = []
for folder in data_path.iterdir():
    if 'downsampled' not in str(folder):
        wav_list.append(list(folder.iterdir()))


voice_seq = []
while len(voice_seq) < 20:
    # make sure wav_list is not empty
    if len(wav_list) == 0:
        wav_list = []
        for folder in data_path.iterdir():
            if 'downsampled' not in str(folder):
                wav_list.append([f for f in folder.iterdir() if 'downsampled' not in str(f)])
    used_voice = []
    voice = random.choice(wav_list)
    used_voice.append(voice)
    voice_seq.append(voice)
    wav_list.remove(voice)

voice_names = []
n = 7  # 5 or 7
for voices in voice_seq:
    # print(voices)
    voice = voices[0]
    if 'voice1' in voice.parts[n]:
        voice_names.append(str('voice1'))
    elif 'voice2' in voice.parts[n]:
        voice_names.append(str('voice2'))
    elif 'voice3' in voice.parts[n]:
        voice_names.append(str('voice3'))
    elif 'voice4' in voice.parts[n]:
        voice_names.append(str('voice4'))

# alternative method:
'''voice_mapping = {
    'voice1': 'voice1',
    'voice2': 'voice2',
    'voice3': 'voice3',
    'voice4': 'voice4'
}

# One-liner to map voices to their string names
voice_names = [voice_mapping[part] for voices in voice_seq for part in voices[0].parts if part in voice_mapping]
'''