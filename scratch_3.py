
import tdt
from win32com.client import Dispatch
import slab
import numpy
import os
import math
from pathlib import Path
sample_freq = 48828
data_path = Path('C:/Users/vrvra/Desktop/attention decoding/voice_recordings_digits_slab_precomputed')

def init_proc():
    proc = Dispatch('RPco.X')
    connected = proc.ConnectRM1('USB', 1)  # connect processor
    proc.ClearCOF() # remove previous program from processor
    proc.LoadCOF('C:/Users/vrvra/Desktop/attention decoding/test2.rcx')  # load target program
    if connected and proc.Run():
        print('connected and running')
    return proc


def load_experiment(voice_idx, n_trials, isi_1, isi_2):
    wav_folders = [folder for folder in os.listdir(data_path)]
    numbers = [1, 2, 3, 4, 5, 6, 8, 9]
    trial_seq = slab.Trialsequence(conditions=[1, 2], n_reps=12)
    wav_folder = wav_folders[voice_idx]
    wav_files = [file for file in os.listdir(data_path / wav_folder) if file.endswith('.wav')]
    n_samples = []
    for number, file in zip(numbers, wav_files): #combine lists into a single iterable->elements from corresponding positions are paired together
        #  print(f'{number} {file}')
        file_path = data_path / wav_folder / file # create file path with the corresponding wav file name
        if file_path.exists():
            s = slab.Binaural(data=file_path)
            s = s.resample(48828)
            n_samples.append(s.n_samples) # places the sound event  duration vals in n_samples list
            proc.WriteTagV(f'{number}', 0, s.data[:, 0]) # loads array on buffer
            proc.SetTagVal(f'{number}_n_samples', s.n_samples) # sets total buffer size according to numeration
    mean_n_samples = int(numpy.mean(n_samples)) # get n_samples mean
    proc.SetTagVal('n_trials', trial_seq.n_trials)
    tlo = isi_1 + int(mean_n_samples / sample_freq * 1000)  # in ms tlo arg for pulse train
    print(proc.SetTagVal('isi', tlo))
    sequence = numpy.array(trial_seq.trials).astype('int32')
    sequence = numpy.append(0, sequence)
    proc.WriteTagV('trial_seq', 0, sequence)
    proc.WriteTagV('trial_seq', 0, numpy.tile(numpy.array((1, 2, 3, 4 ,5 ,6 , 8, 9)), 5))

    # proc.ReadTagV('trial_seq', 0, len(trial_seq.trials))

    # Run

if __name__ == "__main__":
    proc = init_proc()
    voice_idx = 0
    n_trials = 96
    isi_1 = 1000
    isi_2 = 0
    load_experiment(voice_idx, n_trials, isi_1, isi_2)
    proc.SoftTrg(1)  # buffer trigger (read and play stim)

"""


# old approach


for voice in dir_path_list:
    signals=[] # create empty list for all wav files of each folder
    wav_files = [file for file in os.listdir(voice) if file.endswith('.wav')]
    index = 0
    indices = []
# Iterate through files in each voice folder
    for file in wav_files:
        file_path = os.path.join(voice, file)

        if os.path.exists(file_path):
            s = slab.Binaural(data=file_path)
            # s.waveform()
            s = s.resample(48828)
            data = s.data[:, 0]
            data = numpy.pad(data, pad_width=(0, 36421-len(data))) # adds all zeros after data, not before
            signals.append(data)
            indices.append((file,index))
            index += len(s.data[:,0])
            sound_events_dur.append(len(data))
    if signals:
        signal=numpy.concatenate(signals)
        folder_name=os.path.basename(voice)
        voice_data[folder_name]={'signal': signal,
                                 'index': indices}


buffer_array = voice_data['voice1']['signal']
s_samples = len(voice_data['voice1']['signal'])
n_pulses = int(96)
trial_seq = numpy.array((8, 9, 9, 5))
trial_seq = numpy.tile(numpy.array((8, 9, 9, 5)), 20)
n_trials = len(trial_seq)
isi = 1000  # isi in ms
n_samples = max(sound_events_dur)
tlo = isi + int(n_samples / sample_freq * 1000)  # in ms tlo arg for pulse train
# how to change the tlo and n_samples dynamically to load into Rpvds

proc.ClearCOF()  # remove previous program from processor
proc.LoadCOF('C:/labplatform/Devices/RCX/test.rcx')  # load target program
proc.SetTagVal('buffer_size', s_samples)  # write a single val to a tag
proc.WriteTagV('trial_seq', 0, trial_seq)
proc.WriteTagV('voice1', 0, buffer_array)
proc.SetTagVal('n_samples', n_samples)
proc.SetTagVal('n_trials', n_trials)
proc.SetTagVal('isi', tlo)
proc.SetTagVal('buffer_size', s_samples)

proc.Run()
proc.SoftTrg(1)  # buffer trigger (read and play stim)

index = trial_seq[0] * n_samples
while True:
    while proc.ReadTagVal('playback'):
        continue
    proc.SetTagVal('index', index)

ild = slab.Binaural.azimuth_to_ild(45)  # degrees azimuth
# -9.12  # correct ILD in dB
signal = signal.ild(ild)  # apply the ILD

signal.play()

proc.SetTagVal('isi_in', isi)  # send initial isi to tdt processor

proc.SoftTrg(3)  # pulse train trigger #todo make better buffer loop

proc.SetTagVal('isi', isi)  # write ISI in rcx pulsetrain tag

proc.SoftTrg(2)
proc.Halt()"""