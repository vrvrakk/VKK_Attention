import time
import slab
import numpy
import os
import math
import freefield
from pathlib import Path
sample_freq = 48828
data_path = Path.cwd()/'voices'

voice_idx = 0
n_trials = 96
sources = (-17.5, 17.5)  # directions for each streams
isi = (1000, 1500)  # isi for both streams in ms
s2_delay = 1000  # delay for the lagging stream in ms

def run_experiment(voice_idx, n_trials, isi=(1000, 1500), sources=(-17.5, 17.5), s2_delay=1000):
    [speaker1] = freefield.pick_speakers((sources[0], 0))
    [speaker2] = freefield.pick_speakers((sources[1], 0))

    wav_folders = [folder for folder in os.listdir(data_path)]
    numbers = [1, 2, 3, 4, 5, 6, 8, 9]
    #todo avoid repetitions
    #todo avoid same number on both streams simultaneously

    trial_seq1 = slab.Trialsequence(conditions=numbers, n_reps=n_trials/len(numbers)) # trials/conditions
    trial_seq2 = slab.Trialsequence(conditions=numbers, n_reps=n_trials/len(numbers)) #todo scale by isi # n reps should be adjusted based on isi difference
    wav_folder = wav_folders[voice_idx]
    wav_files = [file for file in os.listdir(data_path / wav_folder) if file.endswith('.wav')]
    n_samples = []
    for number, file in zip(numbers, wav_files): #combine lists into a single iterable->elements from corresponding positions are paired together
        #  print(f'{number} {file}')
        file_path = data_path / wav_folder / file # create file path with the corresponding wav file name
        if file_path.exists():
            s = slab.Sound(data=file_path)
            s = s.resample(48828)
            n_samples.append(s.n_samples) # places the sound event  duration vals in n_samples list
            freefield.write(f'{number}', s.data,['RX81','RX82']) # loads array on buffer
            freefield.write(f'{number}_n_samples', s.n_samples,['RX81','RX82']) # sets total buffer size according to numeration

    mean_n_samples = int(numpy.mean(n_samples)) # get n_samples mean todo talk to marc
    # set n_trials to pulse trains sheet0/sheet1
    freefield.write('n_trials1', trial_seq1.n_trials, speaker1.analog_proc) # analog_proc attribute from speakertable dom txt file
    freefield.write('n_trials2', trial_seq2.n_trials, speaker2.analog_proc)
    # assign tlo for each pulse train
    tlo1 = isi[0] + int(mean_n_samples / sample_freq * 1000)  # in ms tlo arg for pulse train
    tlo2 = isi[1] + int(mean_n_samples / sample_freq * 1000)  # in ms tlo arg for pulse train
    freefield.write('isi1', tlo1, speaker1.analog_proc)
    freefield.write('isi2', tlo2, speaker2.analog_proc)

    # convert sequence numbers to integers, add a 0 at the beginning and write to trial sequence buffers
    sequence1 = numpy.array(trial_seq1.trials).astype('int32')
    sequence1 = numpy.append(0, sequence1)
    sequence2 = numpy.array(trial_seq2.trials).astype('int32')
    sequence2 = numpy.append(0, sequence2)
    freefield.write('trial_seq1', sequence1,speaker1.analog_proc)
    freefield.write('trial_seq2', sequence2,speaker2.analog_proc)

    # set output speakers for both streams
    freefield.write('channel1',speaker1.analog_channel, speaker1.analog_proc)
    freefield.write('channel2', speaker2.analog_channel, speaker2.analog_proc)

    # start playing
    freefield.play(kind='zBusA')  # buffer trigger (read and play stim)
    time.sleep(s2_delay / 1000)
    freefield.play(kind='zBusB')  # buffer trigger (read and play stim)

    responses = []
    index = readtag
    while index <= n_trials:
        s1_number = freefield.read('s1_number', speaker1.analog_proc)
        s2_number = freefield.read('s2_number', speaker2.analog_proc)
        response = freefield.read('button', 'RP2')
        if response != 0:
            if response != [responses[-1][0]]:
                responses.append([response, s1_number, s2_number])
        # todo add button response (RP2) and compare to current number in both sequences, save response
        # and make sure that button response is only appended once per button press
        # end loop if trial sequence has finished
        index = readtag
        # read tag of current number from trialseq buffer 1 and 2
        # read tag of button response

if __name__ == "__main__":
    proc_list=[['RX81','RX8',  Path.cwd()/'test2.rcx'],
               ['RX82','RX8',  Path.cwd()/'test2.rcx'],
               ['RP2','RP2',  Path.cwd()/'9_buttons.rcx']]
    freefield.initialize('dome',device=proc_list)
    run_experiment(voice_idx, n_trials, isi=isi, sources=sources, s2_delay=s2_delay)

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
proc.Halt()
def init_proc():
    proc = Dispatch('RPco.X')
    connected = proc.ConnectRM1('GB', 1)  # connect processor
    proc.ClearCOF() # remove previous program from processor
    proc.LoadCOF('C:/Users/vrvra/Desktop/attention decoding/test2.rcx')  # load target program
    if connected and proc.Run():
        print('connected and running')
    return proc"""