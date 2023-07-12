import logging
import threading
import time
import slab

def play_voice(sound, isi, times, event):
   while True:
      sound.play()
      times.append(time.time())
      time.sleep(isi)
      logging.info('playing sound')
      if event.is_set():
                  break

def play_experiment(duration):
   global sound_1_times, sound_2_times
   # sounds
   sound_1 = slab.Sound.tone(frequency=350, duration=1.0)
   sound_2 = slab.Sound.tone(frequency=250, duration=1.0)
   # times and responses
   responses = []
   sound_1_times = []
   sound_2_times = []
   # create threads
   event = threading.Event()
   stream_1 = threading.Thread(target=play_voice, args=(sound_1, 1, sound_1_times, event))
   stream_2 = threading.Thread(target=play_voice, args=(sound_2, 1.5, sound_2_times, event))
   # start multithreading
   stream_1.start()
   stream_2.start()
   start_time = time.time()
   while time.time() < start_time + duration:  # play for specified duration
      response = int(input())
      responses.append(tuple((response, time.time())))
      logging.info('user response')
      continue
   event.set()  # stop sounds
   return responses, sound_1_times, sound_2_times

if __name__ == "__main__":
   responses, sound_1_times, sound_2_times = play_experiment(5)




   #
   # format = "%(asctime)s: %(message)s"
   # logging.basicConfig(format=format, level=logging.INFO, datefmt="%H:%M:%S")
   # stream_1 = threading.Thread(target=play_voice, args=(sound_1, 1, sound_1_times))
   # stream_2 = threading.Thread(target=play_voice, args=(sound_2, 1.5, sound_2_times))
   #
   # response = threading.Thread(target=get_response, args=())
   # stream_1.start()
   # stream_2.start()
   # response.start()
   #
   #
   #
   # time.sleep(20)
   # stream_1.stop()
   # stream_2.stop()


