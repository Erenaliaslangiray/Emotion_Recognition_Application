import pyaudio
import math
import struct
import wave
import time
import os

Threshold = 1
SHORT_NORMALIZE = (1.0 / 32768.0)
chunk = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 22050
swidth = 2
TIMEOUT_LENGTH = 1.5

f_name_directory = './Data'


class Recorder:
    def __init__(self):
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(format=FORMAT,
                                  channels=CHANNELS,
                                  rate=RATE,
                                  input=True,
                                  output=True,
                                  frames_per_buffer=chunk)

    @staticmethod
    def rms(frame):
        count = len(frame) / swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)
        sum_squares = 0.0
        for sample in shorts:
            n = sample * SHORT_NORMALIZE
            sum_squares += n * n
            rms = math.pow(sum_squares / count, 0.5)
            return rms * 1000

    def record(self, file_name='speech_audio.wav'):
        print('Noise detected, recording beginning')
        rec = []
        current = time.time()
        end = time.time() + TIMEOUT_LENGTH
        while current <= end:
            data = self.stream.read(chunk, exception_on_overflow=False)
            if self.rms(data) >= Threshold: end = time.time() + TIMEOUT_LENGTH
            current = time.time()
            rec.append(data)
        self.write(b''.join(rec), file_name=file_name)

    def write(self, recording, file_name='speech_audio.wav'):
        n_files = len(os.listdir(f_name_directory))
        filename = os.path.join(f_name_directory, file_name)
        wf = wave.open(filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(recording)
        wf.close()
        print('Written to file: {}'.format(filename))
        print('Returning to listening')



a = Recorder()
print('Listening beginning')
a.listen()

