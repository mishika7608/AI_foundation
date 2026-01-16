import numpy as np 
import matplotlib.pyplot as plt

import librosa
import librosa.display
import soundfile as sf
import speech_recognition as sr
from jiwer import wer, cer
from IPython.display import Audio
import whisper
import csv
import os
import tempfile
import wave
from gtts import gTTS

# wave of sound file
#integer - audio sound captured per sec
audio_signal, sample_rate = librosa.load(r'nlp_foundation\speech_01.wav', sr=None)
print(sample_rate)
plt.figure(figsize=(12,4))
librosa.display.waveshow(audio_signal, sr = sample_rate)
plt.title('Wavefom')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.show()
Audio(r'nlp_foundation\speech_01.wav')

