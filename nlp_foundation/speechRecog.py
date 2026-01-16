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


#clipping = distortion - change that makes sound harsh
recognizer = sr.Recognizer()
file_path = r'nlp_foundation\speech_01.wav'

def transcribe_audio(file_path):
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        text = recognizer.recognize_google(audio_data)
        print(text)
        return text
transcribed_text = transcribe_audio(file_path) 
print(transcribed_text)

# (Word Error rate)WER = (insertion(cat->the cat) + deletion(the ball->ball) + substitution(cat->bat))/no. of words [less WER high accuracy]
# (character Error rate)CER = (insertion(char added) + deletion(char deleted) + substitution(char changed))/no. of words [less WER high accuracy]

# Reference transcript - ground truth
ground_truth = """My name is Ivan ans I am excited to have as a part of our learning community!
Before we get started, I'd like to tell you a little bit about myself. I'm a sound engineer turned data scientist,
curious about machine learning and Artificial Intelligence. My professional background is primarily in media production with a focus on audio, IT, and communications
"""

calculated_wer = wer(ground_truth, transcribed_text)
calculated_cer = cer(ground_truth, transcribed_text)

print(calculated_wer)
print(calculated_cer)


S = librosa.stft(audio_signal)
S_db = librosa.amplitude_to_db(abs(S), ref=np.max)
np.max(S_db)
plt.figure(figsize=(12,4))
librosa.display.specshow(data=S_db, sr=sample_rate, x_axis='time',y_axis='log')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectogram')
plt.xlabel('Time')
plt.ylabel('Frequency')
plt.show()