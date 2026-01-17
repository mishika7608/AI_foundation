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
# audio_signal, sample_rate = librosa.load(r'nlp_foundation\speech_01.wav', sr=None)
# print(sample_rate)
# plt.figure(figsize=(12,4))
# librosa.display.waveshow(audio_signal, sr = sample_rate)
# plt.title('Wavefom')
# plt.xlabel('Time (s)')
# plt.ylabel('Amplitude')
# plt.show()
# Audio(r'nlp_foundation\speech_01.wav')


# #clipping = distortion - change that makes sound harsh
# recognizer = sr.Recognizer()
# file_path = r'nlp_foundation\speech_01.wav'

# def transcribe_audio(file_path):
#     with sr.AudioFile(file_path) as source:
#         audio_data = recognizer.record(source)
#         text = recognizer.recognize_google(audio_data)
#         print(text)
#         return text
# transcribed_text = transcribe_audio(file_path) 
# print(transcribed_text)

# # (Word Error rate)WER = (insertion(cat->the cat) + deletion(the ball->ball) + substitution(cat->bat))/no. of words [less WER high accuracy]
# # (character Error rate)CER = (insertion(char added) + deletion(char deleted) + substitution(char changed))/no. of words [less WER high accuracy]

# # Reference transcript - ground truth
# ground_truth = """My name is Ivan ans I am excited to have as a part of our learning community!
# Before we get started, I'd like to tell you a little bit about myself. I'm a sound engineer turned data scientist,
# curious about machine learning and Artificial Intelligence. My professional background is primarily in media production with a focus on audio, IT, and communications
# """

# calculated_wer = wer(ground_truth, transcribed_text)
# calculated_cer = cer(ground_truth, transcribed_text)

# print(calculated_wer)
# print(calculated_cer)


# S = librosa.stft(audio_signal)
# S_db = librosa.amplitude_to_db(abs(S), ref=np.max)
# np.max(S_db)
# plt.figure(figsize=(12,4))
# librosa.display.specshow(data=S_db, sr=sample_rate, x_axis='time',y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectogram')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.show()

# signal_filtered = librosa.effects.preemphasis(audio_signal, coef=0.97)
# sf.write('filtered_speech_01.wav', signal_filtered, sample_rate)

# filtered_signal, sr = librosa.load('filtered_speech_01.wav', sr=None)

# Sb = librosa.stft(filtered_signal)
# S_dbb = librosa.amplitude_to_db(np.abs(Sb), ref=np.max)

# plt.figure(figsize=(12,4))
# librosa.display.specshow(S_dbb, sr=sr, x_axis='time', y_axis='log')
# plt.colorbar(format='%+2.0f dB')
# plt.title('Spectrogram (Pre-emphasized Signal)')
# plt.xlabel('Time')
# plt.ylabel('Frequency')
# plt.show()

# # calculated_wer = wer(ground_truth, transcribed_text)
# # calculated_cer = cer(ground_truth, transcribed_text)
# # print(calculated_wer)
# # print(calculated_cer)

# model = whisper.load_model("base")
# result = model.transcribe(file_path)

# transcribed_text_whisper = result["text"]
# print(transcribed_text_whisper)
# print(result["language"])
# calculated_wer = wer(ground_truth, transcribed_text_whisper)
# calculated_cer = cer(ground_truth, transcribed_text_whisper)
# print(calculated_wer)
# print(calculated_cer)

# # Preprocessing -> MEl spectogram -> pattern identify -> Language modelling(encoder-transformer-decoder) -> postprocessing

# directory_path = r'D:\PythonFolder\nlp_foundation\Recordings.rar'
# def transcribe_directory_whisper(directory_path):
#     transcriptions = []
#     for file_name in os.listdir(directory_path):
#         if file_name.endswith(".wav"):
#             files_path = os.path.join(directory_path, file_name)
#             result = model.transcribe(files_path)
#             transcription = result['text']
#             transcriptions.append({'file_name':file_name, "transcription":transcription})
#     return transcriptions
# transcriptions = transcribe_directory_whisper(directory_path)

# output_file = "transcriptions.csv"
# with open(output_file, mode='w',newline='') as file:
#     writer=csv.writer(file)
#     writer.writerow(["Track Number", "File Name", "Transcription"])
#     for number, transcription in enumerate(transcriptions, start=1):
#         writer.writerow([number, transcription['file_name'], transcription['transcription']])

text = """Thank you for joining us!"""
tts = gTTS(text=text, lang='en')
tts.save("output.mp3")
os.system('start output.mp3')