import threading
import tensorflow as tf
import tensorflow_io as tfio
import pyaudio
import numpy as np
import joblib
import tensorflow.keras as keras
import sys
import wave
import cv2
from ultralytics import YOLO
import logging
import requests
import json
import time
from pydub import AudioSegment
import io
import imutils
import subprocess
import librosa

model = keras.models.load_model('2c_mel_librosa_1200_1400x300_model')
folders = joblib.load("3c_mel_class_1200_labels.pkl")
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
AUTH_TOKEN="MFnFu8ZiTVhNqnSoavQbhsT3dcx9uvAz"
deactivate_camera = False
LIST_OF_VALID = ['person','bicycle','car','motorcycle','bus','truck','bird','cat','dog','horse','sheep',
                 'cow','elephant','bear','zebra']

BATCH_SIZE = 4
SAMPLE_RATE = 44100  # Fișierele tale sunt 44.1 kHz
TARGET_SAMPLE_RATE = 16000  # Resamplează la 16 kHz
DURATION = 3  # Durata fișierului audio în secunde
N_MELS = 300  # Număr de benzi Mel
N_FFT = 1024  # Număr de puncte FFT
HOP_LENGTH = int((TARGET_SAMPLE_RATE * DURATION) / 1400)


def load_sound(filename):

    file_path = filename
    if (isinstance(filename,tf.Tensor)):
        file_path = filename.numpy().decode('utf-8')


    wav, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)

    # Padding la o durată fixă de 3 secunde
    wav = librosa.util.fix_length(wav, size=TARGET_SAMPLE_RATE * DURATION)

    return wav


def create_spectrogram(file_path):
    wav = load_sound(file_path)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wav, sr=TARGET_SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=8000
    )
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

    # Verificăm dimensiunea spectrogramei
    num_frames = mel_spectrogram.shape[1]

    if num_frames < 1400:
        # Adăugăm padding cu zero pentru a face spectrograma de dimensiunea 1400
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, 1400 - num_frames)), mode='constant')
    elif num_frames > 1400:
        # Decupăm spectrograma pentru a face dimensiunea 1400
        mel_spectrogram = mel_spectrogram[:, :1400]

    # Adăugăm o dimensiune suplimentară pentru intrarea modelului
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)
    print(mel_spectrogram.shape)

    #mel_spectrogram.shape = (N_MELS, 1400, 1)  # Asigură-te că setăm dimensiunile

    return mel_spectrogram

def classify_audio():
    temp_filename = 'temp_audio2.mp3'
    mel_spectrogram = create_spectrogram(temp_filename)  # Folosește funcția create_spectrogram
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)  # Adaugă dimensiunea batch-ului
    print("S: ",mel_spectrogram.shape)
    prediction = model.predict(mel_spectrogram)

    print(prediction)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_probabilities = prediction[0]
    max_probability = np.max(predicted_probabilities)

    from matplotlib import pyplot as plt
    print(f"Spectrogram shape: {mel_spectrogram.shape}")
    plt.figure(figsize=(30, 20))
    plt.imshow(mel_spectrogram[0, :, :, 0].T, aspect='auto', origin='lower', cmap='viridis')  # Afișează corect spectrograma
    plt.title(f"{folders[predicted_class[0]]}-{max_probability*100}")
    output_path = 'spectrogram.png'
    plt.savefig(output_path)
    print(f"Spectrogram saved at: {output_path}")
    subprocess.run(["xdg-open", output_path])
    
    if max_probability >= 0.5:
        return folders[predicted_class[0]], max_probability
    else:
        return "Niciun Sunet", 0
        
classify_audio()
