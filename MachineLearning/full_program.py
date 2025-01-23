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

# Ca să îl fac mai rapid
'''
import torch
torch.set_num_threads(4)
'''

logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
model = keras.models.load_model('mel_librosa_1400x300_model')
#folders = joblib.load("class_labels2.pkl")
#folders = ['door', 'voice', 'glass','silence']
#folders = ['door', 'voice', 'glass', 'footsteps','silence','dog']
#folders = joblib.load("class_labels2.pkl")
folders = ['door', 'voice', 'glass', 'silence', 'dog', 'footsteps']

command = ['ffmpeg',
            '-f', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s','640x480',
            '-i','-',
            '-ar', '44100',
            '-ac', '2',
            '-acodec', 'pcm_s16le',
            '-f', 's16le',
            '-ac', '2',
            '-i','/dev/zero',   
            '-acodec','aac',
            '-ab','128k',
            '-strict','experimental',
            '-vcodec','h264',
            '-pix_fmt','yuv420p',
            '-g', '50',
            '-vb','1000k',
            '-profile:v', 'baseline',
            '-preset', 'ultrafast',
            '-r', '30',
            '-f', 'flv', 
            'rtmp://a.rtmp.youtube.com/live2/d4jp-wysv-7e8q-67sp-3efu']



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

def fetch_camera_deactivate():
    global deactivate_camera
    while True and not deactivate_camera:
        try:
            url = "http://127.0.0.1:8000/api/deactivate/"
            response = requests.get(url, params={"auth_token": AUTH_TOKEN})
            if response.status_code == 200:
                data = response.json()

                if (data['user_token'] == AUTH_TOKEN and data['state'] == True):
                    deactivate_camera = True
            else:
                print(f"No document found")
        except Exception as e:
            print(f"Eroare la cerere: {e}")
        
        time.sleep(5)
'''
def save_audio_to_wav(audio_data_bytes, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio_data_bytes)
'''
'''
def save_audio_to_mp3(audio_data_bytes, filename):
    audio = AudioSegment.from_raw(io.BytesIO(audio_data_bytes), sample_width=2, frame_rate=RATE, channels=CHANNELS)
    audio.export(filename, format="mp3")
'''
def save_audio_to_mp3(audio_data_bytes, filename):
    audio = AudioSegment.from_raw(io.BytesIO(audio_data_bytes), sample_width=2, frame_rate=44100, channels=1)
    bitrate = "122k"
    audio.export(filename, format="mp3", bitrate=bitrate)
'''
def load_sound(filename):
    file_contents = tf.io.read_file(filename)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=-1)
    return spectrogram
'''



'''
def load_sound(filename):
    res = tfio.audio.AudioIOTensor(filename, dtype=tf.float32)
    tensor = res.to_tensor()
    tensor = tf.math.reduce_sum(tensor, axis=1) / 2  # Calculul mediei pe canale
    sample_rate = res.rate

    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    wav = tfio.audio.resample(tensor, rate_in=sample_rate, rate_out=16000)

    # Trunchierea și completarea cu zero-uri la 48000
    wav = wav[:48000]
    zero_padding = tf.zeros([48000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav], 0)

    # Generarea spectrogramelor folosind STFT
    spectrogram = tf.signal.stft(wav, frame_length=320, frame_step=32)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)  # Adăugăm dimensiunea corectă
    return spectrogram
'''

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

def classify_audio(stream):
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    audio_data_bytes = b''.join(frames)

    temp_filename = 'temp_audio.mp3'
    save_audio_to_mp3(audio_data_bytes, temp_filename)

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

def send_alert(confidence, object_detected, detection_type):
    url = 'http://127.0.0.1:8000/api/upload/'

    file_type = ""
    if detection_type == "Audio":
        file_path = "temp_audio.mp3"
        file_type = "audio/mp3"
    elif detection_type == "Video":
        file_path = "video.avi"
        file_type = "video/avi"
    else:
        return "Nu a mers ceva"

    data = {
        "classification": object_detected,
        "confidence": confidence,
        "detection_type": detection_type,
        "auth_token": AUTH_TOKEN
    }

    with open(file_path, "rb") as file:
        files = {
            "file": (file_path, file, file_type),
        }
        response = requests.post(url, data=data, files=files)


def audio_classification_thread():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Audio Classification Running...")
    while True and not deactivate_camera:
        label = classify_audio(stream)
        #send_alert(int(label[1]*100), label[0],"Audio")
        print(f"Audio classified as: {label}")

    stream.stop_stream()
    stream.close()
    p.terminate()


def object_detection_thread():
    model = YOLO('yolov8n.pt')
    
    print(model.names)
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 9
    output_file = 'video.avi'
    out = cv2.VideoWriter(output_file, fourcc, fps, (640, 480))
    frames = []
    start_time = time.time()

    print("Object Detection Running...")

    last_alert_time = 0
    alert_interval = 5

    # Motion detection
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    _, start_frame = cap.read()
    start_frame = imutils.resize(start_frame, width = 500)
    start_frame = cv2.cvtColor(start_frame,cv2.COLOR_BGR2GRAY)
    start_frame = cv2.GaussianBlur(start_frame, (21,21), 0)


    while True and not deactivate_camera:
        ret, frame = cap.read()
        results = model(frame, imgsz=440)
        object_detected = False
        detected_object_name = ""
        confidence_detected = ""

        frame = imutils.resize(frame, width = 500)
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.GaussianBlur(frame_bw, (5,5), 0)
        difference = cv2.absdiff(frame_bw, start_frame)

        threshold = cv2.threshold(difference, 25, 255, cv2.THRESH_BINARY)[1]
        start_frame = frame_bw
        if threshold.sum() > 200000:
            for result in results:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = box.conf[0]
                    if confidence > 0.5 and model.names[cls] in LIST_OF_VALID:
                        object_detected=True
                        detected_object_name=model.names[cls]
                        confidence_detected = int(confidence * 100)
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        label = f"{model.names[cls]} {confidence:.2f}"
                        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if object_detected == False and (current_time - last_alert_time) > alert_interval:
                #send_alert(100, "Necunoscut", "Video")
                last_alert_time = current_time

        frames.append(frame)

        current_time = time.time()
        if object_detected and (current_time - last_alert_time) > alert_interval:
            #send_alert(confidence_detected, detected_object_name, "Video")
            last_alert_time = current_time

        if current_time - start_time >= 10:
            video_name = 'video.avi'
            height, width, _ = frames[0].shape
            out = cv2.VideoWriter(video_name, fourcc, fps, (width, height))
            for f in frames:
                out.write(f)
            out.release()
            frames = []
            start_time = time.time()

        cv2.imshow('YOLOv8 Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    audio_thread = threading.Thread(target=audio_classification_thread)
    #object_detection_thread = threading.Thread(target=object_detection_thread)
    #deactivate_thread = threading.Thread(target=fetch_camera_deactivate) 

    audio_thread.start()
    #object_detection_thread.start()
    #deactivate_thread.start()

    audio_thread.join()
    #object_detection_thread.join()
    #deactivate_thread.join()

