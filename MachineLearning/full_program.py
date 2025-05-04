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
import os
import signal

logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
model = keras.models.load_model('new_3c_mel_librosa_1200_1400x300_model')
#folders = joblib.load("class_labels2.pkl")
folders = joblib.load("3c_mel_class_1200_labels.pkl")
#folders = ['door', 'voice', 'glass', 'silence', 'dog', 'footsteps']


LIVE_KEY = ""
#LIVE_KEY=""
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3
AUTH_TOKEN="Shvoe7L48P4sILqOI9dhKpSvpnXv6Ndu"
deactivate_camera = False
deactivate_actual_camera = False
deactivate_actual_microphone = False
truly_deactivate = False
LIST_OF_VALID = ['person','bicycle','car','motorcycle','bus','truck','bird','cat','dog','horse','sheep',
                 'cow','elephant','bear','zebra']

BATCH_SIZE = 4
SAMPLE_RATE = 44100
TARGET_SAMPLE_RATE = 16000
DURATION = 3
N_MELS = 300
N_FFT = 1024
HOP_LENGTH = int((TARGET_SAMPLE_RATE * DURATION) / 1400)

try:
    url = "http://127.0.0.1:8000/api/check_user/"
    response = requests.get(url, params={"auth_token": AUTH_TOKEN})
    if response.status_code == 200:
        data = response.json()
        print(data)
        if (data['state'] == "NoUser"):
            os._exit(0)
except Exception as e:
    print(f"Eroare la cerere: {e}")


try:
    url = "http://127.0.0.1:8000/api/initial_clean/"
    response = requests.get(url, params={"auth_token": AUTH_TOKEN})
    data = response.json()
    print(data)
except Exception as e:
    print(f"Eroare la cerere: {e}")



def livestream():
    global truly_deactivate
    global deactivate_camera

    cap = cv2.VideoCapture(2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if LIVE_KEY == "":
        cap.release()
        return

    command = [
        'ffmpeg',
        '-f', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', '640x480',
        '-i', '-',
        '-ar', '44100',
        '-ac', '2',
        '-acodec', 'pcm_s16le',
        '-f', 's16le',
        '-ac', '2',
        '-i', '/dev/zero',
        '-acodec', 'aac',
        '-ab', '128k',
        '-vcodec', 'h264',
        '-pix_fmt', 'yuv420p',
        '-g', '50',
        '-vb', '2500k',
        '-profile:v', 'baseline',
        '-preset', 'ultrafast',
        '-r', '30',
        '-f', 'flv',
        f'rtmp://a.rtmp.youtube.com/live2/{LIVE_KEY}'
    ]

    pipe = subprocess.Popen(command, stdin=subprocess.PIPE, preexec_fn=os.setsid)

    try:
        while not truly_deactivate and not deactivate_camera:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[WARN] Camera read failed, exiting livestream loop.")
                break
            pipe.stdin.write(frame.tobytes())
    except Exception as e:
        print(f"[ERROR] Exception during livestream: {e}")
    finally:
        print("[INFO] Cleaning up livestream...")
        cap.release()
        if pipe.stdin:
            try:
                pipe.stdin.close()
            except Exception:
                pass
        if pipe.poll() is None:  # if still running
            try:
                os.killpg(os.getpgid(pipe.pid), signal.SIGTERM)
            except Exception:
                pass


def fetch_camera_activate_deactivate():
    global deactivate_camera
    global deactivate_actual_camera
    global deactivate_actual_microphone
    global truly_deactivate

    while truly_deactivate == False:
        url = "http://127.0.0.1:8000/api/get_status_info/"
        response = requests.get(url, params={"auth_token": AUTH_TOKEN})
        data = response.json()
        print(data)
        if response.status_code == 200 and deactivate_camera == True:
            if data.get("activate") == "yes":
                current_path = os.getcwd()
                parent_path = os.path.dirname(current_path)

                os.chdir(parent_path)
                truly_deactivate = True
                os.system("./startup.sh")
            else:
                print("User is not yet activated.")

        elif response.status_code == 200 and (deactivate_camera == False or deactivate_actual_camera == False or deactivate_actual_microphone == False):
            if data.get('deactivateSystem') == "yes":
                deactivate_camera = True

            if data.get('deactivateCam') == "yes":
                deactivate_actual_camera = True

            if data.get('deactivateMic') == "yes":
                deactivate_actual_microphone = True
        else:
            print("No document found")

        time.sleep(12)
        

def activity_thread():
    global deactivate_camera
    global truly_deactivate

    while truly_deactivate == False:
        print("Info Working -------------")
        try:
            url = "http://127.0.0.1:8000/api/activity_info/"
            response = requests.get(url, params={"auth_token": AUTH_TOKEN,'deactivate_variable': deactivate_camera})
            if response.status_code == 200:
                data = response.json()
                print(data)
            else:
                print(f"No document found to info")
        except Exception as e:
            print(f"Eroare la cerere: {e}")
        
        time.sleep(12)


def save_audio_to_mp3(audio_data_bytes, filename):
    audio = AudioSegment.from_raw(io.BytesIO(audio_data_bytes), sample_width=2, frame_rate=44100, channels=1)
    bitrate = "122k"
    audio.export(filename, format="mp3", bitrate=bitrate)


def load_sound(filename):
    file_path = filename
    if (isinstance(filename,tf.Tensor)):
        file_path = filename.numpy().decode('utf-8')

    wav, sr = librosa.load(file_path, sr=TARGET_SAMPLE_RATE)
    wav = librosa.util.fix_length(wav, size=TARGET_SAMPLE_RATE * DURATION)
    return wav


def create_spectrogram(file_path):
    wav = load_sound(file_path)
    mel_spectrogram = librosa.feature.melspectrogram(
        y=wav, sr=TARGET_SAMPLE_RATE, n_mels=N_MELS, n_fft=N_FFT, hop_length=HOP_LENGTH, fmax=8000
    )
    mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
    num_frames = mel_spectrogram.shape[1]

    if num_frames < 1400:
        mel_spectrogram = np.pad(mel_spectrogram, ((0, 0), (0, 1400 - num_frames)), mode='constant')
    elif num_frames > 1400:
        mel_spectrogram = mel_spectrogram[:, :1400]

    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=-1)

    return mel_spectrogram

def classify_audio(stream):
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    audio_data_bytes = b''.join(frames)

    temp_filename = 'temp_audio.mp3'
    save_audio_to_mp3(audio_data_bytes, temp_filename)

    mel_spectrogram = create_spectrogram(temp_filename)
    mel_spectrogram = np.expand_dims(mel_spectrogram, axis=0)
    prediction = model.predict(mel_spectrogram)

    print(prediction)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_probabilities = prediction[0]
    max_probability = np.max(predicted_probabilities)
    
    '''
    from matplotlib import pyplot as plt
    print(f"Spectrogram shape: {mel_spectrogram.shape}")
    plt.figure(figsize=(30, 20))
    plt.imshow(mel_spectrogram[0, :, :, 0].T, aspect='auto', origin='lower', cmap='viridis')
    plt.title(f"{folders[predicted_class[0]]}-{max_probability*100}")
    output_path = 'spectrogram.png'
    plt.savefig(output_path)

    #print(f"Spectrogram saved at: {output_path}")
    #subprocess.run(["xdg-open", output_path])
    '''
    
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
    while not deactivate_camera and not deactivate_actual_microphone:
        label = classify_audio(stream)
        print(label[0])
        print(int(label[1]*100))
        if label[1]*100 > 50 and label[0] != "silence":
            if (label[0] == "dog"):
                send_alert(int(label[1]*100), "dog_audio","Audio")
            else:
                send_alert(int(label[1]*100), label[0],"Audio")
        print(f"Audio classified as: {label}")

    stream.stop_stream()
    stream.close()
    p.terminate()


def object_detection_thread():
    model = YOLO('yolov8n.pt')
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

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    _, start_frame = cap.read()
    start_frame = imutils.resize(start_frame, width = 500)
    start_frame = cv2.cvtColor(start_frame,cv2.COLOR_BGR2GRAY)
    start_frame = cv2.GaussianBlur(start_frame, (21,21), 0)

    while not deactivate_camera and not deactivate_actual_camera:
        ret, frame = cap.read()
        results = model(frame, imgsz=440)
        object_detected = False
        detected_object_name = ""
        confidence_detected = ""

        frame = imutils.resize(frame, width = 500)
        frame_bw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_bw = cv2.GaussianBlur(frame_bw, (5,5), 0)
        difference = cv2.absdiff(frame_bw, start_frame)
        current_time = time.time()
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
                send_alert(100, "Necunoscut", "Video")
                last_alert_time = current_time

        frames.append(frame)

        
        if object_detected and (current_time - last_alert_time) > alert_interval:
            send_alert(confidence_detected, detected_object_name, "Video")
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
    object_detection_thread = threading.Thread(target=object_detection_thread)
    livestream_thread = threading.Thread(target=livestream)
    activate_deactivate_thread = threading.Thread(target=fetch_camera_activate_deactivate)
    activity_thread = threading.Thread(target=activity_thread)

    audio_thread.start()
    object_detection_thread.start()
    livestream_thread.start()
    activate_deactivate_thread.start()
    activity_thread.start()

    audio_thread.join()
    object_detection_thread.join()
    livestream_thread.join()
    activate_deactivate_thread.join()
    activity_thread.join()

