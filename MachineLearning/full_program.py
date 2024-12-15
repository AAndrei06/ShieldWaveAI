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

'''
import torch
torch.set_num_threads(4)
'''

logging.getLogger('ultralytics').setLevel(logging.CRITICAL)
model = keras.models.load_model('4batches_fr320_32_4epochs_model/4batches_fr320_32_4epochs_model')
folders = joblib.load('class_labels2.pkl')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3

def save_audio_to_wav(audio_data_bytes, filename):
    with wave.open(filename, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)
        wf.setframerate(RATE)
        wf.writeframes(audio_data_bytes)

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

def classify_audio(stream):
    frames = []
    for _ in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(np.frombuffer(data, dtype=np.int16))

    audio_data_bytes = b''.join(frames)

    temp_filename = 'temp_audio.wav'
    save_audio_to_wav(audio_data_bytes, temp_filename)
    audio_tensor = load_sound(temp_filename)
    audio_tensor = tf.expand_dims(audio_tensor, axis=0)
    prediction = model.predict(audio_tensor)
    predicted_class = np.argmax(prediction, axis=1)
    predicted_probabilities = prediction[0]
    max_probability = np.max(predicted_probabilities)
    if max_probability >= 0.8:
        return folders[predicted_class[0]], max_probability
    else:
        return "Niciun Sunet", 0

def send_alert(confidence, object_detected, detection_type):
    url = 'http://127.0.0.1:8000/api/upload/'

    file_type = ""
    if detection_type == "Audio":
        file_path = "temp_audio.wav"
        file_type = "audio/wav"
    elif detection_type == "Video":
        file_path = "video.avi"
        file_type = "video/avi"
    else:
        return "Nu a mers ceva"

    data = {
        "classification": object_detected,
        "confidence": confidence,
        "detection_type": detection_type,
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
    while True:
        label = classify_audio(stream)
        #send_alert(int(label[1]*100), label[0],"Audio")
        #print(f"Audio classified as: {label}")

    stream.stop_stream()
    stream.close()
    p.terminate()


def object_detection_thread():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    fps = 10
    output_file = 'video.avi'
    out = cv2.VideoWriter(output_file, fourcc, fps, (640, 480))
    frames = []
    start_time = time.time()

    print("Object Detection Running...")

    last_alert_time = 0
    alert_interval = 5

    while True:
        ret, frame = cap.read()
        results = model(frame, imgsz=440)
        object_detected = False
        detected_object_name = ""
        confidence_detected = ""
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                if confidence > 0.5:
                    object_detected=True
                    detected_object_name=model.names[cls]
                    confidence_detected = int(confidence * 100)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{model.names[cls]} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('YOLOv8 Object Detection', frame)
        frames.append(frame)

        current_time = time.time()
        if object_detected and (current_time - last_alert_time) > alert_interval:
            send_alert(confidence_detected, detected_object_name, "Video")
            last_alert_time = current_time

        if time.time() - start_time >= 20:
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

    audio_thread.start()
    object_detection_thread.start()

    audio_thread.join()
    object_detection_thread.join()
