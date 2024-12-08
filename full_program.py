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

logging.getLogger('ultralytics').setLevel(logging.CRITICAL)

model = keras.models.load_model('4batches_fr320_32_4epochs_model/4batches_fr320_32_4epochs_model')
folders = joblib.load('class_labels2.pkl')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3

def calculate_db(audio_data):
    audio_tensor = tf.convert_to_tensor(audio_data, dtype=tf.float32)
    rms = tf.sqrt(tf.reduce_mean(tf.square(audio_tensor)))
    db = 20 * tf.math.log(rms) / tf.math.log(10.0)
    return db.numpy()

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
    db_level = calculate_db(np.frombuffer(audio_data_bytes, dtype=np.int16))

    if db_level > 60:
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
    else:
        return " < 60db"

def audio_classification_thread():
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
    print("Audio Classification Running...")
    while True:
        label = classify_audio(stream)
        print(f"Audio classified as: {label}")

    stream.stop_stream()
    stream.close()
    p.terminate()


def object_detection_thread():
    model = YOLO('yolov8n.pt')
    cap = cv2.VideoCapture(0)

    print("Object Detection Running...")

    while True:
        ret, frame = cap.read()
        results = model(frame)
        for result in results:
            for box in result.boxes:
                cls = int(box.cls[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = box.conf[0]
                if confidence > 0.7:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"{model.names[cls]} {confidence:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
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
