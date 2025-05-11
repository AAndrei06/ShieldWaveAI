
import sounddevice
import speech_recognition as sr
import time
import tensorflow
import json
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.text import tokenizer_from_json
from tensorflow.keras.preprocessing.sequence import pad_sequences
import unicodedata
import string
from nltk.corpus import stopwords
import nltk


# Asigură-te că ai descărcat stopwords și punctuație
nltk.download('stopwords')
nltk.download('punkt')

# Cargar stopwords românesti
stop = set(stopwords.words('romanian'))

# Încărcarea modelului și tokenizer-ului
model = load_model("nlp_model_shieldwave")

with open("tokenizer.json", "r") as json_file:
    tokenizer_json = json.load(json_file)
    tokenizer = tokenizer_from_json(tokenizer_json)

# Funcție pentru a elimina diacriticele
def remove_accents(text):
    return ''.join((c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn'))

# Funcție de preprocesare
def preprocess_text(text):
    # Elimină diacriticele
    text = remove_accents(text)

    # Elimină semnele de punctuație
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Tokenizare simplă folosind split()
    words = text.lower().split()

    # Elimină stopwords
    filtered_words = [word for word in words if word not in stop]

    return ' '.join(filtered_words)
    


r = sr.Recognizer()

while True:
    try:
        with sr.Microphone(device_index=1) as source:
            print("Calibrare zgomot de fundal...")
            r.adjust_for_ambient_noise(source, duration=1)  # Calibrarea zgomotului de fundal
            print("Vorbește acum...")
            audio = r.listen(source, timeout=5)  # Ascultă pentru 5 secunde
            print("Procesare...")
            text = r.recognize_google(audio, language="ro-RO")  # Recunoaște ce s-a spus
            sample_text = text
            
            sample_text_processed = preprocess_text(sample_text)
            sample_seq = tokenizer.texts_to_sequences([sample_text_processed])
            sample_pad = pad_sequences(sample_seq, padding='post', maxlen=16)
            prediction = model.predict(sample_pad)
            
            print(sample_text)
            print(prediction)
        source.close()
    except sr.UnknownValueError:
        print("❌ Nu s-a înțeles ce ai spus.")
    except sr.RequestError as e:
        print("❌ Eroare cu serviciul Google:", e)
    except Exception as e:
        print("❌ Alte probleme:", e)
    time.sleep(1)
