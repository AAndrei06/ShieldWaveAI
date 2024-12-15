import requests

# Adresa URL la care trimitem request-ul
url = "http://127.0.0.1:8000/api/upload/"

# Deschidem fișierul video cu context manager pentru a ne asigura că este închis după trimitere
with open('./video.avi', 'rb') as video_file:
    files = {'file': video_file}

    # Datele suplimentare pentru request
    data = {
        "classification": 'person',
        "confidence": 90,
        "detection_type": "Video",
    }

    # Trimiterea request-ului POST cu fișierul video
    response = requests.post(url, files=files, data=data)

    # Verificăm dacă request-ul a avut succes
    if response.status_code == 201:
        print("Video uploaded successfully!")
        print(response.json())  # Poți afișa răspunsul de la server, dacă dorești
    else:
        print(f"Failed to upload video. Status code: {response.status_code}")
