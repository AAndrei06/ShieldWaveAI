import sounddevice
import pyaudio

p = pyaudio.PyAudio()
# Afișează toate dispozitivele de intrare care conțin 'USB' în numele lor și indexurile lor
usb_devices = []
for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0 and 'USB' in info['name']:  # Filtrăm doar dispozitivele care conțin 'USB'
        usb_devices.append(i)  # Adăugăm indexul dispozitivului

# Afișează indexurile dispozitivelor USB
print("Indexurile dispozitivelor USB de intrare disponibile:")
print(usb_devices)
