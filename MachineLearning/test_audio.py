import pyaudio

p = pyaudio.PyAudio()

for i in range(p.get_device_count()):
    info = p.get_device_info_by_index(i)
    if info['maxInputChannels'] > 0:
        print(f"Index: {i} | Name: {info['name']} | Channels: {info['maxInputChannels']}")

info = p.get_default_input_device_info()
print(f"Default device index: {info['index']}, name: {info['name']}")