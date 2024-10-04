import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parameters
duration = 1  # seconds
sample_rate = 48000  # Hz

# List available audio devices
# print("Available audio devices:")
# print(sd.query_devices())

# Get the default input device
default_device = sd.query_devices(kind='input')
#print(f"\nDefault input device: {default_device['name']}")

# Prompt user to press Enter to start recording
input(f"Press Enter to start recording for {duration} seconds...")

print(f"Recording audio for {duration} seconds...")
try:
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, blocking=True)
    print("Recording finished.")
except Exception as e:
    print(f"An error occurred during recording: {e}")
    exit(1)

# Check if we actually recorded anything
# if np.all(recording == 0):
#     print("Warning: The recording appears to be silent. Please check your microphone.")



# Convert the recorded data to a 1D array
print(type(recording))
print(recording.shape)
audio_time_data = recording.flatten()
print(audio_time_data)
print(len(audio_time_data))


N = len(audio_time_data)

# Create time array
ts = np.linspace(0, duration, len(audio_time_data))
fs = fftfreq(N, d=(1/sample_rate))
#fs = np.linspace(0, 10000, len(audio_time_data))

# Plot the audio data
plt.figure(figsize=(10, 4))
plt.plot(ts, audio_time_data)
plt.title('Time Domain Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

# Add vertical lines every second
for i in range(1, int(duration)):
    plt.axvline(x=i, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()


#perform FFT
audio_freq_data = np.abs(fft(audio_time_data)) # take absolute val

plt.figure(figsize=(10, 4))
plt.plot(fs, audio_freq_data)
plt.title('Frequency Domain Audio Waveform')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.grid(True)

# Add vertical lines every second
for i in range(1, int(duration)):
    plt.axvline(x=i, color='r', linestyle='--', alpha=0.5)

plt.tight_layout()
plt.show()
print("Close the plot window to end the program.")


