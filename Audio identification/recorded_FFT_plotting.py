import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# Parameters
duration = 0.5  # seconds
sample_rate = 48000  # Hz
threshold = 40  # amplitude threshold to be categorized as a prominent frequency

# List available audio devices
# print("Available audio devices:")
# print(sd.query_devices())

# Use default input device
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
audio_time_data = recording.flatten()


# Perform FFT
audio_freq_data = np.abs(fft(audio_time_data)) # take absolute val

# Create time and frequency domain array (x-axis)
N = len(audio_time_data)  # num bins
ts = np.linspace(0, duration, N)
fs = fftfreq(N, d=(1/sample_rate))

# Identifying prominent frequencies
freqs_index = []  # indices of high amplitude frequencies
for index, amplitude in enumerate(audio_freq_data):
    if amplitude > threshold:
        freqs_index.append(index)

freqs = [fs[k] for k in freqs_index]  # will always be an even number (has positive and negative for every value)
freqs = freqs[:len(freqs) // 2]  # slices so only get first half (second half is identical mirror but negative)

freqs_refined = []

def refine(freqs):  # for consecutive group of freqs, pick out middle one and list it
    mid_freqs = []  # initializes list. Will be list of middle frequency for each group
    num_groups = 0  # tracks number of clusters/groupings of freqs (groups of one included)
    group_len = 1  # initialized at 1

    for k in range(1, len(freqs)):
        diff = freqs[k] - freqs[k - 1]
        if diff < 5:
            group_len += 1
        else:
            num_groups += 1
            print(f"Group {num_groups} ended. Size: {group_len}")
            mid_freqs.append(freqs[k - int(group_len / 2)])  # appends element in middle of group
            group_len = 1  # group ended, reinitialize at 1
    num_groups += 1
    print(f"FINAL Group {num_groups} ended. Size: {group_len}")
    print(len(freqs))
    print(len(mid_freqs))
    print(len(freqs) - int(group_len/2))
    mid_freqs.append(freqs[len(freqs)-1 - int(group_len/2)])  # appends element in middle of group (for last group)

    return mid_freqs, num_groups

refined_freqs, num_groups = refine(freqs)

print(len(refined_freqs))
print(num_groups)

for i in range(num_groups):
   print(f"Group {i}: {refined_freqs[i]} Hz")

# PLOTTING
plt.figure(figsize=(15, 6))

# Plot the time-domain data
plt.subplot(2, 1, 1)  # (number of rows, number of columns, plot index)
plt.subplot(1, 2, 1)  # (number of rows, number of columns, plot index)
plt.plot(ts, audio_time_data)
plt.title('Time Domain Audio Waveform')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True)

"""
for i in range(1, int(duration)): # Add vertical lines every second
    plt.axvline(x=i, color='r', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
"""
# Plot frequency domain data
#plt.subplot(2, 1, 2)  # (number of rows, number of columns, plot index)
plt.subplot(1, 2, 2)  # (number of rows, number of columns, plot index)
plt.plot(fs, audio_freq_data)
plt.title('Frequency Domain Audio Waveform')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')
plt.xlim(0, 1300)  # for plotting0
plt.grid(True)

plt.tight_layout()
plt.show()

print("Close the plot window to end the program.")


