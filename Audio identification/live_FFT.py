import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
duration = 1  # sec
sample_rate = 48000  # samples per second
block_size = 4096  # num samples per block

# Create a buffer to store the audio data
buffer = np.zeros(block_size)

# Calculate the frequency array for FFT
freq_array = np.fft.rfftfreq(block_size, d=1 / sample_rate)

# Create the plot
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
line1, = ax1.plot([], [])
line2, = ax2.plot([], [], lw=2)

ax1.set_xlim(0, block_size)
ax1.set_ylim(-1, 1)
ax1.set_title('Raw Audio Signal')
ax1.set_xlabel('Sample')
ax1.set_ylabel('Amplitude')

ax2.set_xlim(20, 1500)
ax2.set_ylim(0, 0.1)
ax2.set_title('Live (1 second) FFT of Microphone Input')
ax2.set_xlabel('Frequency (Hz)')
ax2.set_ylabel('Magnitude')

plt.tight_layout()


# Function to update the plot
def update_plot(frame):
    global buffer

    # Plot raw audio signal
    line1.set_data(range(block_size), buffer)

    # Compute and plot FFT
    buffer_FFT = np.abs(np.fft.rfft(buffer)) / block_size # normalize with  / block_size
    line2.set_data(freq_array, buffer_FFT)

    return line1, line2


def audio_callback(indata, frames, time, status):
    global buffer
    if status:
        print(status)
    buffer = indata[:, 0]

    # Print max value to console for debugging
    #print(f"Max amplitude: {np.max(np.abs(buffer))}")


# List available audio devices
print("Available audio devices:")
print(sd.query_devices())

# Get the default input device
default_device = sd.query_devices(kind='input')
print(f"\nDefault input device: {default_device['name']}")

# Create the animation
ani = FuncAnimation(fig, update_plot, frames=None, interval=30, blit=True)

# Start the audio stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=block_size)

# Start the stream and show the plot
with stream:
    plt.show()