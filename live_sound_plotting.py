# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 12:02:56 2024

@author: brand
"""
import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
duration = 10  # sec
sample_rate = 48000  # samples per second
block_size = 1024  # num samples per block

# Create a buffer to store the audio data
buffer = np.zeros(sample_rate * duration)

# Create the plot
fig, ax = plt.subplots()
line, = ax.plot([], [], lw=2)
ax.set_xlim(0, duration)
ax.set_ylim(-1, 1)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Microphone Amplitude (Line In)')
ax.grid(True)

# Function to update the plot
def update_plot(frame):
    global buffer
    line.set_data(np.linspace(0, duration, len(buffer)), buffer)
    return line,

def audio_callback(indata, frames, time, status):
    global buffer
    if status:
        print(status)
    buffer = np.roll(buffer, -len(indata))
    buffer[-len(indata):] = indata[:, 0]

# Create the animation
ani = FuncAnimation(fig, update_plot, frames=None, interval=30, blit=True)

# Start the audio stream
stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=sample_rate, blocksize=block_size)

# Start the stream and show the plot
with stream:
    plt.show()