import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hamming
import sounddevice as sd
import time
from scipy.fft import fft, fftfreq
from collections import deque

# Simulation parameters
demo_mode = True  # Set to True for demo mode (single mic input simulation)
Fs = 1000  # Sampling frequency (Hz)
freq_range = (250, 300)  # Frequency range for analysis (in Hz)
NO_SIGNAL_THRESHOLD = -80  # Threshold in dB to determine if there is no significant signal
SMOOTHING_FACTOR = 0.8  # Smoothing factor for exponential moving average

# Butterworth Bandpass Filter for specified frequency range
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

# Function to estimate distances based on microphone magnitudes
# Using a model where sound attenuates by approximately 6 dB per doubling of distance
# Assuming a reference distance of 1 meter
REF_DISTANCE = 1.0  # Reference distance in meters
REF_DB = 0  # Reference dB at 1 meter
def estimate_distances(magnitudes):
    distances = REF_DISTANCE * 10 ** ((REF_DB - magnitudes) / 20)
    return distances

# Function to estimate drone position based on triangulation
# Using simplified triangulation with mic1, mic2, mic3, and mic4
# Mic positions are assumed at (0, 0) for mic1, (0, 10) for mic2, (10, 0) for mic3, and (10, 10) for mic4 respectively
# Updated to increase sensitivity and correctly account for signal attenuation
ATTENUATION_PER_METER = 6  # Attenuation in dB per meter
def estimate_position(mic1_db, mic2_db, mic3_db, mic4_db):
    # Check if all signals are below the threshold
    if mic1_db <= NO_SIGNAL_THRESHOLD and mic2_db <= NO_SIGNAL_THRESHOLD and mic3_db <= NO_SIGNAL_THRESHOLD and mic4_db <= NO_SIGNAL_THRESHOLD:
        return np.array([5, 5])  # Default to midpoint if no signal

    # Estimate distances based on microphone dB values
    dist_mic1 = 10 ** ((REF_DB - mic1_db) / ATTENUATION_PER_METER)
    dist_mic2 = 10 ** ((REF_DB - mic2_db) / ATTENUATION_PER_METER)
    dist_mic3 = 10 ** ((REF_DB - mic3_db) / ATTENUATION_PER_METER)
    dist_mic4 = 10 ** ((REF_DB - mic4_db) / ATTENUATION_PER_METER)

    # Avoid division by zero
    epsilon = 1e-10
    dist_mic1 = max(dist_mic1, epsilon)
    dist_mic2 = max(dist_mic2, epsilon)
    dist_mic3 = max(dist_mic3, epsilon)
    dist_mic4 = max(dist_mic4, epsilon)

    # Calculate weights inversely proportional to distances
    weight_mic1 = 1 / dist_mic1
    weight_mic2 = 1 / dist_mic2
    weight_mic3 = 1 / dist_mic3
    weight_mic4 = 1 / dist_mic4

    total_weight = weight_mic1 + weight_mic2 + weight_mic3 + weight_mic4

    # Microphone positions
    x_mic1, y_mic1 = 0, 0
    x_mic2, y_mic2 = 0, 10
    x_mic3, y_mic3 = 10, 0
    x_mic4, y_mic4 = 10, 10

    # Calculate weighted average positions
    x_position = (weight_mic1 * x_mic1 + weight_mic2 * x_mic2 + weight_mic3 * x_mic3 + weight_mic4 * x_mic4) / total_weight
    y_position = (weight_mic1 * y_mic1 + weight_mic2 * y_mic2 + weight_mic3 * y_mic3 + weight_mic4 * y_mic4) / total_weight

    # Constrain the estimated position to be within the bounds of the area
    estimated_position = np.clip([x_position, y_position], 0, 10)
    return estimated_position

# Function to capture real-time microphone input and calculate dB using peak value within frequency range
def get_real_time_mic_input():
    duration = 0.2  # Increased duration for better averaging (0.2 seconds)
    recording = sd.rec(int(duration * Fs), samplerate=Fs, channels=1, dtype='float64')
    sd.wait()  # Wait until the recording is finished
    audio_time_data = recording.flatten()

    # Debug: Print raw RMS value
    rms_value = np.sqrt(np.mean(audio_time_data ** 2))
    print(f"Raw RMS value: {rms_value:.6f}")

    # Apply Hamming window to reduce spectral leakage
    windowed_data = audio_time_data * hamming(len(audio_time_data))

    # Apply FFT to get frequency content and normalize by the length of the signal
    fft_result = np.abs(fft(windowed_data))[:len(windowed_data) // 2] / len(windowed_data)
    freqs = fftfreq(len(windowed_data), d=1/Fs)[:len(windowed_data) // 2]

    # Filter to only include the desired frequency range
    freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    filtered_fft_result = fft_result[freq_mask]

    # Get the peak value within the frequency range
    if len(filtered_fft_result) == 0 or np.max(filtered_fft_result) < 1e-6:
        return NO_SIGNAL_THRESHOLD  # No significant signal detected
    peak_value = np.max(filtered_fft_result)
    return 20 * np.log10(peak_value + 1e-10) + 30  # Convert to dB and increase gain adjustment for higher sensitivity

# Function to start the live plot
def start_live_plot():
    input("Press Enter to start calculating position...")
    # Live plot setup
    plt.ion()
    fig, ax = plt.subplots()
    sc = ax.scatter([0], [10], c='red')  # Start at default position
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_title('Live Drone Position')

    # Function to handle plot close event
    def on_close(event):
        nonlocal running
        running = False

    fig.canvas.mpl_connect('close_event', on_close)

    # Smoothing with exponential moving average
    smoothed_position = np.array([5, 5])

    # Main loop for live plotting
    t = 0
    running = True
    try:
        while running:
            t += 1 / Fs
            if demo_mode:
                # Demo mode: Use real-time input for mic1 and fixed values for other mics
                mic1_db = get_real_time_mic_input()  # Get the signal in dB for mic1
                mic2_db = -60  # Fixed dB value for mic2
                mic3_db = -60  # Fixed dB value for mic3
                mic4_db = -60  # Fixed dB value for mic4

                estimated_position = estimate_position(mic1_db, mic2_db, mic3_db, mic4_db)
                
                # Apply smoothing
                smoothed_position = SMOOTHING_FACTOR * smoothed_position + (1 - SMOOTHING_FACTOR) * estimated_position
                
                # Print the updated dB within the BPF range for the real-time mic every second
                if int(t) % 1 == 0:
                    print(f"Updated dB for real-time mic: {mic1_db:.2f} dB")
            else:
                # Simulate drone movement (you can replace this with actual microphone input)
                drone_position = np.array([5 + 2 * np.sin(0.1 * t), 5 + 2 * np.cos(0.1 * t)])
                mic_signals = get_drone_signal(t, drone_position)
                
                # Filter the signals to focus on frequency range
                filtered_signals = bandpass_filter(mic_signals, freq_range[0], freq_range[1], Fs)
                
                # Estimate drone position based on filtered signal magnitudes
                estimated_position = estimate_position(np.abs(filtered_signals))
                
                # Apply smoothing
                smoothed_position = SMOOTHING_FACTOR * smoothed_position + (1 - SMOOTHING_FACTOR) * estimated_position
            
            # Update live plot
            sc.set_offsets([smoothed_position])
            plt.pause(0.2)  # Adjust to control the speed of the live update
    except KeyboardInterrupt:
        print("Live plotting interrupted by user.")
    finally:
        plt.ioff()
        plt.show()

# Start live plotting
start_live_plot()