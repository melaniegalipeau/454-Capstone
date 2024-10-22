import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, hamming
import sounddevice as sd
import time
from scipy.fft import fft, fftfreq

# Simulation parameters
demo_mode = True  # Set to True for demo mode (single mic input simulation)
Fs = 1000  # Sampling frequency (Hz)
mic_positions = np.array([[0, 0], [0, 10], [10, 0], [10, 10]])  # Microphones at corners of a 10m x 10m square
freq_range = (100, 1000)  # Frequency range for analysis (in Hz)

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
def estimate_position(magnitudes):
    if np.sum(magnitudes) == 0:
        return np.array([0, 10])  # Default to corner if no signal

    distances = estimate_distances(magnitudes)
    
    # Using an iterative method to find the intersection point of circles centered at each microphone
    A = 2 * (mic_positions[1:] - mic_positions[0])
    b = distances[0] ** 2 - distances[1:] ** 2 + np.sum(mic_positions[1:] ** 2, axis=1) - np.sum(mic_positions[0] ** 2)
    estimated_position = np.linalg.lstsq(A, b, rcond=None)[0]
    
    # Constrain the estimated position to be within the bounds of the area
    estimated_position = np.clip(estimated_position, 0, 10)
    return estimated_position

# Function to capture real-time microphone input and calculate dB using peak value within frequency range
def get_real_time_mic_input():
    duration = 0.1  # Reduced duration for faster updates (0.1 seconds)
    recording = sd.rec(int(duration * Fs), samplerate=Fs, channels=1, dtype='float64')
    sd.wait()  # Wait until the recording is finished
    audio_time_data = recording.flatten()

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
        return -120.0  # No significant signal detected
    peak_value = np.max(filtered_fft_result)
    return 20 * np.log10(peak_value + 1e-10)  # Convert to dB

# Function to simulate drone signal for given position (placeholder implementation)
def get_drone_signal(t, drone_position):
    # Placeholder implementation: generate synthetic microphone signals based on drone position
    distances = np.linalg.norm(mic_positions - drone_position, axis=1)
    signal_strengths = REF_DB - 20 * np.log10(distances + 1e-6)  # Inverse relation to distance in dB
    return signal_strengths

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

    # Main loop for live plotting
    t = 0
    running = True
    try:
        while running:
            t += 1 / Fs
            if demo_mode:
                # Demo mode: Use real-time input for 1 mic and fixed values for other 3 mics
                mic_signal_db = get_real_time_mic_input()  # Get the signal in dB
                fixed_db = -10  # Fixed dB value for the other 3 microphones, should not exceed 0 dB
                mic_signals = np.array([fixed_db, fixed_db, fixed_db, mic_signal_db])
                estimated_position = estimate_position(np.maximum(mic_signals, 0))
                
                # Print the updated dB within the BPF range for the real-time mic every second
                if int(t) % 1 == 0:
                    print(f"Updated dB for real-time mic: {mic_signal_db:.2f} dB")
            else:
                # Simulate drone movement (you can replace this with actual microphone input)
                drone_position = np.array([5 + 2 * np.sin(0.1 * t), 5 + 2 * np.cos(0.1 * t)])
                mic_signals = get_drone_signal(t, drone_position)
                
                # Filter the signals to focus on frequency range
                filtered_signals = bandpass_filter(mic_signals, freq_range[0], freq_range[1], Fs)
                
                # Estimate drone position based on filtered signal magnitudes
                estimated_position = estimate_position(np.abs(filtered_signals))
            
            # Update live plot
            sc.set_offsets([estimated_position])
            plt.pause(0.1)  # Adjust to control the speed of the live update
    except KeyboardInterrupt:
        print("Live plotting interrupted by user.")
    finally:
        plt.ioff()
        plt.show()

# Start live plotting
start_live_plot()