import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hamming
import sounddevice as sd
from scipy.fft import fft, fftfreq
from mpl_toolkits.mplot3d import Axes3D

# Simulation parameters
demo_mode = True  # Set to False to capture from multiple microphones
Fs = 1000  # Sampling frequency (Hz)
freq_range = (250, 300)  # Frequency range for analysis (in Hz)
NO_SIGNAL_THRESHOLD = -80  # Threshold in dB to determine if there is no significant signal
SMOOTHING_FACTOR = 0.8  # Smoothing factor for exponential moving average

# Reference values for distance estimation
REF_DISTANCE = 1.0  # Reference distance in meters
REF_DB = 0  # Reference dB at 1 meter
ATTENUATION_PER_METER = 6  # Attenuation in dB per meter

# Distance between microphones (variable)
mic_distance = 10  # Distance between microphones in meters (can be changed)

# Maximum Z distance
MAX_Z_DISTANCE = 10  # Maximum z distance the system can detect

# Microphone positions (variables)
mic_positions = np.array([
    [0, 0, 0],  # Mic1 at (0, 0, 0)
    [mic_distance, 0, 0],  # Mic2 at (mic_distance, 0, 0)
    [0, mic_distance, 0],  # Mic3 at (0, mic_distance, 0)
    [mic_distance, mic_distance, 0],  # Mic4 at (mic_distance, mic_distance, 0)
])

# Function to estimate drone position based on trilateration
def estimate_position(mic_db_values):
    mic_db_values = np.array(mic_db_values)
    
    # Check if all signals are below the threshold
    if np.all(mic_db_values <= NO_SIGNAL_THRESHOLD):
        return np.array([mic_distance / 2, mic_distance / 2, 0])  # Default to center if no signal
    
    # Estimate distances based on microphone dB values
    distances = 10 ** ((REF_DB - mic_db_values) / ATTENUATION_PER_METER)
    
    # Avoid division by zero or negative distances
    distances = np.maximum(distances, 1e-6)
    
    # Extract positions
    x1, y1, z1 = mic_positions[0]
    x2, y2, z2 = mic_positions[1]
    x3, y3, z3 = mic_positions[2]
    x4, y4, z4 = mic_positions[3]
    
    d1, d2, d3, d4 = distances[:4]
    
    # Set up equations for trilateration using mics 1, 2, and 3
    # Equation A
    A = np.array([
        [x2 - x1, y2 - y1],
        [x3 - x1, y3 - y1]
    ])
    
    # Equation B
    B = 0.5 * np.array([
        d1**2 - d2**2 - x1**2 + x2**2 - y1**2 + y2**2,
        d1**2 - d3**2 - x1**2 + x3**2 - y1**2 + y3**2
    ])
    
    try:
        # Solve for x and y
        position_2d = np.linalg.solve(A, B)
        x, y = position_2d
    except np.linalg.LinAlgError:
        # Singular matrix, cannot solve
        x, y = mic_distance / 2, mic_distance / 2  # Default to center
    
    # Constrain x and y within bounds
    x = np.clip(x, 0, mic_distance)
    y = np.clip(y, 0, mic_distance)
    
    # Calculate z using mic4
    r_xy = np.sqrt((x - x4)**2 + (y - y4)**2)
    d4 = distances[3]
    z_squared = d4**2 - r_xy**2
    
    if z_squared >= 0:
        z = np.sqrt(z_squared)
    else:
        z = 0  # If negative due to noise, set z to 0
    
    # Constrain z to be within [0, MAX_Z_DISTANCE]
    z = np.clip(z, 0, MAX_Z_DISTANCE)
    
    # Return the estimated position
    return np.array([x, y, z])

# Function to capture real-time microphone input and calculate dB
def get_real_time_mic_input(channels, device=None):
    duration = 0.2  # Duration in seconds
    recording = sd.rec(int(duration * Fs), samplerate=Fs, channels=len(channels), dtype='float64', device=device)
    sd.wait()  # Wait until the recording is finished
    audio_time_data = recording  # This will be a 2D array with shape (samples, channels)
    
    mic_db_values = []
    for idx, ch in enumerate(channels):
        # Extract channel data
        channel_data = audio_time_data[:, idx]
        # Apply Hamming window
        windowed_data = channel_data * hamming(len(channel_data))
        # Apply FFT and normalize
        fft_result = np.abs(fft(windowed_data))[:len(windowed_data) // 2] / len(windowed_data)
        freqs = fftfreq(len(windowed_data), d=1/Fs)[:len(windowed_data) // 2]
        # Filter to desired frequency range
        freq_mask = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
        filtered_fft_result = fft_result[freq_mask]
        # Get peak value
        if len(filtered_fft_result) == 0 or np.max(filtered_fft_result) < 1e-6:
            mic_db = NO_SIGNAL_THRESHOLD  # No significant signal detected
        else:
            peak_value = np.max(filtered_fft_result)
            mic_db = 20 * np.log10(peak_value + 1e-10) + 30  # Convert to dB
        mic_db_values.append(mic_db)
    return mic_db_values

# Function to simulate mic dB values (for demo mode)
def simulate_mic_db(drone_position, mic_position):
    distance = np.linalg.norm(drone_position - mic_position)
    # Simulate dB value with attenuation
    mic_db = REF_DB - ATTENUATION_PER_METER * np.log10(distance / REF_DISTANCE)
    return mic_db

# Placeholder driver function for targeting system
def targeting_system_driver(position):
    # Currently does nothing
    # In future, this function will point a camera to the drone based on the position
    pass

# Function to start the live plot
def start_live_plot():
    input("Press Enter to start calculating position...")
    # Live plot setup
    plt.ion()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Initial smoothed position
    smoothed_position = np.array([mic_distance / 2, mic_distance / 2, 0])
    
    # Create initial scatter plot with one point
    sc, = ax.plot([smoothed_position[0]], [smoothed_position[1]], [smoothed_position[2]], 'ro')
    
    ax.set_xlim(0, mic_distance)
    ax.set_ylim(0, mic_distance)
    ax.set_zlim(0, MAX_Z_DISTANCE)
    ax.set_xlabel('X Position (m)')
    ax.set_ylabel('Y Position (m)')
    ax.set_zlabel('Z Position (m)')
    ax.set_title('Live Drone Position')
    
    # Function to handle plot close event
    def on_close(event):
        nonlocal running
        running = False
    
    fig.canvas.mpl_connect('close_event', on_close)
    
    # Device selection for multi-channel input
    if not demo_mode:
        print("Available audio devices:")
        print(sd.query_devices())
        device_index = int(input("Enter the device index for your multi-channel audio interface: "))
        channels = [1, 2, 3, 4]  # Channels to record from
    else:
        device_index = None
        channels = [1]  # Only one channel in demo mode
    
    # Main loop for live plotting
    t = 0
    running = True
    try:
        while running:
            t += 0.2  # Update interval
            if demo_mode:
                # Demo mode: Use real-time input for mic1 and fixed values for other mics
                mic_db_values = [get_real_time_mic_input([1])[0], -60, -60, -60]
                estimated_position = estimate_position(mic_db_values)
                
                # Apply smoothing
                smoothed_position = SMOOTHING_FACTOR * smoothed_position + (1 - SMOOTHING_FACTOR) * estimated_position
                
                # Print the updated dB values for all mics
                print(f"Updated dB values: Mic1: {mic_db_values[0]:.2f} dB, Mic2: {mic_db_values[1]:.2f} dB, Mic3: {mic_db_values[2]:.2f} dB, Mic4: {mic_db_values[3]:.2f} dB")
                
                # Print the estimated position
                print(f"Estimated Position - X: {smoothed_position[0]:.2f} m, Y: {smoothed_position[1]:.2f} m, Z: {smoothed_position[2]:.2f} m")
                
                # Call the targeting system driver with the estimated position
                targeting_system_driver(smoothed_position)
            else:
                # Capture real-time input from four microphones
                mic_db_values = get_real_time_mic_input(channels, device=device_index)
                estimated_position = estimate_position(mic_db_values)
                
                # Apply smoothing
                smoothed_position = SMOOTHING_FACTOR * smoothed_position + (1 - SMOOTHING_FACTOR) * estimated_position
                
                # Print the updated dB values for all mics
                print(f"Updated dB values: Mic1: {mic_db_values[0]:.2f} dB, Mic2: {mic_db_values[1]:.2f} dB, Mic3: {mic_db_values[2]:.2f} dB, Mic4: {mic_db_values[3]:.2f} dB")
                
                # Print the estimated position
                print(f"Estimated Position - X: {smoothed_position[0]:.2f} m, Y: {smoothed_position[1]:.2f} m, Z: {smoothed_position[2]:.2f} m")
                
                # Call the targeting system driver with the estimated position
                targeting_system_driver(smoothed_position)
    
            # Update live plot
            sc.set_data(smoothed_position[0], smoothed_position[1])
            sc.set_3d_properties(smoothed_position[2])
            plt.draw()
            plt.pause(0.2)  # Adjust to control the speed of the live update
    except KeyboardInterrupt:
        print("Live plotting interrupted by user.")
    finally:
        plt.ioff()
        plt.show()

# Start live plotting
start_live_plot()
