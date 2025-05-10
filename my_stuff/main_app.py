#!/usr/bin/env python3
###############################################################################
###############################################################################
#
# Basic example of using bladeRF Python bindings for full duplex TX/RX.
# Review the companion my_configuration.ini to adjust configuration options.
#
###############################################################################

from datetime import datetime
from multiprocessing.pool import ThreadPool
from configparser import ConfigParser
import pyqtgraph as pg
from bladerf import _bladerf
import time
import queue
import os
from configparser import ConfigParser
import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg


current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Format: "YYYY-MM-DD_HH-MM-SS"
bw_file_path = f"./bw_summing_data_{current_datetime}.bin"
stacked_file_path = f"./exposure_stacking_data_{current_datetime}.bin"

# Bandwidth summing queue (optional) and stacking memory
bw_powers = []         # All power measurements
stacked_powers = []    # Sliding window or average of power

POWER_WINDOW = 100     # Number of points to retain in stacked plot
power_times = []       # Timestamps of each power measure (for x-axis)

# Load configuration
config = ConfigParser()
config.read('/Users/giannis/PycharmProjects/final_radiotelescope/my_stuff/my_configuration.ini')


# Get parameters from config
N_SAMPLES = int(float(config['bladerf2-rx']['rx_num_samples']))
BW = float(config['bladerf2-rx']['rx_bandwidth'])      # Hz
rx_freq = float(config['bladerf2-rx']['rx_frequency']) # Hz
fs = float(config['bladerf2-rx']['rx_samplerate'])     # Hz


# TODO ignore this class
# TODO create another thread that constantly checks solely the rx buffer for data and plots them
# TODO main folder should be the rx forlder so that overwriting becomes easier
# TODO make a separate folder (permanent data logger) in which you append the current rx_buffer-folder data
# TODO refresh the plot every 1 sec for starters and clear the buffer every...




# Start the Qt event loop
# app.exec_()


prev_fft_db = None
alpha = 0.5  # Smoothing factor, adjust as needed
# The smoothing factor controls how much the new FFT data affects the plot.
# 	•	If alpha = 1.0: you show only the new FFT (no smoothing).
# 	•	If alpha = 0.0: you show only the old plot (fully frozen).
# 	•	If alpha = 0.2: you blend 20% of the new FFT with 80% of the previous plot.
#
# It reduces flickering by slowing down how fast the display reacts to small changes.





# =============================================================================
# Close the device and exit
# =============================================================================
def shutdown(error=0, board=None):
    print("Shutting down with error code: " + str(error))
    if (board != None):
        board.close()
    sys.exit(error)


# =============================================================================
# Version information
# =============================================================================
def print_versions(device=None):
    print("libbladeRF version: " + str(_bladerf.version()))
    if (device != None):
        try:
            print("Firmware version: " + str(device.get_fw_version()))
        except:
            print("Firmware version: " + "ERROR")
            raise

        try:
            print("FPGA version: " + str(device.get_fpga_version()))
        except:
            print("FPGA version: " + "ERROR")
            raise

    return 0


# =============================================================================
# Search for a bladeRF device attached to the host system
# Returns a bladeRF device handle.
# =============================================================================
def probe_bladerf():
    device = None
    print("Searching for bladeRF devices...")
    try:
        devinfos = _bladerf.get_device_list()
        if (len(devinfos) == 1):
            device = "{backend}:device={usb_bus}:{usb_addr}".format(**devinfos[0]._asdict())
            print("Found bladeRF device: " + str(device))
        if (len(devinfos) > 1):
            print("Unsupported feature: more than one bladeRFs detected.")
            print("\n".join([str(devinfo) for devinfo in devinfos]))
            shutdown(error=-1, board=None)
    except _bladerf.BladeRFError:
        print("No bladeRF devices found.")
        pass

    return device


# =============================================================================
# Load FPGA
# =============================================================================
def load_fpga(device, image):
    image = os.path.abspath(image)

    if (not os.path.exists(image)):
        print("FPGA image does not exist: " + str(image))
        return -1

    try:
        print("Loading FPGA image: " + str(image))
        device.load_fpga(image)
        fpga_loaded = device.is_fpga_configured()
        fpga_version = device.get_fpga_version()

        if (fpga_loaded):
            print("FPGA successfully loaded. Version: " + str(fpga_version))

    except _bladerf.BladeRFError:
        print("Error loading FPGA.")
        raise

    return 0


# =============================================================================
# TRANSMIT
# =============================================================================

# =============================================================================
# RECEIVE
# =============================================================================



def update_bw_plots():
    global bw_powers, stacked_powers, power_times

    if not bw_powers:
        return

    # 1) Convert to dB
    powers_db = 10 * np.log10(np.array(bw_powers) + 1e-12)
    times_all = np.array(power_times)

    # 2) Convert time to relative time in seconds (or milliseconds if needed)
    times_relative = times_all - times_all[0]  # Time in seconds since the first sample
    # For milliseconds: times_relative = (times_all - times_all[0]) * 1000

    # 3) Trim both to the last POWER_WINDOW samples
    if len(powers_db) > POWER_WINDOW:
        powers_db = powers_db[-POWER_WINDOW:]
        times_relative = times_relative[-POWER_WINDOW:]

    # 4) Plot raw data
    bw_curve.setData(times_relative, powers_db)

    # 5) Save Bandwidth Summing Data to file
    with open(bw_file_path, 'ab') as bw_file:
        np.array([times_relative, powers_db], dtype=np.float64).tofile(bw_file)

    # 6) If you have at least as many points as your moving-average window:
    window = 10
    if len(powers_db) >= window:
        kernel = np.ones(window) / window
        stacked_db = np.convolve(powers_db, kernel, mode='valid')
        times_stacked = times_relative[window - 1:]  # Adjust times for stacked plot
    else:
        stacked_db = np.array([])
        times_stacked = np.array([])

    # 7) Plot the exposure-stacked curve
    stacked_curve.setData(times_stacked, stacked_db)

    # 8) Save Exposure Stacking Data to file
    with open(stacked_file_path, 'ab') as stacked_file:
        np.array([times_stacked, stacked_db], dtype=np.float64).tofile(stacked_file)

    # Dynamic Y-axis scaling for Bandwidth-Summed Plot
    max_power = np.max(powers_db)
    min_power = np.min(powers_db)
    bw_plot.setYRange(min_power - 10, max_power + 10)  # Add some padding for clarity

    # Dynamic Y-axis scaling for Stacked Power Plot
    if len(stacked_db) > 0:
        max_stacked = np.max(stacked_db)
        min_stacked = np.min(stacked_db)
        stacked_plot.setYRange(min_stacked - 10, max_stacked + 10)  # Add some padding for clarity




# Global buffer and lock for thread-safe access
import threading
TARGET_FFT_SIZE = 8192*4  # or 4096 for faster updates
shared_buffer = queue.Queue(maxsize=TARGET_FFT_SIZE)
buffer_lock = threading.Lock()

def compute_and_plot_fft(buffer, curve, sample_rate, center_freq_hz):
    global prev_fft_db
    """
    Extracts FFT from buffer and updates the plot.

    Args:
        buffer (np.array): Complex I/Q samples.
        curve (pyqtgraph.PlotCurveItem): The curve to update.
        sample_rate (float): Sample rate in Hz.
        center_freq_hz (float): Center frequency in Hz.
    """
    # TODO clear buffer after plot


    if np.any(buffer):  # Skip empty buffer
        buffer = buffer - np.mean(buffer)
        window = np.hanning(len(buffer))  # Or np.blackman, np.hamming, etc.
        buffer_windowed = buffer * window
        buffer_windowed = buffer_windowed - np.mean(buffer_windowed) # Remove DC offset (optional but useful)


        # Normalize window to preserve signal power
        normalization_factor = np.sum(window) / len(window)
        fft_vals = np.fft.fftshift(np.fft.fft(buffer_windowed))
        fft_vals = fft_vals / (len(buffer) * normalization_factor)  # Normalize properly

        # Convert to dB scale
        fft_db = 20 * np.log10(np.abs(fft_vals) + 1e-12)

        #computes the frequency bins corresponding to the FFT of your signal buffer:
        freqs = np.fft.fftshift(np.fft.fftfreq(len(buffer), 1 / sample_rate)) # (1 / sample_rate) == The time between samples, or sample period (T).
        # np.fft.fftfreq: This returns an array of frequency bins in Hz: [0, 1, 2, ..., fs/2 - 1, -fs/2, ..., -1] ...
        # ...spanning from 0 up to just below sample_rate, then wrapping around to negative frequencies.

        #  np.fft.fftshift(...): Reorders the frequencies to center 0 Hz

        freqs_mhz = (freqs + center_freq_hz) / 1e6  # Convert to MHz


        ## So that only the fc±BW/2 spectrum is shown
        bw_half_mhz = BW / 2 / 1e6  # Convert to MHz
        valid_indices = ((freqs_mhz >= (center_freq_hz / 1e6 - bw_half_mhz)) &
                         (freqs_mhz <= (center_freq_hz / 1e6 + bw_half_mhz)))


        if prev_fft_db is None:
            smoothed_fft_db = fft_db
        else:
            smoothed_fft_db = alpha * fft_db + (1 - alpha) * prev_fft_db

        prev_fft_db = smoothed_fft_db
        # curve.setData(freqs_mhz, smoothed_fft_db)
        curve.setData(freqs_mhz[valid_indices], smoothed_fft_db[valid_indices])





def update_fft_gui():
    global shared_buffer, buffer_lock, fft_curve, fs, rx_freq

    # Initialize data to None to handle case where the queue is empty
    data = None

    try:
        # Try to get data from the queue without blocking
        data = shared_buffer.get_nowait()  # Non-blocking, get data
    except queue.Empty:
        pass  # If the queue is empty, just skip

    if data is not None:
        compute_and_plot_fft(data, fft_curve, fs, rx_freq)

    # Optional: If you want to clear the queue after processing (to free memory), you can do this
    # while not shared_buffer.empty():
    #     shared_buffer.get()

    # TODO clear buffer after plot




def receive(device, channel: int, freq: int, rate: int, gain: int,
            tx_start=None, rx_done=None,
            rxfile: str = '', num_samples: int = 1024):

    global shared_buffer, buffer_lock  # Access the global buffer


    if (device == None):
        print("RX: Invalid device handle.")
        return -1

    if (channel == None):
        print("RX: Invalid channel.")
        return -1

    # Configure BladeRF
    ch = device.Channel(channel)
    ch.frequency = freq
    ch.sample_rate = rate
    ch.gain = gain

    # Setup synchronous stream
    device.sync_config(layout=_bladerf.ChannelLayout.RX_X1,
                       fmt=_bladerf.Format.SC16_Q11,
                       num_buffers=16,
                       buffer_size=8192,
                       num_transfers=8,
                       stream_timeout=3500)

    # Enable module
    # print("RX: Start")
    ch.enable = True

    # Create receive buffer
    bytes_per_sample = 4
    buf = bytearray(num_samples * bytes_per_sample)
    num_samples_read = 0

    # Tell TX thread to begin
    if (tx_start != None):
        tx_start.set()

    # Save the samples
    with open(rxfile, 'wb') as outfile:
        while True:
            if num_samples > 0 and num_samples_read == num_samples:
                break
            elif num_samples > 0:
                num = min(len(buf) // bytes_per_sample,
                          num_samples - num_samples_read)
            else:
                num = len(buf) // bytes_per_sample

            # This reads num samples from the receiver (BladeRF) into the buf buffer.
            device.sync_rx(buf, num)
            # This keeps track of how many samples have been read so far.
            num_samples_read += num

            # Write to file
            outfile.write(buf[:num * bytes_per_sample])

            # converts the raw byte data in buf into 16-bit integers, representing the received signal samples (I/Q data).
            data = np.frombuffer(buf[:num * bytes_per_sample], dtype=np.int16)

            # Converts the interleaved real (I) and imaginary (Q) parts of the signal into a complex number array (I + jQ).
            iq = data[::2] + 1j * data[1::2]

            # === Bandwidth Summing (raw full IQ) ===
            power = np.sum(np.abs(iq) ** 2) / len(iq)
            bw_powers.append(power)
            power_times.append(time.time())

            # Downsample for GUI only
            step = max(len(iq) // TARGET_FFT_SIZE, 1)
            iq_downsampled = iq[::step]

            # Queue for FFT
            with buffer_lock:
                shared_buffer.put(iq_downsampled)

            # Downsample the data for real-time plotting (e.g., take 1 out of every 500 samples)
            # Adjust step based on how many you want to show — aim for 4K or 8K points max

            step = max(len(iq) // TARGET_FFT_SIZE, 1)
            iq_downsampled = iq[::step]


            # Update shared_buffer safely
            with buffer_lock:
                # Updates shared_buffer with the new complex samples (I/Q), ensuring the buffer size does not exceed its allocated length.
                # shared_buffer[:len(iq_downsampled)] = iq_downsampled[:len(shared_buffer)]
                shared_buffer.put(iq_downsampled)  # This will add the data to the queue

    # Disable module
    # print("RX: Stop")
    ch.enable = False

    if (rx_done != None):
        rx_done.set()

    print("RX: Done")

    return 0


# =============================================================================
# Load Configuration
# =============================================================================



# Set libbladeRF verbosity level
verbosity = config.get('common', 'libbladerf_verbosity').upper()
if (verbosity == "VERBOSE"):
    _bladerf.set_verbosity(0)
elif (verbosity == "DEBUG"):
    _bladerf.set_verbosity(1)
elif (verbosity == "INFO"):
    _bladerf.set_verbosity(2)
elif (verbosity == "WARNING"):
    _bladerf.set_verbosity(3)
elif (verbosity == "ERROR"):
    _bladerf.set_verbosity(4)
elif (verbosity == "CRITICAL"):
    _bladerf.set_verbosity(5)
elif (verbosity == "SILENT"):
    _bladerf.set_verbosity(6)
else:
    print("Invalid libbladerf_verbosity specified in configuration file:",
          verbosity)
    shutdown(error=-1, board=None)

# =============================================================================
# BEGIN !!!
# =============================================================================

uut = probe_bladerf()

if (uut == None):
    print("No bladeRFs detected. Exiting.")
    shutdown(error=-1, board=None)

b = _bladerf.BladeRF(uut)
board_name = b.board_name
print(board_name)
fpga_size = b.fpga_size

if (config.getboolean(board_name + '-load-fpga', 'enable')):
    print("Loading FPGA...")
    try:
        status = load_fpga(b, config.get(board_name + '-load-fpga',
                                         'image_' + str(fpga_size) + 'kle'))
    except:
        print("ERROR loading FPGA.")
        raise

    if (status < 0):
        print("ERROR loading FPGA.")
        shutdown(error=status, board=b)
else:
    print("Skipping FPGA load due to configuration setting.")

status = print_versions(device=b)

# TODO: can we have >1 rx/tx pool workers because 2x2 MIMO?
rx_pool = ThreadPool(processes=1)
tx_pool = ThreadPool(processes=1)




# TODO append to rx data buffer
# TODO check if rx buffer is full
# TODO check if main file is full



def rx_loop():
    global rx_freq
    while True:
        rx_ch = _bladerf.CHANNEL_RX(config.getint('bladerf2-rx', 'rx_channel'))
        rx_freq = int(config.getfloat('bladerf2-rx', 'rx_frequency'))
        rx_rate = int(config.getfloat('bladerf2-rx', 'rx_samplerate'))
        rx_gain = int(config.getfloat('bladerf2-rx', 'rx_gain'))
        rx_ns = int(config.getfloat('bladerf2-rx', 'rx_num_samples'))
        # rx_file = config.get('bladerf2-rx', 'rx_file')
        rx_file_path = f'./aaanew_my_rx_samples_{current_datetime}.bin'

        # Assign the new path to the config
        config.set('bladerf2-rx', 'rx_file', rx_file_path)




        status = receive(
            device=b,
            channel=rx_ch,
            freq=rx_freq,
            rate=rx_rate,
            gain=rx_gain,
            tx_start=None,
            rx_done=None,
            rxfile=rx_file_path,
            num_samples=rx_ns
        )

        if status < 0:
            print(f"Receive operation failed with error {status}")
            break
        time.sleep(0.05)
        print(f"bw_powers: {bw_powers[-5:]}")  # Print last 5 entries
        print(f"power_times: {power_times[-5:]}")



if __name__ == "__main__":
    # TODO start the GUI



    app = QtWidgets.QApplication(sys.argv)

    # Set up the GUI as usual
    win = pg.GraphicsLayoutWidget(title="Live Signal and FFT Plot")
    win.show()

    # Setup the FFT plot
    fft_plot = win.addPlot(title="FFT of Signal")
    fft_curve = fft_plot.plot(pen='g')

    # Label axes
    fft_plot.setLabel('bottom', 'Frequency (MHz)')
    fft_plot.setLabel('left', 'Magnitude (dB)')

    # FIX THE AXES — no auto-rescale
    center_freq_mhz = rx_freq / 1e6  # Convert to MHz
    bandwidth_mhz = BW / 1e6  # Convert to MHz
    fft_plot.setXRange(center_freq_mhz - bandwidth_mhz / 2,
                       center_freq_mhz + bandwidth_mhz / 2, padding=0)

    fft_plot.setYRange(-100, 0, padding=0)  # dB range — adjust based on your signal

    # Add Bandwidth-Summed Power Plot
    win.nextRow()
    bw_plot = win.addPlot(title="Bandwidth Summed Power (dB)")
    bw_curve = bw_plot.plot(pen='y')
    bw_plot.setLabel('bottom', 'Time (s)')
    bw_plot.setLabel('left', 'Power (dB)')
    bw_plot.setYRange(-100, 0)

    # Add Exposure Stacked Power Plot
    win.nextRow()
    stacked_plot = win.addPlot(title="Exposure Stacked Power (dB avg)")
    stacked_curve = stacked_plot.plot(pen='c')
    stacked_plot.setLabel('bottom', 'Time (s)')
    stacked_plot.setLabel('left', 'Smoothed Power (dB)')
    stacked_plot.setYRange(-100, 0)

    bw_timer = QtCore.QTimer()
    bw_timer.timeout.connect(update_bw_plots)
    bw_timer.start(50)  # Every 0.5 sec or as needed

    # Set up timer to refresh GUI
    timer = QtCore.QTimer()
    timer.timeout.connect(update_fft_gui)  # <- Connect to your function
    timer.start(50)  # Refresh every 50ms

    # Start RX thread
    rx_thread = threading.Thread(target=rx_loop, daemon=True)
    rx_thread.start()



    # TODO start threads!!!!
    # ✔️ Only rx_thread and QTimer are needed now — fft_plot_worker is no longer used

    # ❌ Removed: fft_plot_worker thread
    # The GUI update via QTimer now handles FFT computation and plotting safely

    # Start GUI loop (this must stay in the main thread!)
    sys.exit(app.exec_())

    # if status < 0:
    #     print(f"Receive operation failed with error {status}")
    #     break  # Exit loop if receive failed

    # Refresh the GUI by updating the plot
    # Since GUI updates must happen in the main thread, we invoke the method
    # that updates the plot
"""
    The return value of status is given by calling the transmit def, which always returns
    an integer. Thus, simply referencing transmit as an argument of tx_pool.apply_async()
    "activates" the transmit def, which shall return an integer.

    !!!!!!!!!!!
"""