#!/usr/bin/env python3
###############################################################################
###############################################################################
#
# Basic example of using bladeRF Python bindings for full duplex TX/RX.
# Review the companion my_configuration.ini to adjust configuration options.
#
###############################################################################
import signal
import threading
from datetime import datetime
from multiprocessing.pool import ThreadPool
from configparser import ConfigParser
from bladerf import _bladerf
import time
import queue
import os
from configparser import ConfigParser
import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg
from collections import deque
from bladerf._bladerf import Correction



# Add these manually if they're not exposed by your bindings
BLADERF_MODULE_RX = 0
BLADERF_MODULE_TX = 1

BLADERF_CORR_DC_I = 0
BLADERF_CORR_DC_Q = 1
BLADERF_CORR_PHASE = 2
BLADERF_CORR_GAIN = 3

CONCENTRATION_FREQUENCY = 2.4e9



# Global shutdown flag
stop_event = threading.Event()

current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Format: "YYYY-MM-DD_HH-MM-SS"
bw_file_path = f"./logs/bw_summing_data_{current_datetime}.bin"
stacked_file_path = f"./logs/exposure_stacking_data_{current_datetime}.bin"

# Bandwidth summing queue (optional) and stacking memory
stacked_powers = []    # Sliding window or average of power


# Defines BW for bw summing
BW_FOR_BW_SUMMING = 2e6  # for example, 2 MHz

POWER_WINDOW = 100     # Number of points to retain in stacked plot

# Load configuration
config = ConfigParser()
config.read('my_stuff/my_configuration.ini')


# Get parameters from config
N_SAMPLES = int(float(config['bladerf2-rx']['rx_num_samples']))
BW = float(config['bladerf2-rx']['rx_bandwidth'])      # Hz
rx_freq = float(config['bladerf2-rx']['rx_frequency']) # Hz
fs = float(config['bladerf2-rx']['rx_samplerate'])     # Hz
rx_channel0 = int(config['bladerf2-rx']['rx_channel0'])
rx_channel1 = int(config['bladerf2-rx']['rx_channel1'])



# TODO ignore this class
# TODO create another thread that constantly checks solely the rx buffer for data and plots them
# TODO main folder should be the rx forlder so that overwriting becomes easier
# TODO make a separate folder (permanent data logger) in which you append the current rx_buffer-folder data
# TODO refresh the plot every 1 sec for starters and clear the buffer every...




# Start the Qt event loop
# app.exec_()


prev_fft_db = None
alpha = 1  # Smoothing factor, adjust as needed
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

my_bladerf = _bladerf.BladeRF(uut)



board_name = my_bladerf.board_name
print(board_name)
fpga_size = my_bladerf.fpga_size

if (config.getboolean(board_name + '-load-fpga', 'enable')):
    print("Loading FPGA...")
    try:
        status = load_fpga(my_bladerf, config.get(board_name + '-load-fpga',
                                         'image_' + str(fpga_size) + 'kle'))
    except:
        print("ERROR loading FPGA.")
        raise

    if (status < 0):
        print("ERROR loading FPGA.")
        shutdown(error=status, board=my_bladerf)
else:
    print("Skipping FPGA load due to configuration setting.")

status = print_versions(device=my_bladerf)

# TODO: can we have >1 rx/tx pool workers because 2x2 MIMO?
rx_pool = ThreadPool(processes=1)
tx_pool = ThreadPool(processes=1)




# TODO append to rx data buffer
# TODO check if rx buffer is full
# TODO check if main file is full



# =============================================================================
# TRANSMIT
# =============================================================================

# =============================================================================
# RECEIVE
# =============================================================================


exposure_stack_sum = None
exposure_stack_count = 0


# Initialize buffers somewhere globally or in class:
bw_powers = deque(maxlen=1000)
power_times = deque(maxlen=1000)
INTEGRATION_TIME = 10.0  # seconds per integration block
integration_start = time.time()
save_dir = "bw_spectra"  # directory to save spectra files
os.makedirs(save_dir, exist_ok=True)

def save_integrated_spectrum():
    global bw_powers, power_times, integration_start

    if len(bw_powers) == 0:
        return  # nothing to save

    # Stack deque elements into a 2D array
    stacked_spectra = np.vstack(bw_powers)  # shape (n_spectra, spectrum_length)
    summed_spectrum = np.sum(stacked_spectra, axis=0)
    avg_spectrum = summed_spectrum / len(bw_powers)

    timestamp = int(integration_start)
    filename = os.path.join(save_dir, f"spectrum_{timestamp}.npy")

    np.save(filename, avg_spectrum)
    print(f"Saved integrated spectrum to {filename}")

    # Clear deques
    bw_powers.clear()
    power_times.clear()
    integration_start = time.time()
















# Accumulator for summing power over time with respect to frequency

cumulative_power_sum = None
num_summed = 0
max_power_sum = 0
min_power_sum = 9999
check3 = False
check4 = False

def update_bw_plots():
    global bw_powers, stacked_powers, power_times, cumulative_power_sum, num_summed, max_power_sum, min_power_sum, check3, check4, integration_start


    if not bw_powers:
        return


    times_all = np.array(power_times)


    # stack into a (N_time, N_freq) array; will error if any row has wrong length
    powers_array = np.stack(bw_powers, axis=0)

    # powers_db = 10 * np.log10(powers_array + 1e-12)

    # 2) Frequency axis around CONCENTRATION_FREQUENCY
    N_freq = powers_array.shape[1]
    half_bw = BW_FOR_BW_SUMMING / 2
    freqs = np.linspace(CONCENTRATION_FREQUENCY - half_bw,
                        CONCENTRATION_FREQUENCY + half_bw,
                        N_freq)

    # 2.5) Keep only last POWER_WINDOW spectra
    if powers_array.shape[0] > POWER_WINDOW:
        powers_array = powers_array[-POWER_WINDOW:]

    # 3) Sum new data into cumulative buffer
    if cumulative_power_sum is None:
        cumulative_power_sum = np.sum(powers_array, axis=0)
        num_summed = powers_array.shape[0]
    else:
        cumulative_power_sum += np.sum(powers_array, axis=0)
        num_summed += powers_array.shape[0]

    # 4) Convert cumulative sum to dB
    cumulative_db = 10 * np.log10(cumulative_power_sum + 1e-12)

    # 5) Plot current (latest) CUMULATIVE spectrum
    bw_curve.setData(freqs, cumulative_db)  # stacked version


    # 6) Plot stacked (cumulative summed) spectrum
    stacked_curve.setData(freqs, cumulative_db)

    current_time = time.time()

    if current_time - integration_start >= INTEGRATION_TIME:
        # Average the summed spectra over integration time
        avg_spectrum = cumulative_power_sum / num_summed

        # Convert to dB for saving
        avg_spectrum_db = 10 * np.log10(avg_spectrum + 1e-12)

        # Save to file as numpy binary for precision and easy reload
        timestamp = int(integration_start)
        filename = os.path.join(save_dir, f"spectrum_{timestamp}.npy")
        np.save(filename, avg_spectrum)
        print(f"Saved integrated spectrum to {filename}")

        # Reset cumulative sums and counters
        cumulative_power_sum = None
        num_summed = 0
        integration_start = current_time

    # 7)save bw summing to file
    spectrum = np.vstack((freqs, cumulative_db)).T  # shape: (N_freq, 2)
    with open(bw_file_path, 'wb') as bw_file:  # write mode (overwrite)
        spectrum.tofile(bw_file)

    # 8)
    with open(stacked_file_path, 'wb') as stacked_file:
        spectrum_stacked = np.vstack((freqs, cumulative_db)).T  # shape: (N_freq, 2)
        spectrum_stacked.tofile(stacked_file)

    # 9) Dynamic Y-axis scaling
    min_pow = np.min(cumulative_db)
    max_pow = np.max(cumulative_db)
    stacked_plot.setYRange(min_pow - 10, max_pow + 10)

    last_db = 10 * np.log10(powers_array[-1] + 1e-12)
    if max_power_sum < np.max(last_db):
        max_power_sum = np.max(last_db)
        check3 = True
    if min_power_sum > np.min(last_db):
        min_power_sum = np.min(last_db)
        check4 = True


    if check3 and check4:
        bw_plot.setYRange(min_power_sum - 10, max_power_sum + 30)
    else:
        bw_plot.setYRange(np.min(last_db) - 10, np.max(last_db) + 30)







    # print(len(bw_powers))
    # print(bw_powers)
    # print("BW powers list:", bw_powers[::int(len(bw_powers) / 10)])


    # 2) Convert time to relative time in seconds (or milliseconds if needed)
    # times_relative = times_all - times_all[0]  # Time in seconds since the first sample
    # For milliseconds: times_relative = (times_all - times_all[0]) * 1000

    # 3) Trim both to the last POWER_WINDOW samples
    # if len(powers_db) > POWER_WINDOW:
        # powers_db = powers_db[-POWER_WINDOW:] #keep only the last POWER_WINDOW number of elements
        # from the powers_db list (which holds your dB values of total power).
        # times_relative = times_relative[-POWER_WINDOW:]

    # 3) Average over time for exposure stacking
    # stacked_db = np.mean(powers_db, axis=0)  # Mean dB per frequency bin


    # 4) Plot the current spectrum (last timestamp)
    # bw_curve.setData(freqs, powers_db[-1])  # Last spectrum

    # 4) Plot raw data, with respect to time
    # bw_curve.setData(times_relative, powers_db)

    # 5) Save Bandwidth Summing Data to file
    # with open(bw_file_path, 'ab') as bw_file:
    #     np.array([times_relative, powers_db], dtype=np.float64).tofile(bw_file)

    # 5) Save Bandwidth Summing Data to file
    # with open(bw_file_path, 'ab') as bw_file:
    #     spectrum_data = np.vstack((freqs, powers_db[-1]))  # Shape: (2, N_freq)
    #     spectrum_data.T.tofile(bw_file)  # Save as (freq, power) pairs

    # 6) Plot the exposure-stacked spectrum
    # stacked_curve.setData(freqs, stacked_db)

    # 7) Save Exposure Stacking Data to file
    # with open(stacked_file_path, 'ab') as stacked_file:
    #     stacked_spectrum_data = np.vstack((freqs, stacked_db[-1])).T  # Shape: (N_freq, 2)
    #     stacked_spectrum_data.tofile(stacked_file)

    # 8) Dynamic Y-axis scaling
    # max_power = np.max(powers_db[-1])
    # min_power = np.min(powers_db[-1])
    # bw_plot.setYRange(min_power - 10, max_power + 10)
    #
    # if len(stacked_db) > 0:
    #     max_stacked = np.max(stacked_db)
    #     min_stacked = np.min(stacked_db)
    #     stacked_plot.setYRange(min_stacked - 10, max_stacked + 10)
    #
    #
    #
    # # 6) If you have at least as many points as your moving-average window:
    # window = 10
    # if len(powers_db) >= window:
    #     kernel = np.ones(window) / window
    #     stacked_db = np.convolve(powers_db, kernel, mode='valid')
    #     times_stacked = times_relative[window - 1:]  # Adjust times for stacked plot
    # else:
    #     stacked_db = np.array([])
    #     times_stacked = np.array([])
    #
    # # 7) Plot the exposure-stacked curve
    # stacked_curve.setData(times_stacked, stacked_db)
    #
    # # 8) Save Exposure Stacking Data to file
    # with open(stacked_file_path, 'ab') as stacked_file:
    #     np.array([times_stacked, stacked_db], dtype=np.float64).tofile(stacked_file)
    #
    # # Dynamic Y-axis scaling for Bandwidth-Summed Plot
    # max_power = np.max(powers_db)
    # min_power = np.min(powers_db)
    # bw_plot.setYRange(min_power - 10, max_power + 10)  # Add some padding for clarity
    #
    # # Dynamic Y-axis scaling for Stacked Power Plot
    # if len(stacked_db) > 0:
    #     max_stacked = np.max(stacked_db)
    #     min_stacked = np.min(stacked_db)
    #     stacked_plot.setYRange(min_stacked - 10, max_stacked + 10)  # Add some padding for clarity




# Global buffer and lock for thread-safe access
import threading
TARGET_FFT_SIZE = 8192*4  # or 4096 for faster updates
# this makes room for TARGET_FFT_SIZE buffers, not samples, in the queue....!!!!!! so i replaced it....
# shared_buffer = queue.Queue(maxsize=TARGET_FFT_SIZE)
# Keep only the 10 most recent FFT buffers
shared_buffer = {
    0: deque(maxlen=10),  # For RX channel 0
    1: deque(maxlen=10),  # For RX channel 1
}
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


        # if prev_fft_db is None:
        #     smoothed_fft_db = fft_db
        # else:
        #     smoothed_fft_db = alpha * fft_db + (1 - alpha) * prev_fft_db
        #
        # prev_fft_db = smoothed_fft_db
        # # curve.setData(freqs_mhz, smoothed_fft_db)
        # curve.setData(freqs_mhz[valid_indices], smoothed_fft_db[valid_indices])
        if prev_fft_db is None or len(prev_fft_db) != len(fft_db[valid_indices]):
            smoothed_fft_db = fft_db[valid_indices]
        else:
            smoothed_fft_db = alpha * fft_db[valid_indices] + (1 - alpha) * prev_fft_db

        prev_fft_db = smoothed_fft_db
        curve.setData(freqs_mhz[valid_indices], smoothed_fft_db)





def update_fft_gui():
    global shared_buffer, buffer_lock, fs, rx_freq, fft_curve_ch0, fft_curve_ch1

    # try:
    #     # Try to get data from the queue without blocking
    #     data = shared_buffer.get_nowait()  # Non-blocking, get data
    # except queue.Empty:
    #     pass  # If the queue is empty, just skip
    with buffer_lock:
        if len(shared_buffer[0]) == 0 or len(shared_buffer[1]) == 0:
            return

        chan0_buf = shared_buffer[0][-1]
        chan1_buf = shared_buffer[1][-1]

    if chan0_buf is not None:
        compute_and_plot_fft(chan0_buf, fft_curve_ch0, fs, rx_freq)
    if chan1_buf is not None:
        compute_and_plot_fft(chan1_buf, fft_curve_ch1, fs, rx_freq)



    # Optional: If you want to clear the queue after processing (to free memory), you can do this
    # while not shared_buffer.empty():
    #     shared_buffer.get()

    # TODO clear buffer after plot- i think its done properly here:


def get_iq_corrections(device, channel):
    dcoff_i = _bladerf.ffi.new("int16_t *")
    ret_i = _bladerf.libbladeRF.bladerf_get_correction(device.dev, channel, Correction.DCOFF_I.value, dcoff_i)

    dcoff_q = _bladerf.ffi.new("int16_t *")
    ret_q = _bladerf.libbladeRF.bladerf_get_correction(device.dev, channel, Correction.DCOFF_Q.value, dcoff_q)

    if ret_i == 0 and ret_q == 0:
        print(f'Real time (I, Q) == ({dcoff_i[0], dcoff_q[0]})')
    else:
        return None, None



# meta = _bladerf.ffi.new("struct bladerf_metadata *")
# _bladerf.libbladeRF.bladerf_init_metadata(meta)
# meta.flags = _bladerf.libbladeRF.BLADERF_META_FLAG_RX_NOW

# my_bladerf.enable_module(_bladerf.libbladeRF.BLADERF_MODULE_RX, True)



def receive(device, freq: int, rate: int, gain: int,
            tx_start=None, rx_done=None,
            rxfile: str = '', num_samples: int = 1024):


    global shared_buffer, buffer_lock  # Access the global buffer



    dcoff_i = _bladerf.ffi.new("int16_t *")
    ret_i = _bladerf.libbladeRF.bladerf_get_correction(device.dev[0], BLADERF_MODULE_RX, BLADERF_CORR_DC_I, dcoff_i)
    dcoff_q = _bladerf.ffi.new("int16_t *")
    ret_q = _bladerf.libbladeRF.bladerf_get_correction(device.dev[0], BLADERF_MODULE_RX, BLADERF_CORR_DC_Q, dcoff_q)

    if ret_i == 0 and ret_q == 0:
        print("DCOFF_I =", dcoff_i[0], "and DCOFF_Q =", dcoff_q[0])
    else:
        print("Failed to get DCOFF_I:", ret_i)




    if (device == None):
        print("RX: Invalid device handle.")
        return -1

    # if (channel == None):
    #     print("RX: Invalid channel.")
    #     return -1

    # Configure both RX0 and RX1 channels
    for ch_index in [0, 1]:
        ch = device.Channel(ch_index)
        ch.frequency = freq
        ch.sample_rate = rate
        ch.gain = gain

    # TODO: this must be done after the gain, freq settings (eg ch.gain = ....)
    my_bladerf.enable_module(_bladerf.CHANNEL_RX(0), True)
    my_bladerf.enable_module(_bladerf.CHANNEL_RX(1), True)


    #Actual gain I am getting - no matter what.
    #Doesn't matter if I set it to 1000000db
    #For manual gain control the range is from -15db to 60db
    #if i set it to 1000db via the my_config.ini file, it will getv clamped to 60db
    #and this print below shows exactly that:
    print("Actual gain I am applying - (no matter what i put in the .ini file):", ch.gain)

    # Setup synchronous stream
    # 8 bytes total per interleaved complex sample pair (CH0 + CH1).
    bytes_per_sample = 4*2 # 2 bytes I + 2 bytes Q == 4 bytes for 1 channel, so for 2, it will be 2_x_that
    num_samples_per_buffer = 8192 * 2  # total samples (I+Q pairs), shared across RX0 and RX1, NOT INTERLEAVED
    num_samples_interleaved = num_samples_per_buffer // 2
    buf = bytearray(num_samples_per_buffer * bytes_per_sample)
    num_samples_read = 0

    # for channel 1 AND channel 2 according to gpt
    device.sync_config(layout=_bladerf.ChannelLayout.RX_X2,
                       fmt=_bladerf.Format.SC16_Q11,
                       num_buffers=16, # 16 was for 1 buffer, for convenience i added 16*4 for 2 RX channels
                       buffer_size=num_samples_interleaved, # interleaved = per_buffer / 2
                       num_transfers=8, #it was 8 for 1 RX channel...
                       stream_timeout=5000) # before 2 RX channels, it was 3500....!

    ###################################################################################################

    # Assuming `device` is your bladeRF device handle

    ch0 = device.Channel(0)  # RX channel 0
    current_lo_freq = ch0.frequency  # This returns the LO frequency in Hz (int or float)

    print(f"Current RX LO frequency for channel 0: {current_lo_freq} Hz")

    # enabled = device.is_module_enabled(_bladerf.CHANNEL_RX(0))
    # print(f"RX module enabled on channel 0: {enabled}")

    #####################################################################################################


    full_chunk_size = len(buf) // bytes_per_sample  # or your nominal buffer size in samples
    num_full_chunks = num_samples_read // full_chunk_size

    # Use only full chunks
    usable_samples = full_chunk_size * num_full_chunks
    buffer_for_bw_sum = buf[:usable_samples * bytes_per_sample]

    # Process buffer_for_bw_sum in fixed-size chunks for summing

    # buffer_size: How many samples per buffer (not bytes!) !!!!!!!!!!!!!!!!!!!!!!!!
    # buffer_size is the actual number of samples each buffer contains — not just a limit, but the exact size you’ll get per transfer.




    # Enable module
    # print("RX: Start")
    # ch.enable = True #TODO: not sure if this should be commented out.... I think (gpt) since I do "device.enable_module(_bladerf.BLADERF_CHANNEL_RX(0), True)" and "device.enable_module(_bladerf.BLADERF_CHANNEL_RX(1), True)" i am good

    print("gain_modes:", ch.gain_modes)
    print("RX gain:", int(config.getfloat('bladerf2-rx', 'rx_gain')))
    print("CH0: manual gain range:", my_bladerf.get_gain_range(_bladerf.CHANNEL_RX(0)))  # ch 0 or 1
    print("CH1: manual gain range:", my_bladerf.get_gain_range(_bladerf.CHANNEL_RX(1)))  # ch 0 or 1

    # print(my_bladerfget_gain_range(_bladerf.CHANNEL_RX(0)))


    # Create receive buffer



    # TODO: Replace num_samples with 1024 or some shit - I did!!!

    # Create receive buffer for interleaved RX_X2 (I0, Q0, I1, Q1)
    # Create receive buffer for interleaved RX_X2 (I0, Q0, I1, Q1)



    #TODO: check if this is correct:
    num_samples_per_channel = num_samples_per_buffer / 2 #TODO: or //2

    # buf0 = _bladerf.ffi.new("int16_t[]", buffer_size * 2)
    # buf1 = _bladerf.ffi.new("int16_t[]", buffer_size * 2)

    # my_bladerf.sync_rx(buf0, buffer_size, meta0)  # For channel 0
    # my_bladerf.sync_rx(buf1, buffer_size, meta1)  # For channel 1



    # print("len of buffer:", len(buf))


    # Tell TX thread to begin
    if (tx_start != None):
        tx_start.set()

    # Save the samples
    with open(rxfile, 'ab') as outfile:
        while not (stop_event and stop_event.is_set()):
            if num_samples > 0 and num_samples_read == num_samples:
                break
            elif num_samples > 0:
                print(f"len buf =={len(buf)}") # this prints (bytes_per_sample * num_samples_per_buffer)
                num = min(len(buf) // bytes_per_sample,
                          num_samples - num_samples_read)
                if (num_samples - num_samples_read) < (len(buf) // bytes_per_sample):
                    print(f"Found it: num_samples - num_samples_read === {num_samples - num_samples_read}")
                else:
                    print(f"Other Case (usual), len(buf) // bytes_per_sample === {len(buf) // bytes_per_sample}")
            else:
                print("What? Num_samples < 0 ????? how can that happen?? Num_samples is: ", num_samples)
                num = len(buf) // bytes_per_sample

            # This receives {num} samples and stores them into the buf
            device.sync_rx(buf, num)

            # This keeps track of how many samples have been read so far.
            num_samples_read += num

            # Write to file main "all-inclusive" I-Q data file
            outfile.write(buf[:num * bytes_per_sample])
            # buf[:num * bytes_per_sample]: a slice of the buffer that includes only the valid portion
            # (i.e. just the bytes that were filled with new data).

            # converts the raw byte data in buf into 16-bit integers, representing the received signal samples (I/Q data).
            data = np.frombuffer(buf[:num * bytes_per_sample], dtype=np.int16)


            # deinterleave: [I0,Q0,I1,Q1,...]
            i0 = data[0::4]
            q0 = data[1::4]
            i1 = data[2::4]
            q1 = data[3::4]

            # complex arrays for each channel
            ch0 = i0 + 1j * q0
            ch1 = i1 + 1j * q1

            # Converts the interleaved real (I) and imaginary (Q) parts of the signal into a complex number array (I + jQ).
            iq = data[::2] + 1j * data[1::2]



            # my_bladerfset_correction(_bladerf.CHANNEL_RX(0), Correction.DCOFF_I, 72)

            # val = my_bladerfget_correction(_bladerf.CHANNEL_RX(0), Correction.DCOFF_I)
            # print("DCOFF_I is now set to:", val)

            # Print current correction values
            # print("DCOFF_I:", device.get_correction(_bladerf.CHANNEL_RX(0), Correction.DCOFF_I))
            # print("DCOFF_Q:", device.get_correction(_bladerf.CHANNEL_RX(0), Correction.DCOFF_Q))




            # ret = _bladerf.bladerf_set_correction(
            #     my_bladerfdev,  # raw device pointer
            #     _bladerf.CHANNEL_RX(0),  # RX channel 0
            #     _bladerf.CORR_DC_I,  # DC correction - I component
            #     0  # Value (range: -2048 to +2047)
            # )

            # print(ret)

            iq_test = iq / (num_samples_per_buffer * 2)

            test_var = iq_test[0:num]

            # print("Wxxx:", np.max(test_var)) # TODO: If this is close to 1, you are overloading the ADC, and should reduce the gain
            #TODO:  you will want to adjust your gain to try to get that value around 0.5 to 0.8.
            #TODO: If it is 0.999 that means your receiver is overloaded/saturated and the signal is going to be distorted (it will look smeared throughout the frequency domain).

            # print("RX: Received", num, "samples")
            # print(iq)

            # === Bandwidth Summing (raw full IQ) ===
            # print("LEN IQ:", len(iq)) #ALWAYS 100K right now

            # power = np.sum(np.abs(iq) ** 2) / len(iq)
            # print("power_current:", power)
            # bw_powers.append(power)

            # FFT-based BW Summing over BW_FOR_BW_SUMMING
            window = np.hanning(len(iq)) #TODO: perhaps i shouldn't window non-plot data !!!!!!!!!
            iq_windowed = iq * window
            norm_factor = np.sum(window) / len(window)

            fft_vals = np.fft.fftshift(np.fft.fft(iq_windowed))
            fft_vals /= (len(iq) * norm_factor)
            power_spectrum = np.abs(fft_vals) ** 2


            # Frequency axis in Hz
            freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1 / fs))
            freqs_abs = freqs + freq  # shift to absolute center freq

            # Find indices within ±BW_FOR_BW_SUMMING/2 around fc
            half_bw = BW_FOR_BW_SUMMING / 2


            valid_indices = np.where(
                (freqs_abs >= CONCENTRATION_FREQUENCY - half_bw) &
                (freqs_abs <= CONCENTRATION_FREQUENCY + half_bw)
            )[0]

            # Define how many points you want in the band-limited spectrum

            # N_BW_POINTS = int(np.round(BW_FOR_BW_SUMMING * fft_len / fs))


            # Center index around CONCENTRATION_FREQUENCY
            # center_idx = np.argmin(np.abs(freqs_abs - CONCENTRATION_FREQUENCY))
            # half_n = N_BW_POINTS // 2
            # start = max(center_idx - half_n, 0)
            # end = min(center_idx + half_n, len(power_spectrum))

            # Assume:
            # - power_spectrum: 1D array of power values (after FFT)
            # - freqs_abs: 1D array of absolute frequencies (same length as power_spectrum)
            # - CONCENTRATION_FREQUENCY = fc
            # - BW_FOR_BW_SUMMING = bw
            # - fs = sampling rate
            # - fft_len = len(iq)

            # Step 1: Determine how many FFT bins correspond to the desired BW
            fft_len = len(iq)
            bin_width = fs / fft_len
            N_BW_POINTS = int(round(BW_FOR_BW_SUMMING / bin_width))
            half_n = N_BW_POINTS // 2

            # Step 2: Find center index in freqs_abs corresponding to fc
            center_idx = np.argmin(np.abs(freqs_abs - CONCENTRATION_FREQUENCY))

            # Step 3: Calculate desired range
            start = center_idx - half_n
            end = center_idx + half_n

            # Step 4: Prepare zero-padded band_spectrum
            band_spectrum = np.zeros(N_BW_POINTS)

            # Step 5: Extract valid portion of power_spectrum and insert it into band_spectrum
            valid_start = max(start, 0)
            valid_end = min(end, len(power_spectrum))

            insert_start = valid_start - start  # Offset if start < 0
            insert_end = insert_start + (valid_end - valid_start)

            band_spectrum[insert_start:insert_end] = power_spectrum[valid_start:valid_end]

            # guard: only append if the length is exactly N_BW_POINTS
            if len(band_spectrum) == N_BW_POINTS:
                bw_powers.append(band_spectrum)
            else:
                # this should never happen; log to catch bugs
                print(f"‼️ unexpected band_spectrum length {len(band_spectrum)} != {N_BW_POINTS}")

            power_times.append(time.time())

            # Downsample for GUI only (e.g., take 1 out of every 500 samples)
            step = max(len(iq) // TARGET_FFT_SIZE, 1)

            ch0_downsampled = ch0[::step]
            ch1_downsampled = ch1[::step]

            # Update shared_buffer safely
            with buffer_lock:
                try:
                    # shared_buffer.put_nowait(iq_downsampled)
                    shared_buffer[0].append(ch0_downsampled)
                    shared_buffer[1].append(ch1_downsampled)

                except queue.Full:
                    pass  # Drop the new data instead of blocking




        ####################   DEBUGGING TO SEE IF IQ PARAMS GET MAINTAINED    ############################################################

                # Get DCOFF_I
                #TODO: uncomment this



                # dcoff_i = _bladerf.ffi.new("int16_t *")
                # ret_i = _bladerf.libbladeRF.bladerf_get_correction(device.dev, channel, Correction.DCOFF_I.value,
                #                                                    dcoff_i)
                # if ret_i == 0:
                #     print("Current DCOFF_I =", dcoff_i[0])
                # else:
                #     print("Failed to get DCOFF_I:", ret_i)
                #
                # # Get DCOFF_Q
                # dcoff_q = _bladerf.ffi.new("int16_t *")
                # ret_q = _bladerf.libbladeRF.bladerf_get_correction(device.dev, channel, Correction.DCOFF_Q.value,
                #                                                    dcoff_q)
                # if ret_q == 0:
                #     print("Current DCOFF_Q =", dcoff_q[0])
                # else:
                #     print("Failed to get DCOFF_Q:", ret_q)

    ########################################################################################################################


    # Disable module
    # print("RX: Stop")
    # ch.enable = False
    #
    #
    #
    # if (rx_done != None):
    #     rx_done.set()
    #
    # print("RX: Done")
    #
    # return 0




    #TODO: leave ch.enable = True for continuous streaming

    if rx_done != None:
        rx_done.set() # if some thread is waiting for per-call completion,
        #TODO: ensure that this flag is set after interrupting the code
    return 0



# =============================================================================
# Load Configuration
# =============================================================================





def rx_loop():
    global rx_freq
    while not stop_event.is_set():
        # rx_ch = _bladerf.CHANNEL_RX(config.getint('bladerf2-rx', 'rx_channel0'))
        rx_freq = int(config.getfloat('bladerf2-rx', 'rx_frequency'))
        rx_rate = int(config.getfloat('bladerf2-rx', 'rx_samplerate'))
        rx_gain = int(config.getfloat('bladerf2-rx', 'rx_gain'))
        rx_ns = int(config.getfloat('bladerf2-rx', 'rx_num_samples'))
        # rx_file = config.get('bladerf2-rx', 'rx_file')
        rx_file_path = f'./logs/all_inclusive_{current_datetime}.bin'

        # Assign the new path to the config
        config.set('bladerf2-rx', 'rx_file', rx_file_path)




        status = receive(
            device=my_bladerf,
            freq=rx_freq,
            rate=rx_rate,
            gain=rx_gain,
            tx_start=None,
            rx_done=None,
            rxfile=rx_file_path,
            num_samples=rx_ns
        )

        print("Actual shit2:", rx_gain)

        if status < 0:
            print(f"Receive operation failed with error {status}")
            break
        # time.sleep(0.05)  #TODO: delete this delay after
        # print(f"bw_powers: {bw_powers[-5:]}")  # Print last 5 entries
        # print(f"power_times: {power_times[-5:]}")
        print("RX thread exiting.")


        # print("gain_modes:", rx_ch.gain_modes)





if __name__ == "__main__":
    # TODO start the GUI



    app = QtWidgets.QApplication(sys.argv)

    # Set up the GUI as usual
    win = pg.GraphicsLayoutWidget(title="Live Signal and FFT Plot")
    win.show()

    # Setup the FFT plot
    fft_plot = win.addPlot(title="FFT of Signal")
    # fft_curve = fft_plot.plot(pen='g')

    # Create two curves: one for each channel, with different colors
    fft_curve_ch0 = fft_plot.plot(pen='r')  # Channel 0: red
    fft_curve_ch1 = fft_plot.plot(pen='b')  # Channel 1: blue

    # Label axes
    fft_plot.setLabel('bottom', 'Frequency (MHz)')
    fft_plot.setLabel('left', 'Magnitude (dB)')

    # FIX THE AXES — no auto-rescale
    center_freq_mhz = rx_freq / 1e6  # Convert to MHz
    bandwidth_mhz = BW / 1e6  # Convert to MHz
    fft_plot.setXRange(center_freq_mhz - bandwidth_mhz / 2,
                       center_freq_mhz + bandwidth_mhz / 2, padding=0)

    fft_plot.setYRange(-100, 10, padding=0)  # dB range — adjust based on your signal

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
    bw_timer.start(500)  # Every 0.05 sec or as needed

    # Set up timer to refresh GUI
    timer = QtCore.QTimer()
    timer.timeout.connect(update_fft_gui)  # <- Connect to your function
    timer.start(500)  # Refresh every 50ms

    my_bladerf.set_correction(rx_channel0, Correction.DCOFF_I, 652) #-880 for 652, -536 for 1000
    my_bladerf.set_correction(rx_channel0, Correction.DCOFF_Q, -496) #-496 for 502, -1000 for 1000

    my_bladerf.set_correction(rx_channel1, Correction.DCOFF_I, 652)  # -880 for 652, -536 for 1000
    my_bladerf.set_correction(rx_channel1, Correction.DCOFF_Q, -496)  # -496 for 502, -1000 for 1000


    time.sleep(1)


    def clean_shutdown():
        print("Shutting down...")
        stop_event.set()  # Signal all threads to stop
        # Disable BladeRF channels before closing
        my_bladerf.enable_module(_bladerf.CHANNEL_RX(0), False)
        my_bladerf.enable_module(_bladerf.CHANNEL_RX(1), False)
        my_bladerf.close()  # Properly close BladeRF device
        app.quit()  # Cleanly close the Qt event loop


    # Connect PyQt5 close event
    app.aboutToQuit.connect(clean_shutdown)
    signal.signal(signal.SIGINT, lambda sig, frame: clean_shutdown())
    signal.signal(signal.SIGTERM, lambda sig, frame: clean_shutdown())

    # Start RX thread
    rx_thread = threading.Thread(target=rx_loop, daemon=True)
    rx_thread.start()

    # time.sleep(2)

    ###############        OPTIMAL IQ PARAMETERS SETUP      ####################################################



    # print(f'I parameter=====: {Correction.DCOFF_I}')

    ###################################################################################################################

    # TODO start threads!!!!
    # ✔️ Only rx_thread and QTimer are needed now — fft_plot_worker is no longer used


    sys.exit(app.exec_())

"""
    The return value of status is given by calling the transmit def, which always returns
    an integer. Thus, simply referencing transmit as an argument of tx_pool.apply_async()
    "activates" the transmit def, which shall return an integer.

    !!!!!!!!!!!
"""