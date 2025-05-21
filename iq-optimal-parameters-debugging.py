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

base_dir = 'iq_logs'
iq_debug_filepath = f'iq_debug_filepath_{datetime}.txt'
iq_debug_filepath = os.path.join(base_dir, iq_debug_filepath)
current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Format: "YYYY-MM-DD_HH-MM-SS"
# bw_file_path = f"./logs/bw_summing_data_{current_datetime}.bin"
# stacked_file_path = f"./logs/exposure_stacking_data_{current_datetime}.bin"

# Bandwidth summing queue (optional) and stacking memory
bw_powers = []         # All power measurements
stacked_powers = []    # Sliding window or average of power


# Defines BW for bw summing
BW_FOR_BW_SUMMING = 2e6  # for example, 2 MHz

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




# Global buffer and lock for thread-safe access
import threading
TARGET_FFT_SIZE = 8192*4  # or 4096 for faster updates


shared_buffer = deque(maxlen=10)
buffer_lock = threading.Lock()



    # TODO clear buffer after plot- i think its done properly here:





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




    #Actual gain I am getting - no matter what.
    #Doesn't matter if I set it to 1000000db
    #For manual gain control the range is from -15db to 60db
    #if i set it to 1000db via the my_config.ini file, it will getv clamped to 60db
    #and this print below shows exactly that:
    print("Actual gain I am applying - (no matter what i put in the .ini file):", ch.gain)

    # Setup synchronous stream
    device.sync_config(layout=_bladerf.ChannelLayout.RX_X1,
                       fmt=_bladerf.Format.SC16_Q11,
                       num_buffers=16,
                       buffer_size=8192, #samples per buffer
                       num_transfers=8,
                       stream_timeout=3500)

    # buffer_size: How many samples per buffer (not bytes!) !!!!!!!!!!!!!!!!!!!!!!!!
    # buffer_size is the actual number of samples each buffer contains — not just a limit, but the exact size you’ll get per transfer.

    # Enable module
    # print("RX: Start")
    ch.enable = True

    print("gain_modes:", ch.gain_modes)
    print("RX gain:", int(config.getfloat('bladerf2-rx', 'rx_gain')))
    print("CH0: manual gain range:", b.get_gain_range(_bladerf.CHANNEL_RX(0)))  # ch 0 or 1
    print("CH1: manual gain range:", b.get_gain_range(_bladerf.CHANNEL_RX(1)))  # ch 0 or 1

    # print(b.get_gain_range(_bladerf.CHANNEL_RX(0)))


    # Create receive buffer
    bytes_per_sample = 4
    # TODO: Replace num_samples with 1024 or some shit - I did!!!

    num_samples_per_buffer = 8192

    buf = bytearray(num_samples_per_buffer * bytes_per_sample)
    num_samples_read = 0

    # print("len of buffer:", len(buf))


    # Tell TX thread to begin
    if (tx_start != None):
        tx_start.set()

    # Save the samples
    with open(rxfile, 'ab') as outfile:
        while True:
            if num_samples > 0 and num_samples_read == num_samples:
                break
            elif num_samples > 0:
                # print(len(buf))
                num = min(len(buf) // bytes_per_sample,
                          num_samples - num_samples_read)
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

            # Converts the interleaved real (I) and imaginary (Q) parts of the signal into a complex number array (I + jQ).
            iq = data[::2] + 1j * data[1::2]



            def calibrate_dc_offset(device, channel, fs, fc, num_samples):
                best_i = 0
                best_q = 0
                min_power = float('inf')
                ch = device.Channel(channel)

                bytes_per_sample = 4
                buf = bytearray(num_samples * bytes_per_sample)

                debugging_bw = 2e6  # Hz — defines ±debugging_bw/2 region around fc to monitor

                for i in range(-2048, 2049, 50):
                    for q in range(-2048, 2049, 50):
                        device.set_correction(channel, Correction.DCOFF_I, i)
                        device.set_correction(channel, Correction.DCOFF_Q, q)
                        time.sleep(0.05)  # Let hardware settle

                        # Capture fresh IQ samples
                        device.sync_rx(buf, num_samples)
                        raw = np.frombuffer(buf[:num_samples * bytes_per_sample], dtype=np.int16)
                        iq = raw[::2] + 1j * raw[1::2]

                        # Window + FFT
                        window = np.hanning(len(iq))
                        iq_windowed = iq * window
                        fft = np.fft.fftshift(np.fft.fft(iq_windowed))
                        power = np.abs(fft) ** 2

                        # Frequency axis in Hz (centered around 0 Hz)
                        freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1 / fs))
                        freqs_abs = freqs + fc  # shift to absolute frequency axis

                        # Select indices within ±debugging_bw/2
                        half_bw = debugging_bw / 2
                        mask = (freqs_abs >= fc - half_bw) & (freqs_abs <= fc + half_bw)

                        # Total power in the band around fc
                        total_band_power = np.sum(power[mask])

                        if total_band_power < min_power:
                            min_power = total_band_power
                            best_i = i
                            best_q = q
                            print(f'New min power = {min_power:.2e} for I = {best_i}, Q = {best_q}')

                # Apply best correction
                device.set_correction(channel, Correction.DCOFF_I, best_i)
                device.set_correction(channel, Correction.DCOFF_Q, best_q)
                print(f"✅ Best correction: I = {best_i}, Q = {best_q}, Power = {min_power:.2e}")

                # Ensure logs directory exists
                os.makedirs("iq_logs", exist_ok=True)

                # Generate log file path with date
                log_filename = datetime.now().strftime("iq_logs/iq_correction_log_%Y-%m-%d.txt")

                # Append to file
                with open(log_filename, 'a') as f:
                    f.write(
                        f"[{datetime.now().strftime('%H:%M:%S')}] I = {best_i}, Q = {best_q}, DC power = {min_power:.2e}\n")


            calibrate_dc_offset(device, channel, fs, rx_freq, num_samples)


####################   DEBUGGING TO SEE IF IQ PARAMS GET MAINTAINED                     ############################################################


            # Get DCOFF_I
            # dcoff_i = _bladerf.ffi.new("int16_t *")
            # ret_i = _bladerf.libbladeRF.bladerf_get_correction(device.dev, channel, Correction.DCOFF_I.value, dcoff_i)
            # if ret_i == 0:
            #     print("Current DCOFF_I =", dcoff_i[0])
            # else:
            #     print("Failed to get DCOFF_I:", ret_i)
            #
            # # Get DCOFF_Q
            # dcoff_q = _bladerf.ffi.new("int16_t *")
            # ret_q = _bladerf.libbladeRF.bladerf_get_correction(device.dev, channel, Correction.DCOFF_Q.value, dcoff_q)
            # if ret_q == 0:
            #     print("Current DCOFF_Q =", dcoff_q[0])
            # else:
            #     print("Failed to get DCOFF_Q:", ret_q)



########################################################################################################################




            iq_test = iq / (num_samples_per_buffer*2)

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

            # # FFT-based BW Summing over BW_FOR_BW_SUMMING
            # window = np.hanning(len(iq)) #TODO: perhaps i shouldnt window non-plot data !!!!!!!!!
            # iq_windowed = iq * window
            # norm_factor = np.sum(window) / len(window)
            #
            # fft_vals = np.fft.fftshift(np.fft.fft(iq_windowed))
            # fft_vals /= (len(iq) * norm_factor)
            # power_spectrum = np.abs(fft_vals) ** 2
            #
            # # Frequency axis in Hz
            # freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1 / fs))
            # freqs_abs = freqs + freq  # shift to absolute center freq
            #
            # # Find indices within ±BW_FOR_BW_SUMMING/2 around fc
            # half_bw = BW_FOR_BW_SUMMING / 2
            # valid_indices = np.where((freqs_abs >= freq - half_bw) & (freqs_abs <= freq + half_bw))[0]
            #
            # # Band power in that range
            # band_power = np.sum(power_spectrum[valid_indices])
            # bw_powers.append(band_power)
            #
            # power_times.append(time.time())
            #
            # # Downsample for GUI only (e.g., take 1 out of every 500 samples)
            # step = max(len(iq) // TARGET_FFT_SIZE, 1)
            # iq_downsampled = iq[::step]


            # Update shared_buffer safely
            with buffer_lock:
                try:
                    # shared_buffer.put_nowait(iq_downsampled)
                    shared_buffer.append(iq_downsampled)
                except queue.Full:
                    pass  # Drop the new data instead of blocking



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
        rx_file_path = f'./logs/all_inclusive_{current_datetime}.bin'

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

        print("Actual shit2:", rx_gain)

        if status < 0:
            print(f"Receive operation failed with error {status}")
            break
        time.sleep(0.1)




if __name__ == "__main__":
    # TODO start the GUI


    # Start RX thread
    rx_thread = threading.Thread(target=rx_loop, daemon=True)
    rx_thread.start()

    # while True:
    #     time.sleep(1)

    # Prevent the main thread from exiting
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting program via Ctrl+C")


