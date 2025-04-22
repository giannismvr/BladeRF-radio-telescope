#!/usr/bin/env python3
###############################################################################
###############################################################################
#
# Basic example of using bladeRF Python bindings for full duplex TX/RX.
# Review the companion my_configuration.ini to adjust configuration options.
#
###############################################################################

import sys
import os
import threading

from multiprocessing.pool import ThreadPool
from configparser import ConfigParser

from bladerf import _bladerf

import threading
import numpy as np
import matplotlib.pyplot as plt
import time

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import os
import struct

import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

# Path to your binary file
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets

import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
from configparser import ConfigParser

# Path to your binary file
file_path = "/Users/giannis/PycharmProjects/final_radiotelescope/my_stuff/my_rx_samples.bin"

# Load configuration
config = ConfigParser()
config.read('/Users/giannis/PycharmProjects/final_radiotelescope/my_stuff/my_configuration.ini')

# GUI setup
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Live Signal and FFT Plot")
win.show()

time_plot = win.addPlot(title="Time Domain I Samples")
time_curve = time_plot.plot(pen='y')

fft_plot = win.addPlot(title="FFT of Signal")
fft_curve = fft_plot.plot(pen='g')
fft_plot.setLabel('bottom', 'Frequency (MHz)')
fft_plot.setLabel('left', 'Magnitude (dB)')

# Track last file position
last_position = 0

# Get parameters from config
N_SAMPLES = int(float(config['bladerf2-rx']['rx_num_samples']))
BW = float(config['bladerf2-rx']['rx_bandwidth'])  # Hz
rx_freq = float(config['bladerf2-rx']['rx_frequency'])  # Hz
fs = float(config['bladerf2-rx']['rx_samplerate'])  # Hz

import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg

import sys
import numpy as np
from PyQt5 import QtCore, QtWidgets
import pyqtgraph as pg


class RealTimeFFTPlot(QtWidgets.QMainWindow):

    # TODO ignore this class
    # TODO create another thread that constantly checks solely the rx buffer for data and plots them
    # TODO main folder should be the rx forlder so that overwriting becomes easier
    # TODO make a separate folder (permanent data logger) in which you append the current rx_buffer-folder data
    # TODO refresh the plot every 1 sec for starters and clear the buffer every...

    def __init__(self, file_path):
        super().__init__()

        self.setWindowTitle("Real-time FFT Plot")
        self.resize(800, 600)

        # Create a PlotWidget and set it as the central widget
        self.plot_widget = pg.PlotWidget()
        self.setCentralWidget(self.plot_widget)

        # Initialize FFT plot
        self.fft_curve = self.plot_widget.plot(pen='y')

        # Time domain data (initially empty)
        self.sample_rate = 1000  # Sample rate (samples per second)
        self.num_samples = 1024  # Number of samples in each FFT window
        self.update_interval = 50  # Time in ms between updates
        self.time_data = np.zeros(self.num_samples)  # Initialize with zeros

        # File path to the received binary signal
        self.file_path = file_path
        self.file = open(self.file_path, 'rb')  # Open the binary file for reading

        # Set up a timer to update the plot every "update_interval" ms
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_fft_plot)
        self.timer.start(self.update_interval)

    def update_fft_plot(self):
        """Update the FFT plot with new data every time the timer fires."""

        # Read a chunk of the binary data from the file
        signal_chunk = np.fromfile(self.file, dtype=np.complex64, count=self.num_samples)

        # Check if the chunk is empty, and reset file pointer if needed
        if len(signal_chunk) == 0:
            self.file.seek(0)  # Reset to the beginning of the file
            signal_chunk = np.fromfile(self.file, dtype=np.complex64, count=self.num_samples)

        # Convert complex signal into real and imaginary parts
        if signal_chunk.size == self.num_samples:
            signal = np.real(signal_chunk)  # Use real part of the signal (or both parts for complex signals)

            # Shift data left (for real-time streaming simulation)
            self.time_data = np.roll(self.time_data, -len(signal))
            self.time_data[-len(signal):] = signal  # Correct assignment with matching size

            # Compute the FFT of the time-domain signal
            fft_data = np.fft.fft(self.time_data)
            fft_freqs = np.fft.fftfreq(len(self.time_data), 1 / self.sample_rate)
            fft_magnitude = np.abs(fft_data)

            # Update the FFT plot (display the positive frequencies)
            self.fft_curve.setData(fft_freqs[:len(fft_freqs) // 2], fft_magnitude[:len(fft_magnitude) // 2])

    def closeEvent(self, event):
        """Ensure the file is closed properly when the application is closed."""
        self.file.close()
        event.accept()


# # Timer for real-time updates
# timer = QtCore.QTimer()
# timer.timeout.connect(update)
# timer.start(100)


# Start the Qt event loop
# app.exec_()


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
def transmit(device, channel: int, freq: int, rate: int, gain: int,
             tx_start=None, rx_done=None,
             txfile: str = '', repeat: int = 1, ):
    if (device == None):
        print("TX: Invalid device handle.")
        return -1

    if (channel == None):
        print("TX: Invalid channel.")
        return -1

    if ((rx_done == None) and (repeat < 1)):
        print("TX: Configuration settings indicate transmitting indefinitely?")
        return -1

    if (tx_start != None):
        print("TX: waiting until receive thread is ready...")
        if (not tx_start.wait(60.0)):
            print("TX: Timeout occurred while waiting for receiver to " +
                  "become ready.")
            return -1

    # Configure bladeRF
    ch = device.Channel(channel)

    ch.frequency = freq
    ch.sample_rate = rate
    ch.gain = gain

    # Setup stream
    device.sync_config(layout=_bladerf.ChannelLayout.TX_X1,
                       fmt=_bladerf.Format.SC16_Q11,
                       num_buffers=16,
                       buffer_size=8192,
                       num_transfers=8,
                       stream_timeout=3500)

    # Enable module
    print("TX: Start")
    ch.enable = True

    # Create buffer
    bytes_per_sample = 4
    buf = bytearray(1024 * bytes_per_sample)

    with open(txfile, 'rb') as infile:
        # Read samples from file into buf
        num = infile.readinto(buf)

        # Convert number of bytes read to samples
        num //= bytes_per_sample
        if (num > 0):
            repeats_remaining = repeat - 1
            repeat_inf = (repeat < 1)
            while True:
                # Write to bladeRF

                device.sync_tx(buf, num)

                if ((rx_done != None) and rx_done.is_set()):
                    break

                if (not repeat_inf):
                    if (repeats_remaining > 0):
                        repeats_remaining -= 1
                    else:
                        break

    # Disable module
    print("TX: Stop")
    ch.enable = False

    return 0


# =============================================================================
# RECEIVE
# =============================================================================
def receive(device, channel: int, freq: int, rate: int, gain: int,
            tx_start=None, rx_done=None,
            rxfile: str = '', num_samples: int = 1024):
    status = 0

    if (device == None):
        print("RX: Invalid device handle.")
        return -1

    if (channel == None):
        print("RX: Invalid channel.")
        return -1

    # Configure BladeRF
    ch = device.Channel(channel)
    print("got here !!!!!!!!!!!!!!!!!!!!!!!!!!!")
    ch.frequency = freq
    print("but not here******************")
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
    print("RX: Start")
    ch.enable = True

    # Create receive buffer
    bytes_per_sample = 4
    buf = bytearray(1024 * bytes_per_sample)
    num_samples_read = 0

    # ---- 👇 Add FFT buffer here ----
    fft_buffer_size = 4096  # samples to store for plotting
    fft_buffer = np.zeros(fft_buffer_size, dtype=np.complex64)

    # Timer for FFT update
    last_fft_plot = time.time()
    plot_interval = 0.5  # seconds

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

            # Read into buffer
            device.sync_rx(buf, num)
            num_samples_read += num

            # Write to file
            outfile.write(buf[:num * bytes_per_sample])

            """"

            arxizouyn ta organa edw!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

            """

            # ---- 👇 Convert buffer to complex samples ----
            data = np.frombuffer(buf[:num * bytes_per_sample], dtype=np.int16)
            iq = data[::2] + 1j * data[1::2]

            # ---- 👇 Update FFT buffer as ring buffer ----
            fft_buffer = np.roll(fft_buffer, -len(iq))
            fft_buffer[-len(iq):] = iq

            # ---- 👇 Plot FFT every plot_interval seconds ----
            if time.time() - last_fft_plot > plot_interval:
                last_fft_plot = time.time()

                # Compute FFT
                fft_vals = np.fft.fftshift(np.fft.fft(fft_buffer))
                fft_db = 20 * np.log10(np.abs(fft_vals) + 1e-6)

                # Frequency axis
                freqs = np.fft.fftshift(np.fft.fftfreq(len(fft_buffer), 1 / rate))

                # Plot
                plt.clf()
                plt.plot(freqs / 1e6, fft_db)
                plt.xlabel('Frequency (MHz)')
                plt.ylabel('Magnitude (dB)')
                plt.title('Real-Time FFT')
                plt.pause(0.001)

    # Disable module
    print("RX: Stop")
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

loops = 0

import threading


# def run_rx():
#     # Your BladeRF receiving logic here
    # Continuously write samples to the .bin file


loops = 0
for s in [ss for ss in config.sections() if board_name + '-' in ss]:

    if (s == board_name + "-load-fpga"):
        # Don't re-loading the FPGA!
        continue

    # Print the section name
    print("{:<35s} : ".format(s), end='')

    if (config.getboolean(s, 'enable')):

        print("RUNNING")

        if (s == board_name + '-rx'):
            # rx_main()
            # Start RX in a separate thread


            # rx_thread = threading.Thread(target=run_rx)
            # rx_thread.daemon = True
            # rx_thread.start()
            print("\nhegfdhjfwaegfawe")
            while True:
                # time.sleep(6)
                loops += 1
                if (loops > 100):
                    break

                print("got into RX loop")

                rx_ch = _bladerf.CHANNEL_RX(config.getint(s, 'rx_channel'))
                # rx_ch = 0
                rx_freq = int(config.getfloat(s, 'rx_frequency'))
                rx_rate = int(config.getfloat(s, 'rx_samplerate'))
                rx_gain = int(config.getfloat(s, 'rx_gain'))
                rx_ns = int(config.getfloat(s, 'rx_num_samples'))
                rx_file = config.get(s, 'rx_file')

                fft_size = N_SAMPLES

                print(rx_freq)

                # Make this blocking for now ...
                status = rx_pool.apply_async(receive,
                                             (),
                                             {'device': b,
                                              'channel': rx_ch,
                                              'freq': rx_freq,
                                              'rate': rx_rate,
                                              'gain': rx_gain,
                                              'tx_start': None,
                                              'rx_done': None,
                                              'rxfile': rx_file,
                                              'num_samples': rx_ns
                                              }).get()

                # TODO append to rx data buffer
                # TODO check if rx buffer is full
                # TODO check if main file is full

                if (status < 0):
                    print("Receive operation failed with error " + str(status))

                    """
                                        The return value of status is given by calling the transmit def, which always returns
                                        an integer. Thus, simply referencing transmit as an argument of tx_pool.apply_async()
                                        "activates" the transmit def, which shall return an integer.

                                        !!!!!!!!!!!
                    """






    else:
        print("SKIPPED [ Disabled ]")

b.close()
print("Done!")