#!/usr/bin/env python3
###############################################################################
#
# Copyright (c) 2018-present Nuand LLC
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
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

# File path to monitor
file_path = "/Users/giannis/PycharmProjects/final_radiotelescope/my_stuff/my_rx_samples.bin"

# Create the application window
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Live IQ Plot")
plot = win.addPlot(title="Time Domain I Samples")
curve = plot.plot(pen='y')
win.show()

# Track where we last read from
last_position = 0

def update():
    global last_position
    try:
        with open(file_path, "rb") as f:
            f.seek(last_position)
            new_data = f.read()
            last_position += len(new_data)

        # Only update if we have new data
        if len(new_data) >= 4:
            # Convert to int16 (IQ interleaved)
            iq = np.frombuffer(new_data, dtype=np.int16)
            i = iq[::2]
            # You could also get Q: q = iq[1::2]
            curve.setData(i[-1024:])  # Show last 1024 samples
    except Exception as e:
        print("Error reading file:", e)

timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(100)




# Plotting thread
def live_plot_thread(shared_buffer, lock, stop_event):
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()
    line, = ax.plot([], [], lw=1)
    ax.set_ylim(-1, 1)
    ax.set_xlim(0, 4096)  # Initial window
    ax.set_title("Real-Time RX Signal (I Component)")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")

    while not stop_event.is_set():
        lock.acquire()
        if shared_buffer['data'] is not None:
            current_samples = shared_buffer['data']
            lock.release()

            line.set_ydata(current_samples.real)
            line.set_xdata(np.arange(len(current_samples)))
            ax.set_xlim(0, len(current_samples))
            ax.set_ylim(current_samples.real.min(), current_samples.real.max())
            fig.canvas.draw()
            fig.canvas.flush_events()
        else:
            lock.release()
        time.sleep(0.05)  # Small delay to avoid CPU burn

# Main RX logic
def rx_main():
    from bladerf import _bladerf
    import numpy as np

    # Open first available device
    dev = _bladerf.BladeRF()

    # Configure RX channel
    ch = dev.Channel(0)  # channel 0 (usually RX)
    ch.frequency = 2.45e9  # Example: 2.45 GHz
    ch.sample_rate = 2e6   # 2 MSPS
    ch.gain = 30           # Adjust gain as needed

    dev.sync_config(layout=_bladerf.ChannelLayout.RX_X1,
                    fmt=_bladerf.Format.SC16_Q11,
                    num_buffers=16,
                    buffer_size=4096,
                    num_transfers=8,
                    stream_timeout=3500)

    ch.enable = True  # Enable RX

    shared_buffer = {'data': None}
    lock = threading.Lock()
    stop_event = threading.Event()

    plot_thread = threading.Thread(target=live_plot_thread, args=(shared_buffer, lock, stop_event))
    plot_thread.start()

    try:
        while True:
            # Create buffer
            buf = bytearray(4096 * 4)  # 4 bytes per sample (16-bit I + 16-bit Q)

            # Read samples
            dev.sync_rx(buf, 4096)

            # Convert to numpy complex64 (Q11 format → float)
            iq = np.frombuffer(buf, dtype=np.int16).astype(np.float32).view(np.complex64)
            iq /= 2048  # Normalize SC16 Q11 format

            lock.acquire()
            shared_buffer['data'] = iq
            lock.release()

    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        stop_event.set()
        plot_thread.join()
        ch.enable = False
        dev.close()

# Run main logic



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
    print("RX: Start")
    ch.enable = True

    # Create receive buffer
    bytes_per_sample = 4
    buf = bytearray(1024 * bytes_per_sample)
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

            # Read into buffer
            device.sync_rx(buf, num)
            num_samples_read += num

            # Write to file
            outfile.write(buf[:num * bytes_per_sample])

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

config = ConfigParser()
config.read('/Users/giannis/PycharmProjects/final_radiotelescope/my_stuff/my_configuration.ini')

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

def run_rx():
    # Your BladeRF receiving logic here
    # Continuously write samples to the .bin file
    loops = 0
    while True:
        # update()
        loops += 1
        if (loops > 100):
            break

        print("got into RX loop")

        rx_ch = _bladerf.CHANNEL_RX(config.getint(s, 'rx_channel'))
        rx_freq = int(config.getfloat(s, 'rx_frequency'))
        rx_rate = int(config.getfloat(s, 'rx_samplerate'))
        rx_gain = int(config.getfloat(s, 'rx_gain'))
        rx_ns = int(config.getfloat(s, 'rx_num_samples'))
        rx_file = config.get(s, 'rx_file')

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
        if (status < 0):
            print("Receive operation failed with error " + str(status))

            """
                                The return value of status is given by calling the transmit def, which always returns
                                an integer. Thus, simply referencing transmit as an argument of tx_pool.apply_async()
                                "activates" the transmit def, which shall return an integer.

                                !!!!!!!!!!!
            """




for s in [ss for ss in config.sections() if board_name + '-' in ss]:

    if (s == board_name + "-load-fpga"):
        # Don't re-loading the FPGA!
        continue

    # Print the section name
    print("{:<35s} : ".format(s), end='')

    if (config.getboolean(s, 'enable')):

        print("RUNNING")

        if (s == board_name + '-tx'):

            tx_ch = _bladerf.CHANNEL_TX(config.getint(s, 'tx_channel'))
            tx_freq = int(config.getfloat(s, 'tx_frequency'))
            tx_rate = int(config.getfloat(s, 'tx_samplerate'))
            tx_gain = int(config.getfloat(s, 'tx_gain'))
            tx_rpt = int(config.getfloat(s, 'tx_repeats'))
            tx_file = config.get(s, 'tx_file')

            # Make this blocking for now ...
            status = tx_pool.apply_async(transmit,
                                         (),
                                         {'device': b,
                                          'channel': tx_ch,
                                          'freq': tx_freq,
                                          'rate': tx_rate,
                                          'gain': tx_gain,
                                          'tx_start': None,
                                          'rx_done': None,
                                          'txfile': tx_file,
                                          'repeat': tx_rpt
                                          }).get()
            if (status < 0):
                print("Transmit operation failed with error " + str(status))

                """
                            The return value of status is given by calling the transmit def, which always returns
                            an integer. Thus, simply referencing transmit as an argument of tx_pool.apply_async()
                            "activates" the transmit def, which shall return an integer.

                            !!!!!!!!!!!
                """

        elif (s == board_name + '-rx'):
            # rx_main()
            # Start RX in a separate thread
            rx_thread = threading.Thread(target=run_rx)
            rx_thread.daemon = True
            rx_thread.start()
            QtWidgets.QApplication.instance().exec()
            # app.exec()



        elif (s == board_name + '-txrx'):

            rx_channels = [x.strip() for x in config.get(s, 'rx_channel').split(',')]
            tx_channels = [x.strip() for x in config.get(s, 'tx_channel').split(',')]
            rx_freqs = [x.strip() for x in config.get(s, 'rx_frequency').split(',')]
            tx_freqs = [x.strip() for x in config.get(s, 'tx_frequency').split(',')]

            if (len(rx_channels) != len(tx_channels)):
                print("Configuration error in section " + s + ": "
                                                              "rx_channels and tx_channels must be the same length.")
                shutdown(error=-1, board=b)

            if (len(rx_freqs) != len(tx_freqs)):
                print("Configuration error in section " + s + ": "
                                                              "rx_frequency and tx_frequency must be the same length.")
                shutdown(error=-1, board=b)

            # Create Events for signaling between RX/TX threads
            rx_done = threading.Event()
            tx_start = threading.Event()

            for ch in range(0, len(rx_channels), 1):
                for freq in range(0, len(rx_freqs), 1):
                    rx_ch = _bladerf.CHANNEL_RX(int(rx_channels[ch]))
                    rx_freq = int(float(rx_freqs[freq]))
                    rx_rate = int(config.getfloat(s, 'rx_samplerate'))
                    rx_gain = int(config.getfloat(s, 'rx_gain'))
                    rx_ns = int(config.getfloat(s, 'rx_num_samples'))
                    rx_file = config.get(s, 'rx_file')
                    if (rx_file == "auto"):
                        rx_file = "rx_" + \
                                  "r" + rx_channels[ch] + \
                                  "t" + tx_channels[ch] + "_" + \
                                  str(int(float(rx_freqs[freq]) / (1e6))) + "M.bin"

                    tx_ch = _bladerf.CHANNEL_TX(int(tx_channels[ch]))
                    tx_freq = int(float(tx_freqs[freq]))
                    tx_rate = int(config.getfloat(s, 'tx_samplerate'))
                    tx_gain = int(config.getfloat(s, 'tx_gain'))
                    tx_rpt = int(config.getfloat(s, 'tx_repeats'))
                    tx_file = config.get(s, 'tx_file')

                    print("rx_ch = {:2d} ".format(int(rx_channels[ch])) +
                          "tx_ch = {:2d} ".format(int(tx_channels[ch])) +
                          "rx_freq = {:10d} ".format(rx_freq) +
                          "tx_freq = {:10d} ".format(tx_freq) +
                          "rx_file = " + rx_file)

                    # Start receiver thread
                    rx_result = rx_pool.apply_async(receive,
                                                    (),
                                                    {'device': b,
                                                     'channel': rx_ch,
                                                     'freq': rx_freq,
                                                     'rate': rx_rate,
                                                     'gain': rx_gain,
                                                     'tx_start': tx_start,
                                                     'rx_done': rx_done,
                                                     'rxfile': rx_file,
                                                     'num_samples': rx_ns
                                                     })

                    # Start transmitter thread
                    tx_result = tx_pool.apply_async(transmit,
                                                    (),
                                                    {'device': b,
                                                     'channel': tx_ch,
                                                     'freq': tx_freq,
                                                     'rate': tx_rate,
                                                     'gain': tx_gain,
                                                     'tx_start': tx_start,
                                                     'rx_done': rx_done,
                                                     'txfile': tx_file,
                                                     'repeat': tx_rpt
                                                     })

                    # Wait for RX thread to finish
                    rx_result.wait()
                    status = rx_result.get()

                    if (status < 0):
                        print("Receive operation failed with error " +
                              str(status))

                    # Wait for TX thread to finish
                    tx_result.wait()
                    status = tx_result.get()

                    if (status < 0):
                        print("Transmit operation failed with error " +
                              str(status))

                    tx_start.clear()
                    rx_done.clear()

    else:
        print("SKIPPED [ Disabled ]")

b.close()
print("Done!")