
import sys
import os
import threading

from multiprocessing.pool import ThreadPool
from configparser         import ConfigParser

from bladerf              import _bladerf

# =============================================================================
# Close the device and exit
# =============================================================================
def shutdown( error = 0, board = None ):
    print( "Shutting down with error code: " + str(error) )
    if( board != None ):
        board.close()
    sys.exit(error)


# =============================================================================
# Version information
# =============================================================================
def print_versions( device = None ):
    print( "libbladeRF version: " + str(_bladerf.version()) )
    if( device != None ):
        try:
            print( "Firmware version: " + str(device.get_fw_version()) )
        except:
            print( "Firmware version: " + "ERROR" )
            raise

        try:
            print( "FPGA version: "     + str(device.get_fpga_version()) )
        except:
            print( "FPGA version: "     + "ERROR" )
            raise

    return 0


# =============================================================================
# Search for a bladeRF device attached to the host system
# Returns a bladeRF device handle.
# =============================================================================
def probe_bladerf():
    device = None
    print( "Searching for bladeRF devices..." )
    try:
        devinfos = _bladerf.get_device_list()
        if( len(devinfos) == 1 ):
            device = "{backend}:device={usb_bus}:{usb_addr}".format(**devinfos[0]._asdict())
            print( "Found bladeRF device: " + str(device) )
        if( len(devinfos) > 1 ):
            print( "Unsupported feature: more than one bladeRFs detected." )
            print( "\n".join([str(devinfo) for devinfo in devinfos]) )
            shutdown( error = -1, board = None )
    except _bladerf.BladeRFError:
        print( "No bladeRF devices found." )
        pass

    return device


# =============================================================================
# Load FPGA
# =============================================================================
def load_fpga( device, image ):

    image = os.path.abspath( image )

    if( not os.path.exists(image) ):
        print( "FPGA image does not exist: " + str(image) )
        return -1

    try:
        print( "Loading FPGA image: " + str(image ) )
        device.load_fpga( image )
        fpga_loaded  = device.is_fpga_configured()
        fpga_version = device.get_fpga_version()

        if( fpga_loaded ):
            print( "FPGA successfully loaded. Version: " + str(fpga_version) )

    except _bladerf.BladeRFError:
        print( "Error loading FPGA." )
        raise

    return 0


# =============================================================================
# TRANSMIT
# =============================================================================
def transmit( device, channel : int, freq : int, rate : int, gain : int,
              tx_start = None, rx_done = None,
              txfile : str = '', repeat : int = 1,  ):

    if( device == None ):
        print( "TX: Invalid device handle." )
        return -1

    if( channel == None ):
        print( "TX: Invalid channel." )
        return -1

    if( (rx_done == None) and (repeat < 1) ):
        print( "TX: Configuration settings indicate transmitting indefinitely?" )
        return -1

    if( tx_start != None ):
        print( "TX: waiting until receive thread is ready..." )
        if( not tx_start.wait(60.0) ):
            print( "TX: Timeout occurred while waiting for receiver to " +
                   "become ready." )
            return -1

    # Configure bladeRF
    ch             = device.Channel(channel)
    ch.frequency   = freq
    ch.sample_rate = rate
    ch.gain        = gain

    # Setup stream
    device.sync_config(layout=_bladerf.ChannelLayout.TX_X1,
                       fmt=_bladerf.Format.SC16_Q11,
                       num_buffers=16,
                       buffer_size=8192,
                       num_transfers=8,
                       stream_timeout=3500)

    # Enable module
    print( "TX: Start" )
    ch.enable = True

    # Create buffer
    bytes_per_sample = 4
    buf = bytearray(1024*bytes_per_sample)

    with open(txfile, 'rb') as infile:
        # Read samples from file into buf
        num = infile.readinto(buf)

        # Convert number of bytes read to samples
        num //= bytes_per_sample
        if( num > 0 ):
            repeats_remaining = repeat - 1
            repeat_inf        = (repeat < 1)
            while True:
                # Write to bladeRF
                device.sync_tx(buf, num)

                if( (rx_done != None) and rx_done.is_set() ):
                    break

                if( not repeat_inf ):
                    if( repeats_remaining > 0 ):
                        repeats_remaining -= 1
                    else:
                        break

    # Disable module
    print( "TX: Stop" )
    ch.enable = False

    return 0


# =============================================================================
# RECEIVE
# =============================================================================
def receive(device, channel : int, freq : int, rate : int, gain : int,
            tx_start = None, rx_done = None,
            rxfile : str = '', num_samples : int = 1024):

    status = 0

    if( device == None ):
        print( "RX: Invalid device handle." )
        return -1

    if( channel == None ):
        print( "RX: Invalid channel." )
        return -1

    # Configure BladeRF
    ch             = device.Channel(channel)
    ch.frequency   = freq
    ch.sample_rate = rate
    ch.gain        = gain

    # Setup synchronous stream
    device.sync_config(layout         = _bladerf.ChannelLayout.RX_X1,
                       fmt            = _bladerf.Format.SC16_Q11,
                       num_buffers    = 16,
                       buffer_size    = 8192,
                       num_transfers  = 8,
                       stream_timeout = 3500)

    # Enable module
    print( "RX: Start" )
    ch.enable = True

    # Create receive buffer
    bytes_per_sample = 4
    buf = bytearray(1024*bytes_per_sample)
    num_samples_read = 0

    # Tell TX thread to begin
    if( tx_start != None ):
        tx_start.set()

    # Save the samples
    with open(rxfile, 'wb') as outfile:
        while True:
            if num_samples > 0 and num_samples_read == num_samples:
                break
            elif num_samples > 0:
                num = min(len(buf)//bytes_per_sample,
                          num_samples-num_samples_read)
            else:
                num = len(buf)//bytes_per_sample

            # Read into buffer
            device.sync_rx(buf, num)
            num_samples_read += num

            # Write to file
            outfile.write(buf[:num*bytes_per_sample])

    # Disable module
    print( "RX: Stop" )
    ch.enable = False

    if( rx_done != None ):
        rx_done.set()

    print( "RX: Done" )

    return 0

# uut = probe_bladerf()
# print_versions(_bladerf.get_device_list())
config = ConfigParser()
if config.has_section("common"):
    print("sdajhvhjk")
