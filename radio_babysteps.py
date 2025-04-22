import numpy as np
import matplotlib.pyplot as plt
# from bladerf import _bladerf
from bladerf import _bladerf

from bladerf import _bladerf

from bladerf import BladeRF
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