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



#Add these manually if they're not exposed by your bindings
BLADERF_MODULE_RX = 0
BLADERF_MODULE_TX = 1

BLADERF_CORR_DC_I = 0
BLADERF_CORR_DC_Q = 1
BLADERF_CORR_PHASE = 2
BLADERF_CORR_GAIN = 3

CONCENTRATION_FREQUENCY = 2.4e9



current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # Format: "YYYY-MM-DD_HH-MM-SS"
bw_file_path = f"./logs/bw_summing_data_{current_datetime}.bin"
stacked_file_path = f"./logs/exposure_stacking_data_{current_datetime}.bin"

# Bandwidth summing queue (optional) and stacking memory
stacked_powers = []    # Sliding window or average of power


# Defines BW for bw summing
BW_FOR_BW_SUMMING = 10e6  # for example, 2 MHz

POWER_WINDOW = 100     # Number of points to retain in stacked plot

# Load configuration
config = ConfigParser()
config.read('my_stuff/my_configuration.ini')



prev_fft_db = None
alpha = 1  # Smoothing factor, adjust as needed
# The smoothing factor controls how much the new FFT data affects the plot.
# 	•	If alpha = 1.0: you show only the new FFT (no smoothing).
# 	•	If alpha = 0.0: you show only the old plot (fully frozen).
# 	•	If alpha = 0.2: you blend 20% of the new FFT with 80% of the previous plot.
#
# It reduces flickering by slowing down how fast the display reacts to small changes.


#---------------- more bloat parameters ----------------------------------------


exposure_stack_sum = None
exposure_stack_count = 0


# Initialize buffers somewhere globally or in class:
bw_powers = deque(maxlen=1000) #TODO: check whether 100 is an overdo here....
power_times = deque(maxlen=1000)
INTEGRATION_TIME = 10.0  # seconds per integration block
integration_start = time.time()
save_dir = "bw_spectra"  # directory to save spectra files
os.makedirs(save_dir, exist_ok=True)





#------------------------- MAIN CLASS --------------------------------------------------------------------

#TODO: replace .ini file with a single file containing everything
#TODO: assess whether config.read/parser is necessary and if not, delete it

class BladeRFController:
    """
    Encapsulates BladeRF initialization, FPGA loading, version reporting, and threading setup.
    """

    def __init__(self, config_path="/all_parameters.ini"):
        self.config_path = config_path
        self.config = ConfigParser()
        self.config.read(self.config_path)
        self.devices = []
        self.boards = []
        self.tx_pool = None
        self.rx_pool = None
        self.stop_event = threading.Event()
        self.rx_num_samples = None
        self.BW = None
        self.fs = None
        self.rx_freq=None
        self.rx_channel0 = None
        self.rx_channel1 = None

        #TODO: perhaps it would be better to initialize everything in the cosntructor, as some attributes may not be visible in the code, if the respective objects are declared in the "dict_init" def.... !!!!
        self.write_queue = {device: queue.Queue(maxsize=50) for device in self.devices}

        # Initialize a log filename per device
        self.log_file = {}  # dictionary to hold file paths per device



        # Call main initialization
        self._initialize_device()
        self._initialize_dictionaries()

        rxfiles = [
            "rx_device0.bin",
            "rx_device1.bin",
            "rx_device2.bin"
        ] #TODO where do i move this shit?

        # self.rx_pool = [ThreadPool(processes=1) for _ in self.boards]

        # TODO: check whether the thread pool is necessary, perhaps unusable.
        # TODO: perhaps we need to do pool.close or pool.join -> investigate!!!

    # ──────────────── Main initializer ────────────────
    def _initialize_device(self):


        """Runs the full initialization sequence."""
        self._set_verbosity()
        self.devices = self._probe_bladerf()
        if not self.devices:
            print("No BladeRF detected. Exiting.")
            self.shutdown(-1)

        # Create BladeRF objects for each detected device
        self.boards = []
        for device_str in self.devices:
            try:
                board = _bladerf.BladeRF(device_str)
                self.boards.append(board)
                print(f"Initialized board: {board.board_name}")
            except _bladerf.BladeRFError as e:
                print(f"Error initializing device {device_str}: {e}")

        # Optionally, still load FPGA for each board
        for board in self.boards:
            self._load_fpga_if_enabled(board)
            self._print_board_info(board)

        # self.board = _bladerf.BladeRF(self.device)

        # Thread pools (for RX/TX)
        # Only RX tasks

        for device in self.devices:
            self.log_file[device] = f"./logs/{device}.bin"
            # Optional: create/truncate the file at initialization
            with open(self.log_file[device], 'wb') as f:
                pass  # just create an empty file





    def _initialize_dictionaries(self, devices):
        global buffer_lock, shared_buffer

        # TODO: where should i move all these initializations?
        # self.tx_pool = ThreadPool(processes=1)
        # Initialize per-device locks and buffers
        buffer_lock = {}  # Will hold a Lock per device
        shared_buffer = {}  # Will hold a deque per channel per device

        for device in devices:  # devices is a list of device identifiers
            buffer_lock[device] = threading.Lock()
            shared_buffer[device] = {
                0: deque(maxlen=10),  # RX channel 0
                1: deque(maxlen=10),  # RX channel 1
            }

        data_of_device = {}

        # Bounded queue to hold data to write
        #TODO: find out what the maxsize should be for this!!!!!!!!!! ---------------------------------------------- SOS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


        # Global shutdown flag
        stop_event = threading.Event()



    # ──────────────── Verbosity ────────────────
    def _set_verbosity(self):
        """Sets libbladeRF verbosity based on configuration."""
        verbosity = self.config.get('common', 'libbladerf_verbosity').upper()
        levels = {
            "VERBOSE": 0, "DEBUG": 1, "INFO": 2, "WARNING": 3,
            "ERROR": 4, "CRITICAL": 5, "SILENT": 6
        }
        _bladerf.set_verbosity(levels.get(verbosity, 2)) #sets verbosity level to "INFO"



    # ──────────────── Probe ────────────────
    def _probe_bladerf(self):

        """Return a list of available BladeRF devices."""
        print("Searching for BladeRF devices...")
        try:
            devinfos = _bladerf.get_device_list()
            if not devinfos:
                print("No BladeRF devices found.")
                return []

            devices = []
            for dev in devinfos:
                device_str = f"{dev.backend}:device={dev.usb_bus}:{dev.usb_addr}"
                devices.append(device_str)
                print(f"Found BladeRF device: {device_str}")
            return devices

        except _bladerf.BladeRFError:
            print("Error enumerating BladeRF devices.")
            return []



    # ──────────────── Board info ────────────────
    def _print_board_info(self, board):
        try:
            print("Board name:", board.board_name)
            print("Firmware version:", board.get_fw_version())
            print("FPGA version:", board.get_fpga_version())
        except Exception as e:
            print(f"Error reading board info for {board}: {e}")



    # ──────────────── FPGA load ────────────────
    def _load_fpga_if_enabled(self, board):
        """Load FPGA image if the configuration enables it."""
        board_name = board.board_name
        enabled = self.config.getboolean(f"{board_name}-load-fpga", 'enable')
        if not enabled:
            print(f"Skipping FPGA load per config, for board: {board_name} ")
            return

        fpga_size = board.fpga_size
        image_path = self.config.get(f"{board_name}-load-fpga", f"image_{fpga_size}kle")
        image_path = os.path.abspath(image_path)
        if not os.path.exists(image_path):
            print("FPGA image does not exist:", image_path)
            self.shutdown(-1)

        try:
            print("Loading FPGA:", image_path)
            board.load_fpga(image_path)
            if board.is_fpga_configured():
                print("FPGA loaded successfully. Version:", board.get_fpga_version())
            else:
                print("FPGA load failed.")
                self.shutdown(-1)
        except _bladerf.BladeRFError as e:
            print("Error loading FPGA:", e)
            self.shutdown(-1)

    # ──────────────── Shutdown ────────────────
    def shutdown(self, error=0):
        """Close all BladeRF boards and exit."""
        print(f"Shutting down with error code {error}")
        for board in self.boards:
            try:
                board.close()
            except Exception:
                pass

        for pool in self.rx_pool:
            pool.close()
            pool.join()

        sys.exit(error)

    def _cofigure_rx_parameters(
            self,
            devices,  # required
            rxfile: str,  # required
            freq: int,
            rate: int,
            gain: int,
            num_samples: int,
            tx_start=None,
            rx_done=None
    ):

        global shared_buffer, buffer_lock  # Access the global buffer

        for idx, device_current in enumerate(devices):
            dcoff_i = _bladerf.ffi.new("int16_t *")
            ret_i = _bladerf.libbladeRF.bladerf_get_correction(device_current.dev[0], BLADERF_MODULE_RX, BLADERF_CORR_DC_I,
                                                               dcoff_i)

            dcoff_q = _bladerf.ffi.new("int16_t *")
            ret_q = _bladerf.libbladeRF.bladerf_get_correction(device_current.dev[0], BLADERF_MODULE_RX, BLADERF_CORR_DC_Q,
                                                               dcoff_q)

            if ret_i == 0 and ret_q == 0:
                print(f"[Device {idx}] DCOFF_I = {dcoff_i[0]}, DCOFF_Q = {dcoff_q[0]}")
            else:
                print(f"[Device {idx}] Failed to get correction values: ret_i={ret_i}, ret_q={ret_q}")

        for board_idx, board in enumerate(self.boards):
            for ch_index in [0, 1]:  # RX channels
                ch = board.Channel(ch_index)  # create channel object
                ch.frequency = freq
                ch.sample_rate = rate
                ch.gain = gain
            print(f"[{board.board_name}] RX channels configured: freq={freq}, rate={rate}, gain={gain}")

        # TODO: this must be done after the gain, freq settings (eg ch.gain = ....)
        for board in self.boards:
            # Enable RX channels 0 and 1
            for ch_index in [0, 1]:
                board.enable_module(_bladerf.CHANNEL_RX(ch_index), True)
            print(f"[{board.board_name}] RX channels enabled")

            # Actual gain I am getting - no matter what.
            # Doesn't matter if I set it to 1000000db
            # For manual gain control the range is from -15db to 60db
            # if i set it to 1000db via the my_config.ini file, it will get clamped to 60db
            # and this print below shows exactly that:

        for board in self.boards:
            for ch_index in [0, 1]:
                ch = board.Channel(ch_index)
                print(f"[{board.board_name}][RX{ch_index}] Actual gain applied:", ch.gain)

        # Setup synchronous stream
        # 8 bytes total per interleaved complex sample pair (CH0 + CH1).
        bytes_per_sample = 4 * 2  # 2 bytes I + 2 bytes Q == 4 bytes for 1 channel, so for 2 channels, it will be 2*that
        num_samples_per_buffer = 8192 * 2  # total samples (I+Q pairs), shared across RX0 and RX1, NOT INTERLEAVED
        num_samples_interleaved = num_samples_per_buffer // 2
        buffer = {}  # dictionary to hold buffers per device
        for i, device in enumerate(devices):
            buffer[i] = bytearray(num_samples_per_buffer * bytes_per_sample)
        num_samples_read = [0 for _ in devices]



        # RX_X1 → Single RX channel active. Buffer contains only CH0 samples: [I0, Q0, I0, Q0, ...].
        #
        # RX_X2 → Both RX channels active. Buffer contains interleaved CH0 + CH1 samples: [I0, Q0, I1, Q1, I0, Q0, I1, Q1, ...].

        # for channel 1 AND channel 2 according to gpt
        # RX_X2 literally enables both RX channels, not just RX2.
        for board in self.boards:
            board.sync_config(
                layout=_bladerf.ChannelLayout.RX_X2,
                fmt=_bladerf.Format.SC16_Q11,
                num_buffers=16,  # keep or adjust per channel count
                buffer_size=num_samples_interleaved,  # interleaved buffer size
                num_transfers=8,  # keep or adjust per channel count
                stream_timeout=5000  # adjust if needed
            )
            print(f"Sync configured for board {board.board_name}")

        for board in self.boards:
            for ch_index in [0, 1]:  # adjust if more RX channels per board
                ch = board.Channel(ch_index)
                current_lo_freq = ch.frequency  # returns LO frequency in Hz
                print(f"Board {board.board_name} - Channel {ch_index} LO frequency: {current_lo_freq} Hz")

        # enabled = device.is_module_enabled(_bladerf.CHANNEL_RX(0))
        # print(f"RX module enabled on channel 0: {enabled}")

        # enabled = device.is_module_enabled(_bladerf.CHANNEL_RX(0))
        # print(f"RX module enabled on channel 0: {enabled}")

        #####################################################################################################

        # full_chunk_size = len(buf) // bytes_per_sample  # or your nominal buffer size in samples
        # num_full_chunks = num_samples_read // full_chunk_size
        #
        # # Use only full chunks
        # usable_samples = full_chunk_size * num_full_chunks
        # buffer_for_bw_sum = buf[:usable_samples * bytes_per_sample]

        # Process buffer_for_bw_sum in fixed-size chunks for summing

        # buffer_size: How many samples per buffer (not bytes!) !!!!!!!!!!!!!!!!!!!!!!!!
        # buffer_size is the actual number of samples each buffer contains — not just a limit, but the exact size you’ll get per transfer.

        # Enable module
        # print("RX: Start")
        # ch.enable = True #TODO: not sure if this should be commented out.... I think (gpt) since I do "device.enable_module(_bladerf.BLADERF_CHANNEL_RX(0), True)" and "device.enable_module(_bladerf.BLADERF_CHANNEL_RX(1), True)" i am good

        # print(my_bladerfget_gain_range(_bladerf.CHANNEL_RX(0)))

        # Create receive buffer

        # TODO: Replace num_samples with 1024 or some shit - I did!!!

        # Create receive buffer for interleaved RX_X2 (I0, Q0, I1, Q1)
        # Create receive buffer for interleaved RX_X2 (I0, Q0, I1, Q1)

        # TODO: check if this is correct:
        num_samples_per_channel = num_samples_per_buffer / 2  # TODO: or //2

        # buf0 = _bladerf.ffi.new("int16_t[]", buffer_size * 2)
        # buf1 = _bladerf.ffi.new("int16_t[]", buffer_size * 2)

        # my_bladerf.sync_rx(buf0, buffer_size, meta0)  # For channel 0
        # my_bladerf.sync_rx(buf1, buffer_size, meta1)  # For channel 1

        # print("len of buffer:", len(buf))

        # Tell TX thread to begin
        # if (tx_start != None):
        #     tx_start.set() #TODO: what does this do?




# -------------------- erase this (up limit) --------------------------------------------------------


        # rxfiles = [f"{rxfile}_dev{i}.bin" for i in range(len(devices))]
        # while not (stop_event and stop_event.is_set()):
        #     for i, device in enumerate(devices):
        #         # Determine how many samples to read in this iteration
        #         if num_samples > 0 and num_samples_read[i] == num_samples:
        #             break  # already reached the target for this device
        #
        #         elif num_samples > 0:
        #             print(f"len buf =={len(buffer[i])}")  # this prints (bytes_per_sample * num_samples_per_buffer)
        #             num = min(len(buffer[i]) // bytes_per_sample, num_samples - num_samples_read[i])
        #             if (num_samples - num_samples_read[i]) < (len(buffer[i]) // bytes_per_sample):
        #                 print(f"Found it: num_samples - num_samples_read === {num_samples - num_samples_read[i]}")
        #             else:
        #                 print(f"Other Case (usual), len(buf) // bytes_per_sample === {len(buffer[i]) // bytes_per_sample}")
        #
        #         else:
        #             print("What? Num_samples < 0 ????? how can that happen?? Num_samples is: ", num_samples)
        #             num = len(buffer[i]) // bytes_per_sample
        #
        #         # This receives {num} samples and stores them into the buffer
        #
        #
        #         device.sync_rx(buffer[i], num)


# # ----------------------------------------- erase this (down limit) --------------------------------------------------------




    def rx_worker(self, device, buffer, num_samples, stop_event, rxfile):
        """
        Worker function for receiving data from a single BladeRF device.
        """

        global data_of_device

        bytes_per_sample = 4 * 2  # 2 bytes I + 2 bytes Q == 4 bytes for 1 channel, so for 2 channels, it will be 2*that
        num_samples_per_buffer = 8192 * 2  # total samples (I+Q pairs), shared across RX0 and RX1, NOT INTERLEAVED
        num_samples_interleaved = num_samples_per_buffer // 2
        buf = bytearray(num_samples_per_buffer * bytes_per_sample)
        num_samples_read = 0

        while not stop_event.is_set():
            if num_samples > 0 and num_samples_read == num_samples:
                break
            elif num_samples > 0:
                print(f"len buf =={len(buf)}")  # this prints (bytes_per_sample * num_samples_per_buffer)
                num = min(len(buf) // bytes_per_sample,
                          num_samples - num_samples_read)
                if (num_samples - num_samples_read) < (len(buf) // bytes_per_sample):
                    print(f"Found it: num_samples - num_samples_read === {num_samples - num_samples_read}")
                else:
                    print(f"Other Case (usual), len(buf) // bytes_per_sample === {len(buf) // bytes_per_sample}")
            else:
                print("What? Num_samples < 0 ????? how can that happen?? Num_samples is: ", num_samples)
                num = len(buf) // bytes_per_sample

            # Receive samples into the buffer
            device.sync_rx(buffer, num)

            # This keeps track of how many samples have been read so far.
            num_samples_read += num

            # Make an immutable copy of buffer for safe passing
            raw_copy = bytes(buf[:num_samples_per_buffer * bytes_per_sample])
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')

            # Push to writer queue (blocks if queue is full)
            self.write_queue.put((timestamp, raw_copy))



            # Write to file main "all-inclusive" I-Q data file
            # outfile.write(buffer[:num * bytes_per_sample])

            # buf[:num * bytes_per_sample]: a slice of the buffer that includes only the valid portion
            # (i.e. just the bytes that were filled with new data).

            # converts the raw byte data in buf into 16-bit integers, representing the received signal samples (I/Q data).
            # raw_copy = bytes(buf[:num * bytes_per_sample])  # immutable copy of bytes
            data_of_device[device] = np.frombuffer(raw_copy, dtype=np.int16).copy()  # or just store raw_copy and convert later
            #important: frombuffer does not copy the data, it just interprets the memory. Normally, that’s fine—but np.frombuffer creates a read-only view if the underlying buffer is immutable (as in bytes).
            #.copy: Forces NumPy to allocate a new, writable array in memory.
            #In Python, bytes() creates an immutable copy of the data you give it.

            # If you do bytes(buf[:n]) where buf is a bytearray, it returns a new bytes object containing the same bytes, but you cannot change it afterwards (unlike a bytearray, which is mutable).
            #
            # This is exactly why we use it here: it prevents the RX thread from accidentally overwriting the data while another thread (FFT or plotting) is reading it.

            # inside class BladeRFController

    def writer_thread_fn(self, device):
        """
        Dedicated writer thread method (instance method).
        Consumes items from self.write_queue with the form:
            (device_id, timestamp_str, raw_bytes)
        Writes appending to per-device files under ./logs.
        Drains the queue after stop_event is set, then exits.
        """
        # Ensure logs dir exists
        # os.makedirs("logs", exist_ok=True)

        # Continue while not asked to stop OR while there are still items to write
        while not (self.stop_event.is_set() and self.write_queue[device].empty()):
            try: #TODO: check if "self.write_queue[device].get(timeout=0.5)" is correct!
                device, timestamp, raw_bytes = self.write_queue[device].get(timeout=0.5) #TODO: timeout is an unexpected argument, also
            except queue.Empty:
                # no item available right now -> loop and re-check stop condition
                continue

            try:
                # Build a stable filename per device/session. timestamp expected to be a string.
                # If you'd rather keep one rolling file per device, change this to f"./logs/{device}.bin"

                # Use 'ab' to append raw bytes. If multiple runs use the same timestamp you may
                # want to avoid collisions (e.g. include a session id).
                with open(self.log_file[device], 'ab') as f:
                    # raw_bytes should already be bytes or a bytes-like object
                    f.write(raw_bytes)
            except Exception as e:
                # Log error but keep going; disk write errors shouldn't kill the writer thread
                print(f"[writer] Error writing to {self.log_file[device]}: {e}")
            # finally:
            #     # Mark the item done so queue.join() will work
            #     try:
            #         self.write_queue.task_done()
            #     except Exception:
            #         pass

        # When we exit the loop, everything enqueued has been processed (or stop was requested
        # and the queue drained). Clean exit.
        # Optional: print a short message for debugging
        print("Writer thread exiting - queue drained.")

    def _data_handler(self, device):
        """
        Processes the raw I/Q data from a device:
        - Deinterleaves channels
        - Computes complex I/Q arrays
        - Computes FFT in both linear and dB scales
        - Returns results as a dictionary keyed by device
        """

        global data_of_device

        # Check if we have data for this device
        if device not in data_of_device or len(data_of_device[device]) == 0:
            print("Device {} not found, or sth else idk")
            return None

        fs = int(config.getfloat('[bladerf2-rx]', 'rx_samplerate'))
        center_freq_hz = int(config.getfloat('[bladerf2-rx]', 'rx_frequency'))

        # ---- Deinterleave [I0,Q0,I1,Q1,...] ----
        raw = data_of_device[device]
        i0 = raw[0::4].astype(np.float32) #TODO: search whether this is better than int16, float64 etc....
        q0 = raw[1::4].astype(np.float32)
        i1 = raw[2::4].astype(np.float32)
        q1 = raw[3::4].astype(np.float32)

        # Complex arrays per channel
        ch0 = i0 + 1j * q0
        ch1 = i1 + 1j * q1

        # Full interleaved IQ for FFT
        iq = raw[::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)

        results = {}

        # ---------------- Approach 1: Linear FFT (for computation/integration) ----------------
        window = np.hanning(len(iq))
        iq_windowed = iq * window
        norm_factor = np.sum(window) / len(window)

        fft_vals_linear = np.fft.fftshift(np.fft.fft(iq_windowed))
        fft_vals_linear /= (len(iq) * norm_factor)
        power_spectrum = np.abs(fft_vals_linear) ** 2

        freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), 1 / fs))
        freqs_abs = freqs + center_freq_hz  # absolute frequency axis

        # Select only BW for summing (around center frequency)
        half_bw = int(config.getfloat('BW-Summing', 'BW_FOR_BW_SUMMING')) / 2
        valid_indices = np.where(
            (freqs_abs >= center_freq_hz - half_bw) &
            (freqs_abs <= center_freq_hz + half_bw)
        )[0]

        results['linear_fft'] = fft_vals_linear
        results['power_spectrum'] = power_spectrum
        results['freqs'] = freqs_abs
        results['valid_indices'] = valid_indices

        # ---------------- Approach 2: FFT in dB scale (for plotting) ----------------
        buffer = iq.copy()  # reuse IQ array
        buffer -= np.mean(buffer)  # remove DC offset
        window = np.hanning(len(buffer))
        buffer_windowed = buffer * window
        buffer_windowed -= np.mean(buffer_windowed)  # optional extra DC removal

        normalization_factor = np.sum(window) / len(window)
        fft_vals = np.fft.fftshift(np.fft.fft(buffer_windowed))
        fft_vals /= (len(buffer) * normalization_factor)
        fft_db = 20 * np.log10(np.abs(fft_vals) + 1e-12)  # dB scale

        results['fft_db'] = fft_db

        # Optionally, downsample for plotting to save CPU
        downsample_factor = max(1, len(fft_db) // 1024)  # keep ~1024 points for real-time plotting
        results['fft_db_plot'] = fft_db[::downsample_factor]
        results['freqs_plot'] = freqs_abs[::downsample_factor]


        # ------------- "Approach" 3 ==> GUI FFT downsampling -----------------------------

        # Downsample for GUI only (e.g., take 1 out of every 500 samples)
        step = max(len(iq) // int(config.getfloat('FFT-Parameters', 'TARGET_FFT_SIZE')), 1)

        ch0_downsampled = ch0[::step]
        ch1_downsampled = ch1[::step]

        # Update shared_buffer safely
        with buffer_lock[device]:
            try:
                # shared_buffer.put_nowait(iq_downsampled)
                shared_buffer[device][0].append(ch0_downsampled)
                shared_buffer[device][1].append(ch1_downsampled)

            except queue.Full:
                pass  # Drop the new data instead of blocking

        return results

    def start_threads(self, devices, buffer, num_samples, stop_event, rxfiles):
        """
        Launch separate threads to receive data from multiple BladeRF devices simultaneously.
        """
        rx_threads = []

        for i, device in enumerate(devices):
            t = threading.Thread(
                target=self.rx_worker,
                args=(device, buffer[i], num_samples, stop_event, rxfiles[i])
            )
            t.start()
            rx_threads.append(t)

        return rx_threads


    def _set_transceiver_parameters(self):
        """ Set RX/TX parameters """




    def _set_experimental_parameters(self):
        """ I am not sure about those yet """


