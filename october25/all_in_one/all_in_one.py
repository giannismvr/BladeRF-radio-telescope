import threading
import datetime
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


#------------------------- MAIN CLASS --------------------------------------------------------------------

#TODO: replace .ini file with a single file containing everything
#TODO: assess whether config.read/parser is necessary and if not, delete it

class BladeRFController:
    """
    Encapsulates BladeRF initialization, FPGA loading, version reporting, and threading setup.
    """

    #TODO: incorporate all def's that have to do with device in "class Device". Careful, not initialization, but stuff like print_board_info()

    class Device(_bladerf.BladeRF):
        def __init__(self, device_str):
            super().__init__(device_str)
            self.device_string = device_str

            # self.device = _bladerf.BladeRF(self.device_string)
            self.name = self.board_name

    def __init__(self, config_path="/all_parameters.ini", calc_ch0 = False, calc_ch1 = False, calc_ch0_ch1 = True):

        self.devices = []







        self.config_path = config_path
        self.config = ConfigParser()
        self.config.read(self.config_path)

        # self.boards = []
        self.tx_pool = None
        self.rx_pool = None

        self.rx_num_samples = None
        self.BW = None
        self.fs = None
        self.rx_freq=None
        self.rx_channel0 = None
        self.rx_channel1 = None
        print("efoihwaoufhaw")
        #TODO: perhaps it would be better to initialize everything in the constructor, as some attributes may not be visible in the code, if the respective objects are declared in the "dict_init" def.... !!!!
        self.write_queue = {}
        self.data_of_device = {}

        # Initialize a log filename per device
        self.log_file = {}  # dictionary to hold file paths per device
        self.error_logs = {}

        self.buffer_lock = {}
        self.shared_buffer = {}

        self.stop_event = threading.Event()
        self.plot_timer = QtCore.QTimer()


        self.calc_ch0 = calc_ch0
        self.calc_ch1 = calc_ch1
        self.calc_ch0_ch1 = calc_ch0_ch1

        self.results = {}

        self.calculation = {}

        self.num_samples = {}

        # self.my_bladerf = []

        self.config_path = 'all_parameters.ini'

        self.config = ConfigParser()
        self.config.read(self.config_path)








        self.fft_curves = {}  # Store pyqtgraph curves per device & mode
        self.prev_fft_db = {}  # Store previous FFT for smoothing
        self.alpha = 0.3  # smoothing factor

        self.device_plots = {}  # store PlotWidget and curves per device

        self.config_path = config_path
        self.bytes_per_sample = int(self.config.getfloat('bladerf2-rx', 'bytes_per_sample')) # 2 bytes I + 2 bytes Q == 4 bytes for 1 channel, so for 2 channels, it will be 2*that = 2* 4 =8
        self.gain = int(self.config.getfloat('bladerf2-rx', 'rx_gain'))
        self.rate = int(self.config.getfloat('bladerf2-rx', 'rx_samplerate'))
        self.freq = int(self.config.getfloat('bladerf2-rx', 'rx_frequency'))


        # Call main initialization
        #TODO: check if this is the correct sequence for initialization




        self.bladerf_object_list = [self.devices]


        # self._print_board_info(self.my_bladerf)
        self._initialize_device()
        # time.sleep(2)
        print("a")
        self._initialize_dictionaries()
        # time.sleep(2)
        print("b")
        self._set_verbosity()
        # time.sleep(2)
        print("a")

        # time.sleep(2)
        print("a")

        print("PRINTEEEED")
        # time.sleep(2)
        print("a")


        print("self:", self.bladerf_object_list)
        self._load_fpga_if_enabled(self.devices)
        # time.sleep(2)
        print("a")
        self._configure_rx_parameters()
        # time.sleep(2)
        print("a")

        self.start_threads(self.devices)







        # self.rx_pool = [ThreadPool(processes=1) for _ in self.boards]

        # TODO: check whether the thread pool is necessary, perhaps unusable.
        # TODO: perhaps we need to do pool.close or pool.join -> investigate!!!

    def _create_devices(self):
        # uuts = self._probe_bladerf()
        for idx, uut_str in enumerate(self.uuts):
            # self.devices[idx] = BladeRFController.Device(uut)
            try:
                dev = BladeRFController.Device(uut_str)
            except Exception:
                # handle per-device init failure gracefully
                raise
            self.devices.append(dev)



    # ──────────────── Main initializer ────────────────
    def _initialize_device(self):


        """Runs the full initialization sequence."""
        self._set_verbosity()
        self.uuts = self._probe_bladerf()
        self._create_devices()


        if not self.devices:
            print("No BladeRF detected. Exiting.")
            self.shutdown(-1)

        # Create BladeRF objects for each detected device
        # self.boards = []
        for device in self.devices:
            try:
                # board = _bladerf.BladeRF(device_str)
                # self.boards.append(board)
                print(f"Initialized board: {device.name}")
            except _bladerf.BladeRFError as e:
                print(f"Error initializing device {device.name}: {e}")

        # Optionally, still load FPGA for each board
        for board in self.bladerf_object_list:
            self._load_fpga_if_enabled(board)
            self._print_board_info(board)

        # self.board = _bladerf.BladeRF(self.device)

        # Thread pools (for RX/TX)
        # Only RX tasks







    def _initialize_dictionaries(self):


        # TODO: where should i move all these initializations?
        # self.tx_pool = ThreadPool(processes=1)
        # Initialize per-device locks and buffers


        for device in self.devices:  # devices is a list of device identifiers
            self.buffer_lock[device] = threading.Lock()
            self.shared_buffer[device] = {
                0: queue.Queue(maxsize=10),  # RX channel 0
                1: queue.Queue(maxsize=10),  # RX channel 1
                2: queue.Queue(maxsize=10),  # optional cross-channel or combined data
            }



        # Bounded queue to hold data to write
        #TODO: find out what the maxsize should be for this!!!!!!!!!! ---------------------------------------------- SOS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




        for device in self.devices:
            self.log_file[device] = f"./logs/{device}.bin"
            # Optional: create/truncate the file at initialization
            with open(self.log_file[device], 'wb') as f:
                pass  # just create an empty file


        for device in self.devices:
            self.error_logs[device] = f"./error_logs/{device}.bin"
            # Optional: create/truncate the file at initialization
            with open(self.error_logs[device], 'wb') as f:
                pass  # just create an empty file

        for device in self.devices:
            self.write_queue[device] = queue.Queue(maxsize=200)
        for device in self.devices:
            self.data_of_device[device] = queue.Queue(maxsize=200)

        for device in self.devices:
            self.prev_fft_db[device] = {"CH0": None, "CH1": None, "CH0+CH1": None}
            self.results[device] = {"CH0": queue.Queue(maxsize=10),
                                 "CH1": queue.Queue(maxsize=10),
                                 "CH0+CH1": queue.Queue(maxsize=10)}

        # Store the frequency axis for each device
        self.last_freqs = {device: np.array([]) for device in self.devices}

        for device in self.devices:
            self.num_samples[device] = int(self.config.getfloat('bladerf2-rx', 'rx_num_samples'))






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
    def _print_board_info(self, bladerf_object_list):
        """
        Print board information for a list of BladeRF devices.
        """
        for i, board in enumerate(bladerf_object_list):
            print(f"\n=== Device {i}: {board} ===")
            try:
                print("Board name:       ", board.board_name)
                print("Firmware version: ", board.get_fw_version())
                print("FPGA version:     ", board.get_fpga_version())
            except Exception as e:
                print(f"Error reading board info for device {i}: {e}")



    # ──────────────── FPGA load ────────────────
    def _load_fpga_if_enabled(self, bladerf_object_list):
        """Load FPGA image(s) if the configuration enables it."""
        print([bladerf_object_list])
        print("DEBUG:", type(bladerf_object_list), bladerf_object_list)
        print(f"bladerf_object_list: {type(bladerf_object_list)}")
        for i, board in enumerate(bladerf_object_list):
            board_name = board.board_name
            enabled = self.config.getboolean(f"{board_name}-load-fpga", 'enable')
            if not enabled:
                print(f"Skipping FPGA load per config, for board: {board_name} ")
                continue

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
        for board in self.devices:
            try:
                board.close()
            except Exception:
                pass



        sys.exit(error)

    def _configure_rx_parameters(
            self,
    ):



        # for idx, device_current in enumerate(self.devices): #TODO: uncomment and fix appropriately
        #     dcoff_i = _bladerf.ffi.new("int16_t *")
        #     ret_i = _bladerf.libbladeRF.bladerf_get_correction(device_current.dev[0], BLADERF_MODULE_RX, BLADERF_CORR_DC_I,
        #                                                        dcoff_i)
        #
        #     dcoff_q = _bladerf.ffi.new("int16_t *")
        #     ret_q = _bladerf.libbladeRF.bladerf_get_correction(device_current.dev[0], BLADERF_MODULE_RX, BLADERF_CORR_DC_Q,
        #                                                        dcoff_q)
        #
        #     if ret_i == 0 and ret_q == 0:
        #         print(f"[Device {idx}] DCOFF_I = {dcoff_i[0]}, DCOFF_Q = {dcoff_q[0]}")
        #     else:
        #         print(f"[Device {idx}] Failed to get correction values: ret_i={ret_i}, ret_q={ret_q}")

        for board_idx, board in enumerate(self.devices):
            for ch_index in [0, 1]:  # RX channels
                ch = board.Channel(ch_index)  # create channel object
                ch.frequency = self.freq
                ch.sample_rate = self.rate
                ch.gain = self.gain
            print(f"[{board.board_name}] RX channels configured: freq={self.freq}, rate={self.rate}, gain={self.gain}")

        # TODO: this must be done after the gain, freq settings (eg ch.gain = ....)
        for board in self.devices:
            # Enable RX channels 0 and 1
            for ch_index in [0, 1]:
                board.enable_module(_bladerf.CHANNEL_RX(ch_index), True)
                print("succesfully enabled channel RX", ch_index)
            print(f"[{board.board_name}] RX channels enabled")

            # Actual gain I am getting - no matter what.
            # Doesn't matter if I set it to 1000000db
            # For manual gain control the range is from -15db to 60db
            # if i set it to 1000db via the my_config.ini file, it will get clamped to 60db
            # and this print below shows exactly that:

        for board in self.devices:
            for ch_index in [0, 1]:
                ch = board.Channel(ch_index)
                print(f"[{board.board_name}][RX{ch_index}] Actual gain applied:", ch.gain)

        # Setup synchronous stream
        # 8 bytes total per interleaved complex sample pair (CH0 + CH1).
        bytes_per_sample = 4 * 2  # 2 bytes I + 2 bytes Q == 4 bytes for 1 channel, so for 2 channels, it will be 2*that = 2* 4 =8
        num_samples_per_buffer = 8192 * 2  # total samples (I+Q pairs), shared across RX0 and RX1, NOT INTERLEAVED
        num_samples_interleaved = num_samples_per_buffer // 2
        # buffer = {}  # dictionary to hold buffers per device
        # for i, device in enumerate(devices):
        #     buffer[i] = bytearray(num_samples_per_buffer * bytes_per_sample)
        # num_samples_read = [0 for _ in devices]



        # RX_X1 → Single RX channel active. Buffer contains only CH0 samples: [I0, Q0, I0, Q0, ...].
        #
        # RX_X2 → Both RX channels active. Buffer contains interleaved CH0 + CH1 samples: [I0, Q0, I1, Q1, I0, Q0, I1, Q1, ...].

        # for channel 1 AND channel 2 according to gpt
        # RX_X2 literally enables both RX channels, not just RX2.
        for board in self.devices:
            board.sync_config(
                layout=_bladerf.ChannelLayout.RX_X2,
                fmt=_bladerf.Format.SC16_Q11,
                num_buffers=16,  # keep or adjust per channel count
                buffer_size=num_samples_interleaved,  # interleaved buffer size
                num_transfers=8,  # keep or adjust per channel count
                stream_timeout=5000  # adjust if needed
            )
            print(f"Sync configured for board {board.board_name}")

        for board in self.devices:
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
        num_samples_per_channel = num_samples_per_buffer // 2  # TODO: or //2

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




    def rx_worker(self, device):
        """
        Worker function for receiving data from a single BladeRF device.
        """



        # bytes_per_sample = 4 * 2  # 2 bytes I + 2 bytes Q == 4 bytes for 1 channel, so for 2 channels, it will be 2*that
        num_samples_per_buffer = 8192 * 2  # total samples (I+Q pairs), shared across RX0 and RX1, NOT INTERLEAVED
        num_samples_interleaved = num_samples_per_buffer // 2
        buf = bytearray(num_samples_per_buffer * self.bytes_per_sample)
        num_samples_read = 0

        while not self.stop_event.is_set():
            if self.num_samples[device] > 0 and num_samples_read == self.num_samples:
                break
            elif self.num_samples[device] > 0:
                print(f"len buf =={len(buf)}")  # this prints (bytes_per_sample * num_samples_per_buffer)
                num = min(len(buf) // self.bytes_per_sample,
                          self.num_samples[device] - num_samples_read)
                if (self.num_samples[device] - num_samples_read) < (len(buf) // self.bytes_per_sample):
                    print(f"Found it: num_samples - num_samples_read === {self.num_samples[device] - num_samples_read}")
                else:
                    print(f"Other Case (usual), len(buf) // bytes_per_sample === {len(buf) // self.bytes_per_sample}")
            else:
                print("What? Num_samples < 0 ????? how can that happen?? Num_samples is: ", self.num_samples)
                num = len(buf) // self.bytes_per_sample

            # Receive samples into the buffer
            device.sync_rx(buf, num)

            # This keeps track of how many samples have been read so far.
            num_samples_read += num

            # Make an immutable copy of buffer for safe passing
            raw_copy = bytes(buf[:num_samples_per_buffer * self.bytes_per_sample])
            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')

            try:
                # Push to writer queue (blocks if queue is full)
                self.write_queue[device].put((timestamp, raw_copy), timeout=1.0)
            except queue.Full:
                print(f"[RX] Error: write queue for device {device} is full! Data may be delayed or lost.")
                # Get current timestamp
                error_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                # Append to error_logs file
                with open("error_logs/error_logs.txt", "a") as ef:
                    ef.write(f"{error_ts} - Device {device} write queue full\n")


            # Write to file main "all-inclusive" I-Q data file
            # outfile.write(buffer[:num * bytes_per_sample])

            # buf[:num * bytes_per_sample]: a slice of the buffer that includes only the valid portion
            # (i.e. just the bytes that were filled with new data).
            try:
                # converts the raw byte data in buf into 16-bit integers, representing the received signal samples (I/Q data).
                # raw_copy = bytes(buf[:num * bytes_per_sample])  # immutable copy of bytes

                #TODO: dont forget to do this on the fft thread side: self.data_of_device[device] = np.frombuffer(raw_copy, dtype=np.int16).copy()  # or just store raw_copy and convert later
                self.data_of_device[device].put(raw_copy)
                print("AAerkgverjovebjierbjgw4e", self.data_of_device[device].qsize(), "hjkefbihqwbf", len(raw_copy))
            except queue.Full:
                print(f"[RX] Data queue full for device {device} - FFT may skip a sample")
                error_ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                # Append to error_logs file
                with open("error_logs/error_logs.txt", "a") as ef:
                    ef.write(f"{error_ts} - Device {device} had a full queue of 'data_of_device'. \n")


                #important: frombuffer does not copy the data, it just interprets the memory. Normally, that’s fine—but np.frombuffer creates a read-only view if the underlying buffer is immutable (as in bytes).
                #.copy: Forces NumPy to allocate a new, writable array in memory.
                #In Python, bytes() creates an immutable copy of the data you give it.

                # If you do bytes(buf[:n]) where buf is a bytearray, it returns a new bytes object containing the same bytes, but you cannot change it afterwards (unlike a bytearray, which is mutable).
                #
                # This is exactly why we use it here: it prevents the RX thread from accidentally overwriting the data while another thread (FFT or plotting) is reading it.



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
                timestamp, raw_bytes = self.write_queue[device].get(timeout=0.5) #TODO: timeout is an unexpected argument, also
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
        while not self.stop_event.is_set():

            fft_results = {}


            # Check if we have data for this device
            if device not in self.data_of_device or self.data_of_device[device].empty():
                print("Device {} not found, or sth else idk")
                print(self.data_of_device[device])
                time.sleep(0.01)  # optional, avoid busy-looping
                continue



            fs = int(self.config.getfloat('bladerf2-rx', 'rx_samplerate'))
            center_freq_hz = int(self.config.getfloat('bladerf2-rx', 'rx_frequency'))

            # Get the raw bytes from the queue (non-blocking)
            try:
                raw_bytes = self.data_of_device[device].get_nowait()  # raw bytes from RX thread
            except queue.Empty:
                # No data yet, skip iteration
                time.sleep(0.001)  # tiny sleep prevents CPU spinning
                continue


            # Convert bytes → int16 NumPy array, make independent copy
            raw = np.frombuffer(raw_bytes, dtype=np.int16).copy()

            # ---- Deinterleave [I0,Q0,I1,Q1,...] ----
            # TODO: search whether this is better than int16, float64 etc....
            i0 = raw[0::4]
            q0 = raw[1::4]
            i1 = raw[2::4]
            q1 = raw[3::4]

            # Complex arrays per channel
            ch0 = i0 + 1j * q0
            ch1 = i1 + 1j * q1

            ch_combined = ch0 + ch1

            # Full interleaved IQ for FFT
            # iq = raw[::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)



            # ---------------- Approach 1: Linear FFT (for computation/integration) ----------------
            # --- Per-channel FFT calculation with proper normalization and DC removal ---

            #TODO: check old main.py fft calculations and change these appropriately...i dont think they are good and comprehensive
            # Channel 0
            if self.calc_ch0:
                window0 = np.hanning(len(ch0))
                norm_factor0 = np.sum(window0) / len(window0)
                ch0_windowed = ch0 * window0
                ch0_windowed -= np.mean(ch0_windowed)  # remove DC after windowing
                fft_ch0_linear = np.fft.fftshift(np.fft.fft(ch0_windowed))
                fft_ch0_linear /= (len(ch0_windowed) * norm_factor0)
                fft_ch0_power = np.abs(fft_ch0_linear) ** 2
                fft_ch0_db = 20 * np.log10(np.abs(fft_ch0_linear) + 1e-12)

                fft_results['ch0_fft_linear'] = fft_ch0_linear
                fft_results['ch0_power'] = fft_ch0_power
                fft_results['ch0_fft_db'] = fft_ch0_db

                try:
                    self.results[device]["CH0"].put_nowait(fft_results['ch0_fft_db'])
                except queue.Full:
                    pass




            # Channel 1
            if self.calc_ch1:
                window1 = np.hanning(len(ch1))
                norm_factor1 = np.sum(window1) / len(window1)
                ch1_windowed = ch1 * window1
                ch1_windowed -= np.mean(ch1_windowed)  # remove DC after windowing
                fft_ch1_linear = np.fft.fftshift(np.fft.fft(ch1_windowed))
                fft_ch1_linear /= (len(ch1_windowed) * norm_factor1)
                fft_ch1_power = np.abs(fft_ch1_linear) ** 2
                fft_ch1_db = 20 * np.log10(np.abs(fft_ch1_linear) + 1e-12)

                fft_results['ch1_fft_linear'] = fft_ch1_linear
                fft_results['ch1_power'] = fft_ch1_power
                fft_results['ch1_fft_db'] = fft_ch1_db

                try:
                    self.results[device]["CH1"].put_nowait(fft_results['ch1_fft_db'])
                except queue.Full:
                    pass


            # Combined channel (ch0 + ch1)
            if self.calc_ch0_ch1:
                # sum channels
                window_comb = np.hanning(len(ch_combined))
                norm_factor_comb = np.sum(window_comb) / len(window_comb)
                combined_windowed = ch_combined * window_comb
                combined_windowed -= np.mean(combined_windowed)  # remove DC after windowing
                fft_comb_linear = np.fft.fftshift(np.fft.fft(combined_windowed))
                fft_comb_linear /= (len(combined_windowed) * norm_factor_comb)
                fft_comb_power = np.abs(fft_comb_linear) ** 2
                fft_comb_db = 20 * np.log10(np.abs(fft_comb_linear) + 1e-12)

                fft_results['ch0_ch1_fft_linear'] = fft_comb_linear
                fft_results['ch0_ch1_power'] = fft_comb_power
                fft_results['ch0_ch1_fft_db'] = fft_comb_db

                try:
                    self.results[device]["CH0+CH1"].put_nowait(fft_results['ch0_ch1_fft_db'])
                except queue.Full:
                    pass

            #----------------------------------------------------------------------------------------------

            # Use the channel length (same for ch0, ch1, ch0+ch1)

            freqs = np.fft.fftshift(np.fft.fftfreq(len(ch0), 1 / fs)) #or ch1, since len(ch0) == len(ch1)
            freqs_abs = freqs + center_freq_hz  # absolute frequency axis

            # Save frequency axis for plotting
            self.last_freqs[device] = freqs_abs

            # Select only BW for summing (around center frequency)
            half_bw = int(self.config.getfloat('BW-Summing', 'BW_FOR_BW_SUMMING')) / 2
            valid_indices = np.where(
                (freqs_abs >= center_freq_hz - half_bw) &
                (freqs_abs <= center_freq_hz + half_bw)
            )[0]

            # Store in results (shared for all channels)
            fft_results['freqs'] = freqs_abs
            fft_results['valid_indices'] = valid_indices

            # ------------- "Approach" 3 ==> GUI FFT downsampling -----------------------------

            # Determine downsampling step based on TARGET_FFT_SIZE
            step = max(len(ch0) // int(self.config.getfloat('FFT-Parameters', 'TARGET_FFT_SIZE')), 1)

            # Downsample channels according to flags
            ch0_downsampled = ch0[::step] if self.calc_ch0 else None
            ch1_downsampled = ch1[::step] if self.calc_ch1 else None
            ch0ch1_downsampled = (ch0 + ch1)[::step] if self.calc_ch0_ch1 else None

            # Update shared_buffer safely

            # Build list of (index, data) to append
            channels = [
                (self.calc_ch0, 0, ch0_downsampled),
                (self.calc_ch1, 1, ch1_downsampled),
                (self.calc_ch0_ch1, 2, ch0ch1_downsampled),
            ]

            with self.buffer_lock[device]:
                for flag, idx, data in channels:
                    if flag and data is not None:
                        try:
                            self.shared_buffer[device][idx].put_nowait(data)
                        except queue.Full:
                            pass  # drop new data if full


    def start_plot_thread(self):
        """
        Creates 1 plot per device, 3 curves per plot (CH0, CH1, CH0+CH1),
        and starts a QTimer to update them.
        """
        for device in self.devices:
            win = pg.GraphicsLayoutWidget(title=f"FFT Plot - {device}")
            win.show()
            plot = win.addPlot(title=f"{device} FFT (dB)")
            plot.setLabel('bottom', 'Frequency (MHz)')
            plot.setLabel('left', 'Magnitude (dB)')
            plot.setYRange(-100, 10, padding=0)  # adjust as needed

            # 3 curves per device
            ch0_curve = plot.plot(pen='r', name="CH0")
            ch1_curve = plot.plot(pen='b', name="CH1")
            ch_comb_curve = plot.plot(pen='g', name="CH0+CH1")

            self.device_plots[device] = {
                "plot": plot,
                "CH0": ch0_curve,
                "CH1": ch1_curve,
                "CH0+CH1": ch_comb_curve
            }

        # Timer updates all device plots
        self.plot_timer.timeout.connect(self._update_device_plots)
        self.plot_timer.start(100)  # every 100 ms

    def _update_device_plots(self):
        """
        Pulls latest FFT data from results queues and updates all device plots.
        """
        for device in self.devices:
            curves = self.device_plots[device]

            for mode in ["CH0", "CH1", "CH0+CH1"]:
                fft_curve = curves[mode]

                # Pull latest FFT from queue (non-blocking)
                latest_fft = None
                while not self.results[device][mode].empty():
                    try:
                        latest_fft = self.results[device][mode].get_nowait()
                    except queue.Empty:
                        break

                if latest_fft is not None:
                    freqs = self.last_freqs.get(device)
                    if freqs is not None:
                        fft_curve.setData(freqs / 1e6, latest_fft)  # MHz

    def start_threads(self, devices):
        """
        Launch separate threads to receive data from multiple BladeRF devices simultaneously,
        plus a single plot timer thread.
        """
        rx_threads = []
        data_threads = []
        writer_threads = []
        print("self. devices print:", self.devices)
        for i, device in enumerate(self.devices):
            # RX worker thread
            t_rx = threading.Thread(
                target=self.rx_worker,
                args=(device,), # <-- note the comma
                daemon=True
            )
            t_rx.start()
            rx_threads.append(t_rx)

            # Data handler thread
            t_data = threading.Thread(
                target=self._data_handler,  # calls _data_handler in a loop
                args=(device,),
                daemon=True
            )
            t_data.start()
            data_threads.append(t_data)

            # Writer thread (consumes self.write_queue[device] and writes to disk)
            t_writer = threading.Thread(
                target=self.writer_thread_fn,
                args=(device,),
                daemon=True
            )
            t_writer.start()
            writer_threads.append(t_writer)



        # Start plot timer (Qt-safe)
        self.start_plot_thread()

        return rx_threads, data_threads, writer_threads


if __name__ == "__main__":
    trial = BladeRFController()
    time.sleep(100000)
