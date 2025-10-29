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
bw_powers = deque(maxlen=1000)
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

    def __init__(self, config_path="my_stuff/my_configuration.ini"):
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

        # Call main initialization
        self._initialize_device()

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
        self.rx_pool = [ThreadPool(processes=1) for _ in self.boards]

        #TODO: check whether the thread pool is necessary, perhaps unusable.
        #TODO: perhaps we need to do pool.close or pool.join -> investigate!!!

        # self.tx_pool = ThreadPool(processes=1)

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

    def receive(self, devices, freq, rate, gain,
                tx_start=None, rx_done=None,
                rxfile: str = '', num_samples: int = 1024):

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
            # if i set it to 1000db via the my_config.ini file, it will getv clamped to 60db
            # and this print below shows exactly that:

        for board in self.boards:
            for ch_index in [0, 1]:
                ch = board.Channel(ch_index)
                print(f"[{board.board_name}][RX{ch_index}] Actual gain applied:", ch.gain)














    def _set_transceiver_parameters(self):
        """ Set RX/TX parameters """




    def _set_experimental_parameters(self):
        """ I am not sure about those yet """


