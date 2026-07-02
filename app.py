#!/usr/bin/env python3

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Optional, Tuple

import numpy as np

try:
    from PyQt5 import QtCore, QtWidgets
    import pyqtgraph as pg
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "PyQt5 and pyqtgraph are required to run the GUI. Install dependencies with `pip install -r requirements.txt`."
    ) from exc

try:
    from bladerf import _bladerf
    from bladerf._bladerf import Correction
except ImportError as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "bladeRF Python bindings are required. Build/install the bundled bladeRF bindings before running this app."
    ) from exc


APP_NAME = "BladeRF Radio Telescope"
DEFAULT_CONFIG_PATH = Path("my_stuff/my_configuration.ini")
DEFAULT_LOG_DIR = Path("logs")
DEFAULT_SPECTRUM_DIR = Path("spectra")
DEFAULT_CENTER_FREQUENCY_HZ = 2.398e9
DEFAULT_BANDWIDTH_HZ = 6.0e6
DEFAULT_SAMPLE_RATE_HZ = 12.0e6
DEFAULT_RX_GAIN = 50
DEFAULT_RX_SAMPLES = 98_304
DEFAULT_FFT_HISTORY = 32
DEFAULT_SPECTRUM_HISTORY = 128
DEFAULT_INTEGRATION_SECONDS = 10.0
DEFAULT_WINDOW_SIZE = 8192 * 4
DEFAULT_UPDATE_INTERVAL_MS = 250


@dataclass(frozen=True)
class RuntimeSettings:
    """Runtime parameters used by the receiver and plots."""

    config_path: Path
    center_frequency_hz: float = DEFAULT_CENTER_FREQUENCY_HZ
    bandwidth_hz: float = DEFAULT_BANDWIDTH_HZ
    sample_rate_hz: float = DEFAULT_SAMPLE_RATE_HZ
    rx_gain: int = DEFAULT_RX_GAIN
    rx_samples: int = DEFAULT_RX_SAMPLES
    rx_channels: Tuple[int, int] = (0, 1)
    fft_history: int = DEFAULT_FFT_HISTORY
    spectrum_history: int = DEFAULT_SPECTRUM_HISTORY
    integration_seconds: float = DEFAULT_INTEGRATION_SECONDS
    window_size: int = DEFAULT_WINDOW_SIZE
    update_interval_ms: int = DEFAULT_UPDATE_INTERVAL_MS
    log_dir: Path = DEFAULT_LOG_DIR
    spectrum_dir: Path = DEFAULT_SPECTRUM_DIR
    load_fpga: bool = False
    fpga_image: Optional[Path] = None
    iq_correction_i: Optional[int] = None
    iq_correction_q: Optional[int] = None
    libbladerf_verbosity: str = "INFO"

    @staticmethod
    def from_file(config_path: Path) -> "RuntimeSettings":
        """Load settings from an optional INI file, falling back to defaults."""

        from configparser import ConfigParser

        parser = ConfigParser()
        parser.read(config_path)

        section = parser["bladerf2-rx"] if parser.has_section("bladerf2-rx") else {}
        common = parser["common"] if parser.has_section("common") else {}
        board_name = "bladerf2"
        fpga_section_name = f"{board_name}-load-fpga"
        fpga_section = parser[fpga_section_name] if parser.has_section(fpga_section_name) else {}

        def _get_float(mapping, key: str, default: float) -> float:
            try:
                return float(mapping.get(key, default))
            except Exception:
                return default

        def _get_int(mapping, key: str, default: int) -> int:
            try:
                return int(float(mapping.get(key, default)))
            except Exception:
                return default

        def _get_bool(mapping, key: str, default: bool) -> bool:
            try:
                return str(mapping.get(key, str(default))).strip().lower() in {"1", "true", "yes", "on"}
            except Exception:
                return default

        fpga_image: Optional[Path] = None
        if fpga_section:
            fpga_size = "301"
            image_key = f"image_{fpga_size}kle"
            candidate = fpga_section.get(image_key, "").strip()
            if candidate:
                fpga_image = Path(candidate)

        return RuntimeSettings(
            config_path=config_path,
            center_frequency_hz=_get_float(section, "rx_frequency", DEFAULT_CENTER_FREQUENCY_HZ),
            bandwidth_hz=_get_float(section, "rx_bandwidth", DEFAULT_BANDWIDTH_HZ),
            sample_rate_hz=_get_float(section, "rx_samplerate", DEFAULT_SAMPLE_RATE_HZ),
            rx_gain=_get_int(section, "rx_gain", DEFAULT_RX_GAIN),
            rx_samples=_get_int(section, "rx_num_samples", DEFAULT_RX_SAMPLES),
            rx_channels=(
                _get_int(section, "rx_channel0", 0),
                _get_int(section, "rx_channel1", 1),
            ),
            load_fpga=_get_bool(fpga_section, "enable", False),
            fpga_image=fpga_image,
            libbladerf_verbosity=str(common.get("libbladerf_verbosity", "INFO")).upper(),
        )


@dataclass
class SpectrumState:
    """Shared spectrum data for one device."""

    fft_history_ch0: Deque[np.ndarray]
    fft_history_ch1: Deque[np.ndarray]
    power_history: Deque[np.ndarray]
    frequency_axis_hz: Optional[np.ndarray] = None
    last_fft_ch0_db: Optional[np.ndarray] = None
    last_fft_ch1_db: Optional[np.ndarray] = None
    last_power_db: Optional[np.ndarray] = None


class BladeRFApplication:
    """High-level application controller."""

    def __init__(self, settings: RuntimeSettings):
        self.settings = settings
        self.stop_event = threading.Event()
        self.logger = logging.getLogger(APP_NAME)
        self.device = None
        self.device_name = None
        self.gui_ready = False
        self._shutdown_lock = threading.Lock()

        self.settings.log_dir.mkdir(parents=True, exist_ok=True)
        self.settings.spectrum_dir.mkdir(parents=True, exist_ok=True)

        self.spectrum_state = SpectrumState(
            fft_history_ch0=deque(maxlen=self.settings.fft_history),
            fft_history_ch1=deque(maxlen=self.settings.fft_history),
            power_history=deque(maxlen=self.settings.spectrum_history),
        )

        self.app: Optional[QtWidgets.QApplication] = None
        self.window: Optional[pg.GraphicsLayoutWidget] = None
        self.fft_curve_ch0 = None
        self.fft_curve_ch1 = None
        self.power_curve = None
        self.integrated_curve = None
        self.plot_timer: Optional[QtCore.QTimer] = None
        self.rx_thread: Optional[threading.Thread] = None

    # ------------------------------------------------------------------
    # bladeRF lifecycle
    # ------------------------------------------------------------------
    def _set_verbosity(self) -> None:
        levels = {
            "VERBOSE": 0,
            "DEBUG": 1,
            "INFO": 2,
            "WARNING": 3,
            "ERROR": 4,
            "CRITICAL": 5,
            "SILENT": 6,
        }
        _bladerf.set_verbosity(levels.get(self.settings.libbladerf_verbosity, 2))

    def _probe_device_string(self) -> str:
        self.logger.info("Searching for bladeRF devices")
        devinfos = _bladerf.get_device_list()
        if not devinfos:
            raise RuntimeError("No bladeRF devices were detected.")

        device_info = devinfos[0]
        device_string = "{backend}:device={usb_bus}:{usb_addr}".format(**device_info._asdict())
        if len(devinfos) > 1:
            self.logger.warning("Multiple bladeRF devices detected; using %s", device_string)
        else:
            self.logger.info("Using bladeRF device %s", device_string)
        return device_string

    def _open_device(self) -> None:
        self.device_name = self._probe_device_string()
        self.device = _bladerf.BladeRF(self.device_name)
        self.logger.info("Opened device %s", self.device.board_name)

    def _load_fpga_if_requested(self) -> None:
        if not self.settings.load_fpga or not self.settings.fpga_image:
            self.logger.info("FPGA load disabled in configuration")
            return

        fpga_path = self.settings.fpga_image.expanduser().resolve()
        if not fpga_path.exists():
            raise FileNotFoundError(f"FPGA image not found: {fpga_path}")

        self.logger.info("Loading FPGA image %s", fpga_path)
        self.device.load_fpga(str(fpga_path))
        if self.device.is_fpga_configured():
            self.logger.info("FPGA loaded successfully: %s", self.device.get_fpga_version())
        else:
            raise RuntimeError("FPGA load completed without configuration confirmation.")

    def _apply_iq_corrections(self) -> None:
        if self.settings.iq_correction_i is None or self.settings.iq_correction_q is None:
            return

        for channel in self.settings.rx_channels:
            self.device.set_correction(channel, Correction.DCOFF_I, self.settings.iq_correction_i)
            self.device.set_correction(channel, Correction.DCOFF_Q, self.settings.iq_correction_q)

    def _configure_device(self) -> None:
        for channel in self.settings.rx_channels:
            rx_channel = self.device.Channel(channel)
            rx_channel.frequency = self.settings.center_frequency_hz
            rx_channel.sample_rate = self.settings.sample_rate_hz
            rx_channel.gain = self.settings.rx_gain

        self.device.enable_module(_bladerf.CHANNEL_RX(self.settings.rx_channels[0]), True)
        self.device.enable_module(_bladerf.CHANNEL_RX(self.settings.rx_channels[1]), True)

        bytes_per_complex_sample = 4
        samples_per_buffer = self.settings.window_size
        self.device.sync_config(
            layout=_bladerf.ChannelLayout.RX_X2,
            fmt=_bladerf.Format.SC16_Q11,
            num_buffers=16,
            buffer_size=samples_per_buffer // 2,
            num_transfers=8,
            stream_timeout=5000,
        )

        self.logger.info(
            "Configured RX: center=%.3f MHz, rate=%.3f MHz, gain=%s, buffer=%s samples, bytes/sample=%s",
            self.settings.center_frequency_hz / 1e6,
            self.settings.sample_rate_hz / 1e6,
            self.settings.rx_gain,
            samples_per_buffer,
            bytes_per_complex_sample,
        )

    def initialize(self) -> None:
        self._set_verbosity()
        self._open_device()
        self._load_fpga_if_requested()
        self._configure_device()
        self._apply_iq_corrections()

    # ------------------------------------------------------------------
    # Signal processing
    # ------------------------------------------------------------------
    @staticmethod
    def _deinterleave_samples(raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        i0 = raw[0::4]
        q0 = raw[1::4]
        i1 = raw[2::4]
        q1 = raw[3::4]
        ch0 = i0.astype(np.float32) + 1j * q0.astype(np.float32)
        ch1 = i1.astype(np.float32) + 1j * q1.astype(np.float32)
        return ch0, ch1

    @staticmethod
    def _compute_fft(samples: np.ndarray, sample_rate_hz: float, center_frequency_hz: float, bandwidth_hz: float) -> Tuple[np.ndarray, np.ndarray]:
        if samples.size == 0:
            return np.array([]), np.array([])

        centered = samples - np.mean(samples)
        window = np.hanning(centered.size)
        normalized = window / np.maximum(window.mean(), 1e-12)
        windowed = centered * normalized

        fft_vals = np.fft.fftshift(np.fft.fft(windowed)) / max(windowed.size, 1)
        fft_db = 20.0 * np.log10(np.abs(fft_vals) + 1e-12)
        freqs_hz = np.fft.fftshift(np.fft.fftfreq(windowed.size, d=1.0 / sample_rate_hz)) + center_frequency_hz

        half_bw = bandwidth_hz / 2.0
        mask = (freqs_hz >= center_frequency_hz - half_bw) & (freqs_hz <= center_frequency_hz + half_bw)
        return freqs_hz[mask], fft_db[mask]

    def _integrate_power(self, fft_db: np.ndarray) -> Optional[np.ndarray]:
        if fft_db.size == 0:
            return None

        power_linear = np.power(10.0, fft_db / 10.0)
        self.spectrum_state.power_history.append(power_linear)

        if not self.spectrum_state.power_history:
            return None

        stacked = np.vstack(self.spectrum_state.power_history)
        mean_power = np.mean(stacked, axis=0)
        return 10.0 * np.log10(mean_power + 1e-12)

    def _save_spectrum_snapshot(self, freqs_hz: np.ndarray, fft_db: np.ndarray) -> None:
        if freqs_hz.size == 0 or fft_db.size == 0:
            return

        timestamp = int(time.time())
        out_path = self.settings.spectrum_dir / f"spectrum_{timestamp}.npy"
        np.save(out_path, np.column_stack((freqs_hz, fft_db)))
        self.logger.info("Saved spectrum snapshot to %s", out_path)

    # ------------------------------------------------------------------
    # Worker thread
    # ------------------------------------------------------------------
    def rx_worker(self) -> None:
        bytes_per_complex_sample = 4 * 2  # two channels, 16-bit I/Q each
        num_samples_per_buffer = self.settings.window_size
        raw_buffer = bytearray(num_samples_per_buffer * bytes_per_complex_sample)
        samples_read = 0
        integration_start = time.time()

        while not self.stop_event.is_set():
            remaining = self.settings.rx_samples - samples_read if self.settings.rx_samples > 0 else num_samples_per_buffer
            if self.settings.rx_samples > 0 and remaining <= 0:
                break

            samples_to_read = min(num_samples_per_buffer, remaining) if self.settings.rx_samples > 0 else num_samples_per_buffer
            self.device.sync_rx(raw_buffer, samples_to_read)
            samples_read += samples_to_read

            raw = np.frombuffer(raw_buffer[: samples_to_read * bytes_per_complex_sample], dtype=np.int16).copy()
            if raw.size == 0:
                continue

            ch0, ch1 = self._deinterleave_samples(raw)
            freqs_hz, fft_ch0_db = self._compute_fft(
                ch0,
                self.settings.sample_rate_hz,
                self.settings.center_frequency_hz,
                self.settings.bandwidth_hz,
            )
            _, fft_ch1_db = self._compute_fft(
                ch1,
                self.settings.sample_rate_hz,
                self.settings.center_frequency_hz,
                self.settings.bandwidth_hz,
            )

            if fft_ch0_db.size:
                self.spectrum_state.frequency_axis_hz = freqs_hz
                self.spectrum_state.last_fft_ch0_db = fft_ch0_db
                self.spectrum_state.last_fft_ch1_db = fft_ch1_db
                self.spectrum_state.fft_history_ch0.append(fft_ch0_db)
                self.spectrum_state.fft_history_ch1.append(fft_ch1_db)

                integrated = self._integrate_power(fft_ch0_db)
                if integrated is not None:
                    self.spectrum_state.last_power_db = integrated

                if time.time() - integration_start >= self.settings.integration_seconds:
                    self._save_spectrum_snapshot(freqs_hz, fft_ch0_db)
                    integration_start = time.time()

    # ------------------------------------------------------------------
    # GUI
    # ------------------------------------------------------------------
    def _build_gui(self) -> None:
        self.app = QtWidgets.QApplication(sys.argv)
        self.window = pg.GraphicsLayoutWidget(title=APP_NAME)
        self.window.resize(1200, 900)
        self.window.show()

        fft_plot = self.window.addPlot(title="Live FFT")
        fft_plot.setLabel("bottom", "Frequency", units="MHz")
        fft_plot.setLabel("left", "Magnitude", units="dB")
        fft_plot.setYRange(-120, 20)
        self.fft_curve_ch0 = fft_plot.plot(pen=pg.mkPen("#d95f02", width=1.5), name="Channel 0")
        self.fft_curve_ch1 = fft_plot.plot(pen=pg.mkPen("#1f77b4", width=1.5), name="Channel 1")

        self.window.nextRow()
        power_plot = self.window.addPlot(title="Integrated Power")
        power_plot.setLabel("bottom", "Frequency", units="MHz")
        power_plot.setLabel("left", "Magnitude", units="dB")
        power_plot.setYRange(-120, 20)
        self.power_curve = power_plot.plot(pen=pg.mkPen("#6a3d9a", width=1.5), name="Integrated")

        self.window.nextRow()
        summary_plot = self.window.addPlot(title="Rolling Spectrum History")
        summary_plot.setLabel("bottom", "Bin")
        summary_plot.setLabel("left", "Magnitude", units="dB")
        summary_plot.setYRange(-120, 20)
        self.integrated_curve = summary_plot.plot(pen=pg.mkPen("#33a02c", width=1.5))

        self.plot_timer = QtCore.QTimer()
        self.plot_timer.timeout.connect(self._refresh_gui)  # type: ignore[attr-defined]
        self.plot_timer.start(self.settings.update_interval_ms)

    def _refresh_gui(self) -> None:
        freqs_hz = self.spectrum_state.frequency_axis_hz
        if freqs_hz is None:
            return

        freqs_mhz = freqs_hz / 1e6
        if self.spectrum_state.last_fft_ch0_db is not None:
            self.fft_curve_ch0.setData(freqs_mhz, self.spectrum_state.last_fft_ch0_db)
        if self.spectrum_state.last_fft_ch1_db is not None:
            self.fft_curve_ch1.setData(freqs_mhz, self.spectrum_state.last_fft_ch1_db)
        if self.spectrum_state.last_power_db is not None:
            self.power_curve.setData(freqs_mhz[: self.spectrum_state.last_power_db.size], self.spectrum_state.last_power_db)

        if self.spectrum_state.power_history:
            stacked = np.vstack(self.spectrum_state.power_history)
            rolling_mean = np.mean(stacked, axis=0)
            self.integrated_curve.setData(np.arange(rolling_mean.size), 10.0 * np.log10(rolling_mean + 1e-12))

    # ------------------------------------------------------------------
    # Shutdown handling
    # ------------------------------------------------------------------
    def shutdown(self) -> None:
        with self._shutdown_lock:
            if self.stop_event.is_set():
                return
            self.stop_event.set()

            try:
                if self.device is not None:
                    for channel in self.settings.rx_channels:
                        self.device.enable_module(_bladerf.CHANNEL_RX(channel), False)
                    self.device.close()
            finally:
                if self.app is not None:
                    self.app.quit()

    def _install_signal_handlers(self) -> None:
        signal.signal(signal.SIGINT, lambda *_: self.shutdown())
        signal.signal(signal.SIGTERM, lambda *_: self.shutdown())

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------
    def run(self) -> int:
        self.initialize()
        self._install_signal_handlers()
        self._build_gui()

        self.rx_thread = threading.Thread(target=self.rx_worker, name="bladeRF-rx", daemon=True)
        self.rx_thread.start()

        assert self.app is not None
        return self.app.exec_()


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BladeRF radio telescope viewer")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to an optional INI configuration file.",
    )
    return parser.parse_args(argv)


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def main(argv: Optional[list[str]] = None) -> int:
    configure_logging()
    args = parse_args(argv)
    settings = RuntimeSettings.from_file(args.config)
    app = BladeRFApplication(settings)

    try:
        return app.run()
    except KeyboardInterrupt:
        app.shutdown()
        return 0
    except Exception as exc:
        logging.getLogger(APP_NAME).exception("Application failed: %s", exc)
        app.shutdown()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

