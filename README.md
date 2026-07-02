# BladeRF Radio Telescope

A polished Python/Qt application for receiving, visualizing, and inspecting bladeRF IQ data in real time.

This repository contains a cleaned-up public version of the project intended for portfolio and presentation purposes. The current codebase focuses on readability, maintainability, and a professional layout.

## Contents

- `app.py` — primary application entry point
- `requirements.txt` — Python dependencies
- `bladeRF/` — bundled bladeRF source tree and Python bindings
- `my_stuff/my_configuration.ini` — example runtime configuration

## Features

- bladeRF device discovery and initialization
- Optional FPGA loading from configuration
- Dual-channel RX configuration
- Real-time FFT visualization with PyQt5 and pyqtgraph
- Rolling power-spectrum integration and snapshot saving
- Clean shutdown handling for GUI and hardware resources

## Requirements

- Python 3.9 or newer
- A working bladeRF host setup
- PyQt5, pyqtgraph, and NumPy
- bladeRF Python bindings

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

## Running the application

From the repository root:

```bash
python app.py --config my_stuff/my_configuration.ini
```

If your configuration file is already at the default path, the `--config` argument may be omitted.

## Configuration

The default configuration file lives at `my_stuff/my_configuration.ini`. It controls:

- bladeRF verbosity
- RX frequency, bandwidth, sample rate, and gain
- Optional FPGA image loading
- RX buffer sizes and sample counts

You can duplicate this file and adjust it for your environment.

## Repository notes

- The `develop` branch preserves the work-in-progress history.
- The `master` branch is intended to present the cleaned-up version.
- The local virtual environment is intentionally not tracked in Git.

## License

This repository includes upstream bladeRF source material under the license terms provided by the bladeRF project. Refer to the bundled `bladeRF/` documentation for details.

