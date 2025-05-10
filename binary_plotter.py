import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser

# === Load config ===
config = ConfigParser()
config.read('/Users/giannis/PycharmProjects/final_radiotelescope/my_stuff/my_configuration.ini')

fs = float(config['bladerf2-rx']['rx_samplerate'])       # in Hz
fc = float(config['bladerf2-rx']['rx_frequency'])        # in Hz
bw = float(config['bladerf2-rx']['rx_bandwidth'])        # in Hz
bin_path = '/Users/giannis/PycharmProjects/final_radiotelescope/my_stuff/aaanew_my_rx_samples.bin'

# === Load binary file ===
raw = np.fromfile(bin_path, dtype=np.int16)

# Check if the file is read correctly and contains data
if len(raw) == 0:
    print("Error: The binary file is empty.")
    exit(1)

# Convert to I/Q samples (interleaved I and Q)
iq = raw[::2] + 1j * raw[1::2]

# Check that we have data in iq
if len(iq) == 0:
    print("Error: No valid I/Q samples in the data.")
    exit(1)

# === Optional: Remove DC offset ===
iq -= np.mean(iq)

# === Apply window ===
window = np.hanning(len(iq))
iq_windowed = iq * window
norm_factor = np.sum(window) / len(window)

# === Compute FFT ===
fft_vals = np.fft.fftshift(np.fft.fft(iq_windowed))
fft_vals /= (len(iq) * norm_factor)  # normalize
fft_db = 20 * np.log10(np.abs(fft_vals) + 1e-12)

# === Frequency axis ===
freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1/fs))
freqs_mhz = (freqs + fc) / 1e6  # Absolute freq in MHz

# === Optionally filter to BW only ===
bw_half = bw / 2 / 1e6
valid = (freqs_mhz >= (fc / 1e6 - bw_half)) & (freqs_mhz <= (fc / 1e6 + bw_half))

# === Plot ===
plt.figure(figsize=(10, 5))
plt.plot(freqs_mhz[valid], fft_db[valid], color='green')
plt.title("FFT from recorded BladeRF samples")
plt.xlabel("Frequency (MHz)")
plt.ylabel("Magnitude (dB)")
plt.grid(True)
plt.tight_layout()
plt.show()