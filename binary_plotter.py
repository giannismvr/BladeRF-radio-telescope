import numpy as np
import matplotlib.pyplot as plt
from configparser import ConfigParser





# === Load config ===
config = ConfigParser()
config.read('/Users/giannis/PycharmProjects/final_radiotelescope/my_stuff/my_configuration.ini')

fs = float(config['bladerf2-rx']['rx_samplerate'])       # in Hz
fc = float(config['bladerf2-rx']['rx_frequency'])        # in Hz
bw = float(config['bladerf2-rx']['rx_bandwidth'])        # in Hz
bin_path = '/Users/giannis/PycharmProjects/final_radiotelescope/logs/all_inclusive_2025-05-21_03-59-18.bin'
special_bw = 10e6  # 10 MHz, or whatever range you want (in Hz)

# === Load binary file ===
raw = np.fromfile(bin_path, dtype=np.int16)
if len(raw) == 0:
    print("Error: The binary file is empty.")
    exit(1)

# Convert to I/Q samples
iq = raw[::2] + 1j * raw[1::2]
if len(iq) == 0:
    print("Error: No valid I/Q samples in the data.")
    exit(1)

#
# chunk_size = 8192  # Or whatever your real-time GUI is using (e.g., 8192 or 4096)
# iq = iq[:chunk_size]
#





def averaged_fft(iq, fs, fc, chunk_size=8192, overlap=0.5, skip_dc=True):
    step = int(chunk_size * (1 - overlap))
    if len(iq) < chunk_size:
        raise ValueError(f"Not enough IQ samples: need at least {chunk_size}, got {len(iq)}")

    num_chunks = (len(iq) - chunk_size) // step
    if num_chunks <= 0:
        raise ValueError("Too few IQ samples for the given chunk size and overlap.")

    fft_accum = None

    for i in range(num_chunks):
        start = i * step
        chunk = iq[start:start + chunk_size]

        # === Remove DC before windowing
        chunk -= np.mean(chunk)

        # === Apply window
        window = np.hanning(len(chunk))
        chunk_windowed = chunk * window

        # === Optionally re-remove DC after windowing
        chunk_windowed -= np.mean(chunk_windowed)

        # === FFT and normalize
        fft_vals = np.fft.fftshift(np.fft.fft(chunk_windowed))
        fft_vals /= (len(chunk) * (np.sum(window) / len(window)))

        # === Power spectrum
        power = np.abs(fft_vals) ** 2

        if skip_dc:
            center_bin = len(power) // 2
            power[center_bin] = 0  # Mute DC bin

        if fft_accum is None:
            fft_accum = power
        else:
            fft_accum += power

    fft_avg = fft_accum / num_chunks
    freqs = np.fft.fftshift(np.fft.fftfreq(chunk_size, d=1 / fs)) + fc
    fft_db = 10 * np.log10(fft_avg + 1e-12)
    return freqs / 1e6, fft_db  # Convert to MHz









# === Optional: Remove DC offset ===
# iq -= np.mean(iq)

# bladerf_set_correction()

freqs_mhz, fft_db = averaged_fft(iq, fs, fc, chunk_size=8192, overlap=0.5, skip_dc=True)


# === Apply window ===
# === DC Removal and Windowing ===
# iq = iq - np.mean(iq)  # Remove DC BEFORE windowing
# window = np.hanning(len(iq))
# iq_windowed = iq * window
# iq_windowed -= np.mean(iq_windowed)  # Optional: also remove DC after windowing
#
# norm_factor = np.sum(window) / len(window)
#
# # === Compute FFT ===
# fft_vals = np.fft.fftshift(np.fft.fft(iq_windowed))
# fft_vals /= (len(iq) * norm_factor)
# fft_db = 20 * np.log10(np.abs(fft_vals) + 1e-12)
#
# # center_freq_hz = fc
#
# # === Frequency axis ===
# freqs = np.fft.fftshift(np.fft.fftfreq(len(iq), d=1/fs))
# freqs_mhz = (freqs + fc) / 1e6  #so now, freqs_mhz == [fc-fs/2, ...., fc+fs/2]

# freqs_mhz = (freqs + center_freq_hz) / 1e6  # Convert to MHz
# === Bandpower Summing over special_bw ===
special_bw_half = special_bw / 2 / 1e6  # in MHz
special_range = ((freqs_mhz >= (fc / 1e6 - special_bw_half)) &
                 (freqs_mhz <= (fc / 1e6 + special_bw_half)))

# Compute power spectrum (linear scale)
# power_spectrum = np.abs(fft_vals)**2

# Sum power within special_bw
# power_in_special_bw = np.sum(power_spectrum[special_range])

# Convert to dB
# power_in_special_bw_db = 10 * np.log10(power_in_special_bw + 1e-12)
# print(f"Summed Bandpower over ±{special_bw_half:.3f} MHz: {power_in_special_bw_db:.2f} dB")

# === Filter FFT to BW ===
bw_half = bw / 2 / 1e6
valid = (freqs_mhz >= (fc / 1e6 - bw_half)) & (freqs_mhz <= (fc / 1e6 + bw_half))

# === BW-Summed Power Computation ===
# Split IQ into chunks of N samples
chunk_size = 4096  # e.g., ~68ms @ 60 MSPS; adjust for granularity
powers = []
times = []

for i in range(0, len(iq), chunk_size):
    chunk = iq[i:i+chunk_size]
    if len(chunk) == chunk_size:
        power = np.sum(np.abs(chunk) ** 2) / chunk_size
        powers.append(power)
        times.append(i / fs)  # Time in seconds

powers = np.array(powers)
times = np.array(times)

# Convert to dB
powers_db = 10 * np.log10(powers + 1e-12)

# === Exposure Stacking ===
window_size = 10
if len(powers_db) >= window_size:
    kernel = np.ones(window_size) / window_size
    stacked_db = np.convolve(powers_db, kernel, mode='valid')
    stacked_time = times[window_size - 1:]
else:
    stacked_db = np.array([])
    stacked_time = np.array([])

# === Plotting ===
fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=False)

# FFT
axs[0].plot(freqs_mhz[valid], fft_db[valid], color='green')
axs[0].set_title("FFT from recorded BladeRF samples")
axs[0].set_xlabel("Frequency (MHz)")
axs[0].set_ylabel("Magnitude (dB)")
axs[0].grid(True)

# === Summed Power over Frequency within special_bw ===
# Define how many bins you want in the frequency range (e.g., 100 bins)
num_bins = 100

# Get only the frequencies and powers within the special_bw range
freqs_bw = freqs_mhz[special_range]
# powers_bw = power_spectrum[special_range]

# Create bins
bin_edges = np.linspace(freqs_bw[0], freqs_bw[-1], num_bins + 1)
bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
binned_power = np.zeros(num_bins)

# Bin the power values
for i in range(num_bins):
    mask = (freqs_bw >= bin_edges[i]) & (freqs_bw < bin_edges[i + 1])
    # binned_power[i] = np.sum(powers_bw[mask])

# Convert to dB
binned_power_db = 10 * np.log10(binned_power + 1e-12)

# Plot: Bandpower vs Frequency
axs[1].plot(bin_centers, binned_power_db, color='orange')
axs[1].set_title(f"Bandpower over Frequency (±{special_bw_half:.1f} MHz)")
axs[1].set_xlabel("Frequency (MHz)")
axs[1].set_ylabel("Summed Power (dB)")
axs[1].grid(True)

# Exposure Stacking
axs[2].plot(stacked_time, stacked_db, color='blue')
axs[2].set_title("Exposure Stacked Power (Smoothed)")
axs[2].set_xlabel("Time (s)")
axs[2].set_ylabel("Power (dB)")
axs[2].grid(True)

# fig, axs = plt.subplots(4, 1, figsize=(10, 12), sharex=False)

# Power Spectrum over frequency (linear or dB)
# axs[3].plot(freqs_mhz[valid], 10 * np.log10(power_spectrum[valid] + 1e-12), color='purple')
axs[3].set_title(f"Power Spectrum (Summed ±{special_bw_half:.1f} MHz shown in log scale)")
axs[3].set_xlabel("Frequency (MHz)")
axs[3].set_ylabel("Power (dB)")
axs[3].grid(True)

plt.tight_layout()
plt.show()