import numpy as np
import matplotlib.pyplot as plt
# from bladerf import _bladerf
from bladerf import _bladerf

from bladerf import _bladerf

print(dir(_bladerf))  # This will list all attributes and classes in _bladerf
dev = _bladerf.BladeRF()  # or the correct class for your version
rx_channel = _bladerf.RX()  # Use RX or another relevant class based on your version

#
# # <font color='cyan'>Parameters</font>
# center_freq = 1.4204e9       # Hydrogen line frequency in Hz
# sample_rate = 2.5e6          # 2.5 MSPS
# num_samples = 8192           # Number of samples to read
#
# # <font color='cyan'>Open device</font>
# dev = _bladerf.SyncRx()
# dev.sample_rate = int(sample_rate)
# dev.frequency = int(center_freq)
# dev.bandwidth = int(sample_rate)
# dev.gain = 30  # Adjust as needed
#
# print("Receiving samples...")
#
# # <font color='cyan'>Read samples</font>
# samples = dev.read(num_samples, timeout=3000)
# samples = np.array(samples, dtype=np.complex64)
#
# # <font color='cyan'>Close device</font>
# dev.close()
#
# # <font color='cyan'>Compute FFT and plot</font>
# spectrum = np.fft.fftshift(np.fft.fft(samples))
# power = 20 * np.log10(np.abs(spectrum))
# freq_axis = np.linspace(-sample_rate/2, sample_rate/2, num_samples) + center_freq
#
# plt.figure(figsize=(10, 6))
# plt.plot(freq_axis / 1e6, power)
# plt.title("Hydrogen Line Observation (Example)")
# plt.xlabel("Frequency (MHz)")
# plt.ylabel("Power (dB)")
# plt.grid(True)
# plt.tight_layout()
# plt.show()
