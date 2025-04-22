import numpy as np
import matplotlib.pyplot as plt

# Read the raw binary file (adjust path accordingly)
file_path = 'my_rx_samples.bin'

# Read all data as int16 (little-endian by default)
raw = np.fromfile(file_path, dtype=np.int16)

# I and Q are interleaved: [I0, Q0, I1, Q1, I2, Q2, ...]
# So we reshape it into pairs
iq_pairs = raw.reshape(-1, 2)

# Create complex IQ samples
iq_data = iq_pairs[:, 0] + 1j * iq_pairs[:, 1]

# Optionally normalize if needed
iq_data = iq_data / 32768.0

# Plot a small portion
plt.figure(figsize=(10, 4))
plt.plot(np.real(iq_data[:1000]), label='I')
plt.plot(np.imag(iq_data[:1000]), label='Q')
plt.legend()
plt.title('First 1000 IQ Samples')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()
