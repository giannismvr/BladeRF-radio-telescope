import numpy as np
import pyqtgraph as pg
from PyQt5 import QtCore, QtWidgets
import time
import os

# Set your binary file path
file_path = "/Users/giannis/PycharmProjects/final_radiotelescope/my_stuff/my_rx_samples.bin"

# BladeRF config constants
N_SAMPLES = 4096
fs = 4e6  # 4 MHz sample rate
rx_freq = 1.42e9  # 1.42 GHz
BW = 2e6

# GUI setup
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(title="Live Signal and FFT Plot")
win.show()

time_plot = win.addPlot(title="Time Domain I Samples")
time_curve = time_plot.plot(pen='y')

fft_plot = win.addPlot(title="FFT of Signal")
fft_curve = fft_plot.plot(pen='g')
fft_plot.setLabel('bottom', 'Frequency (MHz)')
fft_plot.setLabel('left', 'Magnitude (dB)')

# FFT function
def compute_fft(i_data):
    window = np.hanning(len(i_data))
    windowed = i_data * window
    fft_data = np.fft.fftshift(np.fft.fft(windowed))
    fft_freqs = np.fft.fftshift(np.fft.fftfreq(len(i_data), d=1/fs))
    fft_magnitude = 20 * np.log10(np.abs(fft_data) + 1e-10)
    fft_freqs += rx_freq
    fft_freqs /= 1e6
    # Limit range
    low = (rx_freq - BW/2)/1e6
    high = (rx_freq + BW/2)/1e6
    valid = (fft_freqs >= low) & (fft_freqs <= high)
    return fft_freqs[valid], fft_magnitude[valid]

# File read offset tracker
last_position = 0

# Update function
def update():
    global last_position
    try:
        with open(file_path, "rb") as f:
            f.seek(last_position)
            new_data = f.read(N_SAMPLES * 2)
            last_position += len(new_data)

        if len(new_data) >= 4:
            iq = np.frombuffer(new_data, dtype=np.int16)
            i = iq[::2]  # Use I samples

            time_curve.setData(i[-N_SAMPLES:])

            fft_freqs, fft_magnitude = compute_fft(i[-N_SAMPLES:])
            fft_curve.setData(fft_freqs, fft_magnitude)

    except Exception as e:
        print("Update error:", e)

# Timer for real-time updates
timer = QtCore.QTimer()
timer.timeout.connect(update)
timer.start(100)

# Run app
if __name__ == '__main__':
    QtWidgets.QApplication.instance().exec_()
