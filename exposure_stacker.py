def perform_exposure_stacking(load_dir, output_path):
    """
    Loads multiple integrated spectrum files from `load_dir`,
    performs exposure stacking (linear average), saves the result to `output_path`,
    and plots the final stacked spectrum in dB.
    """
    import numpy as np
    import os
    import glob
    import matplotlib.pyplot as plt

    # 1. Locate all spectrum files
    file_list = sorted(glob.glob(os.path.join(load_dir, "spectrum_*.npy")))
    if not file_list:
        print("No spectrum files found.")
        return

    # 2. Load all spectra into array
    spectra = []
    for file in file_list:
        data = np.load(file)
        spectra.append(data)

    spectra = np.stack(spectra, axis=0)  # shape: (N_files, N_freq)

    # 3. Average in linear scale
    stacked = np.mean(spectra, axis=0)  # shape: (N_freq,)

    # 4. Convert to dB for visualization
    stacked_db = 10 * np.log10(stacked + 1e-12)

    # 5. Frequency axis (reconstructed from known values)
    N_freq = stacked.shape[0]
    half_bw = BW_FOR_BW_SUMMING / 2
    freqs = np.linspace(CONCENTRATION_FREQUENCY - half_bw,
                        CONCENTRATION_FREQUENCY + half_bw,
                        N_freq)

    # 6. Save both linear and dB-stacked spectra
    np.save(output_path, stacked)  # linear
    np.save(output_path.replace(".npy", "_db.npy"), stacked_db)  # dB version

    # 7. Plot
    plt.figure(figsize=(10, 5))
    plt.plot(freqs, stacked_db, label='Exposure Stacked Spectrum (dB)')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.title('Exposure Stacked Spectrum')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Exposure stacked spectrum saved to {output_path} and plotted.")

BW_FOR_BW_SUMMING = 2e6  # for example, 2 MHz
CONCENTRATION_FREQUENCY = 2.4e9
perform_exposure_stacking(load_dir="bw_spectra", output_path="stacked_exposure_spectra/stacked_exposure.npy")

