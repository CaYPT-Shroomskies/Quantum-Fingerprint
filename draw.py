import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analyzer_ccd import gaussian_mag, raman_wavenumbers
from scipy.ndimage import gaussian_filter


def plot_spectra(file_paths=None, marching_window=0):
    if file_paths is None:
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        file_paths = filedialog.askopenfilenames(
            title="Select Spectrum CSV files (multiple files possible)",
            filetypes=[("CSV files", "*.csv")],
        )

        if not file_paths:  # If user cancels
            print("No files selected")
            return

    if isinstance(file_paths, (str, Path)):
        file_paths = [file_paths]

    fig, ax = plt.subplots()

    for file_path in file_paths:
        try:
            df = pd.read_csv(file_path)
            filename = Path(file_path).stem
            sensor_data = df["intensity"]
            if gaussian_mag != 0:
                sensor_data = gaussian_filter(sensor_data, gaussian_mag)

            # Apply marching average subtraction if window size provided
            if marching_window > 0:
                window = np.ones(marching_window) / marching_window
                background = np.convolve(sensor_data, window, mode="same")
                sensor_data = sensor_data - background

            sensor_data -= np.min(sensor_data)
            ax.plot(
                raman_wavenumbers,
                sensor_data,
                "-",
                linewidth=1.5,
                label=filename,
            )

        except Exception as e:
            print(f"Error reading or plotting file {file_path}: {e}")

    ax.set_xlabel("Wavenumber ($cm^{-1}$)")
    ax.set_ylabel("Intensity (arb.)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    plt.show()


def normalize_spectrum(intensity):
    """Normalize spectrum to range [0,1]"""
    return (intensity - intensity.min()) / (intensity.max() - intensity.min())


if __name__ == "__main__":
    plot_spectra(marching_window=0)
