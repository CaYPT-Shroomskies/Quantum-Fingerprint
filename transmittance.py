import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path

from analyzer_ccd import wavelengths

def calculate_absorbance():
    root = tk.Tk()
    root.withdraw()

    files = filedialog.askopenfilenames(
        title="Select SAMPLE and REFERENCE spectra (select 2 files)",
        filetypes=[("CSV files", "*.csv")],
    )

    if len(files) != 2:
        print("Please select exactly 2 files (sample and reference)")
        return

    sample_path, reference_path = files

    try:
        sample_df = pd.read_csv(sample_path)
        reference_df = pd.read_csv(reference_path)
        if np.average(sample_df["intensity"]) > np.average(reference_df["intensity"]):
            print("Invert sample and reference")
            reference_df, sample_df = sample_df, reference_df

        epsilon = 1e-10  # Prevent division by zero
        transmittance = sample_df["intensity"] / (reference_df["intensity"] + epsilon)

        result_df = pd.DataFrame(
            {
                "Sample_Intensity": sample_df["intensity"],
                "Reference_Intensity": reference_df["intensity"],
                "Transmittance": transmittance,
            }
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Spectral Analysis", fontsize=14)

        ax1.plot(
            wavelengths,
            sample_df["intensity"],
            "b-",
            label=Path(sample_path).stem,
        )
        ax1.plot(
            wavelengths,
            reference_df["intensity"],
            "r-",
            label=Path(reference_path).stem,
        )
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Intensity (arb.)")
        ax1.set_title("Raw Spectra")
        ax1.set_ylim(0,2500)
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(wavelengths, result_df["Transmittance"], "g-")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Transmittance (I/I₀)")
        ax2.set_ylim(0, 1.1)
        ax2.set_title("Transmittance")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        plt.show()

        return result_df

    except Exception as e:
        print(f"Error processing files: {e}")
        return None


if __name__ == "__main__":
    result = calculate_absorbance()
