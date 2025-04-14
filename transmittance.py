import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog
from pathlib import Path


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
        if np.average(sample_df["Intensity"]) > np.average(reference_df["Intensity"]):
            print("Invert sample and reference")
            reference_df, sample_df = sample_df, reference_df

        epsilon = 1e-10  # Prevent division by zero
        transmittance = sample_df["Intensity"] / (reference_df["Intensity"] + epsilon)

        result_df = pd.DataFrame(
            {
                "Wavelength": sample_df["Wavelength"],
                "Sample_Intensity": sample_df["Intensity"],
                "Reference_Intensity": reference_df["Intensity"],
                "Transmittance": transmittance,
            }
        )
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        fig.suptitle("Spectral Analysis", fontsize=14)

        ax1.plot(
            sample_df["Wavelength"],
            sample_df["Intensity"],
            "b-",
            label=Path(sample_path).stem,
        )
        ax1.plot(
            reference_df["Wavelength"],
            reference_df["Intensity"],
            "r-",
            label=Path(reference_path).stem,
        )
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Intensity (arb.)")
        ax1.set_title("Raw Spectra")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        ax2.plot(result_df["Wavelength"], result_df["Transmittance"], "g-")
        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Transmittance (I/I₀)")
        ax2.set_ylim(0, 1)
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
