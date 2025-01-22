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
        title='Select SAMPLE and REFERENCE spectra (select 2 files)',
        filetypes=[('CSV files', '*.csv')]
    )
    
    if len(files) != 2:
        print("Please select exactly 2 files (sample and reference)")
        return
    
    sample_path, reference_path = files
    
    try:
        sample_df = pd.read_csv(sample_path)
        reference_df = pd.read_csv(reference_path)
        
        epsilon = 1e-10
        absorbance = -np.log10((sample_df['Intensity'] + epsilon) / (reference_df['Intensity'] + epsilon))
        
        result_df = pd.DataFrame({
            'Wavelength': sample_df['Wavelength'],
            'Sample_Intensity': sample_df['Intensity'],
            'Reference_Intensity': reference_df['Intensity'],
            'Absorbance': absorbance
        })
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        fig.suptitle('Absorbance Analysis', fontsize=14)
        
        ax1.plot(sample_df['Wavelength'], sample_df['Intensity'], 'b-', 
                 label=Path(sample_path).stem)
        ax1.plot(reference_df['Wavelength'], reference_df['Intensity'], 'r-', 
                 label=Path(reference_path).stem)
        ax1.set_xlabel('Wavelength (nm)')
        ax1.set_ylabel('Intensity (arb.)')
        ax1.set_title('Raw Spectra')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        ax2.plot(result_df['Wavelength'], result_df['Absorbance'], 'k-')
        ax2.set_xlabel('Wavelength (nm)')
        ax2.set_ylabel('Absorbance')
        ax2.set_title('Absorbance vs Wavelength')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
                
        print("\nAbsorbance Statistics:")
        print(f"Maximum absorbance: {result_df['Absorbance'].max():.3f}")
        print(f"Minimum absorbance: {result_df['Absorbance'].min():.3f}")
        print(f"Mean absorbance: {result_df['Absorbance'].mean():.3f}")
        
        plt.show()
        
        return result_df
        
    except Exception as e:
        print(f"Error processing files: {e}")
        return None

if __name__ == "__main__":
    result = calculate_absorbance()
