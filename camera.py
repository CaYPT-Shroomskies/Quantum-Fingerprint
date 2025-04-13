import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
import pandas as pd
from scipy.ndimage import minimum_filter1d  # Added import
import os

print("\n\033[1m[RAMAN SPECTROMETER TERMINAL]\033[0m")

print("Loading images from folder...")
t0 = time.perf_counter()

# Read all JPEG images from folder
folder_path = '/home/chrysanthemum/ethanol/'  # Specify your input folder path
image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg'))]

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))

for image_file in image_files:
    image_path = os.path.join(folder_path, image_file)
    frame = cv2.imread(image_path)
    length = frame.shape[1]
    height = frame.shape[0]

    print(f"Processing image {image_file}:", int(1e3 * (time.perf_counter()-t0)), "ms\n")

    # Calibration
    calibrate = [0.5378783977636364, 251.83884117409121]
    wavelengths = (1920/length)*np.arange(length)*calibrate[0] + calibrate[1]
    laser_wavenumber = 10000000/532
    max_wave = 4000
    wavenumbers = (10000000/wavelengths) - laser_wavenumber

    y1, y2 = int(0.52*height), int(0.58*height)
    rolling = 1
    roll = np.zeros((length, rolling))
    roll_i = 0

    # Load dark frame
    dark_frame = pd.read_csv('dark_frame.csv')
    dark_intensities = dark_frame['Intensity'].values

    line, = ax.plot([], [], label=image_file)
    nocorrect, = ax.plot([], [], alpha=0.5)

    ax.set_title('Spectrometer Output')
    ax.set_ylim(0, 255)
    ax.set_xlim(np.min(wavelengths),np.max(wavelengths))
    ax.set_ylabel('Intensity (arb.)')
    plt.tight_layout()

    def save_spectrum(wavelengths, intensities):
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'spectrum_{timestamp}.csv'

        # Create DataFrame with wavelength and intensity data
        df = pd.DataFrame({
            'Wavelength': wavelengths,
            'Wavenumber': wavenumbers,
            'Intensity': intensities
        })

        # Save to CSV
        df.to_csv(filename, index=False)
        print(f'Spectrum saved to {filename}')
        return filename

    def removeFluor(intensities, window_size=10):
        baseline = minimum_filter1d(intensities, window_size, mode='reflect')
        corrected = intensities - baseline
        return np.clip(corrected, 0, None)  # Ensure non-negative values

    # Add lines to image
    frame = cv2.line(frame, (0, y1), (frame.shape[1], y1), (0, 255, 0), 1)
    frame = cv2.line(frame, (0, y2), (frame.shape[1], y2), (0, 255, 0), 1)
    frame = cv2.flip(frame, 1)

    # Resize frame for display
    scale_percent = 25 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    display_frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    cv2.imshow('Image', display_frame)

    frame = np.array(frame)
    spectrum = frame[y1:y2, :, 1].mean(axis=0)  # Only use green channel (index 1)
    roll[:, roll_i%rolling] = spectrum

    data = np.average(roll, axis=1)

    data_noremove = data - np.min(data)
    data = removeFluor(data)

    line.set_data(wavelengths, data)
    nocorrect.set_data(wavelengths,data_noremove)

    plt.legend()
    plt.draw()
    plt.pause(0.1)

while True:
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = save_spectrum(wavelengths, data_noremove)

cv2.destroyAllWindows()
plt.close('all')
