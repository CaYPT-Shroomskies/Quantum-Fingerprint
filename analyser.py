import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

length = 1920
height = 1080
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, length)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', '2'))
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
cap.set(cv2.CAP_PROP_EXPOSURE, -4.0)

# Calibration
calibrate = [0.5378783977636364, 251.83884117409121]
wavelengths = np.arange(length)*calibrate[0] + calibrate[1]
laser_wavenumber = 10000000/532
max_wave = 4000
wavenumbers = (10000000/wavelengths) - laser_wavenumber

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], 'k', alpha=0.7)
ax.set_title('Spectrometer Output')
ax.set_ylim(0, 100)
ax.set_xlabel('Wavenumber (1/cm)')
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

y1, y2 = 560, 700
rolling = 1
roll = np.zeros((length, rolling))
roll_i = 0

while True:
    ret, frame = cap.read()
    frame = cv2.line(frame, (0, y1), (frame.shape[1], y1), (0, 255, 0), 1)
    frame = cv2.line(frame, (0, y2), (frame.shape[1], y2), (0, 255, 0), 1)
    frame = cv2.flip(frame, 1)
    cv2.imshow('Image', frame)
    
    spectrum = np.mean(frame[y1:y2], axis=(0, 2))
    roll[:, roll_i%rolling] = spectrum
    
    data = np.average(roll, axis=1)
    processed_data = data - np.min(data)
    line.set_data(wavenumbers, processed_data)
    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)
    
    roll_i += 1
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_spectrum(wavelengths, processed_data)

cap.release()
cv2.destroyAllWindows()
plt.close('all')
