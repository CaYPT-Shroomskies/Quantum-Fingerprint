import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import time
from Main import main as similarity
import pandas as pd
from scipy.ndimage import minimum_filter1d  # Added import

print("\n\033[1m[RAMAN SPECTROMETER TERMINAL]\033[0m")

print("Loading camera...")
t0 = time.perf_counter()

length = 1920
height = 1080
cap = cv2.VideoCapture(0,cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, length)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', '2'))
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)
# cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)
# cap.set(cv2.CAP_PROP_EXPOSURE, -4.0)

print("Loaded Camera:",int(1e3* (time.perf_counter()-t0) ),"ms\n")

# Calibration
calibrate = [0.5378783977636364, 251.83884117409121]
wavelengths = (1920/length)*np.arange(length)*calibrate[0] + calibrate[1]
laser_wavenumber = 10000000/532
max_wave = 4000
wavenumbers = (10000000/wavelengths) - laser_wavenumber

y1, y2 = int(0.52*height), int(0.65*height)
rolling = 3
roll = np.zeros((length, rolling))
roll_i = 0


plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], 'k')
nocorrect, = ax.plot([], [], 'k', alpha=0.5)

ax.set_title('Spectrometer Output')
ax.set_ylim(0, 100)
ax.set_xlim(np.min(wavenumbers),np.max(wavenumbers))
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
    return filename

'''
def removeFluor(frame):
    for i in np.linspace(0,frame.shape[0]-200,int(frame.shape[0]/100)):
        i = int(i)
        batch = frame[i:(i+100)]
        batch -= np.min(batch)
        frame[i:(i+100)] = batch
    return frame
'''
def removeFluor(intensities, window_size=10):
    baseline = minimum_filter1d(intensities, window_size, mode='reflect')
    corrected = intensities - baseline
    return np.clip(corrected, 0, None)  # Ensure non-negative values
    

while True:
    _, frame = cap.read()
    frame = cv2.line(frame, (0, y1), (frame.shape[1], y1), (0, 255, 0), 1)
    frame = cv2.line(frame, (0, y2), (frame.shape[1], y2), (0, 255, 0), 1)
    frame = cv2.flip(frame, 1)
    cv2.imshow('Image', frame)

    frame = np.array(frame)
    spectrum = np.mean(frame[y1:y2], axis=(0, 2))
    roll[:, roll_i%rolling] = spectrum
    
    data = np.average(roll, axis=1)

    data_noremove = data - np.min(data)
    data = removeFluor(data)
    
    line.set_data(wavenumbers, data)
    nocorrect.set_data(wavenumbers,data_noremove)

    plt.pause(0.1)
    
    roll_i += 1
    
    key = cv2.waitKey(10) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        filename = save_spectrum(wavelengths, data)
        similarity(filename)
        

cap.release()
cv2.destroyAllWindows()
plt.close('all')
