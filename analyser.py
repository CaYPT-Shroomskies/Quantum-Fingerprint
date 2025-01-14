import cv2
import numpy as np
import matplotlib.pyplot as plt

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
calibrate = [0.5378783977636364,251.83884117409121]
wavelengths = np.arange(length)*calibrate[0] + calibrate[1]

laser_wavenumber = 10000000/532
max_wave = 4000

wavenumbers = (10000000/wavelengths) - laser_wavenumber

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], 'k',alpha=0.7)
ax.set_title('Spectrometer Output')
ax.set_ylim(0,100)
#ax.set_xlim(calibrate[1],calibrate[1]+calibrate[0]*length)
#ax.set_xlim(0,4000)
ax.set_xlabel('Wavenumber (1/cm)')
ax.set_ylabel('Intensity (arb.)')
plt.tight_layout()

y1, y2 = 560, 700

rolling = 10
roll = np.zeros((length,rolling))
roll_i = 0


while True:
    ret, frame = cap.read()

    frame = cv2.line(frame, (0, y1), (frame.shape[1], y1), (0, 255, 0), 1)
    frame = cv2.line(frame, (0, y2), (frame.shape[1], y2), (0, 255, 0), 1)
    frame = cv2.flip(frame,1)
    cv2.imshow('Image', frame)
    
    
    spectrum = np.mean(frame[y1:y2], axis=(0, 2))
    roll[:,roll_i%rolling] = spectrum
    
    data = np.average(roll,axis=1)
    line.set_data(wavenumbers, data-np.min(data))

    ax.relim()
    ax.autoscale_view()
    plt.pause(0.01)
    
    roll_i += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close('all')
