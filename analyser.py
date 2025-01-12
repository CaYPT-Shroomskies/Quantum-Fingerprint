import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', '2'))
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, -1)
cap.set(cv2.CAP_PROP_EXPOSURE, -1)

plt.ion()
fig, ax = plt.subplots(figsize=(8, 6))
line, = ax.plot([], [], 'k', label='Spectrum')
ax.set_title('Spectrometer Output')
ax.set_ylim(0,25)
ax.set_xlim(0,1920)
ax.set_xlabel('Pixel Position')
ax.set_ylabel('Intensity')
ax.legend()
plt.tight_layout()

y1, y2 = 550, 700


while True:
    ret, frame = cap.read()

    frame = cv2.line(frame, (0, y1), (frame.shape[1], y1), (0, 255, 0), 1)
    frame = cv2.line(frame, (0, y2), (frame.shape[1], y2), (0, 255, 0), 1)
    
    cv2.imshow('Image', frame)
    
    
    spectrum = np.mean(frame[y1:y2], axis=(0, 2))
    line.set_data(range(len(spectrum)), spectrum)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
plt.close('all')
