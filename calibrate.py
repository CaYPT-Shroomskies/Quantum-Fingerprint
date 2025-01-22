import numpy as np
import matplotlib.pyplot as plt

# Your data points
points = np.array([[343, 436.6],
                  [540, 542.2],
                  [549, 546.5],
                  [668, 611.6]])

# Separate x and y values
pixels = points[:, 0]
wavelengths = points[:, 1]

# Calculate linear fit
slope, intercept = np.polyfit(pixels, wavelengths, 1)

print(slope,intercept)
def pixel_to_wavelength(pixel):
    return slope * pixel + intercept

pixel_range = np.linspace(pixels.min(), pixels.max(), 100)
wavelength_fit = pixel_to_wavelength(pixel_range)

plt.figure(figsize=(10, 6))
plt.plot(pixel_range, wavelength_fit, 'b-', label=f'Linear fit (y = {slope:.3f}x + {intercept:.1f})')
plt.plot(pixels, wavelengths, 'ro', label='Measured points',alpha = 0.5)
plt.xlabel('Pixel Position')
plt.ylabel('Wavelength (nm)')
plt.title('Spectrometer Calibration')
plt.grid(True)
plt.legend()
plt.show()
