import numpy as np
import matplotlib.pyplot as plt

wavelengths = np.arange(500,700)

laser_wavenumber = 10000000/532
max_wave = 4000

wavenumbers = (10000000/wavelengths) - laser_wavenumber

plt.plot(wavelengths,wavenumbers)
plt.grid()
plt.show()
