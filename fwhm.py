import time

import matplotlib.pyplot as plt
import numpy as np
import serial
from scipy.optimize import curve_fit

# Constants
length = 3648  # Number of pixels
bytes_expected = length * 2  # 2 bytes per pixel (12-bit data in 16-bit container)
averages = 1
baudrate: int = 921600
timeout: float = 1

"""
# Raman
laser_wavenumber = 10000000 / 532
max_wave = 4000
calibrate = np.array([[0.5378783977636364, 251.83884117409121],[0,0]]) # Wavelength pixel pairs
calibrate = np.polyfit(calibrate.reshape(1, -1),1) # obtain calibrate linear fit

wavelengths = np.arange(length) * calibrate[0] + calibrate[1]
raman_wavenumbers = (10000000 / wavelengths) - laser_wavenumber
"""


def gaussian(x, a, mu, sigma):
    return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2))


def find_fwhm(x, y):
    try:
        # Initial parameter estimates
        a0 = np.max(y)
        mu0 = x[np.argmax(y)]
        sigma0 = max(8, (x[-1] - x[0]) / 10)  # Prevent too small initial sigma

        # Set bounds to prevent unrealistic fits
        bounds = (
            [a0 * 0.5, x[0], 2],  # Lower bounds
            [a0 * 1.5, x[-1], (x[-1] - x[0])],  # Upper bounds
        )

        # Fit Gaussian with bounds
        popt, _ = curve_fit(gaussian, x, y, p0=[a0, mu0, sigma0], bounds=bounds)

        # Calculate FWHM = 2.355 * sigma
        fwhm = 2.355 * abs(popt[2])

        return fwhm if fwhm > 2 else None, popt

    except Exception:
        return None, None


def read_sensor_data_12bpp(ser):
    """
    Reads 12-bit data (packed in 16-bit words) from the TCD1304 sensor.
    """
    ser.reset_input_buffer()

    packet = bytearray()
    packet.append(0xA1)  # read TCD1304 module (12 bpp, 7296 bytes)
    # packet.append(0xA2)  # read TCD1304 module (8 bpp, 3648 bytes)
    packet.append(0xB8)  # set integration time (Max D7, min B0)
    ser.write(packet)  # Request data

    buffer = bytearray()
    start_time = time.time()
    timeout_duration = 5

    while len(buffer) < bytes_expected:
        if time.time() - start_time > timeout_duration:
            print(
                f"Timeout reached. Received {len(buffer)} bytes out of {bytes_expected}."
            )
            break
        if ser.in_waiting > 0:
            buffer += ser.read(ser.in_waiting)

    if len(buffer) < bytes_expected:
        print("Warning: Incomplete data received.")
        return np.zeros(length, dtype=np.uint16)

    # Interpret as 16-bit words, assuming little-endian format
    raw_data = np.frombuffer(buffer[:bytes_expected], dtype=np.uint16)

    # Mask to 12-bit data
    pixel_data = raw_data & 0x0FFF

    return pixel_data


def update_plot_12bpp(sensor_data, line, ax):
    """
    Updates the live plot with new 12-bit intensity values from the sensor.
    """
    if len(sensor_data) != length:
        print(
            f"Warning: Expected {length} pixels, got {len(sensor_data)}. Plotting available data."
        )

    sensor_data = 4095 - sensor_data
    sensor_data[0:4] = sensor_data[5]

    # Calculate and print FWHM
    pixels = np.arange(length)
    maxima_index = np.argmax(sensor_data)
    start_index = max(0, int(maxima_index - 10))
    end_index = min(length, int(maxima_index + 10))

    fwhm, popt = find_fwhm(
        pixels[start_index:end_index], sensor_data[start_index:end_index]
    )

    if fwhm is not None:
        print(f"FWHM: {fwhm:.2f} pixels")

        # Plot data and fit
        line.set_data(pixels, sensor_data)

        # Plot gaussian fit
        x_fit = pixels[start_index:end_index]
        y_fit = gaussian(x_fit, *popt)
        if not hasattr(ax, "fit_line"):
            (ax.fit_line,) = ax.plot(x_fit, y_fit, "r--", label="Gaussian fit")
        else:
            ax.fit_line.set_data(x_fit, y_fit)

        # Zoom to fit region
        ax.set_xlim(start_index - 10, end_index + 10)
        ax.set_ylim(
            min(sensor_data[start_index:end_index]) * 0.9,
            max(sensor_data[start_index:end_index]) * 1.1,
        )
    else:
        line.set_ydata(sensor_data)
        ax.set_xlim(0, length)
        ax.set_ylim(0, 4095)

    plt.draw()
    plt.pause(0.01)


if __name__ == "__main__":
    port_name = "/dev/ttyCH341USB0"
    ser = serial.Serial(port_name, baudrate, timeout=timeout)
    print(f"Connected to {port_name}")

    # Set up the live plot
    plt.ion()
    fig, ax = plt.subplots(figsize=(10, 6))
    pixels = np.arange(length)
    (line,) = ax.plot(pixels, np.zeros(length))
    ax.set_title("TCD1304 Spectrum (12-bit mode) - Live Update")
    ax.set_xlabel("Pixel")
    ax.set_ylabel("Intensity (12-bit)")
    ax.set_ylim(0, 4095)
    ax.grid(True)
    plt.show()

    while True:
        try:
            data = np.zeros(length, dtype=np.float64)
            for i in range(averages):
                data += read_sensor_data_12bpp(ser)
                time.sleep(0.2)
            data /= averages
            update_plot_12bpp(data, line, ax)
        except serial.SerialException as e:
            print(f"Serial port error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
    ser.close()
