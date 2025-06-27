import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
from scipy.optimize import curve_fit

# Constants
length = 3694  # Number of pixels (updated per documentation - 7388 bytes / 2)
bytes_expected = 7388  # Total bytes expected per documentation
SH = 10  # integration time in microsecond

ICG = 10000  # ICG in microseconds
balanced = False
averages = 15
baudrate: int = 115200
dark_frame_file = None  # "dark_frame.csv"
save = False
save_dark_Frame = False

txfull = np.uint8([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
SHperiod = np.uint32(SH * 2)
ICGperiod = np.uint32(ICG * 2)

if ICGperiod % SHperiod:
    print("TIMING VIOLATION: NOT DIVISIBLE")
    exit()
elif SHperiod < 20:
    print("TIMING VIOLATION: SH PERIOD TOO SMALL")
    exit()
elif ICGperiod < 14776:
    print("TIMING VIOLATION: ICG PERIOD TOO SMALL")
    exit()

AVGn = np.uint8([0, averages])


# Raman
laser_wavenumber = 10000000 / 632.80
max_wave = 4000
points = np.array(
    [
        [19, 587.6],
        [127, 593.4],
        [262, 599.7],
        [501, 611.6],
        [920, 631.6],
        [962.5, 632.80],
        [1358, 650.8],
        [1638, 662.6],
        [2276, 687.7],
        [2439, 693.7],
    ]
)  # Wavelength pixel pairs
points += 9.25 - points[5, 0]
calibrate = np.polyfit(points[:, 0], points[:, 1], 1)  # obtain calibrate linear fit
calibrate[1] -= calibrate[0] * 9.25 + calibrate[1] - 632.8
wavelengths = np.arange(length) * calibrate[0] + calibrate[1]
raman_wavenumbers = laser_wavenumber - (10000000 / wavelengths)


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

    # Create packet with ER key and timing parameters
    txfull[0] = 69
    txfull[1] = 82
    # split 32-bit integers to be sent into 8-bit data
    txfull[2] = (SHperiod >> 24) & 0xFF
    txfull[3] = (SHperiod >> 16) & 0xFF
    txfull[4] = (SHperiod >> 8) & 0xFF
    txfull[5] = SHperiod & 0xFF
    txfull[6] = (ICGperiod >> 24) & 0xFF
    txfull[7] = (ICGperiod >> 16) & 0xFF
    txfull[8] = (ICGperiod >> 8) & 0xFF
    txfull[9] = ICGperiod & 0xFF
    # averages to perfom
    txfull[10] = AVGn[0]
    txfull[11] = AVGn[1]

    ser.write(txfull)  # Send command packet

    buffer = ser.read(7388)
    rxData16 = np.zeros(3694, np.uint16)
    for rxi in range(3694):
        rxData16[rxi] = (buffer[2 * rxi + 1] << 8) + buffer[2 * rxi]

    if len(buffer) < bytes_expected:
        print("Warning: Incomplete data received.")

    return rxData16


def convert_and_plot_12bpp(sensor_data, save_csv=False, dark_frame_file=None):
    pixels = np.arange(len(sensor_data))
    sensor_data = (sensor_data[10] + sensor_data[11]) / 2 - sensor_data
    sensor_data = np.flip(sensor_data)
    if balanced:
        offset = (
            sensor_data[18]
            + sensor_data[20]
            + sensor_data[22]
            + sensor_data[24]
            - sensor_data[19]
            - sensor_data[21]
            - sensor_data[23]
            - sensor_data[24]
        ) / 4
        for i in range(1847):
            sensor_data[2 * i] = sensor_data[2 * i] - offset

    # Subtract dark frame if provided
    if dark_frame_file and not save_dark_Frame:
        try:
            dark_frame = pd.read_csv(dark_frame_file)["intensity"].values
            sensor_data = sensor_data - dark_frame
        except Exception as e:
            print(f"Error loading dark frame: {e}")

    # Save to CSV if requested
    if save_csv:
        df = pd.DataFrame({"intensity": sensor_data})
        df.to_csv(f"spectrum_{time.strftime('%Y%m%d_%H%M%S')}.csv", index=False)
    if save_dark_Frame:
        df = pd.DataFrame({"intensity": sensor_data})
        df.to_csv("dark_frame.csv", index=False)

    plt.clf()
    plt.plot(raman_wavenumbers, sensor_data)
    plt.title("TCD1304 Spectrum (12-bit mode)")
    plt.xlabel("Wavenumber ($cm^{-1}$)")
    plt.ylabel("Intensity (12-bit)")
    plt.ylim(0, np.max(sensor_data))
    #plt.xlim(np.argmax(sensor_data) - 30, np.argmax(sensor_data) + 30)
    plt.grid(True)

    fwhm, popt = find_fwhm(
        pixels[np.argmax(sensor_data) - 8 : np.argmax(sensor_data) + 8],
        sensor_data[np.argmax(sensor_data) - 8 : np.argmax(sensor_data) + 8],
    )
    if fwhm:
        print(int(fwhm * 10) / 10)
    plt.draw()
    plt.pause(0.001)


if __name__ == "__main__":
    port_name = "/dev/ttyACM0"
    ser = serial.Serial(port_name, baudrate)
    print(f"Connected to {port_name}")
    plt.ion()
    plt.figure(figsize=(10, 6))
    while True:
        try:
            convert_and_plot_12bpp(
                read_sensor_data_12bpp(ser),
                save_csv=save,
                dark_frame_file=dark_frame_file,
            )
        except serial.SerialException as e:
            print(f"Serial port error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
    ser.close()
