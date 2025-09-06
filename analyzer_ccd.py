import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
from scipy.ndimage import gaussian_filter

# Constants
length = 3694  # Number of pixels (updated per documentation - 7388 bytes / 2)

SH = 20  # integration time in seconds

ICG = 20  # ICG in seconds

SH *= 1e6
ICG *= 1e6
balanced = False
averages = 1
gaussian_mag = 6
baudrate: int = 115200
dark_frame_file = None #"240_dark_large.csv" # "120_dark_large.csv" # "240_dark_large.csv"  # "240_dark.csv"
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
"""
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
"""
points = np.array(
    [
        [388.6, 667.43],
        [524, 673.27],
        [714, 682.46],
        [1084, 696.83],
        [3164, 773.83],
        [3226, 776.23],
        [3323, 779.86],
    ]
)

# new optical system slope
# slope = np.array([[292.2, 542.2], [379.2, 546.5], [1550, 599.7], [1820, 611.6]])
calibrate = np.polyfit(points[:, 0], points[:, 1], 1)  # obtain calibrate linear fit
wavelengths = np.arange(length) * calibrate[0] + calibrate[1]
raman_wavenumbers = laser_wavenumber - (10000000 / wavelengths)


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
    txfull[10] = AVGn[0]
    txfull[11] = AVGn[1]

    ser.write(txfull)  # Send command packet

    buffer = ser.read(7388)
    rxData16 = np.zeros(3694, np.uint16)
    for rxi in range(3694):
        rxData16[rxi] = (buffer[2 * rxi + 1] << 8) + buffer[2 * rxi]

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
        print("Saved", f"spectrum_{time.strftime('%Y%m%d_%H%M%S')}.csv")
    if save_dark_Frame:
        print("Saved dark_frame.csv")
        df = pd.DataFrame({"intensity": sensor_data})
        df.to_csv("dark_frame.csv", index=False)

    plt.figure(figsize=(10, 6))
    if gaussian_mag != 0:
        sensor_data = gaussian_filter(sensor_data, gaussian_mag)
    plt.plot(raman_wavenumbers, sensor_data)
    plt.xlabel("Wavenumber ($cm^{-1}$)")
    # plt.xlabel("Wavelength")
    # plt.xlabel("Pixel")
    plt.ylabel("Intensity (12-bit)")
    plt.ylim(-500, 2500)
    plt.grid(True)

    plt.show()


if __name__ == "__main__":
    port_name = "/dev/ttyACM0"
    while True:
        try:
            ser = serial.Serial(port_name, baudrate)
            print(f"\n\nConnected to {port_name}")

            print("\nRead Sensor w/ T-INT TIME", averages * (ICG / 1e6))
            print("Start time:", time.strftime("%H:%M"))
            convert_and_plot_12bpp(
                read_sensor_data_12bpp(ser),
                save_csv=save,
                dark_frame_file=dark_frame_file,
            )
            time.sleep(0.1)
            ser.close()
        except serial.SerialException as e:
            print(f"Serial port error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
