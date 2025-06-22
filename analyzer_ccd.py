import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import serial
from tqdm import tqdm

# Constants
length = 3648  # Number of pixels
bytes_expected = length * 2  # 2 bytes per pixel (12-bit data in 16-bit container)
averages = 1
baudrate: int = 921600
timeout: float = 1
dark_frame_file = None # "dark_frame.csv"
save = False
save_dark_Frame = False

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


def read_sensor_data_12bpp(ser):
    """
    Reads 12-bit data (packed in 16-bit words) from the TCD1304 sensor.
    """
    ser.reset_input_buffer()

    packet = bytearray()
    packet.append(0xA1)  # read TCD1304 module (12 bpp, 7296 bytes)
    # packet.append(0xA2)  # read TCD1304 module (8 bpp, 3648 bytes)
    packet.append(0xBA)  # set integration time (Max D7, min B0)
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


def convert_and_plot_12bpp(sensor_data, save_csv=False, dark_frame_file=None):
    """
    Plots the 12-bit intensity values from the sensor.
    """
    if len(sensor_data) != length:
        print(
            f"Warning: Expected {length} pixels, got {len(sensor_data)}. Plotting available data."
        )

    pixels = np.arange(len(sensor_data))
    sensor_data = 4095 - sensor_data
    sensor_data[0:4] = sensor_data[5]

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

    plt.figure(figsize=(10, 6))
    plt.plot(wavelengths, sensor_data)
    plt.title("TCD1304 Spectrum (12-bit mode)")
    plt.xlabel("Wavenumber ($cm^{-1}$)")
    plt.ylabel("Intensity (12-bit)")
    plt.ylim(-100, 2500)
    #tplt.xlim(500,3000)
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    port_name = "/dev/ttyCH341USB0"
    ser = serial.Serial(port_name, baudrate, timeout=timeout)
    print(f"Connected to {port_name}")


    try:
        data = np.zeros(length, dtype=np.float64)
        for i in tqdm(range(averages)):
            data += read_sensor_data_12bpp(ser)
            time.sleep(1)
        data /= averages
        convert_and_plot_12bpp(data, save_csv=save, dark_frame_file=dark_frame_file)
    except serial.SerialException as e:
        print(f"Serial port error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    ser.close()
