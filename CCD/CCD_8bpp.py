import time

import matplotlib.pyplot as plt
import numpy as np
import serial
from tqdm import tqdm

# Variables
length = 3648
averages = 1
baudrate: int = 921600
timeout: float = 1


def read_sensor_data(ser):
    """
    Reads 8-bit data from a USB-connected TCD1304 sensor and returns it as a numpy array.
    :param port: The serial port (e.g., 'COM3' on Windows or '/dev/ttyUSB0' on Linux).
    :param baudrate: Communication speed (default: 921600).
    :param timeout: Serial timeout in seconds.
    :return: Numpy array of raw sensor data (3648 pixels in 8-bit mode)
    """
    # Initialize array for 3648 pixels (8-bit mode)
    sensor_raw_data = np.zeros(length, dtype=np.uint8)

    # Send initialization command (0xA2 for 8-bit mode)
    packet = bytearray()
    # packet.append(0xA1)  # read TCD1304 module (12 bpp, 7296 bytes)
    packet.append(0xA2)  # read TCD1304 module (8 bpp, 3648 bytes)

    ser.write(packet)

    i = 0
    start_time = time.time()
    timeout_duration = 5  # Maximum time to wait for data in seconds

    while i < length:
        # Check for timeout
        if time.time() - start_time > timeout_duration:
            print(f"Timeout reached. Received {i} bytes out of 3648.")
            break

        # Read available bytes
        bytes_available = ser.in_waiting
        if bytes_available > 0:
            # Read up to the remaining bytes we need
            bytes_to_read = min(bytes_available, length - i)
            data = ser.read(bytes_to_read)

            # Store the bytes in our array
            sensor_raw_data[i : i + len(data)] = np.frombuffer(data, dtype=np.uint8)
            i += len(data)

    return sensor_raw_data


def convert_and_plot(sensor_raw_data):
    """
    Plots the 8-bit intensity values from the sensor.
    :param sensor_raw_data: Raw byte array from the sensor (8-bit values)
    """
    if len(sensor_raw_data) != length:
        print(
            f"Warning: Expected 3648 bytes, got {len(sensor_raw_data)}. Plotting available data."
        )

    # Create pixel indices for plotting
    pixels = np.arange(0, len(sensor_raw_data))
    sensor_raw_data = 255 - sensor_raw_data
    sensor_raw_data[0:4] = sensor_raw_data[5]

    plt.figure(figsize=(10, 6))
    plt.plot(pixels, sensor_raw_data)
    plt.title("TCD1304 Spectrum (8-bit mode)")
    plt.xlabel("Pixel")
    plt.ylabel("Intensity (8-bit)")
    plt.ylim(0, 255)  # 8-bit range
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    port_name = "/dev/ttyCH341USB0"  # Adjust for your system
    ser = serial.Serial(port_name, baudrate, timeout=timeout)
    print(f"Connected to {port_name}")

    while True:
        try:
            data = np.zeros(length, dtype=np.float64)
            for i in tqdm(range(averages)):
                data += read_sensor_data(ser)
            data /= averages
            convert_and_plot(data)
        except serial.SerialException as e:
            print(f"Serial port error: {e}")
        except Exception as e:
            print(f"An error occurred: {e}")
    ser.close()
