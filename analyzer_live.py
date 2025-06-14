import time

import matplotlib.pyplot as plt
import numpy as np
import serial

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


def read_sensor_data_12bpp(ser):
    """
    Reads 12-bit data (packed in 16-bit words) from the TCD1304 sensor.
    """
    ser.reset_input_buffer()

    packet = bytearray()
    packet.append(0xA1)  # read TCD1304 module (12 bpp, 7296 bytes)
    # packet.append(0xA2)  # read TCD1304 module (8 bpp, 3648 bytes)
    packet.append(0xB4)  # set integration time (Max D7, min B0)
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

    line.set_ydata(sensor_data)
    ax.relim()
    ax.autoscale_view()
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
