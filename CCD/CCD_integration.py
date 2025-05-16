import time
import matplotlib.pyplot as plt
import numpy as np
import serial
from tqdm import tqdm

# Constants
length = 3648  # Number of pixels
bytes_expected = length * 2  # 2 bytes per pixel (12-bit data in 16-bit container)
averages = 1
baudrate = 921600
timeout = 1

def read_sensor_data_12bpp(ser, integration_cmd):
    """
    Reads 12-bit data (packed in 16-bit words) from the TCD1304 sensor using a given integration time command byte.
    """
    ser.reset_input_buffer()

    packet = bytearray()
    packet.append(0xA1)  # read TCD1304 module (12 bpp)
    packet.append(integration_cmd)  # integration time setting
    # Max D6
    # Min B0

    ser.write(packet)

    buffer = bytearray()
    start_time = time.time()
    timeout_duration = 5

    while len(buffer) < bytes_expected:
        if time.time() - start_time > timeout_duration:
            print(f"Timeout reached. Received {len(buffer)} bytes out of {bytes_expected}.")
            break
        if ser.in_waiting > 0:
            buffer += ser.read(ser.in_waiting)

    if len(buffer) < bytes_expected:
        print("Warning: Incomplete data received.")
        return np.zeros(length, dtype=np.uint16)

    raw_data = np.frombuffer(buffer[:bytes_expected], dtype=np.uint16)
    pixel_data = raw_data & 0x0FFF  # Mask to 12-bit data

    return pixel_data

def plot_multiple_exposures(exposure_data_dict):
    """
    Plot all spectra from different exposure settings on the same plot.
    """
    plt.figure(figsize=(12, 7))
    for integration_cmd, data in exposure_data_dict.items():
        pixels = np.arange(len(data))
        corrected_data = 4095 - data
        corrected_data[0:4] = corrected_data[5]
        label = f"Exp cmd: 0x{integration_cmd:02X}"
        plt.plot(pixels, corrected_data, label=label)

    plt.title("TCD1304 Spectrum at Various Exposure Settings")
    plt.xlabel("Pixel")
    plt.ylabel("Intensity (12-bit, inverted)")
    plt.ylim(0, 4095)
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == "__main__":
    port_name = "/dev/ttyCH341USB0"
    ser = serial.Serial(port_name, baudrate, timeout=timeout)
    print(f"Connected to {port_name}")

    exposure_data = {}
    try:
        for integration_cmd in tqdm(range(0xB0, 0xD6)):  # You can increase the upper limit if needed
            data = np.zeros(length, dtype=np.float64)
            for _ in range(averages):
                data += read_sensor_data_12bpp(ser, integration_cmd)
            data /= averages
            exposure_data[integration_cmd] = data
            time.sleep(0.5)
            #ser.write(bytearray([0xA1]))
            #time.sleep(5)


        plot_multiple_exposures(exposure_data)
    except serial.SerialException as e:
        print(f"Serial port error: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        ser.close()
