import serial
import joblib
import numpy as np
from collections import deque
from scipy.signal import butter, filtfilt

PORT = '/dev/cu.usbserial-140'
BAUD = 9600
ser = serial.Serial(PORT, BAUD)
model = joblib.load('emg_model.pkl')

def bandpass_filter(data, lowcut=20, highcut=450, fs=150, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = min(highcut / nyq, 0.99)
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def extract_features(window):
    mav = np.mean(np.abs(window))
    rms = np.sqrt(np.mean(window**2))
    zc = np.sum(np.diff(np.sign(window)) != 0)
    wl = np.sum(np.abs(np.diff(window)))
    return [mav, rms, zc, wl]

buffer = deque(maxlen=50)
ser.reset_input_buffer()

print("Reading live... flex your arm (Ctrl+C to stop)")
while True:
    try:
        value = int(ser.readline().decode().strip())
        buffer.append(value)
        if len(buffer) == 50:
            window = bandpass_filter(np.array(buffer))
            feats = extract_features(window)
            prediction = model.predict([feats])[0]
            print(f"Current state: {prediction}")
    except (ValueError, UnicodeDecodeError):
        continue
    except KeyboardInterrupt:
        break