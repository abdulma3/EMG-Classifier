import serial
import csv
import time

PORT = '/dev/cu.usbserial-130'  # your actual port
BAUD = 9600
ser = serial.Serial(PORT, BAUD)
FILENAME = 'emg_data.csv'

def record_session(label, duration_sec):
    print(f"\nGet ready for '{label}'...")
    time.sleep(2)
    # Actively drain everything, not just a single reset
    ser.reset_input_buffer()
    drain_start = time.time()
    while time.time() - drain_start < 0.3:
        if ser.in_waiting > 0:
            ser.read(ser.in_waiting)
    ser.reset_input_buffer()
    print("GO!")
    start = time.time()
    with open(FILENAME, 'a', newline='') as f:
        writer = csv.writer(f)
        while time.time() - start < duration_sec:
            try:
                value = int(ser.readline().decode().strip())
                writer.writerow([time.time(), value, label])
            except (ValueError, UnicodeDecodeError):
                continue
    print(f"Done with '{label}'.")

classes = ['rest', 'light_flex', 'hard_flex']
reps_per_class = 3
rep_duration = 10       # seconds per rep
rest_between_reps = 5   # seconds to recover

for rep in range(1, reps_per_class + 1):
    for label in classes:
        print(f"\n=== Rep {rep}/{reps_per_class} — {label} ===")
        record_session(label, rep_duration)
        print(f"Rest {rest_between_reps}s before next one...")
        time.sleep(rest_between_reps)

print("\nAll done! Data saved to", FILENAME)