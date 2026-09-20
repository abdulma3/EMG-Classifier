import serial, time

PORT = '/dev/cu.usbserial-130'
ser = serial.Serial(PORT, 9600)
ser.reset_input_buffer()
time.sleep(0.5)
ser.reset_input_buffer()

count = 0
start = time.time()
while time.time() - start < 5:
       ser.readline()
       count += 1
       print(f"{count} lines in 5 seconds = {count/5:.1f} samples/sec")