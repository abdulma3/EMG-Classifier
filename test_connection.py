import serial

PORT = '/dev/cu.usbserial-130'
BAUD = 9600
ser = serial.Serial(PORT, BAUD)

print("Reading 20 values... flex your arm now")
for _ in range(20):
    print(ser.readline().decode().strip())