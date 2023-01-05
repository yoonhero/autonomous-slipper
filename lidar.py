import serial


class Lidar():
    def __init__(self):
        ser = serial.Serial(port='/dev/tty.usbserial-0001',
                    baudrate=230400,
                    timeout=5.0,
                    bytesize=8,
                    parity='N',
                    stopbits=1)

    
    def get(self):