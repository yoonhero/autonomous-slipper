import RPi.GPIO as GPIO
from dataclasses import dataclass
import time

GPIO.setwarnings(False)


@dataclass
class MotorPin:
    in1: int
    in2: int
    ena: int


class RCDriver():
    def __init__(self):
        self.leftMotor = MotorPin(36, 38, 32)
        self.rightMotor = MotorPin(16, 18, 12)

        GPIO.setmode(GPIO.BCM)

        self.init_motor()

    def init_motor(self):
        GPIO.setup(self.leftMotor.in1, GPIO.OUT)
        GPIO.setup(self.leftMotor.in2, GPIO.OUT)
        GPIO.setup(self.leftMotor.ena, GPIO.OUT)

        GPIO.setup(self.rightMotor.in1, GPIO.OUT)
        GPIO.setup(self.rightMotor.in2, GPIO.OUT)
        GPIO.setup(self.rightMotor.ena, GPIO.OUT)

        # TODO: Test PWM
        self.power_left = GPIO.PWM(self.leftMotor.ena, 100)
        self.power_right = GPIO.PWM(self.rightMotor.ena, 100)

        self.power_left.start(0)
        self.power_right.start(0)

        self.stop_all()

    def stop_all(self):
        GPIO.setup(self.leftMotor.in1, GPIO.LOW)
        GPIO.setup(self.leftMotor.in2, GPIO.LOW)

        GPIO.setup(self.rightMotor.in1, GPIO.LOW)
        GPIO.setup(self.rightMotor.in2, GPIO.LOW)

    def changePWM(self, pwm):
        self.power_left.ChangeDutyCycle(pwm)
        self.power_right.changeDutyCycle(pwm)

    def forward(self, velocity):
        GPIO.output(self.leftMotor.in1, GPIO.LOW)
        GPIO.output(self.leftMotor.in2, GPIO.HIGH)

        GPIO.output(self.rightMotor.in1, GPIO.HIGH)
        GPIO.output(self.rightMotor.in2, GPIO.LOW)

        self.changePWM(velocity)

    def back(self, velocity):
        GPIO.output(self.leftMotor.in1, GPIO.HIGH)
        GPIO.output(self.leftMotor.in2, GPIO.LOW)

        GPIO.output(self.rightMotor.in1, GPIO.LOW)
        GPIO.output(self.rightMotor.in2, GPIO.HIGH)

        self.changePWM(velocity)

    def right(self, angle):
        # TODO : Test Angle for accurate turning

        GPIO.output(self.leftMotor.in1, GPIO.LOW)
        GPIO.output(self.leftMotor.in2, GPIO.HIGH)

        GPIO.output(self.rightMotor.in1, GPIO.LOW)
        GPIO.output(self.rightMotor.in2, GPIO.LOW)

    def left(self, angle):
        GPIO.output(self.leftMotor.in1, GPIO.LOW)
        GPIO.output(self.leftMotor.in2, GPIO.LOW)

        GPIO.output(self.rightMotor.in1, GPIO.HIGH)
        GPIO.output(self.rightMotor.in2, GPIO.LOW)

    def reset(self):
        GPIO.cleanup()


if __name__ == "__main__":
    rcdriver = RCDriver()

    while True:
        for i in range(3):
            vel = (i+1)*30

            rcdriver.forward(vel)
