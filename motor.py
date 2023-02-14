try:
    import RPi.GPIO as GPIO
except ImportError:
    pass
from dataclasses import dataclass
import time

GPIO.setwarnings(False)


@dataclass
class MotorPin:
    in1: int
    in2: int
    ena: int


# class CustomPID():
#     def __init__(self):
#         self.pwms = []

#     def update(self, ):

class Motor():
    def __init__(self):
        self.leftMotor = MotorPin(5, 6, 13)
        self.rightMotor = MotorPin(16, 26, 12)

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
        self.changePWM(0, 0)

    def changePWM(self, left_pwm, right_pwm):
        self.power_left.ChangeDutyCycle(left_pwm)
        self.power_right.ChangeDutyCycle(right_pwm)

    def back(self, velocity):
        pwm1, pwm2 = velocity[0], velocity[1]
        self.changePWM(pwm1, pwm2)

        GPIO.output(self.leftMotor.in1, GPIO.LOW)
        GPIO.output(self.leftMotor.in2, GPIO.HIGH)
        GPIO.output(self.rightMotor.in1, GPIO.HIGH)
        GPIO.output(self.rightMotor.in2, GPIO.LOW)

    def forward(self, velocity):
        pwm1, pwm2 = velocity[0], velocity[1]
        self.changePWM(pwm1, pwm2)

        GPIO.output(self.leftMotor.in1, GPIO.HIGH)
        GPIO.output(self.leftMotor.in2, GPIO.LOW)
        GPIO.output(self.rightMotor.in1, GPIO.LOW)
        GPIO.output(self.rightMotor.in2, GPIO.HIGH)

    def left(self, velocity):
        right_pwm, left_pwm = velocity[0], velocity[1]
        self.changePWM(right_pwm, left_pwm)
        # TODO : Test Angle for accurate turning

        GPIO.output(self.leftMotor.in1, GPIO.LOW)
        GPIO.output(self.leftMotor.in2, GPIO.HIGH)
        GPIO.output(self.rightMotor.in1, GPIO.LOW)
        GPIO.output(self.rightMotor.in2, GPIO.HIGH)

    def right(self, velocity):
        right_pwm, left_pwm = velocity[0], velocity[1]
        self.changePWM(right_pwm, left_pwm)

        GPIO.output(self.leftMotor.in1, GPIO.HIGH)
        GPIO.output(self.leftMotor.in2, GPIO.LOW)
        GPIO.output(self.rightMotor.in1, GPIO.HIGH)
        GPIO.output(self.rightMotor.in2, GPIO.LOW)

    def reset(self):
        GPIO.cleanup()


if __name__ == "__main__":
    rcdriver = Motor()
    import time

    while True:
        for i in range(3):
            vel = (i+1)*30
            velocity = [vel, vel]

            time.sleep(1)
            print(f"PWM: {vel}")

            rcdriver.forward(velocity)

            time.sleep(1)

            rcdriver.left(velocity)


            time.sleep(1)

            rcdriver.right(velocity)
