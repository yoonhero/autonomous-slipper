import numpy as np
import cv2
import pygame
from pygame.locals import *
import time
import os
import glob
import matplotlib.pyplot as plt
import pandas as pd

from camera import CAM
import config
from motor import Motor

IMAGE_WIDTH, IMAGE_HEIGHT = config.IMAGE_WIDTH, config.IMAGE_HEIGHT


class CollectTrainingData(object):

    def __init__(self):
        self.rc_driver = Motor()

        # Create Labels
        self.k = np.eye(3).astype(np.int32)
        columns = ["X", "Y"]
        if os.path.isfile("training_data.csv"):
            self.df = pd.read_csv("training_data.csv")
        else:
            self.df = pd.DataFrame(columns=columns)

        self.camera = CAM(IMAGE_WIDTH, IMAGE_HEIGHT)

        pygame.init()
        self.screen = pygame.display.set_mode((IMAGE_WIDTH, IMAGE_HEIGHT))

        self.save_path = "./training_data/raw"
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)

    def collect(self):
        total_frame = self.get_total_frames()
        saved_frame = total_frame

        print("Start Collecting Training Data...")
        print("Press 'q' or 'x' to finish..")

        start = cv2.getTickCount()

        # [right_pwm, left_pwm]
        motor = np.zeros((2, ))
        # TODO: INITIAL PWM Forward
        initial_pwm = config.MOTOR_PWM

        # One hot vector for Behavior Decision
        Y = 0

        try:
            while True:
                pygame.display.update()
                frame = self.camera.get_frame()

                # plt.imshow(frame)
                # plt.show()
                self.screen.fill([255, 255, 255])
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # font = cv2.FONT_HERSHEY_SIMPLEX
                # org = (50, 50)
                # fontScale = 1
                # color = (255, 0, 0)
                # thickness = 2
                # image = cv2.putText(image, 'Collect Traning Data', org, font,
                #                     fontScale, color, thickness)

                # print(image.shape)
                pygame_image = pygame.image.frombuffer(
                    image.tobytes(), (864, 480), "RGB")
                self.screen.blit(pygame_image, (0, 0))

                for event in pygame.event.get():
                    if event.type == KEYDOWN:
                        key_input = pygame.key.get_pressed()

                        if key_input[pygame.K_RIGHT]:
                            # TODO: SET PWM for Driving
                            print("Forward Right")
                            motor[0] = initial_pwm
                            motor[1] = initial_pwm/2
                            Y = 0
                            self.rc_driver.right(motor)

                        elif  key_input[pygame.K_LEFT]:
                            print("Forward Left")
                            motor[0] = initial_pwm/2
                            motor[1] = initial_pwm
                            Y = 1
                            self.rc_driver.left(motor)

                        elif key_input[pygame.K_UP]:
                            print("Forward")
                            motor[0] = initial_pwm
                            motor[1] = initial_pwm
                            Y = 2
                            self.rc_driver.forward(motor)

                        elif key_input[pygame.K_SPACE]:
                            print("STOP Motor!!")
                            self.rc_driver.stop_all()
                            continue

                        elif key_input[pygame.K_x] or key_input[pygame.K_q]:
                            print("Exit")
                            self.camera.quit()
                            break

                        else:
                            continue

                        # after send pwms to motor saved the result
                        self.save_data(frame, saved_frame, Y)
                        saved_frame += 1

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

        finally:
            self.camera.quit()

    def save_data(self, frame, fname, y):
        destination = f"{self.save_path}/image_{fname}.png"
        cv2.imwrite(destination, frame)

        new_row = {'X': destination, "Y": [y]}
        new_df = pd.DataFrame(new_row)
        self.df = pd.concat([self.df, new_df])

        self.df.to_csv("training_data.csv")

    def get_total_frames(self):
        target = f"{self.save_path}/*"
        images_file_path = glob.glob(target)
        return len(images_file_path)


if __name__ == "__main__":
    ctd = CollectTrainingData()

    ctd.collect()
