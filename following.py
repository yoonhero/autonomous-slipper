import mediapipe as mp
import cv2
import mediapipe as mp
import pygame
from pygame.locals import *

from motor import Motor
from camera import CAM


class Following(object):
    WIDTH, HEIGHT = 640, 360

    def __init__(self):
        self.rc_driver = Motor()
        self.camera = CAM(Following.WIDTH, Following.HEIGHT)
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils

        pygame.init()
        self.screen = pygame.display.set_mode(
            (Following.WIDTH, Following.HEIGHT))

        self.target_w = Following.WIDTH / 3
        self.target_h = Following.HEIGHT / 3
        self.threshold = 20
        self.initial_pwm = [25, 25]
        self.safe_area = (Following.WIDTH/4, Following.WIDTH-Following.WIDTH/4)

    # Start Following
    def run(self):
        while True:
            pygame.display.update()
            self.screen.fill([255, 255, 255])

            try:
                image = self.camera.get_frame()
                image = cv2.flip(image, 1)

                with self.mp_face_detection.FaceDetection(
                        model_selection=0, min_detection_confidence=0.5) as face_detection:
                    image.flags.writeable = False
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    results = face_detection.process(image)

                    image.flags.writeable = True
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    if results.detections:
                        detection = results.detections[0]

                        location_data = detection.location_data
                        x1 = location_data.relative_bounding_box.xmin * Following.WIDTH
                        width = location_data.relative_bounding_box.width * Following.WIDTH
                        x2 = x1 + width
                        cx = (x1 + x2) / 2

                        print(
                            f"Center: {cx} | x1: {x1} | x2: {x2} | Safe Area: {self.safe_area}")

                        if cx < self.safe_area[0]:
                            print("Turn Left")
                            self.rc_driver.left(self.initial_pwm)
                        elif cx > self.safe_area[1]:
                            print("Turn Right")
                            self.rc_driver.right(self.initial_pwm)
                        elif width >= self.target_w + self.threshold:
                            print("Back")
                            self.rc_driver.back(self.initial_pwm)
                        elif width <= self.target_w - self.threshold:
                            print("Forward")
                            self.rc_drier.forward(self.initial_pwm)

                        self.mp_drawing.draw_detection(image, detection)

                        pygame_image = pygame.image.frombuffer(
                            image.tobytes(), (720, 1280), "RGB")
                        self.screen.blit(pygame_image, (0, 0))

            except:
                self.end()

    def end(self):
        self.rc_driver.stop_all()
        self.camera.quit()


if __name__ == "__main__":
    fl = Following()

    fl.run()
