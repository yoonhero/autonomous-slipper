import cv2
import numpy as np 
from matplotlib import pyplot as plt


def calibration_frame(frame):
    return ## TODO: Calibration

class CAM(object):
    def __init__(self, target_img_width, target_img_height):
        self.cap = cv2.VideoCapture(0)

        self.t_width = target_img_width
        self.t_height = target_img_height

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.t_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.t_height)

    def get_frame(self):
        assert self.cap.isOpened() == True, "Please Check Camera Connection"

        ret, frame = self.cap.read()
        
        assert ret == True, "Camera Access Failed"

        return frame

    def quit(self):
        self.cap.release()
        cv2.destroyAllWindows()





if __name__ == "__main__":
    cap = cv2.VideoCapture(0)

    for i in range(20):
        ret, frame = cap.read()

        if ret:
            cv2.imshow("hi", frame)

            import time
            time.sleep(1)

            cv2.imwrite(f'./calibration/cal{i}.png', frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else: 
            break

    cap.release()
    cv2.destroyAllWindows()
