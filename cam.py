import cv2
import numpy as np 
from matplotlib import pyplot as plt

cap = cv2.VideoCapture(0)
...
while True:
    ret, frame = cap.read()
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if ret:
        plt.imshow(frame, cmap="gray")
        plt.show()

        # cv2.imshow("hi", frame)
        # import time
        # time.sleep(2)
        # break

    if cv2.waitKey(0) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()