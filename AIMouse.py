import cv2
import numpy as np
import HandTrackingMoulde as htm
import time
import autopy



cap = cv2.VideoCapture(1)
cap.set(3, 640)
cap.set(4, 480)


while True:
    _, img = cap.read()


    cv2.imshow("Image", img)
    cv2.waitKey(1)

