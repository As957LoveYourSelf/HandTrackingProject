import HandTrackingMoulde as hd
import cv2
import os
import time
import numpy as np

logopath = "AirPianterlogo"
logolist = os.listdir(logopath)
logoImgs = []

# add logo images to the list
for lp in logolist:
    image = cv2.imread(f'{logopath}/{lp}')
    logoImgs.append(image)
# 1 2 3 4 5 6 7.png
topLogo = logoImgs[-1]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
detector = hd.HandDetector(max_hands=1, min_detectionCon=0.6)
# image canvas by numpy
ImgCanvas = np.zeros((720, 1280, 3), np.uint8)
# bgr: pen color
pencolor = (60, 63, 65)
# radius
radius = 15
while True:
    # 1. import images
    _, img = cap.read()
    img = cv2.flip(img, 1)

    # 2.Find hand landmarks
    img = detector.findHands(img)
    lmlist = detector.findHandPosition(img)

    if len(lmlist) != 0:

        Cfingers = detector.countFingerUp(lmlist)
        x1, y1 = lmlist[8][1], lmlist[8][2]
        x2, y2 = lmlist[12][1], lmlist[12][2]

        # init modes
        isTwoFingersUp = Cfingers[1] and Cfingers[2]
        isOtherFingersUp = Cfingers[0] or Cfingers[3] or Cfingers[4]
        isSelectionMode = isTwoFingersUp and not isOtherFingersUp

        isOneFingerUp = Cfingers[1]
        isOthersFingerUp = Cfingers[0] or Cfingers[3] or Cfingers[4] or Cfingers[2]
        isDrawMode = isOneFingerUp and not isOthersFingerUp

        # 3.selection mode
        if isSelectionMode:
            # print("selection mode ")
            #  y: 0:125
            cv2.rectangle(img, (x1, y1), (x2, y2), (60, 140, 250), cv2.FILLED)
            if y1 <= 125 or y2 <= 125:
                if x1 >= 150 and x2 <= 300:
                    print("white")
                    pencolor = (255, 255, 255)
                    radius = 15
                    topLogo = logoImgs[0]
                elif x1 >= 340 and x2 <= 490:
                    print("black")
                    pencolor = (32, 40, 43)
                    radius = 15
                    topLogo = logoImgs[1]
                elif x1 >= 530 and x2 <= 680:
                    print("red")
                    pencolor = (0, 0, 255)
                    radius = 15
                    topLogo = logoImgs[2]
                elif x1 >= 730 and x2 <= 880:
                    print("green")
                    pencolor = (0, 255, 0)
                    radius = 15
                    topLogo = logoImgs[3]
                elif x1 >= 915 and x2 <= 1060:
                    print("blue")
                    pencolor = (255, 0, 0)
                    radius = 15
                    topLogo = logoImgs[4]
                elif x1 >= 1100 and x2 <= 1270:
                    print("rubber")
                    topLogo = logoImgs[5]
                    pencolor = (0, 0, 0)
                    radius = 30
                else:
                    topLogo = logoImgs[-1]
                    pencolor = (60, 63, 65)
                    radius = 15
                    print("wait.")

        # 4.draw mode
        if isDrawMode:
            xp, yp = 0, 0
            print("draw mode")
            cv2.circle(img, (x1, y1), radius, pencolor, cv2.FILLED)
            if xp == 0 and yp == 0:
                xp, yp = x1, y1
            cv2.line(img, (xp, yp), (x1, y1), pencolor, radius)
            cv2.line(ImgCanvas, (xp, yp), (x1, y1), pencolor, radius)
            xp, yp = x1, y1
    # main idea
    imgGray = cv2.cvtColor(ImgCanvas, cv2.COLOR_BGR2GRAY)
    retavl, imgInv = cv2.threshold(imgGray, 1, 255, cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img, imgInv)
    img = cv2.bitwise_or(img, ImgCanvas)

    img[0:125, 0:1280] = topLogo
    # img = cv2.addWeighted(img, 0.4, ImgCanvas, 0.6, 0)
    cv2.imshow("AirPainter", img)
    # cv2.imshow("GrayImage", imgGray)
    # cv2.imshow("Image INV", imgInv)
    # cv2.imshow("ImageCanvas", ImgCanvas)
    cv2.waitKey(1)
