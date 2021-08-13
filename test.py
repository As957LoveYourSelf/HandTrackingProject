import HandTrackingMoulde as hd
import cv2

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
detector = hd.HandDetector(max_hands=1, min_detectionCon=0.6)
Topfingers = [4, 8, 12, 16, 20]
while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    img = detector.findHands(img)
    lmlist = detector.findHandPosition(img)
    # print(lmlist)
    if len(lmlist) != 0:
        fingers = []
        if lmlist[Topfingers[0]][1] < lmlist[Topfingers[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        for id in range(1, 5):
            if lmlist[Topfingers[id]][2] > lmlist[Topfingers[id] - 2][2]:
                # print("Hand close")
                fingers.append(0)
            else:
                fingers.append(1)
                # print("Hand open")
        print(fingers)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
