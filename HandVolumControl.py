import sys
import cv2
import time
import numpy as np
import HandTrackingMoulde as htm
import math
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

######################################
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None
)
volume = cast(interface, POINTER(IAudioEndpointVolume))
######################################
CamWight, CamHeight = 800, 720
cTime = 0
pTime = 0
######################################
cap = cv2.VideoCapture(0)
cap.set(3, CamWight)
cap.set(4, CamHeight)
######################################
detector = htm.HandDetector(max_hands=2, min_detectionCon=0.6)
volRange = volume.GetVolumeRange()
print(volume.GetVolumeRange())
minvol = volRange[0]
maxvol = volRange[1]
vol = 0
volBar = 400
volPer = 0
# min: 30 max:300
minl = 30
maxl = 250
#######################Begin##########################
while True:
    scc , img = cap.read()
    img = detector.findHands(img)
    lmlist = detector.findHandPosition(img, draw=False)
    if len(lmlist) != 0:
        # print(lmlist[4], lmlist[8])

        x1, y1 = lmlist[4][1], lmlist[4][2]
        x2, y2 = lmlist[8][1], lmlist[8][2]
        cx, cy = (x1+x2)//2, (y1+y2)//2

        cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)
        cv2.line(img,(x1, y1), (x2, y2), (255, 0, 255), 3)


        length = int(math.hypot(x1-x2, y1-y2))
        # print(length)
        # 将长度转化为音量范围值
        vol = np.interp(length, [minl, maxl], [minvol, maxvol])

        # 将长度转化为对应控件的高度以及百分比值
        volBar = np.interp(length, [minl, maxl], [400, 150])
        volPer = np.interp(length, [minl, maxl], [0, 100])

        print(vol)
        volume.SetMasterVolumeLevel(vol, None)

        if length < minl or length > maxl:
            cv2.circle(img, (cx, cy), 15, (255, 255, 0), cv2.FILLED)

    # 窗口音量控件
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 2)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)} %', (50, 430),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1, color=(0, 255, 0), thickness=2)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img, f'FPS: {int(fps)}', (30, 60),
                fontFace=cv2.FONT_HERSHEY_COMPLEX,
                fontScale=1, color=(0, 255, 0), thickness=2)
    cv2.imshow("Image",img)
    cv2.waitKey(1)

