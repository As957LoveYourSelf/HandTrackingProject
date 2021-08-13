import cv2
import mediapipe as mp


class HandDetector():
    def __init__(self, mode=False, max_hands=2, min_detectionCon=0.5, min_trackCon=0.5):
        self.mode = mode
        self.maxHands = max_hands
        self.min_detectionCon = min_detectionCon
        self.min_trackCon = min_trackCon
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.min_detectionCon,
                                        self.min_trackCon)
        self.mpDraws = mp.solutions.drawing_utils
        self.Topfingers = [4, 8, 12, 16, 20]

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    # 为识别的手部画点 (mpHands.HAND_CONNECTIONS则将点用线相连)
                    self.mpDraws.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    # 寻找指定的点
    def findHandPosition(self, img, handNo=0, draw=True):

        landMarkList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                height, weight, channel = img.shape
                # lm是该点在图片坐标的相对于图片尺寸的比例值，与图片高度和宽度相乘即可得出该点的像素值坐标
                cx, cy = int(lm.x * weight), int(lm.y * height)
                landMarkList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx, cy), 6, (255, 0, 0), cv2.FILLED)

        return landMarkList

    def countFingerUp(self, lmlist):
        fingers = []
        #  判断大拇指状态
        if lmlist[self.Topfingers[0]][1] < lmlist[self.Topfingers[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        #  判断四指状态
        for id in range(1, 5):
            if lmlist[self.Topfingers[id]][2] > lmlist[self.Topfingers[id] - 2][2]:
                fingers.append(0)
            else:
                fingers.append(1)
        return fingers
