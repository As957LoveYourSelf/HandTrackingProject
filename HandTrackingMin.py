import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraws = mp.solutions.drawing_utils
fpoint = (0,4,8,12,16,20)

cTime = 0
pTime = 0

while True:
    success, img = cap.read()
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    # print(results.multi_hand_landmarks)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id, lm)
                height, weight, channel = img.shape
                # lm是该点在图片坐标的相对于图片尺寸的比例值，与图片高度和宽度相乘即可得出该点的像素值坐标
                cx, cy = int(lm.x * weight),int(lm.y * height)
                print(id, cx,cy)
                #  获取特定点
                if id in fpoint:
                    cv2.circle(img, (cx,cy),15,(255,0,255),cv2.FILLED)

            # 为识别的手部画点 (mpHands.HAND_CONNECTIONS则将点用线相连)
            mpDraws.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)


    cTime = time.time()
    fps = 1/(cTime - pTime)
    pTime = cTime

    cv2.putText(img,str(int(fps)),(10,60),cv2.FONT_HERSHEY_PLAIN,3
                , (255, 255, 0), 3)
    cv2.imshow("Image",img)
    cv2.waitKey(1)