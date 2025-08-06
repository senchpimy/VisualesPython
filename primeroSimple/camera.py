import os

os.environ["QT_QPA_PLATFORM"] = "xcb"
from cvzone.HandTrackingModule import HandDetector
import cv2
import math
from multiprocessing import Process, Queue

for i in range(10):
    cap = cv2.VideoCapture(i)
    if cap.isOpened():
        print(f"Cámara abierta con éxito en el índice {i}.")
        break
    else:
        print(f"Error: No se pudo abrir la cámara con índice {i}. Probando con el siguiente índice...")

detector = HandDetector(
    staticMode=False, maxHands=1, modelComplexity=1, detectionCon=0.5, minTrackCon=0.5
)


def findAngle(p1, p2, p3, img=None, color=(255, 0, 255), scale=5, invert=False):
    """
    Finds angle between three points.

    :param p1: Point1 - (x1,y1)
    :param p2: Point2 - (x2,y2)
    :param p3: Point3 - (x3,y3)
    :param img: Image to draw output on. If no image input output img is None
    :return:
    """

    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3

    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    if invert:
        angle = 360 - angle

    if img is not None:
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), max(1, scale // 5))
        cv2.line(img, (x3, y3), (x2, y2), (255, 255, 255), max(1, scale // 5))
        cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
        cv2.circle(img, (x1, y1), scale + 5, color, max(1, scale // 5))
        cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
        cv2.circle(img, (x2, y2), scale + 5, color, max(1, scale // 5))
        cv2.circle(img, (x3, y3), scale, color, cv2.FILLED)
        cv2.circle(img, (x3, y3), scale + 5, color, max(1, scale // 5))
        cv2.putText(
            img,
            str(int(angle)),
            (x2 - 50, y2 + 50),
            cv2.FONT_HERSHEY_PLAIN,
            2,
            color,
            max(1, scale // 5),
        )
    return angle, img


def angleCheck(myAngle, targetAngle, offset=20):
    return targetAngle - offset < myAngle < targetAngle + offset


def get_data(que: Queue):
    while True:
        success, img = cap.read()
    
        hands, img = detector.findHands(img, draw=False, flipType=True)
    
        if hands:
            hand1 = hands[0]
            lmList1 = hand1["lmList"]
            #bbox1 = hand1["bbox"]
            #center1 = hand1["center"]
            #handType1 = hand1["type"]
    
            fingers1 = detector.fingersUp(hand1)
            print(f"H1 = {fingers1.count(1)} %% {fingers1} %%", end=" ")
    
            coords = []
            for i, finger in enumerate(fingers1):
                if finger == 1:
                    i += 1
                    coords.append(lmList1[i * 4][0:2])
            print(coords)
            ang = findAngle(
                lmList1[8][0:2],
                lmList1[0][0:2],
                [0, 0],
                img,
                color=(0, 255, 0),
                scale=5,
                invert=True,
            )
            data = f"vy {int(ang[0])}"
            que.put(data)
            for coord in coords:
                cv2.circle(img, coord, 10, (255, 255, 255), 1)
            print(" ")
    
        cv2.imshow("Image", img)
        cv2.waitKey(1)
