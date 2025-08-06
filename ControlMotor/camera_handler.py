# Mano expandida: Se separan todos los elementos y si gira gira el ultimo
# Dedos L: Gira la parte de el centro
# 3 Dedos (L mas uno): gira la parte de el principio
# Dice el nombre de cada pieza seleccionada en una fuente y con perspectiva
# Dos flechas de doreccion
# Unos aros de colores medio raros

import os

os.environ["QT_QPA_PLATFORM"] = "xcb"

import cv2
import math
import time
from multiprocessing import Queue
from cvzone.HandTrackingModule import HandDetector

def findAngle(p1, p2, p3, img, color=(255, 0, 255), scale=10, dibujar=False):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    angle = math.degrees(math.atan2(y3 - y2, x3 - x2) - math.atan2(y1 - y2, x1 - x2))
    if angle < 0:
        angle += 360
    if img is not None and dibujar:
        cv2.line(img, (x1, y1), (x2, y2), (255, 255, 255), 3)
        cv2.line(img, (x2, y2), (x3, y3), (255, 255, 255), 3)
        cv2.circle(img, (x1, y1), scale, color, cv2.FILLED)
        cv2.circle(img, (x2, y2), scale, color, cv2.FILLED)
        cv2.circle(img, (x3, y3), scale, color, cv2.FILLED)
        cv2.putText(img, str(int(angle)), (x2 - 50, y2 + 50), cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    return angle, img

def run_detector(gesture_data_queue: Queue):
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            print(f"Cámara encontrada en {i}.")
            break
    else:
        print("ERROR: No se encontró cámara.")
        return

    detector = HandDetector(maxHands=1, modelComplexity=1, detectionCon=0.7, minTrackCon=0.7)

    while True:
        success, img = cap.read()
        if not success:
            time.sleep(0.1)
            continue
    
        hands, img_processed = detector.findHands(img, flipType=True, draw=False)
        
        detected_gesture = "Ninguno"
        rotation_angle = 90.0

        if hands:
            hand = hands[0]
            lmList = hand["lmList"]
            fingers = detector.fingersUp(hand)
            coords = []
            for i, finger in enumerate(fingers):
                if finger == 1:
                    i += 1
                    coords.append(lmList[i * 4][0:2])
            for coord in coords:
                cv2.circle(img, coord, 10, (255, 255, 255), 1)
            
            if fingers == [1, 1, 1, 1, 1]:
                detected_gesture = "Mano expandida"
            elif fingers == [1, 1, 0, 0, 0]:
                detected_gesture = "Dedos L"
            elif fingers == [1, 1, 1, 0, 0]:
                detected_gesture = "3 Dedos"

            if detected_gesture != "Ninguno":
                p1, p2 = lmList[9][0:2], lmList[0][0:2]
                p3 = (p2[0], img_processed.shape[0])
                angle, img_processed = findAngle(p1, p2, p3, img_processed, color=(0, 255, 0), scale=10)
                rotation_angle = 360 - angle if angle > 180 else angle

        gesture_packet = {"gesture": detected_gesture, "angle": rotation_angle}
        
        if not gesture_data_queue.full():
            gesture_data_queue.put(gesture_packet)

        cv2.imshow("Image", img_processed)
        cv2.waitKey(1)
