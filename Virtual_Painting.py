import cv2
import  numpy as np

cap = cv2.VideoCapture(1)          #0 main Camera, 1 Secound camera.....
cap.set(3,640)         #Width
cap.set(3,280)         #Hight
cap.set(10,150)        #Brightness

myColors = [[0,69,105,255,194,255],     #Orange
            [79,158,172,255,0,170],     #Blue
            [39,73,90,238,147,255]]     #Neon


def findColor(img):
    imgHSV = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imgHSV, lower, upper)
        cv2.imshow(str(color[0]),mask)

while True:
    success, img = cap.read()
    findColor(img)
    cv2.imshow("Video", img)                               #Video Import from Webcam
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break