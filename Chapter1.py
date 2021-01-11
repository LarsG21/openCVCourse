import cv2
import numpy as np
""""
img = cv2.imread("Recources/Bird.jpg")
cv2.imshow("Output", img)                                    #Image Import 
cv2.waitKey(0)
"""

""""
vidcap = cv2.VideoCapture("Recources/Video.mp4")
while True:
    success, img = vidcap.read()
    cv2.imshow("Video", img)                                  #Video Import
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
"""
"""
cap = cv2.VideoCapture(0)          #0 main Camera, 1 Secound camera.....
cap.set(2,640)         #Width
cap.set(3,280)         #Hight
cap.set(10,10)        #Brightness

while True:
    success, img = cap.read()
    cv2.imshow("Video", img)                               #Video Import from Webcam
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break
"""

"""
img = cv2.imread("Recources/Bird.jpg")
imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)             #Colorconversions
imgBlur = cv2.GaussianBlur(imgGray,(13,13),0)              #Blur

cv2.imshow("Gray Image", imgGray)
cv2.imshow("Blur Image", imgBlur)
cv2.waitKey(0)
"""

img = cv2.imread("Recources/Bird.jpg")                      #Edge detector
imgEdges = cv2.Canny(img,100,700)

kernel = np.ones((5,5),np.uint8)                #kernel = matrix

imgDilation = cv2.dilate(imgEdges,kernel,iterations=3)          #dickere Edges
imgErodet = cv2.erode(imgDilation,kernel,iterations=2)          #dünnere Edges



cv2.imshow("Edges Image", imgEdges)
cv2.imshow("Edges Dilation", imgDilation)
cv2.imshow("Edges Erosion",imgErodet)
cv2.waitKey(0)
