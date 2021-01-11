import cv2
import numpy as np

img = cv2.imread("Recources/Porsche-918-Spyder-11-1-750x375.jpg")
print(img.shape)


imgRezize = cv2.resize(img,(900,500))           #Resizeing


cv2.imshow("Image",img)
cv2.imshow("Resize",imgRezize)
print(imgRezize.shape);

imgCropped = img[0:200,200:500]                   #First hight secound width
cv2.imshow("Image",imgCropped)


cv2.waitKey(0)