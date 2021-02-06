import cv2
import numpy as np


img = np.zeros((512,512,3),np.uint8)
print(img.shape)
print(img.shape[0])    #Hight
print(img.shape[1])    #Width
print(img.shape[2])    #channels

#img[:] = 255,0,0                        #all blue
#img[200:300,100:300] = 0,255,0          #selected green

cv2.line(img,(0,0),(img.shape[1],img.shape[0]),(0,0,255),3)         #Bilder bearbeiten
cv2.rectangle(img,(30,30),(80,120),(255,0,0),cv2.FILLED)
cv2.circle(img,(230,400),30,(0,255,0),2)
cv2.putText(img,"Hello",(400,300),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,0),2)


cv2.imshow("Image",img)

cv2.waitKey(0)