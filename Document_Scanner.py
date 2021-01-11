import cv2
import numpy as np

widgtImg,hightImg =640,720

cap = cv2.VideoCapture(2)          #0 main Camera, 1 Secound camera.....
cap.set(3,widgtImg)         #Width
cap.set(3,hightImg)         #Hight
cap.set(10,10)        #Brightness


def preProcessing(img):
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    imgBlur = cv2.GaussianBlur(imgGray,(5,5),1)
    imgCanny = cv2.Canny(imgBlur,100,200)
    kernel = np.ones((5,5))
    imgDialation = cv2.dilate(imgCanny,kernel,iterations=2)
    imgThreshold = cv2.erode(imgDialation,kernel,iterations=1)
    return imgThreshold

def getContours(img):
    biggest = np.array([])
    maxArea = 0
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area >5000:
            #cv2.drawContours(imgContours, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt,True)
            approx = cv2.approxPolyDP(cnt,0.02*perimeter,True)      #approximates number of corners
            if len(approx) == 4 and area > maxArea:
                biggest = approx
                maxArea = area
            print(len(approx))
    cv2.drawContours(imgContours, biggest, -1, (255, 0, 0), 16)
    return  biggest

def getWarp(img,biggest):
    if biggest.size >0:
        print(biggest)
        biggest = reorder(biggest)
        pts1 = np.float32(biggest)
        pts2 = np.float32([[0, 0], [widgtImg, 0], [0, hightImg], [widgtImg, hightImg]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        imgOut = cv2.warpPerspective(img, matrix, (widgtImg, hightImg))

        return imgOut

def reorder(myPoints):
    if myPoints.size >0:
        myPoints = myPoints.reshape((4,2))
        myPointsNew = np.zeros((4,1,2),np.int32)

        add = myPoints.sum(axis=1)       #Axis 1
        print("Add ",add)
        myPointsNew[0] = myPoints[np.argmin(add)]       #index of the smallest
        myPointsNew[3] = myPoints[np.argmax(add)]
        differ = np.diff(myPoints,axis=1)
        myPointsNew[1] = myPoints[np.argmin(differ)]
        myPointsNew[2] = myPoints[np.argmax(differ)]
        print("NewPoints",myPointsNew)
        return myPointsNew
while True:
    success, img = cap.read()
    cv2.resize(img,(widgtImg,hightImg))
    imgContours = img.copy()
    imgThreshold = preProcessing(img)
    biggest = getContours(imgThreshold)
    output = getWarp(img,biggest)
    cv2.imshow("Video", imgThreshold)
    cv2.imshow("Contours",imgContours)
    cv2.imshow("Scanner",output)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

