import cv2
import numpy as np

path = 'Recources/shapes.png'

img = cv2.imread(path)
imageContours = img.copy()

def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape [:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        ver = hor
    return ver

def getContours(img):
    contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area >500:
            cv2.drawContours(imageContours, cnt, -1, (255, 0, 0), 3)
            perimeter = cv2.arcLength(cnt,True)
            print(perimeter)
            approx = cv2.approxPolyDP(cnt,0.02*perimeter,True)      #approximates number of corners
            print(len(approx))
            objectCorners = len(approx)
            x ,y ,w ,h = cv2.boundingRect(approx)

            if objectCorners == 3: objectType = "Triangle"
            elif objectCorners == 4:
                aspRatio = w/float(h)
                if aspRatio >0.95 and aspRatio <1.05: objectType ="Square"
                else: objectType ="Rectangle"
            else: objectType = "Circle"


            cv2.rectangle(imageContours,(x,y),(x+w,y+h),(255,0,255),2)
            cv2.putText(imageContours,objectType,(x,y-6),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0),1)

imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgBlur = cv2.GaussianBlur(imgGray,(7,7),1)
imageCanny = cv2.Canny(imgBlur,50,50)
imgBlack = np.zeros_like(img)

getContours(imageCanny)

imgResult = stackImages(0.7,([img,imgGray,imgBlur],
                             [imageCanny,imageContours,imgBlack]))


cv2.imshow("Result",imgResult)
cv2.waitKey(0)