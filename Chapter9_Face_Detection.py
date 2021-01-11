import cv2

cap = cv2.VideoCapture(1)

faceCascade = cv2.CascadeClassifier("Recources/haarcascade_frontalface_default.xml")


#imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
#faces = faceCascade.detectMultiScale(imgGray,1.1,4)

while True:
    succsess, img = cap.read()
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(imgGray,1.1,4)

    cv2.imshow("Tracking",img)
    if succsess:
        for (x, y, w, h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)
    else:
        cv2.putText(img,"Lost",(5,50),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,255))

#for(x,y,w,h) in faces:
#    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),2)

#cv2.imshow("Result",img)

