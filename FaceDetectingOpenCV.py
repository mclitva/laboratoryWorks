import cv2
import numpy

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
capture = cv2.VideoCapture(0)
print(capture)
while capture.isOpened():
    ret, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0),2)
    cv2.imshow('Your face',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
capture.release()
cv2.destroyAllWindows()