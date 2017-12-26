import cv2
import numpy

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')
capture = cv2.VideoCapture(0)
print(capture)
while capture.isOpened():
    ret, img = capture.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    for (x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h),(255,0,0),1)
    cv2.imshow('Your face',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break    
capture.release()
cv2.destroyAllWindows()
"""face_gray = gray[y:y+h,x:x+w]
        face_color = img[y:y+h,x:x+w]
        eyes = smile_cascade.detectMultiScale(face_gray, 1.3, 5)
        for(x1,y1,w1,h1) in eyes:
            cv2.rectangle(img,(x1,y1), (x1+w1, y1+h1),(0,255,0),1)"""