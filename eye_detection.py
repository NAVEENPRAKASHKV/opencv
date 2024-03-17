import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread('group.jpg')
img_gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)

face_detect =cv.CascadeClassifier('haarcascade_eye.xml')
face=face_detect.detectMultiScale(img_gray,scaleFactor=1.5)
print(face)


for (x,y,w,h) in face:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
    

cv.imshow("eye",img)
cv.waitKey(0)
cv.destroyAllWindows()