import cv2 as cv


img=cv.imread('group2.jpg')
img_gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

face_dect=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
eye_dect=cv.CascadeClassifier("haarcascade_eye.xml")

face=face_dect.detectMultiScale(img_gray,scaleFactor=1.1,minSize=(70,70),)
eye=eye_dect.detectMultiScale(img_gray,scaleFactor=1.1)

for (fx,fy,fw,fh) in face:
    cv.rectangle(img,(fx,fy),(fx+fw,fy+fh),(255,0,0),5)
    print(fw,fh)
    
for (ex,ey,ew,eh) in eye:
    cv.rectangle(img,(ex,ey),(ex+ew,ey+eh),(0,255,0),5)
    
cv.imshow("image",img)

cv.waitKey(0)
cv.destroyAllWindows()