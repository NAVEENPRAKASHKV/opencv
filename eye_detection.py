import cv2 as cv
import matplotlib.pyplot as plt

img=cv.imread('group.jpg')
img_gray =cv.cvtColor(img,cv.COLOR_BGR2GRAY)

face_detect =cv.CascadeClassifier('haarcascade_eye.xml')
face=face_detect.detectMultiScale(img_gray,scaleFactor=1.5)
print(face)


for (x,y,w,h) in face:
    cv.rectangle(img,(x,y),(x+w,y+h),(255,0,0),5)
    

# Load the pre-trained face detection cascade
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(img_gray, scaleFactor=1.2, minNeighbors=5, minSize=(30, 30))


# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


# Display the image with detected faces
cv.imshow('Detected Faces', img)
cv.waitKey(0)
cv.destroyAllWindows()