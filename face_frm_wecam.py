import cv2 as cv

img=cv.VideoCapture(0)

while True :
    rat,frame=img.read()
    img_gray=cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    detection=cv.CascadeClassifier("haarcascade_frontalface_default.xml")
    face = detection.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x,y,w,h) in face:
        cv.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),5)
        cv.imshow("grouped",frame)
   
    
    if cv.waitKey(20) & 0xFF == ord('q'):
        break



cv.destroyAllWindows()