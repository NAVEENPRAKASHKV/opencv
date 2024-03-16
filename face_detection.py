import cv2 as cv



# Load the image
img = cv.imread('group2.jpg')
# Convert the image to grayscale
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)


# Load the pre-trained face detection cascade
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')
# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))


# Draw rectangles around the detected faces
for (x, y, w, h) in faces:
    cv.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)


# Display the image with detected faces
cv.imshow('Detected Faces', img)
cv.waitKey(0)
cv.destroyAllWindows()
