import cv2
from random import randrange
#Load some pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    




#image to detect faces on
img = cv2.imread('jp2.jpg')

#video to capture faces
#webcam = cv2.VideoCapture(0)

#convert to grayscale
grayscaled_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Detect Faces
face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

#Draw rectangle
for (x,y,w,h) in face_coordinates:
    cv2.rectangle(img, (x,y), (x+w,y+h),(0,255,0),3)



cv2.imshow('Face Detector', img)
cv2.waitKey()

print("Code Completed")