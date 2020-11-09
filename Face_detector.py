import cv2
from random import randrange

cv2.namedWindow("Face Detector", cv2.WINDOW_NORMAL)

#Load some pre-trained data
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#Loading Webcam
webcam = cv2.VideoCapture(0)

while True:
    #Reading current frame
    successful_frame_read, frame = webcam.read()
    
    #Converting to grayscale
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #Detect Faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #Draw rectangle
    for (x,y,w,h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),3)
        #cv2.rectangle(frame, (x,y), (x+w,y+h),(randrange(256),randrange(256),randrange(256)),3)

    cv2.imshow('Face Detector', frame)
    
    key = cv2.waitKey(1)

    if key==27:
        break

#release webcam
webcam.release()
cv2.destroyAllWindows()