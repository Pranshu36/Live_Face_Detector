import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while True:

    successfull_fram_read, fram = webcam.read()
    grayscaled_img = cv2.cvtColor(fram, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(fram, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 2)

    cv2.imshow('Face Detector', fram)
    key = cv2.waitKey(1)
    if key==81 or key==113:
        break

webcam.release()