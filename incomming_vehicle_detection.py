import cv2
import numpy as np

cap = cv2.VideoCapture("footage_path_here")

car_classifier = cv2.CascadeClassifier("path_here")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:  
        break
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    cars = car_classifier.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    for (x, y, w, h) in cars:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

    cv2.imshow("Car Detection", frame)

    if cv2.waitKey(1) == 13: 
        break

cap.release()
cv2.destroyAllWindows()
