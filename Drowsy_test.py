import cv2
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time

# Load the pre-trained drowsiness detection model
model = load_model("D:/ML_model/New_Drowsy_model.keras")

# Define class names
class_names = ['DROWSY', 'NATURAL']

# Load Haar cascades for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize webcam
cap = cv2.VideoCapture(0)  # 0 is usually the default camera, adjust if necessary

# Set up the plot for real-time visualization
plt.ion()  # Interactive mode on
fig, ax = plt.subplots()
x_data = deque(maxlen=100)  # Store up to 100 time points
y_data = deque(maxlen=100)  # Store up to 100 prediction results
line, = ax.plot([], [], lw=2)
ax.set_ylim(-0.1, 1.1)  # Adjust based on your prediction output
ax.set_xlabel('Time (seconds)')
ax.set_ylabel('Prediction Probability')
ax.set_title('Drowsiness Detection Over Time')

start_time = time.time()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        # Extract the face region from the frame
        face_roi = frame[y:y+h, x:x+w]

        # Resize the face region to match the model's input size
        img = cv2.resize(face_roi, (224, 224))

        # Convert the face region to a format suitable for prediction
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        img_array = img_array / 255.0  # Normalize if needed

        # Make a prediction
        predictions = model.predict(img_array)
        predicted_probabilities = predictions[0]  # Extract probabilities for the classes

        # Update plot data
        elapsed_time = time.time() - start_time
        x_data.append(elapsed_time)
        y_data.append(predicted_probabilities[np.argmax(predicted_probabilities)])  # Append the probability of the predicted class

        # Update the graph
        line.set_xdata(x_data)
        line.set_ydata(y_data)
        ax.set_xlim(max(0, elapsed_time - 10), elapsed_time)  # Show last 10 seconds

        # Redraw the plot
        fig.canvas.draw()
        fig.canvas.flush_events()

        # Display the result on the frame
        predicted_class = class_names[np.argmax(predictions)]
        cv2.putText(frame, f'Prediction: {predicted_class}', (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2, cv2.LINE_AA)

    # Show the frame
    cv2.imshow('Webcam - Drowsy Detection', frame)

    # Press 'q' to quit the webcam window
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break   

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
