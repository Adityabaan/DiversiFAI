import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
import os

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  

# Load trained model
model = tf.keras.models.load_model("gender_model_diversifai_v1.h5")

# Define labels
labels = ["Male", "Female"]  

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face = frame[y:y + h, x:x + w]
        face = cv2.resize(face, (128, 128))
        face = img_to_array(face) / 255.0
        face = np.expand_dims(face, axis=0)

        print(f"Processing face at {x}, {y}, width: {w}, height: {h}")  # Debugging

        # Safety check
        try:
            prediction = float(model(face).numpy()[0][0])  # Converts to a single float
            gender = labels[int(round(prediction))]
        except Exception as e:
            print(f"Prediction error: {e}")
            continue  # Skip faulty predictions

        color = (0, 255, 0) if gender == "Male" else (255, 0, 255)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.imshow("DiversiFAI - Gender Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
