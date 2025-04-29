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
labels = ["Female", "Male"]

# Initialize OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# --- NEW: Load image ---
image_path = "test5.png"  # <<< Change this to your image path
image = cv2.imread(image_path)

if image is None:
    print(f"Error: Could not load image at {image_path}")
    exit()

# Process image
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=6, minSize=(50, 50))

if len(faces) == 0:
    print("No faces detected.")
else:
    for (x, y, w, h) in faces:
        face = image[y:y + h, x:x + w]
        face = cv2.resize(face, (128, 128))
        face = img_to_array(face) / 255.0
        face = np.expand_dims(face, axis=0)

        print(f"Processing face at {x}, {y}, width: {w}, height: {h}")  # Debugging

        # Safety check
        try:
            prediction = float(model(face).numpy()[0][0])
            gender = labels[int(round(prediction))]
        except Exception as e:
            print(f"Prediction error: {e}")
            continue

        color = (0, 255, 0) if gender == "Male" else (255, 0, 255)
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, gender, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

# --- NEW: Save the result image ---
output_path = "output_gender_detection.jpg"
cv2.imwrite(output_path, image)
print(f"Output image saved as {output_path}")

# Optionally show the image
cv2.imshow("Gender Detection Result", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
