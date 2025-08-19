import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import gdown
from keras.models import load_model

# ------------------------------
# CONFIG
# ------------------------------
MODEL_PATH = "emotion_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1aH3MFMHrEuUUFH0XZmiR09eOhkFYXQIU"  # Replace with your own

# ------------------------------
# LOAD MODEL (cached in Streamlit)
# ------------------------------
@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = load_model(MODEL_PATH)
    return model

model = load_emotion_model()

# ------------------------------
# FACE DETECTOR + LABELS
# ------------------------------
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# ------------------------------
# STREAMLIT APP
# ------------------------------
st.title("😊 Facial Emotion Detection (Streamlit)")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and convert to RGB
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert('RGB'))

    # Convert to grayscale for face detection
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    # Process each face
    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = np.expand_dims(roi, axis=-1)   # (48, 48, 1)
        roi = np.expand_dims(roi, axis=0)    # (1, 48, 48, 1)

        # Predict emotion
        preds = model.predict(roi)[0]
        emotion = emotion_labels[np.argmax(preds)]

        # Draw results on image
        cv2.rectangle(img_np, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_np, emotion, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # Show output
    st.image(img_np, caption="Detected Emotion(s)", use_column_width=True)
