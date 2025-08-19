import streamlit as st
import numpy as np
import cv2
from PIL import Image
import os
import gdown
from keras.models import load_model
from keras.preprocessing.image import img_to_array


MODEL_PATH = "emotion_model.h5"
MODEL_URL = "https://drive.google.com/uc?id=1aH3MFMHrEuUUFH0XZmiR09eOhkFYXQIU"  # Replace with your own

@st.cache_resource
def load_emotion_model():
    if not os.path.exists(MODEL_PATH):
        gdown.download(MODEL_URL, MODEL_PATH, quiet=False)
    model = load_model(MODEL_PATH)
    return model

model = load_emotion_model()
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

st.title("😊 Facial Emotion Detection (Streamlit)")
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file)
    img_np = np.array(image.convert('RGB'))
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi = gray[y:y+h, x:x+w]
        roi = cv2.resize(roi, (48, 48))
        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi)[0]
        emotion = emotion_labels[np.argmax(preds)]

        cv2.rectangle(img_np, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(img_np, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    st.image(img_np, caption="Detected Emotion", use_column_width=True)
