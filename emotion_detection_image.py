
import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array

model = load_model('emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

image_path = 'person.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_classifier.detectMultiScale(gray, 1.3, 5)

for (x, y, w, h) in faces:
    face = gray[y:y+h, x:x+w]
    face = cv2.resize(face, (48, 48))
    face = face.astype('float') / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    preds = model.predict(face)[0]
    emotion = emotion_labels[np.argmax(preds)]
    cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
    cv2.putText(image, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)

cv2.imshow('Emotion Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
