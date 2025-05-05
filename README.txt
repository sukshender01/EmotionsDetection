
Emotion Detection Package
--------------------------
Files included:
- emotion_detection_image.py: Run emotion detection on a static image (person.jpg)
- emotion_detection_webcam.py: Run real-time emotion detection using your webcam
- emotion_model.h5: Download this from the official GitHub repo (see below)
- person.jpg: Place a sample image with a face here

How to run:
1. Install required packages:
   pip install opencv-python keras tensorflow numpy

2. Download model file:
   https://github.com/oarriaga/face_classification/blob/master/trained_models/emotion_models/fer2013_mini_XCEPTION.102-0.66.hdf5

   Save it as 'emotion_model.h5' in the same folder.

3. Run the desired script:
   python emotion_detection_image.py
   OR
   python emotion_detection_webcam.py

Press 'q' to exit webcam mode.
