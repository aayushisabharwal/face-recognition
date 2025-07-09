import os
import cv2 as cv
import numpy as np

# List of people (labels)
people = ['Aayushi Sabharwal' , 'Ben afflek', 'elton john', 'jerry seinfield', 'madonna', 'mindy kaling']

# Haar cascade for face detection
haar_cascade = cv.CascadeClassifier('haarcascade_frontalface_default.xml')

# Directory containing face folders
DIR = r'C:\Users\INTEL\Desktop\OPENCV\faces'

# Data containers
features = []  # Cropped face images
labels = []    # Corresponding numeric labels

def create_train():
    for person in people:
        path = os.path.join(DIR, person)
        label = people.index(person)

        for img in os.listdir(path):
            img_path = os.path.join(path, img)

            img_array = cv.imread(img_path)
            if img_array is None:
                print(f"[WARNING] Failed to read image: {img_path}. Skipping.")
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

            for (x, y, w, h) in faces_rect:
                faces_roi = gray[y:y+h, x:x+w]
                features.append(faces_roi)
                labels.append(label)

create_train()
print('Training data collection done --------------')

# Convert to numpy arrays
features = np.array(features, dtype=object)
labels = np.array(labels, dtype=np.int32)

# Create LBPH face recognizer
face_recognizer = cv.face.LBPHFaceRecognizer_create()

# Train recognizer on features and labels
face_recognizer.train(features, labels)

# Save the trained model and data
face_recognizer.save('face_trained.yml')
np.save('features.npy', features)
np.save('labels.npy', labels)

print('Training complete. Model and data saved.')