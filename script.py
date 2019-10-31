import cv2
import numpy as np
import os, os.path


def face_segment(path):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faceCascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=3,
        minSize=(30, 30)
    )
    mask = np.zeros(image.shape, np.uint8)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]
        cv2.imwrite('./out/segmented_' + path, mask )

VALID_IMAGE_EXTS = [".jpg", ".png", ]
if __name__ == "__main__":
    for img_path in os.listdir('./'):
        ext = os.path.splitext(img_path)[1]
        if ext.lower() in VALID_IMAGE_EXTS:
            face_segment('' + img_path)
            
