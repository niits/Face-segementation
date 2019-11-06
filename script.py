import cv2
import numpy as np
import os, os.path

LOWER = np.array([0, 50, 60], dtype = "uint8")
UPPER = np.array([13, 150, 255], dtype = "uint8")
VALID_IMAGE_EXTS = [".jpg", ".png", ]

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
        mask[y:y+h, x:x+w] = image[y:y+h, x:x+w]
    converted = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
    skinMask = cv2.inRange(converted, LOWER, UPPER)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 9))
    skinMask = cv2.erode(skinMask, kernel, iterations = 3)
    skinMask = cv2.dilate(skinMask, kernel, iterations = 5)
    skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
    skin = cv2.bitwise_and(image, image, mask = skinMask)
    cv2.imwrite('./out/segmented_' + path, skin )

if __name__ == "__main__":
    for img_path in os.listdir('./'):
        ext = os.path.splitext(img_path)[1]
        if ext.lower() in VALID_IMAGE_EXTS:
            face_segment(img_path)