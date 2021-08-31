import numpy as np
import cv2
import pickle
from keras.models import load_model
######
width = 640
height = 480
######

# cap = cv2.VideoCapture(1)
# cap.set(3, width)
# cap.set(4, height)

model = load_model("my_model.h5")

# def preProcessing(img):
#     img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img = cv2.equalizeHist(img)
#     img = img/255
#     return img

# while True:
#     success, imgOriginal = cap.read()
#     img = np.asarray(imgOriginal)
#     img = cv2.resize(img, (320, 320))
#     img = preProcessing(img)
#     cv2.imshow("Processed Image", img)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break