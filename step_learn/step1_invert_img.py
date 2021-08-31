import cv2
from matplotlib import pyplot as plt    
import numpy as np
import imutils
# import easyocr
import numpy as np
from keras.models import load_model
def invert_img(img):
    inverted_img = cv2.bitwise_not(img)
    cv2.imwrite("opencv_img/invert_no_noise_quangnam.jpg", inverted_img)
    return inverted_img

img = cv2.imread('opencv_img/no_noise_quangnam.jpg')
img = invert_img(img)
