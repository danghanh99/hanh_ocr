import cv2
from matplotlib import pyplot as plt    
import numpy as np
import imutils
import easyocr


def bw_img(img):
    thresh, im_bw = cv2.threshold(img, 200, 230, cv2.THRESH_BINARY)
    cv2.imwrite("opencv_img/bw_img.jpg", im_bw)
    return im_bw

def gray_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def show_img(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

img = cv2.imread('dongnai.png')
gray = gray_img(img)
img2 = bw_img(gray)
show_img(img2)


