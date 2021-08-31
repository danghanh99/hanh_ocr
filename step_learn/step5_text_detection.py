import cv2
from matplotlib import pyplot as plt    
import numpy as np
import imutils
import easyocr 
import pytesseract

def invert_img(img):
    inverted_img = cv2.bitwise_not(img)
    cv2.imwrite("opencv_img/step1_invert.jpg", inverted_img)
    return inverted_img

def bw_img(img):
    thresh, im_bw = cv2.threshold(img, 180, 250, cv2.THRESH_BINARY)
    cv2.imwrite("opencv_img/step3_bw_img.jpg", im_bw)
    return im_bw

def gray_img(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def show_img(img):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

def rm_noise_img(img):
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.erode(img, kernel, iterations=1)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    img = cv2.medianBlur(img, 3)
    cv2.imwrite("opencv_img/step4_no_noise.jpg", img)
    return img

def draw_box(img, default_img):
    reader = easyocr.Reader(['vi'])
    result = reader.readtext(img)
    for detection in result:
        print(detection[1])
        cv2.putText(default_img, text=detection[1], org=(int(detection[0][0][0]), int(detection[0][0][1])), fontFace=cv2.FONT_HERSHEY_COMPLEX, fontScale=1, color=(255, 0, 0), thickness=2, lineType=cv2.LINE_AA)
        cv2.rectangle(default_img, (int(detection[0][0][0]), int(detection[0][0][1])), (int(detection[0][2][0]), int(detection[0][2][1])), (255, 0, 0), 2)
    return default_img

# def print_easyocr(img):
#     reader = easyocr.Reader(['en'])
#     result = reader.readtext(img)
#     for detection in result:
#         print(detection[1])


img1 = cv2.imread('opencv_img/no_noise_dongnai.jpg')

# img = gray_img(img1)
# img = bw_img(img)
# img = rm_noise_img(img)
# img = rm_noise_img(img)
# img = rm_noise_img(img)
# cv2.imwrite("opencv_img/no_noise_soctrang.jpg", img)

pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
print(pytesseract.image_to_string(img1))
show_img(img1)

