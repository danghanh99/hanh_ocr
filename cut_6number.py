import cv2
from matplotlib import pyplot as plt    
import numpy as np
import imutils
# import easyocr
import numpy as np
from keras.models import load_model
import math
from operator import itemgetter
from numpy.lib.function_base import average
def cut6number(img):
    (h, w, d) = img.shape
    h2 = h*0.67
    roi = img[int(h2):h, 0:w] #[y1:y2, x1:x2]
    (roi_h, roi_w, roi_d) = roi.shape
    roi2 = roi[0:roi_h, int(roi_w*0.2):roi_w] #[y1:y2, x1:x2]
    return roi2

def detection_number(img, model):
    img = cv2.resize(img, (32, 32))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    img = img.reshape(1, 32, 32, 1)

    a = np.argmax(model.predict(img), axis=-1)
    predictions = model.predict(img)
    val = np.amax(predictions)
    return [a[0], val]

def calculateDistance(x1,y1,x2,y2):
    dist = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    return dist

def make_box(box):
    x_arr = []
    y_arr = []
    for point in box:
        x_arr.append(point[0])
        y_arr.append(point[1])
    # x_arr = sorted(x_arr)
    # y_arr = sorted(y_arr)
    return [min(x_arr), min(y_arr)], [max(x_arr), max(y_arr)]

def is_similarity(first_box, second_box):
    count = 0
    for point in second_box:
        if(point in first_box):
            count += 1
    if(count >= 2): return True
    return False

model = load_model('cnn_trainning/hanh_model_l4.h5')
# img = cv2.imread('opencv_img/no_noise_binhthuan.jpg')
# img = cv2.imread('opencv_img/no_noise_danang.jpg')
img = cv2.imread('opencv_img/no_noise_dongnai.jpg')
cropped_img = cut6number(img)
# cv2.imwrite("detected_imgs/hinh2.png", cropped_img)
# cropped_img = cv2.imread('detected_imgs/hinh2.png')
gray = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
canny = cv2.Canny(gray, 130, 255, 1)
cnts = cv2.findContours(canny, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# cv2.drawContours(cropped_img, cnts, -1, (0,255,0), 3)

boxs = []
crosses = []

def box_img(left_top, right_bottom, img):
    image_box = img[left_top[0]:right_bottom[0], left_top[1]:right_bottom[1]] #x1:x2, y1,y2
    return image_box

def unique_arr(arr):
    uni_arr = []
    for e in arr:
        if e not in uni_arr:
            uni_arr.append(e)
    return uni_arr

def check(boxes, box):
    for b in boxes:
        if np.array_equal(np.asarray(b), np.asarray(box)):
            return True

    return False

def uni(ar):
    newlist = []
    for item in ar:
         if item not in newlist:
             newlist.append(item)
    return newlist

def cutCountour(img, path, con):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    idx = 0 # The index of the contour that surrounds your object
    mask = np.zeros_like(img) # Create mask where white is what we want, black otherwise
    cv2.drawContours(mask, [con], idx, (255, 0, 0), -1) # Draw filled contour in mask
    out = np.zeros_like(img) # Extract out the object and place into output image
    out[mask == 255] = img[mask == 255]

    # Now crop
    (y, x) = np.where(mask == 255)
    (topy, topx) = (np.min(y), np.min(x))
    (bottomy, bottomx) = (np.max(y), np.max(x))
    out = out[topy:bottomy+1, topx:bottomx+1]

    # Show the output image
    # cv2.imshow('Output', out)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite(path, out)
    return out

new_a = []
count = 0
area_a = []

for c in cnts:
    rect = cv2.minAreaRect(c)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    aa = cv2.contourArea(box)
    
    if(aa not in area_a):
        if(len(new_a) > 1):
            if(is_similarity(new_a[len(new_a)-1][0], box) == False):
                new_a.append([box, aa])
                area_a.append(aa)
        else:
            new_a.append([box, aa])
            area_a.append(aa)

new_a = sorted(new_a,key=itemgetter(1))

count = 0
for a in new_a[len(new_a) - 6:len(new_a)]:
    box_cou= a[0]
    box = make_box(box_cou)
    left_top = box[0] #x:y
    right_bottom = box[1] #x:y
    image_box = cropped_img[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]] # y1,y2, x1:x2
    # print(image_box.shape)
    path = "image_boxs/hinh{}.png".format(count)
    img_cut_con = cutCountour(cropped_img, path, box_cou)
    [val, percent] = detection_number(img_cut_con, model)
    count+=1
    if(percent > 0):
        instr = "{}_{:0.0f}%".format(val, percent*100)
        print(instr)
        print(box_cou)
        print("---------------------")
        cv2.drawContours(cropped_img, [box_cou], 0, (0,255,255), 3)
        cv2.putText(cropped_img, str(instr), (left_top[0], left_top[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 1, cv2.LINE_AA)

# boxs = sorted(boxs,key=itemgetter(1))
# print("len1: ", len(boxs))
# boxs=np.unique(boxs,axis=0) #remove duplicate
# print("len2: ", len(boxs))

# no_dup_boxs = [] # box which no duplicate
# for box in boxs:
#     left_top = box[0] #x:y
#     right_bottom = box[1] #x:y
#     cross = calculateDistance(left_top[0], left_top[1], right_bottom[0], right_bottom[1]) #x1,y1,x2,y2

#     image_box = cropped_img[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]] # y1,y2, x1:x2
#     (h, w, d) = image_box.shape
#     if(h*w >0): 
#         no_dup_boxs.append([box, int(cross)])

# print("")
# print("len3 new box: ", len(no_dup_boxs))
# sorted_boxs = sorted(no_dup_boxs,key=itemgetter(1)) # box which sorted by cross like 1-2-5-9...


# def rectangle_area(left_top, right_bottom):
#     w = right_bottom[0] - left_top[0]
#     h = right_bottom[1] - left_top[1]
#     return abs(h*w)

# new_boxs = []
# for sorted_box in  sorted_boxs[len(sorted_boxs) - 20:len(sorted_boxs)]:
#     # box = sorted_box[0]
#     # left_top = box[0] #x:y
#     # right_bottom = box[1] #x:y
#     # image_box = cropped_img[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]] # y1,y2, x1:x2
#     # [val, percent] = detection_number(image_box, model)
#     # if(percent > 0.5):
#     #     print("val {} {}".format(val, percent))
#     #     print(image_box.shape)
#     #     cv2.rectangle(cropped_img, left_top, right_bottom, (0,255,0), 3)
#     #     cv2.putText(cropped_img, str(val), (left_top[0], left_top[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
#     #     path = "image_boxs/hinh{}.png".format(val)
#     #     cv2.imwrite(path, image_box)

#     box = sorted_box[0]
#     left_top = box[0] #x:y
#     right_bottom = box[1] #x:y
#     image_box = cropped_img[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]] # y1,y2, x1:x2
#     [val, percent] = detection_number(image_box, model)
#     if(percent > 0.4):
#         new_boxs.append([box, int(rectangle_area(left_top, right_bottom))])
#         # cv2.rectangle(cropped_img, left_top, right_bottom, (0,255,0), 3)

# new_boxs = sorted(new_boxs,key=itemgetter(1))
# # print(new_boxs)
# count = 0
# for new_box in  new_boxs[len(new_boxs) - 10:len(new_boxs)]:
#     box = new_box[0]
#     left_top = box[0] #x:y
#     right_bottom = box[1] #x:y
#     image_box = cropped_img[left_top[1]:right_bottom[1], left_top[0]:right_bottom[0]] # y1,y2, x1:x2
#     [val, percent] = detection_number(image_box, model)
#     if(percent > 0.4):
#         instr = "{}_{:0.2f}_{}".format(val, percent, count)
#         print(instr)
#         cv2.rectangle(cropped_img, left_top, right_bottom, (0,255,0), 3)
#         count += 1
#         cv2.putText(cropped_img, str(instr), (left_top[0], left_top[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)


cv2.imwrite("detected_imgs/hinh3.png", cropped_img)
plt.imshow(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
plt.show()