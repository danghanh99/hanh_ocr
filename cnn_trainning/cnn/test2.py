from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import cv2
import matplotlib.pyplot as plt
# dimensions of our images
img_width, img_height = 320, 240

# load the model we saved
model = load_model('my_model.h5')

img = cv2.imread('so0.png')
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()