from typing import Sequence
from matplotlib import image
import numpy as np
import cv2
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import adam_v2
from keras.layers import Dropout, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
import pickle
from keras.models import load_model
################
path = 'myData'
# pathLabels = 'labels.csv'
testRatio = 0.2
valRatio = 0.2
imageDimensions = (32, 32, 3)
# ############
count = 0
images = []
classNu = []
myList = os.listdir(path)
print("Total num of classes detected: ", len(myList))
nuOfClasses = len(myList)
print("Importing Classes: ", end="")
for x in range (0, nuOfClasses):
    myPicList = os.listdir(path + "/" + str(x))
    # print(myPicList)
    for y in myPicList:
        curImg = cv2.imread(path + "/" + str(x) + "/" + y)
        curImg = cv2.resize(curImg , (imageDimensions[0], imageDimensions[1]))
        images.append(curImg)
        classNu.append(x)
    print(x, end=" ")
    count+=1
print(" ")
print("total images in images list: ", len(images))
print("total IDS in classNu list: ", len(classNu))

images = np.array(images)
classNu = np.array(classNu)
print("images.shape: ", images.shape)


# ## spliting the data
X_train, X_test, y_train, y_test = train_test_split(images, classNu, test_size=testRatio)
X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=valRatio)
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
print("X_validation: ", X_validation.shape)

numOfSample = []
for x in range(0, nuOfClasses):
    # print(len(np.where(y_train==x)[0]))
    numOfSample.append(len(np.where(y_train==x)[0]))
print(numOfSample)

plt.figure(figsize=(10, 5))
plt.bar(range(0, nuOfClasses), numOfSample)
plt.title("Nu of images for each class")
plt.xlabel("Class ID")
plt.ylabel("Number of Images")
# plt.show()

def preProcessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

img = preProcessing(X_train[30])
img = cv2.resize(img, (300, 300))
# cv2.imshow("preProcessed", img)
# cv2.waitKey(0)

X_train = np.array(list(map(preProcessing, X_train)))
X_test = np.array(list(map(preProcessing, X_test)))
X_validation = np.array(list(map(preProcessing, X_validation)))

X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_validation = X_validation.reshape(X_validation.shape[0], X_validation.shape[1], X_validation.shape[2], 1)

dataGen = ImageDataGenerator(width_shift_range= 0.1,
                            height_shift_range=0.1,
                            zoom_range=0.1,
                            shear_range=0.2,
                            rotation_range=10)

dataGen.fit(X_train)

y_train = to_categorical(y_train, nuOfClasses)
y_test = to_categorical(y_test, nuOfClasses)
y_validation = to_categorical(y_validation, nuOfClasses)

def myModel():
    nuOfFilters = 60
    sizeOfFilter1 = (5, 5)
    sizeOfFilter2 = (3, 3)
    sizeOfPool = (2, 2)
    nuOfNode = 500

    model = Sequential()
    model.add((Conv2D(nuOfFilters, sizeOfFilter1, input_shape= (imageDimensions[0], imageDimensions[1], 1), activation= 'relu')))
    model.add((Conv2D(nuOfFilters, sizeOfFilter1, activation= 'relu')))
    model.add(MaxPooling2D(pool_size= sizeOfPool))
    model.add((Conv2D(nuOfFilters//2, sizeOfFilter2, activation= 'relu')))
    model.add((Conv2D(nuOfFilters//2, sizeOfFilter2, activation= 'relu')))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(nuOfNode, activation= 'relu'))
    model.add(Dense(nuOfClasses, activation= 'softmax'))
    model.compile(adam_v2.Adam(learning_rate=0.001), loss='categorical_crossentropy',
                metrics=['accuracy'])
    return model

model = myModel()
print(model.summary())

###########################
batchSizeVal = 32
epochsVal = 10
stepsPerEpochVal = len(X_train)//batchSizeVal
#############################

history = model.fit(dataGen.flow(X_train, y_train, batch_size= batchSizeVal),
                    steps_per_epoch= stepsPerEpochVal,
                    epochs= epochsVal,
                    validation_data=(X_validation, y_validation),
                    shuffle=1)
print(history.history)
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['training', 'validation'])
plt.title("Loss")
plt.xlabel("epoch")

plt.figure(2)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'validation'])
plt.title("Accuracy")
plt.xlabel("epoch")

# plt.show()

score = model.evaluate(X_test, y_test, verbose= 0)
print('Test score = ', score[0])
print('Test accuracy = ', score[1])

# filename = 'hanh_model.sav'
# pickle.dump(model, open(filename, 'wb'))

model.save('hanh_model_l4.h5')
# model1 = load_model('my_model.h5')
