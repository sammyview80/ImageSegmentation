import cv2 as cv 
import numpy as np
import matplotlib.pyplot as plt

def readImage(path):
    image = cv.imread(path, 1)
    image = cv.resize(image, (96, 96))
    return image

def addMasks(masks):
    addImage = np.zeros((96, 96, 1), dtype=np.uint8)
    for maskIds in range(0, len(masks)):
        image = cv.imread(masks[maskIds], 0)
        image = cv.resize(image, (96, 96))
        image = cv.add(addImage, image)
        addImage = image
    return addImage

def showImage(image):
    plt.imshow(image, plt.cm.binary)
    plt.show()

def normalize(ds):
    DS = []
    for i in ds:
        DS.append(i/255)
    return DS

def reshapeMasks(masks):
    reshaped = []
    for i in masks:
        i = np.expand_dims(i, axis=3)
        reshaped.append(i)

    return reshaped