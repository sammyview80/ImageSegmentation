import cv2 as cv 
import numpy as np
from data.dataUtils import readImage, showImage
from main import load_model


TESTIMAGE = './DataSet/test.png'
MODEL = load_model('./Demo/unet.h5')

def predict(imagePath):
    image = preprocess(imagePath)
    image = image[np.newaxis, :, :, :]
    print(image.shape)
    prediction = MODEL.predict(image)
    showImage(prediction.squeeze())

    return image
    

def preprocess(image):
    image = readImage(image)
    print(image.shape)
    image = image/255
    return image


predict(TESTIMAGE)