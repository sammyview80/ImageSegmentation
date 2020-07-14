import os 
import numpy as np
from .dataUtils import readImage, normalize, reshapeMasks, addMasks

def getDataSetAsDir(Dir):
    """
    Get the directory and return the mask and the images.
    Dir: Path of the train or test set.

    return: images, masks
    """

    masks = []
    images = []
    for roots, _, files in os.walk(Dir):
        list_roots = roots.split('/')
        if list_roots[-1] == 'masks':
            files = [os.path.join(roots, file) for file in files]
            masks.append((files))
        if list_roots[-1] == 'images':
            files = [os.path.join(roots, file) for file in files]
            images.append((files))
    if len(masks) != 0:
        return images, masks
    else:
        return images

def getDateSetAsArray(imagePath, mask=False, image=False):
    """
    Get the imagePath, read image and return list of array of images.
    imagePath: list from getDataSetAsDir()

    return: list of image array.
    """
    images = []
    if image:
        print('Images:')
        for image in imagePath:
            image = readImage(image[0]) # image is a list. image[0] selecting the 1 element of list
            images.append(image)

    if mask:
        print('Masks:')
        for mask in imagePath:
            image = addMasks(mask)
            images.append(image)
        images = reshapeMasks(images)

    images = np.array(normalize(images))
    return images