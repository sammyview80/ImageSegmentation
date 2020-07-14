import tensorflow as tf 
from data.getdata import getDataSetAsDir, getDateSetAsArray
from models.simpleUnet import get_unet


TRAINPATH = './DataSet/trainds'
TESTPATH = './DataSet/testds1'
VALIDATIONPATH = './DataSet/testds2'

def getValData():
    valImagesDir = getDataSetAsDir(VALIDATIONPATH)
    testImagesDir = getDataSetAsDir(TESTPATH)

    valImages = getDateSetAsArray(valImagesDir, image=True)
    testImages = getDateSetAsArray(testImagesDir, image=True)

    print('ValidationImages Shapes:{}'.format(valImages.shape))
    print('TestImages Shapes:{}'.format(testImages.shape))

    return valImages, testImages

def getTrainData():
    trainImagesDir, trainMasksDir = getDataSetAsDir(TRAINPATH)

    trainImages = getDateSetAsArray(trainImagesDir, image=True)
    trainMasks = getDateSetAsArray(trainMasksDir, mask=True)
    

    print('TrainImages Shapes: {}\nTrainMasks Shapes: {}'.format(trainImages.shape, trainMasks.shape))

    return trainImages, trainMasks
    

def modelFit(plot_model=False, summary=False, save=True):
    trainImages, trainMasks = getTrainData()
    model = get_unet(input_shape=(96, 96, 3), n_filters=12, dropout=0.5, batchNormal=True)
    model.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(trainImages, trainMasks, verbose=True, epochs=2)

    if summary:
        print(model.summary())
    if plot_model:
        tf.keras.utils.plot_model(model, to_file='simpleUnet.png', show_shapes=True)
    if save:
        model.save('unet.h5')
    return model

def load_model(path):
    return tf.keras.models.load_model(path)




