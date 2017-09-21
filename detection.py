import numpy as np
import os
from os.path import join
from skimage.io import imread
from skimage.transform import resize
from keras import metrics
from keras.models import load_model, Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
import matplotlib.pyplot as plt
import scipy.misc
from scipy.misc import imresize



size1 = 40
size2 = 40

def train_detector(train_gt, train_img_dir, fast_train=True):
    ep = 50
    np.random.seed(22)
    model = Sequential()

    model.add(Convolution2D(16, (5, 5), input_shape = (size1, size2, 1), activation='tanh'))    
    #model.add(Convolution2D(64, (3, 3), activation = "relu"))
    model.add(MaxPooling2D(pool_size=(2,2)))
    #model.add(BatchNormalization())
    model.add(Dropout(0.2))
    model.add(Convolution2D(48, (3, 3), activation='tanh'))
    #model.add(Convolution2D(48, (2, 2), activation='tanh'))
    #model.add(Convolution2D(96, (2, 2)))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())

    model.add(Dropout(0.3))
    model.add(Convolution2D(64, (3, 3), activation='tanh'))
    #model.add(Convolution2D(64, (2, 2), activation='tanh'))
    #model.add(Convolution2D(128, (2, 2), activation='tanh'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    #model.add(BatchNormalization())

    model.add(Convolution2D(64, (2, 2), activation='tanh'))

    #model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dropout(0.4))
    #model.add(Dense(300, activation= 'tanh', kernel_initializer="normal"))
    model.add(Dense(28, activation = 'linear'))
    print(model.summary())
    model.compile(lr=0.1, loss="mean_squared_error", optimizer="adam", momentum = 0.99)    

    cdlist = []
    imglist = []
    
    flist = os.listdir(train_img_dir)
       
    for file in flist:
        img = imread(join(train_img_dir, file))
        coords = train_gt[file]
        print(coords)
        for j in range (coords.shape[0]):
            if (j % 2 == 0):
                coords[j] /= img.shape[0]
            else:
                coords[j] /= img.shape[1]
        cdlist.append(coords)
        if (len(img.shape) != 2):
            img = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        img = imresize(img, (size1, size2))
        img = np.float32(img)
        img = norm(img)  
        imglist.append(img) 
        
    imgtrain = np.asarray(imglist)
    cdtrain = np.asarray(cdlist)
    imgtrain = imgtrain.astype('float32')
    
    datagen = ImageDataGenerator()
    train_generator = datagen.flow(imgtrain, cdtrain, batch_size=50)

    if (fast_train == True):
        ep = 1
    model.fit_generator(train_generator, steps_per_epoch=140,epochs=ep)
    return model


def detect(model, test_img_dir):
    
    imglist = []
    namelist = []
    shapelist = []
    flist = os.listdir(test_img_dir)
    
    for file in flist:
        img = imread(join(test_img_dir, file))
        shapelist.append(img.shape)
        namelist.append(file)
        if (len(img.shape) != 2):
            img = 0.299 * img[:,:,0] + 0.587 * img[:,:,1] + 0.114 * img[:,:,2]
        img = imresize(img, (size1, size2))
        img = np.float32(img)
        img = norm(img)
        imglist.append(img)
    imgtest = np.asarray(imglist)
    #imgtest = imgtest.reshape(imgtest.shape[0], imgtest.shape[1] * imgtest.shape[2] * imgtest.shape[3])
    imgtest = imgtest.astype('float32')
    predictions = model.predict(imgtest)
    res = {}
    for i in range (len(namelist)):
        res[namelist[i]] = predictions[i]
        if (i % 2 == 0):
            res[namelist[i]] *= shapelist[i][0]
        else:
            res[namelist[i]] *= shapelist[i][1]
    return res

def norm(img):
    img = prevnorm(img)
    return img
def prevnorm(img):
    mean = img.sum() / (size1 * size2)
    dispers = (img * img).sum() / (size1 * size2) - mean ** 2
    if (dispers == 0):
        dispers = 1
    img = (img - mean) / dispers
    return img.reshape(size1, size2, 1)
