import numpy as np
from math import atan2, pi
from sklearn import svm
import scipy.misc

def extract_hog(img1):
    img1 = scipy.misc.imresize(img1, (80, 80))
    img = np.float64(img1)
    height, width = img.shape[0:2]
    blocksz = 2
    cellsz = 8
    bincnt = 9
    mod = np.empty([height, width], np.float64)
    angle = np.empty([height, width], np.float64)
    ppx = np.zeros((height, width, 3), np.float64)
    ppy = np.zeros((height, width, 3), np.float64)
       
    edgeimg = np.zeros([height + 2, width + 2, 3], np.float64)
    edgeimg[0, 0, :] = img[0, 1, :]
    edgeimg[height + 1, 0, :] = img[height - 1, 1, :]
    edgeimg[0, width + 1,:] = img[1, width - 1, :]
    edgeimg[height + 1, width + 1, :] = img[height - 2, width - 1, :]
    
    edgeimg[height + 1, 1: - 1, :] = img[height - 1, :, :]
    edgeimg[0, 1: width + 1, :] = img[0, :, :]
    edgeimg[1: height + 1, width + 1, :] = img[:, width - 1, :]
    edgeimg[1 : height + 1, 0, :] = img[:, 0, :]  
    
    edgeimg[1: height + 1, 1: width + 1, :] = img[:, :, :]
        
    ppx[:, :, :] = edgeimg[2:, 1: width + 1, :] - edgeimg[0: height, 1: width + 1, :]
    ppy[:, :, :] = edgeimg[1: height + 1, 2:, :] - edgeimg[1: height + 1, 0: width, :]
    
    mod = np.empty([height, width, 3], np.float64)
    modm = np.empty([height, width], np.float64)
    mod = (ppx ** 2 + ppy ** 2) ** (1.0 / 2)
    ind = np.argmax(mod, 2)
    for x in range (height):
        for y in range (width):
            maxi = ind[x, y]
            modm[x, y] = mod[x, y, maxi]
            angle[x, y] = atan2(ppy[x, y, maxi], ppx[x, y, maxi])
    angle[angle < 0] += pi          
         
    bins = np.zeros([height // cellsz, width // cellsz, bincnt], np.float64)
    for row in range (bins.shape[0]):
        for col in range (bins.shape[1]):
            rr = row * cellsz
            cc = col * cellsz        
            for x in range(cellsz):
                for y in range(cellsz):
                    ang = angle[rr + x, cc + y] / pi * bincnt
                    num = int(ang)
                    if (ang == int(ang)):
                        pr = 1
                    else:
                        pr = ang - int(ang)
                    if (num == bincnt):
                        num = 0
                    bins[row, col, num] += (1 - pr) * modm[rr + x][cc + y]
                    if (num != bincnt - 1): 
                        bins[row, col, num + 1] += pr * modm[rr + x][cc + y]
                    else:
                        bins[row, col, 0] += pr * modm[rr + x][cc + y]
    height = bins.shape[0] - 1
    width = bins.shape[1] - 1
    blocks = np.empty([height, width, blocksz * blocksz * bincnt], np.float64)
    for row in range (blocks.shape[0]):
        for col in range (blocks.shape[1]):
            for num in range (bincnt):
                for num1 in range (2):
                    for num2 in range (2):
                        blocks[row, col, bincnt * (2 * num1 + num2) + num] = bins[row + num1, col + num2, num]
            modu = (sum(blocks[row, col, :] ** 2)) ** (1.0 / 2) + 1e-6
            blocks[row, col, :] /= modu
    vector = np.empty([blocks.size], np.float64)

    sz0 = blocks.shape[0]
    sz1 = blocks.shape[1]
    sz2 = blocks.shape[2]
    for row in range (sz0):
        for col in range (sz1):
            for num in range (sz2):
                vector[row + sz0 * (col + sz1 * num)] = blocks[row, col, num]
    return vector
    

def fit_and_classify(train_features, train_labels, test_features):
    classif = svm.LinearSVC()
    classif.fit(train_features, train_labels)
    predict = classif.predict(test_features)
    return predict