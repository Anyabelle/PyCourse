import numpy as np
from numpy import zeros
from math import sqrt
import pylab as plt

def crop(img, mode, mask):
    height, width, c = img.shape
    if (mode == 'vertical shrink' or mode == 'horizontal shrink'):
        resized_img = np.empty([height, width - 1, c])
    else:
        resized_img = np.empty([height, width + 1, c])
    if mask is None:
        resized_mask = None
    else:
        height, width = mask.shape
        resized_mask = np.empty(resized_img.shape)

    carve_mask = zeros(img.shape[0:2])
    height, width, c = img.shape
    height, width, c = img.shape
    yuv = img[:, :, 0] * 0.299 + img[:, :, 1] * 0.587 + img[:, :, 2] * 0.114
    a = np.empty([height, width])
    ppx = np.empty([height, width])
    ppy = np.empty([height, width])
    ppx[0, 0:width] = yuv[1, 0:width] - yuv[0, 0:width]
    ppx[height - 1, 0:width] = yuv[height - 1, 0:width] - yuv[height - 2, 0:width]
    ppy[0:height, 0] = yuv[0:height, 1] - yuv[0:height, 0]
    ppy[0:height, width - 1] = yuv[0:height, width - 1] - yuv[0:height, width - 2]
    ppx[1:height - 1, 0:width] = yuv[2:height, 0:width] - yuv[0:height - 2, 0:width]
    ppy[0:height, 1:width - 1] = yuv[0:height, 2:width] - yuv[0:height, 0:width - 2]
    a = (ppx ** 2 + ppy ** 2) ** (1.0 / 2)
    if (mask is not None):
        addm = 256 * width * height
        a = a + mask * addm
    grad = a
    fromg = zeros(img.shape[0:2])
    fromg.astype(int)
    for x in range (1, height):
        for y in range (0, width):
            add = grad[x - 1, y]
            fr = y
            if (y != 0):
                add = grad[x - 1, y - 1]
                fr = y - 1
            if (grad[x - 1, y]< add):
                add = grad[x - 1, y]
                fr = y
            if ((y != width - 1) and (grad[x - 1, y + 1] < add)):
                add = grad[x - 1, y + 1]
                fr = y + 1
            grad[x, y] += add
            fromg[x, y] = fr
    indmin = 0
    for y in range (1, width):
        if (grad[height - 1, y] < grad[height - 1, indmin]):
            indmin = y
    carve_mask[height - 1, indmin] = 1
    if (mode == 'vertical shrink' or mode == 'horizontal shrink'):      
        for y in range (width - 1, -1, -1):
            if (y == indmin):
                continue
            if (y > indmin):
                resized_img[height - 1, y - 1] = img[height - 1, y]
                if (mask is not None):
                    resized_mask[height - 1, y - 1] = mask[height - 1, y]
            else:
                resized_img[height - 1, y] = img[height - 1, y]
                if (mask is not None):
                    resized_mask[height - 1, y] = mask[height - 1, y]
    else:
        for y in range (width - 1, -1, -1):
            if (y == indmin + 1):
                if (y != width - 1):
                    resized_img[height - 1, y] = (img[height - 1, y] + img[height - 1, y + 1]) / 2
                else:
                    resized_img[height - 1, y] = img[height - 1, y]
                if (mask is not None):
                    resized_mask[height - 1, y] += addm
            if (y > indmin):
                resized_img[height - 1, y + 1] = img[height - 1, y]
                if (mask is not None):
                    resized_mask[height - 1, y + 1] = mask[height - 1, y]
            else:
                resized_img[height - 1, y] = img[height - 1, y]
                if (mask is not None):
                    resized_mask[height - 1, y] = mask[height - 1, y]
    for xn in range (height - 1, 0, -1):
        indmin = int(fromg[xn, indmin])
        carve_mask[xn - 1, indmin] = 1
        if (mode == 'vertical shrink' or mode == 'horizontal shrink'):      
            for y in range (width - 1, -1, -1):
                if (y == indmin):
                    continue
                if (y > indmin):
                    resized_img[xn - 1, y - 1] = img[xn - 1, y]
                    if (mask is not None):
                        resized_mask[xn - 1, y - 1] = mask[xn - 1, y]
                else:
                    resized_img[xn - 1, y] = img[xn - 1, y]
                    if (mask is not None):
                        resized_mask[xn - 1, y] = mask[xn - 1, y]
        else:
            for y in range (width - 1, -1, -1):
                if (y == indmin + 1):
                    if (y != width - 1):
                        resized_img[xn - 1, y] = (img[xn - 1, y] + img[xn - 1, y + 1]) / 2
                    else:
                        resized_img[xn - 1, y] = img[xn - 1, y]
                    if (mask is not None):
                        resized_mask[xn - 1, y] += addm
                if (y > indmin):
                    resized_img[xn - 1, y + 1] = img[xn - 1, y]
                    if (mask is not None):
                        resized_mask[xn - 1, y + 1] = mask[xn - 1, y]
                else:
                    resized_img[xn - 1, y] = img[xn - 1, y]
                    if (mask is not None):
                        resized_mask[xn - 1, y] = mask[xn - 1, y]
    return (resized_img, resized_mask, carve_mask)
    
def seam_carve(img, mode, mask=None):
    height, width, c = img.shape  
    if (mode == 'vertical shrink' or mode == 'vertical expand'):
        img = np.swapaxes(img,0,1)
        if (mask is not None):
            mask = np.swapaxes(mask,0,1)
    resized_img, resized_mask, carve_mask = crop(img, mode, mask)
    if (mode == 'vertical shrink' or mode == 'vertical expand'):
        resized_img = np.swapaxes(resized_img,0,1)
        if (mask is not None):
            resized_mask = np.swapaxes(resized_mask,0,1)
        carve_mask = np.swapaxes(carve_mask,0,1)  
    return (resized_img, resized_mask, carve_mask)