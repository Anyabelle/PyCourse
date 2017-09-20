import numpy as np
import scipy.ndimage
from skimage.io import imread, imsave

def count(imgn, imgc, x1, x2, y1, y2):
    minn = 100
    minx1 = 0
    miny1 = 0
    for mx in range(x1, x2):
        for my in range (y1, y2):
            now = 0
            cpyimgn = np.roll(imgn, mx, axis = 0);
            cpyimgn = np.roll(cpyimgn, my, axis = 1)
            now = np.sum((imgc[15:,:-15].astype("float") - cpyimgn[15:,:-15].astype("float")) ** 2)
            now /= np.float64(imgc[15:, -15].shape[0] * cpyimgn[15:, :-15].shape[1])
            if (now < minn):
                minn = now
                minx1 = mx
                miny1 = my
    return (minx1, miny1)

def align(bgr_image, g_coord):  
    g_row, g_col = g_coord  
    img = bgr_image
    height, width = img.shape
    height -= height % 3;
    img = img[:height, :]
    #split into 3
    img1, img2, img3 = np.split(img, 3)
    #slice img1
    height, width = img1.shape
    cut_h = height // 20
    cut_w = width // 20
    heightwas = height
    img1 = img1[cut_h:height - cut_h, cut_w:width - cut_w]
    #slice img2
    g_row -= heightwas
    img2 = img2[cut_h:height - cut_h, cut_w:width - cut_w]
    #slice img3
    img3 = img3[cut_h:height - cut_h, cut_w:width - cut_w]
    #Reducing size
    lst1 = []
    lst2 = []
    lst3 = []
    cnt = 0
    maxn = 100
    while (1):
        a, b = img1.shape
        if (a > 5 * maxn or b > 5 * maxn):
            lst1.append(img1)
            lst2.append(img2)
            lst3.append(img3)
            height, width = img1.shape
            img1 = scipy.ndimage.zoom(img1, 0.5, order = 0) 
            img2 = scipy.ndimage.zoom(img2, 0.5, order = 0) 
            img3 = scipy.ndimage.zoom(img3, 0.5, order = 0) 
            cnt += 1
        else:
            break   
    #count move
    minx1, miny1 = count(img3, img2, -15, 16, -15, 16)
    minx2, miny2 = count(img1, img2, -15, 16, -15, 16)
    #go up
    while (cnt > 0):
        cnt -= 1
        img1 = lst1[cnt]
        img2 = lst2[cnt]
        img3 = lst3[cnt]
        minx1 *= 2
        miny1 *= 2
        minx2 *= 2
        miny2 *= 2
        minx1, miny1 = count(img3, img2, minx1 - 1, minx1 + 2, miny1 - 1, miny1 + 2)
        minx2, miny2 = count(img1, img2, minx2 - 1, minx2 + 2, miny2 - 1, miny2 + 2)
    #really move
    cpyimg3 = np.roll(img3, minx1, axis = 0);
    cpyimg3 = np.roll(cpyimg3, miny1, axis = 1)
    img3 = cpyimg3
    r_row = g_row - minx1
    r_col = g_col - miny1
    r_row += 2 * heightwas
    
    cpyimg1 = np.roll(img1, minx2, axis = 0);
    cpyimg1 = np.roll(cpyimg1, miny2, axis = 1)
    img1 = cpyimg1
    b_row = g_row - minx2
    b_col = g_col - miny2
    #add to one img
    imgnew = np.dstack((img3,img2))
    imgnew = np.dstack((imgnew,img1))    
    bgr_image = imgnew
    return bgr_image, (b_row, b_col), (r_row, r_col)

img4 = imread("01886a.png", plugin='matplotlib')
plt.imshow(img4)
imgnn, a, b = align(img4, (444, 1155))
print(a, b)
plt.imshow(imgnn)
imsave("my.png", imgnn)