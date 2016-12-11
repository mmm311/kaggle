'''
Created on 2016年12月11日

@author: liu
'''
import numpy as np
import scipy as sp
import scipy.ndimage as ndi
import pandas as pd

from skimage import measure
from sklearn import metrics
import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from pylab import rcParams

rcParams['figure.figsize'] = (6,6)
# ---------- I/O ------------------
def read_img(img_no):
    ''' reads img from disk'''
    return mpimg.imread('../LeafData/images/' + str(img_no) +'.jpg')

def get_img(num):
    '''convenience function, yields random sample from 1'''
    if type(num) == int:
        imgs = range(1, 1584)
        num = np.random.choice(imgs, size = num, replace = True)
    for img_no in num:
        yield img_no, preprocess(read_img(img_no))
 
# ----------------- preprocessing --------------
''' splits img to 0 and 255 values at threshold '''
def threshold(img, threshold = 250):
    return ((img > threshold) * 255).astype(int)

def portrait(img):
    ''' make all leaves stand straight'''
    y , x = np.shape(img)
    return img.transpose() if x > y else img

def resample(img ,size):
    ''' resample img to size without distorsion '''
    ratio = size / max(np.shape(img))
    return sp.misc.imresize(img, ratio, mode = 'L', interp = 'nearest')

def fill(img, size = 500, tolerance = 0.95):  
    ''' extends the image if it is signifficantly smaller than size'''
    y, x = np.shape(img)
    
    if x <= size * tolerance:
        pad = np.zeros((y, int ((size - x) / 2)),dtype = int)
        img = np.concatenate((pad, img, pad) , axis = 1)
    if y <= size * tolerance:
        pad = np.zeros((int ((size -y ) /2 ), x), dtype = int)
        img = np.concatenate((pad, img , pad), axis = 0)
    return img
# ---------------------feature enginerring---------------
def extract_shape(img):
    '''
        Expects prepared image, returns leaf shape in img format.
        The strength of smoothing had to be dynamically set
        in order to get consistent results for different sizes.
    '''
    size = int(np.count_nonzero(img) / 1000)
   
    brush = int(5 * size / size ** 0.75)
    # 高斯过滤
    # ?
    return ndi.gaussian_filter(img, sigma = brush, mode = 'nearest') > 200


# ----------------- wrapping function -----------------
def preprocess(img , do_portrait = True, do_resample = 500,
               do_fill = True, do_threshold = 250):
    if do_portrait:
        img = portrait(img)
    if do_resample:
        img = resample(img, size = do_resample)
    if do_fill:
        img = fill(img, size = do_resample)
    if do_threshold:
        img = threshold(img, threshold = do_threshold)
    return img

img = read_img(1)
title , img = list(get_img([1]))[0]
blur = extract_shape(img)

plt.imshow(img, cmap='Set3')
blade = max(measure.find_contours(img, .8),key = len)
shape = max(measure.find_contours(blur, 0.8), key = len)
plt.scatter(blade[:,1],blade[:,0],s = 1,linewidths= 0,c = 'r')
plt.scatter(shape[:,1],shape[:,0], s = 1, linewidths= 0 , c = 'k')
plt.show()
