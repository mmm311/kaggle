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
    return mpimg.imread('../LeafData/imges/' + str(img_no) +'.jpg')

def get_img(num):
    '''convenience function, yields random sample from 1'''
    if type(num) == int:
        imgs = range(1, 1584)
        num = np.random.choice(imgs, size = num, replace = True)
    for img_no in num:
        yield img_no, preprocess(read_img(img_no))
 
# ----------------- preprocessing --------------
''' splits img to 0 and 255 values at threshold ''''
def threshold(img,):       