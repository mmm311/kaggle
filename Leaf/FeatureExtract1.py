'''
Created on 2016年12月8日

@author: liu
'''
import numpy as np

import scipy as sp
import scipy.ndimage as ndi
from scipy.signal import argrelextrema
import matplotlib.pyplot as plt
import pandas as pd

from skimage import measure
from sklearn import metrics
import csv as csv
import matplotlib.image as mpimg
from pylab import  rcParams
rcParams['figure.figsize'] = (6,6)
img = mpimg.imread('../leafData/images/1.jpg')
cy, cx = ndi.center_of_mass(img)
dist_2d = ndi.distance_transform_edt(img)
plt.imshow(img, cmap='Greys', alpha=.2)
plt.imshow(dist_2d, cmap='plasma', alpha=.2)
plt.contour(dist_2d, cmap='plasma')
print(dist_2d.shape)
plt.show()
contours = measure.find_contours(img, .8)
contour = max(contours, key = len)

def cart2pol(x, y):
    rho = np.sqrt(x**2 + y ** 2)
    phi = np.arctan2(y,x)
    return [rho, phi]
polar_contour = np.array([cart2pol(x, y) for x, y in contour])

contour[:,1] -= cx
contour[:,0] -= cy

# just calling the transformation on all pairs in the set
polar_contour = np.array([cart2pol(x, y) for x,y in contour] )
# and plotting the result
rcParams['figure.figsize'] = (12, 6)
plt.subplot(121)
plt.scatter(polar_contour[:,1], polar_contour[:,0], linewidths=0,
            s = 2, c= polar_contour[:,1])
plt.title('in Polar Coordinates')
plt.grid()
plt.subplot(122)
plt.scatter(contour[:,1], contour[:,0],
            linewidths=0, s = 2, c = range(len(contour)))
plt.scatter(0, 0)
plt.title('in Cartesian Coordinates')
plt.grid()


from scipy.signal import argrelextrema
# for local maxima
c_max_index = argrelextrema(polar_contour[:,0], np.greater,order = 50)
c_min_index = argrelextrema(polar_contour[:,0], np.less, order = 50)

plt.subplot(121)
plt.scatter(polar_contour[:,1],polar_contour[:,0], linewidths=0, s = 2, c='k')
plt.scatter(polar_contour[:,1][c_max_index],polar_contour[:,0][c_max_index],
            linewidths=0, s = 30, c = 'b')
plt.scatter(polar_contour[:,1][c_min_index],polar_contour[:,0][c_min_index],
            linewidths=0, s = 30, c = 'r')
plt.subplot(122)
plt.scatter(contour[:,1], contour[:,0], linewidths=0, s = 2, c = 'k')
plt.scatter(contour[:,1][c_max_index], contour[:,0][c_max_index],linewidths= 0,
            s = 30, c='b')
plt.scatter(contour[:,1][c_min_index], contour[:,0][c_min_index],linewidths=0,
            s = 30, c= 'r')
plt.show()




