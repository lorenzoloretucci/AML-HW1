# import packages: numpy, math (you might need pi for gaussian functions)
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.signal import convolve2d as conv2



"""
Gaussian function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian values Gx computed at the indexes x
"""
def gauss(sigma):

    sig = int(sigma)
    x = list(range(-3*sig, 3*sig+1))
    Gx = np.array([1/(math.sqrt(2*math.pi)*sig)*math.exp(-y**2/(2*sig**2)) for y in x])

    return Gx, x



"""
Implement a 2D Gaussian filter, leveraging the previous gauss.
Implement the filter from scratch or leverage the convolve2D method (scipy.signal)
Leverage the separability of Gaussian filtering
Input: image, sigma (standard deviation)
Output: smoothed image
"""

def gaussianfilter(img, sigma):

    Gx, x = gauss(sigma)
    Gauss_kernel = np.outer(Gx, Gx)
    Gauss_filter = Gauss_kernel / Gauss_kernel.sum()
    smooth_img = conv2(img, Gauss_filter,
                       mode='same', boundary='fill', fillvalue=0)

    return smooth_img



"""
Gaussian derivative function taking as argument the standard deviation sigma
The filter should be defined for all integer values x in the range [-3sigma,3sigma]
The function should return the Gaussian derivative values Dx computed at the indexes x
"""
def gaussdx(sigma):

    sig = int(sigma)
    x = list(range(-3*sig, 3*sig+1))
    Dx = np.array([-(1/(math.sqrt(2*math.pi)*sig**3))*y*math.exp(-y**2/(2*sig**2)) for y in x])
    
    return Dx, x



def gaussderiv(img, sigma):

    Gx, x = gauss(sigma)
    Dx, x = gaussdx(sigma)

    Gx = Gx.reshape(1, Gx.size)
    Dx = Dx.reshape(1, Dx.size)

    imgDx = conv2(conv2(img, Dx, 'same'), Gx.T, 'same')
    imgDy = conv2(conv2(img, Gx, 'same'), Dx.T, 'same')

    return imgDx, imgDy

