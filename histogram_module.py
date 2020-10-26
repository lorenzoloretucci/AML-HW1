import numpy as np
from numpy import histogram as hist



#Add the Filtering folder, to import the gauss_module.py file, where gaussderiv is defined (needed for dxdy_hist)
import sys, os, inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
filteringpath = os.path.join(parentdir, 'Filtering')
sys.path.insert(0,filteringpath)
import gauss_module



#  compute histogram of image intensities, histogram should be normalized so that sum of all values equals 1
#  assume that image intensity varies between 0 and 255
#
#  img_gray - input image in grayscale format
#  num_bins - number of bins in the histogram
def normalized_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'

    img_vector = img_gray.reshape(img_gray.size)

    n = 255 / num_bins
    hists = np.zeros(num_bins)

    for el in img_vector:

        if el == 255:
            k = num_bins - 1
        else:
            k = int(el // n)

        hists[k] += 1

    hists = hists / hists.sum()
    bins = np.arange(0, 256, n)

    return hists, bins



#  Compute the *joint* histogram for each color channel in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^3
#
#  E.g. hists[0,9,5] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
#       - their B values fall in bin 5
def rgb_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    n = 255/num_bins

    vector_image = img_color_double.reshape(img_color_double.shape[0] * img_color_double.shape[1], 3)

    true_bins = np.arange(0, 255, n)

    #Define a 3D histogram  with "num_bins^3" number of entries
    hists = np.zeros((num_bins, num_bins, num_bins))
    
    # Loop for each pixel i in the image 
    for i in range(img_color_double.shape[0] * img_color_double.shape[1]):

        for j in range(len(vector_image[i])):

            if vector_image[i, j] == 255:
                k = num_bins - 1
            else:
                k = int(vector_image[i, j] // n)

            if j == 0:
                R = k
            elif j == 1:
                G = k
            else:
                B = k

        hists[R, G, B] += 1

        pass


    #Normalize the histogram such that its integral (sum) is equal 1
    hists = hists / hists.sum()

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists



#  Compute the *joint* histogram for the R and G color channels in the image
#  The histogram should be normalized so that sum of all values equals 1
#  Assume that values in each channel vary between 0 and 255
#
#  img_color - input color image
#  num_bins - number of bins used to discretize each channel, total number of bins in the histogram should be num_bins^2
#
#  E.g. hists[0,9] contains the number of image_color pixels such that:
#       - their R values fall in bin 0
#       - their G values fall in bin 9
def rg_hist(img_color_double, num_bins):
    assert len(img_color_double.shape) == 3, 'image dimension mismatch'
    assert img_color_double.dtype == 'float', 'incorrect image type'

    n = 255/num_bins

    vector_image = img_color_double.reshape(img_color_double.shape[0] * img_color_double.shape[1], 3)
    true_bins = np.arange(0, 255, n)

    # Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))

    for i in range(img_color_double.shape[0] * img_color_double.shape[1]):

        for j in range(len(vector_image[i])):

            if vector_image[i, j] == 255:
                k = num_bins - 1
            else:
                k = int(vector_image[i, j] // n)

            if j == 0:
                R = k
            elif j == 1:
                G = k

        hists[R, G] += 1

    hists = hists / hists.sum()

    # Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)

    return hists




#  Compute the *joint* histogram of Gaussian partial derivatives of the image in x and y direction
#  Set sigma to 3.0 and cap the range of derivative values is in the range [-6, 6]
#  The histogram should be normalized so that sum of all values equals 1
#
#  img_gray - input gray value image
#  num_bins - number of bins used to discretize each dimension, total number of bins in the histogram should be num_bins^2
#
#  Note: you may use the function gaussderiv from the Filtering exercise (gauss_module.py)
def dxdy_hist(img_gray, num_bins):
    assert len(img_gray.shape) == 2, 'image dimension mismatch'
    assert img_gray.dtype == 'float', 'incorrect image type'


    #.....your code here
    n_of_integers = list(range(-6, 7))
    #n = len(n_of_integers) / num_bins
    n = 12/num_bins
    true_bins = np.arange(-6, 6, n)
    true_bins[-1] = 6.0

    [imgDx, imgDy] = gauss_module.gaussderiv(img_gray, 3.0)
    vector_imgDx = imgDx.reshape(imgDx.size)
    vector_imgDy = imgDy.reshape(imgDy.size)

    #Define a 2D histogram  with "num_bins^2" number of entries
    hists = np.zeros((num_bins, num_bins))


    #... (your code here)
    for i in range(imgDx.size):
        #for j, threshold in enumerate(true_bins):
        if vector_imgDx[i] == 6:
            kx = num_bins - 1
            #elif vector_imgDx[i] >= threshold and vector_imgDx[i] < threshold + n:
                #kx = j
        else:
            kx = int((vector_imgDx[i] + 6) // n)

        if vector_imgDy[i] == 6:
            ky = num_bins - 1
            #elif vector_imgDy[i] >= threshold and vector_imgDy[i] < threshold + n:
                #ky = j
        else:
            ky = int((vector_imgDy[i] + 6) // n)

        hists[kx, ky] += 1

    hists = hists / hists.sum()

    #Return the histogram as a 1D vector
    hists = hists.reshape(hists.size)
    return hists



def is_grayvalue_hist(hist_name):
  if hist_name == 'grayvalue' or hist_name == 'dxdy':
    return True
  elif hist_name == 'rgb' or hist_name == 'rg':
    return False
  else:
    assert False, 'unknown histogram type'


def get_hist_by_name(img, num_bins_gray, hist_name):
  if hist_name == 'grayvalue':
    return normalized_hist(img, num_bins_gray)
  elif hist_name == 'rgb':
    return rgb_hist(img, num_bins_gray)
  elif hist_name == 'rg':
    return rg_hist(img, num_bins_gray)
  elif hist_name == 'dxdy':
    return dxdy_hist(img, num_bins_gray)
  else:
    assert False, 'unknown distance: %s'%hist_name

