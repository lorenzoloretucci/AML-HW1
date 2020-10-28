import numpy as np
import math



# Compute the intersection distance between histograms x and y
# Return 1 - hist_intersection, so smaller values correspond to more similar histograms
# Check that the distance range in [0,1]

def dist_intersect(x,y):
    
    #... (your code here)
    minima = []

    for i in range(len(x)):
        minima.append(min(x[i], y[i]))

    intersect = 1 / 2 * (np.array(minima).sum() / x.sum() + np.array(minima).sum() / y.sum())

    result = 1 - intersect

    return result


# Compute the L2 distance between x and y histograms
# Check that the distance range in [0,sqrt(2)]

def dist_l2(x,y):
    
    #... (your code here)
    l2_distance = ((x - y) ** 2).sum()

    return l2_distance


# Compute chi2 distance between x and y
# Check that the distance range in [0,Inf]
# Add a minimum score to each cell of the histograms (e.g. 1) to avoid division by 0

def dist_chi2(x,y):
    def dist_chi2(x, y):

        chi2_distance = 0

        for i in range(len(x)):

            if x[i] > 0 or y[i] > 0:
                chi2_distance += (((x[i] - y[i]) ** 2)) / (x[i] + y[i])
            else:
                chi2_distance += 0

        return chi2_distance


def get_dist_by_name(x, y, dist_name):
  if dist_name == 'chi2':
    return dist_chi2(x,y)
  elif dist_name == 'intersect':
    return dist_intersect(x,y)
  elif dist_name == 'l2':
    return dist_l2(x,y)
  else:
    assert False, 'unknown distance: %s'%dist_name
  




