import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import histogram_module
import dist_module

def rgb2gray(rgb):

    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray



# model_images - list of file names of model images
# query_images - list of file names of query images
#
# dist_type - string which specifies distance type:  'chi2', 'l2', 'intersect'
# hist_type - string which specifies histogram type:  'grayvalue', 'dxdy', 'rgb', 'rg'
#
# note: use functions 'get_dist_by_name', 'get_hist_by_name' and 'is_grayvalue_hist' to obtain 
#       handles to distance and histogram functions, and to find out whether histogram function 
#       expects grayvalue or color image

def find_best_match(model_images, query_images, dist_type, hist_type, num_bins):
    hist_isgray = histogram_module.is_grayvalue_hist(hist_type)
    
    model_hists = compute_histograms(model_images, hist_type, hist_isgray, num_bins)
    query_hists = compute_histograms(query_images, hist_type, hist_isgray, num_bins)

    D = np.zeros((len(model_images), len(query_images)))

    # ... (your code here)
    best_match = [0] * len(query_hists)

    #Computing the distance according to the chosen metric
    for i, model in enumerate(model_hists):
        for j, query in enumerate(query_hists):

            if dist_type == 'intersect':
                D[i, j] = dist_module.dist_intersect(model, query)

            elif dist_type == 'l2':
                D[i, j] = dist_module.dist_l2(model, query)

            elif dist_type == 'chi2':
                D[i, j] = dist_module.dist_chi2(model, query)

            #Keeping the closest image
            if i == 0:
                best_match[j] = i
            elif D[i, j] < D[best_match[j], j]:
                best_match[j] = i

    return np.array(best_match), D



def compute_histograms(image_list, hist_type, hist_isgray, num_bins):
    
    image_hist = []

    # Compute histogram for each image and add it at the bottom of image_hist

    #... (your code here)
    for path_img in image_list:

        img = np.array(Image.open(path_img))

        if hist_isgray == True:
            img = rgb2gray(img.astype('double'))

        #Computing the right histogram
        if hist_type == 'grayvalue':
            image_hist.append(histogram_module.normalized_hist(img, num_bins)[0])
        elif hist_type == 'rgb':
            image_hist.append(histogram_module.rgb_hist(img.astype('double'), num_bins))
        elif hist_type == 'rg':
            image_hist.append(histogram_module.rg_hist(img.astype('double'), num_bins))
        elif hist_type == 'dxdy':
            image_hist.append(histogram_module.dxdy_hist(img, num_bins))

    return image_hist



# For each image file from 'query_images' find and visualize the 5 nearest images from 'model_image'.
#
# Note: use the previously implemented function 'find_best_match'
# Note: use subplot command to show all the images in the same Python figure, one row per query image

def show_neighbors(model_images, query_images, dist_type, hist_type, num_bins):

    plt.figure()

    num_nearest = 5  # show the top-5 neighbors

    # ... (your code here)
    #Computing distances
    best_match, D = find_best_match(model_images, query_images, dist_type, hist_type, num_bins)

    #Keeping the top k
    top_k = []

    for col in range(D.shape[1]):
        top_k.append(D[:, col].argsort()[0:num_nearest])

    #Plotting the results
    cc = 1
    
    for i, best in enumerate(top_k):


        plt.tight_layout()

        plt.rcParams["figure.figsize"] = [30, 10]

        img_color = np.array(Image.open(query_images[i]))
        ax1 = plt.subplot(i + 1, num_nearest + 1, cc)
        que = 'Q' + str(i)
        ax1.set_title(que, size=25)
        plt.sca(ax1)
        plt.imshow(img_color, vmin=0, vmax=255)
        plt.axis('off')
        cc += 1

        for j, img in enumerate(best):
            img_color = np.array(Image.open(model_images[img]))
            ax = plt.subplot(i + 1, num_nearest + 1, cc)
            title = 'M' + str(round(D[img, i], 2))
            ax.set_title(title, size=25)
            plt.sca(ax)
            plt.imshow(img_color, vmin=0, vmax=255)
            cc += 1
            plt.axis('off')
            plt.tight_layout()

        plt.tight_layout()

        plt.savefig('Intersect-rg {}'.format(i), dpi=100)
        plt.show()

    plt.close()







