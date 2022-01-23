import numpy as np
from scipy.stats import skew
from skimage.feature import local_binary_pattern, hog


def cm_8x8_image(image):
    windowsize_r = 8
    windowsize_c = 8
    meanConcate, stdConcate, skewConcate = [], [], []
    # Iterate through each 8x8 window in the 64x64 image
    for r in range(0, image.shape[0], windowsize_r):
        for c in range(0, image.shape[1], windowsize_c):
            # The current 8x8 window
            window = image[r:r + windowsize_r, c:c + windowsize_c]
            # Calculate the mean of the window
            meanConcate.append(np.mean(window))
            # Calculate the standard deviation of the window
            stdConcate.append(np.std(window))
            # Calculate the skew of the window
            temp = skew(window)
            skewConcate.append(skew(temp))

    return meanConcate, stdConcate, skewConcate


def elbp_image(image):
    radius = 4
    nPoints = 8 * radius
    return local_binary_pattern(image, nPoints, radius, 'uniform')


def hog_image(image):
    return hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)[1]
