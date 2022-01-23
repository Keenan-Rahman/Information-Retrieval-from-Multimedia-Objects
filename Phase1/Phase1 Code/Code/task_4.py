from matplotlib.backends.backend_pdf import PdfPages
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from matplotlib import image
from scipy.stats import skew

import os, os.path
import matplotlib.pyplot as plt
import numpy as np
import sys


def read_folder_and_get_similar_images(folder_path, image_id, k):
    valid_images = [".jpg", ".png"]

    # Dictionaries are created to keep track of cost of individual descriptors and the total cost for all descriptors.
    total_cost = {}
    cm_cost_dic = {}
    lbp_cost_dic = {}
    hog_cost_dic = {}

    # For each of the descriptors a max and min value is being saved in order to scale the values in the end
    max_cm = -(sys.maxsize - 1)
    min_cm = sys.maxsize
    max_lbp = -(sys.maxsize - 1)
    min_lbp = sys.maxsize
    max_hog = -(sys.maxsize - 1)
    min_hog = sys.maxsize

    # Image is read based on the image ID and folder path provided in input
    main_image_matrix = image.imread(os.path.join(folder_path, image_id))

    # All descriptors are applied to the actual_image and results are saved.
    main_image_color_moments = color_moment_model_single_image(main_image_matrix)
    main_image_lbp = local_binary_pattern_model_single_image(main_image_matrix)
    main_image_hog = histogram_of_oriented_gradients_model_single_image(main_image_matrix)

    # Whole folder is now iterated to calculate the cost of each image to actual_image
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        if f == image_id:
            continue

        print("\nApplying Feature Descriptors for \"" + f + "\"")
        image_matrix = image.imread(os.path.join(folder_path, f))

        # All descriptors are applied to the actual_image and results are saved for individual image.
        image_color_moments = color_moment_model_single_image(image_matrix)
        image_lbp = local_binary_pattern_model_single_image(image_matrix)
        image_hog = histogram_of_oriented_gradients_model_single_image(image_matrix)

        # Cost is being calculated for each descriptor.
        cm_cost = np.linalg.norm(np.array(main_image_color_moments) - np.array(image_color_moments), ord=2)
        lbp_cost = np.linalg.norm(np.array(main_image_lbp) - np.array(image_lbp), ord=2)
        hog_cost = np.linalg.norm(np.array(main_image_hog) - np.array(image_hog), ord=2)

        # Costs are saved in dictionary with image id as Key and descriptors output as Value.
        cm_cost_dic[f] = cm_cost
        lbp_cost_dic[f] = lbp_cost
        hog_cost_dic[f] = hog_cost

        # Max and Min are calculated for each model
        max_cm = max(cm_cost, max_cm)
        max_lbp = max(lbp_cost, max_lbp)
        max_hog = max(hog_cost, max_hog)
        min_cm = min(cm_cost, min_cm)
        min_lbp = min(lbp_cost, min_lbp)
        min_hog = min(hog_cost, min_hog)

    print("\nScaling All the Costs")
    # All dictionaries are iterated and the values of each descriptor is scaled between 0 - 1 and saved to total_cost.
    for key, value in cm_cost_dic.items():

        std_cm_cost = (value - min_cm) / (max_cm - min_cm)
        std_lbp_cost = (lbp_cost_dic[key] - min_lbp) / (max_lbp - min_lbp)
        std_hog_cost = (hog_cost_dic[key] - min_hog) / (max_hog - min_hog)

        total_image_cost = (std_cm_cost + std_lbp_cost + std_hog_cost) / 3
        total_cost[key] = total_image_cost

    # Cost dictionary is sorted in ascending order.
    total_cost = sorted(total_cost.items(), key=lambda x: x[1])

    # Dictionary is sliced to get k-neater images
    if k <= len(total_cost):
        total_cost = total_cost[:k]

    print("\nGenerating Output File for \""+image_id+"\"")

    # An output file is generated in the following location to save PDF file.
    output_file = PdfPages("/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web "
                           "Databases/Project/Phase-1/output/task_4/" + os.path.splitext(image_id)[0] + '_output.pdf')
    first_page = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(50)), dpi=80, facecolor='w', edgecolor='r')
    first_page.clf()
    title = 'ALL MODEL OUTPUT FOR THE ' + os.path.splitext(image_id)[0] + " for " + \
            str(k) + "-similar images"
    first_page.text(0.5, 0.5, title, transform=first_page.transFigure, size=300, ha="center")
    output_file.savefig()
    plt.close(first_page)

    actual_image = image.imread(os.path.join(folder_path, image_id))
    plot_actual_image = plt.figure(figsize=(cm_to_inch(200), cm_to_inch(200)))
    plt.title("Actual Image", fontdict={'fontsize': 200})
    plt.imshow(actual_image, interpolation='nearest', cmap='gray')
    output_file.savefig(plot_actual_image)
    plt.close(plot_actual_image)

    # All k-nearest images are iterated and saved in the file.
    for key, value in total_cost:
        image_file = image.imread(os.path.join(folder_path, key))
        plot_0 = plt.figure(figsize=(cm_to_inch(200), cm_to_inch(300)))
        plt.title("Color Moment Score = " + str(cm_cost_dic.get(key)) + "\nLBP Score = " +
                  str(lbp_cost_dic.get(key)) + "\nHOG Score = " + str(hog_cost_dic.get(key)) +
                  "\nOverall Score = " + str(value), fontdict={'fontsize': 200})
        plt.imshow(image_file, interpolation='nearest', cmap='gray')
        output_file.savefig(plot_0)
        plt.close(plot_0)

    output_file.close()
    print("\nOutput File Generated\n")

    return


def color_moment_model_single_image(image_array):
    color_moments = [[0] * 64 for _ in range(3)]
    count = 0

    # Image matrix of 64x64 is break into blocks of 8x8 matrix inorder to perform individual color moment on the image.
    for row in range(0, 8):
        for col in range(0, 8):
            sliced_image = image_array[row * 8:(row * 8) + 8, col * 8:(col * 8) + 8]

            color_moments[0][count] = sliced_image.mean()
            color_moments[1][count] = np.std(sliced_image)
            color_moments[2][count] = skew(skew(sliced_image))
            count += 1

    return color_moments


def local_binary_pattern_model_single_image(image_array):

    # neighbour_set_points and radius values are set to 8 and 1 respectively.
    neighbour_set_points = 8
    radius = 1.00

    # ELBP is applied on the image, we use method='ror' this method provides gray scale and rotation invariant.
    return local_binary_pattern(image_array, neighbour_set_points, radius, method='ror')


def histogram_of_oriented_gradients_model_single_image(image_array):

    # A Histogram of Oriented Gradients is being applied on actual_image with the following arguments as seen below
    feature_vector, hog_visualization = hog(image_array, orientations=9, pixels_per_cell=(8, 8),
                                            cells_per_block=(8, 8), block_norm='L2-Hys',
                                            visualize=True, multichannel=0)
    return feature_vector


def cm_to_inch(value):
    return value / 2.54


folder_path = input("\nPlease Enter Folder Path: ")
image_id = input("\nPlease Enter Image ID {image-392.png}: ")
k = int(input("\nPlease Enter K-Value: "))
read_folder_and_get_similar_images(folder_path, image_id, k)







# read_folder_and_get_similar_images(
#     "/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web Databases/Project/Phase-1/olivetti_faces",
#     "image-392.png", 3)
#
# read_folder_and_get_similar_images(
#     "/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web Databases/Project/Phase-1/olivetti_faces",
#     "image-108.png", 5)
#
# read_folder_and_get_similar_images(
#     "/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web Databases/Project/Phase-1/olivetti_faces",
#     "image-284.png", 8)
read_folder_and_get_similar_images("/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web Databases/Project/Phase-1/test_imgage_sets/set1",
                                   "image-0.png", 2)