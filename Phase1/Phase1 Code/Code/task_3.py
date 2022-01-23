from matplotlib.backends.backend_pdf import PdfPages
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from matplotlib import image
from scipy.stats import skew

import os, os.path
import matplotlib.pyplot as plt
import numpy as np


def read_folder_and_get_similar_images(folder_path, image_id, model_name, k):
    valid_images = [".jpg", ".png"]

    print("\nGenerating Output File for \""+image_id+"\" using Model \""+model_name+"\"\n")

    # Applied individual feature descriptors based on the model requested in the input
    if model_name == "color_moment":
        color_moment_model(folder_path, image_id, valid_images, k)
    elif model_name == "local_binary_pattern":
        local_binary_pattern_model(folder_path, image_id, valid_images, k)
    elif model_name == "histogram_of_oriented_gradients":
        histogram_of_oriented_gradients_model(folder_path, image_id, valid_images, k)
    else:
        print("No Model of the given name exist = " + model_name)
    print("\nOutput File Generated\n")
    return


def cm_to_inch(value):
    return value / 2.54


def color_moment_model(folder_path, image_id, valid_images, k):

    # Image is read based on the image ID and folder path provided in input
    main_image_matrix = image.imread(os.path.join(folder_path, image_id))

    # A dictionary is created to contian the image file as Key and its cost(Euclidean distance) as total cost.
    total_cost = {}

    main_image_color_moments = [[0] * 64 for _ in range(3)]
    main_count = 0

    # Actual image is being iterated and the values for each color moment is being stored in an array of size 3x64
    for row in range(0, 8):
        for col in range(0, 8):
            sliced_image = main_image_matrix[row * 8:(row * 8) + 8, col * 8:(col * 8) + 8]

            main_image_color_moments[0][main_count] = sliced_image.mean()
            main_image_color_moments[1][main_count] = np.std(sliced_image)
            main_image_color_moments[2][main_count] = skew(skew(sliced_image))
            main_count += 1

    # Whole folder is now iterated to calculate the cost of each image to actual_image
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        if f == image_id:
            continue

        print("\nCalculating Color Moments for \"" + f + "\"")
        image_matrix = image.imread(os.path.join(folder_path, f))
        color_moments = [[0] * 64 for _ in range(3)]
        count = 0

        # Image matrix of 64x64 is break into blocks of 8x8 matrix inorder to perform individual color moment on the image.
        for row in range(0, 8):
            for col in range(0, 8):
                sliced_image = image_matrix[row * 8:(row * 8) + 8, col * 8:(col * 8) + 8]

                color_moments[0][count] = sliced_image.mean()
                color_moments[1][count] = np.std(sliced_image)
                color_moments[2][count] = skew(skew(sliced_image))
                count += 1

        # A cost is calculated from the current image and the actual_image using L2 Normalization
        total_cost[f] = np.linalg.norm(np.array(main_image_color_moments) - np.array(color_moments), ord=2)

    # Cost dictionary is sorted in ascending order.
    total_cost = sorted(total_cost.items(), key=lambda x: x[1])

    # Dictionary is sliced to get k-neater images
    if k <= len(total_cost):
        total_cost = total_cost[:k]

    # An output file is generated in the following location to save PDF file.
    output_file = PdfPages("/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web "
                           "Databases/Project/Phase-1/output/task_3/" + os.path.splitext(image_id)[0] + '_color-moment_output.pdf')
    first_page = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(50)), dpi=80, facecolor='w', edgecolor='r')
    first_page.clf()
    title = 'COLOR MOMENTS OUTPUT FOR THE ' + os.path.splitext(image_id)[0] + " for " + str(k) + "-similar images"
    first_page.text(0.5, 0.5, title, transform=first_page.transFigure, size=500, ha="center")
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
        plot_0 = plt.figure(figsize=(cm_to_inch(200), cm_to_inch(200)))
        plt.title("Overall Matching Score = " + str(value), fontdict={'fontsize': 200})
        plt.imshow(image_file, interpolation='nearest', cmap='gray')
        output_file.savefig(plot_0)
        plt.close(plot_0)

    output_file.close()
    return


def local_binary_pattern_model(folder_path, image_id, valid_images, k):

    # neighbour_set_points and radius values are set to 8 and 1 respectively.
    neighbour_set_points = 8
    radius = 1.00

    # Image is read based on the image ID and folder path provided in input
    main_image_matrix = image.imread(os.path.join(folder_path, image_id))

    # A dictionary is created to contian the image file as Key and its cost(Euclidean distance) as total cost.
    total_cost = {}

    # ELBP is applied on the image, we use method='ror' this method provides gray scale and rotation invariant.
    main_image_lbp = local_binary_pattern(main_image_matrix, neighbour_set_points, radius, method='ror')

    # Whole folder is now iterated to calculate the cost of each image to actual_image
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        if f == image_id:
            continue

        print("\nCalculating Local Binary Patterns for \"" + f + "\"")
        image_matrix = image.imread(os.path.join(folder_path, f))
        image_lbp = local_binary_pattern(image_matrix, neighbour_set_points, radius, method='ror')

        # A cost is calculated from the current image and the actual_image using L2 Normalization
        total_cost[f] = np.linalg.norm(np.array(main_image_lbp) - np.array(image_lbp), ord=2)

    # Cost dictionary is sorted in ascending order.
    total_cost = sorted(total_cost.items(), key=lambda x: x[1])

    # Dictionary is sliced to get k-neater images
    if k <= len(total_cost):
        total_cost = total_cost[:k]

    # An output file is generated in the following location to save PDF file.
    output_file = PdfPages("/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web "
                           "Databases/Project/Phase-1/output/task_3/" + os.path.splitext(image_id)[0] + '_lpb_output.pdf')
    first_page = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(50)), dpi=80, facecolor='w', edgecolor='r')
    first_page.clf()
    title = 'LOCAL BINARY PATTERN OUTPUT FOR THE ' + os.path.splitext(image_id)[0] + " for " + \
            str(k) + "-similar images"
    first_page.text(0.5, 0.5, title, transform=first_page.transFigure, size=400, ha="center")
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
        plot_0 = plt.figure(figsize=(cm_to_inch(200), cm_to_inch(200)))
        plt.title("Overall Matching Score = " + str(value), fontdict={'fontsize': 200})
        plt.imshow(image_file, interpolation='nearest', cmap='gray')
        output_file.savefig(plot_0)
        plt.close(plot_0)

    output_file.close()

    return


def histogram_of_oriented_gradients_model(folder_path, image_id, valid_images, k):

    # Image is read based on the image ID and folder path provided in input
    main_image_matrix = image.imread(os.path.join(folder_path, image_id))

    # A dictionary is created to contian the image file as Key and its cost(Euclidean distance) as total cost.
    total_cost = {}

    # A Histogram of Oriented Gradients is being applied on actual_image with the following arguments as seen below
    main_image_feature_vector, main_image_hog = hog(main_image_matrix, orientations=9, pixels_per_cell=(8, 8),
                                                    cells_per_block=(8, 8), block_norm='L2-Hys', visualize=True,
                                                    multichannel=0)

    # Whole folder is now iterated to calculate the cost of each image to actual_image
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1]
        if ext.lower() not in valid_images:
            continue
        if f == image_id:
            continue

        print("\nCalculating Histogram of Oriented Gradients for \"" + f + "\"")
        image_matrix = image.imread(os.path.join(folder_path, f))

        # A Histogram of Oriented Gradients is being applied with the following arguments as seen below
        feature_vector, hog_image = hog(image_matrix, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(8, 8),
                                        block_norm='L2-Hys', visualize=True, multichannel=0)

        # A cost is calculated from the current image and the actual_image using L2 Normalization
        total_cost[f] = np.linalg.norm(np.array(main_image_feature_vector) - np.array(feature_vector), ord=2)

    # Cost dictionary is sorted in ascending order.
    total_cost = sorted(total_cost.items(), key=lambda x: x[1])

    # Dictionary is sliced to get k-neater images
    if k <= len(total_cost):
        total_cost = total_cost[:k]

    # An output file is generated in the following location to save PDF file.
    output_file = PdfPages("/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web "
                           "Databases/Project/Phase-1/output/task_3/" + os.path.splitext(image_id)[0] + '_hog_output.pdf')
    first_page = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(50)), dpi=80, facecolor='w', edgecolor='r')
    first_page.clf()
    title = 'HISTOGRAM OF ORIENTED GRADIENTS MODEL OUTPUT FOR THE ' + os.path.splitext(image_id)[0] + " for " + \
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
        plot_0 = plt.figure(figsize=(cm_to_inch(200), cm_to_inch(200)))
        plt.title("Overall Matching Score = " + str(value), fontdict={'fontsize': 200})
        plt.imshow(image_file, interpolation='nearest', cmap='gray')
        output_file.savefig(plot_0)
        plt.close(plot_0)

    output_file.close()

    return


folder_path = input("\nPlease Enter Folder Path: ")
image_id = input("\nPlease Enter Image ID {image-392.png}: ")
model_name = input("\nEnter Model Names from\n- color_moment\n- local_binary_pattern\n- "
                   "histogram_of_oriented_gradients\n\nPlease Enter Model Name: ")
k = int(input("\nPlease Enter K-Value: "))
read_folder_and_get_similar_images(folder_path, image_id, model_name, k)







# read_folder_and_get_similar_images(
#     "/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web Databases/Project/Phase-1/olivetti_faces",
#     "image-392.png", "histogram_of_oriented_gradients", 3)
#
# read_folder_and_get_similar_images(
#     "/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web Databases/Project/Phase-1/olivetti_faces",
#     "image-108.png", "local_binary_pattern", 5)
#
# read_folder_and_get_similar_images(
#     "/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web Databases/Project/Phase-1/olivetti_faces",
#     "image-284.png", "color_moment", 8)