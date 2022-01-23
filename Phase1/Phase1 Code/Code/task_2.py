from matplotlib.backends.backend_pdf import PdfPages
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from matplotlib import image
from scipy.stats import skew
from skimage import exposure

import os, os.path
import matplotlib.pyplot as plt
import numpy as np


def read_folder_and_generate_report(folder_path):
    valid_images = [".jpg", ".png"]

    # Iterate through all the files in the folder.
    for f in os.listdir(folder_path):
        ext = os.path.splitext(f)[1]

        # Does not process any file that is not within the acceptable extension.
        if ext.lower() not in valid_images:
            continue

        image_file = image.imread(os.path.join(folder_path, f))

        # An output file is generated in the following location to save PDF file.
        output_file = PdfPages("/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web "
                               "Databases/Project/Phase-1/output/task_2/" + os.path.splitext(f)[0] + '_output.pdf')

        first_page = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(50)), dpi=80, facecolor='w',
                                edgecolor='r')
        first_page.clf()
        title = 'OUTPUT FOR THE ' + os.path.splitext(f)[0]
        first_page.text(0.5, 0.5, title, transform=first_page.transFigure, size=500, ha="center")
        output_file.savefig()
        plt.close(first_page)

        # Original image is printed on the file initially.
        plot_0 = plt.figure(figsize=(cm_to_inch(200), cm_to_inch(200)))
        plt.imshow(image_file, interpolation='nearest', cmap='gray')
        output_file.savefig(plot_0)
        plt.close(plot_0)

        # All the feature descriptors are applied to the image file.
        color_moment_model(f, image_file, output_file)
        local_binary_pattern_model(f, image_file, output_file)
        histogram_of_oriented_gradients_model(f, image_file, output_file)

        output_file.close()
        print("\nOutput File for \""+os.path.splitext(f)[0]+"\" has been created\n")

    return


def cm_to_inch(value):
    return value / 2.54


def color_moment_model(image_id, image_matrix, output_file):
    # created 3 dictionaries to keep track of the values. Data is saved in dictionatries having image id as Key and
    # color moment result as Value.
    mean_of_sliced_image = {}
    std_dev_of_sliced_image = {}
    skew_of_sliced_image = {}

    # Image matrix of 64x64 is break into blocks of 8x8 matrix inorder to perform individual color moment on the image.
    for row in range(0, 8):
        for col in range(0, 8):
            # matrix is being sliced to 8x8 and values saved in the dictionary
            sliced_image = image_matrix[row * 8:(row * 8) + 8, col * 8:(col * 8) + 8]
            mean_of_sliced_image["[" + str(row * 8) + ":" + str((row * 8) + 8) + "] \n[" + str(col * 8) + ":" + str(
                (col * 8) + 8) + "]"] = sliced_image.mean()
            std_dev_of_sliced_image["[" + str(row * 8) + ":" + str((row * 8) + 8) + "] \n[" + str(col * 8) + ":" + str(
                (col * 8) + 8) + "]"] = np.std(sliced_image)
            skew_of_sliced_image["[" + str(row * 8) + ":" + str((row * 8) + 8) + "] \n[" + str(col * 8) + ":" + str(
                (col * 8) + 8) + "]"] = skew(skew(sliced_image))

    color_moment_page = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(50)), dpi=80, facecolor='w',
                                   edgecolor='r')
    color_moment_page.clf()
    color_moment_page_title = 'I - Color Moment Result'
    color_moment_page.text(0.5, 0.5, color_moment_page_title, transform=color_moment_page.transFigure, size=250,
                           ha="center")
    output_file.savefig()
    plt.close(color_moment_page)

    labels = []
    ys = []
    for x, y in mean_of_sliced_image.items():
        labels.append(x)
        ys.append(y)

    # A bar graph is generated for the Mean - color moment result obtained from the image
    mean_page = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(50)), dpi=80, facecolor='w', edgecolor='r')
    mean_page.clf()
    mean_title = 'i - Mean Values for Image Sliced into 8*8 Matrices'
    mean_page.text(0.5, 0.5, mean_title, transform=mean_page.transFigure, size=250, ha="center")
    output_file.savefig()
    plt.close(mean_page)

    # Graph is saved to PDF file
    xs = np.arange(len(labels))
    width = 0.6
    plot_1 = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(150)), dpi=80, facecolor='w', edgecolor='r')
    plt.bar(xs, ys, width, align='edge')
    plt.xticks(xs, labels, fontsize=40, weight='bold', horizontalalignment="center")
    plt.yticks(fontsize=100, weight='bold')
    output_file.savefig(plot_1, orientation='portrait')
    plt.close(plot_1)

    labels = []
    ys = []
    for x, y in std_dev_of_sliced_image.items():
        labels.append(x)
        ys.append(y)

    # A bar graph is generated for the Standard Deviation - color moment result obtained from the image
    std_dev_page = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(50)), dpi=80, facecolor='w', edgecolor='r')
    std_dev_page.clf()
    std_dev_title = 'ii - Standard Deviation Values for Image Sliced into 8*8 Matrices'
    std_dev_page.text(0.5, 0.5, std_dev_title, transform=std_dev_page.transFigure, size=250, ha="center")
    output_file.savefig()
    plt.close(std_dev_page)

    # Graph is saved to PDF file
    xs = np.arange(len(labels))
    width = 0.6
    plot_2 = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(150)), dpi=80, facecolor='w', edgecolor='r')
    plt.bar(xs, ys, width, align='edge')
    plt.xticks(xs, labels, fontsize=40, weight='bold', horizontalalignment="center")
    plt.yticks(fontsize=100, weight='bold')
    output_file.savefig(plot_2)
    plt.close(plot_2)

    labels = []
    ys = []
    for x, y in skew_of_sliced_image.items():
        labels.append(x)
        ys.append(y)

    # A bar graph is generated for the Skewness - color moment result obtained from the image
    skew_page = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(50)), dpi=80, facecolor='w', edgecolor='r')
    skew_page.clf()
    skew_title = 'iii - Skewness for Image Sliced into 8*8 Matrices'
    skew_page.text(0.5, 0.5, skew_title, transform=skew_page.transFigure, size=250, ha="center")
    output_file.savefig()
    plt.close(skew_page)

    # Graph is saved to PDF file
    xs = np.arange(len(labels))
    width = 0.6
    plot_3 = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(150)), dpi=80, facecolor='w', edgecolor='r')
    plt.bar(xs, ys, width, align='edge')
    plt.xticks(xs, labels, fontsize=40, weight='bold', horizontalalignment="center")
    plt.yticks(fontsize=100, weight='bold')
    output_file.savefig(plot_3)
    plt.close(plot_3)

    return


def local_binary_pattern_model(image_id, image_matrix, output_file):
    local_binary_pattern_page = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(50)), dpi=80, facecolor='w',
                                           edgecolor='r')
    local_binary_pattern_page.clf()
    local_binary_pattern_page_title = 'II - Local Binary Pattern Result'
    local_binary_pattern_page.text(0.5, 0.5, local_binary_pattern_page_title,
                                   transform=local_binary_pattern_page.transFigure, size=250, ha="center")
    output_file.savefig()
    plt.close(local_binary_pattern_page)

    # neighbour_set_points and radius values are set to 8 and 1 respectively.
    neighbour_set_points = 8
    radius = 1.00

    # ELBP is applied on the image, we use method='ror' this method provides gray scale and rotation invariant.
    patterns = local_binary_pattern(image_matrix, neighbour_set_points, radius, method='ror')

    plot_4 = plt.figure(figsize=(cm_to_inch(200), cm_to_inch(200)))
    plt.imshow(patterns, interpolation='nearest', cmap='gray')
    output_file.savefig(plot_4)
    plt.close(plot_4)

    # A histogram is being generated from the output to visualize the output from the ELBP model
    hist, _ = np.histogram(patterns, bins=np.arange(2 ** 3 + 1), density=True)

    # Graph is saved to PDF file
    plot_5 = plt.figure(figsize=(cm_to_inch(200), cm_to_inch(200)))
    plt.ylim([0, hist.max()])
    plt.yticks(fontsize=100, weight='bold')
    plt.xlim([0, 7])
    plt.xticks(fontsize=100, weight='bold')
    plt.plot(hist)
    output_file.savefig(plot_5)
    plt.close(plot_5)

    return


def histogram_of_oriented_gradients_model(image_id, image_matrix, output_file):
    hog_page = plt.figure(num=None, figsize=(cm_to_inch(600), cm_to_inch(50)), dpi=80, facecolor='w', edgecolor='r')
    hog_page.clf()
    hog_page_title = 'III - Histogram of Oriented Gradients Result'
    hog_page.text(0.5, 0.5, hog_page_title, transform=hog_page.transFigure, size=250, ha="center")
    output_file.savefig()
    plt.close(hog_page)

    # A Histogram of Oriented Gradients is being applied with the following arguments as seen below
    fd, hog_image = hog(image_matrix, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(8, 8), block_norm='L2-Hys', visualize=True, multichannel=0)

    # A bar graph is being generated from the output to visualize
    plot_6, ax2 = plt.subplots(figsize=(cm_to_inch(200), cm_to_inch(200)), sharex=True, sharey=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    ax2.imshow(hog_image_rescaled, cmap='gray')

    # Graph is saved to PDF file
    output_file.savefig(plot_6)
    plt.close(plot_6)

    return


folder_path = input("\nPlease Enter Folder Path: ")
read_folder_and_generate_report(folder_path)

# read_folder_and_generate_report(
#     "/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web Databases/Project/Phase-1/olivetti_faces")
