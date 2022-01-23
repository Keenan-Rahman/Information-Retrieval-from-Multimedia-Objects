from scipy.stats import skew
from skimage.feature import local_binary_pattern
from skimage.feature import hog
from pymongo import MongoClient
from skimage import exposure

import pickle
import matplotlib.pyplot as plt
import numpy as np


def apply_model_on_olivetti_face(image_id, model_name):
    if image_id < 0 or image_id > 399:
        print("No Image of this ID Exist " + str(image_id))
        return

    # create database connection on mongodb server with port=27017, appname = MongoDB Compass, and no username and
    # password
    client = MongoClient("mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false")
    db = client.admin
    mwdb_database = client["mwdb_database"]
    olivetti_faces_col = mwdb_database["olivetti_faces"]

    print("\nGenerating Results for Image ID \"" + str(image_id) + "\"\n")

    # fetch the data from the database using the image id provided in the input
    data_image = olivetti_faces_col.find_one({'_id': image_id})
    binary_data = data_image['image_matrix']

    # convert the data back to its original format by using pickle.load
    image_matrix = pickle.loads(binary_data)
    plt.figure()
    plt.title("Image For ID " + str(image_id))
    plt.imshow(image_matrix, interpolation='nearest')
    plt.gray()

    # applied individual feature descriptors based on the model requested in the input
    if model_name == "color_moment":
        color_moment_model(image_id, image_matrix)
    elif model_name == "local_binary_pattern":
        local_binary_pattern_model(image_id, image_matrix)
    elif model_name == "histogram_of_oriented_gradients":
        histogram_of_oriented_gradients_model(image_id, image_matrix)
    else:
        print("No Model of the given name exist = " + model_name)
    print("\nResults Generated\n")

    return


def cm_to_inch(value):
    return value / 2.54


def color_moment_model(image_id, image_matrix):
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
    print("\nColor Moment Result for Image " + str(image_id) + "\n")
    labels = []
    ys = []
    for x, y in mean_of_sliced_image.items():
        labels.append(x)
        ys.append(y)

    # A bar graph is generated for the Mean - color moment result obtained from the image
    xs = np.arange(len(labels))
    width = 0.4
    plt.figure(figsize=(cm_to_inch(500), cm_to_inch(50)), dpi=30, facecolor='w', edgecolor='r')
    plt.bar(xs, ys, width)
    plt.title("Mean Values for Image Sliced into 8*8 Matrices")
    plt.xticks(xs, labels, fontsize=15, weight='bold',
               horizontalalignment="center")  # Replace default x-ticks with xs, then replace xs with labels
    plt.yticks(fontsize=20, weight='bold')
    plt.plot()

    labels = []
    ys = []
    for x, y in std_dev_of_sliced_image.items():
        labels.append(x)
        ys.append(y)

    # A bar graph is generated for the Standard Deviation - color moment result obtained from the image
    xs = np.arange(len(labels))
    width = 0.4
    plt.figure(figsize=(cm_to_inch(500), cm_to_inch(50)), dpi=30, facecolor='w', edgecolor='r')
    plt.bar(xs, ys, width)
    plt.title("Standard Deviation Values for Image Sliced into 8*8 Matrices")
    plt.xticks(xs, labels, fontsize=15, weight='bold',
               horizontalalignment="center")  # Replace default x-ticks with xs, then replace xs with labels
    plt.yticks(fontsize=20, weight='bold')
    plt.plot()

    labels = []
    ys = []
    for x, y in skew_of_sliced_image.items():
        labels.append(x)
        ys.append(y)

    # A bar graph is generated for the Skewness - color moment result obtained from the image
    xs = np.arange(len(labels))
    width = 0.4
    plt.figure(figsize=(cm_to_inch(500), cm_to_inch(50)), dpi=30, facecolor='w', edgecolor='r')
    plt.bar(xs, ys, width)
    plt.title("Skewness for Image Sliced into 8*8 Matrices")
    plt.xticks(xs, labels, fontsize=15, weight='bold',
               horizontalalignment="center")  # Replace default x-ticks with xs, then replace xs with labels
    plt.yticks(fontsize=20, weight='bold')
    plt.plot()
    plt.show()

    return


def local_binary_pattern_model(image_id, image_matrix):
    print("\nLocal Binary Pattern Result for Image " + str(image_id) + "\n")

    # neighbour_set_points and radius values are set to 8 and 1 respectively.
    neighbour_set_points = 8
    radius = 1.00

    # ELBP is applied on the image, we use method='ror' this method provides gray scale and rotation invariant.
    patterns = local_binary_pattern(image_matrix, neighbour_set_points, radius, method='ror')

    plt.figure()
    plt.title("Image After Applying ELBP")
    plt.imshow(patterns, interpolation='nearest')
    plt.gray()

    # A histogram is being generated from the output to visualize the output from the ELBP model
    hist, _ = np.histogram(patterns, bins=np.arange(2 ** 3 + 1), density=True)

    # A bar graph is being generated from the output to visualize
    plt.figure()
    plt.title("Image Graph after ELBP")
    plt.plot(hist)
    plt.ylim([0, hist.max()])
    plt.xlim([0, 7])
    plt.show()

    return


def histogram_of_oriented_gradients_model(image_id, image_matrix):
    print("\nHistogram of Oriented Gradients Result for Image " + str(image_id) + "\n")

    # A Histogram of Oriented Gradients is being applied with the following arguments as seen below
    fd, hog_image = hog(image_matrix, orientations=9, pixels_per_cell=(8, 8),
                        cells_per_block=(8, 8), block_norm='L2-Hys', visualize=True, multichannel=0)

    # A bar graph is being generated from the output to visualize
    fig, ax2 = plt.subplots(figsize=(8, 4), sharex=True, sharey=True)
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

    ax2.imshow(hog_image_rescaled)
    ax2.set_title('Image After Applying HOG')
    plt.show()

    return


image_id = int(input("\nPlease Enter Image ID {0 - 399}: "))
model_name = input("\nEnter Model Names from\n- color_moment\n- local_binary_pattern\n- "
                   "histogram_of_oriented_gradients\n\nPlease Enter Model Name: ")
apply_model_on_olivetti_face(image_id, model_name)

# apply_model_on_olivetti_face(100, "local_binary_pattern")
# apply_model_on_olivetti_face(100, "color_moment")
# apply_model_on_olivetti_face(0, "histogram_of_oriented_gradients")
