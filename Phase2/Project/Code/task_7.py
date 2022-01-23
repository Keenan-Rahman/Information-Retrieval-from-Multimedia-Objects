import sys
import os
import pandas as pd
import pathlib
import glob
import numpy as np
import SVD
import pca
import csv
from PIL import Image as im
from features import Features
from matplotlib import image
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn import preprocessing
import math as m
import database
from numpy import math
import matplotlib.pyplot as plt

# database.insert_image_dataset("C:/Users/ipbol/Downloads/phase2_data/all")

"""
 Task 7
This task takes in a query image and latent semantics file and, by weighing each
of the latent semantics, attempts to calculate the most likely subject ID
for the given query image. The provided image may not be present in the database.

This program was made with contributions from Preston Mott's work in Task 5, and
Brandon Bayles' work in Task 6.

This code was written with an assumption that the input latent semantics file is
corresponding to that generated when running task 1, and as such has the same
file naming convention.
"""


def task_7(query_image_filename, latent_semantics_file):
    # parse input files for relevant information
    model, subject_id, k, reduction_method = latent_semantics_file.split("-")

    # remove absolute path fragments to retrive model
    splitPath = model.split("/")
    model = splitPath[len(splitPath)-1]
    windows_splitPath = model.split("\\")
    model = windows_splitPath[len(windows_splitPath)-1]
    # remove trailing file type (.csv)
    reduction_method = reduction_method[:-4]

    # retrieve all test images from MongoDB database
    image_database = database.get_all_images()

    # normalize the query image
    query_numpy_image = (np.asarray(im.open(query_image_filename))) / 255

    # add query to image database (in memory)
    image_database[query_image_filename] = query_numpy_image

    # generate a feature matrix database by running the provided feature model (derived from the LS filename)
    # on each image in the database
    feature_matrix = apply_feature_model(model, image_database)
    df = pd.DataFrame(feature_matrix, index=image_database.keys())

    # apply the provided dimensionality reduction type to the feature database
    dimensional_reduction_results = apply_dimensional_reduction(reduction_method, df, int(k))

    # create a Latent Semantic dataframe from the dimensionality reduction results
    columns = ['Latent-Semantic-' + str(i) for i in range(1, int(k) + 1)]
    data_frame = pd.DataFrame(dimensional_reduction_results[0], index=image_database.keys(), columns=columns)

    # convert to dictionary for image similarity calculations
    latent_semantic_dict = {}
    for i in range(len(dimensional_reduction_results[0])):
        latent_semantic_dict[list(image_database.keys())[i]] = dimensional_reduction_results[0][i]

    # find most similar images
    sim_dict = image_similarity(query_image_filename, latent_semantic_dict)

    # read contents of the latent semantics input file as a dataframe
    subject_weight_df = pd.read_csv(latent_semantics_file, index_col=0)

    # calculate the most likely subject ID for the given query image using the similar images output,,
    # latent semantics dataframe, and the query image
    most_likely_subject = find_most_likely_subject(sim_dict, subject_weight_df, query_image_filename)

    print("\n\nMost likely subject: " + most_likely_subject + "\n")
    return most_likely_subject


def apply_feature_model(feature_model, images):
    feature_model_to_matrix_sie = {"color_moment": 64, "local_binary_pattern": 4096,
                                   "histogram_of_oriented_gradients": 4096}
    feature_matrix = np.zeros((len(images), feature_model_to_matrix_sie[feature_model]))
    count = 0
    for img_id in images:
        thisNumpyImage = images[img_id]
        these_features = get_single_image_features(thisNumpyImage, feature_model)
        feature_matrix[count] = np.reshape(these_features, (1, len(these_features) * len(these_features[0])))
        count += 1
    return feature_matrix


def image_similarity(query_image_filename, latent_semantic_dict):
    image_similarities = {}
    for imgName, imgFeatures in latent_semantic_dict.items():
        if imgName != query_image_filename:
            img_type = imgName.split("-")[1]
            sim = np.linalg.norm(latent_semantic_dict[query_image_filename] - imgFeatures)
            image_similarities[imgName] = abs(sim)
    return image_similarities


def get_single_image_features(numpy_image, feature_model):
    features = Features()
    featuresOut = None
    if (feature_model == 'color_moment'):
        featuresOut = np.array([np.array(features.cm8x8Image(numpy_image)[0])])
    elif (feature_model == 'local_binary_pattern'):
        featuresOut = features.elbpImage(numpy_image)
    elif (feature_model == 'histogram_of_oriented_gradients'):
        featuresOut = features.hogsImage(numpy_image)
    else:
        print('ERROR: invalid model given.')
    return featuresOut


def apply_dimensional_reduction(reduction_method, object_feature_matrix, k):
    model_output = None
    if reduction_method == "PCA":
        model_output = pca.pca(k, object_feature_matrix)
    elif reduction_method == "SVD":
        model_output = SVD.SVD(object_feature_matrix, k)
    elif reduction_method == "LDA":
        lda = LatentDirichletAllocation(n_components=k, total_samples=len(object_feature_matrix)).fit(
            object_feature_matrix)
        model_output = [lda.transform(object_feature_matrix), lda.components_]
    elif reduction_method == "KMeans":
        k_means = KMeans(n_clusters=k, random_state=0).fit(object_feature_matrix)
        model_output = [k_means.transform(object_feature_matrix), k_means.cluster_centers_]
    else:
        print("No Dimensionality Reduction Model of the given name exist = " + reduction_method)
    return model_output


# Task 7: Calculates the most likely subject based on the provided latent semantics file
def find_most_likely_subject(sim_dict, subject_weight_pair, query_image_filename):
    # initialize dictionary of the average subject counts in the provided similarity dictionary to be populated
    subject_avgs = {}
    for i in range(1, 41):
        subject_avgs[i] = 0

    # calculate subject averages
    for img_subject in range(1, 41):
        subject_count = 0
        for item in sim_dict:
            item_subject = item.split("-")[2]

            # found subject ID in similarity dictionary, increment count
            # add value to averages dictionary
            if (int(item_subject) == int(img_subject)):
                subject_avgs[img_subject] += sim_dict[item]
                subject_count += 1

        # calculate average for subject IDs if they appear in the similarity dictionary
        if (subject_count != 0):
            subject_avgs[img_subject] = subject_avgs[img_subject] / subject_count

    # construct dataframe using provided subject_weight_pair dataframe, which is created
    # using the provided latent semantics file
    weighedDict = pd.DataFrame(columns=subject_weight_pair.columns, index=subject_weight_pair.index)

    # initialize weight value to assist in classification of subject IDs
    semanticWeight = 2

    # iterate through latent semantics dataframe, adding the value (weighted) at each cell
    for col in subject_weight_pair.columns:
        for subject_name in subject_weight_pair.index:
            # retrieve subject ID of iamge in processing
            _, sbj_id = subject_name.split('-')

            weighedDict.at[subject_name, col] = abs(
                subject_weight_pair.at[subject_name, col] * semanticWeight * subject_avgs[int(sbj_id)])

        # halve the weight to provide less preference to further latent semantics
        semanticWeight = semanticWeight / 2

    # initialize final dictionary for calculating weight sums of each subject ID
    totalsDict = {}
    for i in range(1, 41):
        totalsDict['Subject-' + str(i)] = 0

    # sum each weighted subject ID
    for subject_name in weighedDict.index:
        totalsDict[subject_name] = sum(weighedDict.loc[subject_name])

    # retrieve the largest value (most likely)
    visualizeScores(totalsDict, 'Weighed Similarity Scores')
    maxSubject = max(totalsDict, key=totalsDict.get)

    return maxSubject

def visualizeScores(dict, title):
    plt.bar([subject.split("-")[1] for subject in dict.keys()], dict.values())
    plt.gcf().set_size_inches(18.5, 10.5)
    plt.title(title)
    plt.show()


query_image = input("\nPlease input the filepath of your query image. ")
input_latent_file = input("\nPlease input the filepath of your latent feature .csv file. ")

task_7(query_image, input_latent_file)
