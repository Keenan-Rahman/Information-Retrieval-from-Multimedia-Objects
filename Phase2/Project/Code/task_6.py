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

"""#########################
task_6.py
CSE 515
Project Phase II
This program takes a filename of a query image and a latent semantics file and then associates a type label to the image under the given latent semantics
#########################"""


def task_6(query_image_filename, latent_semantics_file):
    # In order to determine the type label for the given file, we need to use the weights of the given latent semantics file and apply them to averaged similarity to images of the different types using the same feature model
    model, subject_id, k, reduction_method = latent_semantics_file.split("-")
    splitPath = model.split("\\")
    model = splitPath[len(splitPath) - 1]
    reduction_method = reduction_method[:-4]
    image_database = database.get_all_images()
    query_numpy_image = (np.asarray(im.open(query_image_filename))) / 255
    image_database[query_image_filename] = query_numpy_image

    feature_matrix = apply_feature_model(model, image_database)
    df = pd.DataFrame(feature_matrix, index=image_database.keys())

    dimensional_reduction_results = apply_dimensional_reduction(reduction_method, df, int(k))
    columns = ['Latent-Semantic-' + str(i) for i in range(1, int(k) + 1)]
    data_frame = pd.DataFrame(dimensional_reduction_results[0], index=image_database.keys(), columns=columns)
    latent_semantic_dict = {}
    for i in range(len(dimensional_reduction_results[0])):
        latent_semantic_dict[list(image_database.keys())[i]] = dimensional_reduction_results[0][i]

    sim_dict = image_similarity(query_image_filename, latent_semantic_dict)
    type_weight_df = pd.read_csv(latent_semantics_file, index_col=0)
    most_likely_type = find_most_likely_type(sim_dict, type_weight_df, query_image_filename)

    print("\n\nMost likely type: " + most_likely_type + "\n")
    return most_likely_type


def apply_feature_model(feature_model, images):
    print("Calculating feature matrix...")
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
    print('Calculating image similarities to query image... ')
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
    if(feature_model == 'color_moment'):
        featuresOut = np.array([np.array(features.cm8x8Image(numpy_image)[0])])
    elif(feature_model == 'local_binary_pattern'):
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


def find_most_likely_type(sim_dict, type_weight_pair, query_image_filename):
    types = ["cc", "con", "detail", "emboss", "jitter", "neg", "noise1", "noise2", "original", "poster", "rot",
             "smooth", "stipple"]
    type_avgs = {"cc": 0, "con": 0, "detail": 0, "emboss": 0, "jitter": 0, "neg": 0, "noise1": 0, "noise2": 0,
                 "original": 0, "poster": 0, "rot": 0, "smooth": 0, "stipple": 0}
    for img_type in types:
        type_count = 0
        for item in sim_dict:
            item_type = item.split("-")[1]
            if (item_type == img_type):
                type_avgs[img_type] += sim_dict[item]
                type_count += 1
        if (type_count != 0):
            type_avgs[img_type] = type_avgs[img_type] / type_count
    #print("Unweighed similarity averages:\n")
    print(type_avgs)
    #visualizeScores(type_avgs, 'Unweighed Similarity Scores')
    print("\nSimilarity averages calculated, weighing by latent semantics")
    weighedDict = pd.DataFrame(columns=type_weight_pair.columns, index=type_weight_pair.index)
    semanticWeight = 2
    for col in type_weight_pair.columns:
        for type_name in type_weight_pair.index:
            weighedDict.at[type_name, col] = abs(
                type_weight_pair.at[type_name, col] * semanticWeight * type_avgs[type_name])
        semanticWeight = semanticWeight / 2
    print("\nSimilarities weighed by latent semantics:")
    print(weighedDict)
    print("\nCalculating most likely type...\n")
    totalsDict = {"cc": 0, "con": 0, "detail": 0, "emboss": 0, "jitter": 0, "neg": 0, "noise1": 0, "noise2": 0,
                  "original": 0, "poster": 0, "rot": 0, "smooth": 0, "stipple": 0}
    for type_name in weighedDict.index:
        totalsDict[type_name] = sum(weighedDict.loc[type_name])
    print("Type likeliness weighed scores: \n")
    print(totalsDict)
    print("\n")
    visualizeScores(totalsDict, 'Weighed Similarity Scores')
    maxType = max(totalsDict, key=totalsDict.get)
    print("Most likely type found...\n\n")
    return maxType

def visualizeScores(dict, title):
    plt.bar(dict.keys(), dict.values())
    plt.title(title)
    plt.show()

query_image_path = input('\nPlease enter path to query image: ')
latent_semantics_file = input('\nPlease enter path to latent semantics file: ')
task_6(query_image_path, latent_semantics_file)

# task_6('E:\Downloads\phase2_data\\all\\image-neg-25-4.png', 'E:\Documents\GitHub\CSE515Phase2\output\\task_2\local_binary_pattern-4-3-SVD.csv')
