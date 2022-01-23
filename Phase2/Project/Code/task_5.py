import os
import pickle
import pandas as pd
import pathlib
import glob
import numpy as np
import SVD
import pca
import csv
from PIL import Image, ImageOps
from matplotlib.image import imread
from features import Features
from matplotlib import image, pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans
from sklearn import preprocessing
import math as m
import database
from operator import itemgetter

images = database.get_image_data_by_subject(2)


def task_5(query_image_filename, latent_semantic_file, n):
    latent_semantic_parser = latent_semantic_file.split("/")
    last_index = len(latent_semantic_parser) - 1
    latent_semantic_filename = latent_semantic_parser[last_index]
    model, img_type, k, reduction_method = latent_semantic_filename.split("-")
    splitPath = model.split("\\")
    model = splitPath[len(splitPath)-1]
    reduction_method = reduction_method[:-4]

    image_dictionary = database.get_all_images()
    query_numpy_image = np.asarray(Image.open(query_image_filename))
    query_numpy_image = query_numpy_image / 255
    image_dictionary[query_image_filename] = query_numpy_image

    query_feature_matrix = get_object_features_matrix(image_dictionary, model)
    df = pd.DataFrame(query_feature_matrix, index=image_dictionary.keys())
    reduction_algorithm_output = get_dimensionality_reduction(reduction_method, df, int(k))
    latent_semantic_dict = {}
    for i in range(len(reduction_algorithm_output[0])):
        latent_semantic_dict[list(image_dictionary.keys())[i]] = reduction_algorithm_output[0][i]

    sim_dict = image_similarity(query_image_filename, latent_semantic_dict)
    most_k_sim = dict(sorted(sim_dict.items(), key=itemgetter(1))[:n])

    source_image_dict = {}
    image_score_dict = {}
    for key, value in most_k_sim.items():
        source_image_dict[key] = image_dictionary[key]
        image_score_dict[key] = value
    print("Image rankings: ", image_score_dict)
    visualizekNearest(query_numpy_image, source_image_dict, n)
    return most_k_sim


def apply_feature_extraction(feature_model, image_array):
    features = Features()
    model_output = None

    if feature_model == "color_moment":
        model_output = np.array([np.array(features.cm8x8Image(image_array)[0])])
    elif feature_model == "local_binary_pattern":
        model_output = features.elbpImage(image_array)
    elif feature_model == "histogram_of_oriented_gradients":
        model_output = features.hogsImage(image_array)
    else:
        print("No Model of the given name exist = " + feature_model)
    return model_output


def get_dimensionality_reduction(dimensionality_reduction, object_feature_matrix, k):
    model_output = None
    if dimensionality_reduction == "PCA":
        model_output = pca.pca(k, object_feature_matrix)
    elif dimensionality_reduction == "SVD":
        model_output = SVD.SVD(object_feature_matrix, k)
    elif dimensionality_reduction == "LDA":
        lda = LatentDirichletAllocation(n_components=k, total_samples=len(object_feature_matrix)).fit(
            object_feature_matrix)
        model_output = [lda.transform(object_feature_matrix), lda.components_]
    elif dimensionality_reduction == "KMeans":
        k_means = KMeans(n_clusters=k, random_state=0).fit(object_feature_matrix)
        model_output = [k_means.transform(object_feature_matrix), k_means.cluster_centers_]
    else:
        print("No Dimensionality Reduction Model of the given name exist = " + dimensionality_reduction)
    return model_output


def get_object_features_matrix(image_dictionary, feature_model):
    feature_model_to_matrix_sie = {"color_moment": 64, "local_binary_pattern": 4096,
                                   "histogram_of_oriented_gradients": 4096}
    object_feature_matrix = np.zeros((len(image_dictionary), feature_model_to_matrix_sie[feature_model]))
    count = 0
    for image_id in image_dictionary:
        image_array = image_dictionary[image_id]
        feature_extraction = apply_feature_extraction(feature_model, image_array)
        object_feature_matrix[count] = np.reshape(feature_extraction,
                                                  (1, len(feature_extraction) * len(feature_extraction[0])))
        count += 1
    return object_feature_matrix


def image_similarity(query_image_filename, latent_semantic_dict):
    print('Calculating image similarities to query image... ')
    image_similarities = {}
    for imgName, imgFeatures in latent_semantic_dict.items():
        if imgName != query_image_filename:
            sim = np.linalg.norm(latent_semantic_dict[query_image_filename] - imgFeatures)
            image_similarities[imgName] = sim
    return image_similarities


def visualizekNearest(query_image, sim_dictionary, n):
    fig, tup = plt.subplots(n + 1, 1, figsize=(16, 16))
    tup[0].axis('off')
    tup[0].imshow(query_image, cmap=plt.cm.gray)
    tup[0].set_title("Most N similar images")

    counter = 0
    for x in tup[1:]:
        currFile = str(list(sim_dictionary.keys())[counter])
        im = list(sim_dictionary.values())[counter]
        x.axis('off')
        x.set_title(currFile + " has rank number " + str(counter + 1))
        x.imshow(im, cmap=plt.cm.gray)
        counter += 1
    plt.show()


query_image = input("\nPlease input the filepath of your query image. ")
input_latent_file = input("\nPlease input the filepath of your latent feature .csv file. ")
n = input("\nPlease input how many results you would like to display. ")

task_5(query_image, input_latent_file, int(n))

# task_5("all/image-cc-1-1.png", "color_moment-smooth-4-KMeans.csv", 7)
