import os
import pandas as pd
import pathlib
import numpy as np
import SVD
import database
import pca
import math
import statistics

from features import Features
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans


def task3(feature_model, k, dimensionality_reduction):
    print("Executing task 3 Started")
    types = ["cc", "con", "detail", "emboss", "jitter", "neg", "noise1", "noise2", "original", "poster", "rot",
             "smooth", "stipple"]

    type_type_similarity_matrix = get_similarity_matrix(types, feature_model)
    reduction_algorithm_output = get_dimensionality_reduction(dimensionality_reduction,
                                                              type_type_similarity_matrix, k)
    type_weight_pair = create_type_weight_pair(reduction_algorithm_output, types, k)
    filename = create_output_file(type_weight_pair, feature_model, k, dimensionality_reduction)
    database.save_similarity_matrix(filename, type_type_similarity_matrix)
    print("Task 3 Execution Completed")



def get_similarity_matrix(types, feature_model):
    type_type_similarity_matrix = pd.DataFrame(columns=types, index=types)

    for type1 in types:
        for type2 in types:
            if not math.isnan(type_type_similarity_matrix[type1][type2]):
                continue
            if type1 == type2:
                type_type_similarity_matrix[type2][type1] = 0
            else:
                type_type_value = get_type_similarities(type1, type2, feature_model)
                type_type_similarity_matrix[type1][type2] = type_type_value
                type_type_similarity_matrix[type2][type1] = type_type_value
    return type_type_similarity_matrix


def get_type_similarities(type1, type2, feature_model):
    images_for_type_1 = database.get_image_data_by_type_and_feature_descriptors(type1, feature_model)
    images_for_type_2 = database.get_image_data_by_type_and_feature_descriptors(type2, feature_model)

    if not bool(images_for_type_1) or not bool(images_for_type_2):
        return 1000

    for type1_file_name, type1_img_matrix in images_for_type_1.items():
        distance_list = []
        for type2_file_name, type2_img_matrix in images_for_type_2.items():
            similarity_value = np.linalg.norm(type1_img_matrix - type2_img_matrix)
            distance_list.append(similarity_value)

    type_similarity = statistics.mean(distance_list)
    return type_similarity


def apply_feature_extraction(feature_model, similarity_matrix):
    features = Features()
    model_output = None

    if feature_model == "color_moment":
        model_output = np.array([np.array(features.cm8x8Image(similarity_matrix)[0])])
    elif feature_model == "local_binary_pattern":
        model_output = features.elbpImage(similarity_matrix)
    elif feature_model == "histogram_of_oriented_gradients":
        model_output = features.hogsImage(similarity_matrix)
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
        model_output = [lda.transform(object_feature_matrix)]
    elif dimensionality_reduction == "KMeans":
        k_means = KMeans(n_clusters=k, random_state=0).fit(object_feature_matrix)
        model_output = [k_means.transform(object_feature_matrix)]
    else:
        print("No Dimensionality Reduction Model of the given name exist = " + dimensionality_reduction)
    return model_output


def create_type_weight_pair(reduction_algorithm_output, types, k):
    columns = ['Latent-Semantic-' + str(i) for i in range(1, k + 1)]
    type_weight_pair = pd.DataFrame(reduction_algorithm_output[0], index=types, columns=columns)
    return type_weight_pair.sort_values(by=columns, ascending=False)


def create_output_file(type_weight_pair, feature_model, k, dimensionality_reduction):
    path = str(pathlib.Path().resolve()) + '/output/task_3/'
    filename = 'type-type-' + feature_model + '-' + str(k) + '-' + dimensionality_reduction + '.csv'
    if not os.path.exists(path):
        os.makedirs(path)
    file = open(
        path + filename, 'w+')
    file.write(type_weight_pair.to_csv(index=True, header=True, sep=',', index_label='TYPE'))
    return filename


model_name = input("\nEnter Feature Model Names From\n- color_moment\n- local_binary_pattern\n- "
                   "histogram_of_oriented_gradients\n\nPlease Enter Model Name: ")
k = int(input("\nPlease Enter K-Value: "))
dimensionality_reduction = input("\nEnter Dimensionality Reduction Model Names From\n- PCA\n- SVD\n- "
                                 "LDA\n- KMeans\n\nPlease Enter Model Name: ")

task3(model_name, k, dimensionality_reduction)

# task3("local_binary_pattern", 4, "KMeans")
