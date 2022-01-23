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


def task4(feature_model, k, dimensionality_reduction):
    subjects = [subject_id for subject_id in range(1, 41)]

    subject_subject_similarity_matrix = get_similarity_matrix(subjects, feature_model)
    reduction_algorithm_output = get_dimensionality_reduction(dimensionality_reduction,
                                                              subject_subject_similarity_matrix, k)
    subject_weight_pair = create_subject_weight_pair(reduction_algorithm_output, subjects, k)
    filename = create_output_file(subject_weight_pair, feature_model, k, dimensionality_reduction)
    database.save_similarity_matrix(filename, subject_subject_similarity_matrix)


def get_similarity_matrix(subjects, feature_model):
    subject_subject_similarity_matrix = pd.DataFrame(columns=subjects, index=subjects)

    for subject1 in subjects:
        for subject2 in subjects:
            if not math.isnan(subject_subject_similarity_matrix[subject1][subject2]):
                continue
            if subject1 == subject2:
                subject_subject_similarity_matrix[subject2][subject1] = 0
            else:
                subject_subject_value = get_subject_similarities(subject1, subject2, feature_model)
                subject_subject_similarity_matrix[subject1][subject2] = subject_subject_value
                subject_subject_similarity_matrix[subject2][subject1] = subject_subject_value
    return subject_subject_similarity_matrix


def get_subject_similarities(subject1, subject2, feature_model):
    images_for_subject1 = database.get_image_data_by_subject_and_feature_descriptors(subject1, feature_model)
    images_for_subject2 = database.get_image_data_by_subject_and_feature_descriptors(subject2, feature_model)

    if not bool(images_for_subject1) or not bool(images_for_subject2):
        return 1000

    for subject1_file_name, subject1_img_matrix in images_for_subject1.items():
        distance_list = []
        for subject2_file_name, subject2_img_matrix in images_for_subject2.items():
            similarity_value = np.linalg.norm(subject1_img_matrix - subject2_img_matrix)
            distance_list.append(similarity_value)

    subject_similarity = statistics.mean(distance_list)
    return subject_similarity


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


def create_subject_weight_pair(reduction_algorithm_output, subjects, k):
    columns = ['Latent-Semantic-' + str(i) for i in range(1, k + 1)]
    subject_weight_pair = pd.DataFrame(reduction_algorithm_output[0], index=subjects, columns=columns)
    return subject_weight_pair.sort_values(by=columns, ascending=False)


def create_output_file(subject_weight_pair, feature_model, k, dimensionality_reduction):
    path = str(pathlib.Path().resolve()) + '/output/task_4/'
    filename = 'subject-subject-' + feature_model + '-' + str(k) + '-' + dimensionality_reduction + '.csv'
    if not os.path.exists(path):
        os.makedirs(path)
    file = open(
        path + filename, 'w+')
    file.write(subject_weight_pair.to_csv(index=True, header=True, sep=',', index_label='Subject'))
    return filename


model_name = input("\nEnter Feature Model Names From\n- color_moment\n- local_binary_pattern\n- "
                   "histogram_of_oriented_gradients\n\nPlease Enter Model Name: ")
k = int(input("\nPlease Enter K-Value: "))
dimensionality_reduction = input("\nEnter Dimensionality Reduction Model Names From\n- PCA\n- SVD\n- "
                                 "LDA\n- KMeans\n\nPlease Enter Model Name: ")

task4(model_name, k, dimensionality_reduction)

# task4("local_binary_pattern", 4, "KMeans")
