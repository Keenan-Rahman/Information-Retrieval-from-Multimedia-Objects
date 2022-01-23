import os
import pandas as pd
import pathlib
import numpy as np
import SVD
import pca
import database

from features import Features
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.cluster import KMeans


def task1(feature_model, X, k, dimensionality_reduction):
    image_dictionary = database.get_image_data_by_type(X)
    object_feature_matrix = get_object_features_matrix(image_dictionary, feature_model)
    data_frame = pd.DataFrame(object_feature_matrix, index=image_dictionary.keys())
    reduction_algorithm_output = get_dimensionality_reduction(dimensionality_reduction, data_frame, k)

    columns = ['Latent-Semantic-' + str(i) for i in range(1, k + 1)]
    data_frame = pd.DataFrame(reduction_algorithm_output[0], index=image_dictionary.keys(), columns=columns)
    database.save_latent_semantics_to_feature_matrix("image_weight_pair-Type-" + X + "-" +
                                                     feature_model + "-" + str(k) + "-" + dimensionality_reduction,
                                                     data_frame)
    subject_weight_pair = create_subject_weight_pair(data_frame, X, columns)
    filename = create_output_file(subject_weight_pair, feature_model, X, k, dimensionality_reduction)
    database.save_latent_semantics_to_feature_matrix(filename, reduction_algorithm_output[1])


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


def create_subject_weight_pair(data_frame, X, columns):
    subjects = [subject_id for subject_id in range(1, 41)]
    subject_weight_pair = pd.DataFrame(columns=columns)
    for subject_id in subjects:
        sub_string = "image-" + X + "-" + str(subject_id) + "-"
        temp_dataframe = data_frame[data_frame.index.str.contains(sub_string)]
        subject_weight_pair.loc['Subject-' + str(subject_id)] = [value / len(temp_dataframe) for value in
                                                                 temp_dataframe.sum().values]
    return subject_weight_pair.sort_values(by=columns, ascending=False)


def create_output_file(subject_weight_pair, feature_model, X, k, dimensionality_reduction):
    path = str(pathlib.Path().resolve()) + '/output/task_1'
    filename = feature_model + '-' + X + '-' + str(k) + '-' + dimensionality_reduction + '.csv'
    if not os.path.exists(path):
        os.makedirs(path)
    file = open(path + '/' + filename, 'w+')
    file.write(subject_weight_pair.to_csv(index=True, header=True, sep=',', index_label='Subjects'))
    return filename

model_name = input("\nEnter Feature Model Names From\n- color_moment\n- local_binary_pattern\n- "
                   "histogram_of_oriented_gradients\n\nPlease Enter Model Name: ")
X = input("\nEnter Values of X From: \n- cc\n- con\n- detail\n- emboss\n- jitter\n- neg\n- noise1"
          "\n- noise2\n- original\n- poster\n- rot\n- smooth\n- stipple\n\nPlease Enter Value of X: ")
k = int(input("\nPlease Enter K-Value: "))
dimensionality_reduction = input("\nEnter Dimensionality Reduction Model Names From\n- PCA\n- SVD\n- "
                                 "LDA\n- KMeans\n\nPlease Enter Model Name: ")

task1(model_name, X, k, dimensionality_reduction)


#
# task1("local_binary_pattern",
#       "smooth",
#       4,
#       "KMeans")
