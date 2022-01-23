import glob
from matplotlib import image as Image
import pandas as pd
import numpy as np
import json
import os
import sys
from matplotlib import image as image
from pathlib import Path
curDir = os.path.dirname(os.path.realpath(__file__)) #add directory above current one so that our files from the parent directoery can be referenced.
upDir = os.path.dirname(curDir)
sys.path.append(upDir)
import helper
import lsh as lsh

def task_7(image_folder, query_image, feature_model, relevance_vector):
    all_images, query_image, query_image_name = fetch_images(image_folder, query_image)
    all_images_feature, query_image_feature = get_features(all_images, query_image, feature_model)

    k_images, lsh_index, vectors, lsh_range = get_k_nearest_images(all_images_feature, query_image_feature,
                                                                   feature_model, query_image_name, 25)

    print("The 25-nearest for the given query image are:\n\n")
    print(k_images)

    
    X_train, Y_train = get_training_data(all_images_feature, relevance_vector)
    all_nearest_images, nearest_images_with_score = get_all_nearest_images(lsh_index, vectors, lsh_range,
                                                                           all_images_feature,
                                                                           query_image_feature, query_image_name)
    X_test, X_images = get_test_data(all_images_feature, all_nearest_images)
    final_results = svm_relevance_feedback(X_train, Y_train, X_test, X_images, nearest_images_with_score)
    print_results(final_results)
    return final_results


def fetch_images(image_folder, query_image):
    all_images_dict = {}
    query_image_dict = {}
    query_image_name = ''
    for img in glob.glob(image_folder + "/image-*.png"):
        all_images_dict[Path(img).name] = image.imread(img)
    for q_img in glob.glob(query_image):
        query_image_name = Path(q_img).name
        query_image_dict[query_image_name] = image.imread(q_img)
    return all_images_dict, query_image_dict, query_image_name


def get_features(all_images, query_image, feature_model):
    object_feature_matrix = helper.get_object_features_matrix(all_images, feature_model)
    data_frame = pd.DataFrame(object_feature_matrix, index=all_images.keys())

    query_object_feature_matrix = helper.get_object_features_matrix(query_image, feature_model)
    query_data_frame = pd.DataFrame(query_object_feature_matrix, index=query_image.keys())

    images_feature = {}
    query_image_feature = {}
    for key, value in data_frame.iterrows():
        images_feature[key] = value.values
    for key, value in query_data_frame.iterrows():
        query_image_feature[key] = value.values
    return images_feature, query_image_feature


def get_k_nearest_images(all_images_feature, query_image_feature, feature_model, query_image_name, k):
    lsh_index, vectors, lsh_range = lsh.get_index_structure(5, 5, lsh.get_feature_length(feature_model), None,
                                                            all_images_feature)
    copy_dict = query_image_feature.copy()
    copy_dict.update(all_images_feature)
    t_images, _0, _1, _2, _3, _4 = lsh.get_t_nearest_images_using_LSH(5, 5, vectors, lsh_index, lsh_range,
                                                                  copy_dict,
                                                                  query_image_name, k)
    return list(t_images.keys()), lsh_index, vectors, lsh_range


def get_all_nearest_images(lsh_index, vectors, lsh_range, all_images_feature, query_image_feature, query_image_name):
    copy_dict = query_image_feature.copy()
    copy_dict.update(all_images_feature)
    all_images = lsh.get_all_nearest_images_using_LSH(5, 5, vectors, lsh_index, lsh_range,
                                                      copy_dict,
                                                      query_image_name)
    return list(all_images.keys()), all_images


def get_training_data(all_images_feature, relevance_vector):
    # relevant = relevant_images.replace('\'', '').replace(' ', '').split(",")
    # irrelevant = irrelevant_images.replace('\'', '').replace(' ', '').split(",")
    relevant = irrelevant = []
    for image in relevance_vector:
        if(image[2] == 0):
            irrelevant.append(image[0])
        else:
            relevant.append(image[0])
    labels = ["relevant", "irrelevant"]
    X_train = []
    Y_train = []
    for image in all_images_feature:
        if image in relevant:
            X_train.append(all_images_feature[image])
            Y_train.append(labels[0])
        elif image in irrelevant:
            X_train.append(all_images_feature[image])
            Y_train.append(labels[1])
    return X_train, Y_train


def get_test_data(all_images_feature, all_nearest_images):
    X_test = []
    X_images = []
    for image in all_images_feature:
        if image in all_nearest_images:
            X_test.append(all_images_feature[image])
            X_images.append(image)
    return X_test, X_images


def svm_relevance_feedback(X_train, Y_train, X_test, X_images, nearest_images_with_score):
    final_results = {}
    clf = helper.train_classifier(np.array(X_train), Y_train, "SVM")
    results = helper.test_classifier(clf, np.array(X_test), "SVM")
    for result, image_name in zip(results, X_images):
        if result == "relevant":
            final_results[image_name] = nearest_images_with_score[image_name]
    final_results = sorted(final_results.items(), key=lambda kv: kv[1])
    return final_results


def print_results(final_results):
    print("Relevant Images based on User Feedback\n\n")
    print(json.dumps(final_results, indent=2))


# images_path = input("\nPlease Provide Input Images Path: ")
# q_image_path = input("\nPlease Provide Query Image Path: ")
# model_name = input("\nEnter Feature Model Names From\n- color_moment\n- local_binary_pattern\n- "
#                    "histogram_of_oriented_gradients\n\nPlease Enter Model Name: ")

# task_7(images_path, q_image_path, model_name)

# task_7("/Users/keenan/Desktop/test/train", "/Users/keenan/Desktop/test/all_images/image-jitter-11-10.png",
# "local_binary_pattern")
