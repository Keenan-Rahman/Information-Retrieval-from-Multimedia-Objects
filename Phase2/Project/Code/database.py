import glob
import pickle

import numpy as np
from bson.binary import Binary
from matplotlib import image as Image
from pymongo import MongoClient
from features import Features


def get_db():
    client = MongoClient("mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false")
    db = client.admin
    mwdb_database = client["mwdb_database_phase_2"]
    return mwdb_database


def insert_image_dataset(folder_path):
    images_dataset = get_db().mwdb_database["image_data"]
    for img in glob.glob(folder_path + "/image-*.png"):
        images_dataset.insert_one({
            "_id": img.split("/")[len(img.split("/")) - 1],
            "image_matrix": Binary(pickle.dumps(Image.imread(img), protocol=2))
        })


def get_image_data_by_type(X):
    image_dictionary = {}
    images_dataset = get_db().mwdb_database["image_data"]
    image_data = images_dataset.find({"_id": {"$regex": ".*image-" + X + "-.*.png"}})
    for image in image_data:
        image_dictionary[image["_id"]] = pickle.loads(image["image_matrix"])
    return image_dictionary


def get_image_data_by_subject(Y):
    image_dictionary = {}
    images_dataset = get_db().mwdb_database["image_data"]
    image_data = images_dataset.find({"_id": {"$regex": ".*image-.*-" + str(Y) + "-.*.png"}})
    for image in image_data:
        image_dictionary[image["_id"]] = pickle.loads(image["image_matrix"])
    return image_dictionary


def get_all_images():
    image_dictionary = {}
    images_dataset = get_db().mwdb_database["image_data"]
    image_data = images_dataset.find({"_id": {"$regex": "image.*.png"}})
    for image in image_data:
        image_dictionary[image["_id"]] = pickle.loads(image["image_matrix"])
    return image_dictionary


def save_latent_semantics_to_feature_matrix(filename, latent_semantic_and_feature_matrix):
    dataset = get_db().mwdb_database["reduction_algorithm_output"]
    dataset.insert_one({
        "_id": filename,
        "latent_semantic_and_feature_matrix": Binary(pickle.dumps(latent_semantic_and_feature_matrix, protocol=2))
    })


def get_type_latent_semantics_matrix(feature_model, X, k, dimensionality_reduction):
    dataset = get_db().mwdb_database["reduction_algorithm_output"]
    matrix = dataset.find({"_id": {"$regex": ".*image_weight_pair-Type-" + X + "-" + feature_model + "-" + str(
        k) + "-" + dimensionality_reduction}})
    for single_value in matrix:
        return pickle.loads(single_value["latent_semantic_and_feature_matrix"])


def get_subject_latent_semantics_matrix(feature_model, Y, k, dimensionality_reduction):
    dataset = get_db().mwdb_database["reduction_algorithm_output"]
    matrix = dataset.find({"_id": {"$regex": ".*image_weight_pair-Subject-" + str(Y) + "-" + feature_model + "-" + str(
        k) + "-" + dimensionality_reduction}})
    for single_value in matrix:
        return pickle.loads(single_value["latent_semantic_and_feature_matrix"])


def save_similarity_matrix(filename, weight_pair):
    dataset = get_db().mwdb_database["matrix_similarity"]
    dataset.insert_one({
        "_id": filename,
        "matrix": Binary(pickle.dumps(weight_pair, protocol=2))
    })


def get_subject_similarity_matrix_by_feature_model_k_dimensionality_reduction(feature_model, k,
                                                                              dimensionality_reduction):
    dataset = get_db().mwdb_database["matrix_similarity"]
    matrix = dataset.find(
        {"_id": {"$regex": "subject-subject-" + feature_model + "-" + str(k) + "-" + dimensionality_reduction}})
    for single_value in matrix:
        return pickle.loads(single_value["matrix"])


def get_subject_similarity_matrix_by_id(id):
    dataset = get_db().mwdb_database["matrix_similarity"]
    matrix = dataset.find({"_id": id})
    for single_value in matrix:
        return pickle.loads(single_value["matrix"])


def create_feature_reduction_for_all_images():
    dataset = get_db().mwdb_database["image_features"]
    features = Features()
    all_images = get_all_images()
    for image_name, image in all_images.items():
        feature1 = features.cm8x8Image(image)[0]
        feature2 = features.elbpImage(image)
        feature3 = features.hogsImage(image)
        dataset.insert_one({
            "_id": "color_moment" + image_name,
            "matrix": Binary(pickle.dumps(feature1, protocol=2))
        })
        dataset.insert_one({
            "_id": "local_binary_pattern" + image_name,
            "matrix": Binary(pickle.dumps(feature2, protocol=2))
        })
        dataset.insert_one({
            "_id": "histogram_of_oriented_gradients" + image_name,
            "matrix": Binary(pickle.dumps(feature3, protocol=2))
        })


def get_image_feature_descriptors(feature_model, image_name):
    dataset = get_db().mwdb_database["image_features"]
    matrix = dataset.find({"_id": feature_model + image_name})
    for single_value in matrix:
        return pickle.loads(single_value["matrix"])


def get_image_data_by_type_and_feature_descriptors(X, feature_model):
    image_dictionary = {}
    images_dataset = get_db().mwdb_database["image_features"]
    image_data = images_dataset.find({"_id": {"$regex": feature_model+".*image-" + X + "-.*.png"}})
    for image in image_data:
        image_dictionary[image["_id"]] = np.array(pickle.loads(image["matrix"]))
    return image_dictionary


def get_image_data_by_subject_and_feature_descriptors(Y, feature_model):
    image_dictionary = {}
    images_dataset = get_db().mwdb_database["image_features"]
    image_data = images_dataset.find({"_id": {"$regex": feature_model+".*image-.*-" + str(Y) + "-.*.png"}})
    for image in image_data:
        image_dictionary[image["_id"]] = np.array(pickle.loads(image["matrix"]))
    return image_dictionary

# create_database("/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web Databases/Project/Phase-2/all_images")
# get_image_data_by_type("con")
# get_image_data_by_subject(2)
# create_database("/home/preston/Desktop/CSE515/Phase2/CSE515Phase2")
# insert_image_dataset("E:\Downloads\phase2_data\\all")
# d = get_image_data_by_type("con")
# print(d)
# d = get_image_data_by_subject(2)
# print(d)

# images_dataset = get_db().mwdb_database["image_data"]
# images_dataset.delete_many({})
# insert_image_dataset("all")
# d = get_all_images()
# print(d)
