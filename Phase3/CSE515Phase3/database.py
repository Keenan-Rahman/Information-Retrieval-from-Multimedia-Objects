import glob
import pickle
import numpy as np
import features

from bson.binary import Binary
from matplotlib import image as Image
from pymongo import MongoClient


def get_db():
    client = MongoClient("mongodb://localhost:27017/?readPreference=primary&appname=MongoDB%20Compass&ssl=false")
    db = client.admin
    mwdb_database = client["mwdb_database_phase_3"]
    return mwdb_database


def insert_image_dataset(folder_path):
    images_dataset = get_db().mwdb_database["image_data"]
    for img in glob.glob(folder_path + "/image-*.png"):
        images_dataset.insert_one({
            "_id": img.split("/")[len(img.split("/")) - 1],
            "image_matrix": Binary(pickle.dumps(Image.imread(img), protocol=2))
        })


def save_latent_semantics(id, latent_semantic_matrix):
    dataset = get_db().mwdb_database["latent_semantics"]
    dataset.insert_one({
        "_id": id,
        "latent_semantic": Binary(pickle.dumps(latent_semantic_matrix, protocol=2))
    })


def check_if_latent_semantic_exist(id):
    dataset = get_db().mwdb_database["latent_semantics"]
    if dataset.find({'_id': id}).count() > 0:
        return True
    return False


def get_latent_semantic(id):
    dataset = get_db().mwdb_database["latent_semantics"]
    latent_semantics = dataset.find(
        #{"_id": {"$regex": id}})
        {'_id': id})
    for single_value in latent_semantics:
        return pickle.loads(single_value["latent_semantic"])


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


def get_image_data_by_sample(Z):
    image_dictionary = {}
    images_dataset = get_db().mwdb_database["image_data"]
    image_data = images_dataset.find({"_id": {"$regex": ".*image-.*-.*-" + str(Z) + ".png"}})
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


def get_all_images_by_feature_descriptor(feature_model):
    image_dictionary = {}
    images_dataset = get_db().mwdb_database["image_features"]
    image_data = images_dataset.find({"_id": {"$regex": feature_model + ".*.png"}})
    for image in image_data:
        image_dictionary[image["_id"]] = pickle.loads(image["matrix"])
    return image_dictionary


def create_feature_reduction_for_all_images():
    dataset = get_db().mwdb_database["image_features"]
    all_images = get_all_images()
    for image_name, image in all_images.items():
        feature1 = features.cm_8x8_image(image)[0]
        feature2 = features.elbp_image(image)
        feature3 = features.hog_image(image)
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
    image_data = images_dataset.find({"_id": {"$regex": feature_model + ".*image-" + X + "-.*.png"}})
    for image in image_data:
        image_dictionary[image["_id"]] = np.array(pickle.loads(image["matrix"]))
    return image_dictionary


def get_image_data_by_subject_and_feature_descriptors(Y, feature_model):
    image_dictionary = {}
    images_dataset = get_db().mwdb_database["image_features"]
    image_data = images_dataset.find({"_id": {"$regex": feature_model + ".*image-.*-" + str(Y) + "-.*.png"}})
    for image in image_data:
        image_dictionary[image["_id"]] = np.array(pickle.loads(image["matrix"]))
    return image_dictionary
