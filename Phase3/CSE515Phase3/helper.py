import features
import reduction.pca as pca
import reduction.svd as svd
import numpy as np
import glob
import database
from collections import Counter
import classifier.svm as svm
import classifier.decision_tree as decision_tree
import classifier.ppr as ppr
from matplotlib import image as Image


def apply_feature_extraction(feature_model, image_array):
    model_output = None

    if feature_model == "color_moment":
        model_output = np.array([np.array(features.cm_8x8_image(image_array)[0])])
    elif feature_model == "local_binary_pattern":
        model_output = features.elbp_image(image_array)
    elif feature_model == "histogram_of_oriented_gradients":
        model_output = features.hog_image(image_array)
    else:
        print("No Model of the given name exist = " + feature_model)
    return model_output


def train_classifier(X_train, Y_train, classifier):
    print("Applying Image Classifier on Training images.")
    clf = None
    if classifier == "SVM":
        clf = svm.SVM(C=0.001, max_iteration=1000, tol=0.001)
        print("Image Classifier on Training started.")
        clf.fit(X_train, Y_train)
    elif classifier == "PPR":
        clf = ppr.PPR(X_train, Y_train)
    elif classifier == "DT":
        clf = decision_tree.train_tree(X_train, Y_train)
    else:
        print("No Classifier of the given name exist = " + classifier)
    print("Image Classifier on Training done.")
    return clf


def test_classifier(clf, X_test, classifier):
    print("Performing Analysis on Testing images.")
    result = None
    if classifier == "SVM":
        result = clf.predict(X_test)
    elif classifier == "PPR":
        print("PPR")
        result = clf.predict(X_test)
    elif classifier == "DT":
        result = decision_tree.test_tree(X_test, clf)
    else:
        print("No Classifier of the given name exist = " + classifier)
    return result


def get_object_features_matrix(image_dictionary, feature_model):
    print("Applying " + feature_model + " Feature Descriptors on Testing images")
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
    print("Feature Descriptors on Testing images Done.")
    return object_feature_matrix


def get_obj_features_matrix(image_dictionary, feature_model):
    feature_model_to_matrix_sie = {"color_moment": 64, "local_binary_pattern": 4096,
                                   "histogram_of_oriented_gradients": 4096}
    object_feature_matrix = np.zeros((len(image_dictionary), feature_model_to_matrix_sie[feature_model]))
    count = 0
    for image_id in image_dictionary:
        image_array = image_dictionary[image_id]
        feature_extraction = image_array.copy()
        if feature_model == 'color_moment':
            object_feature_matrix[count] = np.reshape(feature_extraction, (1, len(feature_extraction)))
        else:
            object_feature_matrix[count] = np.reshape(feature_extraction,
                                                      (1, len(feature_extraction) * len(feature_extraction[0])))
        count += 1
    return object_feature_matrix


def get_dimensionality_reduction(dimensionality_reduction, object_feature_matrix, k):
    print("Applying " + dimensionality_reduction + " Dimensionality Reduction on Testing images")
    model_output = object_feature_matrix

    if dimensionality_reduction == "PCA":
        model_output = pca.pca(k, object_feature_matrix)
    elif dimensionality_reduction == "SVD":
        model_output = svd.svd(object_feature_matrix, k)
    elif dimensionality_reduction == "ALL":
        model_output = [object_feature_matrix.to_numpy()]
        print("Not Applying any reduction on Images")
    else:
        print("No Dimensionality Reduction Model of the given name exist = " + dimensionality_reduction)
    print("Dimensionality Reduction on Testing images Done.")
    return model_output


def fetch_training_images(first_image_folder, feature_model):
    all_images = {}
    print("Fetching Training images from folder: " + first_image_folder)
    if first_image_folder == 'None':
        all_images = database.get_all_images_by_feature_descriptor(feature_model)
    else:
        for img in glob.glob(first_image_folder + "/image-*.png"):
            all_images[img.split("/")[len(img.split("/")) - 1]] = Image.imread(img)
    print("All Training images are fetched.")
    return all_images


def fetch_testing_images(second_image_folder, feature_model):
    test_images = {}
    print("Fetching Testing images from folder: " + second_image_folder)
    if second_image_folder == 'None':
        test_images = database.get_all_images_by_feature_descriptor(feature_model)
    else:
        for img in glob.glob(second_image_folder + "/image-*.png"):
            test_images[img.split("/")[len(img.split("/")) - 1]] = Image.imread(img)
    print("All Testing images are fetched.")
    return test_images


def get_image_labels(images, task_id, analysis_type):
    Y_train = []
    for key in images:
        name = key.split('-')
        if task_id == 3:
            name[task_id] = name[task_id][:len(name[task_id]) - 4]
        Y_train.append(name[task_id])
    print("\n\n\n" + analysis_type + " Data Analysis: ")
    print(Counter(Y_train))
    return Y_train


def display_results(actual, predicted):
    print_list(actual, predicted)

    print("\n\n\n\nAnalysis Results\n\n")
    total_misses = 0
    total_correct = 0
    total_false_positives = 0
    distinct_feature = set.union(set(actual), set(predicted))
    for feature in distinct_feature:
        miss_score = 0
        correct_score = 0
        false_positive_score = 0
        for real, guess in zip(actual, predicted):
            if real == feature and guess != feature:
                miss_score += 1
            elif real == feature and guess == feature:
                correct_score += 1
            elif real != feature and guess == feature:
                false_positive_score += 1
        current_feature_total = miss_score + correct_score + false_positive_score
        print("\n\nFeature Selected: " + str(feature))
        print("Miss Score = " + str(miss_score) + " | Percentage = " + str(miss_score/current_feature_total * 100))
        print("Correct Score = " + str(correct_score) + " | Percentage = "  + str(correct_score/current_feature_total * 100))
        print("False Positive Score = " + str(false_positive_score) + " | Percentage = "  + str(false_positive_score/current_feature_total * 100))
        total_misses += miss_score
        total_correct += correct_score
        total_false_positives += false_positive_score
    total = total_misses + total_correct + total_false_positives
    print("\n\nTotal Analysis in No. = " + str(total))
    print("Miss Rate = " + str(total_misses))
    print("Correct Rate = " + str(total_correct))
    print("False Positive Rate = " + str(total_false_positives))
    print("\n\nTotal Analysis in %")
    print("Miss Rate = " + str(total_misses / total * 100))
    print("Correct Rate = " + str(total_correct / total * 100))
    print("False Positive Rate = " + str(total_false_positives / total * 100))


def print_list(actual, predicted):
    print("Actual = [", end='')
    for i in actual:
        print(str(i) + ", ", end='')
    print("]")
    print("Predicted = [", end='')
    for i in predicted:
        print(str(i) + ", ", end='')
    print("]")


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def euclidean_distance(value1, value2):
    return np.sum((np.array(value1) - np.array(value2)) ** 2)
