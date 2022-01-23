import pandas as pd
import sys
import os

curDir = os.path.dirname(os.path.realpath(
    __file__))  # add directory above current one so that our files from the parent directoery can be referenced.
upDir = os.path.dirname(curDir)
sys.path.append(upDir)
import database
import helper


def task1(first_image_folder, second_image_folder, feature_model, dimensionality_reduction, classifier='SVM',
          k=5):
    X_train_id = first_image_folder + '-' + feature_model + '-' + dimensionality_reduction + '-' + classifier + '-' + \
                 str(k) + '-X_train'
    Y_train_id = first_image_folder + '-' + feature_model + '-' + dimensionality_reduction + '-' + classifier + '-' + \
                 str(k) + '-Y_train'
    print("\n\n\n\nSTARTING TASK 1...\n\n\n")
    # TRAINING PHASE IMAGES

    if database.check_if_latent_semantic_exist(X_train_id):
        print("Fetching Already Saved Latent Semantics")
        X_train = database.get_latent_semantic(X_train_id)
        images_names = database.get_latent_semantic(Y_train_id)
        Y_train = helper.get_image_labels(images_names, 1, "Training")
    else:
        all_images = helper.fetch_training_images(first_image_folder, feature_model)
        object_feature_matrix = helper.get_object_features_matrix(all_images, feature_model)
        data_frame = pd.DataFrame(object_feature_matrix, index=all_images.keys())
        reduction_algorithm_output = helper.get_dimensionality_reduction(dimensionality_reduction, data_frame, k)

        Y_train = helper.get_image_labels(all_images, 1, "Training")
        X_train = reduction_algorithm_output[0]
        if dimensionality_reduction != "ALL":
            database.save_latent_semantics(
                first_image_folder + '-' + feature_model + '-' + dimensionality_reduction + '-'
                + classifier + '-' + str(k) + '-X_train', X_train)
            database.save_latent_semantics(
                first_image_folder + '-' + feature_model + '-' + dimensionality_reduction + '-'
                + classifier + '-' + str(k) + '-Y_train', list(all_images.keys()))
    clf = helper.train_classifier(X_train, Y_train, classifier)

    # TESTING PHASE IMAGES

    test_images = helper.fetch_testing_images(second_image_folder, feature_model)
    test_object_feature_matrix = helper.get_object_features_matrix(test_images, feature_model)
    test_data_frame = pd.DataFrame(test_object_feature_matrix, index=test_images.keys())
    test_reduction_algorithm_output = helper.get_dimensionality_reduction(dimensionality_reduction, test_data_frame, k)

    X_test = test_reduction_algorithm_output[0]
    Y_test = helper.get_image_labels(test_images, 1, "Testing")
    results = helper.test_classifier(clf, X_test, classifier)

    helper.display_results(Y_test, results)
    print("\n\n\n\nTASK ENDING...")


if __name__ == '__main__':
    """
    training_folder_path = input("\nPlease Provide Image Folder Path for Training")
    testing_folder_path = input("\nPlease Provide Image Folder Path for Testing")
    model_name = input("\nEnter Feature Model Names From\n- color_moment\n- local_binary_pattern\n- "
                    "histogram_of_oriented_gradients\n\nPlease Enter Model Name: ")
    dimensionality_reduc = input("\nEnter Dimensionality Reduction Model Names From\n- PCA\n\nPlease Enter Model Name: ")
    image_classifier = input("\nEnter Image Classifier Names From\n- SVM\n- DT\n- PPR\n\nPlease Enter Classifier Name: ")
    k = int(input("\nPlease Enter K-Value: "))
    task1(training_folder_path, testing_folder_path, model_name, dimensionality_reduc, image_classifier, k)"""
    # task1('/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web Databases/Project/Phase3/CSE515Phase3/train',
    #       '/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web Databases/Project/Phase3/CSE515Phase3/test',
    #       'local_binary_pattern', 'PCA', 'SVM', 4)

    #task1('E:\Downloads\phase2_data\\test_1','E:\Downloads\phase2_data\\test_2','local_binary_pattern', 'PCA', 'PPR', 4)

    training_folder_path = input("\nPlease Provide Image Folder Path for Training")
    testing_folder_path = input("\nPlease Provide Image Folder Path for Testing")
    model_name = input("\nEnter Feature Model Names From\n- color_moment\n- local_binary_pattern\n- "
                    "histogram_of_oriented_gradients\n\nPlease Enter Model Name: ")
    dimensionality_reduc = input("\nEnter Dimensionality Reduction Model Names From\n- PCA\n- ALL\n\nPlease Enter Model Name: ")
    image_classifier = input("\nEnter Image Classifier Names From\n- SVM\n- DT\n- PPR\n\nPlease Enter Classifier Name: ")
    k = int(input("\nPlease Enter K-Value: "))
    task1(training_folder_path, testing_folder_path, model_name, dimensionality_reduc, image_classifier, k)
    # task1('/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web Databases/Project/Phase3/train',
    #       '/Users/keenan/Desktop/ASU/Semester 1/Multimedia and Web Databases/Project/Phase3/test',
    #       'local_binary_pattern', 'PCA', 'SVM', 4)
    #task1('E:\Downloads\phase2_data\\test_1','E:\Downloads\phase2_data\\test_2','local_binary_pattern', 'PCA', 'PPR', 4)
