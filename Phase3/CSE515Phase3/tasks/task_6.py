import pandas as pd
import sys
import os
import csv
import glob
from pathlib import Path
import numpy as np
from matplotlib import image, pyplot as plt
from PIL import Image
curDir = os.path.dirname(os.path.realpath(__file__)) #add directory above current one so that our files from the parent directoery can be referenced.
upDir = os.path.dirname(curDir)
sys.path.append(upDir)
import database
import helper
import classifier.decision_tree as decision_tree
"""Decision-tree-based relevance feedback: Implement a decision tree based relevance feedback system to improve nearest neighbor matches, which enables the user to label some of the results returned by the search task as relevant
or irrelevant and then returns a new set of ranked results, either by revising the query or by re-ordering the existing results.
"""

def task_6(rankDict, relevenaceVect):
    print("Starting task 6...")
    """ if inputPath is None:
        inputPath = input("Please provide path to input nearest neighbor search results file: ")

    testing_path = input("Please provide path to image folder: ")

    items = inputPath.split(')')
    nn_type = items[0]
    nn_type = nn_type.split("\\")
    nn_type = nn_type[len(nn_type)-1]
    nn_type = nn_type.split("/")
    nn_type = nn_type[len(nn_type)-1]
    feature_model = items[1]
    image_folder_path = items[2]
    image_folder_path = image_folder_path.replace("(", ":")
    image_folder_path = image_folder_path.replace("]", "/")
    image_folder_path = image_folder_path.replace(",", ".")
    query_image_path = Path(image_folder_path)
    image_folder_path = query_image_path.parent
    k_val = items[3]
    k_val = k_val.split('.')[0]
    inputPath = Path(inputPath)
    nn_scores = []
    with open(inputPath, 'r') as csvf:
        reader = csv.reader(csvf)
        nn_scores = list(reader)
    if len(nn_scores) == 0:
        print('Input file not able to be read, please ensure valid data input file')
        return
    print("Input file successfully collected, beginning relevance collection") """
    nn_scores = rankDict
    query_image = nn_scores[0]
    #visualizeImageComparison(query_image, nn_scores, int(k_val), image_folder_path)

    # scores that will create the tree
    """ relevance_scores = []
    for i in range(int(len(relevenaceVect))):
        cur_nn_score = nn_scores[i+1]
        #relevance = input('Is image ' + cur_nn_score[0] + ' relevant? Enter 1 for yes and 0 for no. ')
        relevance_scores.append([cur_nn_score[0], float(cur_nn_score[1]), relevenaceVect[i][1]])
 """
    # extract image names, labels from relevance_scores
    images = []
    all_data = []
    for image in relevenaceVect:
        data = []

        images.append(image[0])
        # extract image names, labels
        image_name = image[0].split('-')

        # image distance
        data.append(image[1])
        # type
        data.append(image_name[1])
        # subject ID
        data.append(int(image_name[2]))
        #sample ID
        sampleID = image_name[3].split('.')
        data.append(int(sampleID[0]))
        # classification label
        data.append(image[-1])

        all_data.append(data)

    # extract labels
    labels = []
    for data in all_data:
        labels.append(int(data[-1]))
        del data[-1]

    print('Reorganizing results based on relevance using decision tree...')

    # decision tree trained on images labeled by user
    print("Training Tree....")
    clf = decision_tree.train_tree(all_data, labels)
    print("Tree trained.\n\nTesting Tree....")
    testing_data, testing_images = build_testing_data(nn_scores, relevenaceVect, query_image)

    # tested on whole folder of images used in Task 4/5?
    results = decision_tree.test_tree(testing_data, clf)
    print("Tree classification ended.")
    merged_list = [(testing_images[i], results[i]) for i in range(0, len(testing_images))]

    # save only initial training results
    relevant_scored = {}
    irrelevant_scored = {}
    for result_key, result_val in dict(merged_list).items():
        if result_key in images:
            if result_val == 1:
                relevant_scored[result_key] = result_val
            else:
                irrelevant_scored[result_key] = result_val
    output_dict = {query_image[0]: query_image[1]}
    output_dict.update(relevant_scored)
    output_dict.update(irrelevant_scored)

    return output_dict


# format testing data from Task 4/5 output file
def build_testing_data(scores, relevenaceVect,query_image):
    relevant = [item[0] for item in relevenaceVect]
    relevant.append(query_image[0])

    images = []
    all_data = []

    # only use data not in training data
    for score in scores:
        data = []

        images.append(score[0])
        # extract image names, labels
        image_name = score[0].split('-')

        # image distance
        data.append(float(score[1]))
        # type
        data.append(image_name[1])
        # subject ID
        data.append(int(image_name[2]))
        #sample ID
        sampleID = image_name[3].split('.')
        data.append(int(sampleID[0]))

        all_data.append(data)

    return all_data, images


def visualizeImageComparison(query_image, nn_scores, k, image_folder_path):
    fig, tup = plt.subplots(k + 1, 1, figsize=(16, 16))
    tup[0].axis('off')
    im_path = image_folder_path / query_image[0]
    qryImage= np.asarray(Image.open(im_path))
    tup[0].imshow(qryImage, cmap=plt.cm.gray)
    tup[0].set_title("Query Image: ")

    counter = 1
    for x in tup[1:]:
        currFile = str(nn_scores[counter][0])
        im = nn_scores[counter][0]
        new_im_path = image_folder_path / im
        thisIm = np.asarray(Image.open(new_im_path))
        x.axis('off')
        x.set_title(currFile + " has rank number " + str(counter) + " with score " + str(nn_scores[counter][1]))
        x.imshow(thisIm, cmap=plt.cm.gray)
        counter += 1
    plt.show()


if __name__ == '__main__':
    print(task_6())

# task 5 example input file: E:\Documents\GitHub\CSE515Phase3\output\task_5\vaRankings)local_binary_pattern)E(]Downloads]phase2_data]test_1]image-con-28-8,png)5.csv
# task 4 example input file: C:\Users\ipbol\Downloads\CSE515\CSE515Phase3\output\task_4\lshRankings)histogram_of_oriented_gradients)C(]Users]ipbol]Downloads]CSE515]test]all]image-jitter-11-10,png)6.csv
