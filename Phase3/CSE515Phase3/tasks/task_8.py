import os
import sys
from matplotlib import image, pyplot as plt
import numpy as np
from PIL import Image
from pathlib import Path
curDir = os.path.dirname(os.path.realpath(__file__)) #add directory above current one so that our files from the parent directoery can be referenced.
upDir = os.path.dirname(curDir)
sys.path.append(upDir)
from task_4 import task_4_runner
from task_5 import task_5
from task_6 import task_6
from task_7 import task_7

def visualizeImageComparisonDict(query_image, nn_scores, k, image_folder_path):

    fig, tup = plt.subplots(k + 1, 1, figsize=(16, 16))
    tup[0].axis('off')
    im_path = image_folder_path + query_image[0]
    im_path = Path(query_image)
    qryImage= np.asarray(Image.open(im_path))
    tup[0].imshow(qryImage, cmap=plt.cm.gray)
    tup[0].set_title("Query Image (" + im_path.name + "): ")

    counter = 1
    for x in tup[1:]:
        currFile = list(nn_scores.keys())[counter]
        im = image_folder_path + str(currFile)
        #new_im_path = image_folder_path / im
        new_im_path = Path(im)
        thisIm = np.asarray(Image.open(new_im_path))
        x.axis('off')
        x.set_title(currFile + " has rank number " + str(counter) + " with score " + str(list(nn_scores.values())[counter]))
        x.imshow(thisIm, cmap=plt.cm.gray)
        counter += 1
    plt.show()

def visualizeImageComparisonList(query_image, nn_scores, k, image_folder_path):

    fig, tup = plt.subplots(k + 1, 1, figsize=(16, 16))
    tup[0].axis('off')
    im_path = image_folder_path + query_image[0]
    im_path = Path(query_image)
    qryImage= np.asarray(Image.open(im_path))
    tup[0].imshow(qryImage, cmap=plt.cm.gray)
    tup[0].set_title("Query Image (" + im_path.name + "): ")

    counter = 1
    for x in tup[1:]:
        currFile = str(nn_scores[counter][0])
        im = image_folder_path + nn_scores[counter][0]
        #new_im_path = image_folder_path / im
        new_im_path = Path(im)
        thisIm = np.asarray(Image.open(new_im_path))
        x.axis('off')
        x.set_title(currFile + " has rank number " + str(counter) + " with score " + str(nn_scores[counter][1]))
        x.imshow(thisIm, cmap=plt.cm.gray)
        counter += 1
    plt.show()

task_decision = input("Which task would you like to use: task_4 or task_5? ")
input_decision = input("Would you like to generate the partitions from a set of vectors or from a directory of images? Type 1 for vectors and type 2 for directory of images: ")
images_path = input("\nPlease Provide Images Path to populate Index, NOTE be sure to add a slash or backslash at the end!: ")
model_name = input("\nEnter Feature Model Names From\n- color_moment\n- local_binary_pattern\n- "
                "histogram_of_oriented_gradients\n\nPlease Enter Model Name: ")
q_image_path = input("\nPlease Provide Query Image Path: ")
t = int(input("\nPlease Enter T-Value: "))

top_t_results = None
if input_decision == "1":
    vector_path = input("\nPlease Provide Vector Path: ")
    if task_decision == "task_4":
        top_t_results = task_4_runner(input_decision, images_path, model_name, q_image_path, t, vector_path)
    elif task_decision == "task_5":
        top_t_results = task_5(input_decision, images_path, model_name, q_image_path, t, vector_path)
    else:
        print("Please input either task_4 or task_5")
elif input_decision == "2":
    if task_decision == "task_4":
        top_t_results = task_4_runner(input_decision, images_path, model_name, q_image_path, t)
    elif task_decision == "task_5":
        top_t_results = task_5(input_decision, images_path, model_name, q_image_path, t)
    else:
        print("Please input either task_4 or task_5")
else:
    print("Please input either 1 or 2")
print(top_t_results[:t])

visualizeImageComparisonList(q_image_path, top_t_results, t, images_path)

relevance_scores = []
for i in range(int(t)):
    cur_nn_score = top_t_results[i+1]
    relevance = input('Is image ' + cur_nn_score[0] + ' relevant? Enter 1 for yes and 0 for no. ')
    relevance_scores.append([cur_nn_score[0], float(cur_nn_score[1]), relevance])

choice = input('Use task_6 or task_7 for releveance feedback? ')
if(choice == 'task_6'):
    results = task_6(top_t_results, relevance_scores)
    print(results)
    visualizeImageComparisonDict(q_image_path, results, t, images_path)
elif(choice == 'task_7'):
    print('task 7')
    results = task_7(images_path, q_image_path, model_name, relevance_scores)
    print(results)
    visualizeImageComparisonList(q_image_path, results, t, images_path)
    print("program ending....")
else:
    print('invalid')
