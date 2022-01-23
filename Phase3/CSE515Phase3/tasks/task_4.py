import csv
import glob
import json
import sys
import os
import copy
from pathlib import Path
from PIL import Image, ImageOps
curDir = os.path.dirname(os.path.realpath(__file__)) #add directory above current one so that our files from the parent directoery can be referenced.
upDir = os.path.dirname(curDir)
sys.path.append(upDir)
import pandas as pd
from matplotlib import image as image
import numpy as np
import helper
import lsh
import features


def get_random_vectors(layer, hashes, vector_length, vector_data):
    random_vectors = {}
    if not vector_data:
        for i in range(layer):
            for j in range(hashes):
                random_vectors[i, j] = list(np.random.randint(0, 9, size=(2, vector_length)) * 1.1)
    else:
        for i in range(layer):
            count = 0
            for j in range(hashes):
                vector_weight = []
                vector_weight.append(vector_data[count % len(vector_data)])
                vector_weight.append(vector_data[(count + 1) % len(vector_data)])
                random_vector = vector_weight * np.random.randint(0, 9, size=(2, len(vector_weight[0]))) * 0.8
                if len(random_vector) < vector_length:
                    temp = []
                    temp.append(random_vector[0].tolist() + list(
                        np.random.randint(0, 9, size=(vector_length - len(random_vector[0]))) * 0.8))
                    temp.append(random_vector[1].tolist() + list(
                        np.random.randint(0, 9, size=(vector_length - len(random_vector[1]))) * 0.1))
                    random_vectors[i, j] = temp
                count += 1
    return random_vectors


def get_distances(vectors, all_images):
    results = {}
    result = []
    for i in vectors:
        dist = {}
        projection = []
        for image in all_images:
            a = np.array(vectors[i], dtype=np.float)
            b = a[1]
            a = a[0]
            p = np.array(all_images[image], dtype=np.float)
            ap = p - a
            ab = b - a
            # distance = np.array(np.around((ab * ap) / Distance.E2_distance(a, b), decimals=4))
            projection.append(np.array(np.around((a + np.dot(ab, ap) / np.dot(ab, ab) * ab), decimals=3)))
            # projection.append(np.array(np.around()))
        min_proj = np.amin(projection, axis=0)
        image_keys = list(all_images.keys())
        j = 0
        # print(min_proj)
        for x in projection:
            distance = helper.euclidean_distance(min_proj, x)
            ind = image_keys[j]
            dist[ind] = distance
            j = j + 1
        result.append(dist)
        results['h' + str(i)] = result
        result = []
    # print(results)
    return results


def get_buckets(results, num_buckets=4):
    buckets = {}
    buckets_range = {}
    for result in results:
        hash = results[result]
        for images in hash:
            # print(images)
            distances = list(images.values())
            # print(distances)
            distances = np.around(np.array(distances, dtype=np.float), decimals=3)
            dist = (np.amax(distances) - np.amin(distances)) / num_buckets
            # bucket_range = range(int(np.amin(distances)), int(np.amax(distances)), int(dist) if int(dist) > 0 else 1)
            bucket_range = np.arange(np.amin(distances), np.amax(distances), dist)
            # print(bucket_range)
            bucket = {}
            for i in range(num_buckets):
                bucket[str(i)] = []
            for image in images:
                # print(images[image])
                x = images[image]
                j = 0
                for b in bucket_range:
                    # print(b)
                    if x < b:
                        if j >= len(bucket):
                            j = len(bucket) - 1
                        bucket[str(j)].append(image)
                        break
                    j = j + 1
            buckets[result] = bucket
            buckets_range[result] = bucket_range
            break
    # print(buckets)
    return buckets, buckets_range


def get_feature_length(feature_model):
    feature_model_to_matrix_sie = {"color_moment": 64, "local_binary_pattern": 4096,
                                   "histogram_of_oriented_gradients": 4096}
    return feature_model_to_matrix_sie[feature_model]


def get_index_structure(layer, hashes, vector_length, vector_data, vectors):
    random_vectors = get_random_vectors(layer, hashes, vector_length, vector_data)

    # projection code
    results = get_distances(random_vectors, vectors)

    # bucketing code
    num_buckets = min(pow(2, hashes), len(vectors), 10)
    buckets, ranges = get_buckets(results, num_buckets)

    return buckets, random_vectors, ranges


def get_t_nearest_images(q_image, unique_images_considered, query_image_name, t):
    t_nearest_images = {}
    for image in unique_images_considered:
        t_nearest_images[image] = helper.euclidean_distance(q_image[image], q_image[query_image_name])

    #sorted does not sort in place and needs to be assigned to a variable
    #l = sorted(t_nearest_images.items(), key=lambda x: x[1])
    #return dict(l[0:t]), dict(l)
    sorted(t_nearest_images.items(), key=lambda x: x[1], reverse=True)
    return dict(list(t_nearest_images.items())[:t]), t_nearest_images


def get_t_nearest_images_using_LSH(layers, hashes, vectors, index_struct, index_ranges, q_image, query_image_name, t):
    results = get_distances(vectors, q_image)
    q_image_dist = {}
    for hash_val in results:
        q_image_dist[hash_val] = results[hash_val][0][query_image_name]

    overall_images, buckets_searched, total_images_considered = find_buckets_of_similar_images(layers, hashes,
                                                                                               q_image_dist,
                                                                                               index_ranges,
                                                                                               index_struct, expand=0)
    unique_images_considered, unique_images_count, overall_images_considered = get_unique_images(layers, overall_images)

    expanding = 0
    while len(unique_images_considered) < t and expanding < 20:
        print("Unable to find k-nearest images, expanding out search")
        expanding += 1
        overall_images, buckets_searched, total_images_considered = find_buckets_of_similar_images(layers, hashes,
                                                                                                   q_image_dist,
                                                                                                   index_ranges,
                                                                                                   index_struct,
                                                                                                   expanding)
        unique_images_considered, unique_images_count, overall_images_considered = get_unique_images(layers,
                                                                                                     overall_images)

    images, image_dict = get_t_nearest_images(q_image, unique_images_considered, query_image_name, t)
    return images, buckets_searched, unique_images_count, overall_images_considered, total_images_considered, image_dict


def get_unique_images(layers, overall_images):
    overall_images_considered = 0
    unique_images_considered = set()
    for layer in range(layers):
        if layer == 0:
            unique_images_considered = overall_images[layer][layer]
        else:
            unique_images_considered = set.union(unique_images_considered, overall_images[layer][layer])
        overall_images_considered += len(overall_images[layer][layer])
    unique_images_count = len(unique_images_considered)
    return unique_images_considered, unique_images_count, overall_images_considered


def find_buckets_of_similar_images(layers, hashes, q_image_dist, index_ranges, index_struct, expand=0):
    total_images_considered = 0
    overall_images = []
    buckets_searched = 0
    print("Expanding")
    for i in range(layers):
        overall_images_in_level = {}
        unique_images = set()
        for j in range(hashes):
            q_dist = q_image_dist['h(' + str(i) + ', ' + str(j) + ')']
            index_range = index_ranges['h(' + str(i) + ', ' + str(j) + ')']
            x = 0
            for b in index_range:
                if q_dist < b:
                    if x >= len(index_struct['h(' + str(i) + ', ' + str(j) + ')']):
                        x = len(index_struct['h(' + str(i) + ', ' + str(j) + ')']) - 1
                    buckets_searched += 1
                    if len(unique_images) == 0:
                        unique_images = set(index_struct['h(' + str(i) + ', ' + str(j) + ')'][str(x)])
                        total_images_considered += len(index_struct['h(' + str(i) + ', ' + str(j) + ')'][str(x)])
                    else:
                        unique_images = set.intersection(unique_images,
                                                         set(index_struct['h(' + str(i) + ', ' + str(j) + ')'][str(x)]))
                        total_images_considered += len(index_struct['h(' + str(i) + ', ' + str(j) + ')'][str(x)])
                    if expand >= 1:
                        if x > 0:
                            count = x
                            for exp in range(expand):
                                if count > 0:
                                    buckets_searched += 1
                                    unique_images = set.union(unique_images,
                                                              index_struct['h(' + str(i) + ', ' + str(j) + ')'][
                                                                  str(count - 1)])
                                    total_images_considered += len(
                                        index_struct['h(' + str(i) + ', ' + str(j) + ')'][str(count - 1)])
                                    count -= 1
                        if x < len(index_struct['h(' + str(i) + ', ' + str(j) + ')']) - 1:
                            count = x
                            for exp in range(expand):
                                if count < len(index_struct['h(' + str(i) + ', ' + str(j) + ')']) - 1:
                                    buckets_searched += 1
                                    unique_images = set.union(unique_images,
                                                              index_struct['h(' + str(i) + ', ' + str(j) + ')'][
                                                                  str(count + 1)])
                                    total_images_considered += len(
                                        index_struct['h(' + str(i) + ', ' + str(j) + ')'][str(count + 1)])
                                    count += 1
                    break
                x = x + 1
        overall_images_in_level[i] = unique_images
        overall_images.append(overall_images_in_level)
    return overall_images, buckets_searched, total_images_considered


def display_results(actual, predicted):
    correct_score = misses = false_positive = 0
    for a in actual:
        if a[0] in predicted:
            correct_score += 1
        else:
            misses += 1

    actual = dict(actual)
    for p in predicted:
        if p not in actual:
            false_positive += 1

    print("\n\n\n\nAnalysis Results\n\n")
    #print("Correctly Identified = " + str(correct_score))
    print("Miss Rate = " + str(misses / len(actual)))
    print("False Positive Rate = " + str(false_positive / len(predicted)))


def find_nearest_k(all_images, query_image, k):
    explored = {}
    for image in all_images:
        explored[image] = helper.euclidean_distance(all_images[image], query_image)
    sorted_list = list(sorted(explored.items(), key=lambda item: item[1]))
    return sorted_list[:k]


# Read in the folder of images and output the corresponding feature_model vectors
def read_image_folder(image_directory, feature_model):
    cur_path = Path(os.path.abspath(os.curdir)).parent
    with open(str(cur_path)+"/lsh_"+feature_model+".csv", 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for file in os.listdir(image_directory):
            if file.endswith(".png"):
                im = Image.open(image_directory + file)
                im = ImageOps.grayscale(im)
                im = np.array(im)
                if feature_model == "color_moment":
                    cm = features.cm_8x8_image(im)
                    mean = cm[0]
                    std = cm[1]
                    skew = cm[2]
                    cm_average = [file]
                    for i in range(len(mean)):
                        cm_average.append((mean[i]+std[i]+skew[i])/3)
                    writer.writerow(cm_average)
                elif feature_model == "local_binary_pattern":
                    elbp = features.elbp_image(im)
                    elbp_csv = [file]
                    for x in elbp:
                        for val in x:
                            elbp_csv.append(val)
                    writer.writerow(elbp_csv)
                elif feature_model == "histogram_of_oriented_gradients":
                    fd = features.hog_image(im)
                    hog_csv = [file]
                    for x in fd:
                        for val in x:
                            hog_csv.append(val)
                    writer.writerow(hog_csv)
                else:
                    print("No model of the given name exists = " + feature_model)
                    return None
        return csvfile


def task_4(layers, hashes, set_of_vectors, image_folder, feature_model, q_image, t):
    all_images = {}
    query_image = {}
    query_image_name = ''
    for img in glob.glob(image_folder + "/image-*.png"):
        all_images[Path(img).name] = image.imread(img)
    for q_img in glob.glob(q_image):
        query_image_name = Path(q_img).name
        query_image[query_image_name] = image.imread(q_img)

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

    vector_data = []
    try:
        for line in csv.reader(open(set_of_vectors), delimiter=','):
            tup = []
            for val in line:
                if helper.isfloat(val):
                    tup.append(float(val))
            if tup:
                vector_data.append(tuple(tup))
    except FileNotFoundError:
        print("Wrong file or file path")

    lsh_index_data_structure, vectors, lsh_index_data_structure_range = lsh.get_index_structure(layers, hashes,
                                                                                                lsh.get_feature_length(
                                                                                                    feature_model),
                                                                                                vector_data,
                                                                                                images_feature)

    print("Index structure built with images\n")
    print(json.dumps(lsh_index_data_structure, indent=2))

    copy_dict = query_image_feature.copy()
    copy_dict.update(images_feature)
    t_nearest_images, buckets_searched, unique_images_count, overall_images_considered, total_images_considered, image_dict = lsh.get_t_nearest_images_using_LSH(
        layers,
        hashes,
        vectors,
        lsh_index_data_structure,
        lsh_index_data_structure_range,
        copy_dict,
        query_image_name,
        t)

    print("\nQuery Result:\n")
    print(t_nearest_images)
    print("\nBuckets Searched = " + str(buckets_searched))
    print("Unique Images Count = " + str(unique_images_count))
    print("Overall Images Considered = " + str(overall_images_considered))
    print("Total Images Considered In Each Bucket = " + str(total_images_considered))

    k_nearest_images = find_nearest_k(images_feature, query_image_feature[query_image_name], t)
    print("\n\nThe actual K-nearest Images")
    print(k_nearest_images)
    display_results(k_nearest_images, t_nearest_images)

    output = save_output_to_csv(image_dict, q_image, feature_model, t)
    return output


def save_output_to_csv(image_dict, query_name, feature_model, t):
    name = query_name.split('/')[-1]

    nearest_neighbors = {}
    for key, value in image_dict.items():
        nearest_neighbors[key] = value

    sorted_list = list(sorted(nearest_neighbors.items(), key=lambda item: item[1]))
    query_image_path = query_name.replace('/', ']').replace('\\', ']')
    query_image_path = query_image_path.replace('.', ',').replace(':', '(')
    # outputPath = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    #cur_path = Path(os.path.abspath(os.curdir)).parent
    #filepath = os.path.join(cur_path.__str__() + "/output/task_4/", "lshRankings)" + feature_model + ")" + query_image_path + ")" + str(t) + ".csv")
    #filepath = Path(filepath).__str__()

    #cur_path = Path(os.path.abspath(os.curdir)).parent
    file_path = Path("output/task_4/lshRankings" + ")" + feature_model + ")" + query_image_path + ")" + str(t) + ".csv")


    actual_path = Path(os.path.abspath(os.curdir))
    output_path = os.path.join(actual_path, file_path)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        for row in sorted_list:
            writer.writerow(row)
    return sorted_list

def task_4_runner(decision, images_path, model_name, q_image_path, t, vector_path=None):
    if decision == "1":
        l = int(input("\nPlease Provide Number of Layers: "))
        h = int(input("\nPlease Provide Number of Hashes Per Layer: "))

        return task_4(l, h, vector_path, images_path, model_name, q_image_path, t)
    elif decision == "2":
        l = int(input("\nPlease Provide Number of Layers: "))
        h = int(input("\nPlease Provide Number of Hashes Per Layer: "))
        csv_output = read_image_folder(images_path, model_name)

        return task_4(l, h, csv_output.name, images_path, model_name, q_image_path, t)
        # csv_output = read_image_folder("/home/preston/Phase2_Sample_Images/", "histogram_of_oriented_gradients")
        # return task_4(4, 6, csv_output.name, "/home/preston/Phase2_Sample_Images/",
        # "histogram_of_oriented_gradients", "/home/preston/Phase2_Sample_Images/image-cc-12-3.png", 6)
    else:
        print("Please enter either 1 or 2")

#task_4(4, 6, "C:/Users/ipbol/Downloads/CSE515/test/local_binary_pattern-cc-5-LDA.csv",
#       "C:/Users/ipbol/Downloads/CSE515/test/train", "histogram_of_oriented_gradients",
#       "C:/Users/ipbol/Downloads/CSE515/test/all/image-jitter-11-10.png", 6)

# task_4(4, 6, "/Users/keenan/Desktop/test/local_binary_pattern-cc-5-LDA.csv", "/Users/keenan/Desktop/test/train",
#        "histogram_of_oriented_gradients", "/Users/keenan/Desktop/test/all_images/image-jitter-11-10.png", 6)

# task_4(4, 6, "/Users/keenan/Desktop/test/local_binary_pattern-cc-5-LDA.csv", "/Users/keenan/Desktop/test/train",
#        "histogram_of_oriented_gradients", "/Users/keenan/Desktop/test/all_images/image-jitter-11-10.png", 6)
