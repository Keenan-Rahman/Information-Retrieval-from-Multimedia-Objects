import csv
import os
import sys
import re
from PIL import Image, ImageOps
import numpy as np
from pathlib import Path
curDir = os.path.dirname(os.path.realpath(__file__)) #add directory above current one so that our files from the parent directoery can be referenced.
upDir = os.path.dirname(curDir)
sys.path.append(upDir)
import features

# Check to see if the value passed in is a float
def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False

# Generates the partitions for each dimension
def generate_partitions(bj, data):
    # Generate parition points
    test_datapoint = data[0]
    cur_min = []
    cur_max = []
    for val in test_datapoint:
        cur_min.append(val)
        cur_max.append(val)

    for datapoint in data:
        counter = 0
        for val in datapoint:
            if val < cur_min[counter]:
                cur_min[counter] = val
            if val > cur_max[counter]:
                cur_max[counter] = val
            counter += 1

    dim_dif = []
    for i in range(len(cur_min)):
        dim_dif.append(cur_max[i]-cur_min[i]+1)

    part_range = []
    for i in range(len(dim_dif)):
        part_range.append((dim_dif[i]/((2**bj[i]))))

    partitions = {}
    for i in range(len(bj)):
        cur_parts = [cur_min[i]]
        for j in range(1,2**bj[i]+1):
            cur_parts.append(cur_min[i]+(part_range[i]*j))
        partitions[i] = cur_parts

    return partitions

# Calculates the number of bits
def bit_value_calculation(bj, partitions, data):
    bit_dict = {}
    va_string = ""
    for datapoint in data:
        bit_val = []
        counter = 0
        # Find the bit value for each dimension
        for val in datapoint:
            for i in range(1, len(partitions[counter])):
                if val <= partitions[counter][i]:
                    bit_val.append(i-1)
                    break
            counter += 1
        return_bit = ""
        counter2 = 0
        # Concatenate the dimensions bit values
        for bit in bit_val:
            if format(bit,"b") == '0':
                return_bit += '0'*bj[counter2]
            elif len(format(bit,"b")) != bj[counter2]:
                return_bit = '0'*(bj[counter2]-len(format(bit,"b"))) + format(bit,"b")
            else:
                return_bit += format(bit,"b")
            counter2 += 1
        va_string += return_bit
        if return_bit in bit_dict:
            bit_dict[return_bit].append(datapoint)
        else:
            bit_dict[return_bit] = [datapoint]
    return (bit_dict, va_string)

# Generate the VA-File structure in memory
def vafile_creation(b, user_input):
    data = []
    data_name_dict = {}
    f = csv.reader(open(user_input), delimiter=',')
    # Convert csv to correct format
    for line in f:
        tup = []
        name = ""
        for val in line:
            if re.search('^image', str(val)) != None:
                name = str(val)
            if isfloat(val):
                tup.append(float(val))
        data.append(tuple(tup))
        data_name_dict[tuple(tup)] = name
    if data[0] == ():
        data.pop(0)
    d = len(data[0])
    bj = []
    # Determine partition points required
    for j in range(1, d+1):
        temp = 0
        if j <= b%d:
            temp = 1
        cur_bj = int(b/d) + temp
        bj.append(cur_bj)

    # Generate parition points
    partitions = generate_partitions(bj, data)
    # Calculate bit value for each datapoint
    result = bit_value_calculation(bj, partitions, data)
    return result, data_name_dict

# Find the bucket that the query_point is located
def find_bucket(bit_dict, query_point):
    for key, val_list in bit_dict.items():
        for val in val_list:
            if val == query_point:
                return key

# Find the nearest t objects and output them with the full list of explored nodes,
# number of buckets explored, and nodes considered
def nearest_t(bit_dict, query_point, t):
    key_list = sorted(list(bit_dict.keys()))
    counter = 1
    bucket = find_bucket(bit_dict, query_point)
    bucket_count = 1
    explored = {}
    cur_bucket = bucket
    while len(explored) < t:
        for val in bit_dict[cur_bucket]:
            dist = np.linalg.norm(np.array(val)-np.array(query_point))
            explored[val] = dist
        cur_bucket_index = key_list.index(cur_bucket)
        if counter % 2 == 0:
            if (cur_bucket_index + counter) < len(key_list):
                cur_bucket = key_list[cur_bucket_index + counter]
                bucket_count += 1
        else:
            if (cur_bucket_index + counter) >= 0:
                cur_bucket = key_list[cur_bucket_index - counter]
                bucket_count += 1

        counter += 1
    sorted_list = list(sorted(explored.items(), key=lambda item: item[1]))
    nodes_considered = len(sorted_list)
    return (sorted_list[:t], sorted_list, bucket_count, nodes_considered)

# Find the true nearest k values
def find_nearest_k(bit_dict, query_point, k):
    explored = {}
    for point in bit_dict.values():
        for val in point:
            dist = np.linalg.norm(np.array(val)-np.array(query_point))
            explored[val] = dist
    sorted_list = list(sorted(explored.items(), key=lambda item: item[1]))
    return (sorted_list[:k], sorted_list)

# Generate the false positive and miss rate
def fal_pos_miss_rate(nearest_t, nearest_k):
    fal_pos_count = 0
    miss_count = 0
    for val in nearest_t:
        if val not in nearest_k:
            fal_pos_count += 1
    for val in nearest_k:
        if val not in nearest_t:
            miss_count += 1
    fal_pos_rate = fal_pos_count/len(nearest_t)
    miss_rate = miss_count/len(nearest_k)

    return (fal_pos_rate, miss_rate)

# Read in the folder of images and output the corresponding feature_model vectors
def read_image_folder(image_directory, feature_model):
    cur_path = Path(os.path.abspath(os.curdir)).parent
    with open(str(cur_path)+"/vafiles_"+feature_model+".csv", 'w', newline='') as csvfile:
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

# Find the query image data values
def find_query_point_name(csv_file, query_image_path):
    query_image_name = Path(query_image_path).name
    with open(csv_file) as csvf:
        reader = csv.reader(csvf)
        str_data = []
        for row in reader:
            if row[0] == query_image_name:
                str_data = row[1:]
                break
        float_data = []
        for str in str_data:
            float_data.append(float(str))
        return tuple(float_data)

# Find all the names of the data points passed in
def find_name_from_query_point(csv_file, top_t_list):
    with open(csv_file, 'r') as csvf:
        output_list = []
        reader = csv.reader(csvf)
        for data, dist in top_t_list:
            csvf.seek(0)
            for row in reader:
                float_data = []
                str_data = row[1:]
                for str in str_data:
                    if isfloat(str):
                        float_data.append(float(str))
                float_tuple_data = tuple(float_data)
                if data == float_tuple_data:
                    output_list.append((row[0], dist))
                    break
        return output_list

def save_output_to_csv(full_output, feature_type, filepath, t):
    cur_path = Path(os.path.abspath(os.curdir))
    file_path = Path("output/task_5/vaRankings" + ")" + feature_type + ")" + filepath + ")" + str(t) + ".csv")
    output_path = os.path.join(cur_path, file_path)

    with open(output_path, 'w', newline='') as outFile:
        writer = csv.writer(outFile)
        for row in full_output:
            writer.writerow(row)

def task_5(decision, images_path, model_name, q_image_path, t, vector_path=None):
    # Might need to change logic to consider folder of images passed in
    if decision == "1":
        b = input("Please enter your value for b: ")
        va_file = vafile_creation(int(b), vector_path)

        query_image_data = find_query_point_name(vector_path, q_image_path)
        closest_t_data = nearest_t(va_file[0][0], query_image_data, int(t))
        closest_t = find_name_from_query_point(vector_path, closest_t_data[0])
        print("Nearest t: ", closest_t)
        print("Buckets considered: ", closest_t_data[2])
        print("Nodes considered: ", closest_t_data[3])

        nearest_k_data = find_nearest_k(va_file[0][0], query_image_data, int(t))
        nearest_k = find_name_from_query_point(vector_path, nearest_k_data[0])
        print("Nearest k: ", nearest_k)
        fal_pos, miss = fal_pos_miss_rate(closest_t_data[1], nearest_k_data[1])
        print("False positive rate:", fal_pos, "Miss rate:", miss)

        all_scores = []
        for datapoint, dist in closest_t_data[1]:
            all_scores.append((va_file[1][datapoint],dist))
        filepath = (Path(q_image_path)).__str__()
        filepath = filepath.replace('/', ']')
        filepath = filepath.replace('\\', ']')
        filepath = filepath.replace(".", ",")
        filepath = filepath.replace(":", "(")
        save_output_to_csv(all_scores, model_name, filepath, t)
        return all_scores


    elif decision == "2":
        # directory_of_images = input("Please input the file path to the directory of images (include / at the end of the file path): ")
        # feature_type = input("Please input the feature type you want to use (color_moment, local_binary_pattern, or histogram_of_oriented_gradients): ")
        # query_image = input("Please input the file name of the query image: ")
        # t = input("Please input your t value: ")

        # directory_of_images = "/home/preston/Phase2_Sample_Images/"
        # feature_type = "histogram_of_oriented_gradients"
        # query_image = "image-cc-12-3.png"
        # t = "5"

        b = input("Please enter your value for b: ")
        csv_output = read_image_folder(images_path, model_name)
        va_file = vafile_creation(int(b), csv_output.name)

        query_image_data = find_query_point_name(csv_output.name, q_image_path)

        closest_t_data = nearest_t(va_file[0][0], query_image_data, int(t))

        closest_t = find_name_from_query_point(csv_output.name, closest_t_data[0])
        print("Nearest t: ", closest_t)
        print("Buckets considered: ", closest_t_data[2])
        print("Nodes considered: ", closest_t_data[3])
        nearest_k_data = find_nearest_k(va_file[0][0], query_image_data, int(t))
        nearest_k = find_name_from_query_point(csv_output.name,nearest_k_data[0])
        print("Nearest k: ", nearest_k)
        fal_pos, miss = fal_pos_miss_rate(closest_t_data[1], nearest_k_data[1])
        print("False positive rate:", fal_pos, "Miss rate:", miss)
        #all_scores = find_name_from_query_point(csv_output.name, closest_t_data[1])
        all_scores = []
        for datapoint, dist in closest_t_data[1]:
            all_scores.append((va_file[1][datapoint],dist))
        filepath = (Path(q_image_path)).__str__()
        filepath = filepath.replace('/', ']')
        filepath = filepath.replace('\\', ']')
        filepath = filepath.replace(".", ",")
        filepath = filepath.replace(":", "(")
        save_output_to_csv(all_scores, model_name, filepath, t)
        return all_scores
    else:
        print("Please enter either 1 or 2")

## SAMPLE INPUT FOR DECISION == 1
# va_file = vafile_creation(3, "/home/preston/Desktop/CSE515/Phase3/CSE515Phase3/color_moments-cc-3-KMeans.csv)

# closest_t = nearest_t(va_file[0], (0.6453648745255728,0.438677225557627,0.6460529036533915), 4)

# nearest_k = find_nearest_k(va_file[0], (0.6453648745255728,0.438677225557627,0.6460529036533915), 4)
# fal_pos, miss = fal_pos_miss_rate(closest_t[0], nearest_k)

# print(closest_t[0])
# print(fal_pos, miss)

## SAMPLE INPUT FOR DECISION == 2
# csv_output = read_image_folder("/home/preston/Phase2_Sample_Images/", "histogram_of_oriented_gradients")
# va_file = vafile_creation(3, csv_output.name)
# query_image_data = find_query_point_name(csv_output.name,"/home/preston/Phase2_Sample_Images/image-cc-12-3.png")
# closest_t_data = nearest_t(va_file[0], query_image_data, 5)
# closest_t = find_name_from_query_point(csv_output.name,closest_t_data[0])
# print(closest_t)
# nearest_k_data = find_nearest_k(va_file[0], query_image_data, 5)
# nearest_k = find_name_from_query_point(csv_output.name,nearest_k_data)
# print(nearest_k)
# fal_pos, miss = fal_pos_miss_rate(closest_t_data[0], nearest_k_data)
# print(fal_pos, miss)
