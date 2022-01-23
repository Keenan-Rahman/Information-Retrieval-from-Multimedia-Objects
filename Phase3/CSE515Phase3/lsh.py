import numpy as np

import helper


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
                        np.random.randint(0, 9, size=(vector_length - len(random_vector[1]))) * 0.8))
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
            projection.append(np.array(np.around((a + np.dot(ab, ap) / np.dot(ab, ab) * ab), decimals=5)))
        min_proj = np.amin(projection, axis=0)
        image_keys = list(all_images.keys())
        j = 0
        for x in projection:
            distance = helper.euclidean_distance(min_proj, x)
            ind = image_keys[j]
            dist[ind] = distance
            j = j + 1
        result.append(dist)
        results['h' + str(i)] = result
        result = []
    return results


def get_buckets(results, num_buckets=4):
    buckets = {}
    buckets_range = {}
    for result in results:
        hash = results[result]
        for images in hash:
            distances = list(images.values())
            distances = np.around(np.array(distances, dtype=np.float), decimals=5)
            dist = (np.amax(distances) - np.amin(distances)) / num_buckets
            bucket_range = np.arange(np.amin(distances), np.amax(distances), dist)
            bucket = {}
            for i in range(num_buckets):
                bucket[str(i)] = []
            for image in images:
                x = images[image]
                j = 0
                for b in bucket_range:
                    if x < b:
                        if j >= len(bucket):
                            j = len(bucket) - 1
                        bucket[str(j)].append(image)
                        break
                    j = j + 1
            buckets[result] = bucket
            buckets_range[result] = bucket_range
            break
    return buckets, buckets_range


def get_feature_length(feature_model):
    feature_model_to_matrix_sie = {"color_moment": 64, "local_binary_pattern": 4096,
                                   "histogram_of_oriented_gradients": 4096}
    return feature_model_to_matrix_sie[feature_model]


def get_index_structure(layer, hashes, vector_length, vector_data, vectors):
    random_vectors = get_random_vectors(layer, hashes, vector_length, vector_data)
    results = get_distances(random_vectors, vectors)
    num_buckets = min(pow(2, hashes), len(vectors))
    buckets, ranges = get_buckets(results, num_buckets)

    return buckets, random_vectors, ranges


def get_t_nearest_images(q_image, unique_images_considered, query_image_name, t):
    t_nearest_images = {}
    for image in unique_images_considered:
        t_nearest_images[image] = helper.euclidean_distance(q_image[image], q_image[query_image_name])
    sorted(t_nearest_images.items(), key=lambda x: x[1], reverse=True)
    return dict(list(t_nearest_images.items())[:t]), t_nearest_images


def get_nearest_images(q_image, unique_images_considered, query_image_name):
    t_nearest_images = {}
    for image in unique_images_considered:
        t_nearest_images[image] = helper.euclidean_distance(q_image[image], q_image[query_image_name])
    sorted(t_nearest_images.items(), key=lambda x: x[1], reverse=True)
    return dict(list(t_nearest_images.items()))


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


def get_all_nearest_images_using_LSH(layers, hashes, vectors, index_struct, index_ranges, q_image, query_image_name):
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
    while len(unique_images_considered) < 30 and expanding < 20:
        print("Unable to find k-nearest images, expanding out search")
        expanding += 1
        overall_images, buckets_searched, total_images_considered = find_buckets_of_similar_images(layers, hashes,
                                                                                                   q_image_dist,
                                                                                                   index_ranges,
                                                                                                   index_struct,
                                                                                                   expanding)
        unique_images_considered, unique_images_count, overall_images_considered = get_unique_images(layers,
                                                                                                     overall_images)

    images = get_nearest_images(q_image, unique_images_considered, query_image_name)
    return images


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
