import numpy as np
import random
import copy
import database
import graph


def task_8(similarity_graph, m):
    s_matrix = initialize_random_graph(similarity_graph)
    n = len(list(s_matrix.values())[0])
    total_subjects = len(similarity_graph)
    converge_parameter = 0.001
    counter = 0
    converged_solutions = {}
    c = 0.7

    while True:
        counter += 1
        prev_s_matrix = copy.deepcopy(s_matrix)
        for i in list(s_matrix.keys()):
            for k in range(n):
                s_matrix[i][k][1] = equation_4(prev_s_matrix, s_matrix[i], c, i, k)
        s_matrix_sum = 0
        for key, val in s_matrix.items():
            for tup in val:
                s_matrix_sum += tup[1]
        converged_solutions[counter] = check_graph_convergence(prev_s_matrix, s_matrix, total_subjects,
                                                               converge_parameter)
        if converged_solutions[counter][0] > 99.00:
            break

        if counter > 100 and (converged_solutions[counter][0] - converged_solutions[counter - 1][0]) < 1:
            # Break here as we have too many iterations and data is not converging much.
            break

    most_important_subjects = get_most_significant_subjects(converged_solutions[counter][1], m, n)
    for subject in most_important_subjects:
        print(subject + '\n')


def check_graph_convergence(prev_s_matrix, s_matrix, total_subjects, converge_parameter):
    nodes_diff = {}
    convergence_count = 0
    for (key1, val1), (key2, val2) in zip(prev_s_matrix.items(), s_matrix.items()):
        sum = 0
        for (tup1, tup2) in zip(val1, val2):
            sum += abs(tup1[1] - tup2[1])
        nodes_diff[key1] = sum / n
        if nodes_diff[key1] < converge_parameter:
            convergence_count += 1
    return [(convergence_count / total_subjects) * 100, copy.deepcopy(s_matrix)]


def initialize_random_graph(similarity_graph):
    s_matrix = copy.deepcopy(similarity_graph)

    # Random initialization of S
    for key, val in s_matrix.items():
        for tup in val:
            tup[1] = random.random()
    return s_matrix


def equation_4(cur_s_matrix, cur_edge_list, c, i, j):
    if i == j:
        return 1
    w_i_star = 0
    for edge in cur_edge_list:
        w_i_star += edge[1]
    if w_i_star == 0:
        return cur_s_matrix[i][j][1]
    sum_val = 0
    for edge in cur_edge_list:
        w_i_k = edge[1]
        s_k_j = cur_s_matrix[i][j][1]
        div_val = w_i_k / w_i_star
        inner_val = 1 - np.exp(-w_i_k)
        sum_val += div_val * inner_val * s_k_j
    return c * sum_val


def get_most_significant_subjects(converged_solution, m, n):
    nodes_rank = []
    for subject, subject_node in converged_solution.items():
        nodes_rank.append([subject, update_pagerank(subject_node, 0.15, n)])
    nodes_rank.sort(key=lambda x: x[1])
    nodes_rank.reverse()
    most_important_subjects = nodes_rank[:m]
    return ['Subject-' + str(subjects[0]) + ' With ASCOS++ Rank = ' + str(subjects[1]) for subjects in
            most_important_subjects]


def update_pagerank(subject_node_in_degree, d, n):
    pagerank_sum = sum((node[1] / len(subject_node_in_degree)) for node in subject_node_in_degree)
    random_walk = d / n
    pagerank = random_walk + (1 - d) * pagerank_sum
    return pagerank


n = int(input("\nPlease Enter n-Value: "))
m = int(input("\nPlease Enter m-Value: "))
similarity_matrix_id = input("\nPlease Enter Similarity Matrix Name: ")
subject_subject_similarity_matrix = database.get_subject_similarity_matrix_by_id(similarity_matrix_id)
task_8(graph.generate_similarity_graph(subject_subject_similarity_matrix, n), m)
