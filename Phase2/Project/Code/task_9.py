import numpy as np

import database
import graph
import networkx as nx
import matplotlib.pyplot as plt

"""#########################
task_9.py
CSE 515
Project Phase II
This program takes a subject-subject similarity matrix, 
a value n, a value m, and 3 subject IDs and uses them to 
create a similarity graph where each subject is one of the
n most similar subjects in the database and identify the most 
significant m objects using personalized page rank.
#########################"""


def task_9(sub_sub_sim_mat, n, m, subID1, subID2, subID3):
    # Create a similarity graph G(V,E) where V corresponds to the subjects in the database and E contains node pairs
    # vi,, vj such that for each subject vi,vj is one of the most similar subjects in the database
    similarityGraph = graph.generate_sim_graph(sub_sub_sim_mat, n)
    # Identify the most significant m subjects (relative to input subjects) using personalized page rank measure.
    mostSigSubs = getMostSigSubs(similarityGraph, m, subID1, subID2, subID3)
    print("\n\nSimilarity Graph:\n")
    for item in similarityGraph:
        print("\nSubject: " + str(item) + " has edges to: ")
        for i in range(len(similarityGraph[item])):
            print(str(similarityGraph[item][i][0]) + ", ", end = " ")
    visualizeGraph(similarityGraph)
    print("\n\nMost significant subjects (relative to input subjects: " + str(subID1) + ", " + str(subID2) + ", " + str(subID3) + ") found to be: ")
    for item in mostSigSubs:
        print("Subject " + str(item[0] + 1) + " with Personalized PageRank Score: " + str(item[1]))
    return [similarityGraph, mostSigSubs]


def getMostSigSubs(similarityGraph, m, subID1, subID2, subID3):
    rankings = personalized_page_rank(similarityGraph, [subID1, subID2, subID3])
    labelledrankings = {a: abs(b) for a, b in enumerate(rankings)}
    sub_list = [subID1, subID2, subID3]
    for item in sub_list:
        index = item - 1
        del labelledrankings[index]
    visualizePPRScores(labelledrankings)
    sortedRankings = dict(sorted(labelledrankings.items(), key=lambda item: item[1], reverse=True))
    return_set = list(sortedRankings.items())[:m]

    print(sortedRankings)
    return return_set


def personalized_page_rank(query_graph, seed_set):
    # p = (I - beta(T))^-1 (1 - beta)c
    I = np.identity(len(query_graph))
    beta = 0.84
    c = np.zeros(len(query_graph))
    c_score = 0.25
    for index in range(len(c)):
        indexSet = False
        for seed in seed_set:
            seed_index = int(seed) - 1
            if (index == int(seed)):
                c[seed_index] = c_score
                indexSet = True
        if (not indexSet):
            c[index] = (0.25 / len(query_graph))
    transitionMatrix = getTransistionMatrixFromGraph(query_graph)

    p = (I - (beta * transitionMatrix)).T @ ((1 - beta) * c)
    return p


def getTransistionMatrixFromGraph(a_graph):
    outMatrix = np.zeros((len(a_graph), len(a_graph)))
    for g in a_graph:
        for edge in a_graph[g]:
            for i in range(len(outMatrix)):
                for j in range(len(outMatrix[i])):
                    if (i == (g - 1) and j == (edge[0] - 1)):
                        outMatrix[i][j] = 1 / (len(a_graph[g]))
    outMatrix = np.asarray(outMatrix)
    return outMatrix

def visualizeGraph(a_graph):
    plot = graphPlot()
    for item in a_graph:
        for edge in a_graph[item]:
            plot.addEdge(item, edge[0])
    plot.draw()

def visualizePPRScores(PPR_vector):
    plt.bar(PPR_vector.keys(), PPR_vector.values())
    plt.title('Personalized PageRank Scores')
    plt.show()

class graphPlot:
    def __init__(self):
        self.edges = []
    def addEdge(self, start, end):
        self.edges.append([start, end])
    def draw(self):
        grph = nx.Graph()
        grph.add_edges_from(self.edges)
        nx.draw_networkx(grph)
        plt.title('Similarity Graph')
        plt.show()

""" sim_matrix = database.get_subject_similarity_matrix_by_feature_model_k_dimensionality_reduction('histogram_of_oriented_gradients', 3, 'LDA')
#print(sim_matrix)
task_9(sim_matrix, 4, 5, 2, 20, 34)  """

simMatrixFeatureStr = input("Enter feature model of similarity matrix to retrieve: ")
simMatrixKStr = input("Enter k value of similarity matrix to retrieve: ")
simMatrixDecompStr = input("Enter dimensionality reduction model of similarity matrix to retrieve: ")
sim_matrix = database.get_subject_similarity_matrix_by_feature_model_k_dimensionality_reduction(simMatrixFeatureStr,
                                                                                                int(simMatrixKStr),
                                                                                                simMatrixDecompStr)
#print(sim_matrix)
n = int(input("Enter n value to use: "))
m = int(input("Enter m value to use: "))
subId1 = int(input("Enter 1st subject ID to use: "))
subId2 = int(input("Enter 2nd subject ID to use: "))
subId3 = int(input("Enter 3rd subject ID to use: "))

task_9(sim_matrix, n, m, subId1, subId2, subId3)
