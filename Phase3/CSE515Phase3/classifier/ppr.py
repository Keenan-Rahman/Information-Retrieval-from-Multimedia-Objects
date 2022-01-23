import numpy as np
import pandas as pd
import math
import sys
import os
curDir = os.path.dirname(os.path.realpath(
    __file__))  # add directory above current one so that our files from the parent directoery can be referenced.
upDir = os.path.dirname(curDir)
sys.path.append(upDir)
from sklearn.preprocessing import normalize
"""#########################
ppr.py
CSE 515
Project Phase III
Brandon Bayles
This program was created for use in phase III of the project
to aid with the classification of images based on a 
Personalized Page Rank classifier. 
#########################"""

class PPR():
    
    def __init__(self, X_train, Y_train):
        self.X_train = X_train
        self.Y_train = Y_train
        
    def predict(self, X_test):
        print("Starting ppr test classifier process")
        sim_matrix = self.generate_sim_matrix(X_test)
        print(sim_matrix)
        sim_graph = self.generate_sim_graph(sim_matrix)
        labels = self.compute_ppr_scores(sim_graph)
        print("PPR Classifications generated, returning...")
        return labels

    def compute_ppr_scores(self, sim_graph):
        print("Starting ppr score calculation")
        labels = []
        transitionMatrix = self.getTransistionMatrixFromGraph(sim_graph)
        #for node in sim_graph:
            #if(node.split("_")[0] == "test"):
        print("Calculating ppr for node scores")
        seed_set = {}
        for item in sim_graph:
            if item.split('_')[0] == 'test':
                seed_set[item] = sim_graph[item]
        rankings = self.personalized_page_rank(seed_set, sim_graph, transitionMatrix)

        labelledRankings = {}
        index = 0
        for indexLabel in sim_graph:
            labelledRankings[indexLabel] = rankings[index]
            index += 1
        sortedRankings = dict(sorted(labelledRankings.items(), key=lambda item: item[1], reverse=True))
        count = 0
        finalRankings = []
        for ranking in sortedRankings:
            name = ranking.split("_")[0]
            if(name != "test"):
                finalRankings.append([ranking, sortedRankings[ranking], self.Y_train[self.findIndexForLabel(ranking, sim_graph)]])
            count += 1

        labelCounts = {}
        for test_node in seed_set:
            nodeLabelCounts = {}
            neighbors = sim_graph[test_node]
            sort_neighbors = sorted(neighbors, key = lambda l:l[1])
            scored_neighbors = {}
            for neighbor in sort_neighbors:
                if(neighbor[0].split('_')[0] == 'train'):
                    score = sortedRankings[neighbor[0]]
                    scored_neighbors[neighbor[0]] = [score, neighbor[1]]
            scored_neighbors = list(scored_neighbors.items())[:20]
            sorted_neighbors = dict(sorted(scored_neighbors, key=lambda l: l[0]))
            neighCount = 0
            for neighbor in sorted_neighbors:
                if(neighCount > 4):
                    break
                if neighbor.split('_')[0] == 'train':
                    neighLabel = self.findLabelByImage(neighbor, finalRankings)
                    if(neighLabel in nodeLabelCounts):
                        nodeLabelCounts[neighLabel] += sortedRankings[neighbor]
                    else:
                        nodeLabelCounts[neighLabel] = sortedRankings[neighbor]
                    neighCount += 1
            labelCounts[test_node] = nodeLabelCounts
        labels = []
        for node in labelCounts:
            maxval = max(labelCounts[node], key=labelCounts[node].get)
            print("Label of node: " + node + " predicted to be: " + maxval)
            labels.append(maxval)
        return labels

    def findLabelByImage(self, image_name, final_rankings):
        for item in final_rankings:
            if(item[0] == image_name):
                return item[2]
        return 'NA'

    def findIndexByVal(self, list, value):
        for i in range(len(list)):
            if list[i] == value:
                return i

    def personalized_page_rank(self, seed_set, sim_graph, transitionMatrix):
        # p = (I - beta(T))^-1 (1 - beta)c
        I = np.identity(len(sim_graph))
        beta = 0.84
        c = np.zeros(len(sim_graph))
        c_score = 1 / (len(seed_set) + 1)
        sim_graph_keys_as_list = list(sim_graph.keys()) 
        for index in range(len(c)):
            indexSet = False
            #seed_index = self.findIndexForLabel(node, sim_graph)
            if sim_graph_keys_as_list[index] in seed_set:
                c[index] = c_score
            else:
                c[index] = ((1-c_score) / len(sim_graph))
                c_not_seed = (1-c_score) / len(sim_graph)
                #print(c_not_seed)

        p = (I - (beta * transitionMatrix)).T @ ((1 - beta) * c)
        return p

    def findIndexForLabel(self, node, sim_graph):
        index = 0
        for item in sim_graph:
            if(item == node):
                return index
            index += 1

    def checkInSet(self, node, set):
        for item in set:
            if item == node:
                return True
        return False

    def getTransistionMatrixFromGraph(self, a_graph): #TODO Fix so that transision graph accurately shows neighbors
        outMatrix = np.zeros((len(a_graph), len(a_graph)))
        for g in a_graph:
            g_index = self.findIndexForLabel(g, a_graph)
            for edge in a_graph[g]:
                e_index = self.findIndexForLabel(edge[0], a_graph)
                outMatrix[g_index][e_index] = edge[1]
                #divisor = len(a_graph[g])
                #outMatrix[g_index][e_index] = 1 #/ (len(a_graph[g]))
        outMatrix = np.asarray(outMatrix)
        outMatrix = normalize(outMatrix, axis = 1, norm='l1')
        return outMatrix

    def generate_sim_graph(self, sim_matrix):
        print("Calculating similarity graph")
        simGraph = {}
        for curNode, edgeDict in sim_matrix.items():
            simGraph[curNode] = []
            sorted_edges = sorted(edgeDict.items(), key=lambda item: item[1])
            count = 0
            i = 0
            while i < len(sorted_edges): #weighted
                if sorted_edges[i][1] != 0:
                    simGraph[curNode].append(sorted_edges[i])
                    count += 1
                i += 1 
        print("Similarity graph created.")
        return simGraph

    def generate_sim_matrix(self, X_test):
        print("Creating similarity matrix...")
        image_labels = []
        counter = 0
        for item in self.X_train:
            image_labels.append("train_" + str(counter))
            counter += 1
        counter = 0
        for item in X_test:
            image_labels.append("test_" + str(counter))
            counter += 1
        sim_matrix = pd.DataFrame(columns=image_labels, index=image_labels)

        for image1 in image_labels:
            for image2 in image_labels:
                if not math.isnan(sim_matrix[image1][image2]):
                    continue
                if(image1 == image2):
                    sim_matrix[image1][image2] = 0
                else:
                    sim_val = self.get_sim_val(image1, image2, X_test)
                    #sim_val = 0
                    sim_matrix[image1][image2] = sim_val
                    sim_matrix[image2][image1] = sim_val
        
        print("Similarity matrix complete")
        #print(sim_matrix)
        return sim_matrix

    def get_sim_val(self, image1, image2, X_test):
        print("Calculating similarity of images " + image1 + " and " + image2)
        im1type, im1index = image1.split('_')
        im2type, im2index = image2.split('_')
        im1index = int(im1index)
        im2index = int(im2index)
        im1Val = 0
        im2Val = 0
        if(im1type == "train"):
            im1Val = self.X_train[im1index]
        elif(im1type == "test"):
            im1Val = X_test[im1index]
        if(im2type == "train"):
            im2Val = self.X_train[im2index]
        elif(im2type == "test"):
            im2Val = X_test[im2index]
        sim_val = np.linalg.norm(im1Val - im2Val)
        print(str(sim_val))
        return sim_val

    #ppr_classifier('E:\Downloads\phase2_data\\test_1', 'E:\Downloads\phase2_data\\test_2', "local_binary_pattern")

