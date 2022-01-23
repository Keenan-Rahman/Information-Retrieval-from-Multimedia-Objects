import os
import pandas as pd
from matplotlib.image import imread
import sys

# -------------------- Define Tree Nodes --------------------

# define leaf nodes
# store prediction proportions for each label
# for a given object
class Leaf:
    def __init__(self, data):
        if data == []:
            self.classifications = {}
        else:
            data = pd.DataFrame(data)
            #rename last column
            data.columns = [*data.columns[:-1], 'labels']

            # get unique labels and the amount
            labels = dict(data['labels'].value_counts())

            self.classifications = labels



# object which is used to split the data
class Query:
    def __init__(self, f, v):
        self.feature = f
        self.val = v

    def check(self, query):
        # compare feature value in query to
        # feature value in object
        if isinstance(query[self.feature], (int, float, complex)): #numerical
            return query[self.feature] > self.val
        else:
            return query[self.feature] == self.val # categorical



class Node:
    def __init__(self, q, tb, fb):
        self.true_branch = tb
        self.false_branch = fb
        self.query = q



def gini_index(data):
    if data == []:
        return 0.0

    data = pd.DataFrame(data)

    #rename last column
    data.columns = [*data.columns[:-1], 'labels']

    # get unique labels and the amount
    labels = dict(data['labels'].value_counts())

    # calculate gini
    gini_index = 1.0
    for label in labels:
        # compute gini for current label and add to index
        gini_index -= (labels[label] / float(len(data))) * (labels[label] / float(len(data)))

    return gini_index

# -------------------- Data Partitioning -------------------

# split the data
def determine_split(data):
    impurity = gini_index(data)

    df = pd.DataFrame(data)

    best_split = [0, None]
    for feature in range(0, len(data[0]) - 1):
        unique_values = df[feature].unique()
        for value in unique_values:
            query = Query(feature, value)

            false_rows = []
            true_rows = []
            for data_point in data:
                if query.check(data_point):
                    true_rows.append(data_point)
                else:
                    false_rows.append(data_point)

            # make sure query splits dataset
            if len(true_rows) != 0 or len(false_rows) != 0:
                # information gain
                probability = float(len(true_rows)) / (len(true_rows) + len(false_rows))
                gain = impurity - probability * gini_index(true_rows) - (1 - probability) * gini_index(false_rows)
                if gain > best_split[0]:
                    best_split = gain, query

    return best_split


# -------------------- Tree Construction and Prediction --------------------

# recursively build the tree
def make_tree(data):
    # split dataset
    gain, query = determine_split(data)

    # base case
    if gain == 0:
        return Leaf(data)

    false_rows = []
    true_rows = []
    for data_point in data:
        if query.check(data_point):
            true_rows.append(data_point)
        else:
            false_rows.append(data_point)

    true_branch = make_tree(true_rows)
    false_branch = make_tree(false_rows)

    # recursively generate tree
    return Node(query, true_branch, false_branch)



def predict(data_point, node):
    if isinstance(node, Leaf):
        # may be several possibilities for classification in a single leaf node
        # take first one, since equal likelihood amongst all classifications
        return list(node.classifications.keys())[0]

    if node.query.check(data_point):
        return predict(data_point, node.true_branch)
    else:
        return predict(data_point, node.false_branch)

# -------------------- Decision Tree Data Retrieval --------------------

# Trains the tree on the given X and Y data
def train_tree(X_train, Y_train):
    df = pd.DataFrame(X_train)
    df['labels'] = Y_train

    tree = make_tree(df.values.tolist())

    return tree


# classify testing data with generated tree
def test_tree(X_test, d_tree):
    results = []
    for x in X_test:
        results.append(predict(x, d_tree))

    return results
