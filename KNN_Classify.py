import csv
import numpy as np
import math
import operator


'''
Loads the Data File and returns the data as a List
'''
def loadDataFile(filename):
    file = csv.reader(open(filename, "r"))

#    for row in file:
    dataset = list(file)
#    print(dataset)

    dataset = np.array(dataset)

    return dataset

#Detarmines the class of the Test Data
def getClass(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if (response not in classVotes.keys()):
            classVotes[response] = 1
        else:
            classVotes[response] += 1

    value_list = list(classVotes.values())
    k_list = list(classVotes.keys())
    return k_list[value_list.index(max(value_list))]


def myknnclassify(X, test, k):
    dataset = loadDataFile(X)

#Find Length of the feature set
    feature_list_len = len(dataset[0])
    print("feature_list_len", feature_list_len)

#Check, if we have same number of features in test as well
    if(len(test) != feature_list_len - 1):
        print("Feature Lengths for X and test features provided does not match ...")
        exit(0)

#k must be greater than equal to 1
    if (k < 0):
        print("k must be >=1 ...")
        exit(0)

    dist_list = []

    for i in range(len(dataset)):
        sqr_dist = 0
        for j in range(feature_list_len -1):
            sqr_dist += ((float(dataset[i][j]) - float(test[j])) * (float(dataset[i][j]) - float(test[j])))
        dist = math.sqrt(sqr_dist)

        dist_list.append((dataset[i], dist))

    dist_list.sort(key=operator.itemgetter(1))

    k_neighbors = []
    for x in range(k):
        k_neighbors.append(dist_list[x][0])

    test_class = getClass(k_neighbors)

    print('Test Instance {} will have the class {}'.format(test, test_class))

    return test_class


#Testing with points data
X = 'points.csv'
test = [33,3]
k = 6

myknnclassify(X, test, k)