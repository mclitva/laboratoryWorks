"""
This example show how to classify wine
based on an existing dataset from UCI Machine Learning Repository.
The classification is made according to seven neighbors.
"""
import random
import math
import operator
import urllib.request
import numpy


def loadDataset(url):
    data_page = urllib.request.urlopen(url)
    dataset = numpy.loadtxt(data_page, delimiter=",")
    return dataset


def GetTestAndTrainingSets(data_set, split_coef, test_set = [], training_set = []):
    for x in range(len(data_set) - 1):
        if random.random() >= split_coef:
            test_set.append(data_set[x, :])
        else:
            training_set.append(data_set[x, :])



def getEuclideanDistance(instance_to, instance_from, length):
    distance = 0
    for x in range(length):
        distance += pow((instance_to[x] - instance_from[x]), 2)
    return math.sqrt(distance)


def getNeighbors(training_set, test_set_instance, k):
    distances = []
    length = len(test_set_instance) - 1
    for x in range(len(training_set)):
        dist = getEuclideanDistance(test_set_instance, training_set[x], length)
        distances.append((training_set[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])      
    return neighbors


def getClassVoteResult(neighbors):
    class_votes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(),
                          key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def calculateAccuracy(test_set, predicted_values):
    correct = 0
    for x in range(len(test_set)):
        if test_set[x][-1] == predicted_values[x]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def main():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    split_coef = 0.67
    k = 7
    data_set = loadDataset(url)
    training_set = []
    test_set = []
    GetTestAndTrainingSets(data_set,split_coef,test_set,training_set)
    print("Count of elements in repository: " + repr(len(data_set)))    
    print("Trainig set length: " + repr(len(training_set)))
    print("Test set length: " + repr(len(test_set)))
    predicted_values = []
    print("Input new data from the test set with count of neighbors = %i:" % k)
    for x in range(len(test_set)):
        neighbors = getNeighbors(training_set, test_set[x], k)
        vote_result = getClassVoteResult(neighbors)
        predicted_values.append(vote_result)        
        print("> Predicted value=%.2f, actual=%.2f" % (vote_result, test_set[x][-1]))
    accuracy = calculateAccuracy(test_set, predicted_values)
    print("Accuracy: " + repr(round(accuracy,4)) + '%')


main()
