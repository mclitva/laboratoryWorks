"""
This eiample show how to classify wine
based on an eiisting dataset from UCI Machine Learning Repository.
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
    for i in range(len(data_set) - 1):
        if random.random() >= split_coef:
            test_set.append(data_set[i, :])
        else:
            training_set.append(data_set[i, :])



def getEuclideanDistance(instance_to, instance_from, length):
    distance = 0
    for i in range(length):
        distance += pow((instance_to[i] - instance_from[i]), 2)
    return math.sqrt(distance)


def getNeighbors(training_set, test_set_instance, k):
    distances = []
    length = len(test_set_instance) - 1
    for i in range(len(training_set)):
        dist = getEuclideanDistance(test_set_instance, training_set[i], length)
        distances.append((training_set[i], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for i in range(k):
        neighbors.append(distances[i][0])      
    return neighbors


def getClassVoteResult(neighbors):
    class_votes = {}
    for i in range(len(neighbors)):
        response = neighbors[i][0]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(),
                          key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


def calculateAccuracy(test_set, predicted_values):
    correct = 0
    for i in range(len(test_set)):
        if test_set[i][0] == predicted_values[i]:
            correct += 1
    return (correct / float(len(test_set))) * 100.0


def main():
    url = "http://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"
    split_coef = 0.67
    k = 5
    data_set = loadDataset(url)
    training_set = []
    test_set = []
    GetTestAndTrainingSets(data_set,split_coef,test_set,training_set)
    print("Count of elements in repository: " + repr(len(data_set)))    
    print("Trainig set length: " + repr(len(training_set)))
    print("Test set length: " + repr(len(test_set)))
    predicted_values = []
    print("Input new data from the test set with count of neighbors = %i:" % k)
    for i in range(len(test_set)):
        neighbors = getNeighbors(training_set, test_set[i], k)
        vote_result = getClassVoteResult(neighbors)
        predicted_values.append(vote_result)        
        print("> Predicted value=%.2f, actual=%.2f" % (vote_result, test_set[i][0]))
    accuracy = calculateAccuracy(test_set, predicted_values)
    print("Accuracy: " + repr(round(accuracy,4)) + '%')


main()
