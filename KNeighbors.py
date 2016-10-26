import pandas as pd
import numpy as np
from sortedcontainers import SortedDict
from operator import itemgetter

test = pd.read_csv("test.csv", header=None).as_matrix()
train = pd.read_csv("train.csv", header=None).as_matrix()
labelsTest = pd.read_csv("label_test.csv", header=None).as_matrix().flatten()#make labels a single list
labelsTrain = pd.read_csv("label_train.csv", header=None).as_matrix().flatten()

class KNN(object):
    def __init__(self, k):
        self.k = k

    def fit(self, trainSet, labels):
        self.set = trainSet
        self.labels = labels

    def getNearestNeighbors(self, testInstance):

        nearestNeighbors = SortedDict()
        for j,trainInstance in enumerate(self.set):
            distance = np.linalg.norm(testInstance - trainInstance)

            #if there arent k neighbors in set then add current trainInstance
            if len(nearestNeighbors) < self.k:
                    nearestNeighbors[distance] = self.labels[j]
            else:
                #if biggest distance in set is bigger than the current one
                # delete it from set and add the new one
                farthestDistanceInSet = nearestNeighbors.viewkeys()[-1]
                if (distance < farthestDistanceInSet):
                    del nearestNeighbors[farthestDistanceInSet]
                    nearestNeighbors[distance] = self.labels[j]

        return nearestNeighbors

    def getWinner(self, nearestNeighbors):
        classVotes = {}
        for neighbor in nearestNeighbors.viewvalues():
            classVotes[neighbor] = classVotes.get(neighbor,0) + 1
        sortedVotes = sorted(classVotes.items(), key=itemgetter(1), reverse = True)
        #return which number is the winner
        return sortedVotes[0][0]

    def predict(self, testSet):

        predictions = []
        for i,testInstance in enumerate(testSet): # return tuples in the form (0,testInstance0),(1,testInstance1)
            nearestNeighbors = self.getNearestNeighbors(testInstance)
            winner = self.getWinner(nearestNeighbors)
            predictions.append(winner)

        return predictions

    def getAccuracy(self, predictions, labelsTest):
        correct = 0
        for x in range(len(labelsTest)):
            if (labelsTest[x] == predictions[x]):
                correct += 1
        ratio = (correct/float(len(labelsTest)))*100.0
        return ratio

for k in (1,3):
    knn = KNN(k)
    knn.fit(train, labelsTrain)
    predictions = knn.predict(test)

    accuracy = knn.getAccuracy(predictions,labelsTest)
    print("K neighbors using k = " + str(k))
    print('Accuracy: ' + str(accuracy) + '%')
