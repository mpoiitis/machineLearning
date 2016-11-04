import pandas as pd
import numpy as np
from operator import itemgetter

test = pd.read_csv("test.csv", header=None).as_matrix()
train = pd.read_csv("train.csv", header=None).as_matrix()
labelsTest = pd.read_csv("label_test.csv", header=None).as_matrix().flatten()#make labels a single list
labelsTrain = pd.read_csv("label_train.csv", header=None).as_matrix().flatten()

class KMeans(object):
    def __init__(self, k):
        self.k = k

    def fit(self, trainSet, labels):
        self.set = trainSet
        self.labels = labels
        self.itemSets = self.createSets()
        self.centroids = self.getCentroids()

    def createSets(self):
        sets = []
        for x in range (self.k):
            subset = []
            sets.append(subset)

        for index,instance in enumerate(self.set):
            label = self.labels[index]
            sets[label].append(instance)

        return sets

    def getCentroids(self):
        centroids = []
        for x in range(self.k):
            mean = np.mean(np.array([set for set in self.itemSets[x]]), axis=0)
            centroids.append(mean)

        return centroids

    def getDistances(self,instance):
        distances = {}
        for x in range(len(self.centroids)):
            distance = np.linalg.norm(instance - self.centroids[x])
            distances[x] = distance
        sortedDistances = sorted(distances.items(), key = itemgetter(1))

        return sortedDistances

    def updateCentroids(self,instance):
        sortedDistances = self.getDistances(instance)
        self.moveCentroids(instance, sortedDistances)

    def moveCentroids(self, instance, sortedDistances):
        #get index of shortest to instance centroid
        index = sortedDistances[0][0]
        #update the corresponding set
        self.itemSets[index].append(instance)
        #recreate centroids according to mean value of updated set
        self.centroids[index] = np.mean(np.array([set for set in self.itemSets[index]]), axis=0)


    def calculateCentroidsFromTest(self, test):
        for j,instance in enumerate(test):
            self.updateCentroids(instance)

    def getWinner(self, instance):
        sortedDistances = self.getDistances(instance)
        winner = sortedDistances[0][0]
        return winner

    def predict(self, test):
        predictions = []
        for index,instance in enumerate(test):
            winner = self.getWinner(instance)
            predictions.append(winner)

        return predictions

    def getAccuracy(self, predictions, labelsTest):
        correct = 0
        for x in range(len(labelsTest)):
            if (labelsTest[x] == predictions[x]):
                correct += 1
        ratio = (correct/float(len(labelsTest)))*100.0

        return ratio

kmeans = KMeans(10)#the number of distinct numbers
kmeans.fit(train, labelsTrain)

predictions = kmeans.predict(test)
accuracy = kmeans.getAccuracy(predictions,labelsTest)

print("K nearest centroids using k = 10")
print('Accuracy: ' + str(accuracy) + '%')
