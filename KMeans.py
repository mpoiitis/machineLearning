import pandas as pd
import numpy as np
import random
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
        self.centroids = self.initializeCentroids()
        self.setsAccordingToCentroids = self.createSets()

    #centers is a list of arrays. Each array is an image
    def initializeCentroids(self):
        #take k random centers
        centroids = []
        for x in range (self.k):
            index = random.randint(0,len(self.set))
            centroids.append(self.set[index])

        return centroids

    def createSets(self):
        sets = []
        for x in range (self.k):
            subset = []
            subset.append(self.centroids[x])
            sets.append(subset)

        return sets

    def updateCentroids(self,instance):
        distances = {}
        for x in range(len(self.centroids)):
            distance = np.linalg.norm(instance - self.centroids[x])
            distances[x] = distance
        sortedDistances = sorted(distances.items(), key = itemgetter(1))
        self.moveCentroids(instance, sortedDistances)

    def moveCentroids(self, instance, sortedDistances):
        #get index of shortest to instance centroid
        index = sortedDistances[0][0]
        #update the corresponding set
        self.setsAccordingToCentroids[index].append(instance)
        #recreate centroids according to mean value of updated set
        self.centroids[index] = np.mean(np.array([set for set in self.setsAccordingToCentroids[index]]), axis=0)


    def calculateCentroidsFromTrain(self):
        for j,instance in enumerate(self.set):
            self.updateCentroids(instance)

for k in (3,5):
    kmeans = KMeans(k)
    kmeans.fit(train, labelsTrain)
    kmeans.calculateCentroidsFromTrain()
